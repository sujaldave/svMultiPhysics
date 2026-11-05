#!/usr/bin/env python3
"""Probe CUDA binding per MPI task, or validate the resulting JSON files.

Works under srun (SLURM_PROCID) and Open MPI/HPC-X mpirun (OMPI_COMM_WORLD_RANK).
Uses the CUDA runtime, or the driver API, through ctypes; no Python CUDA
packages are required. Each record also holds the CPU affinity of the rank and
the NUMA-local CPUs of its GPU, so core binding can be audited after the run.
"""
import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import socket


def _cuda_bus_id():
    """Return (visible device count, runtime version, PCI bus ID of device 0).

    Tries the CUDA runtime first, then the driver API (libcuda.so.1 ships with
    the driver, so it is found even when the CUDA toolkit is not on the path).
    """
    for name in [ctypes.util.find_library('cudart'), 'libcudart.so',
                 'libcudart.so.12', 'libcudart.so.11.0']:
        if not name:
            continue
        try:
            runtime = ctypes.CDLL(name)
        except OSError:
            continue
        runtime.cudaGetErrorString.restype = ctypes.c_char_p

        def check(status):
            if status:
                raise RuntimeError(runtime.cudaGetErrorString(status).decode())

        count, version = ctypes.c_int(), ctypes.c_int()
        check(runtime.cudaGetDeviceCount(ctypes.byref(count)))
        check(runtime.cudaRuntimeGetVersion(ctypes.byref(version)))
        if count.value != 1:
            return count.value, version.value, None
        check(runtime.cudaSetDevice(0))
        bus = ctypes.create_string_buffer(64)
        check(runtime.cudaDeviceGetPCIBusId(bus, len(bus), 0))
        return count.value, version.value, bus.value.decode()
    try:
        driver = ctypes.CDLL('libcuda.so.1')
    except OSError:
        raise RuntimeError('Cannot load libcudart or libcuda; use the CUDA library environment from the GPU build')

    def check(status, what):
        if status:
            raise RuntimeError(f'{what} failed with CUDA driver error {status}')

    count, version, device = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    check(driver.cuInit(0), 'cuInit')
    check(driver.cuDeviceGetCount(ctypes.byref(count)), 'cuDeviceGetCount')
    check(driver.cuDriverGetVersion(ctypes.byref(version)), 'cuDriverGetVersion')
    if count.value != 1:
        return count.value, version.value, None
    check(driver.cuDeviceGet(ctypes.byref(device), 0), 'cuDeviceGet')
    bus = ctypes.create_string_buffer(64)
    check(driver.cuDeviceGetPCIBusId(bus, len(bus), device), 'cuDeviceGetPCIBusId')
    return count.value, version.value, bus.value.decode()


def _compact(cpus):
    """Format a CPU set as a Linux cpulist, e.g. 0-13,112-125."""
    cpus, ranges = sorted(cpus), []
    for cpu in cpus:
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ','.join(f'{a}' if a == b else f'{a}-{b}' for a, b in ranges)


def _parse_cpulist(text):
    cpus = set()
    for part in text.strip().split(','):
        if part:
            first, _, last = part.partition('-')
            cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def _gpu_locality(bus_id):
    """NUMA node and local CPUs of a PCI device, from sysfs (None if unknown)."""
    if not bus_id:
        return None, None
    device = Path('/sys/bus/pci/devices')/bus_id.lower()[-12:]
    try:
        numa = int((device/'numa_node').read_text())
        local = _parse_cpulist((device/'local_cpulist').read_text())
    except (OSError, ValueError):
        return None, None
    return numa, local


def probe(output):
    count, version, bus_id = _cuda_bus_id()
    if count != 1:
        raise RuntimeError(f'Expected exactly one visible GPU per task, got {count}')
    # Slurm direct launch (srun) or Open MPI/HPC-X launch (mpirun).
    rank = int(os.environ.get('SLURM_PROCID') if 'OMPI_COMM_WORLD_RANK' not in os.environ
               else os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = int(os.environ.get('SLURM_LOCALID') if 'OMPI_COMM_WORLD_LOCAL_RANK' not in os.environ
                     else os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    affinity = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else set()
    numa, local = _gpu_locality(bus_id)
    result = dict(rank=rank, local_rank=local_rank,
                  hostname=socket.gethostname(), visible_count=count,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  pci_bus_id=bus_id, cuda_runtime_version=version,
                  cpu_affinity=_compact(affinity), cpu_count=len(affinity),
                  gpu_numa_node=numa,
                  gpu_local_cpus=_compact(local) if local else None,
                  affinity_within_gpu_numa=(bool(affinity) and affinity <= local) if local else None,
                  bind_mode=os.environ.get('SVMP_BIND'))
    output.mkdir(parents=True, exist_ok=True)
    (output/f'rank-{rank}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def validate(output, ranks, multi_node=False, require_numa_local=False):
    rows = [json.loads(p.read_text()) for p in sorted(output.glob('rank-*.json'))]
    if len(rows) != ranks or {r['rank'] for r in rows} != set(range(ranks)):
        raise RuntimeError('Missing or unexpected MPI rank binding records')
    hosts = {r['hostname'] for r in rows}
    if len(hosts) != 1 and not multi_node:
        raise RuntimeError('This comparison requires one node')
    if any(r['visible_count'] != 1 for r in rows) or \
            len({(r['hostname'], r['pci_bus_id']) for r in rows}) != ranks:
        raise RuntimeError('Tasks are not bound to distinct GPUs')
    if require_numa_local and not all(r.get('affinity_within_gpu_numa') for r in rows):
        bad = [r['rank'] for r in rows if not r.get('affinity_within_gpu_numa')]
        raise RuntimeError(f'Ranks {bad} are not bound to their GPU NUMA-local CPUs')
    print(f'Validated {ranks} ranks on distinct GPUs on {len(hosts)} node(s): {", ".join(sorted(hosts))}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--validate', type=int, metavar='RANKS')
    parser.add_argument('--multi-node', action='store_true', help='Allow ranks on several nodes.')
    parser.add_argument('--require-numa-local', action='store_true',
                        help='Fail unless every rank runs only on its GPU NUMA-local CPUs.')
    args = parser.parse_args()
    if args.validate is None:
        probe(args.output)
    else:
        validate(args.output, args.validate, args.multi_node, args.require_numa_local)


if __name__ == '__main__':
    main()
