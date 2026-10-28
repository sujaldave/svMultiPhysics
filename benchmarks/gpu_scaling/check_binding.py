#!/usr/bin/env python3
"""Probe CUDA binding per Slurm task, or validate the resulting JSON files.

Uses the CUDA runtime through ctypes; no Python CUDA packages are required.
"""
import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import socket


def probe(output):
    candidates = [ctypes.util.find_library('cudart'), 'libcudart.so',
                  'libcudart.so.12', 'libcudart.so.11.0']
    runtime = None
    for name in candidates:
        if not name:
            continue
        try:
            runtime = ctypes.CDLL(name)
            break
        except OSError:
            pass
    if runtime is None:
        raise RuntimeError('Cannot load libcudart; use the CUDA library environment from the GPU build')
    runtime.cudaGetErrorString.restype = ctypes.c_char_p

    def check(status):
        if status:
            raise RuntimeError(runtime.cudaGetErrorString(status).decode())

    count, version = ctypes.c_int(), ctypes.c_int()
    check(runtime.cudaGetDeviceCount(ctypes.byref(count)))
    check(runtime.cudaRuntimeGetVersion(ctypes.byref(version)))
    if count.value != 1:
        raise RuntimeError(f'Expected exactly one visible GPU per task, got {count.value}')
    check(runtime.cudaSetDevice(0))
    bus = ctypes.create_string_buffer(64)
    check(runtime.cudaDeviceGetPCIBusId(bus, len(bus), 0))
    rank = int(os.environ['SLURM_PROCID'])
    result = dict(rank=rank, local_rank=int(os.environ['SLURM_LOCALID']),
                  hostname=socket.gethostname(), visible_count=count.value,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  pci_bus_id=bus.value.decode(), cuda_runtime_version=version.value)
    output.mkdir(parents=True, exist_ok=True)
    (output/f'rank-{rank}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def validate(output, ranks):
    rows = [json.loads(p.read_text()) for p in sorted(output.glob('rank-*.json'))]
    if len(rows) != ranks or {r['rank'] for r in rows} != set(range(ranks)):
        raise RuntimeError('Missing or unexpected MPI rank binding records')
    if len({r['hostname'] for r in rows}) != 1:
        raise RuntimeError('This comparison requires one node')
    if any(r['visible_count'] != 1 for r in rows) or len({r['pci_bus_id'] for r in rows}) != ranks:
        raise RuntimeError('Tasks are not bound to distinct GPUs')
    print(f'Validated {ranks} ranks on distinct GPUs on {rows[0]["hostname"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--validate', type=int, metavar='RANKS')
    args = parser.parse_args()
    if args.validate is None:
        probe(args.output)
    else:
        validate(args.output, args.validate)


if __name__ == '__main__':
    main()
