#!/usr/bin/env python3
"""Run an isolated iliac artery CPU/GPU comparison without modifying the source case.

Each run gets its own directory, solver XML, full log, and JSON timings.
The mesh is reused by symlink. Timed runs must be run sequentially.
"""
import argparse
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preconditioner', choices=['diagonal', 'schur-amg', 'ml', 'fsils'], required=True)
    parser.add_argument('--gmres-preconditioner', choices=['diagonal','resistance'], default='diagonal')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--launcher', choices=['mpirun','srun'], default='mpirun')
    parser.add_argument('--gpus-per-task', type=int, default=0)
    parser.add_argument('--gpus-per-node', type=int, default=0, help='Explicit GPU count for each Slurm step.')
    parser.add_argument('--srun-exact', action='store_true', help='Limit each Slurm step to its requested CPU resources.')
    parser.add_argument('--cpu-bind', choices=['cores'], help='Optional Slurm CPU binding.')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--assembly', choices=['auto','legacy','kokkos'], default='auto')
    parser.add_argument('--assembly-stats', action='store_true')
    parser.add_argument('--schur-reuse', choices=['transfers','fresh'], default='transfers')
    parser.add_argument('--schur-operator', choices=['explicit','factored'], default='explicit')
    parser.add_argument('--runtime-details', action='store_true')
    parser.add_argument('--profile-mpi', action='store_true', help='Include MPI tracing in a separate Nsight profile.')
    parser.add_argument('--profile', choices=['nsys'], help='Profile a separate short run; not a timing benchmark.')
    parser.add_argument('--stats', action='store_true', help='Enable per-Jacobian stats (new executable only).')
    args = parser.parse_args()
    if min(args.steps, args.ranks, args.repeat, args.threads) < 1:
        parser.error("steps, ranks, repeat, and threads must be positive")
    if args.preconditioner == 'fsils' and args.gmres_preconditioner != 'diagonal':
        parser.error('gmres-preconditioner selects a Trilinos policy; it is not applied to FSILS inputs')
    if args.gpus_per_task < 0 or (args.gpus_per_task and args.launcher != "srun"):
        parser.error("gpus-per-task requires srun and a nonnegative count")
    if args.gpus_per_node < 0 or (args.gpus_per_node and args.launcher != 'srun'):
        parser.error('gpus-per-node requires srun and a nonnegative count')
    exe = args.exe.resolve()
    if not exe.is_file():
        parser.error(f"executable does not exist: {exe}")
    case = Path(__file__).resolve().parents[2] / 'tests/cases/fluid/iliac_artery'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for i in range(args.repeat):
        run = output / f'run-{i + 1}'
        run.mkdir()  # Never overwrite prior measurements.
        (run / 'mesh').symlink_to(case / 'mesh', target_is_directory=True)
        tree = ET.parse(case / 'solver.xml')
        root = tree.getroot()
        general = root.find('GeneralSimulationParameters')
        general.find('Number_of_time_steps').text = str(args.steps)
        # Write identical final-state output only, keeping intermediate I/O out
        # of the timing comparison. Leave all physics and tolerances unchanged.
        general.find('Increment_in_saving_VTK_files').text = str(args.steps)
        general.find('Increment_in_saving_restart_files').text = str(args.steps)
        la = root.find('Add_equation/LS/Linear_algebra')
        if args.preconditioner != 'fsils':
            la.set('type', 'trilinos')
            for child in list(la):
                la.remove(child)
            ET.SubElement(la, 'Assembly').text = 'trilinos'
            ET.SubElement(la, 'Preconditioner').text = 'trilinos-diagonal'
            ET.SubElement(la, 'GMRES_Preconditioner').text = 'trilinos-' + args.gmres_preconditioner
            ET.SubElement(la, 'CG_Preconditioner').text = 'trilinos-' + args.preconditioner
        tree.write(run / 'solver.xml', encoding='utf-8', xml_declaration=True)
        command = [str(exe), 'solver.xml']
        if args.profile == 'nsys':
            command = ['nsys','profile','--trace='+('cuda,nvtx,mpi,osrt' if args.profile_mpi else 'cuda,nvtx,osrt'),'--sample=none',
                       '--stats=true','--output='+str(run/'trace')] + command
        if args.launcher == 'srun':
            prefix = ['srun','--ntasks',str(args.ranks),'--cpus-per-task',str(args.threads)]
            if args.srun_exact:
                prefix += ['--exact']
            if args.cpu_bind:
                prefix += ['--cpu-bind='+args.cpu_bind]
            if args.gpus_per_task:
                prefix += ['--gpus-per-task',str(args.gpus_per_task),'--gpu-bind=single:1']
            if args.gpus_per_node:
                prefix += ['--gpus-per-node',str(args.gpus_per_node)]
            command = prefix + command
        elif args.ranks > 1:
            command = ['mpirun', '-np', str(args.ranks)] + command
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(args.threads)
        env['KOKKOS_NUM_THREADS'] = str(args.threads)
        env['OPENBLAS_NUM_THREADS'] = '1'
        env['VECLIB_MAXIMUM_THREADS'] = '1'
        env.pop('SV_TRILINOS_BIPN_STATS', None)
        env['SV_TRILINOS_ASSEMBLY'] = args.assembly
        env['SV_TRILINOS_SCHUR_REUSE'] = args.schur_reuse
        env['SV_TRILINOS_SCHUR_OPERATOR'] = args.schur_operator
        env.pop('SV_TRILINOS_RUNTIME_DETAILS',None)
        env.pop('SV_TRILINOS_AMG_DETAILS',None)
        if args.runtime_details:
            env['SV_TRILINOS_RUNTIME_DETAILS']='1'
            env['SV_TRILINOS_AMG_DETAILS']='1'
        env.pop('SV_TRILINOS_ASSEMBLY_STATS',None)
        if args.assembly_stats:
            env['SV_TRILINOS_ASSEMBLY_STATS'] = '1'
        if args.stats:
            env['SV_TRILINOS_BIPN_STATS'] = '1'
        start = time.perf_counter()
        with (run / 'stdout.log').open('w') as log:
            result = subprocess.run(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        log = (run / 'stdout.log').read_text()
        history_path = run / f'{args.ranks}-procs/histor.dat'
        history = history_path.read_text() if history_path.exists() else ''
        nonlinear_lines = [line for line in history.splitlines()
                           if re.match(r'\s*NS\s+\d+-', line)]
        last_step = int(re.search(r'NS\s+(\d+)-', nonlinear_lines[-1]).group(1)) if nonlinear_lines else 0
        completed = last_step == args.steps and (run / f'{args.ranks}-procs/result_{args.steps:03d}.vtu').exists()

        stats = [{key: float(value) for key, value in re.findall(r'(\w+)=([-+\deE.]+)', line)}
                 for line in log.splitlines() if line.startswith('BIPN_STATS ')]
        assembly_stats = []
        for line in log.splitlines():
            if line.startswith(('ASSEMBLY_STATS ', 'ASSEMBLY_EXPORT_STATS ', 'ASSEMBLY_ALLOC_STATS ')):
                item = dict(re.findall(r'(\w+)=([^\s]+)', line))
                assembly_stats.append(dict(kind=line.split()[0], **item))
        boundary_stats = [dict(re.findall(r'(\w+)=([^\s]+)', line))
                          for line in log.splitlines() if line.startswith('BOUNDARY_SETUP_STATS ')]
        record = dict(schur_coarse=env.get('SV_TRILINOS_SCHUR_COARSE','klu'), gpu_aware_mpi=env.get('TPETRA_ASSUME_GPU_AWARE_MPI','auto'), schur_reuse=args.schur_reuse, schur_operator=args.schur_operator,
                      profiled=bool(args.profile), boundary_stats=boundary_stats,
                      executable=str(exe), preconditioner=args.preconditioner,
                      gmres_preconditioner=args.gmres_preconditioner,
                      assembly_mode=args.assembly, assembly_stats=assembly_stats,
                      steps=args.steps, ranks=args.ranks, threads=args.threads, command=command,
                      wall_seconds=elapsed, returncode=result.returncode,
                      completed=completed, nonlinear_iterations=len(nonlinear_lines),
                      simulation_seconds=float(nonlinear_lines[-1].split()[2]) if nonlinear_lines else None,
                      platform=platform.platform(), jacobians=stats)
        (run / 'measurement.json').write_text(json.dumps(record, indent=2) + '\n')
        print(f'{run}: {elapsed:.3f} s, exit={result.returncode}', flush=True)
        if result.returncode or not completed:
            raise SystemExit(f'Run failed; see {run / "stdout.log"}')


if __name__ == '__main__':
    main()
