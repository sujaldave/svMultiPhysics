#!/usr/bin/env python3
"""Run the iliac artery CPU comparison without modifying the source case.

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
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--stats', action='store_true', help='Enable per-Jacobian stats (new executable only).')
    args = parser.parse_args()
    if min(args.steps, args.ranks, args.repeat) < 1:
        parser.error("steps, ranks, and repeat must be positive")
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
            ET.SubElement(la, 'GMRES_Preconditioner').text = 'trilinos-diagonal'
            ET.SubElement(la, 'CG_Preconditioner').text = 'trilinos-' + args.preconditioner
        tree.write(run / 'solver.xml', encoding='utf-8', xml_declaration=True)
        command = [str(exe), 'solver.xml']
        if args.ranks > 1:
            command = ['mpirun', '-np', str(args.ranks)] + command
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '1'
        env['OPENBLAS_NUM_THREADS'] = '1'
        env['VECLIB_MAXIMUM_THREADS'] = '1'
        env.pop('SV_TRILINOS_BIPN_STATS', None)
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
        record = dict(executable=str(exe), preconditioner=args.preconditioner,
                      steps=args.steps, ranks=args.ranks, command=command,
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
