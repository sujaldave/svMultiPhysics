#!/usr/bin/env python3
"""Matched scalar/solid/FSI experiments; never edits source cases or overwrites runs."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
CASES = {'scalar': 'fluid/precomputed_dye_AD', 'solid': 'linear-elasticity/beam',
         'fsi': 'fsi/pipe_3d'}

def variants(case):
    result = [('gmres-'+p, 'GMRES', p, 'CG')
              for p in ('diagonal', 'blockjacobi', 'riluk0', 'ml')]
    if case == 'solid':
        result += [('cg-'+p, 'CG', p, 'CG') for p in ('diagonal', 'blockjacobi')]
    if case == 'fsi':
        result += [('gmres-riluk0-mesh-gmres', 'GMRES', 'riluk0', 'GMRES')]
    return result

def put(parent, tag, value):
    node = parent.find(tag)
    if node is None:
        node = ET.SubElement(parent, tag)
    node.text = str(value)

def prepare(case, run, variant, assembly, steps):
    source = REPO / 'tests/cases' / CASES[case]
    for item in source.iterdir():
        if item.name == 'mesh' or item.suffix == '.dat' or item.name == 'precomputed_velocity.vtu':
            (run/item.name).symlink_to(item, target_is_directory=item.is_dir())
    tree = ET.parse(source/'solver.xml')
    general = tree.getroot().find('GeneralSimulationParameters')
    for tag, value in {'Number_of_time_steps': steps, 'Increment_in_saving_VTK_files': steps,
                       'Increment_in_saving_restart_files': steps, 'Debug': 0}.items():
        put(general, tag, value)
    _, solver, prec, mesh_solver = variant
    for eq in tree.getroot().findall('Add_equation'):
        if case == 'scalar':
            put(eq, 'Max_iterations', 80)
        ls = eq.find('LS')
        mesh = eq.get('type').lower() == 'mesh'
        ls.set('type', mesh_solver if mesh else solver)
        la = ls.find('Linear_algebra')
        la.clear()
        la.set('type', 'trilinos')
        put(la, 'Assembly', assembly)
        put(la, 'Preconditioner', 'trilinos-diagonal' if mesh else 'trilinos-'+prec)
        # Keep original tolerances and caps. Explicit common mesh defaults allow
        # GMRES/CG comparisons without implicit solver-specific input defaults.
        if mesh:
            put(ls, 'Max_iterations', 1000)
        if ls.find('Krylov_space_dimension') is None:
            put(ls, 'Krylov_space_dimension', 50)
    tree.write(run/'solver.xml', encoding='utf-8', xml_declaration=True)

def measure(run, args, case, variant, assembly):
    command = [str(args.exe.resolve()), 'solver.xml']
    if args.launcher == 'srun':
        command = ['srun', '--nodes=1', '--exact', '--ntasks=1', '--cpus-per-task=1',
                   '--cpu-bind=cores', '--gpus-per-node=1', '--gpus-per-task=1',
                   '--gpu-bind=single:1'] + command
    env = os.environ.copy()
    for name in list(env):
        if name.startswith('SV_TRILINOS_'):
            del env[name]
    env.update(OMP_NUM_THREADS='1', KOKKOS_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', SV_TRILINOS_ASSEMBLY='auto',
               SV_TRILINOS_ASSEMBLY_STATS='1', SV_TRILINOS_RUNTIME_DETAILS='1',
               SV_TRILINOS_SOLVER_STATS='1')
    env.update(args.solver_env)
    start = time.perf_counter()
    timed_out = False
    with (run/'stdout.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=run, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = proc.returncode
    elapsed = time.perf_counter()-start
    history_path = run/'1-procs/histor.dat'
    history = history_path.read_text() if history_path.exists() else ''
    rows = [l for l in history.splitlines() if re.match(r'\s*\S+\s+\d+-\d+', l)]
    log = (run/'stdout.log').read_text(errors='replace')
    final = run/f'1-procs/result_{args.steps:03d}.vtu'
    record = dict(case=case, variant=variant[0], assembly=assembly, steps=args.steps,
                  command=command, wall_seconds=elapsed, returncode=code, timeout=timed_out,
                  completed=code == 0 and final.exists(), history_rows=len(rows),
                  simulation_seconds=float(rows[-1].split()[2]) if rows else None,
                  linear_failures=history.count('linear system solution has not converged'),
                  nonlinear_warnings=history.count('has exceeded the maximum'),
                  linear_iterations=sum(int(m.group(1)) for l in rows
                                        if (m := re.search(r'[\]!][ \t]+[\[!]\s*(\d+)', l))),
                  assembly_stats=[l for l in log.splitlines() if l.startswith(('ASSEMBLY_', 'TRILINOS_RUNTIME'))],
                  solver_env=args.solver_env, solve_stats=solve_totals(log))
    (run/'measurement.json').write_text(json.dumps(record, indent=2)+'\n')
    print(f'{case}/{variant[0]}/{assembly}/{run.name}: {elapsed:.3f}s completed={record["completed"]} linear_failures={record["linear_failures"]}', flush=True)

def solve_totals(log):
    """Sum GENERIC_SOLVE_STATS lines (preconditioner setup and Krylov time) per DOF count."""
    totals = {}
    for line in log.splitlines():
        if not line.startswith('GENERIC_SOLVE_STATS '):
            continue
        item = dict(re.findall(r'(\w+)=([-+\deE.]+)', line))
        key = 'dof'+item.get('dof', '?')
        entry = totals.setdefault(key, dict(solves=0, setup_s=0.0, krylov_s=0.0, iterations=0))
        entry['solves'] += 1
        entry['setup_s'] += float(item.get('setup_s', 0))
        entry['krylov_s'] += float(item.get('krylov_s', 0))
        entry['iterations'] += int(float(item.get('iterations', 0)))
        for counter in ('muelu_builds', 'muelu_refreshes', 'muelu_skips'):
            entry[counter] = int(float(item.get(counter, 0)))
    return totals

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    p.add_argument('--variants', nargs='+')
    p.add_argument('--assemblies', nargs='+', choices=['trilinos', 'fsils'], default=['trilinos', 'fsils'])
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--repeat', type=int, default=2)
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--launcher', choices=['local', 'srun'], default='local')
    p.add_argument('--env', action='append', default=[], metavar='SV_TRILINOS_NAME=VALUE',
                   help='Solver option passed to every run (all other SV_TRILINOS_* are cleared).')
    args = p.parse_args()
    args.solver_env = {}
    for item in args.env:
        name, sep, value = item.partition('=')
        if not sep or not name.startswith('SV_TRILINOS_'):
            p.error('--env expects SV_TRILINOS_NAME=VALUE')
        args.solver_env[name] = value
    if not args.exe.is_file() or min(args.steps, args.repeat, args.timeout) <= 0:
        p.error('Executable must exist; steps/repeat/timeout must be positive')
    if args.variants and not any(v[0] in args.variants for case in args.cases for v in variants(case)):
        p.error('No requested variant applies to the selected cases')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(platform=platform.platform(), python=platform.python_version(), steps=args.steps, repeat=args.repeat, cases=args.cases,
                    assemblies=args.assemblies, variants=args.variants, solver_env=args.solver_env,
                    executable=str(args.exe.resolve()),
                    sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(),
                    expected_groups=[(case, v[0], a) for case in args.cases for v in variants(case) if not args.variants or v[0] in args.variants for a in args.assemblies],
                    commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip())
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    # Interleave policies by repeat; all timings sequential, never concurrent.
    for repeat in range(1, args.repeat+1):
        for case in args.cases:
            for variant in variants(case):
                if args.variants and variant[0] not in args.variants:
                    continue
                for assembly in args.assemblies:
                    run = args.output/case/variant[0]/assembly/f'run-{repeat}'
                    run.mkdir(parents=True)
                    prepare(case, run, variant, assembly, args.steps)
                    measure(run, args, case, variant, assembly)

if __name__ == '__main__':
    main()
