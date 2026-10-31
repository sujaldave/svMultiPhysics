#!/usr/bin/env python3
"""Compare a rerun of all A40 suites (submit_all.sh) against the earlier results.

    compare_builds.py NEW_ROOT --old-root OLD_ROOT [--old SUITE=DIR ...]

NEW_ROOT holds schur/, pc/, pipe/, scale/, physics/ from submit_all.sh. OLD_ROOT
holds the earlier job directories; --old overrides a default directory name.
Writes NEW_ROOT/comparison.md and comparison.json.

Two comparisons are reported:
  * same-allocation: old and new executables in one job (schur, pipe). This
    isolates the build because node, GPU and time are shared.
  * historical: new suite results against the earlier jobs. These also include
    node/time variation and any source changes between the two binaries.
Final fields of the new build are compared with the old build for every
matching run. Requires Python VTK for field checks and for refreshing the
summaries; timing tables need only the standard library.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

BENCH = Path(__file__).resolve().parents[1]
OLD_DEFAULTS = {'schur': 'sherlock-49893846', 'pc': 'preconditioners-49978759',
                'pipe': 'large-pipe-50440494', 'scale': 'scaling-49899116',
                'physics': 'multiphysics-50018693'}
AB_PAIRS = {'schur': [('.', 'old', 'reuse-explicit')],
            'pipe': [('pipe', 'old-amg-resistance', 'gpu-amg-resistance'),
                     ('iliac', 'old-amg-resistance', 'gpu-amg-resistance')]}
SKIP_GROUPS = {'runtime-details', 'profile'}
ARCH_WARNING = 'compute capability 8.0 on device with compute capability 8.6'


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def ok(run):
    return not run.get('returncode') and run.get('completed', True) and not run.get('timeout')


def metrics(runs):
    """Median metrics over completed solver-benchmark runs."""
    runs = [r for r in runs if ok(r)]
    if not runs:
        return None
    total = lambda key: median(r.get('solver_totals', {}).get(key) for r in runs)
    cg_s, cg_iter = total('cg_s'), total('cg_iterations')
    return dict(repeats=len(runs), wall_s=median(r['wall_seconds'] for r in runs),
                simulation_s=median(r.get('simulation_seconds') for r in runs),
                newton=median(r.get('nonlinear_iterations') for r in runs),
                cg_iterations=cg_iter, cg_s=cg_s,
                us_per_cg_iter=cg_s / cg_iter * 1e6 if cg_s and cg_iter else None,
                schur_setup_s=total('schur_setup_s'), amg_setup_s=total('pressure_setup_s'),
                gmres_s=total('gmres_s'), bipn_s=total('total_s'),
                assembly_s=median(r.get('phase_totals', {}).get('ASSEMBLY_STATS', {}).get('total_s')
                                  for r in runs),
                cg_failures=total('cg_failures'))


def load_solver(directory):
    """Groups of a solver_optimization-style summary.json -> {label: (metrics, runs)}."""
    path = directory / 'summary.json'
    if not path.exists():
        return {}
    groups = json.loads(path.read_text())['groups']
    return {label: (metrics(g['runs']), g['runs']) for label, g in groups.items()
            if label not in SKIP_GROUPS}


def load_scaling(directory):
    path = directory / 'scaling.json'
    if not path.exists():
        return {}
    groups = json.loads(path.read_text())['groups']
    return {f'gpus-{n}': (metrics(g['runs']), g['runs']) for n, g in groups.items()}


def load_physics(directory):
    path = directory / 'summary.json'
    if not path.exists():
        return {}
    result = {}
    for g in json.loads(path.read_text())['groups']:
        label = f"{g['case']}/{g['variant']}/{g['assembly']}"
        runs = g['runs']
        m = dict(repeats=sum(ok(r) for r in runs), wall_s=g.get('wall_seconds'),
                 simulation_s=g.get('simulation_seconds'), linear_iterations=g.get('linear_iterations'),
                 linear_failures=g.get('linear_failures'), validated=g.get('validated'),
                 timeouts=sum(bool(r.get('timeout')) for r in runs))
        result[label] = (m, runs)
    return result


def result_vtu(run_dir, record):
    return run_dir / f"{record.get('ranks', 1)}-procs" / f"result_{record['steps']:03d}.vtu"


def first_result(directory, label, runs):
    """Final VTU of the first completed run of a group."""
    for index, run in enumerate(runs, start=1):
        path = result_vtu(directory / label / f'run-{index}', run)
        if ok(run) and path.exists():
            return path
    return None


def compare_fields(reference, candidate):
    """Relative L2 difference of every common point array, matched by coordinates."""
    import vtk
    def read(path):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(path))
        reader.Update()
        return reader.GetOutput()
    a, b = read(reference), read(candidate)
    if a.GetNumberOfPoints() != b.GetNumberOfPoints() or not a.GetNumberOfPoints():
        raise ValueError('empty grid or different point counts')
    count = a.GetNumberOfPoints()
    if all(a.GetPoint(i) == b.GetPoint(i) for i in range(count)):
        order = range(count)
    else:
        # Different partitions write points in different orders. Matching by
        # coordinates is valid only without coincident points (FSI interfaces).
        lookup = {b.GetPoint(i): i for i in range(count)}
        if len(lookup) != count:
            raise ValueError('point order differs and coordinates are not unique')
        order = [lookup[a.GetPoint(i)] for i in range(count)]
    result = {}
    for k in range(a.GetPointData().GetNumberOfArrays()):
        av = a.GetPointData().GetArray(k)
        bv = b.GetPointData().GetArray(av.GetName())
        if bv is None or av.GetNumberOfComponents() != bv.GetNumberOfComponents():
            continue
        norm, error = [], []
        for i, j in enumerate(order):
            for x, y in zip(av.GetTuple(i), bv.GetTuple(j)):
                norm.append(x * x)
                error.append((y - x) ** 2)
        n, e = math.sqrt(math.fsum(norm)), math.sqrt(math.fsum(error))
        result[av.GetName()] = e / n if n else e
    return result


def tolerance(suite, label):
    if suite == 'physics' and not label.startswith('fsi/'):
        return 1e-4
    return 1e-7


def informational(label, *groups):
    """Capped or unconverged policies may legitimately follow a different
    iteration path in another Trilinos version; report, but do not fail."""
    for m in groups:
        if m and (m.get('validated') is False or (m.get('cg_failures') or 0) > 0 or m.get('timeouts')):
            return True
    return 'classic' in label


def refresh_summaries(suite, directory, log):
    """Rerun the suite's own summarizer with this Python (adds VTK field checks)."""
    commands = []
    if suite == 'schur':
        ref = ['--reference-group', 'old'] if (directory / 'old').is_dir() else []
        commands.append([BENCH / 'solver_optimization/summarize.py', directory, '--expect-cuda'] + ref)
    elif suite == 'pc':
        commands.append([BENCH / 'solver_optimization/summarize.py', directory, '--reference-group',
                         'amg-diagonal', '--expect-cuda', '--allow-diagonal-inner-nonconvergence'])
    elif suite == 'pipe':
        for case in ('pipe', 'iliac'):
            if (directory / case).is_dir():
                ref = 'old-amg-resistance' if (directory / case / 'old-amg-resistance').is_dir() \
                    else 'gpu-amg-resistance'
                commands.append([BENCH / 'solver_optimization/summarize.py', directory / case,
                                 '--reference-group', ref, '--expect-cuda',
                                 '--allow-diagonal-inner-nonconvergence'])
    elif suite == 'scale':
        commands.append([BENCH / 'gpu_scaling/summarize_scaling.py', directory, '--expect-cuda'])
    elif suite == 'physics':
        commands.append([BENCH / 'multiphysics/summarize.py', directory, '--expect-cuda'])
    for command in commands:
        result = subprocess.run([sys.executable] + [str(c) for c in command],
                                capture_output=True, text=True)
        log.append({'command': ' '.join(map(str, command)), 'returncode': result.returncode,
                    'tail': (result.stdout + result.stderr).strip().splitlines()[-6:]})


def arch_warnings(directory):
    logs = [Path(top) / 'stdout.log' for top, _, files in os.walk(directory, followlinks=True)
            if 'stdout.log' in files]
    return dict(logs=len(logs), with_sm80_on_sm86_warning=sum(ARCH_WARNING in p.read_text(errors='replace')
                                                               for p in logs))


def first_hash(path):
    try:
        token = path.read_text().split()[0]
    except (OSError, IndexError):
        return None
    return token if len(token) == 64 else None


def historical_binary(suite, old_dir, report):
    """Whether the earlier job ran the binary that the old build holds now."""
    config = old_dir.parent / (old_dir.name + '-config') if suite == 'physics' else old_dir / 'config'
    then = first_hash(config / 'executable.sha256')
    now = next((line.split()[0] for line in report.get('build-old.txt', '').splitlines()
                if len(line.split()[0:1]) and len(line.split()[0]) == 64), None)
    if not then or not now:
        return 'unknown'
    return 'same binary as the current old build' if then == now else \
        'different binary from the current old build (source/build changed since)'


def ratio(old, new):
    return old / new if old and new else None


def fmt(value, digits=1):
    if value is None:
        return '—'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, float) and (abs(value) < 1e-3 and value != 0):
        return f'{value:.1e}'
    return f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('new_root', type=Path)
    parser.add_argument('--old-root', type=Path, required=True)
    parser.add_argument('--old', action='append', default=[], metavar='SUITE=DIR')
    parser.add_argument('--no-refresh', action='store_true', help='Do not rerun suite summarizers.')
    parser.add_argument('--no-fields', action='store_true', help='Skip final-field comparisons.')
    args = parser.parse_args()
    new_root, old_root = args.new_root.resolve(), args.old_root.resolve()
    old_dirs = {s: old_root / d for s, d in OLD_DEFAULTS.items()}
    for item in args.old:
        suite, _, path = item.partition('=')
        old_dirs[suite] = Path(path).resolve()
    have_vtk = importlib.util.find_spec('vtk') is not None
    report = {'new_root': str(new_root), 'old_dirs': {k: str(v) for k, v in old_dirs.items()},
              'vtk': have_vtk, 'summarizers': [], 'suites': {}}
    for name in ('build-new.txt', 'build-old.txt'):
        if (new_root / name).exists():
            report[name] = (new_root / name).read_text()

    for suite in OLD_DEFAULTS:
        new_dir, old_dir = new_root / suite, old_dirs[suite]
        if not new_dir.is_dir():
            continue
        if have_vtk and not args.no_refresh:
            refresh_summaries(suite, new_dir, report['summarizers'])
        kind = {'scale': 'scaling', 'physics': 'physics'}.get(suite, 'solver')
        loader = {'scaling': load_scaling, 'physics': load_physics, 'solver': load_solver}[kind]
        parts = ['pipe', 'iliac'] if suite == 'pipe' else ['.']
        entry = report['suites'][suite] = {'historical': [], 'same_allocation': [],
                                           'arch_new': arch_warnings(new_dir),
                                           'arch_old': arch_warnings(old_dir) if old_dir.is_dir() else None,
                                           'historical_binary': historical_binary(suite, old_dir, report)}
        for part in parts:
            new_groups = loader(new_dir / part)
            old_groups = loader(old_dir / part) if old_dir.is_dir() else {}
            prefix = '' if part == '.' else part + '/'
            baseline_labels = {o for p, o, _ in AB_PAIRS.get(suite, []) if p == part}
            for label in sorted(set(new_groups) - baseline_labels):
                new_m, new_runs = new_groups[label]
                row = {'group': prefix + label, 'new': new_m}
                if label in old_groups:
                    old_m, old_runs = old_groups[label]
                    row['old'] = old_m
                    if not args.no_fields and have_vtk and new_m and old_m:
                        a = first_result(old_dir / part, label, old_runs)
                        b = first_result(new_dir / part, label, new_runs)
                        row['fields'] = field_row(a, b, suite, label, informational(label, old_m, new_m))
                entry['historical'].append(row)
            for ab_part, old_label, new_label in AB_PAIRS.get(suite, []):
                if ab_part != part or old_label not in new_groups or new_label not in new_groups:
                    continue
                (old_m, old_runs), (new_m, new_runs) = new_groups[old_label], new_groups[new_label]
                row = {'group': prefix + new_label, 'old': old_m, 'new': new_m}
                if not args.no_fields and have_vtk and old_m and new_m:
                    row['fields'] = field_row(first_result(new_dir / part, old_label, old_runs),
                                              first_result(new_dir / part, new_label, new_runs),
                                              suite, new_label, informational(new_label, old_m, new_m))
                entry['same_allocation'].append(row)

    (new_root / 'comparison.json').write_text(json.dumps(report, indent=1) + '\n')
    (new_root / 'comparison.md').write_text(markdown(report))
    print((new_root / 'comparison.md').read_text())


def field_row(reference, candidate, suite, label, info):
    if reference is None or candidate is None:
        return {'status': 'no completed run with a final VTU'}
    try:
        values = compare_fields(reference, candidate)
    except Exception as error:  # Report and keep comparing the other groups.
        return {'status': f'error: {error}'}
    worst = max(values.items(), key=lambda kv: kv[1]) if values else (None, None)
    limit = tolerance(suite, label)
    passed = worst[1] is not None and worst[1] <= limit
    return {'status': 'passed' if passed else ('differs (informational)' if info else 'FAILED'),
            'tolerance': limit, 'worst_field': worst[0], 'worst_relative_l2': worst[1],
            'velocity': values.get('Velocity'), 'pressure': values.get('Pressure'), 'all': values}


def markdown(report):
    out = ['# Old build vs new build', '',
           'Speedup = old / new wall time (above 1 means the new build is faster). '
           'Phase times are sums of per-Jacobian maximum-rank times.', '']
    for name in ('build-old.txt', 'build-new.txt'):
        if name in report:
            out += [f'<details><summary>{name}</summary>', '', '```', report[name].strip(), '```', '', '</details>', '']
    for suite, entry in report['suites'].items():
        out += [f'## {suite}', '']
        arch = entry['arch_new']
        old_arch = entry['arch_old'] or {}
        out.append(f"SM 8.0-on-8.6 Kokkos warning: new build {arch['with_sm80_on_sm86_warning']}/{arch['logs']} "
                   f"logs; old {old_arch.get('with_sm80_on_sm86_warning', '—')}/{old_arch.get('logs', '—')}.")
        out.append(f"Earlier job vs current old build: {entry['historical_binary']}.")
        out.append('')
        for key, title in (('same_allocation', 'Same allocation (old and new executables in one job)'),
                           ('historical', 'Against the earlier job')):
            rows = entry[key]
            if not rows:
                continue
            physics = suite == 'physics'
            out += [f'### {title}', '']
            if physics:
                out += ['| Group | Old wall s | New wall s | Speedup | Old valid | New valid | New timeouts | Fields |',
                        '|---|---:|---:|---:|---|---|---:|---|']
            else:
                out += ['| Group | Old wall s | New wall s | Speedup | Newton old/new | CG iters old/new | '
                        'µs/CG iter old/new | CG s | Schur setup s | AMG setup s | GMRES s | Assembly s | Fields |',
                        '|---|---:|---:|---:|---|---|---|---|---|---|---|---|---|']
            for row in rows:
                o, n = row.get('old') or {}, row.get('new') or {}
                f = row.get('fields', {})
                field = f.get('status', '—')
                if f.get('worst_relative_l2') is not None:
                    field += f" (max {f['worst_field']} {f['worst_relative_l2']:.1e})"
                pair = lambda k, d=1: f"{fmt(o.get(k), d)} / {fmt(n.get(k), d)}"
                if not n:
                    out.append(f"| {row['group']} | {fmt(o.get('wall_s'))} | FAILED |" + ' |' * (5 if physics else 10))
                elif physics:
                    out.append(f"| {row['group']} | {fmt(o.get('wall_s'))} | {fmt(n.get('wall_s'))} | "
                               f"{fmt(ratio(o.get('wall_s'), n.get('wall_s')), 2)} | {fmt(o.get('validated'))} | "
                               f"{fmt(n.get('validated'))} | {n.get('timeouts')} | {field} |")
                else:
                    out.append(f"| {row['group']} | {fmt(o.get('wall_s'))} | {fmt(n.get('wall_s'))} | "
                               f"{fmt(ratio(o.get('wall_s'), n.get('wall_s')), 2)} | {pair('newton', 0)} | "
                               f"{pair('cg_iterations', 0)} | {pair('us_per_cg_iter', 0)} | {pair('cg_s')} | "
                               f"{pair('schur_setup_s', 2)} | {pair('amg_setup_s', 2)} | {pair('gmres_s', 2)} | "
                               f"{pair('assembly_s', 2)} | {field} |")
            out.append('')
    if report['summarizers']:
        out += ['## Summarizer reruns', '']
        for item in report['summarizers']:
            out.append(f"- `{item['command']}` → exit {item['returncode']}: " + ' / '.join(item['tail'][-2:]))
        out.append('')
    if not report['vtk']:
        out += ['Field checks skipped: run with a Python that has VTK.', '']
    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    main()
