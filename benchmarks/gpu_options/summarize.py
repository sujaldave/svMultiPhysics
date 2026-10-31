#!/usr/bin/env python3
"""Summarize run_options.sh output: timing, iterations and final-field agreement.

    summarize.py OUTPUT_DIR [--reference-fluid A_old] [--reference-physics ml_old]

Writes OUTPUT_DIR/report.md and report.json. Speedup is reference wall time /
variant wall time. Field differences are the largest relative L2 difference over
all saved point fields against the reference variant of the same case (needs
Python VTK; otherwise reported as unavailable).
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trilinos_comparison'))
import compare_builds  # noqa: E402  (shared field comparison)

HAVE_VTK = importlib.util.find_spec('vtk') is not None


def fluid_row(run_dir):
    path = run_dir / 'run-1' / 'measurement.json'
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    jac = d.get('jacobians', [])
    total = lambda key: sum(j.get(key, 0.0) for j in jac)
    assembly = sum(float(r.get('total_s', 0)) for r in d.get('assembly_stats', [])
                   if r.get('kind') == 'ASSEMBLY_STATS')
    cg_it, cg_s, calls = total('cg_iterations'), total('cg_s'), total('cg_calls')
    vtu = run_dir / 'run-1' / f"{d['ranks']}-procs" / f"result_{d['steps']:03d}.vtu"
    return dict(wall_s=d['wall_seconds'], simulation_s=d.get('simulation_seconds'),
                completed=d.get('completed'), newton=d.get('nonlinear_iterations'),
                cg_iterations=cg_it, cg_per_call=cg_it / calls if calls else None,
                cg_s=cg_s, us_per_cg=cg_s / cg_it * 1e6 if cg_it else None,
                schur_setup_s=total('schur_setup_s'), amg_setup_s=total('pressure_setup_s'),
                gmres_s=total('gmres_s'), bipn_s=total('total_s'), assembly_s=assembly,
                cg_failures=total('cg_failures'),
                ri_failures=sum(j.get('ri_success', 1) == 0 for j in jac),
                product_cached=bool(jac and jac[-1].get('schur_product_cached')),
                amg_skips=jac[-1].get('amg_skips') if jac else None,
                vtu=str(vtu) if vtu.exists() else None)


def physics_rows(label_dir):
    rows = {}
    for path in sorted(label_dir.glob('*/*/*/run-1/measurement.json')):
        d = json.loads(path.read_text())
        stats = d.get('solve_stats', {})
        vtu = path.parent / '1-procs' / f"result_{d['steps']:03d}.vtu"
        rows[d['case']] = dict(
            wall_s=d['wall_seconds'], simulation_s=d.get('simulation_seconds'),
            completed=d.get('completed'), timeout=d.get('timeout'),
            linear_failures=d.get('linear_failures'), linear_iterations=d.get('linear_iterations'),
            setup_s=sum(s['setup_s'] for s in stats.values()),
            krylov_s=sum(s['krylov_s'] for s in stats.values()),
            muelu_builds=sum(s.get('muelu_builds', 0) for s in stats.values()),
            muelu_refreshes=sum(s.get('muelu_refreshes', 0) for s in stats.values()),
            vtu=str(vtu) if d.get('completed') and vtu.exists() else None)
    return rows


def field_difference(reference, candidate):
    if not HAVE_VTK:
        return None, 'VTK unavailable'
    if not reference or not candidate:
        return None, 'no final state'
    try:
        values = compare_builds.compare_fields(Path(reference), Path(candidate))
    except Exception as error:  # report and continue
        return None, f'error: {error}'
    name, value = max(values.items(), key=lambda kv: kv[1])
    return value, name


def fmt(value, digits=1):
    if value is None:
        return '—'
    if isinstance(value, float) and value != 0 and abs(value) < 1e-3:
        return f'{value:.1e}'
    return f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference-fluid', default='A_old')
    parser.add_argument('--reference-physics', default='ml_old')
    args = parser.parse_args()
    root = args.output.resolve()
    report, lines = {'fluid': {}, 'physics': {}}, ['# Solver-option variants', '']
    failed = (root / 'failed.txt').read_text().split() if (root / 'failed.txt').exists() else []
    if failed:
        lines += ['Failed launches: ' + ', '.join(failed), '']

    for case_dir in sorted((root / 'fluid').glob('*')):
        rows = {p.name: fluid_row(p) for p in sorted(case_dir.iterdir()) if p.is_dir()}
        ref = rows.get(args.reference_fluid)
        lines += [f'## Fluid: {case_dir.name}', '',
                  '| Variant | Wall s | Speedup | Newton | CG iters | CG/call | µs/CG iter | CG s | '
                  'Schur setup s | AMG setup s | GMRES s | Assembly s | Cached BᵀB | Failures | Max field diff |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|']
        for label, row in rows.items():
            if row is None or not row['completed']:
                lines.append(f'| {label} | FAILED (see stdout.log) |' + ' |' * 13)
                continue
            row['field_diff'], row['field'] = field_difference(ref and ref['vtu'], row['vtu'])
            speed = ref['wall_s'] / row['wall_s'] if ref and ref['completed'] and row['wall_s'] else None
            lines.append(
                f"| {label} | {fmt(row['wall_s'])} | {fmt(speed, 2)} | {row['newton']} | "
                f"{fmt(row['cg_iterations'], 0)} | {fmt(row['cg_per_call'])} | {fmt(row['us_per_cg'], 0)} | "
                f"{fmt(row['cg_s'])} | {fmt(row['schur_setup_s'], 2)} | {fmt(row['amg_setup_s'], 2)} | "
                f"{fmt(row['gmres_s'], 2)} | {fmt(row['assembly_s'], 2)} | {'yes' if row['product_cached'] else 'no'} | "
                f"{fmt(row['cg_failures'], 0)}/{row['ri_failures']} | "
                f"{fmt(row['field_diff'])} ({row['field']}) |")
        lines.append('')
        report['fluid'][case_dir.name] = rows

    labels = sorted(p.name for p in (root / 'physics').glob('*') if p.is_dir())
    physics = {label: physics_rows(root / 'physics' / label) for label in labels}
    cases = sorted({case for rows in physics.values() for case in rows})
    for case in cases:
        ref = physics.get(args.reference_physics, {}).get(case)
        lines += [f'## Generic solve (GMRES + MueLu): {case}', '',
                  '| Variant | Wall s | Speedup | Completed | Linear iters | Linear failures | '
                  'Precond setup s | Krylov s | MueLu builds/refreshes | Max field diff |',
                  '|---|---:|---:|---|---:|---:|---:|---:|---|---|']
        for label in labels:
            row = physics[label].get(case)
            if row is None:
                continue
            row['field_diff'], row['field'] = field_difference(ref and ref['vtu'], row['vtu'])
            speed = ref['wall_s'] / row['wall_s'] if ref and ref['completed'] and row['completed'] else None
            status = 'timeout' if row['timeout'] else ('yes' if row['completed'] else 'no')
            lines.append(
                f"| {label} | {fmt(row['wall_s'])} | {fmt(speed, 2)} | {status} | {row['linear_iterations']} | "
                f"{row['linear_failures']} | {fmt(row['setup_s'], 2)} | {fmt(row['krylov_s'], 2)} | "
                f"{row['muelu_builds']}/{row['muelu_refreshes']} | {fmt(row['field_diff'])} ({row['field']}) |")
        lines.append('')
    report['physics'] = physics
    (root / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
