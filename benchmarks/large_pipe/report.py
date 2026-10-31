#!/usr/bin/env python3
"""Print one Markdown table from summary.json files written by
benchmarks/solver_optimization/summarize.py.

Usage: report.py DIR [DIR ...]   (each DIR contains summary.json)
Times are medians over completed repeats. Solver phases are sums of the
per-Jacobian maximum-rank times, so they are diagnostic, not additive.
"""
import json
from pathlib import Path
import statistics
import sys


def median(runs, get):
    values = [get(r) for r in runs]
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=1):
    return '—' if value is None else f'{value:.{digits}f}'


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print('| Case | Group | Ranks | Repeats | Wall s | Simulation s | Newton | '
          'CG iters | CG/call | CG s | µs/CG iter | Schur setup s | AMG setup s | '
          'GMRES s | Assembly s | CG failures |')
    print('|' + '---|' * 16)
    for directory in map(Path, sys.argv[1:]):
        path = directory / 'summary.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        for label, group in sorted(summary['groups'].items()):
            if label in ('runtime-details', 'profile'):
                continue
            runs = [r for r in group['runs'] if not r['returncode'] and r['completed']]
            if not runs:
                print(f'| {directory.name} | {label} | | 0 | FAILED |' + ' |' * 11)
                continue
            total = lambda key: (lambda r: r['solver_totals'].get(key))
            cg_iters = median(runs, total('cg_iterations'))
            cg_calls = median(runs, total('cg_calls'))
            cg_s = median(runs, total('cg_s'))
            assembly = median(runs, lambda r: r['phase_totals'].get('ASSEMBLY_STATS', {}).get('total_s'))
            failures = median(runs, total('cg_failures'))
            print(f"| {directory.name} | {label} | {runs[0]['ranks']} | {len(runs)} | "
                  f"{fmt(median(runs, lambda r: r['wall_seconds']))} | "
                  f"{fmt(median(runs, lambda r: r['simulation_seconds']))} | "
                  f"{fmt(median(runs, lambda r: r['nonlinear_iterations']), 0)} | "
                  f"{fmt(cg_iters, 0)} | {fmt(cg_iters / cg_calls if cg_iters and cg_calls else None)} | "
                  f"{fmt(cg_s)} | {fmt(cg_s / cg_iters * 1e6 if cg_s and cg_iters else None, 0)} | "
                  f"{fmt(median(runs, total('schur_setup_s')), 2)} | "
                  f"{fmt(median(runs, total('pressure_setup_s')), 2)} | "
                  f"{fmt(median(runs, total('gmres_s')), 2)} | {fmt(assembly, 2)} | {fmt(failures, 0)} |")
    print('\nFailures/warnings are listed in each summary.json.')


if __name__ == '__main__':
    main()
