#!/usr/bin/env python3
"""Summarize Marlowe H100 scaling jobs (marlowe_h100_scaling.sbatch).

Pass one or more job output directories, or a campaign directory that contains
them (every directory holding a sweep.json is a job). The 1-node (1/4/8 GPU)
and 2-node (16 GPU) jobs are combined into one table per case:

    case | binding | communication | GPUs | median wall | pressure-CG ms/iteration | speedups

Speedups use the 1-GPU median of the same binding mode and, separately, the
fastest 1-GPU median of the case. Runs are validated as in summarize_scaling.py
(completed, solver statistics present, no CG failures, CUDA assembly). The
binding probe of each (binding, GPU count) is validated as well; `numa` must
show every rank on its GPU's NUMA-local CPUs. With Python VTK available the
final fields of every run are compared with the 1-GPU result of the same case.

Writes marlowe_scaling.{md,csv,json} into --output (default: first directory).
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

import check_binding
from summarize_scaling import FIELDS, read_run


def job_roots(paths):
    roots = []
    for path in paths:
        path = path.resolve()
        found = [path] if (path/'sweep.json').is_file() else sorted(p.parent for p in path.rglob('sweep.json'))
        if not found:
            raise SystemExit(f'No sweep.json under {path}')
        roots += found
    return list(dict.fromkeys(roots))


def median_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def cg_ms_per_iteration(run):
    totals = run['solver_totals']
    iterations = totals.get('cg_iterations', 0)
    return 1e3*totals['cg_s']/iterations if iterations and 'cg_s' in totals else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expect-cuda', action='store_true')
    args = parser.parse_args()
    roots = job_roots(args.directories)
    output = (args.output or roots[0]).resolve()
    output.mkdir(parents=True, exist_ok=True)

    failures, groups, steps_by_case, binding = [], {}, {}, {}
    for root in roots:
        manifest = json.loads((root/'sweep.json').read_text())
        for case, steps in manifest['steps'].items():
            if case in manifest['cases']:
                if steps_by_case.setdefault(case, steps) != steps:
                    failures.append(f'{root}: {case} uses {steps} steps, other jobs {steps_by_case[case]}')
        for probe in sorted(p for p in (root/'binding').glob('*/gpus-*') if re.fullmatch(r'gpus-\d+', p.name)):
            bind, gpus = probe.parent.name, int(probe.name.split('-')[-1])
            try:
                check_binding.validate(probe, gpus, multi_node=True, require_numa_local=(bind == 'numa'))
                rows = [json.loads(p.read_text()) for p in probe.glob('rank-*.json')]
                status = f"ok: {len({r['hostname'] for r in rows})} node(s)"
                if bind != 'numa':
                    local = sum(bool(r.get('affinity_within_gpu_numa')) for r in rows)
                    status += f', {local}/{len(rows)} ranks NUMA-local'
            except (RuntimeError, KeyError) as error:
                status = f'FAILED: {error}'
                failures.append(f'{probe}: {error}')
            binding[(bind, gpus)] = status
        for path in sorted(root.glob('*/*/gpus-*/run-*/measurement.json')):
            if not re.fullmatch(r'gpus-\d+', path.parent.parent.name):
                continue   # partial results set aside after a preemption
            variant_dir = path.parent.parent.parent
            case, variant = variant_dir.parent.name, variant_dir.name
            if variant == 'warmup' or '__' not in variant:
                continue
            bind, comm = variant.split('__', 1)
            gpus = int(path.parent.parent.name.split('-')[-1])
            data = read_run(path, args.expect_cuda, failures)
            data['nodes'] = manifest['nodes']
            data['measurement'] = str(path)
            if data['ranks'] != gpus:
                failures.append(f'{path}: rank count differs from directory label')
                data['valid'] = False
            groups.setdefault((case, bind, comm, gpus), []).append(data)
        failed_list = root/'failed.txt'
        if failed_list.is_file():
            for line in failed_list.read_text().split():
                failures.append(f'failed or timed out: {line}')
    if not groups:
        parser.error('No */<binding>__<communication>/gpus-*/run-*/measurement.json files found')

    rows = []
    for (case, bind, comm, gpus), runs in sorted(groups.items()):
        good = [r for r in runs if r['valid']]
        rows.append(dict(case=case, binding=bind, communication=comm, gpus=gpus,
            nodes=sorted({r['nodes'] for r in runs}), valid_repeats=len(good), runs=len(runs),
            timeouts=sum(bool(r.get('timed_out')) for r in runs),
            median_wall_s=median_or_none([r['wall_seconds'] for r in good]),
            min_wall_s=min((r['wall_seconds'] for r in good), default=None),
            max_wall_s=max((r['wall_seconds'] for r in good), default=None),
            median_simulation_s=median_or_none([r['simulation_seconds'] for r in good]),
            median_solver_s=median_or_none([r['solver_totals'].get('total_s') for r in good]),
            median_cg_s=median_or_none([r['solver_totals'].get('cg_s') for r in good]),
            median_gmres_s=median_or_none([r['solver_totals'].get('gmres_s') for r in good]),
            median_cg_iterations=median_or_none([r['solver_totals'].get('cg_iterations') for r in good]),
            cg_ms_per_iteration=median_or_none([cg_ms_per_iteration(r) for r in good]),
            nonlinear_iterations=sorted({r['nonlinear_iterations'] for r in good}),
            gpu_aware_mpi=sorted({r.get('gpu_aware_mpi', 'auto') for r in runs}),
            ucx_tls=sorted({r.get('communication', {}).get('UCX_TLS', 'default') for r in runs}),
            binding_status=binding.get((bind, gpus), 'no probe record'),
            measurements=[str(Path(r['measurement'])) for r in runs]))

    for row in rows:
        singles = [r for r in rows if r['case'] == row['case'] and r['gpus'] == 1 and r['median_wall_s']]
        same = next((r for r in singles if r['binding'] == row['binding']), None)
        best = min(singles, key=lambda r: r['median_wall_s'], default=None)
        wall = row['median_wall_s']
        row['speedup_same_binding'] = same['median_wall_s']/wall if same and wall else None
        row['efficiency_same_binding'] = row['speedup_same_binding']/row['gpus'] if row['speedup_same_binding'] else None
        row['speedup_vs_best_1gpu'] = best['median_wall_s']/wall if best and wall else None
        row['gpu_seconds'] = row['gpus']*wall if wall else None

    field_checks = {}
    if importlib.util.find_spec('vtk') is None:
        field_status = 'pending: rerun this script locally with Python VTK'
    else:
        compare = Path(__file__).resolve().parents[1]/'schur_amg/compare_fields.py'
        for case in sorted({key[0] for key in groups}):
            references = [r for key, runs in sorted(groups.items()) if key[0] == case and key[3] == 1
                          for r in runs if r['valid']]
            if not references:
                failures.append(f'{case}: no valid 1-GPU reference for field comparison')
                continue
            ref = references[0]
            ref_file = Path(ref['measurement']).parent/f"1-procs/result_{ref['steps']:03d}.vtu"
            for key, runs in sorted(groups.items()):
                if key[0] != case:
                    continue
                for run in runs:
                    if not run['valid']:
                        continue
                    candidate = Path(run['measurement']).parent/f"{run['ranks']}-procs/result_{run['steps']:03d}.vtu"
                    process = subprocess.run([sys.executable, str(compare), str(ref_file), str(candidate),
                                              '--fields', *FIELDS], capture_output=True, text=True)
                    if process.returncode:
                        failures.append(f"{run['measurement']}: field comparison failed")
                        field_checks[run['measurement']] = {'error': process.stdout+process.stderr}
                    else:
                        field_checks[run['measurement']] = json.loads(process.stdout)
        field_status = 'completed (reference: first valid 1-GPU run of each case)'

    columns = ['case','binding','communication','gpus','valid_repeats','median_wall_s','min_wall_s','max_wall_s',
               'median_simulation_s','median_solver_s','median_cg_s','median_gmres_s','median_cg_iterations',
               'cg_ms_per_iteration','speedup_same_binding','efficiency_same_binding','speedup_vs_best_1gpu',
               'gpu_seconds','timeouts','binding_status']
    with (output/'marlowe_scaling.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    (output/'marlowe_scaling.json').write_text(json.dumps(dict(
        jobs=[str(r) for r in roots], rows=rows, failures=failures,
        field_check_status=field_status, field_checks=field_checks), indent=2)+'\n')

    def fmt(value, digits=1, percent=False):
        if value is None:
            return '-'
        return f'{100*value:.0f}%' if percent else f'{value:.{digits}f}'
    lines = []
    for case in sorted({r['case'] for r in rows}):
        lines += [f'## {case}', '',
                  '| Binding | Communication | GPUs | Wall (s) median [min-max] | CG ms/it | CG its | '
                  'GMRES (s) | Speedup (same binding) | Eff. | vs best 1 GPU | Repeats | Binding probe |',
                  '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for r in sorted((r for r in rows if r['case'] == case),
                        key=lambda r: (r['gpus'], r['communication'], r['binding'])):
            lines.append(f"| {r['binding']} | {r['communication']} | {r['gpus']} | "
                         f"{fmt(r['median_wall_s'])} [{fmt(r['min_wall_s'])}-{fmt(r['max_wall_s'])}] | "
                         f"{fmt(r['cg_ms_per_iteration'], 2)} | {fmt(r['median_cg_iterations'], 0)} | "
                         f"{fmt(r['median_gmres_s'])} | {fmt(r['speedup_same_binding'], 2)} | "
                         f"{fmt(r['efficiency_same_binding'], percent=True)} | {fmt(r['speedup_vs_best_1gpu'], 2)} | "
                         f"{r['valid_repeats']}/{r['runs']} | {r['binding_status']} |")
        lines.append('')
    lines += [field_status, '',
              'CG ms/it = pressure-CG seconds / pressure-CG iterations (summed over Jacobians, maximum rank).',
              'Efficiency = (T1/TN)/N against the 1-GPU run with the same binding mode.',
              'Wall time includes mpirun launch, setup and output of the final step.']
    if failures:
        lines += ['', 'Validation failures:', *['- '+f for f in failures]]
    (output/'marlowe_scaling.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
