#!/usr/bin/env python3
"""Summarize an isolated CPU/GPU comparison; VTK enables final-field checks.

Can be rerun locally after downloading the output directory from Sherlock.
Requires only the Python standard library for timing/residual checks.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--expect-cuda', action='store_true')
    parser.add_argument('--reference-group', help='Directory label used for field and timing comparisons.')
    parser.add_argument('--allow-diagonal-inner-nonconvergence', action='store_true',
                        help='Report unconverged diagonal-pressure inner solves as warnings; outer/field checks remain strict.')
    args = parser.parse_args()
    directory = args.directory.resolve()
    groups = {}
    failures = []
    warnings = []
    for path in sorted(directory.glob('*/run-*/measurement.json')):
        data = json.loads(path.read_text())
        label = path.parent.parent.name
        data['path'] = str(path.relative_to(directory))
        phases = {}
        for row in data['assembly_stats']:
            target = phases.setdefault(row['kind'], {})
            for key, value in row.items():
                if key.endswith('_s') or key == 'rebuilt':
                    target[key] = target.get(key, 0.0)+float(value)
        data['phase_totals'] = phases
        data['solver_totals'] = {
            key: sum(row.get(key, 0.0) for row in data['jacobians'])
            for key in {key for row in data['jacobians'] for key in row}
            if key.endswith('_s') or key.endswith('_iterations') or key.endswith('_calls') or key.endswith('_failures')}
        data['amg_counters'] = {key: data['jacobians'][-1].get(key) if data['jacobians'] else None
                                for key in ['amg_builds','amg_refreshes']}
        data['boundary_setup_s'] = sum(float(row['total_s']) for row in data.get('boundary_stats', []))
        data['failed_ri_solves'] = sum(row.get('ri_success', 1)==0 for row in data['jacobians'])
        spaces = {row.get('execution_space') for row in data['assembly_stats'] if 'execution_space' in row}
        data['execution_spaces'] = sorted(spaces)
        if data['returncode'] or not data['completed'] or data['failed_ri_solves']:
            failures.append(data['path']+': incomplete or failed solve')
        for key in ('cg_failures','cg_true_residual_failures'):
            if data['solver_totals'].get(key, 0):
                message = data['path']+': '+key+'='+str(data['solver_totals'][key])
                if args.allow_diagonal_inner_nonconvergence and data['preconditioner'] == 'diagonal':
                    warnings.append(message)
                else:
                    failures.append(message)
        if args.expect_cuda and label!='cpu' and data['assembly_mode']=='kokkos' and spaces!={'Cuda'}:
            failures.append(data['path']+': expected CUDA assembly, got '+str(spaces))
        groups.setdefault(label, []).append(data)
    if not groups:
        parser.error('no */run-*/measurement.json files found')
    summary = {'groups': {}, 'failures': failures, 'warnings': warnings, 'field_checks': {}}
    for label, runs in groups.items():
        successful = [r for r in runs if not r['returncode'] and r['completed']]
        summary['groups'][label] = {'median_wall_s': statistics.median(r['wall_seconds'] for r in successful) if successful else None,
                                   'runs': runs}
    reference_label = args.reference_group or ('old' if 'old' in groups else 'fresh-factored')
    if args.reference_group and reference_label not in groups:
        failures.append('requested reference group is missing: '+reference_label)
    if reference_label in groups and all(not r['returncode'] and r['completed'] for r in groups[reference_label]):
        reference_run = directory / groups[reference_label][0]['path']
        reference = reference_run.parent / f"{groups[reference_label][0]['ranks']}-procs/result_{groups[reference_label][0]['steps']:03d}.vtu"
        summary['reference'] = str(reference.relative_to(directory))
        baseline = summary['groups'][reference_label]['median_wall_s']
        for label, data in summary['groups'].items():
            if label not in ('profile','runtime-details') and baseline and data['median_wall_s']:
                data['speedup_vs_'+reference_label] = baseline/data['median_wall_s']
        if importlib.util.find_spec('vtk') is None:
            summary['field_check_status'] = 'pending: run this script with Python VTK after copying results locally'
        else:
            compare = Path(__file__).resolve().parents[1]/'schur_amg/compare_fields.py'
            for label, runs in groups.items():
                for index, run in enumerate(runs):
                    if run['returncode'] or not run['completed'] or run['steps'] != groups[reference_label][0]['steps']:
                        continue
                    candidate = (directory/run['path']).parent / f"{run['ranks']}-procs/result_{run['steps']:03d}.vtu"
                    key = f'{label}/run-{index+1}'
                    result = subprocess.run([sys.executable, str(compare), str(reference), str(candidate),
                        '--fields','Velocity','Pressure','WSS','Traction','Vorticity','Divergence'], capture_output=True, text=True)
                    if result.returncode:
                        failures.append(key+': field comparison failed')
                        summary['field_checks'][key] = {'error': result.stdout+result.stderr}
                    else:
                        summary['field_checks'][key] = json.loads(result.stdout)
            summary['field_check_status'] = 'completed'
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for label, data in summary['groups'].items():
        if data['median_wall_s'] is None:
            print(f'{label:24} FAILED: no completed measurements')
        else:
            print(f"{label:24} median wall {data['median_wall_s']:.3f} s")
    print(summary.get('field_check_status','no matching field reference'))
    print('Summary:', directory/'summary.json')
    for warning in warnings:
        print('WARNING:',warning)
    if failures:
        raise SystemExit('\n'.join(failures))


if __name__ == '__main__':
    main()
