#!/usr/bin/env python3
"""Report 1/2/4-GPU strong scaling and validate matching final fields.

Run on the cluster for timing checks, then rerun with Python VTK locally for
field comparisons. Only the standard library is needed for timing analysis.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

import check_binding


FIELDS = ['Velocity', 'Pressure', 'WSS', 'Traction', 'Vorticity', 'Divergence']


def read_run(path, expect_cuda, failures):
    data = json.loads(path.read_text())
    data['measurement'] = str(path)
    solver = data['jacobians']
    totals = {key: sum(row.get(key, 0) for row in solver)
              for key in {key for row in solver for key in row}
              if key.endswith('_s') or key.endswith('_iterations') or key.endswith('_calls') or key.endswith('_failures')}
    data['solver_totals'] = totals
    data['phase_totals'] = {}
    for row in data['assembly_stats']:
        phase = data['phase_totals'].setdefault(row['kind'], {})
        for key, value in row.items():
            if key.endswith('_s') or key == 'rebuilt':
                phase[key] = phase.get(key, 0.0)+float(value)
    data['amg_counters'] = {key: solver[-1].get(key) if solver else None
                            for key in ['amg_builds','amg_refreshes']}
    data['transport'] = sorted({r['transport'] for r in data['assembly_stats'] if 'transport' in r})
    spaces = {r['execution_space'] for r in data['assembly_stats'] if 'execution_space' in r}
    data['execution_spaces'] = sorted(spaces)
    errors = []
    if data['returncode'] or not data['completed'] or not solver:
        errors.append('incomplete run or missing solver statistics')
    if solver and len(solver) != data['nonlinear_iterations']:
        errors.append('solver statistics do not cover all nonlinear iterations')
    if any(row.get('ri_success', 1) == 0 or row.get('cg_success', 1) == 0 for row in solver):
        errors.append('failed outer or pressure solve')
    if any(totals.get(key, 0) for key in ['cg_failures','cg_true_residual_failures']):
        errors.append('pressure CG/residual failure')
    if expect_cuda and spaces != {'Cuda'}:
        errors.append('native assembly did not report CUDA')
    if (data['preconditioner'],data['assembly_mode'],data.get('schur_reuse'),data.get('schur_operator')) != ('schur-amg','kokkos','transfers','explicit'):
        errors.append('unexpected solver/assembly settings')
    for error in errors:
        failures.append(str(path)+': '+error)
    data['valid'] = not errors
    data['input_xml'] = ET.tostring(ET.parse(path.parent/'solver.xml').getroot(), encoding='unicode')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--expect-cuda', action='store_true')
    args = parser.parse_args()
    root = args.directory.resolve()
    failures, groups = [], {}
    manifest = json.loads((root/'sweep.json').read_text()) if (root/'sweep.json').is_file() else None
    for path in sorted(root.glob('gpus-*/run-*/measurement.json')):
        ranks = int(path.parent.parent.name.split('-')[-1])
        data = read_run(path, args.expect_cuda, failures)
        if data['ranks'] != ranks:
            failures.append(str(path)+': rank count differs from directory label')
            data['valid'] = False
        groups.setdefault(ranks, []).append(data)
    if not groups:
        parser.error('No gpus-*/run-*/measurement.json files found')
    if manifest:
        for ranks in manifest['gpu_counts']:
            if len(groups.get(ranks, [])) != manifest['repeat']:
                failures.append(f'{ranks} GPUs: missing expected repeats')
        if any(r['steps'] != manifest['steps'] for runs in groups.values() for r in runs):
            failures.append('step counts differ from the sweep manifest')
    inputs = {r['input_xml'] for runs in groups.values() for r in runs}
    if len(inputs) != 1:
        failures.append('Input XML differs between scaling runs')
    report = dict(manifest=manifest, failures=failures, groups={}, field_checks={})
    for ranks, runs in sorted(groups.items()):
        binding = root/'binding'/f'gpus-{ranks}'
        try:
            check_binding.validate(binding, ranks)
            binding_status = 'validated: distinct GPU per rank on one node'
        except RuntimeError as error:
            binding_status = str(error)
            if args.expect_cuda:
                failures.append(f'{ranks} GPUs: '+binding_status)
        successful = [r for r in runs if r['valid']]
        item = dict(binding_status=binding_status, runs=runs, valid_repeats=len(successful))
        if successful:
            item.update(median_wall_s=statistics.median(r['wall_seconds'] for r in successful),
                        min_wall_s=min(r['wall_seconds'] for r in successful),
                        max_wall_s=max(r['wall_seconds'] for r in successful),
                        median_simulation_s=statistics.median(r['simulation_seconds'] for r in successful),
                        median_solver_s=statistics.median(r['solver_totals']['total_s'] for r in successful),
                        median_cg_iterations=statistics.median(r['solver_totals']['cg_iterations'] for r in successful),
                        nonlinear_iterations=sorted({r['nonlinear_iterations'] for r in successful}))
        for run in runs:
            run.pop('input_xml')
            # Relative paths survive copying the whole sweep directory locally.
            run['measurement'] = str(Path(run['measurement']).relative_to(root))
        report['groups'][ranks] = item
    baseline = report['groups'].get(1, {})
    if baseline.get('median_wall_s'):
        for ranks, item in report['groups'].items():
            if item.get('median_wall_s'):
                item.update(wall_speedup=baseline['median_wall_s']/item['median_wall_s'],
                            wall_efficiency=baseline['median_wall_s']/item['median_wall_s']/ranks,
                            simulation_speedup=baseline['median_simulation_s']/item['median_simulation_s'],
                            solver_speedup=baseline['median_solver_s']/item['median_solver_s'],
                            gpu_seconds=ranks*item['median_wall_s'])
        if importlib.util.find_spec('vtk') is None:
            report['field_check_status'] = 'pending: rerun this script locally with Python VTK'
        else:
            reference_run = next(r for r in baseline['runs'] if r['valid'])
            reference = (root/reference_run['measurement']).parent/f"1-procs/result_{reference_run['steps']:03d}.vtu"
            compare = Path(__file__).resolve().parents[1]/'schur_amg/compare_fields.py'
            for ranks, item in report['groups'].items():
                for run in item['runs']:
                    if not run['valid']:
                        continue
                    candidate = (root/run['measurement']).parent/f"{ranks}-procs/result_{run['steps']:03d}.vtu"
                    key = run['measurement']
                    process = subprocess.run([sys.executable,str(compare),str(reference),str(candidate),'--fields',*FIELDS],
                                             capture_output=True,text=True)
                    if process.returncode:
                        failures.append(key+': field comparison failed')
                        report['field_checks'][key] = {'error':process.stdout+process.stderr}
                    else:
                        report['field_checks'][key] = json.loads(process.stdout)
            report['field_check_status'] = 'completed'
    else:
        report['field_check_status'] = 'pending: no valid 1-GPU reference in this sweep'
    rows = []
    for ranks, item in report['groups'].items():
        rows.append(dict(gpus=ranks,**{key:item.get(key) for key in [
            'median_wall_s','min_wall_s','max_wall_s','median_simulation_s','median_solver_s',
            'wall_speedup','wall_efficiency','simulation_speedup','solver_speedup','gpu_seconds','median_cg_iterations']}))
    with (root/'scaling.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (root/'scaling.json').write_text(json.dumps(report,indent=2)+'\n')
    lines = ['| GPUs | Median wall (s) | Simulation (s) | Wall speedup | Efficiency |',
             '|---:|---:|---:|---:|---:|']
    for row in rows:
        def fmt(key, percent=False):
            value = row[key]
            return 'pending' if value is None else (f'{100*value:.1f}%' if percent else f'{value:.3f}')
        lines.append(f"| {row['gpus']} | {fmt('median_wall_s')} | {fmt('median_simulation_s')} | {fmt('wall_speedup')} | {fmt('wall_efficiency',True)} |")
    lines += ['',report['field_check_status'],'',
              'Strong-scaling efficiency = (T1/TN)/N. GPU-seconds = N*TN.',
              'Simulation time is the last history timestamp; wall time also includes launcher/startup/shutdown.',
              'Phase times sum the maximum rank per Jacobian; compare iteration counts as well as elapsed time.']
    if failures:
        lines += ['', 'Validation failures:', *['- '+failure for failure in failures]]
    (root/'scaling.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
