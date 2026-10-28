#!/usr/bin/env python3
"""Audit completion, linear/nonlinear convergence and fields before ranking timings."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET

REFERENCE = {'scalar': 'gmres-diagonal', 'solid': 'cg-diagonal', 'fsi': 'gmres-riluk0'}
SYMBOL = {'scalarTransport': 'HF', 'lElas': 'LE', 'FSI': 'FS', 'mesh': 'MS'}
TOLERANCE = {'scalar': 1e-4, 'solid': 1e-4, 'fsi': 1e-7}
ROW = re.compile(r'^\s*(\S+)\s+(\d+)-(\d+)s?\s+(\S+)\s+[\[!]\s*(-?\d+)\s+(\S+)\s+(\S+)\s+(\S+)[\]!]\s+[\[!]\s*(\d+)')

def audit(run, record):
    equations = ET.parse(run/'solver.xml').getroot().findall('Add_equation')
    tolerances = {SYMBOL[e.get('type')]: float(e.findtext('Tolerance')) for e in equations}
    history = run/'1-procs/histor.dat'
    last = {}
    iterations = 0
    for line in history.read_text().splitlines() if history.exists() else []:
        match = ROW.match(line)
        if not match:
            continue
        sym, step, itr, seconds, db, relative_step, relative_initial, residual, linear = match.groups()
        last[sym, int(step)] = (float(relative_step), float(relative_initial))
        iterations += int(linear)
    # The integrator accepts either residual relative to the initial global
    # norm or relative to this time step. Account for history's 4-digit rounding.
    capped = [{'equation': sym, 'step': step, 'relative_residuals': vals}
              for (sym, step), vals in last.items()
              if not all(math.isfinite(v) for v in vals) or min(vals) > tolerances[sym]*1.001]
    complete = all((sym, step) in last for sym in tolerances
                   for step in range(1, record['steps']+1))
    return dict(nonlinear_unconverged=capped, history_complete=complete,
                linear_iterations=iterations,
                convergence_passed=bool(record['completed'] and complete and not capped
                                        and record['linear_failures'] == 0))

def fields(reference, candidate):
    import vtk
    def read(path):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(path)); reader.Update()
        return reader.GetOutput()
    a, b = read(reference), read(candidate)
    if a.GetNumberOfPoints() != b.GetNumberOfPoints() or not a.GetNumberOfPoints():
        raise ValueError('Empty or different point counts')
    # Same mesh and one MPI rank. Confirm coordinate identity before using IDs;
    # the output stores original mesh coordinates, with displacement separately.
    if any(a.GetPoint(i) != b.GetPoint(i) for i in range(a.GetNumberOfPoints())):
        raise ValueError('Point ordering/coordinates differ')
    result = {}
    for j in range(a.GetPointData().GetNumberOfArrays()):
        av = a.GetPointData().GetArray(j)
        bv = b.GetPointData().GetArray(av.GetName())
        if bv is None or av.GetNumberOfComponents() != bv.GetNumberOfComponents():
            raise ValueError('Field missing or incompatible: '+av.GetName())
        norm, error = [], []
        for i in range(a.GetNumberOfPoints()):
            for x, y in zip(av.GetTuple(i), bv.GetTuple(i)):
                norm.append(x*x); error.append((y-x)**2)
        n, e = math.sqrt(math.fsum(norm)), math.sqrt(math.fsum(error))
        value = e/n if n else e
        result[av.GetName()] = value if math.isfinite(value) else None
    return result

def summarize(root, expect_cuda=False):
    manifest = json.loads((root/'manifest.json').read_text())
    try:
        import vtk
        have_vtk = True
    except ImportError:
        have_vtk = False
    groups = {}
    failures = []
    for path in sorted(root.glob('*/*/*/run-*/measurement.json')):
        r = json.loads(path.read_text()); run = path.parent
        r.update(audit(run, r))
        if not r['completed']:
            failures.append(str(run.relative_to(root))+': solver failed, timed out, or missing final output')
        ref = root/r['case']/REFERENCE[r['case']]/'trilinos/run-1'/f'1-procs/result_{r["steps"]:03d}.vtu'
        candidate = run/f'1-procs/result_{r["steps"]:03d}.vtu'
        r['fields'] = 'pending'
        if have_vtk and ref.exists() and candidate.exists():
            try:
                r['fields'] = fields(ref, candidate)
                r['fields_passed'] = bool(r['fields']) and all(v is not None and v <= TOLERANCE[r['case']] for v in r['fields'].values())
                reference_run = ref.parent.parent
                reference_measurement = reference_run/'measurement.json'
                r['reference_converged'] = reference_measurement.exists() and audit(reference_run, json.loads(reference_measurement.read_text()))['convergence_passed']
                if r['convergence_passed'] and not r['fields_passed']:
                    failures.append(str(run.relative_to(root))+': converged solver disagrees with reference fields')
            except ValueError as exc:
                r['fields'] = str(exc); r['fields_passed'] = False
        log = (run/'stdout.log').read_text(errors='replace')
        r['cuda_evidence'] = bool(re.search(r'Default Device:\s*Cuda|KOKKOS_ENABLE_CUDA:\s*yes', log))
        if expect_cuda and not r['cuda_evidence']:
            failures.append(str(run.relative_to(root))+': no CUDA runtime evidence')
        r['state_checks_passed'] = bool(r['completed'] and r['history_complete'] and not r['nonlinear_unconverged'] and r.get('fields_passed', False) and r.get('reference_converged', False))
        r['validated'] = r['convergence_passed'] and r.get('fields_passed', False) and r.get('reference_converged', False)
        groups.setdefault((r['case'], r['variant'], r['assembly']), []).append(r)
    for expected in manifest.get('expected_groups', []):
        if tuple(expected) not in groups:
            failures.append('/'.join(expected)+': missing group')
    output = []
    for key, runs in groups.items():
        case, variant, assembly = key
        count_ok = len(runs) == manifest['repeat']
        if not count_ok:
            failures.append('/'.join(key)+': missing repeats')
        row = dict(case=case, variant=variant, assembly=assembly, repeats=len(runs),
                   wall_seconds=statistics.median(r['wall_seconds'] for r in runs),
                   simulation_seconds=statistics.median(r['simulation_seconds'] for r in runs if r['simulation_seconds'] is not None) if any(r['simulation_seconds'] is not None for r in runs) else None,
                   wall_range=[min(r['wall_seconds'] for r in runs),max(r['wall_seconds'] for r in runs)],
                   linear_failures=sum(r['linear_failures'] for r in runs),
                   nonlinear_failures=sum(len(r['nonlinear_unconverged']) for r in runs),
                   linear_iterations=statistics.median(r['linear_iterations'] for r in runs),
                   state_checks_passed=count_ok and all(r['state_checks_passed'] for r in runs),
                   validated=count_ok and all(r['validated'] for r in runs), runs=runs)
        output.append(row)
    lookup = {(r['case'],r['variant'],r['assembly']):r for r in output}
    for row in output:
        other = lookup.get((row['case'],row['variant'],'fsils'))
        row['speedup_vs_fsils_assembly'] = other['wall_seconds']/row['wall_seconds'] if row['assembly']=='trilinos' and row['validated'] and other and other['validated'] else None
    result = dict(field_checks='completed' if have_vtk else 'pending: install vtk and rerun locally',
                  field_tolerances=TOLERANCE, failures=failures, groups=output)
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines = ['| Case | Solver/preconditioner | Assembly | Wall s | Simulation s | Linear failures | Nonlinear failures | Validated |',
             '|---|---|---|---:|---:|---:|---:|---|']
    for r in output:
        sim = f'{r["simulation_seconds"]:.3f}' if r['simulation_seconds'] is not None else '—'
        lines.append(f'| {r["case"]} | {r["variant"]} | {r["assembly"]} | {r["wall_seconds"]:.3f} | {sim} | {r["linear_failures"]} | {r["nonlinear_failures"]} | {r["validated"]} |')
    lines += ['', 'Field comparisons: '+result['field_checks']] + failures
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--expect-cuda',action='store_true')
    args=p.parse_args()
    result=summarize(args.output.resolve(),args.expect_cuda)
    if result['failures']:
        raise SystemExit(1)
