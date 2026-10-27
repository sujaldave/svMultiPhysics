#!/usr/bin/env python3
"""Compare saved VTU fields using VTK (no NumPy dependency)."""
import argparse
import json
import math
from pathlib import Path
import vtk


def read(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    if not grid.GetNumberOfPoints():
        raise ValueError(f'No points read from {path}')
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tolerance', type=float, default=1e-7)
    parser.add_argument('--fields', nargs='+', default=['Velocity','Pressure'])
    args = parser.parse_args()
    ref, cand = read(args.reference), read(args.candidate)
    if ref.GetNumberOfPoints() != cand.GetNumberOfPoints():
        raise ValueError('Point counts differ')
    # Match coordinates, also supporting different partition output orderings.
    lookup = {cand.GetPoint(i): i for i in range(cand.GetNumberOfPoints())}
    order = [lookup[ref.GetPoint(i)] for i in range(ref.GetNumberOfPoints())]
    results = {}
    for name in args.fields:
        a, b = ref.GetPointData().GetArray(name), cand.GetPointData().GetArray(name)
        if a is None or b is None:
            raise ValueError(f'Missing {name} field')
        if a.GetNumberOfComponents() != b.GetNumberOfComponents():
            raise ValueError(f'{name} component counts differ')
        reference_values, errors = [], []
        for i, j in enumerate(order):
            for av, bv in zip(a.GetTuple(i), b.GetTuple(j)):
                reference_values.append(av)
                errors.append(bv - av)
        norm = math.sqrt(math.fsum(v*v for v in reference_values))
        delta = math.sqrt(math.fsum(v*v for v in errors))
        rel = delta / norm if norm else delta
        results[name] = dict(relative_l2=rel, maximum_absolute=max(map(abs, errors)),
                             passed=math.isfinite(rel) and rel <= args.tolerance)
    text = json.dumps(results, indent=2, allow_nan=False) + '\n'
    print(text, end='')
    if args.output:
        args.output.write_text(text)
    if not all(result['passed'] for result in results.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
