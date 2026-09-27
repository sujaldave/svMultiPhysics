#!/usr/bin/env python3
"""Generate a structured quadrilateral mesh of the Turek-Hron elastic flap.

Written while diagnosing the 2D FSI benchmarks; see
README_TurekHron_2D_benchmark_investigation.md section 4.3 for the CSM3
convergence study these meshes came from.

Geometry follows Turek & Hron (2006): the flap occupies y in [0.19, 0.21] and
runs from the cylinder surface (centre (0.2, 0.2), radius 0.05) to x = 0.6, so
its root edge is an arc rather than a straight line.

Writes, per resolution, a directory tree that svMultiPhysics can read directly:

    <out>/flap_<nlen>x<nthk>/solid/mesh-complete.mesh.vtu
    <out>/flap_<nlen>x<nthk>/solid/mesh-surfaces/fixed.vtp      (root, clamped)
    <out>/flap_<nlen>x<nthk>/solid/mesh-surfaces/interface.vtp  (wetted boundary)

Usage:
    python3 generate_flap_mesh.py <output-dir> [nlen x nthk ...]

    python3 generate_flap_mesh.py ./meshes              # default set
    python3 generate_flap_mesh.py ./meshes 90x8         # just the one

Measured CSM3 accuracy (quads, <Penalty_parameter> ~ 0, tip amplitude vs the
65.160e-3 reference): 45x4 -7.9%, 45x8 -6.1%, 90x8 -2.8%, 90x16 -2.0%.
Refining only through the thickness does not help much -- refine both
directions and keep the elements roughly square. 90x8 is the sweet spot.
"""

import vtk, os, sys, math

# Turek-Hron flap: y in [0.19,0.21]; left edge is the cylinder arc
# (centre (0.2,0.2), r=0.05); right edge x=0.6.
R, CX, CY, XR = 0.05, 0.2, 0.2, 0.6
YB, YT = 0.19, 0.21

def xleft(y):
    return CX + math.sqrt(max(R*R - (y-CY)**2, 0.0))

def build(nlen, nthk, outdir):
    os.makedirs(f"{outdir}/mesh-surfaces", exist_ok=True)
    pts = vtk.vtkPoints()
    idx = {}
    for j in range(nthk+1):
        y = YB + (YT-YB)*j/nthk
        xl = xleft(y)
        for i in range(nlen+1):
            x = xl + (XR-xl)*i/nlen
            idx[(i,j)] = pts.GetNumberOfPoints()
            pts.InsertNextPoint(x, y, 0.0)
    g = vtk.vtkUnstructuredGrid(); g.SetPoints(pts)
    for j in range(nthk):
        for i in range(nlen):
            q = vtk.vtkQuad()
            for k,(a,b) in enumerate([(i,j),(i+1,j),(i+1,j+1),(i,j+1)]):
                q.GetPointIds().SetId(k, idx[(a,b)])
            g.InsertNextCell(q.GetCellType(), q.GetPointIds())
    gn = vtk.vtkIntArray(); gn.SetName("GlobalNodeID")
    for i in range(g.GetNumberOfPoints()): gn.InsertNextValue(i+1)
    g.GetPointData().AddArray(gn)
    ge = vtk.vtkIntArray(); ge.SetName("GlobalElementID")
    for i in range(g.GetNumberOfCells()): ge.InsertNextValue(i+1)
    g.GetCellData().AddArray(ge)
    w = vtk.vtkXMLUnstructuredGridWriter(); w.SetFileName(f"{outdir}/mesh-complete.mesh.vtu")
    w.SetInputData(g); w.Write()

    def face(name, node_pairs, elem_ids):
        fp = vtk.vtkPoints(); local = {}
        poly = vtk.vtkPolyData(); lines = vtk.vtkCellArray()
        for (a,b) in node_pairs:
            for nd in (a,b):
                if nd not in local:
                    local[nd] = fp.GetNumberOfPoints()
                    p = g.GetPoint(nd); fp.InsertNextPoint(p)
        for (a,b) in node_pairs:
            ln = vtk.vtkLine(); ln.GetPointIds().SetId(0, local[a]); ln.GetPointIds().SetId(1, local[b])
            lines.InsertNextCell(ln)
        poly.SetPoints(fp); poly.SetLines(lines)
        fgn = vtk.vtkIntArray(); fgn.SetName("GlobalNodeID")
        inv = {v:k for k,v in local.items()}
        for i in range(fp.GetNumberOfPoints()): fgn.InsertNextValue(inv[i]+1)
        poly.GetPointData().AddArray(fgn)
        fge = vtk.vtkIntArray(); fge.SetName("GlobalElementID")
        for e in elem_ids: fge.InsertNextValue(e+1)
        poly.GetCellData().AddArray(fge)
        w2 = vtk.vtkXMLPolyDataWriter(); w2.SetFileName(f"{outdir}/mesh-surfaces/{name}.vtp")
        w2.SetInputData(poly); w2.Write()

    def eid(i,j): return j*nlen + i
    fixed_pairs = [(idx[(0,j)], idx[(0,j+1)]) for j in range(nthk)]
    fixed_elems = [eid(0,j) for j in range(nthk)]
    face("fixed", fixed_pairs, fixed_elems)
    ip, ie = [], []
    for i in range(nlen):                       # bottom
        ip.append((idx[(i,0)], idx[(i+1,0)])); ie.append(eid(i,0))
    for j in range(nthk):                       # right
        ip.append((idx[(nlen,j)], idx[(nlen,j+1)])); ie.append(eid(nlen-1,j))
    for i in range(nlen):                       # top
        ip.append((idx[(i,nthk)], idx[(i+1,nthk)])); ie.append(eid(i,nthk-1))
    face("interface", ip, ie)
    print(f"  {outdir}: {g.GetNumberOfPoints()} pts, {g.GetNumberOfCells()} quads ({nlen} x {nthk})")

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    if len(sys.argv) > 2:
        sizes = []
        for a in sys.argv[2:]:
            nl, _, nt = a.lower().partition("x")
            sizes.append((int(nl), int(nt)))
    else:
        sizes = [(45, 4), (45, 8), (90, 8), (90, 16), (120, 24)]
    for nlen, nthk in sizes:
        build(nlen, nthk, f"{out}/flap_{nlen}x{nthk}/solid")


if __name__ == "__main__":
    main()
