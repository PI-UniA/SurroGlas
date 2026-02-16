import typing
from pathlib import Path

from dolfinx import cpp as _cpp
from dolfinx.cpp.graph import AdjacencyList_int32
from dolfinx.mesh import Mesh

from dolfinx.io import gmshio
from mpi4py import MPI
import gmsh
import numpy as np


def create_mesh(
    path: str,
    dim: int,
    name: str,
    t_start: int,
    t_end: int,
    thickness_mm: float = 20,
    n_thickness_nodes: int = 20,   # nodes across thickness (x)
    length_m: float = 1.0,         # plate length (y) for 2D/3D
    n_length_nodes: int = 51,      # nodes along y
    width_m: float = 0.1,          # plate width (z) for 3D
    n_width_nodes: int = 11,       # nodes along z
    recombine: bool = True         # quads (2D) / hexes (3D) if possible
):
    import gmsh

    init_here = False
    if not gmsh.isInitialized():
        gmsh.initialize()
        init_here = True
    gmsh.model.add(f"Glass {dim}D mesh")

    # --- geometry sizes ---
    t_m  = thickness_mm / 1000.0     # e.g. 19 mm -> 0.019 m
    xL   = -0.5 * t_m
    xR   =  0.5 * t_m
    tol  = 1e-10

    if dim == 1:
        # line along x in [xL, xR]
        pL = gmsh.model.occ.addPoint(xL, 0.0, 0.0)
        pR = gmsh.model.occ.addPoint(xR, 0.0, 0.0)
        line = gmsh.model.occ.addLine(pL, pR)
        gmsh.model.occ.synchronize()

        # EXACT node count along thickness
        gmsh.model.mesh.setTransfiniteCurve(line, n_thickness_nodes)

        # physical groups
        gmsh.model.addPhysicalGroup(0, [pL], 10)      # left
        gmsh.model.addPhysicalGroup(0, [pR], 12)      # right
        gmsh.model.addPhysicalGroup(1, [line], 0)     # cells
        gmsh.model.setPhysicalName(1, 0, "cells")

        gmsh.model.mesh.generate(1)
        gmsh.write(path)
        if init_here: gmsh.finalize()
        return

    if dim == 2:
        # rectangle in (x,y): x in [xL,xR], y in [0,length_m]
        p1 = gmsh.model.occ.addPoint(xL, 0.0, 0.0)
        p2 = gmsh.model.occ.addPoint(xR, 0.0, 0.0)
        p3 = gmsh.model.occ.addPoint(xR, length_m, 0.0)
        p4 = gmsh.model.occ.addPoint(xL, length_m, 0.0)

        l1 = gmsh.model.occ.addLine(p1, p2)  # bottom
        l2 = gmsh.model.occ.addLine(p2, p3)  # right
        l3 = gmsh.model.occ.addLine(p3, p4)  # top
        l4 = gmsh.model.occ.addLine(p4, p1)  # left

        cloop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        surf  = gmsh.model.occ.addPlaneSurface([cloop])
        gmsh.model.occ.synchronize()

        # ---- transfinite constraints = exact node counts ----
        # x-direction curves: bottom/top
        gmsh.model.mesh.setTransfiniteCurve(l1, n_thickness_nodes)  # 20 nodes
        gmsh.model.mesh.setTransfiniteCurve(l3, n_thickness_nodes)  # 20 nodes

        # y-direction curves: left/right
        gmsh.model.mesh.setTransfiniteCurve(l4, n_length_nodes)     # 10 nodes
        gmsh.model.mesh.setTransfiniteCurve(l2, n_length_nodes)     # 10 nodes

        gmsh.model.mesh.setTransfiniteSurface(surf)

        # optional: quads instead of triangles
        if recombine:
            gmsh.model.mesh.setRecombine(2, surf)

        # ---- physical groups (boundaries + cells) ----
        # boundary tags
        gmsh.model.addPhysicalGroup(1, [l4], 10)  # left
        gmsh.model.addPhysicalGroup(1, [l2], 12)  # right
        gmsh.model.addPhysicalGroup(1, [l1], 13)  # bottom
        gmsh.model.addPhysicalGroup(1, [l3], 11)  # top

        # cell tag
        gmsh.model.addPhysicalGroup(2, [surf], 0)
        gmsh.model.setPhysicalName(2, 0, "cells")

        gmsh.model.mesh.generate(2)
        gmsh.write(path)

        if init_here:
            gmsh.finalize()
        return
    if dim == 3:
        # start from 2D base surface and extrude in z to get a volume
        A = gmsh.model.occ.addPoint(xL, 0.0,       0.0)
        B = gmsh.model.occ.addPoint(xR, 0.0,       0.0)
        C = gmsh.model.occ.addPoint(xR, length_m,  0.0)
        D = gmsh.model.occ.addPoint(xL, length_m,  0.0)

        lB = gmsh.model.occ.addLine(A, B)
        lR = gmsh.model.occ.addLine(B, C)
        lT = gmsh.model.occ.addLine(C, D)
        lL = gmsh.model.occ.addLine(D, A)

        loop = gmsh.model.occ.addCurveLoop([lB, lR, lT, lL])
        surf = gmsh.model.occ.addPlaneSurface([loop])

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setTransfiniteCurve(lB, n_thickness_nodes)
        gmsh.model.mesh.setTransfiniteCurve(lT, n_thickness_nodes)
        gmsh.model.mesh.setTransfiniteCurve(lL, n_length_nodes)
        gmsh.model.mesh.setTransfiniteCurve(lR, n_length_nodes)
        gmsh.model.mesh.setTransfiniteSurface(surf)
        if recombine:
            gmsh.model.mesh.setRecombine(2, surf)

        # extrude base surface by width_m in +z
        out = gmsh.model.occ.extrude([(2, surf)], 0, 0, width_m, numElements=[],
                                     heights=[], recombine=recombine)
        gmsh.model.occ.synchronize()

        # get the volume tag
        vols = [e[1] for e in out if e[0] == 3]
        if not vols:
            vols = [t for (d, t) in gmsh.model.getEntities(3)]
        vol = vols[0]

        # set transfinite counts on all boundary edges (by dominant axis)
        edges = [t for (d, t) in gmsh.model.getBoundary([(3, vol)], oriented=False, recursive=True) if d == 1]
        for c in edges:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, c)
            dx, dy, dz = abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin)
            if dx >= dy and dx >= dz:
                gmsh.model.mesh.setTransfiniteCurve(c, n_thickness_nodes)  # x
            elif dy >= dx and dy >= dz:
                gmsh.model.mesh.setTransfiniteCurve(c, n_length_nodes)     # y
            else:
                gmsh.model.mesh.setTransfiniteCurve(c, n_width_nodes)      # z

        # mark faces as transfinite (+ recombine) and set physical groups by face position
        faces = [t for (d, t) in gmsh.model.getBoundary([(3, vol)], oriented=False, recursive=False) if d == 2]
        for f in faces:
            gmsh.model.mesh.setTransfiniteSurface(f)
            if recombine:
                gmsh.model.mesh.setRecombine(2, f)

            cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, f)
            if abs(cx - xL) < tol:
                gmsh.model.addPhysicalGroup(2, [f], 12)  # left (x = xL)
            elif abs(cx - xR) < tol:
                gmsh.model.addPhysicalGroup(2, [f], 13)  # right (x = xR)
            elif abs(cy - 0.0) < tol:
                gmsh.model.addPhysicalGroup(2, [f], 15)  # bottom (y = 0)
            elif abs(cy - length_m) < tol:
                gmsh.model.addPhysicalGroup(2, [f], 14)  # top (y = L)
            elif abs(cz - 0.0) < tol:
                gmsh.model.addPhysicalGroup(2, [f], 10)  # front (z = 0)
            elif abs(cz - width_m) < tol:
                gmsh.model.addPhysicalGroup(2, [f], 11)  # back  (z = W)

        # volume "cells"
        gmsh.model.mesh.setTransfiniteVolume(vol)
        gmsh.model.addPhysicalGroup(3, [vol], 0)
        gmsh.model.setPhysicalName(3, 0, "cells")

        gmsh.model.mesh.generate(3)
        gmsh.write(path)
        if init_here: gmsh.finalize()
        return

    raise ValueError("dim must be 1, 2, or 3")
