import time
from pathlib import Path
import gmsh
import numpy as np
import trimesh
import meshio

def build_wake_mesh(
    stl_path: str = "mesh/sr71_faceted_clean.stl",
    output_msh: str = "mesh/sr71_wake_test.msh",
    lc_far: float = 6.0,
    lc_wake: float = 0.6, # 60cm in wake for quick local test, 25cm on RunPod
):
    print("=" * 76)
    print("TESTING WAKE-REFINED MESH GENERATION")
    print(f"  STL:        {stl_path}")
    print(f"  Wake Cell:  {lc_wake} m")
    print(f"  Farfield:   {lc_far} m")
    print("=" * 76)

    gmsh.initialize()
    gmsh.model.add("sr71_wake")
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 1e-4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_far)

    gmsh.merge(stl_path)
    stl_surf = 1
    stl_loop = gmsh.model.geo.addSurfaceLoop([stl_surf])

    # Wind tunnel box
    x0, x1 = -30.0, 55.0
    y0, y1 = -18.0, 20.0
    z0, z1 = -25.0, 25.0

    p1 = gmsh.model.geo.addPoint(x0, y0, z0, lc_far)
    p2 = gmsh.model.geo.addPoint(x1, y0, z0, lc_far)
    p3 = gmsh.model.geo.addPoint(x1, y1, z0, lc_far)
    p4 = gmsh.model.geo.addPoint(x0, y1, z0, lc_far)

    p5 = gmsh.model.geo.addPoint(x0, y0, z1, lc_far)
    p6 = gmsh.model.geo.addPoint(x1, y0, z1, lc_far)
    p7 = gmsh.model.geo.addPoint(x1, y1, z1, lc_far)
    p8 = gmsh.model.geo.addPoint(x0, y1, z1, lc_far)

    l1 = gmsh.model.geo.addLine(p1, p2); l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4); l4 = gmsh.model.geo.addLine(p4, p1)
    l5 = gmsh.model.geo.addLine(p5, p6); l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8); l8 = gmsh.model.geo.addLine(p8, p5)
    l9 = gmsh.model.geo.addLine(p1, p5); l10 = gmsh.model.geo.addLine(p2, p6)
    l11 = gmsh.model.geo.addLine(p3, p7); l12 = gmsh.model.geo.addLine(p4, p8)

    s_zmin = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
    s_zmax = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])])
    s_inlet = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l4, -l12, -l8, l9])])
    s_outlet = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])])
    s_ymin = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])])
    s_ymax = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])])

    box_loop = gmsh.model.geo.addSurfaceLoop([s_zmin, s_zmax, s_inlet, s_outlet, s_ymin, s_ymax])
    vol = gmsh.model.geo.addVolume([box_loop, stl_loop])
    gmsh.model.geo.synchronize()

    # Physical Groups
    gmsh.model.addPhysicalGroup(3, [vol], 1, name="fluid")
    gmsh.model.addPhysicalGroup(2, [s_inlet], 11, name="inlet")
    gmsh.model.addPhysicalGroup(2, [s_outlet], 12, name="outlet")
    gmsh.model.addPhysicalGroup(2, [s_zmin, s_zmax, s_ymin, s_ymax], 13, name="walls")
    gmsh.model.addPhysicalGroup(2, [stl_surf], 14, name="airfoil_wall")

    # Wake Refinement Box Field: X in [14, 50], Y in [-3.5, 4.5], Z in [-10, 10]
    f_box = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f_box, "VIn", lc_wake)
    gmsh.model.mesh.field.setNumber(f_box, "VOut", lc_far)
    gmsh.model.mesh.field.setNumber(f_box, "XMin", 14.0)
    gmsh.model.mesh.field.setNumber(f_box, "XMax", 50.0)
    gmsh.model.mesh.field.setNumber(f_box, "YMin", -3.5)
    gmsh.model.mesh.field.setNumber(f_box, "YMax", 4.5)
    gmsh.model.mesh.field.setNumber(f_box, "ZMin", -10.0)
    gmsh.model.mesh.field.setNumber(f_box, "ZMax", 10.0)
    gmsh.model.mesh.field.setNumber(f_box, "Thickness", 2.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_box)

    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)

    t0 = time.perf_counter()
    gmsh.model.mesh.generate(3)
    tets = gmsh.model.mesh.getElementsByType(4)[0]
    print(f"Generated {len(tets)} tetrahedra in {time.perf_counter() - t0:.2f}s!")

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(output_msh)
    gmsh.finalize()
    print(f"Test mesh written to {output_msh}")

if __name__ == "__main__":
    build_wake_mesh()
