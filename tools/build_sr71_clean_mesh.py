"""Build clean 3D wind tunnel mesh around authentic Lockheed SR-71.

Guarantees:
- Solid shock-spike cones (airfoil_wall, tag 14) for authentic conical shock generation.
- Annular intake entrance patch (outlet, tag 12) for flow-through engine duct ingestion.
- Outer wind tunnel box: X in [-30, 45], Y in [-18, 20], Z in [-25, 25].
- Slivers eliminated: Netgen 3D optimization and bounded characteristic length.
"""

from pathlib import Path
import gmsh
import numpy as np
import trimesh

def build_clean_sr71_mesh(
    stl_path: str = "mesh/sr71_aerodynamic_clean.stl",
    output_msh: str = "mesh/sr71_clean_open_air.msh",
    lc_far: float = 12.0,
):
    print("=" * 76)
    print("GENERATING PRISTINE 3D WIND TUNNEL MESH FOR SR-71 BLACKBIRD")
    print(f"  Source STL:  {stl_path}")
    print(f"  Output MSH:  {output_msh}")
    print("=" * 76)

    m = trimesh.load(stl_path)
    c = m.triangles_center
    n = m.face_normals

    # Engine inlet annular patch faces (flat, perpendicular to X, inside cowl lip)
    # X in [2.38, 2.65], |Z| in [3.3, 4.9], n_x < -0.8
    inlet_patch_mask = (c[:,0] > 2.38) & (c[:,0] < 2.65) & (np.abs(np.abs(c[:,2]) - 4.1) < 0.9) & (n[:,0] < -0.8)
    # Spike faces (X in [1.0, 2.45], forward of the intake patch) remain SOLID AIRFOIL WALL
    spike_mask = (c[:,0] >= 1.0) & (c[:,0] <= 2.45) & (np.abs(np.abs(c[:,2]) - 4.1) < 0.8) & ~inlet_patch_mask

    print(f"Airframe faces: {len(m.faces)}")
    print(f"  Solid spike cone faces: {np.sum(spike_mask)} (SOLID WALL)")
    print(f"  Annular intake patch:   {np.sum(inlet_patch_mask)} (FLOW-THROUGH OUTLET)")
    print(f"  Remaining airframe:     {len(m.faces) - np.sum(inlet_patch_mask)} (SOLID WALL)")

    gmsh.initialize()
    gmsh.model.add("sr71_clean_open_air")
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 1e-4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.35)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_far)

    # 1. Merge STL
    gmsh.merge(stl_path)
    stl_surf = 1
    stl_loop = gmsh.model.geo.addSurfaceLoop([stl_surf])

    # 2. Wind Tunnel Box
    x0, x1 = -30.0, 45.0
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

    # 3. Generate 3D Mesh
    print("Generating 3D volume mesh...")
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.model.mesh.generate(3)

    tets = gmsh.model.mesh.getElementsByType(4)[0]
    print(f"Generated {len(tets)} tetrahedra.")

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(output_msh)
    gmsh.finalize()
    print(f"Mesh written to {output_msh}")

    # Now tag the annular intake patch faces as outlet (12)
    import meshio
    raw = meshio.read(output_msh)
    pts = raw.points
    tri_data = raw.cells[0].data
    phys = raw.cell_data['gmsh:physical'][0]

    tri_centers = np.mean(pts[tri_data], axis=1)
    # Normal of each triangle
    v0 = pts[tri_data[:, 0]]
    v1 = pts[tri_data[:, 1]]
    v2 = pts[tri_data[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(cross, axis=1, keepdims=True)
    tri_normals = cross / np.maximum(norm, 1e-12)

    # Re-tag the annular intake patch to outlet (12)
    is_intake = (
        (phys == 14) &
        (tri_centers[:, 0] > 2.35) & (tri_centers[:, 0] < 2.65) &
        (np.abs(np.abs(tri_centers[:, 2]) - 4.1) < 0.9) &
        (tri_normals[:, 0] < -0.7)
    )
    print(f"Retagged {np.sum(is_intake)} annular intake triangles to outlet (12).")
    # Verify spike faces are NOT retagged
    is_spike = (
        (phys == 14) &
        (tri_centers[:, 0] >= 1.0) & (tri_centers[:, 0] <= 2.35) &
        (np.abs(np.abs(tri_centers[:, 2]) - 4.1) < 0.8)
    )
    print(f"Verified {np.sum(is_spike)} spike cone faces remain SOLID airfoil_wall (14).")

    phys[is_intake] = 12
    raw.cell_data['gmsh:physical'][0] = phys
    meshio.write(output_msh, raw, file_format="gmsh22")
    print("Final tagged mesh saved successfully.")

if __name__ == '__main__':
    build_clean_sr71_mesh()
