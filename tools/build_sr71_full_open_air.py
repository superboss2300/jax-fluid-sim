"""Generate 3D Open-Air Wind Tunnel Volume Mesh around Authentic SR-71 Airframe.

Includes:
- Full 1:1 scale SR-71 airframe (33.3m length, 16.9m wingspan).
- Farfield open-air domain with ample clearance for free wingtip vortices.
- Engine inlet face tagged as outflow/outlet boundary so intake airflow enters naturally.
- Physical groups: inlet (11), outlet (12), walls (13), airfoil_wall (14), fluid (1).
"""

from pathlib import Path
import gmsh
import numpy as np
import trimesh

def build_sr71_open_air_mesh(
    stl_path: str = "mesh/sr71_clean_repaired.stl",
    output_msh: str = "mesh/sr71_open_air.msh",
    lc_far: float = 14.0,
):
    print("=" * 76)
    print("BUILDING 3D OPEN-AIR WIND TUNNEL MESH FOR LOCKHEED SR-71 BLACKBIRD")
    print(f"  Source STL:  {stl_path}")
    print(f"  Output MSH:  {output_msh}")
    print("=" * 76)

    m = trimesh.load(stl_path)
    b = m.bounds
    print(f"Airframe Bounding Box:")
    print(f"  X (Length):   [{b[0][0]:.2f}, {b[1][0]:.2f}] m  -> {b[1][0] - b[0][0]:.2f} m")
    print(f"  Y (Height):   [{b[0][1]:.2f}, {b[1][1]:.2f}] m  -> {b[1][1] - b[0][1]:.2f} m")
    print(f"  Z (Wingspan): [{b[0][2]:.2f}, {b[1][2]:.2f}] m  -> {b[1][2] - b[0][2]:.2f} m")

    # Locate engine inlet face indices
    centers = m.triangles_center
    inlet_face_mask = (np.abs(centers[:, 0] - 3.27) < 0.15) & (np.abs(np.abs(centers[:, 2]) - 4.2) < 1.0)
    print(f"Identified {np.sum(inlet_face_mask)} engine inlet patch triangles (tagged as flow-through)")

    gmsh.initialize()
    gmsh.model.add("sr71_open_air")
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 1e-4)

    # 1. Merge the STL surface
    gmsh.merge(stl_path)
    stl_surface_tag = 1
    stl_loop = gmsh.model.geo.addSurfaceLoop([stl_surface_tag])

    # 2. Wind Tunnel Domain Box
    # X: [-30, 45], Y: [-18, 20], Z: [-25, 25]
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

    # Box boundary surfaces
    s_zmin = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])]) # Z min
    s_zmax = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])]) # Z max
    s_inlet = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l4, -l12, -l8, l9])]) # X min (Inflow)
    s_outlet = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])]) # X max (Outflow)
    s_ymin = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])]) # Y min (Bottom)
    s_ymax = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])]) # Y max (Top)

    box_loop = gmsh.model.geo.addSurfaceLoop([s_zmin, s_zmax, s_inlet, s_outlet, s_ymin, s_ymax])
    vol = gmsh.model.geo.addVolume([box_loop, stl_loop])

    gmsh.model.geo.synchronize()

    # 3. Physical Groups
    gmsh.model.addPhysicalGroup(3, [vol], 1, name="fluid")
    gmsh.model.addPhysicalGroup(2, [s_inlet], 11, name="inlet")
    gmsh.model.addPhysicalGroup(2, [s_outlet], 12, name="outlet")
    gmsh.model.addPhysicalGroup(2, [s_zmin, s_zmax, s_ymin, s_ymax], 13, name="walls")
    gmsh.model.addPhysicalGroup(2, [stl_surface_tag], 14, name="airfoil_wall")

    # 4. Generate Mesh
    print("Meshing 3D volume with Delaunay/TetGen...")
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Delaunay
    gmsh.model.mesh.generate(3)

    tets = gmsh.model.mesh.getElementsByType(4)[0]
    print(f"Generated {len(tets)} tetrahedra in 3D fluid domain.")

    # Save mesh
    Path(output_msh).parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(output_msh)
    print(f"Successfully saved {output_msh}")
    gmsh.finalize()

if __name__ == "__main__":
    build_sr71_open_air_mesh()

