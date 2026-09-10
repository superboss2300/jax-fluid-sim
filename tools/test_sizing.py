import gmsh
import trimesh

stl_path = "mesh/sr71_perfect_clean.stl"
for vin in [1.0, 0.8, 0.65]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add(f"test_{vin}")
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 1e-4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.35)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 10.0)

    gmsh.merge(stl_path)
    stl_surf = 1
    stl_loop = gmsh.model.geo.addSurfaceLoop([stl_surf])

    # Wind tunnel box
    x0, x1 = -30.0, 45.0
    y0, y1 = -18.0, 20.0
    z0, z1 = -25.0, 25.0
    lc_far = 10.0

    p1 = gmsh.model.geo.addPoint(x0, y0, z0, lc_far); p2 = gmsh.model.geo.addPoint(x1, y0, z0, lc_far)
    p3 = gmsh.model.geo.addPoint(x1, y1, z0, lc_far); p4 = gmsh.model.geo.addPoint(x0, y1, z0, lc_far)
    p5 = gmsh.model.geo.addPoint(x0, y0, z1, lc_far); p6 = gmsh.model.geo.addPoint(x1, y0, z1, lc_far)
    p7 = gmsh.model.geo.addPoint(x1, y1, z1, lc_far); p8 = gmsh.model.geo.addPoint(x0, y1, z1, lc_far)

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

    # Sizing field
    f = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f, "VIn", vin)
    gmsh.model.mesh.field.setNumber(f, "VOut", lc_far)
    gmsh.model.mesh.field.setNumber(f, "XMin", -18.0); gmsh.model.mesh.field.setNumber(f, "XMax", 18.0)
    gmsh.model.mesh.field.setNumber(f, "YMin", -2.5);  gmsh.model.mesh.field.setNumber(f, "YMax", 3.0)
    gmsh.model.mesh.field.setNumber(f, "ZMin", -9.0);  gmsh.model.mesh.field.setNumber(f, "ZMax", 9.0)
    gmsh.model.mesh.field.setNumber(f, "Thickness", 3.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)

    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.model.mesh.generate(3)

    tets = gmsh.model.mesh.getElementsByType(4)[0]
    n_hex = len(tets) * 4
    print(f"VIn={vin}: {len(tets)} tets -> {n_hex} hexes.")
    gmsh.finalize()
