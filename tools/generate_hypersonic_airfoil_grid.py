"""Universal Hypersonic Airfoil Structured C-Grid Generator.

Produces 100% structured pure hexahedral meshes with ZERO triangles, ZERO tetrahedra,
and ZERO wedges for the JAX high-order discontinuous Galerkin fluid solver.

Supports:
- Space Shuttle Orbiter re-entry cross-section (blunted TPS profile)
- NASA X-43A scramjet waverider profile (thin blunted diamond/double-wedge)
- Arbitrary angle of attack alpha (pitched clockwise so horizontal freestream creates positive lift)
"""

import argparse
from pathlib import Path
import subprocess
import numpy as np
import meshio

from hybrid.mesh import _ELEMENT_ALIASES, _LOCAL_FACES


def generate_hypersonic_airfoil_geo(
    profile: str = "shuttle",
    chord: float = 1.0,
    alpha_deg: float = 35.0,
    r_far: float = 3.0,
    x_out: float = 4.5,
    dz: float = 0.1,
    n_wall: int = 28,
    n_rad: int = 36,
    n_wake: int = 40,
    prog_rad: float = 1.06,
    prog_wake: float = 1.04,
    p_degree: int = 2,
) -> str:
    """Generate Gmsh OpenCASCADE script for structured 6-block hypersonic airfoil C-grid."""
    alpha = np.radians(alpha_deg)
    x_rot = 0.25 * chord
    y_rot = 0.0

    def rot(xb, yb):
        # Clockwise rotation: pitches nose UP (+y), belly faces oncoming horizontal flow (+x)
        dx = xb - x_rot
        dy = yb - y_rot
        xw = x_rot + dx * np.cos(alpha) + dy * np.sin(alpha)
        yw = y_rot - dx * np.sin(alpha) + dy * np.cos(alpha)
        return float(xw), float(yw)

    if profile == "shuttle":
        # Space Shuttle Orbiter re-entry cross section
        r_le = 0.06 * chord
        x_nose_c, y_nose_c = rot(r_le, 0.0)
        x_le, y_le = rot(0.0, 0.0)
        x_nose_top, y_nose_top = rot(r_le, r_le)
        x_nose_bot, y_nose_bot = rot(r_le, -r_le)
        # Cambered upper shoulder (max thickness t/c ≈ 10% at x/c ≈ 0.35)
        x_sh_top, y_sh_top = rot(0.35 * chord, 0.09 * chord)
        # Flat thermal protection tile lower belly tangency
        x_sh_bot, y_sh_bot = rot(0.35 * chord, -0.045 * chord)
        # Trailing edge
        x_te, y_te = rot(1.0 * chord, 0.0)
    elif profile == "x43a":
        # NASA X-43A thin blunted diamond (double-wedge) profile
        r_le = 0.008 * chord
        x_nose_c, y_nose_c = rot(r_le, 0.0)
        x_le, y_le = rot(0.0, 0.0)
        x_nose_top, y_nose_top = rot(r_le, r_le)
        x_nose_bot, y_nose_bot = rot(r_le, -r_le)
        # Symmetrical mid-chord diamond ridge at x/c = 0.5 (t/c = 5%)
        x_sh_top, y_sh_top = rot(0.50 * chord, 0.025 * chord)
        x_sh_bot, y_sh_bot = rot(0.50 * chord, -0.025 * chord)
        # Trailing edge
        x_te, y_te = rot(1.0 * chord, 0.0)
    else:
        raise ValueError(f"Unknown profile: {profile}. Supported: 'shuttle', 'x43a'")

    # Outer Farfield points
    x_far_le = -r_far + x_rot
    y_far_le = y_rot
    x_far_top = x_rot
    y_far_top = r_far
    x_far_bot = x_rot
    y_far_bot = -r_far
    x_far_top_te = 1.0 * chord
    y_far_top_te = r_far
    x_far_bot_te = 1.0 * chord
    y_far_bot_te = -r_far

    y_wake = y_te

    lines = [
        'SetFactory("OpenCASCADE");',
        '',
        '// Airfoil landmarks (profile: %s, rotated alpha = %.1f deg)' % (profile, alpha_deg),
        'p_le = newp; Point(p_le) = {%.6f, %.6f, 0};' % (x_le, y_le),
        'p_nose_c = newp; Point(p_nose_c) = {%.6f, %.6f, 0};' % (x_nose_c, y_nose_c),
        'p_nose_top = newp; Point(p_nose_top) = {%.6f, %.6f, 0};' % (x_nose_top, y_nose_top),
        'p_nose_bot = newp; Point(p_nose_bot) = {%.6f, %.6f, 0};' % (x_nose_bot, y_nose_bot),
        'p_sh_top = newp; Point(p_sh_top) = {%.6f, %.6f, 0};' % (x_sh_top, y_sh_top),
        'p_sh_bot = newp; Point(p_sh_bot) = {%.6f, %.6f, 0};' % (x_sh_bot, y_sh_bot),
        'p_te = newp; Point(p_te) = {%.6f, %.6f, 0};' % (x_te, y_te),
        '',
        '// Outer boundary landmarks',
        'p_far_c = newp; Point(p_far_c) = {%.6f, %.6f, 0};' % (x_rot, y_rot),
        'p_far_le = newp; Point(p_far_le) = {%.6f, %.6f, 0};' % (x_far_le, y_far_le),
        'p_far_top = newp; Point(p_far_top) = {%.6f, %.6f, 0};' % (x_far_top, y_far_top),
        'p_far_bot = newp; Point(p_far_bot) = {%.6f, %.6f, 0};' % (x_far_bot, y_far_bot),
        'p_far_top_te = newp; Point(p_far_top_te) = {%.6f, %.6f, 0};' % (x_far_top_te, y_far_top_te),
        'p_far_bot_te = newp; Point(p_far_bot_te) = {%.6f, %.6f, 0};' % (x_far_bot_te, y_far_bot_te),
        'p_out_top = newp; Point(p_out_top) = {%.6f, %.6f, 0};' % (x_out, r_far),
        'p_out_bot = newp; Point(p_out_bot) = {%.6f, %.6f, 0};' % (x_out, -r_far),
        'p_out_mid = newp; Point(p_out_mid) = {%.6f, %.6f, 0};' % (x_out, y_wake),
        '',
        '// Inner airfoil curves: single curve per block for pure quad mapping',
        'c_fwd_top = newc; BSpline(c_fwd_top) = {p_le, p_nose_top, p_sh_top};',
        'c_fwd_bot = newc; BSpline(c_fwd_bot) = {p_le, p_nose_bot, p_sh_bot};',
        'l_aft_top = newl; Line(l_aft_top) = {p_sh_top, p_te};',
        'l_aft_bot = newl; Line(l_aft_bot) = {p_sh_bot, p_te};',
        '',
        '// Outer boundary curves',
        'c_far_top = newc; Circle(c_far_top) = {p_far_top, p_far_c, p_far_le};',
        'c_far_bot = newc; Circle(c_far_bot) = {p_far_le, p_far_c, p_far_bot};',
        'l_far_top_mid = newl; Line(l_far_top_mid) = {p_far_top, p_far_top_te};',
        'l_far_bot_mid = newl; Line(l_far_bot_mid) = {p_far_bot, p_far_bot_te};',
        'l_far_top_wake = newl; Line(l_far_top_wake) = {p_far_top_te, p_out_top};',
        'l_far_bot_wake = newl; Line(l_far_bot_wake) = {p_far_bot_te, p_out_bot};',
        '',
        '// Radial rays (wall-normal clustering)',
        'l_rad_le = newl; Line(l_rad_le) = {p_le, p_far_le};',
        'l_rad_top_sh = newl; Line(l_rad_top_sh) = {p_sh_top, p_far_top};',
        'l_rad_bot_sh = newl; Line(l_rad_bot_sh) = {p_sh_bot, p_far_bot};',
        'l_rad_te_top = newl; Line(l_rad_te_top) = {p_te, p_far_top_te};',
        'l_rad_te_bot = newl; Line(l_rad_te_bot) = {p_te, p_far_bot_te};',
        '',
        '// Wake and outlet',
        'l_wake = newl; Line(l_wake) = {p_te, p_out_mid};',
        'l_out_top = newl; Line(l_out_top) = {p_out_mid, p_out_top};',
        'l_out_bot = newl; Line(l_out_bot) = {p_out_bot, p_out_mid};',
        '',
        '// Block 1: Top-Front',
        'cl1 = newcl; Curve Loop(cl1) = {l_rad_top_sh, c_far_top, -l_rad_le, -c_fwd_top};',
        's1 = news; Plane Surface(s1) = {cl1};',
        '',
        '// Block 2: Bottom-Front',
        'cl2 = newcl; Curve Loop(cl2) = {l_rad_le, c_far_bot, -l_rad_bot_sh, -c_fwd_bot};',
        's2 = news; Plane Surface(s2) = {cl2};',
        '',
        '// Block 3: Top-Mid',
        'cl3 = newcl; Curve Loop(cl3) = {l_rad_top_sh, l_far_top_mid, -l_rad_te_top, -l_aft_top};',
        's3 = news; Plane Surface(s3) = {cl3};',
        '',
        '// Block 4: Bottom-Mid',
        'cl4 = newcl; Curve Loop(cl4) = {l_aft_bot, l_rad_te_bot, -l_far_bot_mid, -l_rad_bot_sh};',
        's4 = news; Plane Surface(s4) = {cl4};',
        '',
        '// Block 5: Top-Wake',
        'cl5 = newcl; Curve Loop(cl5) = {l_wake, l_out_top, -l_far_top_wake, -l_rad_te_top};',
        's5 = news; Plane Surface(s5) = {cl5};',
        '',
        '// Block 6: Bottom-Wake',
        'cl6 = newcl; Curve Loop(cl6) = {l_wake, -l_out_bot, -l_far_bot_wake, -l_rad_te_bot};',
        's6 = news; Plane Surface(s6) = {cl6};',
        '',
        '// Transfinite structured grid mapping (100% pure quads)',
        'n_wall = %d;' % n_wall,
        'n_rad = %d;' % n_rad,
        'n_wake = %d;' % n_wake,
        '',
        'Transfinite Curve {l_rad_le, l_rad_top_sh, l_rad_bot_sh, l_rad_te_top, l_rad_te_bot, l_out_top, l_out_bot} = n_rad Using Progression %.4f;' % prog_rad,
        'Transfinite Curve {c_far_top, c_far_bot, c_fwd_top, c_fwd_bot} = n_wall;',
        'Transfinite Curve {l_aft_top, l_aft_bot, l_far_top_mid, l_far_bot_mid} = n_wall;',
        'Transfinite Curve {l_wake, l_far_top_wake, l_far_bot_wake} = n_wake Using Progression %.4f;' % prog_wake,
        '',
        'Transfinite Surface {s1}; Recombine Surface {s1};',
        'Transfinite Surface {s2}; Recombine Surface {s2};',
        'Transfinite Surface {s3}; Recombine Surface {s3};',
        'Transfinite Surface {s4}; Recombine Surface {s4};',
        'Transfinite Surface {s5}; Recombine Surface {s5};',
        'Transfinite Surface {s6}; Recombine Surface {s6};',
        '',
        '// Extrude 1 layer in z to form pure 3D hexahedra',
        'ext[] = Extrude {0, 0, %.4f} { Surface{s1, s2, s3, s4, s5, s6}; Layers{1}; Recombine; };' % dz,
        'Coherence;',
        '',
        'Physical Volume("fluid", 1) = {Volume{:}};',
        'Mesh.ElementOrder = %d;' % p_degree,
        'Mesh.MshFileVersion = 4.1;',
    ]
    return '\n'.join(lines)


def generate_hypersonic_airfoil_mesh(
    profile: str = "shuttle",
    chord: float = 1.0,
    alpha_deg: float = 35.0,
    output_path: str = "mesh/shuttle_airfoil_a35.msh",
    r_far: float = 3.0,
    x_out: float = 4.5,
    dz: float = 0.1,
    n_wall: int = 28,
    n_rad: int = 36,
    n_wake: int = 40,
    prog_rad: float = 1.06,
    prog_wake: float = 1.04,
    p_degree: int = 2,
) -> Path:
    """Generate and tag a structured 3D hexahedral hypersonic airfoil mesh."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_geo = output_file.with_suffix(".temp.geo")
    temp_msh = output_file.with_suffix(".temp.msh")

    geo_content = generate_hypersonic_airfoil_geo(
        profile=profile,
        chord=chord,
        alpha_deg=alpha_deg,
        r_far=r_far,
        x_out=x_out,
        dz=dz,
        n_wall=n_wall,
        n_rad=n_rad,
        n_wake=n_wake,
        prog_rad=prog_rad,
        prog_wake=prog_wake,
        p_degree=p_degree,
    )

    with open(temp_geo, "w") as f:
        f.write(geo_content)

    print(f"Generating structured C-grid for {profile} airfoil at alpha={alpha_deg} deg...")
    cmd = ["gmsh", "-3", str(temp_geo), "-o", str(temp_msh)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Gmsh failed:\n{res.stderr}")

    raw = meshio.read(str(temp_msh))
    pts = np.asarray(raw.points, dtype=np.float64)
    pts[:, 2] -= dz * 0.5  # center span at z=0

    cells_3d = []
    tags_3d = []
    for cb in raw.cells:
        alias = _ELEMENT_ALIASES.get(cb.type)
        if alias is not None and alias[0] in ("hexahedron", "wedge", "tetra"):
            cells_3d.append((cb.type, cb.data))
            tags_3d.append(np.full(len(cb.data), 1, dtype=np.int32))

    # Identify external boundary faces
    pending = {}
    for cb_type, cb_data in cells_3d:
        elem_type, corner_count = _ELEMENT_ALIASES[cb_type]
        corners = cb_data[:, :corner_count]
        local_faces = _LOCAL_FACES[elem_type]
        for cell_idx, cell in enumerate(corners):
            for lf_idx, lf in enumerate(local_faces):
                nodes = tuple(int(cell[v]) for v in lf)
                key = tuple(sorted(nodes))
                if key in pending:
                    pending.pop(key)
                else:
                    pending[key] = (elem_type, cell_idx, lf_idx, nodes)

    tol = 1.0e-3
    quad_faces, quad_tags = [], []
    tag_counts = {}

    for elem_type, cell_idx, lf_idx, nodes in pending.values():
        face_pts = pts[list(nodes)]
        fx, fy, fz = face_pts[:, 0], face_pts[:, 1], face_pts[:, 2]

        # 1. Front/back symmetry planes
        if np.all(np.abs(fz - (-dz * 0.5)) < tol) or np.all(np.abs(fz - (dz * 0.5)) < tol):
            t = 15  # symmetry
        # 2. Downstream outlet plane
        elif np.all(np.abs(fx - x_out) < tol):
            t = 12  # outlet
        # 3. Top and bottom farfield boundaries
        elif np.all(np.abs(fy - r_far) < tol) or np.all(np.abs(fy - (-r_far)) < tol):
            t = 13  # walls / slip farfield
        # 4. Upstream circular C-boundary
        elif np.all(np.sqrt((fx - 0.25 * chord)**2 + fy**2) > r_far - 0.2):
            t = 11  # inlet
        # 5. Inner airfoil solid wall
        else:
            t = 14  # airfoil_wall (no-slip solid wall)

        tag_counts[t] = tag_counts.get(t, 0) + 1
        quad_faces.append(nodes)
        quad_tags.append(t)

    tag_names = {
        11: "inlet",
        12: "outlet",
        13: "walls (farfield slip)",
        14: "airfoil_wall (no-slip)",
        15: "symmetry (z-planes)",
    }
    for t, cnt in sorted(tag_counts.items()):
        print(f"  Boundary Tag {t} ({tag_names.get(t, 'other')}): {cnt} faces")

    output_cells = [("quad", np.asarray(quad_faces, dtype=np.int32))]
    output_cell_data = [np.asarray(quad_tags, dtype=np.int32)]
    for (cb_type, cb_data), cb_tags in zip(cells_3d, tags_3d):
        output_cells.append((cb_type, cb_data))
        output_cell_data.append(cb_tags)

    field_data = {
        "fluid": np.array([1, 3]),
        "inlet": np.array([11, 2]),
        "outlet": np.array([12, 2]),
        "walls": np.array([13, 2]),
        "airfoil_wall": np.array([14, 2]),
        "symmetry": np.array([15, 2]),
    }

    out_mesh = meshio.Mesh(
        pts,
        output_cells,
        cell_data={"gmsh:physical": output_cell_data},
        field_data=field_data,
    )
    out_mesh.write(str(output_file), file_format="gmsh22", binary=False)
    print(f"Successfully generated structured hypersonic airfoil mesh: {output_file}")
    print(f"  Total elements: {sum(len(b[1]) for b in cells_3d)} ({cells_3d[0][0]})")

    temp_geo.unlink(missing_ok=True)
    temp_msh.unlink(missing_ok=True)
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hypersonic Airfoil Structured C-Grid Generator")
    parser.add_argument("--profile", default="shuttle", choices=["shuttle", "x43a"])
    parser.add_argument("--alpha", type=float, default=35.0, help="Angle of attack in degrees")
    parser.add_argument("--chord", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n_wall", type=int, default=28)
    parser.add_argument("--n_rad", type=int, default=36)
    parser.add_argument("--n_wake", type=int, default=40)
    parser.add_argument("--prog_rad", type=float, default=1.06)
    parser.add_argument("--prog_wake", type=float, default=1.04)
    parser.add_argument("--p_degree", type=int, default=2)

    args = parser.parse_args()
    if args.output is None:
        args.output = f"mesh/{args.profile}_a{int(args.alpha)}.msh"

    generate_hypersonic_airfoil_mesh(
        profile=args.profile,
        chord=args.chord,
        alpha_deg=args.alpha,
        output_path=args.output,
        n_wall=args.n_wall,
        n_rad=args.n_rad,
        n_wake=args.n_wake,
        prog_rad=args.prog_rad,
        prog_wake=args.prog_wake,
        p_degree=args.p_degree,
    )

