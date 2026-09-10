"""Generate smooth aerospace O-grid with downstream wake elongation matching cylinder style."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import meshio


def generate_aerospace_ogrid(
    r_inner=0.5,
    r_outer_base=3.0,
    x_wake_stretch=1.0,  # extends wake to x = 4.0
    n_face=10,
    n_radial=24,
    progression=1.06,
):
    # Equiangular coordinates on [-pi/4, pi/4]
    theta = np.linspace(-np.pi / 4, np.pi / 4, n_face + 1)
    tan_vals = np.tan(theta)

    def face_unit(face, a, b):
        if face == 0:    # -X (Nose / Upstream)
            v = np.array([-1.0, b, a])
        elif face == 1:  # +X (Wake / Downstream)
            v = np.array([1.0, b, -a])
        elif face == 2:  # +Y (Top)
            v = np.array([a, 1.0, b])
        elif face == 3:  # -Y (Bottom)
            v = np.array([a, -1.0, -b])
        elif face == 4:  # +Z (Right)
            v = np.array([a, b, 1.0])
        elif face == 5:  # -Z (Left)
            v = np.array([-a, b, -1.0])
        return v / np.linalg.norm(v)

    def outer_radius(u):
        # u is unit direction vector. u[0] is x-component (inflow is -1, wake is +1)
        # Smoothly elongate in +x direction
        if u[0] > 0:
            return r_outer_base + x_wake_stretch * (u[0] ** 1.5)
        return r_outer_base

    node_dict = {}
    node_coords = []

    def get_nid(pt):
        k = (round(float(pt[0]), 7), round(float(pt[1]), 7), round(float(pt[2]), 7))
        if k not in node_dict:
            nid = len(node_coords)
            node_dict[k] = nid
            node_coords.append([pt[0], pt[1], pt[2]])
            return nid
        return node_dict[k]

    grids = np.zeros((6, n_face + 1, n_face + 1, n_radial + 1), dtype=np.int32)

    for face in range(6):
        for i in range(n_face + 1):
            a = tan_vals[i]
            for j in range(n_face + 1):
                b = tan_vals[j]
                u = face_unit(face, a, b)
                r_out = outer_radius(u)
                
                # Geometric radial layers
                geom = (progression ** np.arange(n_radial + 1) - 1.0) / (progression ** n_radial - 1.0)
                r_layers = r_inner + (r_out - r_inner) * geom

                for k in range(n_radial + 1):
                    pt = r_layers[k] * u
                    grids[face, i, j, k] = get_nid(pt)

    pts = np.asarray(node_coords, dtype=np.float64)
    print(f"Generated {len(pts)} unique nodes.")

    edges_3d = set()
    hex_list = []
    for face in range(6):
        for i in range(n_face):
            for j in range(n_face):
                for k in range(n_radial):
                    v = [
                        grids[face, i, j, k],
                        grids[face, i + 1, j, k],
                        grids[face, i + 1, j + 1, k],
                        grids[face, i, j + 1, k],
                        grids[face, i, j, k + 1],
                        grids[face, i + 1, j, k + 1],
                        grids[face, i + 1, j + 1, k + 1],
                        grids[face, i, j + 1, k + 1],
                    ]
                    hex_list.append(v)
                    hex_edges = [
                        (v[0], v[1]), (v[1], v[2]), (v[2], v[3]), (v[3], v[0]),
                        (v[4], v[5]), (v[5], v[6]), (v[6], v[7]), (v[7], v[4]),
                        (v[0], v[4]), (v[1], v[5]), (v[2], v[6]), (v[3], v[7]),
                    ]
                    for u1, u2 in hex_edges:
                        edges_3d.add(tuple(sorted((u1, u2))))

    # Midplane segments exactly on z=0
    midplane_segments = []
    for u1, u2 in edges_3d:
        p1, p2 = pts[u1], pts[u2]
        if abs(p1[2]) < 1e-5 and abs(p2[2]) < 1e-5:
            midplane_segments.append([(p1[0], p1[1]), (p2[0], p2[1])])

    # Plot
    fig = plt.figure(figsize=(18, 9), facecolor="#030712")

    # 1. Full Domain
    ax1 = fig.add_subplot(1, 2, 1, facecolor="#030712")
    lc1 = LineCollection(midplane_segments, colors="#38bdf8", linewidths=0.9, alpha=0.85)
    ax1.add_collection(lc1)
    th = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(r_inner * np.cos(th), r_inner * np.sin(th), color="#f43f5e", linewidth=2.5, label="Sphere Wall (R=0.5m)")
    ax1.set_xlim(-3.3, 4.3)
    ax1.set_ylim(-3.3, 3.3)
    ax1.set_aspect("equal")
    ax1.set_title("Aerospace Conformal 3D Spherical O-Grid (Full Midplane Z=0)", color="white", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("x (m)", color="#94a3b8")
    ax1.set_ylabel("y (m)", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    for spine in ax1.spines.values():
        spine.set_color("#334155")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#64748b")
    ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white")

    # 2. Zoomed View
    ax2 = fig.add_subplot(1, 2, 2, facecolor="#030712")
    lc2 = LineCollection(midplane_segments, colors="#38bdf8", linewidths=1.2, alpha=0.9)
    ax2.add_collection(lc2)
    ax2.plot(r_inner * np.cos(th), r_inner * np.sin(th), color="#f43f5e", linewidth=3.0)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect("equal")
    ax2.set_title("Zoomed Near-Body: Concentric Shells & Boundary Orthogonality", color="white", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("x (m)", color="#94a3b8")
    ax2.set_ylabel("y (m)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_color("#334155")
    ax2.grid(True, linestyle="--", alpha=0.2, color="#64748b")

    plt.suptitle("Pure Hexahedral 3D Sphere O-Grid (100% Conformal, Extended Wake)", color="white", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    out = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/sphere_aerospace_ogrid.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out}")

generate_aerospace_ogrid(n_face=10, n_radial=24)
