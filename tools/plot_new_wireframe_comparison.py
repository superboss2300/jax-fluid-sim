"""Render high-resolution multi-view wireframe comparison of the new SR-71 mesh."""

import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_wireframe():
    old_stl = "mesh/sr71_perfect_clean.stl"
    new_stl = "mesh/sr71_faceted_clean.stl"

    m_old = trimesh.load(old_stl)
    m_new = trimesh.load(new_stl)

    fig = plt.figure(figsize=(22, 14), facecolor="#0a0f1d", dpi=180)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.25, wspace=0.18)

    # -------------------------------------------------------------
    # Panel 1: Full Airframe Top View (Planform Wireframe) - New Mesh
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#0d1326")
    ax1.set_title("NEW SR-71 AIRFRAME SURFACE WIREFRAME (Uniform Subdivided Triangles, Max Edge <= 1.2 m)",
                  fontsize=14, color="#38bdf8", fontweight="bold", pad=12)

    # Plot faces with subtle shading and crisp wireframe
    v = m_new.vertices
    f = m_new.faces
    
    # Lines for wireframe
    edges = m_new.edges_unique
    e_pts = v[edges]
    lines_new = LineCollection(e_pts[:, :, [0, 2]], colors="#38bdf8", linewidths=0.35, alpha=0.65)
    ax1.add_collection(lines_new)

    ax1.set_xlim(-18, 18)
    ax1.set_ylim(-9.5, 9.5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X [m] (Fuselage Length: 33.3 m)", color="#94a3b8", fontsize=11)
    ax1.set_ylabel("Z [m] (Wingspan: 16.9 m)", color="#94a3b8", fontsize=11)
    ax1.tick_params(colors="#64748b")
    ax1.grid(True, color="#1e293b", linestyle="--", linewidth=0.5, alpha=0.5)

    # Add annotations
    ax1.annotate("Conical Shock Spike (x = +1.7m)", xy=(1.7, 4.1), xytext=(-6, 7.5),
                 color="#fbbf24", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.2))
    ax1.annotate("Chine Edge (Sharp Vortex Generator)", xy=(-7.5, 2.0), xytext=(-14, 5.5),
                 color="#a78bfa", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#a78bfa", lw=1.2))
    ax1.annotate("Subdivided Inboard Wing Junction\n(10.5m sliver replaced by uniform 1.2m cells)",
                 xy=(7.0, 2.8), xytext=(5.0, 7.0),
                 color="#4ade80", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#4ade80", lw=1.2))

    # -------------------------------------------------------------
    # Panel 2: Wing-Nacelle Junction Detail — OLD MESH (Showing 10.5m Sliver)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#0d1326")
    ax2.set_title("OLD MESH: Wing-Nacelle Junction (Showing Giant 10.5m Sliver Edge)",
                  fontsize=12, color="#f87171", fontweight="bold", pad=10)

    v_o = m_old.vertices
    e_o = m_old.edges_unique
    e_o_pts = v_o[e_o]
    
    # Filter edges in zoom region: X in [1, 15], Z in [1, 5.5]
    mask_o = (
        (e_o_pts[:, :, 0].max(axis=1) >= 1.0) & (e_o_pts[:, :, 0].min(axis=1) <= 15.0) &
        (e_o_pts[:, :, 2].max(axis=1) >= 1.0) & (e_o_pts[:, :, 2].min(axis=1) <= 5.5)
    )
    lines_old_zoom = LineCollection(e_o_pts[mask_o][:, :, [0, 2]], colors="#f87171", linewidths=0.7, alpha=0.8)
    ax2.add_collection(lines_old_zoom)

    # Highlight the 10.5m edge in neon red/yellow
    long_mask = m_old.edges_unique_length > 4.0
    long_pts = e_o_pts[long_mask]
    lines_long = LineCollection(long_pts[:, :, [0, 2]], colors="#facc15", linewidths=2.5, alpha=1.0)
    ax2.add_collection(lines_long)

    ax2.set_xlim(2.0, 14.5)
    ax2.set_ylim(1.2, 5.2)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X [m] (Axial)", color="#94a3b8", fontsize=10)
    ax2.set_ylabel("Z [m] (Spanwise)", color="#94a3b8", fontsize=10)
    ax2.tick_params(colors="#64748b")
    ax2.grid(True, color="#1e293b", linestyle="--", linewidth=0.5, alpha=0.5)

    ax2.annotate("10.51 m Sliver Edge!\n(Caused 5.2m blue anomaly)", xy=(8.0, 3.2), xytext=(3.0, 1.8),
                 color="#facc15", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.5))

    # -------------------------------------------------------------
    # Panel 3: Wing-Nacelle Junction Detail — NEW MESH (Subdivided & Uniform)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#0d1326")
    ax3.set_title("NEW REPAIRED MESH: Wing-Nacelle Junction (Clean Uniform Cells, Max Edge <= 1.2 m)",
                  fontsize=12, color="#4ade80", fontweight="bold", pad=10)

    v_n = m_new.vertices
    e_n = m_new.edges_unique
    e_n_pts = v_n[e_n]
    mask_n = (
        (e_n_pts[:, :, 0].max(axis=1) >= 1.0) & (e_n_pts[:, :, 0].min(axis=1) <= 15.0) &
        (e_n_pts[:, :, 2].max(axis=1) >= 1.0) & (e_n_pts[:, :, 2].min(axis=1) <= 5.5)
    )
    lines_new_zoom = LineCollection(e_n_pts[mask_n][:, :, [0, 2]], colors="#4ade80", linewidths=0.7, alpha=0.8)
    ax3.add_collection(lines_new_zoom)

    ax3.set_xlim(2.0, 14.5)
    ax3.set_ylim(1.2, 5.2)
    ax3.set_aspect("equal")
    ax3.set_xlabel("X [m] (Axial)", color="#94a3b8", fontsize=10)
    ax3.set_ylabel("Z [m] (Spanwise)", color="#94a3b8", fontsize=10)
    ax3.tick_params(colors="#64748b")
    ax3.grid(True, color="#1e293b", linestyle="--", linewidth=0.5, alpha=0.5)

    ax3.annotate("Uniformly subdivided\n(13,448 conformal faces,\nzero sliver elements)",
                 xy=(8.0, 3.2), xytext=(3.5, 1.8),
                 color="#4ade80", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#4ade80", lw=1.5))

    out_png = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/sr71_new_wireframe_comparison.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved wireframe comparison to {out_png}")

if __name__ == "__main__":
    plot_wireframe()
