"""Render a single full-frame 16:9 widescreen cinematic view of the SR-71."""

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def render_cinematic():
    m = trimesh.load("mesh/sr71_faceted_clean.stl")
    v = m.vertices
    f = m.faces
    
    # In matplotlib 3d: X is x (length), Y is z (span), Z is y (height)
    v_3d = np.column_stack([v[:, 0], v[:, 2], v[:, 1]])
    f_3d = v_3d[f]

    normals = m.face_normals
    n_3d = np.column_stack([normals[:, 0], normals[:, 2], normals[:, 1]])

    x_range = v[:, 0].max() - v[:, 0].min()
    z_range = v[:, 2].max() - v[:, 2].min()
    y_range = v[:, 1].max() - v[:, 1].min()
    
    x_mid = 0.5 * (v[:, 0].max() + v[:, 0].min())
    z_mid = 0.5 * (v[:, 2].max() + v[:, 2].min())
    y_mid = 0.5 * (v[:, 1].max() + v[:, 1].min())

    fig = plt.figure(figsize=(18, 9), facecolor="#030611", dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_facecolor("#030611")

    light = np.array([-0.45, 0.45, 0.75])
    light /= np.linalg.norm(light)
    diffuse = np.clip(np.abs(np.dot(n_3d, light)), 0.15, 1.0)
    colors = np.outer(diffuse, np.array([0.08, 0.22, 0.45]))

    poly = Poly3DCollection(f_3d, alpha=0.95, edgecolor="#0284c7", linewidth=0.25)
    poly.set_facecolor(colors)
    ax.add_collection3d(poly)

    ax.set_box_aspect([x_range, z_range, y_range * 1.5])
    ax.set_xlim(x_mid - x_range*0.48, x_mid + x_range*0.48)
    ax.set_ylim(z_mid - z_range*0.48, z_mid + z_range*0.48)
    ax.set_zlim(y_mid - y_range*0.8, y_mid + y_range*0.8)

    ax.view_init(elev=22, azim=-55)
    ax.set_axis_off()
    ax.dist = 6.2 # Zoomed in close

    out_png = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/sr71_cinematic_16x9.png"
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved 16:9 cinematic shot to {out_png}")

if __name__ == "__main__":
    render_cinematic()
