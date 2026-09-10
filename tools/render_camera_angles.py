"""Render showcase of the 4 premier cinematic camera angles with TRUE physical aspect ratio."""

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def render_angles():
    m = trimesh.load("mesh/sr71_faceted_clean.stl")
    v = m.vertices
    f = m.faces
    
    # In matplotlib 3d: X is x (length), Y is z (span), Z is y (height)
    v_3d = np.column_stack([v[:, 0], v[:, 2], v[:, 1]])
    f_3d = v_3d[f]

    normals = m.face_normals
    n_3d = np.column_stack([normals[:, 0], normals[:, 2], normals[:, 1]])

    # Aircraft physical spans
    x_range = v[:, 0].max() - v[:, 0].min() # 33.3m
    z_range = v[:, 2].max() - v[:, 2].min() # 16.9m
    y_range = v[:, 1].max() - v[:, 1].min() # 4.18m
    
    x_mid = 0.5 * (v[:, 0].max() + v[:, 0].min())
    z_mid = 0.5 * (v[:, 2].max() + v[:, 2].min())
    y_mid = 0.5 * (v[:, 1].max() + v[:, 1].min())

    angles = [
        {
            "id": 1,
            "title": "Angle 1: Front 3/4 Chase-Plane (Close-Up Flight Formation)",
            "desc": "Shows true 33.3m length & 16.9m wingspan: nose bow shock, chine vortex, and engine spike.",
            "elev": 20, "azim": -55, "light": [-0.5, 0.4, 0.7]
        },
        {
            "id": 2,
            "title": "Angle 2: Top-Down Planform Sweep (True Wingspan & Double-Delta Planform)",
            "desc": "Shows full spanwise vortex roll-up over both port and starboard delta wings.",
            "elev": 58, "azim": -42, "light": [0.0, 0.0, 1.0]
        },
        {
            "id": 3,
            "title": "Angle 3: Rear 3/4 Afterbody Wake (Twin 15 deg Inward Canted Rudders)",
            "desc": "Shows engine exhaust cowls, aft boattail, and directional stability fins.",
            "elev": 18, "azim": 128, "light": [0.5, -0.4, 0.7]
        },
        {
            "id": 4,
            "title": "Angle 4: Low-Angle Supersonic Flyby (Windward Compression Ramp)",
            "desc": "Shows the flat high-pressure supersonic compression belly and chine knife edge.",
            "elev": -10, "azim": -62, "light": [-0.6, 0.6, -0.5]
        },
    ]

    fig = plt.figure(figsize=(24, 15), facecolor="#040711", dpi=180)
    plt.suptitle("Lockheed SR-71 Blackbird — Authentic Full-Scale (33.3m / 109.3 ft) Perspectives",
                 fontsize=18, color="#f8fafc", fontweight="bold", y=0.985)

    for i, a in enumerate(angles):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.set_facecolor("#070c18")
        ax.set_title(a["title"], color="#38bdf8", fontsize=12, fontweight="bold", pad=10)

        # Lighting calculation
        light = np.array(a["light"])
        light /= np.linalg.norm(light)
        diffuse = np.clip(np.abs(np.dot(n_3d, light)), 0.18, 1.0)
        colors = np.outer(diffuse, np.array([0.08, 0.22, 0.44]))

        poly = Poly3DCollection(f_3d, alpha=0.94, edgecolor="#0284c7", linewidth=0.22)
        poly.set_facecolor(colors)
        ax.add_collection3d(poly)

        # Enforce TRUE physical aspect ratio: length : span : height = 33.3 : 16.9 : 4.2
        ax.set_box_aspect([x_range, z_range, y_range * 1.6])

        # Tight bounds centered on aircraft
        ax.set_xlim(x_mid - x_range*0.52, x_mid + x_range*0.52)
        ax.set_ylim(z_mid - z_range*0.55, z_mid + z_range*0.55)
        ax.set_zlim(y_mid - y_range*0.9, y_mid + y_range*0.9)

        ax.view_init(elev=a["elev"], azim=a["azim"])
        ax.set_axis_off()

        # In matplotlib 3d, adjust zoom
        ax.dist = 7.8 # Closer zoom (default is 10)

        ax.text2D(0.5, 0.02, a["desc"], transform=ax.transAxes,
                  color="#94a3b8", fontsize=9.5, ha="center", style="italic")

    out_png = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/camera_angles_showcase_close.png"
    plt.tight_layout()
    plt.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved close-up camera showcase to {out_png}")

if __name__ == "__main__":
    render_angles()
