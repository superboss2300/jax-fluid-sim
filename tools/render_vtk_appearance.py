import meshio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

m = meshio.read("output/sr71_cinematic/sr71_surface.vtu")
pts = m.points
quads = m.cells_dict["quad"]
p = m.point_data["pressure"]
mach = m.point_data["mach"]

fig = plt.figure(figsize=(20, 14), facecolor="#0a0f1d")

# Subplot 1: 3/4 Perspective View - Surface Pressure
ax1 = fig.add_subplot(2, 2, 1, projection='3d', facecolor="#0e172a")
quad_pts = pts[quads]
quad_p = np.mean(p[quads], axis=1)

# Sort quads from back to front for crude 3D painters algorithm or use Poly3DCollection
norm_p = plt.Normalize(vmin=0.8, vmax=1.4)
colors = plt.cm.turbo(norm_p(quad_p))

poly1 = Poly3DCollection(quad_pts, facecolors=colors, edgecolors='#1e293b', linewidths=0.2, alpha=0.95)
ax1.add_collection3d(poly1)
ax1.set_xlim(-18, 18); ax1.set_ylim(-10, 10); ax1.set_zlim(-5, 5)
ax1.view_init(elev=25, azim=-125)
ax1.set_title("Perspective View: Surface Pressure (p / p_inf)", color="white", fontsize=12, fontweight="bold")
ax1.axis('off')

# Subplot 2: Top View - Mach number
ax2 = fig.add_subplot(2, 2, 2, projection='3d', facecolor="#0e172a")
quad_m = np.mean(mach[quads], axis=1)
norm_m = plt.Normalize(vmin=1.5, vmax=2.6)
colors_m = plt.cm.turbo(norm_m(quad_m))
poly2 = Poly3DCollection(quad_pts, facecolors=colors_m, edgecolors='#1e293b', linewidths=0.2, alpha=0.95)
ax2.add_collection3d(poly2)
ax2.set_xlim(-18, 18); ax2.set_ylim(-10, 10); ax2.set_zlim(-5, 5)
ax2.view_init(elev=90, azim=-90)
ax2.set_title("Top Planform View: Surface Mach Number", color="white", fontsize=12, fontweight="bold")
ax2.axis('off')

# Subplot 3: Underside Keel View - Pressure
ax3 = fig.add_subplot(2, 2, 3, projection='3d', facecolor="#0e172a")
poly3 = Poly3DCollection(quad_pts, facecolors=colors, edgecolors='#1e293b', linewidths=0.2, alpha=0.95)
ax3.add_collection3d(poly3)
ax3.set_xlim(-18, 18); ax3.set_ylim(-10, 10); ax3.set_zlim(-5, 5)
ax3.view_init(elev=-75, azim=-90)
ax3.set_title("Underside Belly Keel View: Surface Pressure", color="white", fontsize=12, fontweight="bold")
ax3.axis('off')

# Subplot 4: Zoomed Cockpit & Forward Chine Area
ax4 = fig.add_subplot(2, 2, 4, projection='3d', facecolor="#0e172a")
poly4 = Poly3DCollection(quad_pts, facecolors=colors, edgecolors='#1e293b', linewidths=0.3, alpha=1.0)
ax4.add_collection3d(poly4)
ax4.set_xlim(-16, -5); ax4.set_ylim(-4, 4); ax4.set_zlim(-2, 2)
ax4.view_init(elev=30, azim=-140)
ax4.set_title("Zoomed Forebody & Chine: Mesh Faceting & Solution", color="white", fontsize=12, fontweight="bold")
ax4.axis('off')

plt.tight_layout()
out_path = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/paraview_vtk_inspection.png"
plt.savefig(out_path, dpi=160, facecolor="#0a0f1d")
print("Saved inspection to", out_path)
