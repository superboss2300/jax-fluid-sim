import meshio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(22, 12), facecolor="#0a0f1d")

# Subplot 1: Old raw barycentric surface mesh (faceted, spider-web)
ax1 = fig.add_subplot(1, 2, 1, projection='3d', facecolor="#0e172a")
m_old = meshio.read("output/sr71_cinematic/sr71_surface.vtu")
pts_old = m_old.points
quads_old = m_old.cells_dict["quad"]
p_old = m_old.point_data["pressure"]
quad_p_old = np.mean(p_old[quads_old], axis=1)

norm_p = plt.Normalize(vmin=0.8, vmax=1.4)
colors_old = plt.cm.turbo(norm_p(quad_p_old))

poly1 = Poly3DCollection(pts_old[quads_old], facecolors=colors_old, edgecolors='#1e293b', linewidths=0.2, alpha=0.95)
ax1.add_collection3d(poly1)
ax1.set_xlim(-16, 16); ax1.set_ylim(-9, 9); ax1.set_zlim(-3, 3)
ax1.view_init(elev=28, azim=-125)
ax1.set_title("BEFORE: Raw Subdivided Hex Mesh (Faceted / Origami Look)", color="#ff6b6b", fontsize=14, fontweight="bold")
ax1.axis('off')

# Subplot 2: New smooth CAD surface (sleek, high-poly, smooth contours)
ax2 = fig.add_subplot(1, 2, 2, projection='3d', facecolor="#0e172a")
m_new = meshio.read("output/sr71_cinematic/sr71_surface_smooth.vtu")
pts_new = m_new.points
tris_new = m_new.cells_dict["triangle"]
p_new = m_new.point_data["pressure"]
tri_p_new = np.mean(p_new[tris_new], axis=1)
colors_new = plt.cm.turbo(norm_p(tri_p_new))

poly2 = Poly3DCollection(pts_new[tris_new], facecolors=colors_new, edgecolors='none', alpha=1.0)
ax2.add_collection3d(poly2)
ax2.set_xlim(-16, 16); ax2.set_ylim(-9, 9); ax2.set_zlim(-3, 3)
ax2.view_init(elev=28, azim=-125)
ax2.set_title("AFTER: Pristine CAD Skin (sr71_surface_smooth.vtu)", color="#00f2fe", fontsize=14, fontweight="bold")
ax2.axis('off')

plt.tight_layout()
out_path = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/sr71_vtk_quality_comparison.png"
plt.savefig(out_path, dpi=160, facecolor="#0a0f1d")
print("Saved comparison to", out_path)
