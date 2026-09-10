import meshio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

m = meshio.read("mesh/sr71_pristine.msh")
pts = m.points
tris = m.cells_dict["triangle"]
phys = m.cell_data_dict["gmsh:physical"]["triangle"]

# Outlet (tag 12) triangles
outlet_mask = (phys == 12)
outlet_tris = tris[outlet_mask]

# All airfoil_wall (tag 14) triangles near the nacelle
nacelle_mask = (phys == 14) & (pts[tris[:, 0], 0] > 0.0) & (pts[tris[:, 0], 0] < 8.0) & (np.abs(pts[tris[:, 0], 2]) > 2.0)
nacelle_tris = tris[nacelle_mask]

fig = plt.figure(figsize=(16, 12), facecolor="#222222")
ax = fig.add_subplot(1, 1, 1, projection='3d', facecolor="#2a2a2a")

# Plot nacelle in gray
poly_nac = Poly3DCollection(pts[nacelle_tris], facecolors='#888888', edgecolors='#555555', linewidths=0.2, alpha=0.5)
ax.add_collection3d(poly_nac)

# Plot outlet triangles in bright cyan
poly_out = Poly3DCollection(pts[outlet_tris], facecolors='#00ffff', edgecolors='#0088ff', linewidths=0.5, alpha=1.0)
ax.add_collection3d(poly_out)

ax.set_xlim(0, 6); ax.set_ylim(-2, 2); ax.set_zlim(2, 6)
ax.view_init(elev=30, azim=-130)
ax.set_title("Outlet (Tag 12) Triangles in MSH (Cyan)", color="white", fontsize=14, fontweight="bold")

plt.tight_layout()
out = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/outlet_tag_visual.png"
plt.savefig(out, dpi=160, facecolor="#222222")
print("Saved to", out)
