import meshio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

m = meshio.read("output/sr71_cinematic/sr71_surface.vtu")
pts = m.points
mach = m.point_data["mach"]
quads = m.cells_dict["quad"]

fig = plt.figure(figsize=(14, 10), facecolor="#222222")
ax = fig.add_subplot(1, 1, 1, projection='3d', facecolor="#2a2a2a")

# Filter quads within 3m of the stagnant node [8.658, -0.848, -2.556]
target = np.array([8.658, -0.848, -2.556])
quad_centers = np.mean(pts[quads], axis=1)
dists = np.linalg.norm(quad_centers - target, axis=1)
sub_quads = quads[dists < 4.0]

sub_pts = pts[sub_quads]
sub_mach = np.mean(mach[sub_quads], axis=1)

norm = plt.Normalize(vmin=0.2, vmax=2.5)
colors = plt.cm.coolwarm(norm(sub_mach))

poly = Poly3DCollection(sub_pts, facecolors=colors, edgecolors='#333333', linewidths=0.5, alpha=1.0)
ax.add_collection3d(poly)

# Mark the stagnant node
ax.scatter([target[0]], [target[2]], [target[1]], color="blue", s=100, label="Stagnant Node (Mach 0.17)")

ax.set_xlim(5, 12); ax.set_ylim(-5, 0); ax.set_zlim(-2, 1)
ax.view_init(elev=45, azim=-120)
ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
ax.set_title("Zoomed View of Stagnant Node Region on Inboard Wing", color="white", fontsize=14, fontweight="bold")

plt.tight_layout()
out = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/stagnant_node_zoom.png"
plt.savefig(out, dpi=160, facecolor="#222222")
print("Saved to", out)
