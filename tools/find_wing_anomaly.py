import meshio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

m = meshio.read("output/sr71_cinematic/sr71_surface.vtu")
pts = m.points
quads = m.cells_dict["quad"]

fields = ["mach", "pressure", "cp", "density", "temperature", "vorticity_magnitude", "Q_criterion"]

fig = plt.figure(figsize=(24, 14), facecolor="#222222")

for idx, f in enumerate(fields):
    ax = fig.add_subplot(2, 4, idx + 1, projection='3d', facecolor="#2a2a2a")
    val = m.point_data[f]
    quad_val = np.mean(val[quads], axis=1)
    
    # ParaView default colormap is coolwarm or rainbow (jet/turbo)
    # In user image: background aircraft is dark red/orange, anomaly is bright blue/cyan!
    # That means the field has high values everywhere else (red) and low value at anomaly (blue)!
    vmin, vmax = np.percentile(quad_val, 1), np.percentile(quad_val, 99)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = plt.cm.turbo(norm(quad_val))
    
    poly = Poly3DCollection(pts[quads], facecolors=colors, edgecolors='none', alpha=1.0)
    ax.add_collection3d(poly)
    
    # View looking at the left wing & nacelle from above/aft
    # Spike is at X ~ 1.0, nacelle at Z ~ -4.1 or +4.1
    ax.set_xlim(0, 15); ax.set_ylim(-6, 6); ax.set_zlim(-1, 3)
    ax.view_init(elev=35, azim=-60)
    ax.set_title(f"{f}\nmin={val.min():.2f}, max={val.max():.2f}", color="white", fontsize=11, fontweight="bold")
    ax.axis('off')

plt.tight_layout()
out = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/wing_anomaly_fields.png"
plt.savefig(out, dpi=150, facecolor="#222222")
print("Saved to", out)
