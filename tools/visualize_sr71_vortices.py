import meshio
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load volume mesh
m = meshio.read("output/sr71_cinematic/sr71_volume.vtu")
pts = m.points
hex_cells = m.cells_dict["hexahedron"]
vort = m.point_data["vorticity_magnitude"]
q_crit = m.point_data["Q_criterion"]
mach = m.point_data["mach"]
p = m.point_data["pressure"]

fig = plt.figure(figsize=(18, 12), facecolor="#0a0f1d")

# Grid layout: 2x2
# Subplot 1: Cross-flow slice at X = -3.0m (Mid-chine vortex formation)
ax1 = fig.add_subplot(2, 2, 1, facecolor="#0e172a")
x_cut1 = -3.0
dx = 0.8
mask_x1 = np.abs(pts[:, 0] - x_cut1) < dx
pts_x1 = pts[mask_x1]
vort_x1 = vort[mask_x1]

zy_grid_z = np.linspace(-6.0, 6.0, 300)
zy_grid_y = np.linspace(-2.0, 2.5, 150)
ZZ, YY = np.meshgrid(zy_grid_z, zy_grid_y)

vort_grid1 = griddata((pts_x1[:, 2], pts_x1[:, 1]), vort_x1, (ZZ, YY), method="linear", fill_value=0.0)
im1 = ax1.imshow(vort_grid1, origin="lower", extent=[-6.0, 6.0, -2.0, 2.5], cmap="inferno", aspect="equal", vmin=0, vmax=20.0)
cb1 = plt.colorbar(im1, ax=ax1, fraction=0.03, pad=0.04)
cb1.set_label("Vorticity Magnitude (1/s)", color="white", fontsize=10)
cb1.ax.tick_params(colors="white")
ax1.set_title(f"Cross-Flow Vorticity at X = {x_cut1:.1f} m (Forward Chine Separation)", color="white", fontsize=12, fontweight="bold")
ax1.set_xlabel("Spanwise Z (m)", color="white")
ax1.set_ylabel("Vertical Y (m)", color="white")
ax1.tick_params(colors="white")
ax1.grid(color="#ffffff", alpha=0.1, linestyle="--")

# Subplot 2: Cross-flow slice at X = +8.0m (Wing / Nacelles / Chine Vortex Stream)
ax2 = fig.add_subplot(2, 2, 2, facecolor="#0e172a")
x_cut2 = 8.0
mask_x2 = np.abs(pts[:, 0] - x_cut2) < dx
pts_x2 = pts[mask_x2]
vort_x2 = vort[mask_x2]
vort_grid2 = griddata((pts_x2[:, 2], pts_x2[:, 1]), vort_x2, (ZZ, YY), method="linear", fill_value=0.0)
im2 = ax2.imshow(vort_grid2, origin="lower", extent=[-6.0, 6.0, -2.0, 2.5], cmap="inferno", aspect="equal", vmin=0, vmax=30.0)
cb2 = plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.04)
cb2.set_label("Vorticity Magnitude (1/s)", color="white", fontsize=10)
cb2.ax.tick_params(colors="white")
ax2.set_title(f"Cross-Flow Vorticity at X = +{x_cut2:.1f} m (Wing Root, Nacelles & Chines)", color="white", fontsize=12, fontweight="bold")
ax2.set_xlabel("Spanwise Z (m)", color="white")
ax2.set_ylabel("Vertical Y (m)", color="white")
ax2.tick_params(colors="white")
ax2.grid(color="#ffffff", alpha=0.1, linestyle="--")

# Subplot 3: Planform Top-Down View of Mach Field at Y = 0.2m
ax3 = fig.add_subplot(2, 2, 3, facecolor="#0e172a")
y_cut = 0.2
dy = 0.5
mask_y = np.abs(pts[:, 1] - y_cut) < dy
pts_y = pts[mask_y]
mach_y = mach[mask_y]

xz_grid_x = np.linspace(-18.0, 20.0, 400)
xz_grid_z = np.linspace(-9.0, 9.0, 200)
XX, ZZ_top = np.meshgrid(xz_grid_x, xz_grid_z)

mach_grid = griddata((pts_y[:, 0], pts_y[:, 2]), mach_y, (XX, ZZ_top), method="linear", fill_value=2.4)
im3 = ax3.imshow(mach_grid, origin="lower", extent=[-18.0, 20.0, -9.0, 9.0], cmap="coolwarm", aspect="equal", vmin=1.4, vmax=2.8)
cb3 = plt.colorbar(im3, ax=ax3, fraction=0.02, pad=0.04)
cb3.set_label("Local Mach Number", color="white", fontsize=10)
cb3.ax.tick_params(colors="white")
ax3.set_title("Planform Mach Field (Y = 0.2 m, Wing & Chine Top Surface)", color="white", fontsize=12, fontweight="bold")
ax3.set_xlabel("Longitudinal X (m)", color="white")
ax3.set_ylabel("Spanwise Z (m)", color="white")
ax3.tick_params(colors="white")
ax3.grid(color="#ffffff", alpha=0.1, linestyle="--")

# Subplot 4: Full Surface Pressure Distribution (Compression & Expansion)
ax4 = fig.add_subplot(2, 2, 4, facecolor="#0e172a")
ms = meshio.read("output/sr71_cinematic/sr71_surface.vtu")
s_pts = ms.points
s_p = ms.point_data["pressure"]

# Centerline points (|Z| < 0.25m)
mask_cl = np.abs(s_pts[:, 2]) < 0.25
cl_x = s_pts[mask_cl, 0]
cl_y = s_pts[mask_cl, 1]
cl_p = s_p[mask_cl]
cl_upper = cl_y > -0.6
cl_lower = cl_y <= -0.6

ax4.scatter(cl_x[cl_upper], cl_p[cl_upper], color="#00f2fe", s=14, label="Upper Spine Pressure (Expansion)", alpha=0.8)
ax4.scatter(cl_x[cl_lower], cl_p[cl_lower], color="#ff6b6b", s=14, label="Lower Belly Keel Pressure (Compression/Lift)", alpha=0.8)
ax4.axhline(1.0, color="#ffffff", linestyle=":", alpha=0.4, label="Freestream p_inf (1.0)")
ax4.set_title("SR-71 Surface Pressure Distribution Along Centerline", color="white", fontsize=12, fontweight="bold")
ax4.set_xlabel("Longitudinal Station X (m)", color="white")
ax4.set_ylabel("Static Pressure Ratio (p / p_inf)", color="white")
ax4.set_xlim(-18.0, 18.0)
ax4.set_ylim(0.5, 1.8)
ax4.tick_params(colors="white")
ax4.grid(color="#ffffff", alpha=0.1, linestyle="--")
ax4.legend(loc="upper right", facecolor="#0e172a", edgecolor="#00a3d9", labelcolor="white")

plt.tight_layout()
out_fig = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/sr71_vorticity_flow_features.png"
plt.savefig(out_fig, dpi=160, facecolor="#0a0f1d")
print("Saved flow features plot to", out_fig)
