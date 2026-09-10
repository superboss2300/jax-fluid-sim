import meshio
import numpy as np
import matplotlib.pyplot as plt

m = meshio.read("output/sr71_cinematic/sr71_flow_slices.vtu")
pts = m.points
quads = m.cells_dict["quad"]
mach = m.point_data["mach"]
p = m.point_data["pressure"]
q = m.point_data["Q_criterion"]

fig = plt.figure(figsize=(20, 10), facecolor="#0a0f1d")

# Subplot 1: Centerline slice (Z = 0) Mach field
ax1 = fig.add_subplot(1, 2, 1, facecolor="#0e172a")
# Filter Z = 0 slice
mask_z0 = np.abs(pts[:, 2]) < 0.05
pts_z0 = pts[mask_z0]
mach_z0 = mach[mask_z0]

from scipy.interpolate import griddata
gx = np.linspace(-20, 22, 350)
gy = np.linspace(-3.5, 4.5, 120)
GX, GY = np.meshgrid(gx, gy)
grid_m = griddata((pts_z0[:, 0], pts_z0[:, 1]), mach_z0, (GX, GY), method="linear")

im1 = ax1.imshow(grid_m, origin="lower", extent=[-20, 22, -3.5, 4.5], cmap="turbo", aspect="equal", vmin=1.2, vmax=2.6)
ax1.set_title("Longitudinal Centerline (Z=0) Slice: Mach Number Contours", color="white", fontsize=12, fontweight="bold")
ax1.set_xlabel("X (m)", color="white")
ax1.set_ylabel("Y (m)", color="white")
ax1.tick_params(colors="white")
cb1 = plt.colorbar(im1, ax=ax1, fraction=0.03, pad=0.04)
cb1.set_label("Mach Number", color="white")
cb1.ax.tick_params(colors="white")

# Subplot 2: Horizontal Wing Planform Slice (Y = 0.2m)
ax2 = fig.add_subplot(1, 2, 2, facecolor="#0e172a")
mask_y0 = np.abs(pts[:, 1] - 0.2) < 0.05
pts_y0 = pts[mask_y0]
mach_y0 = mach[mask_y0]

gz = np.linspace(-8.5, 8.5, 150)
GX_w, GZ_w = np.meshgrid(gx, gz)
grid_mw = griddata((pts_y0[:, 0], pts_y0[:, 2]), mach_y0, (GX_w, GZ_w), method="linear")

im2 = ax2.imshow(grid_mw, origin="lower", extent=[-20, 22, -8.5, 8.5], cmap="turbo", aspect="equal", vmin=1.4, vmax=2.6)
ax2.set_title("Horizontal Wing Planform Slice (Y=0.2m): Chine & Wing Shocks", color="white", fontsize=12, fontweight="bold")
ax2.set_xlabel("X (m)", color="white")
ax2.set_ylabel("Z (m)", color="white")
ax2.tick_params(colors="white")
cb2 = plt.colorbar(im2, ax=ax2, fraction=0.02, pad=0.04)
cb2.set_label("Mach Number", color="white")
cb2.ax.tick_params(colors="white")

plt.tight_layout()
out_path = "/home/unnik/.gemini/antigravity/brain/79dbcdf7-3a5e-4fb5-9755-a9daf0901abb/paraview_slices_inspection.png"
plt.savefig(out_path, dpi=160, facecolor="#0a0f1d")
print("Saved slices inspection to", out_path)
