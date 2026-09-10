import time
from pathlib import Path
import trimesh
import meshio
import numpy as np
from scipy.spatial import cKDTree
from pyevtk.hl import imageToVTK

out_dir = Path("output/sr71_cinematic")
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 76)
print("GENERATING PREMIUM HIGH-QUALITY VTK EXPORTS FOR PARAVIEW")
print("=" * 76)

# 1. Load volume simulation solution
print("Loading simulation volume solution...")
t0 = time.perf_counter()
m_vol = meshio.read(out_dir / "sr71_volume.vtu")
pts_vol = m_vol.points
p_vol = m_vol.point_data["pressure"]
rho_vol = m_vol.point_data["density"]
mach_vol = m_vol.point_data["mach"]
cp_vol = m_vol.point_data["cp"]
vel_vol = m_vol.point_data["velocity"]
vort_vol = m_vol.point_data["vorticity"]
vort_mag_vol = m_vol.point_data["vorticity_magnitude"]
q_vol = m_vol.point_data["Q_criterion"]

tree = cKDTree(pts_vol)
print(f"Loaded {len(pts_vol)} simulation nodes in {time.perf_counter() - t0:.2f}s")

# -----------------------------------------------------------------------------
# A. SMOOTH CAD SURFACE VTU (sr71_surface_smooth.vtu)
# -----------------------------------------------------------------------------
print("\nA. Building smooth, high-fidelity CAD airframe skin...")
cad = trimesh.load("mesh/sr71_perfect_clean.stl")
cad_sub = cad.subdivide()  # Subdivide to 21,120 smooth triangles

dists, idx = tree.query(cad_sub.vertices, k=4)
weights = 1.0 / np.maximum(dists, 1e-6)
weights /= np.sum(weights, axis=1, keepdims=True)

cad_p = np.sum(p_vol[idx] * weights, axis=1)
cad_rho = np.sum(rho_vol[idx] * weights, axis=1)
cad_mach = np.sum(mach_vol[idx] * weights, axis=1)
cad_cp = np.sum(cp_vol[idx] * weights, axis=1)
cad_vel = np.sum(vel_vol[idx] * weights[:, :, None], axis=1)
cad_vort = np.sum(vort_vol[idx] * weights[:, :, None], axis=1)
cad_vort_mag = np.sum(vort_mag_vol[idx] * weights, axis=1)
cad_q = np.sum(q_vol[idx] * weights, axis=1)

surf_smooth = meshio.Mesh(
    points=cad_sub.vertices,
    cells=[("triangle", cad_sub.faces)],
    point_data={
        "pressure": cad_p.astype(np.float32),
        "density": cad_rho.astype(np.float32),
        "mach": cad_mach.astype(np.float32),
        "cp": cad_cp.astype(np.float32),
        "velocity": cad_vel.astype(np.float32),
        "vorticity": cad_vort.astype(np.float32),
        "vorticity_magnitude": cad_vort_mag.astype(np.float32),
        "Q_criterion": cad_q.astype(np.float32),
        "normals": cad_sub.vertex_normals.astype(np.float32),
    }
)
surf_path = out_dir / "sr71_surface_smooth.vtu"
surf_smooth.write(str(surf_path))
print(f"  -> Exported {surf_path} ({len(cad_sub.faces)} triangles, {len(cad_sub.vertices)} smooth vertices)")

# -----------------------------------------------------------------------------
# B. HIGH-RESOLUTION MULTI-PLANE FLOW CUTS (sr71_flow_slices.vtu)
# -----------------------------------------------------------------------------
print("\nB. Building high-resolution planar flow slices...")
slice_points = []
slice_cells = []
slice_p = []
slice_mach = []
slice_cp = []
slice_vel = []
slice_vort_mag = []
slice_q = []
node_offset = 0

def add_plane_grid(origin, u_vec, v_vec, nu, nv):
    global node_offset
    u_vals = np.linspace(0.0, 1.0, nu)
    v_vals = np.linspace(0.0, 1.0, nv)
    UU, VV = np.meshgrid(u_vals, v_vals)
    plane_pts = (
        origin[None, None, :]
        + UU[:, :, None] * u_vec[None, None, :]
        + VV[:, :, None] * v_vec[None, None, :]
    ).reshape(-1, 3)

    # Grid quad connectivity
    quads = []
    for j in range(nv - 1):
        for i in range(nu - 1):
            n0 = node_offset + j * nu + i
            n1 = node_offset + j * nu + (i + 1)
            n2 = node_offset + (j + 1) * nu + (i + 1)
            n3 = node_offset + (j + 1) * nu + i
            quads.append([n0, n1, n2, n3])

    dists, idx = tree.query(plane_pts, k=4)
    w = 1.0 / np.maximum(dists, 1e-6)
    w /= np.sum(w, axis=1, keepdims=True)

    slice_points.append(plane_pts)
    slice_cells.extend(quads)
    slice_p.append(np.sum(p_vol[idx] * w, axis=1))
    slice_mach.append(np.sum(mach_vol[idx] * w, axis=1))
    slice_cp.append(np.sum(cp_vol[idx] * w, axis=1))
    slice_vel.append(np.sum(vel_vol[idx] * w[:, :, None], axis=1))
    slice_vort_mag.append(np.sum(vort_mag_vol[idx] * w, axis=1))
    slice_q.append(np.sum(q_vol[idx] * w, axis=1))
    node_offset += len(plane_pts)

# 1. Longitudinal Centerline Slice (Z = 0)
add_plane_grid(
    origin=np.array([-20.0, -3.5, 0.0]),
    u_vec=np.array([42.0, 0.0, 0.0]),
    v_vec=np.array([0.0, 8.0, 0.0]),
    nu=350, nv=120
)

# 2. Chine Vortex Cross-Flow Slices (X = -4, 0, 6, 12m)
for x_cut in [-4.0, 0.0, 6.0, 12.0]:
    add_plane_grid(
        origin=np.array([x_cut, -2.5, -8.0]),
        u_vec=np.array([0.0, 0.0, 16.0]),
        v_vec=np.array([0.0, 6.0, 0.0]),
        nu=160, nv=90
    )

# 3. Horizontal Wing Planform Slice (Y = 0.2m)
add_plane_grid(
    origin=np.array([-18.0, 0.2, -8.5]),
    u_vec=np.array([36.0, 0.0, 0.0]),
    v_vec=np.array([0.0, 0.0, 17.0]),
    nu=300, nv=150
)

slices_mesh = meshio.Mesh(
    points=np.vstack(slice_points),
    cells=[("quad", np.array(slice_cells))],
    point_data={
        "pressure": np.concatenate(slice_p).astype(np.float32),
        "mach": np.concatenate(slice_mach).astype(np.float32),
        "cp": np.concatenate(slice_cp).astype(np.float32),
        "velocity": np.vstack(slice_vel).astype(np.float32),
        "vorticity_magnitude": np.concatenate(slice_vort_mag).astype(np.float32),
        "Q_criterion": np.concatenate(slice_q).astype(np.float32),
    }
)
slices_path = out_dir / "sr71_flow_slices.vtu"
slices_mesh.write(str(slices_path))
print(f"  -> Exported {slices_path} ({len(slice_cells)} quads, {node_offset} slice nodes)")

# -----------------------------------------------------------------------------
# C. UNIFORM RECTILINEAR 3D VOLUME GRID (sr71_flow_volume.vti)
# -----------------------------------------------------------------------------
print("\nC. Building uniform 3D Cartesian volume grid for ray-casting...")
nx, ny, nz = 160, 60, 90  # 864,000 Cartesian voxels
gx = np.linspace(-18.0, 22.0, nx, dtype=np.float32)
gy = np.linspace(-3.0, 4.0, ny, dtype=np.float32)
gz = np.linspace(-9.0, 9.0, nz, dtype=np.float32)

dx = float(gx[1] - gx[0])
dy = float(gy[1] - gy[0])
dz = float(gz[1] - gz[0])

# Coordinate mesh
X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
grid_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

dists, idx = tree.query(grid_pts, k=4)
w = 1.0 / np.maximum(dists, 1e-6)
w /= np.sum(w, axis=1, keepdims=True)

grid_p = np.sum(p_vol[idx] * w, axis=1).reshape((nx, ny, nz))
grid_mach = np.sum(mach_vol[idx] * w, axis=1).reshape((nx, ny, nz))
grid_cp = np.sum(cp_vol[idx] * w, axis=1).reshape((nx, ny, nz))
grid_vort_mag = np.sum(vort_mag_vol[idx] * w, axis=1).reshape((nx, ny, nz))
grid_q = np.sum(q_vol[idx] * w, axis=1).reshape((nx, ny, nz))

vx = np.sum(vel_vol[idx, 0] * w, axis=1).reshape((nx, ny, nz))
vy = np.sum(vel_vol[idx, 1] * w, axis=1).reshape((nx, ny, nz))
vz = np.sum(vel_vol[idx, 2] * w, axis=1).reshape((nx, ny, nz))

vti_path = str(out_dir / "sr71_flow_volume")
imageToVTK(
    vti_path,
    origin=(-18.0, -3.0, -9.0),
    spacing=(dx, dy, dz),
    pointData={
        "pressure": grid_p,
        "mach": grid_mach,
        "cp": grid_cp,
        "vorticity_magnitude": grid_vort_mag,
        "Q_criterion": grid_q,
        "velocity": (vx, vy, vz),
    }
)
print(f"  -> Exported {vti_path}.vti ({nx}x{ny}x{nz} = {nx*ny*nz:,} voxels)")
print("\nAll premium VTK datasets exported successfully!")
