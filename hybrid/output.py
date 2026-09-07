"""Visualization output for discontinuous P2 hybrid fields."""

from pathlib import Path

import meshio
import numpy as np
from scipy.interpolate import griddata


def shock_colormap():
    """Dark high-contrast palette suited to schlieren and CFD gradients."""
    from matplotlib.colors import LinearSegmentedColormap

    color_map = LinearSegmentedColormap.from_list(
        "shock_dark",
        [
            (0.00, "#01051f"), (0.16, "#031448"),
            (0.34, "#07549a"), (0.52, "#00a9cf"),
            (0.68, "#20d49a"), (0.82, "#d9ef38"),
            (0.93, "#ff8b1e"), (1.00, "#ff2915"),
        ],
    )
    color_map.set_bad("#01051f")
    return color_map


def _robust_limits(values, lower=0.5, upper=99.5):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return 0.0, 1.0
    low, high = np.percentile(finite, [lower, upper])
    if high <= low:
        padding = max(abs(float(low)) * 0.05, 1.0e-6)
        low, high = low - padding, high + padding
    return float(low), float(high)


def _field_limits(values, minimum_relative_span):
    """Do not turn roundoff in a nearly uniform field into full color range."""
    low, high = _robust_limits(values)
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return low, high
    center = float(np.median(finite))
    scale = max(abs(center), 1.0)
    minimum_span = minimum_relative_span * scale
    if high - low < minimum_span:
        return center - 0.5 * minimum_span, center + 0.5 * minimum_span
    return low, high


_P1_SUBCELLS = {
    "tetra": ("tetra", np.asarray([[0, 1, 2, 3]], dtype=np.int32)),
    "wedge": ("wedge", np.asarray([[0, 1, 2, 3, 4, 5]], dtype=np.int32)),
    "hexahedron": ("hexahedron", np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)),
}

_P2_SUBCELLS = {
    "tetra": (
        "tetra",
        np.asarray([
            [0, 4, 5, 6], [4, 1, 7, 8], [5, 7, 2, 9], [6, 8, 9, 3],
            [4, 5, 6, 8], [4, 7, 5, 8], [5, 6, 8, 9], [5, 8, 7, 9],
        ], dtype=np.int32),
    ),
    "wedge": (
        "wedge",
        np.asarray([
            [0, 3, 5, 12, 15, 17], [3, 1, 4, 15, 13, 16],
            [5, 4, 2, 17, 16, 14], [3, 4, 5, 15, 16, 17],
            [12, 15, 17, 6, 9, 11], [15, 13, 16, 9, 7, 10],
            [17, 16, 14, 11, 10, 8], [15, 16, 17, 9, 10, 11],
        ], dtype=np.int32),
    ),
}


def _hex_p2_subcells():
    slot = (0, 2, 1)  # internal Q2 coordinate slots are (0, 1, 1/2)
    index = lambda ix, iy, iz: iz * 9 + iy * 3 + ix
    cells = []
    for kz in range(2):
        for jy in range(2):
            for ix in range(2):
                x0, x1 = slot[ix], slot[ix + 1]
                y0, y1 = slot[jy], slot[jy + 1]
                z0, z1 = slot[kz], slot[kz + 1]
                cells.append([
                    index(x0, y0, z0), index(x1, y0, z0),
                    index(x1, y1, z0), index(x0, y1, z0),
                    index(x0, y0, z1), index(x1, y0, z1),
                    index(x1, y1, z1), index(x0, y1, z1),
                ])
    return "hexahedron", np.asarray(cells, dtype=np.int32)


_P2_SUBCELLS["hexahedron"] = _hex_p2_subcells()

# P=3 subcells (27 subcells per element)
_TETRA_P3_SUBCELLS = np.asarray([
    [10, 11, 16, 19], [14, 7, 2, 11], [14, 15, 18, 19], [13, 12, 17, 19],
    [5, 10, 12, 1], [9, 13, 15, 3], [8, 17, 16, 19], [8, 18, 19, 16],
    [8, 4, 16, 17], [8, 4, 0, 16], [8, 6, 18, 16], [8, 6, 16, 0],
    [14, 18, 16, 19], [14, 6, 11, 16], [14, 11, 19, 16], [14, 6, 16, 18],
    [14, 7, 11, 6], [5, 17, 19, 16], [5, 10, 19, 12], [5, 12, 19, 17],
    [5, 4, 17, 16], [5, 10, 16, 19], [9, 18, 19, 17], [9, 13, 17, 19],
    [9, 15, 19, 18], [9, 8, 18, 17], [9, 13, 19, 15],
], dtype=np.int32)

_WEDGE_P3_SUBCELLS = np.asarray([
    [0, 3, 8, 20, 23, 28], [4, 1, 5, 24, 21, 25], [7, 6, 2, 27, 26, 22],
    [3, 4, 9, 23, 24, 29], [5, 6, 9, 25, 26, 29], [8, 9, 7, 28, 29, 27],
    [3, 9, 8, 23, 29, 28], [4, 5, 9, 24, 25, 29], [6, 7, 9, 26, 27, 29],
    [20, 23, 28, 30, 33, 38], [24, 21, 25, 34, 31, 35], [27, 26, 22, 37, 36, 32],
    [23, 24, 29, 33, 34, 39], [25, 26, 29, 35, 36, 39], [28, 29, 27, 38, 39, 37],
    [23, 29, 28, 33, 39, 38], [24, 25, 29, 34, 35, 39], [26, 27, 29, 36, 37, 39],
    [30, 33, 38, 10, 13, 18], [34, 31, 35, 14, 11, 15], [37, 36, 32, 17, 16, 12],
    [33, 34, 39, 13, 14, 19], [35, 36, 39, 15, 16, 19], [38, 39, 37, 18, 19, 17],
    [33, 39, 38, 13, 19, 18], [34, 35, 39, 14, 15, 19], [36, 37, 39, 16, 17, 19],
], dtype=np.int32)


def _hex_p3_subcells():
    slot = (0, 2, 3, 1)  # internal Q3 line order [0, 1, 1/3, 2/3] -> spatial (0, 1/3, 2/3, 1)
    index = lambda ix, iy, iz: iz * 16 + iy * 4 + ix
    cells = []
    for kz in range(3):
        for jy in range(3):
            for ix in range(3):
                x0, x1 = slot[ix], slot[ix + 1]
                y0, y1 = slot[jy], slot[jy + 1]
                z0, z1 = slot[kz], slot[kz + 1]
                cells.append([
                    index(x0, y0, z0), index(x1, y0, z0),
                    index(x1, y1, z0), index(x0, y1, z0),
                    index(x0, y0, z1), index(x1, y0, z1),
                    index(x1, y1, z1), index(x0, y1, z1),
                ])
    return "hexahedron", np.asarray(cells, dtype=np.int32)


_P3_SUBCELLS = {
    "tetra": ("tetra", _TETRA_P3_SUBCELLS),
    "wedge": ("wedge", _WEDGE_P3_SUBCELLS),
    "hexahedron": _hex_p3_subcells(),
}


def get_element_subcells(element_type, polynomial_degree):
    """Return (vtk_cell_type, subcell_indices) for a given element and degree."""
    if polynomial_degree == 1:
        return _P1_SUBCELLS[element_type]
    elif polynomial_degree == 2:
        return _P2_SUBCELLS[element_type]
    elif polynomial_degree == 3:
        return _P3_SUBCELLS[element_type]
    raise ValueError(
        f"Unsupported polynomial degree {polynomial_degree} for element {element_type!r}"
    )


_LINEAR_CELL_EDGES = {
    "tetra": ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    "wedge": (
        (0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5),
    ),
    "hexahedron": (
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ),
}


def _midplane_triangles(operator, states, plane_z=0.0):
    """Intersect visualization subcells with z=plane_z without slab projection."""
    vertices = []
    values = []
    triangles = []
    tolerance = 1.0e-10

    for block in operator.mesh.element_blocks:
        element_type = block.element_type
        coordinates = np.asarray(
            operator.geometry.solution_node_coordinates[element_type]
        )
        primitive = np.asarray(
            operator.solver.conserved_to_primitive(states[element_type])
        )
        output_type, subcells = get_element_subcells(
            element_type, operator.solver.polynomial_degree
        )
        edges = _LINEAR_CELL_EDGES[output_type]

        parent_crosses = (
            (np.min(coordinates[..., 2], axis=1) <= plane_z + tolerance)
            & (np.max(coordinates[..., 2], axis=1) >= plane_z - tolerance)
        )
        for element in np.flatnonzero(parent_crosses):
            for local_cell in subcells:
                xyz = coordinates[element, local_cell]
                state = primitive[element, local_cell]
                signed_z = xyz[:, 2] - plane_z
                if np.min(signed_z) > tolerance or np.max(signed_z) < -tolerance:
                    continue

                intersections = {}

                def insert(point, value):
                    key = tuple(np.round(point[:2], 12))
                    intersections.setdefault(key, (point[:2], value))

                for local_index in np.flatnonzero(np.abs(signed_z) <= tolerance):
                    insert(xyz[local_index], state[local_index])
                for first, second in edges:
                    z_first, z_second = signed_z[first], signed_z[second]
                    if z_first * z_second < -(tolerance * tolerance):
                        fraction = -z_first / (z_second - z_first)
                        insert(
                            xyz[first] + fraction * (xyz[second] - xyz[first]),
                            state[first] + fraction * (state[second] - state[first]),
                        )

                polygon = list(intersections.values())
                if len(polygon) < 3:
                    continue
                xy = np.asarray([item[0] for item in polygon])
                primitive_values = np.asarray([item[1] for item in polygon])
                center = np.mean(xy, axis=0)
                order = np.argsort(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]))
                xy = xy[order]
                primitive_values = primitive_values[order]
                offset = len(vertices)
                vertices.extend(xy)
                values.extend(primitive_values)
                for index in range(1, len(xy) - 1):
                    triangles.append((offset, offset + index, offset + index + 1))

    if not triangles:
        raise ValueError(f"The mesh does not intersect z={plane_z}")
    return (
        np.asarray(vertices),
        np.asarray(triangles, dtype=np.int32),
        np.asarray(values),
    )


def write_tessellated_vtu(path, operator, states):
    """Write each P1/P2/P3 parent element as conforming linear visualization subcells."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points, primitives, cells = [], [], []
    point_offset = 0
    p_degree = operator.solver.polynomial_degree

    for block in operator.mesh.element_blocks:
        element_type = block.element_type
        coordinates = np.asarray(
            operator.geometry.solution_node_coordinates[element_type]
        )
        W = np.asarray(operator.solver.conserved_to_primitive(states[element_type]))
        points.append(coordinates.reshape(-1, 3))
        primitives.append(W.reshape(-1, W.shape[-1]))
        nodes_per_element = coordinates.shape[1]

        output_type, local_subcells = get_element_subcells(element_type, p_degree)

        element_offsets = (
            point_offset
            + np.arange(len(coordinates), dtype=np.int32) * nodes_per_element
        )
        connectivity = (
            element_offsets[:, None, None] + local_subcells[None, :, :]
        ).reshape(-1, local_subcells.shape[1])
        cells.append((output_type, connectivity))
        point_offset += coordinates.size // 3

    points = np.concatenate(points, axis=0)
    W = np.concatenate(primitives, axis=0)
    sound_speed = np.sqrt(
        np.maximum(operator.solver.gamma * W[:, 4] / np.maximum(W[:, 0], 1.0e-12), 0.0)
    )
    speed = np.linalg.norm(W[:, 1:4], axis=-1)
    point_data = {
        "density": W[:, 0],
        "velocity": W[:, 1:4],
        "pressure": W[:, 4],
        "speed": speed,
        "mach": speed / np.maximum(sound_speed, 1.0e-30),
    }
    if operator.solver.layout.species_names:
        for s_idx, s_name in enumerate(operator.solver.layout.species_names):
            clean_name = s_name.replace("+", "_plus").replace("-", "_minus")
            point_data[f"Y_{clean_name}"] = W[:, 5 + s_idx]
    meshio.write(path, meshio.Mesh(points, cells, point_data=point_data))
    return path


def _resample_to_cartesian_grid(
    unique_xy, unique_fields, extent, resolution, sphere_radius=0.5, sphere_center=(0.0, 0.0)
):
    """Resample unstructured slice data onto a uniform grid with boundary inpainting."""
    xmin, xmax, ymin, ymax = extent
    nx, ny = resolution
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    GX, GY = np.meshgrid(gx, gy)

    r_sq = (GX - sphere_center[0]) ** 2 + (GY - sphere_center[1]) ** 2
    sphere_mask = r_sq < (sphere_radius + 1.0e-4) ** 2

    num_vars = unique_fields.shape[-1]
    resampled = np.zeros((ny, nx, num_vars), dtype=float)
    for v in range(num_vars):
        # 1. High-quality linear interpolation inside mesh convex hull
        val = griddata(unique_xy, unique_fields[:, v], (GX, GY), method="linear")

        # 2. Inpaint boundary gap pixels (between mesh boundary chords and physical circle)
        nan_fluid = np.isnan(val) & ~sphere_mask
        if np.any(nan_fluid):
            val_near = griddata(
                unique_xy, unique_fields[:, v], (GX[nan_fluid], GY[nan_fluid]), method="nearest"
            )
            val[nan_fluid] = val_near

        val[sphere_mask] = np.nan
        resampled[:, :, v] = val

    return gx, gy, GX, GY, resampled, sphere_mask


def compute_schlieren(grid_rho, dx, dy, sphere_mask, noise_floor=0.02, contrast=15.0):
    """Compute physical, noise-gated numerical Schlieren log10(1 + k*|grad rho|/rho_ref)."""
    med_rho = max(abs(float(np.nanmedian(grid_rho))), 1.0)
    rho_clean = np.nan_to_num(grid_rho, nan=med_rho)
    grad_y, grad_x = np.gradient(rho_clean, dy, dx)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag[sphere_mask] = np.nan

    valid = grad_mag[~sphere_mask]
    finite_valid = valid[np.isfinite(valid)]
    max_g = float(np.max(finite_valid)) if len(finite_valid) else 0.0

    if max_g < noise_floor:
        # Uniform or early startup: zero out so colormap doesn't stretch micro noise
        return np.zeros_like(grad_mag)

    return np.log10(1.0 + contrast * grad_mag / med_rho)


def save_midplane_heatmap(path, operator, states, title="Hybrid P2 DG", resolution=(900, 450)):
    """Render an exact, continuous z=0 cut resampled to a Cartesian pixel grid."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xy, _, W = _midplane_triangles(operator, states)

    # Deduplicate coincident slice points and average coincident DG traces
    rounded = np.round(xy, decimals=5)
    unique_xy, inverse = np.unique(rounded, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    unique_W = np.zeros((len(unique_xy), W.shape[1]), dtype=W.dtype)
    for var in range(W.shape[1]):
        unique_W[:, var] = np.bincount(inverse, weights=W[:, var]) / counts

    xmin, xmax = float(np.min(unique_xy[:, 0])), float(np.max(unique_xy[:, 0]))
    ymin, ymax = float(np.min(unique_xy[:, 1])), float(np.max(unique_xy[:, 1]))
    extent = (xmin, xmax, ymin, ymax)

    gx, gy, GX, GY, grid_W, sphere_mask = _resample_to_cartesian_grid(
        unique_xy, unique_W, extent, resolution
    )

    speed = np.linalg.norm(grid_W[:, :, 1:4], axis=-1)
    sound = np.sqrt(
        np.maximum(
            operator.solver.gamma * grid_W[:, :, 4]
            / np.maximum(grid_W[:, :, 0], 1.0e-12),
            0.0,
        )
    )
    mach = speed / np.maximum(sound, 1.0e-30)

    dx = float(gx[1] - gx[0])
    dy = float(gy[1] - gy[0])
    schlieren = compute_schlieren(grid_W[:, :, 0], dx, dy, sphere_mask)

    fields = [grid_W[:, :, 0], grid_W[:, :, 4], mach, schlieren]
    names = (
        "Density",
        "Pressure",
        "Mach number",
        r"Schlieren $\log_{10}(1+|\nabla\rho|)$",
    )

    figure, axes = plt.subplots(
        2, 2, figsize=(15, 9), constrained_layout=True,
        facecolor="#01051f",
    )
    color_map = shock_colormap()

    for field_index, (axis, name) in enumerate(zip(axes.flat, names)):
        field = fields[field_index]
        valid_field = field[~sphere_mask]
        if field_index < 3:
            minimum_spans = (0.02, 0.05, 0.02)
            vmin, vmax = _field_limits(valid_field, minimum_spans[field_index])
        else:
            finite = valid_field[np.isfinite(valid_field)]
            if len(finite) and np.max(finite) > 0.0:
                vmin = float(np.percentile(finite, 2))
                vmax = float(np.percentile(finite, 99.5))
                if vmax <= vmin:
                    vmax = vmin + 1.0
            else:
                vmin, vmax = 0.0, 1.0

        shown = axis.imshow(
            np.ma.masked_invalid(field),
            origin="lower",
            extent=extent,
            cmap=color_map,
            interpolation="bilinear",
            vmin=vmin,
            vmax=vmax,
        )
        # Draw physical sphere boundary
        sphere = plt.Circle(
            (0.0, 0.0), 0.5,
            facecolor="#01051f",
            edgecolor="#4cc9f0",
            linewidth=1.5,
            zorder=5,
        )
        axis.add_patch(sphere)

        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
        axis.set_aspect("equal")
        axis.set_facecolor("#01051f")
        axis.set_title(name, color="white")
        axis.set_xlabel("x", color="white")
        axis.set_ylabel("y", color="white")
        axis.tick_params(colors="white")
        for spine in axis.spines.values():
            spine.set_color("#4cc9f0")
        colorbar = figure.colorbar(shown, ax=axis, shrink=0.84)
        colorbar.ax.tick_params(colors="white")
        colorbar.outline.set_edgecolor("#4cc9f0")

    figure.suptitle(title, color="white")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def save_cinematic_schlieren(
    path,
    operator,
    states,
    resolution=(1920, 1080),
    title="Compressible Flow Past Sphere",
    sim_time=None,
    step=None,
    mach=None,
    reynolds=None,
    plane_z=0.0,
    sphere_radius=0.5,
    sphere_center=(0.0, 0.0),
):
    """Render a full-bleed 1080p cinematic numerical Schlieren image."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xy, _, W = _midplane_triangles(operator, states, plane_z=plane_z)

    # Deduplicate coincident slice points and average coincident DG traces
    rounded = np.round(xy, decimals=5)
    unique_xy, inverse = np.unique(rounded, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    unique_rho = np.bincount(inverse, weights=W[:, 0]) / counts

    xmin, xmax = float(np.min(unique_xy[:, 0])), float(np.max(unique_xy[:, 0]))
    ymin, ymax = float(np.min(unique_xy[:, 1])), float(np.max(unique_xy[:, 1]))
    extent = (xmin, xmax, ymin, ymax)

    gx, gy, GX, GY, resampled, sphere_mask = _resample_to_cartesian_grid(
        unique_xy, unique_rho[:, None], extent, resolution, sphere_radius, sphere_center
    )
    grid_rho = resampled[:, :, 0]

    dx = float(gx[1] - gx[0])
    dy = float(gy[1] - gy[0])
    schlieren = compute_schlieren(grid_rho, dx, dy, sphere_mask)

    # Color limits
    valid = schlieren[~sphere_mask]
    finite_valid = valid[np.isfinite(valid)]
    if len(finite_valid) and np.max(finite_valid) > 0.0:
        s_min = float(np.percentile(finite_valid, 2))
        s_max = float(np.percentile(finite_valid, 99.5))
        if s_max <= s_min:
            s_max = s_min + 1.0
    else:
        s_min, s_max = 0.0, 1.0

    fig = plt.figure(figsize=(16, 9), facecolor="#01051f", dpi=120)
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#01051f")

    color_map = shock_colormap()
    ax.imshow(
        np.ma.masked_invalid(schlieren),
        origin="lower",
        extent=extent,
        cmap=color_map,
        interpolation="bilinear",
        vmin=s_min,
        vmax=s_max,
        aspect="auto",
    )

    # Draw physical obstacle body with glowing boundary
    sphere = plt.Circle(
        sphere_center,
        sphere_radius,
        facecolor="#000000",
        edgecolor="#20d49a",
        linewidth=2.0,
        zorder=10,
    )
    ax.add_patch(sphere)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")

    # Cinematic HUD typography
    p_degree = operator.solver.polynomial_degree
    flow_desc = f"Mach {mach:.1f} " if mach is not None else ""
    hdr_title = (
        title
        if title != "Compressible Flow Past Sphere"
        else f"{flow_desc}Compressible Flow Past Sphere (P={p_degree} DG/FV)"
    )
    ax.text(
        0.03, 0.95, hdr_title,
        color="white", fontsize=18, fontweight="bold",
        transform=ax.transAxes, zorder=20,
    )

    info_parts = []
    if sim_time is not None:
        info_parts.append(f"t = {sim_time:.5f} s")
    if step is not None:
        info_parts.append(f"Step {step}")
    if reynolds is not None:
        info_parts.append(f"Re = {int(reynolds):,}")
    info_parts.append("Numerical Schlieren")
    hud_text = " | ".join(info_parts)
    ax.text(
        0.03, 0.91, hud_text,
        color="#00a9cf", fontsize=13,
        transform=ax.transAxes, zorder=20,
    )

    fig.savefig(path, facecolor="#01051f", edgecolor="none", dpi=120)
    plt.close(fig)
    return path


def write_pvd(path, frames):
    """Write a ParaView time-series index for ``[(time, vtu_path), ...]``."""
    path = Path(path)
    datasets = "\n".join(
        f'    <DataSet timestep="{time:.16g}" group="" part="0" '
        f'file="{Path(frame).name}"/>'
        for time, frame in frames
    )
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        '  <Collection>\n' + datasets + '\n  </Collection>\n</VTKFile>\n'
    )
    return path


def save_reacting_heatmap(
    path,
    operator,
    states,
    title="Air-11 Reacting Hypersonic Flow",
    resolution=(900, 450),
    plane_z=0.0,
    sphere_radius=0.5,
    sphere_center=(0.0, 0.0),
):
    """Render a 4-panel diagnostic plot of T, Y_O, Y_NO, and Y_N at midplane."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xy, _, W = _midplane_triangles(operator, states, plane_z=plane_z)

    rounded = np.round(xy, decimals=5)
    unique_xy, inverse = np.unique(rounded, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    unique_W = np.zeros((len(unique_xy), W.shape[1]), dtype=W.dtype)
    for var in range(W.shape[1]):
        unique_W[:, var] = np.bincount(inverse, weights=W[:, var]) / counts

    xmin, xmax = float(np.min(unique_xy[:, 0])), float(np.max(unique_xy[:, 0]))
    ymin, ymax = float(np.min(unique_xy[:, 1])), float(np.max(unique_xy[:, 1]))
    extent = (xmin, xmax, ymin, ymax)

    gx, gy, GX, GY, grid_W, sphere_mask = _resample_to_cartesian_grid(
        unique_xy, unique_W, extent, resolution, sphere_radius, sphere_center
    )

    rho = grid_W[:, :, 0]
    p = grid_W[:, :, 4]
    R_air = 287.05
    T = p / np.maximum(rho * R_air, 1.0e-6)

    layout = operator.solver.layout
    idx_O = layout.index("O") if "O" in layout.species_names else None
    idx_NO = layout.index("NO") if "NO" in layout.species_names else None
    idx_N = layout.index("N") if "N" in layout.species_names else None

    Y_O = grid_W[:, :, idx_O] if idx_O is not None else np.zeros_like(T)
    Y_NO = grid_W[:, :, idx_NO] if idx_NO is not None else np.zeros_like(T)
    Y_N = grid_W[:, :, idx_N] if idx_N is not None else np.zeros_like(T)

    fields = [T, Y_O, Y_NO, Y_N]
    names = (
        "Temperature T [K]",
        "Atomic Oxygen Mass Fraction $Y_O$",
        "Nitric Oxide Mass Fraction $Y_{NO}$",
        "Atomic Nitrogen Mass Fraction $Y_N$",
    )

    figure, axes = plt.subplots(
        2, 2, figsize=(15, 9), constrained_layout=True,
        facecolor="#01051f",
    )
    fire_cmap = plt.get_cmap("inferno").copy()
    fire_cmap.set_bad("#01051f")
    color_map = shock_colormap()

    for field_index, (axis, name) in enumerate(zip(axes.flat, names)):
        field = fields[field_index]
        valid_field = field[~sphere_mask]
        finite = valid_field[np.isfinite(valid_field)]
        if len(finite) and np.max(finite) > np.min(finite):
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        else:
            vmin, vmax = 0.0, 1.0

        cmap = fire_cmap if field_index == 0 else color_map
        shown = axis.imshow(
            np.ma.masked_invalid(field),
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="bilinear",
            vmin=vmin,
            vmax=vmax,
        )
        sphere = plt.Circle(
            sphere_center, sphere_radius,
            facecolor="#01051f",
            edgecolor="#ff2915" if field_index == 0 else "#4cc9f0",
            linewidth=1.5,
            zorder=5,
        )
        axis.add_patch(sphere)

        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
        axis.set_aspect("equal")
        axis.set_facecolor("#01051f")
        axis.set_title(name, color="white", fontsize=12)
        axis.set_xlabel("x", color="white")
        axis.set_ylabel("y", color="white")
        axis.tick_params(colors="white")
        for spine in axis.spines.values():
            spine.set_color("#4cc9f0")
        colorbar = figure.colorbar(shown, ax=axis, shrink=0.84)
        colorbar.ax.tick_params(colors="white")
        colorbar.outline.set_edgecolor("#4cc9f0")

    figure.suptitle(title, color="white", fontsize=15)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path

