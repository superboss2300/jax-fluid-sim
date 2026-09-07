"""Visualization output for discontinuous P2 hybrid fields."""

from pathlib import Path

import meshio
import numpy as np


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


_P2_SUBCELLS = {
    "tetra": (
        "tetra",
        np.asarray([
            [0,4,5,6], [4,1,7,8], [5,7,2,9], [6,8,9,3],
            [4,5,6,8], [4,7,5,8], [5,6,8,9], [5,8,7,9],
        ], dtype=np.int32),
    ),
    "wedge": (
        "wedge",
        np.asarray([
            [0,3,5,12,15,17], [3,1,4,15,13,16],
            [5,4,2,17,16,14], [3,4,5,15,16,17],
            [12,15,17,6,9,11], [15,13,16,9,7,10],
            [17,16,14,11,10,8], [15,16,17,9,10,11],
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
                    index(x0,y0,z0), index(x1,y0,z0),
                    index(x1,y1,z0), index(x0,y1,z0),
                    index(x0,y0,z1), index(x1,y0,z1),
                    index(x1,y1,z1), index(x0,y1,z1),
                ])
    return "hexahedron", np.asarray(cells, dtype=np.int32)


_P2_SUBCELLS["hexahedron"] = _hex_p2_subcells()


_LINEAR_CELL_EDGES = {
    "tetra": ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)),
    "wedge": (
        (0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5),
    ),
    "hexahedron": (
        (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
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
        if operator.solver.polynomial_degree == 2:
            output_type, subcells = _P2_SUBCELLS[element_type]
        else:
            output_type = element_type
            subcells = np.arange(coordinates.shape[1], dtype=np.int32)[None, :]
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
                order = np.argsort(np.arctan2(xy[:,1]-center[1], xy[:,0]-center[0]))
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
    """Write each P2 parent as eight linear visualization subcells."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points, primitives, cells = [], [], []
    point_offset = 0
    for block in operator.mesh.element_blocks:
        element_type = block.element_type
        coordinates = np.asarray(
            operator.geometry.solution_node_coordinates[element_type]
        )
        W = np.asarray(operator.solver.conserved_to_primitive(states[element_type]))
        points.append(coordinates.reshape(-1, 3))
        primitives.append(W.reshape(-1, W.shape[-1]))
        nodes_per_element = coordinates.shape[1]
        if operator.solver.polynomial_degree == 2:
            output_type, local_subcells = _P2_SUBCELLS[element_type]
        else:
            output_type = element_type
            local_subcells = np.arange(nodes_per_element, dtype=np.int32)[None, :]
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
        np.maximum(operator.solver.gamma * W[:, 4] / W[:, 0], 0.0)
    )
    speed = np.linalg.norm(W[:, 1:4], axis=-1)
    point_data = {
        "density": W[:, 0], "velocity": W[:, 1:4], "pressure": W[:, 4],
        "speed": speed, "mach": speed / np.maximum(sound_speed, 1.0e-30),
    }
    meshio.write(path, meshio.Mesh(points, cells, point_data=point_data))
    return path


def save_midplane_heatmap(path, operator, states, title="Hybrid P2 DG", resolution=(900, 450)):
    """Render an exact, continuous z=0 cut resampled to a Cartesian pixel grid."""
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

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

    # Resample onto a uniform Cartesian grid to eliminate triangular facet artifacts
    nx, ny = resolution
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    GX, GY = np.meshgrid(gx, gy)
    sphere_mask = (GX**2 + GY**2) < (0.5 + 1.0e-4)**2

    grid_W = np.zeros((ny, nx, 5))
    for v in range(5):
        val = griddata(unique_xy, unique_W[:, v], (GX, GY), method="linear")
        val[sphere_mask] = np.nan
        grid_W[:, :, v] = val

    speed = np.linalg.norm(grid_W[:, :, 1:4], axis=-1)
    sound = np.sqrt(
        np.maximum(
            operator.solver.gamma * grid_W[:, :, 4]
            / np.maximum(grid_W[:, :, 0], 1.0e-12),
            0.0,
        )
    )
    mach = speed / np.maximum(sound, 1.0e-30)

    # Compute continuous Schlieren gradient on the Cartesian grid
    dy = gy[1] - gy[0]
    dx = gx[1] - gx[0]
    grad_y, grad_x = np.gradient(np.nan_to_num(grid_W[:, :, 0], nan=1.0), dy, dx)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag[sphere_mask] = np.nan
    density_span = np.nanmax(grid_W[:, :, 0]) - np.nanmin(grid_W[:, :, 0])
    if density_span < 1.0e-5 * max(abs(float(np.nanmedian(grid_W[:, :, 0]))), 1.0):
        schlieren = np.zeros_like(grad_mag)
    else:
        schlieren = np.log10(1.0 + grad_mag)

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
            vmin, vmax = (0.0, 1.0) if not np.any(np.isfinite(valid_field)) else _robust_limits(valid_field)

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
