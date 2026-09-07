"""Run a short DG Navier--Stokes wind-tunnel test around a sphere."""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tetra_dg import TetrahedralDG


class VTKTimeSeriesWriter:
    """Write compact, ParaView-ready VTU frames and a PVD animation index."""

    def __init__(self, solver, directory):
        try:
            import meshio
        except ImportError as error:
            raise RuntimeError(
                "VTK output requires meshio; install it with `pipenv install meshio`."
            ) from error

        self.meshio = meshio
        self.solver = solver
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames = []

        if solver.polynomial_degree == 2 and solver.geometry_order == 2:
            # Convert the solver's edge ordering back to meshio/Gmsh tetra10
            # ordering: 01, 12, 20, 03, 13, 23.
            order = np.array([0, 1, 2, 3, 4, 7, 5, 6, 8, 9])
            self.connectivity = np.asarray(solver.geometry_connectivity)[:, order]
            self.local_connectivity = np.asarray(solver.geometry_connectivity)
            self.cell_type = "tetra10"
        else:
            self.connectivity = np.asarray(solver.elements)
            self.local_connectivity = self.connectivity
            self.cell_type = "tetra"

        self.points = np.asarray(solver.points)

    def _continuous_point_data(self, U):
        """Average coincident DG traces only for smooth visualization output."""
        W = np.asarray(self.solver.conserved_to_primitive(U))
        sums = np.zeros((self.points.shape[0], 5), dtype=W.dtype)
        counts = np.zeros(self.points.shape[0], dtype=np.int32)
        np.add.at(sums, self.local_connectivity.reshape(-1), W.reshape(-1, 5))
        np.add.at(counts, self.local_connectivity.reshape(-1), 1)
        valid = counts > 0
        projected = np.zeros_like(sums)
        projected[valid] = sums[valid] / counts[valid, None]

        velocity = projected[:, 1:4]
        speed = np.linalg.norm(velocity, axis=1)
        sound_speed = np.sqrt(
            self.solver.gamma
            * np.maximum(projected[:, 4], 0.0)
            / np.maximum(projected[:, 0], 1.0e-14)
        )
        return {
            "density": projected[:, 0],
            "velocity": velocity,
            "pressure": projected[:, 4],
            "Mach": speed / np.maximum(sound_speed, 1.0e-14),
            "temperature": projected[:, 4]
            / np.maximum(projected[:, 0] * self.solver.gas_constant, 1.0e-14),
        }

    def write(self, U, simulation_time):
        frame_name = f"frame_{len(self.frames):05d}.vtu"
        self.meshio.write_points_cells(
            self.directory / frame_name,
            self.points,
            [(self.cell_type, self.connectivity)],
            point_data=self._continuous_point_data(U),
            binary=True,
        )
        self.frames.append((float(simulation_time), frame_name))
        self._write_index()

    def _write_index(self):
        datasets = "\n".join(
            f'    <DataSet timestep="{time_value:.16g}" group="" part="0" file="{name}"/>'
            for time_value, name in self.frames
        )
        contents = (
            '<?xml version="1.0"?>\n'
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
            "  <Collection>\n"
            f"{datasets}\n"
            "  </Collection>\n"
            "</VTKFile>\n"
        )
        (self.directory / "sphere_animation.pvd").write_text(contents)


class TetrahedralPlaneSlice:
    """A genuine plane/tetrahedron intersection with parent DG coordinates."""

    _edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    _p2_subtetrahedra = (
        (0, 4, 5, 6),
        (1, 4, 7, 8),
        (2, 5, 7, 9),
        (3, 6, 8, 9),
        (4, 5, 6, 9),
        (4, 5, 7, 9),
        (4, 6, 8, 9),
        (4, 7, 8, 9),
    )

    def __init__(self, solver, z_value=0.0):
        self.solver = solver
        self.z_value = float(z_value)
        self._build()

    @staticmethod
    def _unique_intersections(points, references, z_value, tolerance):
        intersections = []

        def add(point, reference):
            if not any(
                np.linalg.norm(point - old_point) <= tolerance
                for old_point, _ in intersections
            ):
                intersections.append((point, reference))

        signed_z = points[:, 2] - z_value
        for vertex in range(4):
            if abs(signed_z[vertex]) <= tolerance:
                add(points[vertex], references[vertex])

        for first, second in TetrahedralPlaneSlice._edges:
            z_first = signed_z[first]
            z_second = signed_z[second]
            if z_first * z_second < -(tolerance**2):
                fraction = z_first / (z_first - z_second)
                add(
                    points[first] + fraction * (points[second] - points[first]),
                    references[first]
                    + fraction * (references[second] - references[first]),
                )
        return intersections

    def _build(self):
        geometry_nodes = np.asarray(self.solver.geometry_nodes)
        if self.solver.geometry_order == 2:
            geometry_reference_nodes = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ]
            )
            subtetrahedra = self._p2_subtetrahedra
        else:
            geometry_reference_nodes = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            subtetrahedra = ((0, 1, 2, 3),)

        intersects_plane = np.flatnonzero(
            (np.min(geometry_nodes[..., 2], axis=1) <= self.z_value)
            & (np.max(geometry_nodes[..., 2], axis=1) >= self.z_value)
        )
        triangle_points = []
        triangle_references = []
        triangle_elements = []
        seen_triangles = set()

        domain_scale = float(np.ptp(np.asarray(self.solver.points), axis=0).max())
        tolerance = max(1.0e-12 * domain_scale, 1.0e-14)

        for element in intersects_plane:
            for local_indices in subtetrahedra:
                local_indices = np.asarray(local_indices)
                points = geometry_nodes[element, local_indices]
                references = geometry_reference_nodes[local_indices]
                intersections = self._unique_intersections(
                    points,
                    references,
                    self.z_value,
                    tolerance,
                )
                if len(intersections) < 3:
                    continue

                physical = np.asarray([entry[0] for entry in intersections])
                reference = np.asarray([entry[1] for entry in intersections])
                center = np.mean(physical[:, :2], axis=0)
                angle = np.arctan2(
                    physical[:, 1] - center[1],
                    physical[:, 0] - center[0],
                )
                order = np.argsort(angle)
                physical = physical[order]
                reference = reference[order]

                for corner in range(1, len(physical) - 1):
                    physical_triangle = physical[[0, corner, corner + 1]]
                    reference_triangle = reference[[0, corner, corner + 1]]
                    first_edge = (
                        physical_triangle[1, :2] - physical_triangle[0, :2]
                    )
                    second_edge = (
                        physical_triangle[2, :2] - physical_triangle[0, :2]
                    )
                    twice_area = abs(
                        first_edge[0] * second_edge[1]
                        - first_edge[1] * second_edge[0]
                    )
                    if twice_area <= tolerance**2:
                        continue
                    triangle_key = tuple(
                        sorted(
                            tuple(np.round(point[:2], 12))
                            for point in physical_triangle
                        )
                    )
                    if triangle_key in seen_triangles:
                        continue
                    seen_triangles.add(triangle_key)
                    triangle_points.append(physical_triangle)
                    triangle_references.append(reference_triangle)
                    triangle_elements.append(element)

        if not triangle_points:
            raise ValueError(f"The mesh does not intersect z={self.z_value}")

        self.points = np.asarray(triangle_points)
        self.reference_coordinates = jnp.asarray(np.asarray(triangle_references))
        self.element_ids = jnp.asarray(np.asarray(triangle_elements, dtype=np.int32))

    def make_pixel_grid(self, extent, width=900, height=600):
        """Scan-convert the tetrahedral cut to pixels without changing the DG state."""
        x_min, x_max, y_min, y_max = extent
        x_pixels = np.linspace(x_min, x_max, width)
        y_pixels = np.linspace(y_min, y_max, height)
        pixel_element = np.full((height, width), -1, dtype=np.int32)
        pixel_reference = np.zeros((height, width, 3), dtype=np.float64)

        triangle_elements = np.asarray(self.element_ids)
        triangle_references = np.asarray(self.reference_coordinates)
        epsilon = 1.0e-10

        for triangle, references, element in zip(
            self.points[..., :2], triangle_references, triangle_elements
        ):
            column_start = max(0, np.searchsorted(x_pixels, triangle[:, 0].min()) - 1)
            column_stop = min(width, np.searchsorted(x_pixels, triangle[:, 0].max(), side="right") + 1)
            row_start = max(0, np.searchsorted(y_pixels, triangle[:, 1].min()) - 1)
            row_stop = min(height, np.searchsorted(y_pixels, triangle[:, 1].max(), side="right") + 1)
            if column_start >= column_stop or row_start >= row_stop:
                continue

            local_x, local_y = np.meshgrid(
                x_pixels[column_start:column_stop],
                y_pixels[row_start:row_stop],
            )
            x0, y0 = triangle[0]
            x1, y1 = triangle[1]
            x2, y2 = triangle[2]
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denominator) < 1.0e-20:
                continue

            weight0 = (
                (y1 - y2) * (local_x - x2)
                + (x2 - x1) * (local_y - y2)
            ) / denominator
            weight1 = (
                (y2 - y0) * (local_x - x2)
                + (x0 - x2) * (local_y - y2)
            ) / denominator
            weight2 = 1.0 - weight0 - weight1
            inside = (
                (weight0 >= -epsilon)
                & (weight1 >= -epsilon)
                & (weight2 >= -epsilon)
            )

            local_owner = pixel_element[
                row_start:row_stop, column_start:column_stop
            ]
            # A point on a shared edge may belong to either adjacent element.
            write = inside & (local_owner < 0)
            local_owner[write] = element
            local_reference = pixel_reference[
                row_start:row_stop, column_start:column_stop
            ]
            interpolated_reference = (
                weight0[..., None] * references[0]
                + weight1[..., None] * references[1]
                + weight2[..., None] * references[2]
            )
            local_reference[write] = interpolated_reference[write]

        valid = pixel_element >= 0
        return x_pixels, y_pixels, pixel_element, pixel_reference, valid

    def evaluate_pixel_fields(
        self,
        U,
        extent=(-2.0, 4.0, -2.0, 2.0),
        width=900,
        height=600,
        batch_size=65536,
    ):
        """Evaluate the continuous presentation projection at heatmap pixels."""
        visualization_W = self.continuous_primitive_nodes(U)
        x_pixels, y_pixels, element_image, reference_image, valid = (
            self.make_pixel_grid(extent, width=width, height=height)
        )
        flat_locations = np.flatnonzero(valid)
        element_ids = element_image.ravel()[flat_locations]
        reference_coordinates = reference_image.reshape(-1, 3)[flat_locations]
        field_values = [np.full(height * width, np.nan) for _ in range(6)]

        for start in range(0, flat_locations.size, batch_size):
            stop = min(start + batch_size, flat_locations.size)
            batch_elements = jnp.asarray(element_ids[start:stop])
            # The singleton dimension is the quadrature/sample-point axis used
            # throughout the solver's geometry routines.
            batch_references = jnp.asarray(
                reference_coordinates[start:stop, None, :]
            )
            batch_fields = self.evaluate_fields(
                U,
                element_ids=batch_elements,
                reference_coordinates=batch_references,
                element_primitive_nodes=visualization_W,
            )

            destination = flat_locations[start:stop]
            for output, values in zip(field_values, batch_fields):
                output[destination] = np.asarray(values).reshape(-1)

        images = tuple(values.reshape(height, width) for values in field_values)
        return x_pixels, y_pixels, images

    def continuous_primitive_nodes(self, U):
        """Project discontinuous DG traces to shared nodes for presentation."""
        W = np.asarray(self.solver.conserved_to_primitive(U))
        if (
            self.solver.polynomial_degree == 2
            and self.solver.geometry_order == 2
        ):
            connectivity = np.asarray(self.solver.geometry_connectivity)
        else:
            connectivity = np.asarray(self.solver.elements)

        sums = np.zeros((self.solver.points.shape[0], 5), dtype=W.dtype)
        counts = np.zeros(self.solver.points.shape[0], dtype=np.int32)
        np.add.at(sums, connectivity.reshape(-1), W.reshape(-1, 5))
        np.add.at(counts, connectivity.reshape(-1), 1)
        valid = counts > 0
        shared_W = np.zeros_like(sums)
        shared_W[valid] = sums[valid] / counts[valid, None]
        return jnp.asarray(shared_W[connectivity])

    def evaluate_fields(
        self,
        U,
        element_ids=None,
        reference_coordinates=None,
        element_primitive_nodes=None,
    ):
        if element_ids is None:
            element_ids = self.element_ids
        if reference_coordinates is None:
            reference_coordinates = self.reference_coordinates

        if element_primitive_nodes is None:
            element_U = U[element_ids]
            element_W = self.solver.conserved_to_primitive(element_U)
        else:
            element_W = element_primitive_nodes[element_ids]
            element_U = self.solver.primitive_to_conserved(element_W)
        basis = self.solver.basis_at_reference_points(
            reference_coordinates
        )
        W_slice = jnp.einsum("tqi,tiv->tqv", basis, element_W)
        U_slice = self.solver.primitive_to_conserved(W_slice)

        reference_gradients = self.solver.basis_gradients_at_reference_points(
            reference_coordinates
        )
        geometry_jacobian = self.solver.geometry_jacobian_at_reference_points(
            element_ids,
            reference_coordinates,
        )
        physical_gradients = jnp.einsum(
            "tqir,tqrx->tqix",
            reference_gradients,
            jnp.linalg.inv(geometry_jacobian),
        )

        nodal_density = element_W[..., 0]
        density_gradient = jnp.einsum(
            "ti,tqix->tqx",
            nodal_density,
            physical_gradients,
        )
        schlieren = jnp.log10(
            1.0 + jnp.linalg.norm(density_gradient, axis=-1)
        )

        viscous_nodes = self.solver.viscous_variables(U)[element_ids]
        viscous_gradient = jnp.einsum(
            "tim,tqix->tqmx",
            viscous_nodes,
            physical_gradients,
        )
        velocity_gradient = viscous_gradient[..., :3, :]
        vorticity = jnp.stack(
            [
                velocity_gradient[..., 2, 1] - velocity_gradient[..., 1, 2],
                velocity_gradient[..., 0, 2] - velocity_gradient[..., 2, 0],
                velocity_gradient[..., 1, 0] - velocity_gradient[..., 0, 1],
            ],
            axis=-1,
        )

        speed = jnp.linalg.norm(W_slice[..., 1:4], axis=-1)
        sound_speed = jnp.sqrt(
            self.solver.gamma * W_slice[..., 4] / W_slice[..., 0]
        )
        if self.solver.use_wale:
            turbulent_viscosity = self.solver.wale_dynamic_viscosity(
                U_slice,
                viscous_gradient,
                self.solver.element_filter_width[element_ids, None],
            )
            wale_ratio = turbulent_viscosity / self.solver.dynamic_viscosity
        else:
            wale_ratio = jnp.zeros_like(speed)

        return (
            W_slice[..., 0],
            W_slice[..., 4],
            speed / sound_speed,
            schlieren,
            jnp.linalg.norm(vorticity, axis=-1),
            wale_ratio,
        )


def plot_center_slice(solver, U, current_time, output_path):
    plane_slice = TetrahedralPlaneSlice(solver, z_value=0.0)
    extent = (-2.0, 4.0, -2.0, 2.0)
    _, _, slice_fields = plane_slice.evaluate_pixel_fields(U, extent=extent)

    # DG derivatives are discontinuous even after the primitive variables are
    # projected to shared nodes. Apply a one-pixel, mask-aware recovery only to
    # derivative-based presentation fields. The raw state in the NPZ is not
    # modified and must be used for quantitative comparisons.
    from scipy.ndimage import gaussian_filter

    def recover_display_field(values, sigma=1.0):
        valid = np.isfinite(values)
        numerator = gaussian_filter(
            np.where(valid, values, 0.0), sigma=sigma, mode="nearest"
        )
        denominator = gaussian_filter(
            valid.astype(float), sigma=sigma, mode="nearest"
        )
        recovered = numerator / np.maximum(denominator, 1.0e-14)
        return np.where(valid, recovered, np.nan)

    slice_fields = (
        slice_fields[0],
        slice_fields[1],
        slice_fields[2],
        recover_display_field(slice_fields[3]),
        recover_display_field(slice_fields[4]),
        recover_display_field(slice_fields[5]),
    )

    fields = (
        (slice_fields[0], "Density", "viridis"),
        (slice_fields[1], "Pressure", "coolwarm"),
        (slice_fields[2], "Mach number", "plasma"),
        (slice_fields[3], r"Schlieren $\log_{10}(1+|\nabla\rho|)$", "gray"),
        (slice_fields[4], "Vorticity magnitude", "inferno"),
        (slice_fields[5], r"WALE $\mu_t/\mu$", "magma"),
    )

    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for axis, (values, title, color_map) in zip(axes.flat, fields):
        value_min = float(np.nanmin(values))
        value_max = float(np.nanmax(values))
        if value_max <= value_min:
            value_max = value_min + max(1.0e-12, abs(value_min) * 1.0e-12)
        image = axis.imshow(
            np.ma.masked_invalid(values),
            origin="lower",
            extent=extent,
            cmap=color_map,
            interpolation="bilinear",
            vmin=value_min,
            vmax=value_max,
        )
        obstacle = plt.Circle((0.0, 0.0), 0.5, color="black", zorder=5)
        axis.add_patch(obstacle)
        axis.set_xlim(-2.0, 4.0)
        axis.set_ylim(-2.0, 2.0)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(title)
        figure.colorbar(image, ax=axis)

    figure.suptitle(
        f"P{solver.polynomial_degree} DG sphere flow at t={current_time:.6f} "
        "(continuous display projection)"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    return figure


def run(args):
    farfield_sound_speed = np.sqrt(1.4)
    farfield_velocity = args.mach * farfield_sound_speed
    sphere_diameter = 1.0
    solver = TetrahedralDG(
        args.mesh,
        polynomial_degree=args.polynomial_degree,
        use_wale=args.wale,
        use_shock_limiter=args.shock_limiter,
        shock_sensor_threshold=args.shock_sensor_threshold,
        farfield_mach=args.mach,
        reynolds=args.reynolds,
        reynolds_reference_density=1.0,
        reynolds_reference_velocity=farfield_velocity,
        reynolds_reference_length=sphere_diameter,
    )

    W_initial = jnp.broadcast_to(
        solver.farfield_primitive,
        (solver.number_of_elements, solver.nodes_per_element, 5),
    )
    U = solver.primitive_to_conserved(W_initial)
    initial_integrals = solver.integrated_conserved_state(U)

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    label = f"sphere_p{args.polynomial_degree}_t{args.target_time:.6f}".replace(".", "p")
    vtk_writer = None
    if args.vtk:
        vtk_writer = VTKTimeSeriesWriter(
            solver,
            output_directory / f"{label}_vtk",
        )
        vtk_writer.write(U, 0.0)
        print(
            "Wrote VTK frame 0; open sphere_animation.pvd in ParaView.",
            flush=True,
        )

    advance = jax.jit(
        solver.advance_navier_stokes_to_time,
        static_argnames=("maximum_steps",),
    )
    current_time = jnp.asarray(0.0, dtype=U.dtype)
    current_step = jnp.asarray(0, dtype=jnp.int32)
    output_times = np.linspace(0.0, args.target_time, args.progress_updates + 1)[1:]

    print(
        f"elements={solver.number_of_elements} p={solver.polynomial_degree} "
        f"geometry=p{solver.geometry_order} Mach={args.mach} "
        f"Re_D={args.reynolds} mu={solver.dynamic_viscosity:.6e}",
        flush=True,
    )
    print("Compiling and advancing the first interval...", flush=True)
    start = time.perf_counter()

    for output_index, requested_time in enumerate(output_times, start=1):
        U, current_time, current_step = advance(
            U,
            current_time,
            current_step,
            requested_time,
            args.maximum_steps,
        )
        U.block_until_ready()
        W = solver.conserved_to_primitive(U)
        elapsed = time.perf_counter() - start
        fraction = output_index / len(output_times)
        eta = elapsed * (1.0 - fraction) / fraction
        troubled = (
            int(jnp.sum(solver.troubled_element_mask(U)))
            if args.shock_limiter
            else 0
        )
        print(
            f"progress={100.0*fraction:6.2f}% "
            f"time={float(current_time):.6f} steps={int(current_step)} "
            f"rho=[{float(jnp.min(W[..., 0])):.6f}, {float(jnp.max(W[..., 0])):.6f}] "
            f"p=[{float(jnp.min(W[..., 4])):.6f}, {float(jnp.max(W[..., 4])):.6f}] "
            f"troubled={troubled} elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
        if vtk_writer is not None:
            vtk_writer.write(U, current_time)
            print(f"  wrote VTK frame {output_index}", flush=True)

    final_integrals = solver.integrated_conserved_state(U)
    final_W = solver.conserved_to_primitive(U)
    image_path = output_directory / f"{label}.png"
    state_path = output_directory / f"{label}.npz"
    figure = plot_center_slice(solver, U, float(current_time), image_path)
    np.savez_compressed(
        state_path,
        U=np.asarray(U),
        time=float(current_time),
        steps=int(current_step),
    )

    print("All finite:", bool(jnp.all(jnp.isfinite(U))))
    print("Minimum density:", float(jnp.min(final_W[..., 0])))
    print("Minimum pressure:", float(jnp.min(final_W[..., 4])))
    print("Integrated-state change:", np.asarray(final_integrals - initial_integrals))
    print("Saved figure:", image_path.resolve())
    print("Saved restart state:", state_path.resolve())
    if vtk_writer is not None:
        print(
            "Saved ParaView animation:",
            (vtk_writer.directory / "sphere_animation.pvd").resolve(),
        )

    if args.show:
        plt.show()
    else:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", default="mesh/sphere_tunnel_gpu6gb.msh")
    parser.add_argument("--target-time", type=float, default=0.002)
    parser.add_argument("--maximum-steps", type=int, default=10000)
    parser.add_argument("--progress-updates", type=int, default=10)
    parser.add_argument("--polynomial-degree", type=int, choices=(1, 2), default=2)
    parser.add_argument("--mach", type=float, default=0.2)
    parser.add_argument(
        "--reynolds",
        type=float,
        default=10000.0,
        help="Reynolds number based on freestream speed and sphere diameter.",
    )
    parser.add_argument("--wale", action="store_true")
    parser.add_argument("--shock-limiter", action="store_true")
    parser.add_argument("--shock-sensor-threshold", type=float, default=0.1)
    parser.add_argument("--output-directory", default="outputs/dg_sphere")
    parser.add_argument(
        "--vtk",
        action="store_true",
        help=(
            "Write one VTU frame per progress interval plus a ParaView PVD "
            "animation. Increase --progress-updates for more frames."
        ),
    )
    parser.add_argument("--show", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
