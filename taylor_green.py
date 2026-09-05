"""Run a periodic Taylor--Green vortex using the unstructured FV solver."""

import argparse
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from solver3d import Solver


DOMAIN_LENGTH = 2.0 * jnp.pi
PERIODIC_PAIRS = [
    (
        "periodic_x_minus",
        "periodic_x_plus",
        (float(DOMAIN_LENGTH), 0.0, 0.0),
    ),
    (
        "periodic_y_minus",
        "periodic_y_plus",
        (0.0, float(DOMAIN_LENGTH), 0.0),
    ),
    (
        "periodic_z_minus",
        "periodic_z_plus",
        (0.0, 0.0, float(DOMAIN_LENGTH)),
    ),
]


def diagnostics(solver, U):
    W = solver.conserved_to_primitive(U)
    density = W[:, 0]
    velocity = W[:, 1:4]
    pressure = W[:, 4]
    volumes = solver.cell_volumes
    total_volume = jnp.sum(volumes)

    kinetic_energy = jnp.sum(
        0.5
        * density
        * jnp.sum(velocity**2, axis=1)
        * volumes
    ) / total_volume

    velocity_gradient, _ = solver.compute_primitive_gradients(U)
    turbulent_viscosity = solver.compute_wale_viscosity(
        U,
        velocity_gradient,
    )

    return {
        "kinetic_energy": kinetic_energy,
        "mass": jnp.sum(density * volumes),
        "momentum": jnp.sum(U[:, 1:4] * volumes[:, None], axis=0),
        "total_energy": jnp.sum(U[:, 4] * volumes),
        "rho_min": jnp.min(density),
        "pressure_min": jnp.min(pressure),
        "mu_t_max": jnp.max(turbulent_viscosity),
        "finite": jnp.all(jnp.isfinite(U)),
    }


def print_diagnostics(step, current_time, values, initial_values):
    kinetic_ratio = (
        values["kinetic_energy"] / initial_values["kinetic_energy"]
    )
    mass_error = (
        values["mass"] - initial_values["mass"]
    ) / initial_values["mass"]
    energy_error = (
        values["total_energy"] - initial_values["total_energy"]
    ) / initial_values["total_energy"]
    momentum_error = jnp.linalg.norm(
        values["momentum"] - initial_values["momentum"]
    ) / initial_values["mass"]

    print(
        f"step={step:5d} "
        f"time={current_time:.6f} "
        f"K/K0={float(kinetic_ratio):.8f} "
        f"mass_error={float(mass_error):+.3e} "
        f"momentum_error={float(momentum_error):.3e} "
        f"energy_error={float(energy_error):+.3e} "
        f"rho_min={float(values['rho_min']):.6f} "
        f"p_min={float(values['pressure_min']):.6f} "
        f"mu_t_max={float(values['mu_t_max']):.6e} "
        f"finite={bool(values['finite'])}"
    )


def plot_results(solver, U, times, kinetic_energies, current_time):
    """Plot energy history and a cell-centered slab near z = pi."""
    W = solver.conserved_to_primitive(U)
    velocity = W[:, 1:4]
    velocity_gradient, _ = solver.compute_primitive_gradients(U)
    turbulent_viscosity = solver.compute_wale_viscosity(
        U,
        velocity_gradient,
    )

    vorticity = jnp.stack(
        [
            velocity_gradient[:, 2, 1] - velocity_gradient[:, 1, 2],
            velocity_gradient[:, 0, 2] - velocity_gradient[:, 2, 0],
            velocity_gradient[:, 1, 0] - velocity_gradient[:, 0, 1],
        ],
        axis=-1,
    )

    centers = np.asarray(jax.device_get(solver.cell_centers))
    speed = np.asarray(
        jax.device_get(jnp.linalg.norm(velocity, axis=1))
    )
    vorticity_magnitude = np.asarray(
        jax.device_get(jnp.linalg.norm(vorticity, axis=1))
    )
    viscosity_ratio = np.asarray(
        jax.device_get(
            turbulent_viscosity / solver.dynamic_viscosity
        )
    )

    # A true tetrahedron/plane intersection is more involved.  For this first
    # viewer, show cell centers in a thin slab around the middle plane.
    characteristic_cell_size = float(
        np.cbrt(np.mean(np.asarray(solver.cell_volumes)))
    )
    slice_mask = (
        np.abs(centers[:, 2] - np.pi)
        <= 0.6 * characteristic_cell_size
    )

    x = centers[slice_mask, 0]
    y = centers[slice_mask, 1]
    marker_size = max(3.0, 18000.0 / solver.number_of_cells)

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    initial_kinetic_energy = kinetic_energies[0]
    axes[0, 0].plot(
        times,
        np.asarray(kinetic_energies) / initial_kinetic_energy,
        marker="o",
    )
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel(r"$K(t)/K(0)$")
    axes[0, 0].set_title("Resolved kinetic energy")
    axes[0, 0].grid(True, alpha=0.3)

    fields = (
        (speed[slice_mask], "Speed", "viridis"),
        (vorticity_magnitude[slice_mask], "Vorticity magnitude", "magma"),
        (viscosity_ratio[slice_mask], r"WALE $\mu_t/\mu$", "plasma"),
    )

    for axis, (field, title, color_map) in zip(
        (axes[0, 1], axes[1, 0], axes[1, 1]),
        fields,
    ):
        points = axis.scatter(
            x,
            y,
            c=field,
            s=marker_size,
            cmap=color_map,
            linewidths=0.0,
        )
        axis.set_xlim(0.0, float(DOMAIN_LENGTH))
        axis.set_ylim(0.0, float(DOMAIN_LENGTH))
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(f"{title} near z = pi")
        figure.colorbar(points, ax=axis)

    figure.suptitle(f"Taylor--Green vortex at t = {current_time:.4f}")
    figure.tight_layout()
    plt.show()


def run(number_of_steps, report_every, make_plot):
    solver = Solver(
        "mesh/taylor_green.msh",
        case="taylor_green",
        periodic_pairs=PERIODIC_PAIRS,
        reynolds=1600.0,
    )

    U = solver.U_initial
    initial_values = diagnostics(solver, U)
    current_time = 0.0
    completed_steps = 0
    times = [0.0]
    kinetic_energies = [float(initial_values["kinetic_energy"])]

    print_diagnostics(0, current_time, initial_values, initial_values)

    compiled_chunk = jax.jit(
        lambda state: solver.simulate(state, report_every)
    )

    wall_start = time.perf_counter()
    while completed_steps + report_every <= number_of_steps:
        U, elapsed_time = compiled_chunk(U)
        U.block_until_ready()
        completed_steps += report_every
        current_time += float(elapsed_time)

        values = diagnostics(solver, U)
        print_diagnostics(
            completed_steps,
            current_time,
            values,
            initial_values,
        )
        times.append(current_time)
        kinetic_energies.append(float(values["kinetic_energy"]))

    remaining_steps = number_of_steps - completed_steps
    if remaining_steps:
        final_chunk = jax.jit(
            lambda state: solver.simulate(state, remaining_steps)
        )
        U, elapsed_time = final_chunk(U)
        U.block_until_ready()
        completed_steps += remaining_steps
        current_time += float(elapsed_time)
        values = diagnostics(solver, U)
        print_diagnostics(
            completed_steps,
            current_time,
            values,
            initial_values,
        )
        times.append(current_time)
        kinetic_energies.append(float(values["kinetic_energy"]))

    print(f"Execution time: {time.perf_counter() - wall_start:.3f} s")

    if make_plot:
        plot_results(
            solver,
            U,
            times,
            kinetic_energies,
            current_time,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--plot", action="store_true")
    arguments = parser.parse_args()

    if arguments.steps <= 0:
        parser.error("--steps must be positive")
    if arguments.report_every <= 0:
        parser.error("--report-every must be positive")

    run(arguments.steps, arguments.report_every, arguments.plot)


if __name__ == "__main__":
    main()
