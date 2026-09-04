import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import numpy as np

number_of_cells = 200
number_of_steps = 200
dx = 1.0 / number_of_cells
x = (jnp.arange(number_of_cells) + 0.5) * dx
owner = jnp.arange(0, number_of_cells - 1)
neighbor = jnp.arange(1, number_of_cells)

density = jnp.where(x < 0.5, 1.0, 0.125)
velocity = jnp.zeros_like(x)
pressure = jnp.where(x < 0.5, 1.0, 0.1)
W = jnp.stack([density, velocity, pressure], axis=-1)

cfl = 0.5
# dt = cfl * dx / jnp.max(jnp.abs(U))
gravity = 9.81
gamma = 1.4

cell_volume = jnp.full(
    (number_of_cells,),
    dx,
)

internal_face_normal = jnp.ones(
    number_of_cells - 1
)

boundary_owner = jnp.array([
    0,
    number_of_cells - 1,
])

boundary_normal = jnp.array([
    -1.0,
    1.0,
])

boundary_area = jnp.array([
    1.0,
    1.0,
])

left_boundary_area = 1.0
right_boundary_area = 1.0


def physical_flux(u):
    momentum = u[:, 1]
    E = u[:, 2]
    W = inverse(u)
    density = W[:, 0]
    velocity = W[:, 1]
    pressure = W[:, 2]
    combined_flux = jnp.stack([momentum, (momentum*velocity) + pressure, velocity * (E + pressure)], axis=-1)
    return combined_flux

def conserved_vars(W):
    density = W[:, 0]
    velocity = W[:, 1]
    pressure = W[:, 2]
    momentum =  jnp.multiply(density, velocity)
    print("Momentum: ", momentum.shape)
    E = jnp.add(pressure/(gamma-1), (1/2*(momentum**2/density)))
    print("E: ", E.shape)
    U = jnp.stack([density, momentum, E], axis=1)
    print("Final shape of U: ", U.shape)
    return U

U_initial = conserved_vars(W)


def inverse(U):
    density = U[..., 0]
    momentum = U[..., 1]
    E = U[..., 2]
    velocity = momentum/density
    pressure = jnp.multiply((gamma - 1), (E - (1/2)*(momentum**2/density)))
    W_new = jnp.stack([density, velocity, pressure], axis=1)
    return W_new


def rusanov_flux(U_L, U_R, normal):
    F_L = physical_flux(U_L)
    F_R = physical_flux(U_R)
    W_L = inverse(U_L)
    W_R = inverse(U_R)
    density_L = W_L[:, 0]
    velocity_L = W_L[:, 1]
    pressure_L = W_L[:, 2]

    density_R = W_R[:, 0]
    velocity_R = W_R[:, 1]
    pressure_R = W_R[:, 2]


    sound_speed_L = jnp.sqrt(
        gamma * pressure_L / density_L
    )

    sound_speed_R = jnp.sqrt(
        gamma * pressure_R / density_R
    )

    normal_velocity_L = velocity_L * normal
    normal_velocity_R = velocity_R * normal

    alpha = jnp.maximum(
        jnp.abs(normal_velocity_L) + sound_speed_L,
        jnp.abs(normal_velocity_R) + sound_speed_R,
    )

    normal_flux_L = normal[:, None] * F_L
    normal_flux_R = normal[:, None] * F_R

    return (
        0.5 * (normal_flux_L + normal_flux_R)
        - 0.5 * alpha[:, None] * (U_R - U_L)
    )


def connectivity_residiual(u):
    U_L = u[owner]
    U_R = u[neighbor]
    internal_flux = rusanov_flux(
        U_L,
        U_R,
        internal_face_normal,
    )
    internal_face_area = jnp.ones(
        number_of_cells - 1
    )
    integrated_flux = (
        internal_face_area[:, None]
        * internal_flux
    )
    residual = jnp.zeros_like(u)
    residual = residual.at[owner].add(
        internal_flux
    )
    residual = residual.at[neighbor].add(
        -internal_flux
    )
    boundary_state = u[boundary_owner]

    boundary_flux = rusanov_flux(
        boundary_state,
        boundary_state,
        boundary_normal,
    )
    integrated_boundary_flux = (
        boundary_area[:, None]
        * boundary_flux
    )
    residual = residual.at[boundary_owner].add(
        integrated_boundary_flux
    )
    return residual


def compute_dt(u, dx):
    W = inverse(u)
    density = W[:, 0]
    velocity = W[:, 1]
    pressure = W[:, 2]
    alpha = jnp.sqrt((gamma*pressure)/density)
    smax = jnp.max(jnp.abs(velocity) + alpha)
    dt = cfl * (dx/smax) 
    return dt

def update(u, dt):
    residual = connectivity_residiual(u)
    U_new = (
        u
        - dt
        * residual
        / cell_volume[:, None]
    )
    return U_new

def simulate(U_initial):

    def loop_body(carry, _):
        # Retrieve the current state and time.
        U, current_time = carry

        # Find a stable timestep.
        cfl_dt = compute_dt(U, dx)

        # Do not advance past t = 0.2.
        remaining_time = jnp.maximum(
            0.2 - current_time,
            0.0,
        )

        dt = jnp.minimum(
            cfl_dt,
            remaining_time,
        )

        # Advance the conserved state.
        U_new = update(U, dt)
        new_time = current_time + dt

        # Values passed into the next iteration.
        new_carry = (U_new, new_time)

        # Values saved in the history.
        saved_output = (U_new, new_time)

        return new_carry, saved_output

    initial_carry = (
        U_initial,
        jnp.array(0.0),
    )

    final_carry, saved_history = jax.lax.scan(
        loop_body,
        initial_carry,
        xs=None,
        length=number_of_steps,
    )

    final_U, final_time = final_carry
    U_history, time_history = saved_history

    return final_U, final_time, U_history, time_history

def main(u):
    compiled_simulate = jax.jit(simulate)
    final_U, final_time, U_history, time_history = (
        compiled_simulate(U_initial)
    )
    final_U.block_until_ready()

    print("Final time:", final_time)
    print("History shape:", U_history.shape)
    W_history = inverse(U_history)
    W_final = inverse(final_U)

    density_history = W_history[..., 0]
    velocity_history = W_history[..., 1]
    pressure_history = W_history[..., 2]

    x_plot = np.asarray(jax.device_get(x))
    W_initial_plot = np.asarray(jax.device_get(W))
    W_final_plot = np.asarray(jax.device_get(W_final))
    
    density_plot = np.asarray(jax.device_get(density_history))
    velocity_plot = np.asarray(jax.device_get(velocity_history))
    pressure_plot = np.asarray(jax.device_get(pressure_history))
    times_plot = np.asarray(jax.device_get(time_history))
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 9),
        sharex=True,
    )

    labels = ["Density", "Velocity", "Pressure"]

    initial_fields = [
        W_initial_plot[:, 0],
        W_initial_plot[:, 1],
        W_initial_plot[:, 2],
    ]

    final_fields = [
        W_final_plot[:, 0],
        W_final_plot[:, 1],
        W_final_plot[:, 2],
    ]

    for ax, label, initial, final in zip(
        axes,
        labels,
        initial_fields,
        final_fields,
    ):
        ax.plot(
            x_plot,
            initial,
            "--",
            color="gray",
            label="Initial",
        )

        ax.plot(
            x_plot,
            final,
            color="royalblue",
            label="t = 0.2",
        )

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylim(0.0, 1.1)
    axes[1].set_ylim(-0.1, 1.1)
    axes[2].set_ylim(0.0, 1.1)

    axes[-1].set_xlabel("x")
    fig.suptitle("Sod Shock Tube")
    plt.tight_layout()
    plt.show()



main(U_initial)