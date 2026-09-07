import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

class dg_solver:
    def __init__(self):
        self.number_of_elements = 40
        self.x_min = 0.0
        self.x_max = 1.0

        self.velocity = 1.0
        self.cfl = 0.2

        self.dx = (
            self.x_max - self.x_min
        ) / self.number_of_elements

        self.reference_nodes = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        self.reference_basis_gradients = jnp.array([
            [-1.0, -1.0, -1.0],
            [ 1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  0.0,  1.0],
        ])

        self.x_nodes = self.solution_nodes()

        edge_vectors = (
            self.solution_nodes[:, 1:, :]
            - self.solution_nodes[:, :1, :]
        )

        self.jacobian = jnp.swapaxes(
            edge_vectors,
            1,
            2,
        )

        self.inverse_jacobian = jnp.linalg.inv(
            self.jacobian
        )

        self.determinant = jnp.linalg.det(
            self.jacobian
        )

        self.element_volumes = (
            jnp.abs(self.determinant) / 6.0
        )

        self.physical_basis_gradients = jnp.einsum(
            "ir,erx->eix",
            self.reference_basis_gradients,
            self.inverse_jacobian,
        )

    def solution_nodes(self):
        reference_nodes = jnp.array([
            -1.0,
             1.0,
        ])

        element_left = (
            self.x_min
            + jnp.arange(
                self.number_of_elements
            ) * self.dx
        )

        return (
            element_left[:, None]
            + 0.5
            * self.dx
            * (
                reference_nodes[None, :]
                + 1.0
            )
        )

    def initial_condition(self, x):
        return jnp.sin(
            2.0 * jnp.pi * x
        )

solver = dg_solver()
u_initial = solver.initial_condition(
    solver.x_nodes
)

print("State shape:", u_initial.shape)