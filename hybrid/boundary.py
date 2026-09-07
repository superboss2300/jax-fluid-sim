"""Boundary-state construction for the hybrid compressible operator."""

import jax.numpy as jnp


class FarfieldWallBoundary:
    """Farfield inlet/outlet with inviscid slip walls.

    This is appropriate for the Euler part of a supersonic wind-tunnel test.
    The viscous operator supplies its own no-slip/thermal wall traces later.
    """

    def __init__(
        self,
        solver,
        farfield_primitive,
        inlet_names=("inlet",),
        outlet_names=("outlet",),
        wall_names=("walls", "sphere_wall"),
    ):
        self.solver = solver
        self.farfield_primitive = jnp.asarray(farfield_primitive)
        if self.farfield_primitive.shape != (solver.layout.number_of_variables,):
            raise ValueError("farfield_primitive has the wrong state size")
        names = solver.mesh.physical_names
        self.inlet_tags = jnp.asarray(
            [names[name] for name in inlet_names if name in names]
        )
        self.outlet_tags = jnp.asarray(
            [names[name] for name in outlet_names if name in names]
        )
        self.wall_tags = jnp.asarray(
            [names[name] for name in wall_names if name in names]
        )

    @staticmethod
    def _matches(tags, choices):
        if choices.size == 0:
            return jnp.zeros(tags.shape, dtype=bool)
        return jnp.any(tags[..., None] == choices, axis=-1)

    def __call__(self, U_inside, normals, tags):
        W_inside = self.solver.conserved_to_primitive(U_inside)
        W_ghost = W_inside

        wall = self._matches(tags, self.wall_tags)[:, None]
        velocity = W_inside[..., 1:4]
        normal_velocity = jnp.sum(velocity * normals, axis=-1)
        reflected_velocity = velocity - 2.0 * normal_velocity[..., None] * normals
        W_wall = W_inside.at[..., 1:4].set(reflected_velocity)
        W_ghost = jnp.where(wall[..., None], W_wall, W_ghost)

        inlet = self._matches(tags, self.inlet_tags)[:, None]
        W_farfield = jnp.broadcast_to(
            self.farfield_primitive, W_inside.shape
        )
        W_ghost = jnp.where(inlet[..., None], W_farfield, W_ghost)

        # Outlet and unknown tags are transmissive: W_ghost = W_inside.
        return self.solver.primitive_to_conserved(W_ghost)

    def viscous_ghost(self, U_inside, normals, tags):
        """Ghost [u,v,w,T] for stationary adiabatic no-slip walls."""
        variables = self.solver.viscous_variables(U_inside)
        ghost = variables
        wall = self._matches(tags, self.wall_tags)[:, None]
        wall_variables = variables.at[..., 0:3].set(-variables[..., 0:3])
        ghost = jnp.where(wall[..., None], wall_variables, ghost)

        inlet = self._matches(tags, self.inlet_tags)[:, None]
        farfield_U = self.solver.primitive_to_conserved(
            jnp.broadcast_to(self.farfield_primitive, U_inside.shape)
        )
        farfield_variables = self.solver.viscous_variables(farfield_U)
        ghost = jnp.where(inlet[..., None], farfield_variables, ghost)
        return ghost
