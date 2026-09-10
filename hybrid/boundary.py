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
        wall_names=("walls", "sphere_wall", "cylinder_wall", "airfoil_wall", "symmetry"),
        no_slip_names=("sphere_wall", "cylinder_wall", "airfoil_wall"),
        slip_names=("walls", "symmetry"),
        wall_temperature: float | None = None,
    ):
        self.solver = solver
        self.wall_temperature = (
            float(wall_temperature) if wall_temperature is not None else None
        )
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
        self.no_slip_tags = jnp.asarray(
            [names[name] for name in no_slip_names if name in names]
        )
        self.slip_tags = jnp.asarray(
            [names[name] for name in slip_names if name in names]
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
        """Ghost [u,v,w,T] for no-slip solid walls and slip/symmetry planes."""
        variables = self.solver.viscous_variables(U_inside)
        ghost = variables

        # No-slip wall: u_ghost = -u_inside, T_ghost = 2*T_wall - T_inside (or adiabatic T_inside)
        no_slip = self._matches(tags, self.no_slip_tags)[:, None]
        no_slip_variables = variables.at[..., 0:3].set(-variables[..., 0:3])
        if self.wall_temperature is not None:
            T_inside = variables[..., 3]
            T_ghost_val = 2.0 * self.wall_temperature - T_inside
            no_slip_variables = no_slip_variables.at[..., 3].set(T_ghost_val)
        ghost = jnp.where(no_slip[..., None], no_slip_variables, ghost)

        # Symmetry / slip wall: normal velocity reflected, tangential preserved
        slip = self._matches(tags, self.slip_tags)[:, None]
        velocity = variables[..., 0:3]
        normal_velocity = jnp.sum(velocity * normals, axis=-1)
        reflected_velocity = velocity - 2.0 * normal_velocity[..., None] * normals
        slip_variables = variables.at[..., 0:3].set(reflected_velocity)
        ghost = jnp.where(slip[..., None], slip_variables, ghost)

        inlet = self._matches(tags, self.inlet_tags)[:, None]
        farfield_U = self.solver.primitive_to_conserved(
            jnp.broadcast_to(self.farfield_primitive, U_inside.shape)
        )
        farfield_variables = self.solver.viscous_variables(farfield_U)
        ghost = jnp.where(inlet[..., None], farfield_variables, ghost)
        return ghost
