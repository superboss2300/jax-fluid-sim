"""DG timestep estimate and SSP-RK3 integration for type-blocked states."""

import jax
import jax.numpy as jnp
import numpy as np


class HybridSSPRK3:
    def __init__(
        self,
        operator,
        cfl=0.15,
        limiter=None,
        density_floor=1.0e-7,
        pressure_floor=1.0e-7,
        maximum_rejections=16,
    ):
        self.operator = operator
        self.solver = operator.solver
        self.cfl = float(cfl)
        self.limiter = limiter
        self.density_floor = float(density_floor)
        self.pressure_floor = float(pressure_floor)
        self.maximum_rejections = int(maximum_rejections)
        self.characteristic_lengths = self._characteristic_lengths()

    def _characteristic_lengths(self):
        """Use 2V / total face area for each physical element."""
        area_sums = {
            block.element_type: jnp.zeros(len(block.connectivity))
            for block in self.solver.mesh.element_blocks
        }
        for face_type, faces in self.solver.mesh.face_blocks.items():
            face_areas = jnp.sum(
                self.operator.geometry.face_metrics[
                    face_type
                ].integration_weights,
                axis=1,
            )
            for block_id, block in enumerate(self.solver.mesh.element_blocks):
                owner_indices = np.flatnonzero(
                    faces.owner_block == block_id
                ).astype(np.int32)
                if len(owner_indices):
                    cells = jnp.asarray(faces.owner_local[owner_indices])
                    area_sums[block.element_type] = area_sums[
                        block.element_type
                    ].at[cells].add(face_areas[jnp.asarray(owner_indices)])
                neighbor_indices = np.flatnonzero(
                    faces.neighbor_block == block_id
                ).astype(np.int32)
                if len(neighbor_indices):
                    cells = jnp.asarray(faces.neighbor_local[neighbor_indices])
                    area_sums[block.element_type] = area_sums[
                        block.element_type
                    ].at[cells].add(face_areas[jnp.asarray(neighbor_indices)])

        lengths = {}
        for block_id, block in enumerate(self.solver.mesh.element_blocks):
            volumes = jnp.sum(
                self.operator.geometry.element_metrics[
                    block_id
                ].integration_weights,
                axis=1,
            )
            lengths[block.element_type] = (
                2.0 * volumes / area_sums[block.element_type]
            )
        return lengths

    def stable_dt(self, states, boundary_state=None):
        """Combined acoustic/viscous CFL using the resolved WALE gradient.

        The same raw polynomial gradient is used to determine eddy viscosity
        in the viscous operator.  BR2 lifting remains part of the physical
        stress gradient, but is not treated as resolved turbulence.
        """
        degree_factor = 2 * self.solver.polynomial_degree + 1
        maximum_rates = []
        for block_id, block in enumerate(self.solver.mesh.element_blocks):
            element_type = block.element_type
            U = states[element_type]
            W = self.solver.conserved_to_primitive(U)
            density = jnp.maximum(W[..., 0], 1.0e-14)
            pressure = jnp.maximum(W[..., 4], 1.0e-14)
            speed = (
                jnp.linalg.norm(W[..., 1:4], axis=-1)
                + jnp.sqrt(self.solver.gamma * pressure / density)
            )
            element_speed = jnp.max(speed, axis=1)
            length = self.characteristic_lengths[element_type]
            rate = degree_factor * element_speed / length

            if self.solver.dynamic_viscosity > 0.0 or self.solver.use_wale:
                minimum_density = jnp.min(density, axis=1)
                effective_viscosity = jnp.full_like(
                    minimum_density, self.solver.dynamic_viscosity
                )
                effective_conductivity = jnp.full_like(
                    minimum_density, self.solver.thermal_conductivity
                )
                if self.solver.use_wale:
                    reference = self.operator.geometry.reference_elements[
                        element_type
                    ]
                    metrics = self.operator.geometry.element_metrics[block_id]
                    variables = self.solver.viscous_variables(U)
                    gradient = jnp.einsum(
                        "eim,eqid->eqmd",
                        variables,
                        metrics.physical_basis_gradients,
                    )
                    U_quadrature = jnp.einsum(
                        "qi,eiv->eqv",
                        reference.basis(reference.volume_points),
                        U,
                    )
                    width = (
                        jnp.sum(metrics.integration_weights, axis=1)
                        ** (1.0/3.0)
                        / (self.solver.polynomial_degree + 1)
                    )
                    turbulent = self.solver.wale_dynamic_viscosity(
                        U_quadrature, gradient, width[:, None]
                    )
                    turbulent = jnp.max(turbulent, axis=1)
                    effective_viscosity = effective_viscosity + turbulent
                    effective_conductivity = (
                        effective_conductivity
                        + turbulent
                        * self.solver.specific_heat_cp
                        / self.solver.turbulent_prandtl
                    )
                diffusivity = jnp.maximum(
                    (4.0/3.0) * effective_viscosity / minimum_density,
                    effective_conductivity
                    / (minimum_density * self.solver.specific_heat_cp),
                )
                penalty = float(getattr(self.operator, "br2_stabilization", 1.0))
                rate = rate + (
                    2.0 * penalty * degree_factor**2
                    * diffusivity / length**2
                )
            maximum_rates.append(jnp.max(rate))
        return self.cfl / jnp.max(jnp.stack(maximum_rates))

    def local_dt(self, states, cfl=None, boundary_state=None):
        """Compute element-wise local stable time step field for rapid steady-state convergence."""
        cfl_val = self.cfl if cfl is None else cfl
        degree_factor = 2 * self.solver.polynomial_degree + 1
        dts = {}
        for block_id, block in enumerate(self.solver.mesh.element_blocks):
            element_type = block.element_type
            U = states[element_type]
            W = self.solver.conserved_to_primitive(U)
            density = jnp.maximum(W[..., 0], 1.0e-14)
            pressure = jnp.maximum(W[..., 4], 1.0e-14)
            speed = (
                jnp.linalg.norm(W[..., 1:4], axis=-1)
                + jnp.sqrt(self.solver.gamma * pressure / density)
            )
            element_speed = jnp.max(speed, axis=1)
            length = self.characteristic_lengths[element_type]
            rate = degree_factor * element_speed / length

            if self.solver.dynamic_viscosity > 0.0 or self.solver.use_wale:
                minimum_density = jnp.min(density, axis=1)
                effective_viscosity = jnp.full_like(
                    minimum_density, self.solver.dynamic_viscosity
                )
                effective_conductivity = jnp.full_like(
                    minimum_density, self.solver.thermal_conductivity
                )
                if self.solver.use_wale:
                    reference = self.operator.geometry.reference_elements[
                        element_type
                    ]
                    metrics = self.operator.geometry.element_metrics[block_id]
                    variables = self.solver.viscous_variables(U)
                    gradient = jnp.einsum(
                        "eim,eqid->eqmd",
                        variables,
                        metrics.physical_basis_gradients,
                    )
                    U_quadrature = jnp.einsum(
                        "qi,eiv->eqv",
                        reference.basis(reference.volume_points),
                        U,
                    )
                    width = (
                        jnp.sum(metrics.integration_weights, axis=1)
                        ** (1.0/3.0)
                        / (self.solver.polynomial_degree + 1)
                    )
                    turbulent = self.solver.wale_dynamic_viscosity(
                        U_quadrature, gradient, width[:, None]
                    )
                    turbulent = jnp.max(turbulent, axis=1)
                    effective_viscosity = effective_viscosity + turbulent
                    effective_conductivity = (
                        effective_conductivity
                        + turbulent
                        * self.solver.specific_heat_cp
                        / self.solver.turbulent_prandtl
                    )
                diffusivity = jnp.maximum(
                    (4.0/3.0) * effective_viscosity / minimum_density,
                    effective_conductivity
                    / (minimum_density * self.solver.specific_heat_cp),
                )
                penalty = float(getattr(self.operator, "br2_stabilization", 1.0))
                rate = rate + (
                    2.0 * penalty * degree_factor**2
                    * diffusivity / length**2
                )
            dt_elem = cfl_val / jnp.maximum(rate, 1.0e-12)
            dts[element_type] = dt_elem[:, None, None]
        return dts

    @staticmethod
    def _add_scaled(left, scale, right):
        if isinstance(scale, dict):
            return {
                name: left[name] + scale[name] * right[name]
                for name in left
            }
        return {
            name: left[name] + scale * right[name]
            for name in left
        }

    @staticmethod
    def _blend(first_scale, first, second_scale, second):
        return {
            name: first_scale * first[name] + second_scale * second[name]
            for name in first
        }

    def _limit(self, states):
        return states if self.limiter is None else self.limiter(states)

    def _positivity_limiter(self):
        if self.limiter is None:
            return None
        return getattr(self.limiter, "positivity", self.limiter)

    def state_admissibility(self, states):
        """Return finite flag and minima directly at solution nodes."""
        finite = jnp.asarray(True)
        density_minima = []
        pressure_minima = []
        for block in self.solver.mesh.element_blocks:
            U = states[block.element_type]
            finite = finite & jnp.all(jnp.isfinite(U))
            rho = U[..., 0]
            density_minima.append(jnp.min(rho))
            safe_rho = jnp.maximum(rho, 1.0e-14)
            kinetic = 0.5 * jnp.sum(U[..., 1:4] ** 2, axis=-1) / safe_rho
            pressure = (self.solver.gamma - 1.0) * (U[..., 4] - kinetic)
            pressure_minima.append(jnp.min(pressure))
        density_minimum = jnp.min(jnp.stack(density_minima))
        pressure_minimum = jnp.min(jnp.stack(pressure_minima))
        admissible = (
            finite
            & (density_minimum >= self.density_floor)
            & (pressure_minimum >= self.pressure_floor)
        )
        return admissible, density_minimum, pressure_minimum

    def try_step(self, states, dt, boundary_state=None):
        """Attempt SSP-RK3, rejecting immediately after an invalid stage."""
        rhs0 = self.operator.rhs(states, boundary_state)
        stage1 = self._limit(self._add_scaled(states, dt, rhs0))
        valid1, rho1, pressure1 = self.state_admissibility(stage1)

        def after_stage1(stage1_states):
            rhs1 = self.operator.rhs(stage1_states, boundary_state)
            euler1 = self._add_scaled(stage1_states, dt, rhs1)
            stage2 = self._limit(self._blend(0.75, states, 0.25, euler1))
            valid2, rho2, pressure2 = self.state_admissibility(stage2)

            def after_stage2(stage2_states):
                rhs2 = self.operator.rhs(stage2_states, boundary_state)
                euler2 = self._add_scaled(stage2_states, dt, rhs2)
                result = self._limit(
                    self._blend(1.0 / 3.0, states, 2.0 / 3.0, euler2)
                )
                valid3, rho3, pressure3 = self.state_admissibility(result)
                accepted = jax.tree.map(
                    lambda new, old: jnp.where(valid3, new, old), result, states
                )
                return accepted, valid3, rho3, pressure3

            return jax.lax.cond(
                valid2,
                after_stage2,
                lambda _: (states, valid2, rho2, pressure2),
                stage2,
            )

        return jax.lax.cond(
            valid1,
            after_stage1,
            lambda _: (states, valid1, rho1, pressure1),
            stage1,
        )

    def _euler_stage(
        self,
        base_states,
        current_states,
        dt,
        base_weight,
        current_weight,
        boundary_state=None,
    ):
        if hasattr(self.operator, "inviscid_rhs"):
            inviscid = self.operator.inviscid_rhs(current_states, boundary_state)
            viscous = self.operator.viscous_rhs(current_states, boundary_state)
            total = {name: inviscid[name] + viscous[name] for name in inviscid}
        else:
            total = self.operator.rhs(current_states, boundary_state)
            viscous = None
        euler = self._add_scaled(current_states, dt, total)
        unlimited = self._blend(base_weight, base_states, current_weight, euler)
        return euler, unlimited, viscous

    def _blend_positivity_and_check(self, base, recomp, alpha, beta):
        blended = self._blend(alpha, base, beta, recomp)
        result = self.limiter.positivity(blended)
        valid, rho_min, p_min = self.state_admissibility(result)
        return result, valid, rho_min, p_min

    def _mood_stage(
        self,
        base_states,
        current_states,
        euler,
        unlimited,
        viscous,
        dt,
        base_weight,
        current_weight,
        boundary_state=None,
    ):
        euler_recomputed = self.limiter.recompute(
            current_states,
            euler,
            dt,
            boundary_state,
            viscous,
            detection_state=unlimited,
        )
        result = self.limiter.positivity(
            self._blend(base_weight, base_states, current_weight, euler_recomputed)
        )
        valid, rho_min, p_min = self.state_admissibility(result)
        return result, valid, rho_min, p_min

    def rk_stage(
        self,
        base_states,
        current_states,
        dt,
        base_weight,
        current_weight,
        boundary_state=None,
    ):
        mood = self.limiter is not None and hasattr(self.limiter, "recompute")
        if mood and hasattr(self.operator, "inviscid_rhs"):
            inviscid = self.operator.inviscid_rhs(current_states, boundary_state)
            viscous = self.operator.viscous_rhs(current_states, boundary_state)
            total = {name: inviscid[name] + viscous[name] for name in inviscid}
        else:
            total = self.operator.rhs(current_states, boundary_state)
            viscous = None

        euler = self._add_scaled(current_states, dt, total)
        unlimited = self._blend(base_weight, base_states, current_weight, euler)
        if mood:
            euler_recomputed = self.limiter.recompute(
                current_states,
                euler,
                dt,
                boundary_state,
                viscous,
                detection_state=unlimited,
            )
            result = self.limiter.positivity(
                self._blend(base_weight, base_states, current_weight, euler_recomputed)
            )
        else:
            result = self._limit(unlimited)
        valid, rho_min, p_min = self.state_admissibility(result)
        return result, valid, rho_min, p_min

    def checked_step(self, states, dt, boundary_state=None):
        """SSP-RK3 plus admissibility diagnostics for memory-limited GPUs.

        Instead of unrolling all three stages into one monolithic kernel,
        each stage is executed through modular JIT-compiled kernels.
        When subcell MOOD limiting is active, the spatial operator and the
        subcell recompute are compiled into decoupled kernels to prevent
        XLA compiler memory explosions on large meshes.
        """
        mood = self.limiter is not None and hasattr(self.limiter, "recompute")
        dtype = next(iter(states.values())).dtype
        if isinstance(dt, dict):
            step_dt = {k: jnp.asarray(v, dtype=dtype) for k, v in dt.items()}
        else:
            step_dt = jnp.asarray(dt, dtype=dtype)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        three_quarters = jnp.asarray(0.75, dtype=dtype)
        one_quarter = jnp.asarray(0.25, dtype=dtype)
        one_third = jnp.asarray(1.0 / 3.0, dtype=dtype)
        two_thirds = jnp.asarray(2.0 / 3.0, dtype=dtype)

        if mood:
            if (
                not hasattr(self, "_compiled_inviscid")
                or getattr(self, "_compiled_stage_boundary", None) is not boundary_state
            ):
                self._compiled_stage_boundary = boundary_state
                self._compiled_inviscid = jax.jit(
                    lambda curr: (
                        self.operator.inviscid_rhs(curr, boundary_state)
                        if hasattr(self.operator, "inviscid_rhs")
                        else self.operator.rhs(curr, boundary_state)
                    )
                )
                self._compiled_viscous = (
                    jax.jit(lambda curr: self.operator.viscous_rhs(curr, boundary_state))
                    if hasattr(self.operator, "viscous_rhs")
                    else None
                )
                self._compiled_advance = jax.jit(
                    lambda curr, base, inv, visc, s_dt, alpha, beta: (
                        self._add_scaled(
                            curr,
                            s_dt,
                            inv if visc is None else {k: inv[k] + visc[k] for k in inv},
                        ),
                        self._blend(
                            alpha,
                            base,
                            beta,
                            self._add_scaled(
                                curr,
                                s_dt,
                                inv if visc is None else {k: inv[k] + visc[k] for k in inv},
                            ),
                        ),
                    )
                )
                self._compiled_recompute = jax.jit(
                    lambda curr, eul, unlim, visc, s_dt: self.limiter.recompute(
                        curr, eul, s_dt, boundary_state, visc, detection_state=unlim
                    )
                )
                self._compiled_blend_positivity = jax.jit(
                    lambda base, recomp, alpha, beta: self._blend_positivity_and_check(
                        base, recomp, alpha, beta
                    )
                )

            # Stage 1
            inv1 = self._compiled_inviscid(states)
            visc1 = self._compiled_viscous(states) if self._compiled_viscous is not None else None
            euler1, unlim1 = self._compiled_advance(states, states, inv1, visc1, step_dt, zero, one)
            del inv1
            recomp1 = self._compiled_recompute(states, euler1, unlim1, visc1, step_dt)
            del euler1, unlim1, visc1
            stage1, _, _, _ = self._compiled_blend_positivity(states, recomp1, zero, one)
            del recomp1

            # Stage 2
            inv2 = self._compiled_inviscid(stage1)
            visc2 = self._compiled_viscous(stage1) if self._compiled_viscous is not None else None
            euler2, unlim2 = self._compiled_advance(
                stage1, states, inv2, visc2, step_dt, three_quarters, one_quarter
            )
            del inv2
            recomp2 = self._compiled_recompute(stage1, euler2, unlim2, visc2, step_dt)
            del euler2, unlim2, visc2
            stage2, _, _, _ = self._compiled_blend_positivity(
                states, recomp2, three_quarters, one_quarter
            )
            del stage1, recomp2

            # Stage 3
            inv3 = self._compiled_inviscid(stage2)
            visc3 = self._compiled_viscous(stage2) if self._compiled_viscous is not None else None
            euler3, unlim3 = self._compiled_advance(
                stage2, states, inv3, visc3, step_dt, one_third, two_thirds
            )
            del inv3
            recomp3 = self._compiled_recompute(stage2, euler3, unlim3, visc3, step_dt)
            del euler3, unlim3, visc3
            result, valid, rho_min, p_min = self._compiled_blend_positivity(
                states, recomp3, one_third, two_thirds
            )
            del stage2, recomp3

            return (
                result,
                valid,
                rho_min,
                p_min,
            )

        if (
            not hasattr(self, "_compiled_stage")
            or getattr(self, "_compiled_stage_boundary", None) is not boundary_state
        ):
            self._compiled_stage_boundary = boundary_state
            self._compiled_stage = jax.jit(
                lambda base, curr, step_dt, alpha, beta: self.rk_stage(
                    base, curr, step_dt, alpha, beta, boundary_state
                )
            )

        stage1, valid1, rho1, p1 = self._compiled_stage(
            states, states, step_dt, zero, one
        )
        stage2, valid2, rho2, p2 = self._compiled_stage(
            states, stage1, step_dt, three_quarters, one_quarter
        )
        del stage1
        result, valid3, rho3, p3 = self._compiled_stage(
            states, stage2, step_dt, one_third, two_thirds
        )
        del stage2
        return (
            result,
            valid1 & valid2 & valid3,
            jnp.minimum(rho1, jnp.minimum(rho2, rho3)),
            jnp.minimum(p1, jnp.minimum(p2, p3)),
        )

    def positivity_or_limit(self, states):
        positivity = self._positivity_limiter()
        return states if positivity is None else positivity(states)

    def advance_interval(self, states, duration, boundary_state=None, max_steps=100000):
        """Advance a time interval entirely on device with adaptive retries.

        Keeping the CFL reduction, RK stages, and retry decisions inside one
        compiled loop avoids a GPU synchronization and Python dispatch for
        every physical timestep.
        """
        dtype = next(iter(states.values())).dtype
        duration = jnp.asarray(duration, dtype=dtype)
        time_tolerance = (
            32.0
            * jnp.finfo(dtype).eps
            * jnp.maximum(jnp.abs(duration), jnp.asarray(1.0e-12, dtype=dtype))
        )
        initial = (
            states,
            jnp.asarray(0.0, dtype=dtype),  # elapsed physical time
            jnp.asarray(0, dtype=jnp.int32),  # accepted steps
            jnp.asarray(0, dtype=jnp.int32),  # rejected attempts
            jnp.asarray(0, dtype=jnp.int32),  # consecutive rejections
            jnp.asarray(1.0, dtype=dtype),  # CFL scale after a rejection
            jnp.asarray(jnp.inf, dtype=dtype),  # minimum accepted dt
        )

        def condition(carry):
            _, elapsed, accepted, _, consecutive, _, _ = carry
            return (
                ((duration - elapsed) > time_tolerance)
                & (accepted < max_steps)
                & (consecutive < self.maximum_rejections)
            )

        def body(carry):
            state, elapsed, accepted, rejected, consecutive, scale, min_dt = carry
            cfl_dt = self.stable_dt(state, boundary_state)
            dt = jnp.minimum(scale * cfl_dt, duration - elapsed)
            candidate, valid, _, _ = self.try_step(state, dt, boundary_state)
            next_elapsed = jnp.where(valid, elapsed + dt, elapsed)
            next_accepted = accepted + valid.astype(jnp.int32)
            next_rejected = rejected + (~valid).astype(jnp.int32)
            next_consecutive = jnp.where(valid, 0, consecutive + 1)
            next_scale = jnp.where(
                valid, jnp.minimum(1.0, 1.25 * scale), 0.5 * scale
            )
            next_min_dt = jnp.where(valid, jnp.minimum(min_dt, dt), min_dt)
            return (
                candidate,
                next_elapsed,
                next_accepted,
                next_rejected,
                next_consecutive,
                next_scale,
                next_min_dt,
            )

        result = jax.lax.while_loop(condition, body, initial)
        final_state, elapsed, accepted, rejected, consecutive, _, min_dt = result
        valid, rho_min, pressure_min = self.state_admissibility(final_state)
        completed = ((duration - elapsed) <= time_tolerance) & valid
        return (
            final_state,
            elapsed,
            accepted,
            rejected,
            consecutive,
            min_dt,
            rho_min,
            pressure_min,
            completed,
        )

    def step(self, states, dt, boundary_state=None):
        """One third-order strong-stability-preserving Runge--Kutta step."""
        rhs0 = self.operator.rhs(states, boundary_state)
        stage1 = self._limit(self._add_scaled(states, dt, rhs0))

        rhs1 = self.operator.rhs(stage1, boundary_state)
        euler1 = self._add_scaled(stage1, dt, rhs1)
        stage2 = self._limit(self._blend(0.75, states, 0.25, euler1))

        rhs2 = self.operator.rhs(stage2, boundary_state)
        euler2 = self._add_scaled(stage2, dt, rhs2)
        return self._limit(self._blend(1.0/3.0, states, 2.0/3.0, euler2))
