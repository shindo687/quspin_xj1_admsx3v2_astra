"""Fixed-grid differentiation for dynamic pure-state Hamiltonians.

The adapter in this module intentionally has a small, explicit contract.  A
Hamiltonian callback returns a dense matrix ``H(t, controls)`` and derivative
metadata returns ``{name: dH/dname}`` at the same point.  This is enough to
differentiate QuSpin dynamic drives without inspecting private callback
tuples.  The integrator is fixed-step RK4; no adaptive solver or numerical
differencing is used.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .rules import ad, _array, _input_gradient, _matrix, _unsupported


@dataclass(frozen=True)
class StateObjective:
    """A scalar trajectory objective with an analytic state gradient.

    ``value`` and ``gradient`` receive ``(time, state)``.  ``mode`` is
    ``"final"`` for the last state, or ``"integral"`` for trapezoidal
    integration over the supplied grid.  The gradient callback must return
    the real-linear gradient satisfying ``dL = Re(vdot(gradient, dstate))``.
    """

    value: Callable[[float, np.ndarray], float]
    gradient: Callable[[float, np.ndarray], np.ndarray]
    mode: str = "final"

    def __post_init__(self):
        if self.mode not in ("final", "integral"):
            raise ValueError("StateObjective mode must be 'final' or 'integral'")


def _validate_inputs(psi0, times, controls, derivatives, checkpoint_interval):
    psi = np.asarray(psi0)
    times = np.asarray(times)
    if psi.ndim not in (1, 2):
        raise TypeError("fixed_grid_trajectory requires a vector or state batch")
    if times.ndim != 1 or times.size == 0 or np.iscomplexobj(times):
        raise TypeError("times must be a non-empty real 1-D fixed grid")
    if times.size > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    if controls is None:
        controls = {}
    if not isinstance(controls, Mapping) or any(not isinstance(k, str) for k in controls):
        raise TypeError("controls must be a mapping from names to scalar values")
    controls = dict(controls)
    for name, val in controls.items():
        arr = np.asarray(val)
        if arr.ndim != 0 or not np.issubdtype(arr.dtype, np.number) or not np.all(np.isfinite(arr)):
            raise TypeError(f"control {name!r} must be a finite scalar")
    if checkpoint_interval is None:
        checkpoint_interval = max(1, int(np.sqrt(max(1, times.size - 1))))
    if isinstance(checkpoint_interval, bool) or not isinstance(checkpoint_interval, (int, np.integer)) or checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be a positive integer")
    return psi, times.astype(float), controls, int(checkpoint_interval)


def _matrix_at(hamiltonian, time, controls):
    if callable(hamiltonian):
        try:
            raw = hamiltonian(float(time), controls)
        except TypeError:
            raw = hamiltonian(float(time), **controls)
    elif hasattr(hamiltonian, "tocsr"):
        raw = hamiltonian.tocsr(time=float(time)).toarray()
    elif hasattr(hamiltonian, "toarray"):
        raw = hamiltonian.toarray(time=float(time))
    else:
        raw = hamiltonian
    matrix = _matrix(raw, name="Hamiltonian")
    if matrix.shape[0] != matrix.shape[1]:
        raise TypeError("Hamiltonian must be square")
    return matrix


def _derivatives_at(metadata, time, controls):
    raw = metadata(float(time), controls) if callable(metadata) else metadata
    if not isinstance(raw, Mapping):
        raise TypeError("hamiltonian_derivatives must be a mapping or mapping callback")
    result = {}
    for name, value in raw.items():
        value = value(float(time), controls) if callable(value) else value
        result[name] = _matrix(value, name=f"dH[{name!r}]")
    return result


def _rhs(matrix, state):
    return -1j * matrix.dot(state)


def _rk4_step(hamiltonian, state, time, dt, controls, derivatives=None, dstate=None, dcontrols=None):
    h1 = _matrix_at(hamiltonian, time, controls)
    hm = _matrix_at(hamiltonian, time + dt / 2, controls)
    h2 = _matrix_at(hamiltonian, time + dt, controls)
    if h1.shape != (state.shape[0], state.shape[0]):
        raise ValueError("Hamiltonian and state dimensions are incompatible")
    k1 = _rhs(h1, state)
    k2 = _rhs(hm, state + dt * k1 / 2)
    k3 = _rhs(hm, state + dt * k2 / 2)
    k4 = _rhs(h2, state + dt * k3)
    out = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    if dstate is None:
        return out, None
    dcontrols = {} if dcontrols is None else dcontrols
    dh = {}
    for at in (time, time + dt / 2, time + dt):
        available = _derivatives_at(derivatives, at, controls)
        combined = np.zeros_like(h1, dtype=np.result_type(h1, np.complex128))
        for name, direction in dcontrols.items():
            if name not in available:
                raise TypeError(f"missing derivative metadata for control {name!r}")
            combined = combined + direction * available[name]
        dh[at] = combined
    def drhs(matrix, dmatrix, state_, dstate_):
        return -1j * (matrix.dot(dstate_) + dmatrix.dot(state_))
    q1 = drhs(h1, dh[time], state, dstate)
    q2 = drhs(hm, dh[time + dt / 2], state + dt*k1/2, dstate + dt*q1/2)
    q3 = drhs(hm, dh[time + dt / 2], state + dt*k2/2, dstate + dt*q2/2)
    q4 = drhs(h2, dh[time + dt], state + dt*k3, dstate + dt*q3)
    dout = dstate + dt * (q1 + 2*q2 + 2*q3 + q4) / 6
    return out, dout


def _forward(hamiltonian, psi, times, controls, derivatives, checkpoint_interval):
    states = np.empty((times.size,) + psi.shape, dtype=np.result_type(psi, np.complex128))
    states[0] = psi
    checkpoints = {0: states[0].copy()}
    for i, dt in enumerate(np.diff(times)):
        states[i + 1], _ = _rk4_step(hamiltonian, states[i], times[i], dt, controls, derivatives=None)
        if (i + 1) % checkpoint_interval == 0 or i + 1 == times.size - 1:
            checkpoints[i + 1] = states[i + 1].copy()
    return states, checkpoints


def _objective_value_and_gradient(objective, states, times, objective_gradient=None):
    if objective is None:
        return None, None
    if callable(objective) and not isinstance(objective, StateObjective):
        if objective_gradient is None or not callable(objective_gradient):
            raise TypeError("objective_gradient metadata is required for callable objectives")
        output = np.moveaxis(states, 0, -1)
        value = objective(output)
        gradient = np.asarray(objective_gradient(output))
        if gradient.shape != output.shape:
            raise ValueError("objective_gradient must match trajectory output shape")
        return value, np.moveaxis(gradient, -1, 0)
    if not isinstance(objective, StateObjective):
        raise TypeError("objective must be StateObjective or callable with objective_gradient")
    values = [objective.value(float(t), states[i]) for i, t in enumerate(times)]
    grads = np.asarray([objective.gradient(float(t), states[i]) for i, t in enumerate(times)])
    if grads.shape != states.shape:
        raise ValueError("objective gradient must match every trajectory state")
    if objective.mode == "final":
        return values[-1], np.array([*([np.zeros_like(states[i]) for i in range(times.size - 1)]), grads[-1]])
    weights = np.zeros(times.size)
    if times.size > 1:
        weights[0] = (times[1] - times[0]) / 2
        weights[-1] = (times[-1] - times[-2]) / 2
        weights[1:-1] = (times[2:] - times[:-2]) / 2
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(values, times)), grads * weights.reshape((-1,) + (1,) * (states.ndim - 1))


def fixed_grid_trajectory(hamiltonian, psi0, times, controls=None,
                          hamiltonian_derivatives=None, objective=None,
                          objective_gradient=None, checkpoint_interval=None,
                          continuous=True, topology_changes=False):
    """Return ``(Ns, Ntime)`` (or ``(Ns, Nstate, Ntime)``) fixed-grid states.

    Callbacks marked ``discontinuous`` or ``topology_changes`` are rejected;
    ``continuous`` is an explicit assertion required by this bounded rule.
    ``objective_gradient`` is retained for compatibility with callable
    objectives but arbitrary callables are intentionally unsupported because
    their derivative contract cannot be inspected safely.
    """
    if not continuous or topology_changes or getattr(hamiltonian, "discontinuous", False):
        raise ValueError("fixed_grid_trajectory requires a continuous fixed-topology callback")
    psi, grid, controls, interval = _validate_inputs(psi0, times, controls, hamiltonian_derivatives, checkpoint_interval)
    states, _ = _forward(hamiltonian, psi, grid, controls, hamiltonian_derivatives, interval)
    output = np.moveaxis(states, 0, -1)
    output_shape = output.shape
    if objective is not None:
        return _objective_value_and_gradient(objective, states, grid, objective_gradient)[0]
    return output


def _linearized(hamiltonian, psi, times, controls, derivatives, interval, dpsi, dcontrols):
    tangent = np.empty((times.size,) + psi.shape, dtype=np.result_type(psi, np.complex128))
    tangent[0] = 0 if dpsi is None else np.asarray(dpsi)
    state = np.asarray(psi, dtype=np.result_type(psi, np.complex128))
    for i, dt in enumerate(np.diff(times)):
        state, tangent[i + 1] = _rk4_step(
            hamiltonian, state, times[i], dt, controls, derivatives,
            dstate=tangent[i], dcontrols=dcontrols,
        )
    return tangent


def _reverse_step(hamiltonian, state, time, dt, controls, derivatives, cotangent, names):
    """Adjoint of one RK4 step; returns input cotangent and control gradients."""
    h1 = _matrix_at(hamiltonian, time, controls)
    hm = _matrix_at(hamiltonian, time + dt / 2, controls)
    h2 = _matrix_at(hamiltonian, time + dt, controls)
    k1 = _rhs(h1, state); y2 = state + dt * k1 / 2
    k2 = _rhs(hm, y2); y3 = state + dt * k2 / 2
    k3 = _rhs(hm, y3); y4 = state + dt * k3
    k4 = _rhs(h2, y4)
    lam = np.asarray(cotangent)
    grad = {name: 0.0 for name in names}
    def add_control(at, stage, bar, weight):
        if not names: return
        available = _derivatives_at(derivatives, at, controls)
        for name in names:
            if name not in available:
                raise TypeError(f"missing derivative metadata for control {name!r}")
            grad[name] += weight * np.real(np.vdot(bar, -1j * available[name].dot(stage)))
    bk1 = dt * lam / 6; bk2 = dt * lam / 3; bk3 = dt * lam / 3; bk4 = dt * lam / 6
    bx = lam.copy()
    by4 = 1j * h2.conj().T.dot(bk4); bx += by4; bk3 += dt * by4
    add_control(time + dt, y4, bk4, 1.0)
    by3 = 1j * hm.conj().T.dot(bk3); bx += by3; bk2 += dt * by3 / 2
    add_control(time + dt / 2, y3, bk3, 1.0)
    by2 = 1j * hm.conj().T.dot(bk2); bx += by2; bk1 += dt * by2 / 2
    add_control(time + dt / 2, y2, bk2, 1.0)
    bx += 1j * h1.conj().T.dot(bk1)
    add_control(time, state, bk1, 1.0)
    return bx, grad


def _reverse(hamiltonian, psi, times, controls, derivatives, interval, cotangent):
    """Checkpointed reverse pass; only one checkpoint block is replayed at once."""
    names = tuple(controls)
    checkpoints = {0: np.asarray(psi, dtype=np.result_type(psi, np.complex128))}
    state = checkpoints[0]
    for i, dt in enumerate(np.diff(times)):
        state, _ = _rk4_step(hamiltonian, state, times[i], dt, controls)
        if (i + 1) % interval == 0 or i + 1 == times.size - 1:
            checkpoints[i + 1] = state.copy()
    lam = np.moveaxis(np.asarray(cotangent), -1, 0).copy()
    if lam.shape[0] != times.size:
        raise ValueError("trajectory cotangent must match output shape")
    input_grad = np.zeros_like(state, dtype=np.result_type(state, np.complex128))
    grads = {name: 0.0 for name in names}
    ends = sorted(checkpoints)
    for end in reversed(ends[1:]):
        start = max(k for k in ends if k < end)
        local = [checkpoints[start]]
        for i in range(start, end):
            nxt, _ = _rk4_step(hamiltonian, local[-1], times[i], times[i+1]-times[i], controls)
            local.append(nxt)
        for i in range(end - 1, start - 1, -1):
            input_grad, part = _reverse_step(
                hamiltonian, local[i-start], times[i], times[i+1]-times[i],
                controls, derivatives, input_grad + lam[i+1], names,
            )
            for name in names: grads[name] += part[name]
    input_grad += lam[0]
    return input_grad, grads


@ad.rules.jvp_for(fixed_grid_trajectory)
def _fixed_jvp(tangents, hamiltonian, psi0, times, controls=None,
               hamiltonian_derivatives=None, objective=None, objective_gradient=None,
               checkpoint_interval=None, continuous=True, topology_changes=False):
    _unsupported(fixed_grid_trajectory, tangents, ("psi0", "controls"))
    if not continuous or topology_changes or getattr(hamiltonian, "discontinuous", False):
        raise ValueError("fixed_grid_trajectory requires a continuous fixed-topology callback")
    psi, grid, controls, interval = _validate_inputs(psi0, times, controls, hamiltonian_derivatives, checkpoint_interval)
    dpsi = tangents.get("psi0", ad.ZERO); dc = tangents.get("controls", ad.ZERO)
    if controls and hamiltonian_derivatives is None:
        raise TypeError("hamiltonian_derivatives metadata is required for controls")
    dc = {} if dc is ad.ZERO else dict(dc)
    tangent = _linearized(hamiltonian, psi, grid, controls, hamiltonian_derivatives, interval,
                          None if dpsi is ad.ZERO else _array(dpsi, name="dpsi0"), dc)
    states, _ = _forward(hamiltonian, psi, grid, controls, hamiltonian_derivatives, interval)
    out = np.moveaxis(states, 0, -1); dout = np.moveaxis(tangent, 0, -1)
    if objective is None: return out, dout
    value, og = _objective_value_and_gradient(objective, states, grid, objective_gradient)
    if dpsi is ad.ZERO and dc is ad.ZERO: return value, ad.ZERO
    return value, np.real(np.vdot(og, tangent))


@ad.rules.vjp_for(fixed_grid_trajectory)
def _fixed_vjp(wrt, hamiltonian, psi0, times, controls=None,
               hamiltonian_derivatives=None, objective=None, objective_gradient=None,
               checkpoint_interval=None, continuous=True, topology_changes=False):
    _unsupported(fixed_grid_trajectory, wrt, ("psi0", "controls"))
    if not continuous or topology_changes or getattr(hamiltonian, "discontinuous", False):
        raise ValueError("fixed_grid_trajectory requires a continuous fixed-topology callback")
    psi, grid, controls, interval = _validate_inputs(psi0, times, controls, hamiltonian_derivatives, checkpoint_interval)
    if controls and hamiltonian_derivatives is None:
        raise TypeError("hamiltonian_derivatives metadata is required for controls")
    states, _ = _forward(hamiltonian, psi, grid, controls, hamiltonian_derivatives, interval)
    output = np.moveaxis(states, 0, -1)
    output_shape = output.shape
    value = output
    objective_cotangent = None
    if objective is not None: value, objective_cotangent = _objective_value_and_gradient(objective, states, grid, objective_gradient)
    def pullback(cotangent):
        if cotangent is ad.ZERO: return dict.fromkeys(wrt, ad.ZERO)
        if objective is not None: g = np.asarray(cotangent) * np.moveaxis(objective_cotangent, 0, -1)
        else:
            g = np.asarray(cotangent)
            if g.shape != output_shape: raise ValueError("trajectory cotangent must match output shape")
        psi_grad, control_grad = _reverse(hamiltonian, psi, grid, controls, hamiltonian_derivatives, interval, g)
        result = {}
        if "psi0" in wrt: result["psi0"] = _input_gradient(np.asarray(psi0), psi_grad)
        if "controls" in wrt: result["controls"] = control_grad
        return result
    return value, pullback


dynamic_trajectory = fixed_grid_trajectory
evolve_fixed_grid = fixed_grid_trajectory
evolve = fixed_grid_trajectory
