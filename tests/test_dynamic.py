"""Analytic fixed-grid drive rules, independent of a QuSpin installation."""
import numpy as np
import pytest

import quspin_ad as qa
import chainrules as ad


X = np.array([[0., 1.], [1., 0.]], dtype=complex)
Z = np.diag([1., -1.]).astype(complex)


def hamiltonian(t, controls):
    return controls["amplitude"] * np.sin(controls["omega"] * t) * X + .2 * Z


def derivatives(t, controls):
    return {
        "amplitude": np.sin(controls["omega"] * t) * X,
        "omega": controls["amplitude"] * t * np.cos(controls["omega"] * t) * X,
    }


def test_dynamic_coefficients_central_fd_and_duality():
    psi = np.array([1., .1j]); times = np.linspace(0., 1., 65)
    controls = {"amplitude": .7, "omega": 1.3}
    rng = np.random.default_rng(3)
    cotangent = rng.normal(size=(2, len(times))) + 1j*rng.normal(size=(2, len(times)))
    value, pullback = ad.vjp(qa.fixed_grid_trajectory, hamiltonian, psi, times,
        controls, derivatives, wrt="controls", checkpoint_interval=8)
    gradients = pullback(cotangent)["controls"]
    assert value.shape == (2, len(times))
    for name in controls:
        _, tangent = ad.jvp(qa.fixed_grid_trajectory, hamiltonian, psi, times,
            controls, derivatives, tangents={"controls": {name: 1.}})
        for eps in (1e-4, 1e-5, 1e-6):
            plus = dict(controls); minus = dict(controls)
            plus[name] += eps; minus[name] -= eps
            oracle = (qa.fixed_grid_trajectory(hamiltonian, psi, times, plus)
                      - qa.fixed_grid_trajectory(hamiltonian, psi, times, minus))/(2*eps)
            np.testing.assert_allclose(tangent, oracle, atol=2e-8, rtol=2e-6)
        np.testing.assert_allclose(gradients[name], np.real(np.vdot(cotangent, tangent)), atol=1e-10)


def test_complex_control_vjp_jvp_real_linear_duality():
    psi = np.array([1., .1j]); times = np.linspace(0., 1., 33)
    controls = {"amplitude": .7 + .2j, "omega": 1.3}
    rng = np.random.default_rng(17)
    cotangent = rng.normal(size=(2, len(times))) + 1j * rng.normal(size=(2, len(times)))
    _, pullback = ad.vjp(qa.fixed_grid_trajectory, hamiltonian, psi, times,
                          controls, derivatives, wrt="controls", checkpoint_interval=7)
    gradient = pullback(cotangent)["controls"]["amplitude"]
    assert np.iscomplexobj(gradient)
    for da in (1.0, 1.0j):
        _, tangent = ad.jvp(qa.fixed_grid_trajectory, hamiltonian, psi, times,
                            controls, derivatives,
                            tangents={"controls": {"amplitude": da}})
        np.testing.assert_allclose(
            np.real(np.conj(gradient) * da),
            np.real(np.vdot(cotangent, tangent)),
            atol=1e-10,
        )


def test_scalar_objective_and_initial_state_direction():
    psi = np.array([1., 0.]); times = np.linspace(0., 1., 33)
    controls = {"amplitude": .7, "omega": 1.3}
    objective = qa.StateObjective(
        lambda t, s: float(abs(s[1])**2),
        lambda t, s: np.array([0, 2*s[1]]),
        mode="final",
    )
    value, tangent = ad.jvp(qa.fixed_grid_trajectory, hamiltonian, psi, times,
        controls, derivatives, objective,
        tangents={"controls": {"amplitude": 1.}})
    same, pullback = ad.vjp(qa.fixed_grid_trajectory, hamiltonian, psi, times,
        controls, derivatives, objective, wrt="controls")
    assert value == same
    np.testing.assert_allclose(tangent, pullback(1.)["controls"]["amplitude"])
    _, dy = ad.jvp(qa.fixed_grid_trajectory, hamiltonian, psi, times, controls,
        derivatives, tangents={"psi0": np.array([.2, -.3])})
    assert np.isfinite(dy).all()


def test_initial_state_tangent_needs_no_control_metadata():
    h = lambda t, controls: X
    psi = np.array([1., 0.], complex)
    dpsi = np.array([0., 1.], complex)
    value, tangent = ad.jvp(
        qa.fixed_grid_trajectory, h, psi, np.linspace(0., 1., 5),
        tangents={"psi0": dpsi},
    )
    assert value.shape == tangent.shape == (2, 5)
    assert np.linalg.norm(tangent) > 0


def test_checkpoint_forward_does_not_store_trajectory():
    from quspin_ad import dynamics
    psi = np.array([1., 0.], complex)
    times = np.linspace(0., 1., 101)
    trajectory, checkpoints = dynamics._forward(
        lambda t, controls: X, psi, times, {}, None, 10,
        store_trajectory=False,
    )
    assert trajectory is None
    assert len(checkpoints) == 11


def test_time_integrated_objective():
    psi = np.array([1., 0.]); times = np.linspace(0., 1., 33)
    controls = {"amplitude": .7, "omega": 1.3}
    objective = qa.StateObjective(
        lambda t, s: float(abs(s[1])**2),
        lambda t, s: np.array([0, 2*s[1]]),
        mode="integral",
    )
    value, pullback = ad.vjp(qa.fixed_grid_trajectory, hamiltonian, psi, times,
                             controls, derivatives, objective, wrt="controls",
                             checkpoint_interval=4)
    _, tangent = ad.jvp(qa.fixed_grid_trajectory, hamiltonian, psi, times,
                        controls, derivatives, objective,
                        tangents={"controls": {"omega": 1.}})
    np.testing.assert_allclose(tangent, pullback(1.)["controls"]["omega"], atol=1e-10)
    assert np.isfinite(value)


def test_missing_contract_and_invalid_grid_fail():
    psi = np.array([1., 0.]); controls = {"amplitude": .7, "omega": 1.3}
    with pytest.raises(TypeError, match="derivatives metadata"):
        ad.jvp(qa.fixed_grid_trajectory, hamiltonian, psi, [0., 1.], controls,
               tangents={"controls": {"amplitude": 1.}})
    with pytest.raises(ValueError, match="strictly increasing"):
        qa.fixed_grid_trajectory(hamiltonian, psi, [0., 0.], controls)


def test_quspin_forward_parity_on_dynamic_drive():
    from quspin.basis import spin_basis_1d
    from quspin.operators import hamiltonian as make_hamiltonian

    basis = spin_basis_1d(L=2)
    def drive(t, amplitude): return amplitude * np.sin(t)
    def build(amplitude):
        return make_hamiltonian([], [["z", [[1., 0]], drive, (amplitude,)]],
                                basis=basis, dtype=np.complex128)
    z_matrix = make_hamiltonian([["z", [[1., 0]]]], [], basis=basis,
                                dtype=np.complex128).tocsr().toarray()
    def matrix(t, controls): return build(controls["amplitude"]).tocsr(time=t).toarray()
    def derivative(t, controls): return {"amplitude": np.sin(t) * z_matrix}
    psi = np.zeros(basis.Ns, dtype=complex); psi[0] = 1
    times = np.linspace(0., 1., 9); controls = {"amplitude": .7}
    reference = build(.7).evolve(psi, 0., times, eom="SE")
    value, tangent = ad.jvp(qa.fixed_grid_trajectory, matrix, psi, times,
                            controls, derivative,
                            tangents={"controls": {"amplitude": 1.}})
    np.testing.assert_allclose(value, reference, atol=3e-7, rtol=3e-7)
    eps = 1e-5
    plus = qa.fixed_grid_trajectory(matrix, psi, times, {"amplitude": .7 + eps})
    minus = qa.fixed_grid_trajectory(matrix, psi, times, {"amplitude": .7 - eps})
    np.testing.assert_allclose(tangent, (plus-minus)/(2*eps), atol=2e-7, rtol=2e-6)
