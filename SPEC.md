# QuSpin AD implementation specification

## Scope and fixed versions

This sidecar targets the public API of QuSpin **1.0.1**, corresponding to
official upstream commit `5bf9e5b266e6d8b70e5cf5973c7c7d59d62e412f`.  The
complete inventory is in `api_inventory.json`; its 76 public entries comprise
64 generated-document APIs plus 12 public re-exports/aliases discovered from
package `__all__` exports and source call paths.  The inventory was cross-checked
against the upstream generated docs, exports, tests and examples.  Alias entries
carry an `alias_of` field and an explicit status, so every exported name has a
decision even when it shares one rule with its canonical callable.

The sidecar uses `chainrules==0.1.0` when present and bundles a small compatible
fallback for isolated/offline installs; both expose the same `jvp`, `vjp`,
`grad` and `value_and_grad` protocol through explicit registration at `import
quspin_ad`.  QuSpin remains the sole source of every primal value.  No source
under `upstream/` is imported or modified by the wheel, and no finite-difference
calculation is used at runtime.

## Implemented interfaces

All rules support NumPy arrays with fixed shape/dtype and continuous values away
from the stated domain boundaries.  VJP pullbacks are reusable and return keys
exactly matching `wrt`; zero tangents/cotangents use `chainrules.ZERO`.

### `quspin.tools.misc.KL_div(p1, p2)`

The original callable is used unchanged.  `p1` and `p2` must be one-dimensional,
strictly positive, same-shaped, real arrays normalized to one.  The ambient
extension has gradients

```
g_p1 = log(p1 / p2) + 1
g_p2 = -p1 / p2
```

for a real cotangent; JVP is their dot product with supplied directions.  Users
should provide normalization-preserving directions (`sum(dp)==0`) when testing
on the constrained probability simplex.  Shape, positivity and normalization
errors remain QuSpin errors.  The scalar output has absolute/relative oracle
tolerance `2e-6` for float64 central differences over steps `1e-4..1e-6`.

### `quspin.basis.coherent_state(a, n, dtype)`

`n` and `dtype` are fixed discrete inputs; only scalar amplitude `a` is active.
For `a != 0`,

```
state[k] = exp(-|a|²/2) a**k / sqrt(k!)
dstate[k] = state[k] * (k*da/a - Re(conj(a)*da))
```

which is real-linear for complex amplitudes.  The upstream implementation is
undefined at `a=0` (it evaluates `log(a)`), so the rule raises
`NonDifferentiablePoint` there.  Real and complex cotangents use the real inner
product `Re(conj(cotangent)*direction)`.  Oracle tolerance is `5e-6`.

### `quspin.operators.commutator(H1, H2)` and `anti_commutator(H1, H2)`

The implemented domain is dense square NumPy arrays of compatible shape.  For
sign `s=-1` (commutator) or `s=+1` (anti-commutator),
`f=H1@H2+s*H2@H1` and
`df=dH1@H2+H1@dH2+s*(dH2@H1+H2@dH1)`.  VJP uses the real Frobenius inner
product and returns only requested matrix inputs; unsupported object types,
sparse matrices and non-square shapes raise `UnsupportedWrt`/`TypeError`.
Oracle tolerance is `2e-6`.

### `quspin.tools.evolution.ED_state_vs_time(psi, E, V, times, iterate=False)`

Only the pure-state, non-iterator path (`psi.ndim == 1`, `iterate=False`) is
registered.  The eigensystem matrix `V` is fixed; active inputs are `psi`, `E`
and `times`.  With `c = V.conj().T @ psi` and
`phase[t,k] = exp(-1j*times[t]*E[k])`, `y = V @ (phase*c).T` (QuSpin's
returned shape is `(Ns, Ntime)`).  Its JVP is formed analytically by
differentiating `c` and `phase`; the VJP is the adjoint
of the same linearization under the real complex inner product.  `times` must
be a one-dimensional real array, `E` one-dimensional real array, and all
dimensions remain fixed.  `iterate=True`, mixed-state inputs and spectral
decomposition/eigensolver differentiation are deferred.  Oracle tolerance is
`2e-5` because long-time phase cancellation can amplify round-off.

### `quspin.tools.lanczos.lin_comb_Q_T(coeff, Q_T, out=None)`

The differentiable array domain accepts `coeff.shape == (m,)` and
`Q_T.shape == (m,n)`, with `out=None`; generator inputs and in-place output are
deferred.  The primal is `coeff @ Q_T`.  JVP is `dcoeff @ Q_T + coeff @ dQ_T`;
VJP for cotangent `g` returns `coeff = Q_T.conj() @ g` and
`Q_T = outer(coeff.conj(), g)`.  Oracle tolerance is `2e-6`.

### `quspin.tools.misc.project_op(Obs, proj, dtype)`

Dense square NumPy observables and dense projectors are supported; basis objects,
sparse matrices and Hamiltonian-object branches are outside the domain.  If
`proj.shape[0] == Obs.shape[0]`, the primal is `P.conj().T @ Obs @ P`; if
`proj.shape[1] == Obs.shape[0]`, it is `P @ Obs @ P.conj().T`.  JVP differentiates
both matrices and VJP accepts the structured cotangent `{"Proj_Obs": G}` and
returns requested `Obs`/`proj` gradients under the real Frobenius inner product.
`dtype` and orientation are fixed during differentiation.  Central differences
and JVP/VJP duality are checked to `2e-6`.

### `quspin_ad.fixed_grid_trajectory(...)`

This rule integrates a vector (or batch of vectors) with fixed-step RK4 on a
strictly increasing real grid. `hamiltonian` is `H(t, controls)` or a QuSpin
object exposing `tocsr(time=t)`. Active controls require explicit
`hamiltonian_derivatives` metadata returning `{name: dH/dname}`; missing
metadata, discontinuous callbacks, topology changes, adaptive solvers and
non-scalar controls fail explicitly. The output keeps QuSpin's
state-by-time shape `(Ns, Ntime)` (or `(Ns, Nstate, Ntime)`).

JVP supports `psi0` and mapping-valued `controls` tangents. VJP uses the real
complex inner product and returns matching keys. `StateObjective(value,
gradient, mode="final"|"integral")` supplies analytic final-state or
trapezoidal time-integrated scalar objectives. `checkpoint_interval` stores
periodic states and replays one block at a time in reverse mode; no runtime
finite differences are used.

## Deferred or unsuitable API

The remaining inventory entries are intentionally not registered.  Basis
constructors, integer/bitwise conversion and `photon_Hspace_dim` are discrete;
operator constructors, predicates, save/load and block construction perform
object creation or I/O; sparse matvec helpers are backend dispatch; entropy and
measurement routines contain sorting, rank changes, SVD/eigenvalue branches or
discrete subsystem choices; `mean_level_spacing` sorts and branches at ties;
`evolve`, Floquet, Lanczos eigensolvers and exponential operators contain
adaptive solvers/iterators.
Hamiltonian/quantum-operator methods (`dot`, `expt_value`, `matrix_ele`) are
object-bound and their parameter dictionaries and dynamic drives require an
explicit adapter contract not present in QuSpin's public API.  These entries
are marked `deferred` or `not_suitable` with evidence paths in
`api_inventory.json`; they fail explicitly with ChainRules `RuleNotFound` if
called through AD.

## Verification contract

For each implemented entry, tests cover primal parity, analytic JVP and VJP
against a five-step central-difference scan (`1e-4`, `3e-5`, `1e-5`, `3e-6`,
`1e-6`), JVP/VJP duality, active-input selection, zero directions, reusable
pullbacks, and invalid-domain behavior.  The 7 canonical implemented callables
are exposed through 9 inventory entries because the measurements module also
re-exports `ED_state_vs_time` and `project_op`; these aliases are tested through
their canonical rules.
The complete test command and environment are recorded in
`tasks/task-8/artifacts/test_receipt.txt`; installation is checked in a clean
virtual environment in `tasks/task-8/artifacts/install_receipt.txt`.
