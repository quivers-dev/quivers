"""Pathfinder variational inference over the unconstrained log joint.

Pathfinder [@zhang2022pathfinder] runs L-BFGS on the Jacobian-corrected
unconstrained log density of a program from a random initial point, forms a
normal approximation at every iterate of the optimization path from the
iterate, its gradient, and the L-BFGS inverse-Hessian estimate, selects the
approximation with the largest Monte Carlo estimate of the evidence lower
bound (ELBO), and draws from it. Multi-path Pathfinder runs several
independent paths and resamples their pooled draws with replacement, with
probabilities given by Pareto-smoothed importance weights.

The pieces follow the paper's algorithms:

* `lbfgs_path`: L-BFGS (Algorithm 7) with a strong Wolfe line search and
  Stan's convergence tests, recording every iterate and gradient;
* `recover_diagonal`: the diagonal inverse-Hessian estimate of Gilbert and
  Lemarechal along the path (Algorithm 3);
* `bfgs_approximation`: the compact BFGS representation of
  [@byrd1994representations] at one iterate and its factorization by a thin
  QR and a Cholesky decomposition (Algorithm 4);
* `single_path`: ELBO selection and draws (Algorithm 1);
* `psis_log_weights`: Pareto-smoothed importance sampling as the loo
  package computes it [@vehtari2024psis];
* `Pathfinder`: the driver over a compiled program (Algorithm 2), whose
  result `PathfinderDraws` the fit, prediction, and artifact machinery reads
  as independent draws.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import math
from typing import Literal, Protocol

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.inference._initialization import unconstrained_initial_values
from quivers.inference._latent_recording import latent_structure_recording
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.mcmc.kernel import PotentialFn, _check_observed_support
from quivers.inference.prior import PriorDraws
from quivers.inference.registry import LatentRegistry, _require_one_run
from quivers.inference.trace import trace

_EPSILON = 2.220446049250313e-16
"""Double-precision machine epsilon, the unit of Stan's relative tolerances."""

_MASK = (1 << 64) - 1

type Termination = Literal[
    "absolute objective",
    "relative objective",
    "absolute gradient",
    "relative gradient",
    "parameter change",
    "maximum iterations",
    "line search failed",
]
"""Why an L-BFGS path stopped."""


class PathfinderTarget(Protocol):
    """A log density on a flat unconstrained vector.

    `PotentialFn` is one, through `Pathfinder.run`; a test target can be any
    object with these members.
    """

    @property
    def dtype(self) -> torch.dtype:
        """Floating type the density is evaluated in."""
        ...

    def log_density(self, z: torch.Tensor) -> torch.Tensor:
        """Return the scalar log density at ``z``.

        Parameters
        ----------
        z : torch.Tensor
            Position, shape ``(N,)``, in `dtype`.

        Returns
        -------
        torch.Tensor
            Scalar log density, ``-inf`` outside the support.
        """
        ...

    def value_and_grad(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the log density and its gradient at ``z``.

        Parameters
        ----------
        z : torch.Tensor
            Position, shape ``(N,)``, in `dtype`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Scalar log density and gradient of shape ``(N,)``.
        """
        ...


class _PotentialTarget:
    """Adapt a `PotentialFn` to `PathfinderTarget` in the default dtype.

    Parameters
    ----------
    potential : PotentialFn
        Jacobian-corrected unconstrained log joint of a program.
    """

    def __init__(self, potential: PotentialFn) -> None:
        self._potential = potential

    @property
    def dtype(self) -> torch.dtype:
        """The default floating type, in which programs score."""
        return torch.get_default_dtype()

    def log_density(self, z: torch.Tensor) -> torch.Tensor:
        """Return the potential's log density without a gradient.

        Parameters
        ----------
        z : torch.Tensor
            Position, shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Scalar log density.
        """
        with torch.no_grad():
            return self._potential.log_density(z.to(self.dtype)).detach()

    def value_and_grad(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the potential's log density and gradient.

        Parameters
        ----------
        z : torch.Tensor
            Position, shape ``(N,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Scalar log density and gradient.
        """
        return self._potential.value_and_grad(z.to(self.dtype))


class LBFGSSettings(ValueModel):
    """L-BFGS settings, named as CmdStan's optimizer and Pathfinder arguments.

    Parameters
    ----------
    history_size : int
        Number of position and gradient updates kept (CmdStan
        ``history_size``, default 5).
    max_iterations : int
        Maximum L-BFGS iterations (``max_lbfgs_iters``, default 1000).
    init_alpha : float
        Step length tried first on the first iteration (``init_alpha``).
    tol_obj : float
        Stop when the objective changes by less than this (``tol_obj``).
    tol_rel_obj : float
        Stop when the relative objective change is below this many machine
        epsilons (``tol_rel_obj``).
    tol_grad : float
        Stop when the gradient norm is below this (``tol_grad``).
    tol_rel_grad : float
        Stop when ``g' H g / max(|f|, 1)`` is below this many machine
        epsilons, with ``H`` the inverse-Hessian estimate (``tol_rel_grad``).
    tol_param : float
        Stop when the position moves by less than this (``tol_param``).
    wolfe_c1 : float
        Sufficient decrease constant of the strong Wolfe conditions.
    wolfe_c2 : float
        Curvature constant of the strong Wolfe conditions. Near zero, the
        line search returns the exact minimizer along the direction, which
        cubic interpolation reaches in one step on a quadratic.
    max_line_search : int
        Maximum function evaluations of one line search.
    """

    history_size: int = 5
    max_iterations: int = 1000
    init_alpha: float = 1e-3
    tol_obj: float = 1e-12
    tol_rel_obj: float = 1e4
    tol_grad: float = 1e-8
    tol_rel_grad: float = 1e7
    tol_param: float = 1e-8
    wolfe_c1: float = 1e-4
    wolfe_c2: float = 0.9
    max_line_search: int = 40

    @dx.model_validator()
    def _validate(self) -> None:
        for name in ("history_size", "max_iterations", "max_line_search"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "init_alpha",
            "tol_obj",
            "tol_rel_obj",
            "tol_grad",
            "tol_rel_grad",
            "tol_param",
        ):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value)) or (
                value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.wolfe_c1 < self.wolfe_c2 < 1:
            raise ValueError("the Wolfe constants must satisfy 0 < c1 < c2 < 1")


class OptimizationPath(ValueModel):
    """The iterates of one L-BFGS run on ``-log p``.

    Parameters
    ----------
    positions : torch.Tensor
        Iterates ``theta(0..L)``, float64, shape ``(L + 1, N)``.
    log_densities : torch.Tensor
        ``log p`` at each iterate, shape ``(L + 1,)``.
    gradients : torch.Tensor
        ``grad log p`` at each iterate, shape ``(L + 1, N)``.
    termination : Termination
        Why the run stopped.
    num_evaluations : int
        Log-density and gradient evaluations, line searches included.
    """

    positions: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_densities: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    gradients: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    termination: Termination
    num_evaluations: int

    @property
    def num_iterations(self) -> int:
        """Accepted L-BFGS steps ``L``."""
        return int(self.positions.shape[0]) - 1


class DiagonalRecovery(ValueModel):
    """Output of the paper's α-RECOVER (Algorithm 3).

    Parameters
    ----------
    alphas : torch.Tensor
        Diagonal initial inverse-Hessian estimate after each step, shape
        ``(L, N)``.
    accepted : torch.Tensor
        Boolean, whether step ``l`` enters the inverse-Hessian estimates,
        shape ``(L,)``.
    steps : torch.Tensor
        Position updates ``s(l) = theta(l) - theta(l-1)``, shape ``(L, N)``.
    gradient_steps : torch.Tensor
        Updates ``z(l) = grad log p(theta(l-1)) - grad log p(theta(l))`` of
        the gradient of ``-log p``, shape ``(L, N)``.
    """

    alphas: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    accepted: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    steps: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    gradient_steps: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


class NormalApproximation(ValueModel):
    """The normal approximation at one iterate, in factored form.

    The covariance is ``Sigma = diag(alpha) + beta gamma beta'`` (the
    compact BFGS inverse-Hessian estimate) and is held through the thin QR
    factorization ``Q R = diag(alpha)^(-1/2) beta`` and the Cholesky factor
    ``L`` of ``I + R gamma R'``, so ``Sigma = T T'`` with
    ``T = diag(alpha)^(1/2) (Q L Q' + I - Q Q')``. Sampling and scoring cost
    O(N J + J^2) per draw for history size J.

    Parameters
    ----------
    mean : torch.Tensor
        ``theta + Sigma grad log p(theta)``, shape ``(N,)``.
    alpha : torch.Tensor
        Diagonal part, shape ``(N,)``.
    q : torch.Tensor
        Orthonormal factor, shape ``(N, k)`` with ``k = min(N, 2 J)``.
    cholesky : torch.Tensor
        Lower Cholesky factor of ``I + R gamma R'``, shape ``(k, k)``.
    beta : torch.Tensor
        ``[diag(alpha) Z, S]``, shape ``(N, 2 J)``.
    gamma : torch.Tensor
        Middle matrix of the compact representation, shape ``(2 J, 2 J)``.
    log_det : float
        ``log |Sigma|``.
    """

    mean: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    alpha: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    q: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    cholesky: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    beta: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    gamma: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_det: float

    def sample(self, noise: torch.Tensor) -> torch.Tensor:
        """Map standard normal noise to draws (Algorithm 4, line 11).

        Parameters
        ----------
        noise : torch.Tensor
            Standard normal ``u``, shape ``(M, N)``.

        Returns
        -------
        torch.Tensor
            ``mean + diag(alpha)^(1/2) (Q (L - I) Q' u + u)``, shape ``(M, N)``.
        """
        rotated = noise @ self.q
        eye = torch.eye(self.cholesky.shape[0], dtype=torch.float64)
        lifted = (rotated @ (self.cholesky - eye).T) @ self.q.T + noise
        return self.mean + lifted * self.alpha.sqrt()

    def log_prob(self, draws: torch.Tensor) -> torch.Tensor:
        """Score draws under the approximation.

        The standard normal coordinates are recovered with the inverse of
        ``I + Q (L - I) Q'``, which is ``I + Q (L^(-1) - I) Q'``, so the cost
        is O(N J + J^2) per draw.

        Parameters
        ----------
        draws : torch.Tensor
            Points, shape ``(M, N)``.

        Returns
        -------
        torch.Tensor
            Log densities, shape ``(M,)``.
        """
        whitened = (draws - self.mean) / self.alpha.sqrt()
        rotated = whitened @ self.q
        solved = torch.linalg.solve_triangular(self.cholesky, rotated.T, upper=False).T
        noise = whitened + (solved - rotated) @ self.q.T
        dim = draws.shape[-1]
        return -0.5 * (
            self.log_det + (noise * noise).sum(-1) + dim * math.log(2 * math.pi)
        )

    def covariance(self) -> torch.Tensor:
        """Form the dense covariance; O(N^2 J), for inspection and tests.

        Returns
        -------
        torch.Tensor
            ``diag(alpha) + beta gamma beta'``, shape ``(N, N)``.
        """
        return torch.diag(self.alpha) + self.beta @ self.gamma @ self.beta.T


def lbfgs_path(
    target: PathfinderTarget, initial: torch.Tensor, settings: LBFGSSettings
) -> OptimizationPath:
    """Maximize ``log p`` by L-BFGS, recording every iterate.

    The search direction is the two-loop recursion over the last
    ``history_size`` pairs with the scaled identity
    ``(s'y / y'y) I`` as the initial inverse Hessian; a pair enters the
    history when ``s'y > eps |y|^2``. The first step tries length
    ``init_alpha`` along the gradient, later steps length one. The line
    search enforces the strong Wolfe conditions by bracketing and cubic
    interpolation [@nocedal2006numerical]. The run stops on the first of
    Stan's tests that holds, in Stan's order: absolute and relative objective
    change, absolute and relative gradient, and parameter change.

    Parameters
    ----------
    target : PathfinderTarget
        The log density.
    initial : torch.Tensor
        Starting position, shape ``(N,)``.
    settings : LBFGSSettings
        Optimizer settings.

    Returns
    -------
    OptimizationPath
        Iterates, log densities, gradients, and the stopping reason. Cost is
        O(L J N) besides the evaluations.

    Raises
    ------
    ValueError
        If the log density or its gradient at ``initial`` is not finite.
    """
    evaluations = [0]

    def evaluate(z: torch.Tensor) -> tuple[torch.Tensor, float, torch.Tensor]:
        """Round ``z`` to the target's dtype and return ``-log p`` there.

        Parameters
        ----------
        z : torch.Tensor
            Position.

        Returns
        -------
        tuple
            The rounded position, objective, and objective gradient.
        """
        rounded = z.to(target.dtype).to(torch.float64)
        evaluations[0] += 1
        value, gradient = target.value_and_grad(rounded.to(target.dtype))
        return rounded, -float(value), -gradient.detach().to(torch.float64)

    x, f, g = evaluate(initial.to(torch.float64))
    if not (math.isfinite(f) and bool(torch.isfinite(g).all())):
        raise ValueError(
            "Pathfinder: the log density or its gradient at the initial point "
            "is not finite"
        )
    positions, objectives, gradients = [x], [f], [g]
    pairs: deque[tuple[torch.Tensor, torch.Tensor]] = deque(
        maxlen=settings.history_size
    )
    termination: Termination = "maximum iterations"
    for iteration in range(settings.max_iterations):
        direction = -_two_loop(g, pairs)
        slope = float(g @ direction)
        if not slope < 0:
            pairs.clear()
            direction = -g
            slope = -float(g @ g)
        start = settings.init_alpha if iteration == 0 else 1.0
        step = _line_search(evaluate, x, f, g, direction, slope, start, settings)
        if step is None:
            termination = "line search failed"
            break
        x_new, f_new, g_new = step
        s, y = x_new - x, g_new - g
        if float(s @ y) > _EPSILON * float(y @ y):
            pairs.append((s, y))
        positions.append(x_new)
        objectives.append(f_new)
        gradients.append(g_new)
        change = abs(f - f_new)
        reason: Termination | None = None
        if change < settings.tol_obj:
            reason = "absolute objective"
        elif change / max(abs(f), abs(f_new), 1.0) < settings.tol_rel_obj * _EPSILON:
            reason = "relative objective"
        elif float(g_new.norm()) < settings.tol_grad:
            reason = "absolute gradient"
        elif (
            float(g_new @ _two_loop(g_new, pairs)) / max(abs(f_new), 1.0)
            < settings.tol_rel_grad * _EPSILON
        ):
            reason = "relative gradient"
        elif float(s.norm()) < settings.tol_param:
            reason = "parameter change"
        x, f, g = x_new, f_new, g_new
        if reason is not None:
            termination = reason
            break
    return OptimizationPath(
        positions=torch.stack(positions),
        log_densities=-torch.tensor(objectives, dtype=torch.float64),
        gradients=-torch.stack(gradients),
        termination=termination,
        num_evaluations=evaluations[0],
    )


def _two_loop(
    gradient: torch.Tensor, pairs: deque[tuple[torch.Tensor, torch.Tensor]]
) -> torch.Tensor:
    """Apply the L-BFGS inverse-Hessian estimate to a vector.

    Parameters
    ----------
    gradient : torch.Tensor
        Vector, shape ``(N,)``.
    pairs : deque
        Stored ``(s, y)`` pairs, oldest first.

    Returns
    -------
    torch.Tensor
        ``H gradient`` by the two-loop recursion, O(J N).
    """
    q = gradient.clone()
    coefficients: list[float] = []
    for s, y in reversed(pairs):
        rho = 1.0 / float(s @ y)
        a = rho * float(s @ q)
        coefficients.append(a)
        q = q - a * y
    if pairs:
        s, y = pairs[-1]
        q = q * (float(s @ y) / float(y @ y))
    for (s, y), a in zip(pairs, reversed(coefficients), strict=True):
        rho = 1.0 / float(s @ y)
        b = rho * float(y @ q)
        q = q + (a - b) * s
    return q


def _cubic_minimizer(
    a: float, fa: float, da: float, b: float, fb: float, db: float
) -> float:
    """Minimizer of the cubic through two values and slopes.

    Parameters
    ----------
    a, fa, da : float
        First abscissa, value, and slope.
    b, fb, db : float
        Second abscissa, value, and slope.

    Returns
    -------
    float
        The interpolant's minimizer (Nocedal and Wright, eq. 3.59), exact
        for a quadratic; NaN when it does not exist.
    """
    d1 = da + db - 3 * (fa - fb) / (a - b)
    radicand = d1 * d1 - da * db
    if not radicand >= 0:
        return math.nan
    d2 = math.copysign(math.sqrt(radicand), b - a)
    denominator = db - da + 2 * d2
    if denominator == 0:
        return math.nan
    return b - (b - a) * (db + d2 - d1) / denominator


def _line_search(
    evaluate,
    x: torch.Tensor,
    f0: float,
    g0: torch.Tensor,
    direction: torch.Tensor,
    slope: float,
    start: float,
    settings: LBFGSSettings,
) -> tuple[torch.Tensor, float, torch.Tensor] | None:
    """Find a step satisfying the strong Wolfe conditions.

    Bracketing (Nocedal and Wright, Algorithm 3.5) with expansion by a
    factor of four, then zoom (Algorithm 3.6) with cubic interpolation and
    bisection when the cubic minimizer leaves the bracket or a value is not
    finite.

    Parameters
    ----------
    evaluate : Callable
        Objective evaluator returning the rounded point, value and gradient.
    x : torch.Tensor
        Current point.
    f0 : float
        Objective at ``x``.
    g0 : torch.Tensor
        Gradient at ``x``.
    direction : torch.Tensor
        Descent direction.
    slope : float
        ``g0 . direction``, negative.
    start : float
        First trial step length.
    settings : LBFGSSettings
        Wolfe constants and evaluation budget.

    Returns
    -------
    tuple or None
        The accepted point, value and gradient, or None when the budget is
        exhausted or the bracket collapses.
    """
    del g0
    c1, c2 = settings.wolfe_c1, settings.wolfe_c2
    budget = [settings.max_line_search]

    def probe(alpha: float) -> tuple[torch.Tensor, float, torch.Tensor, float]:
        """Evaluate the objective along the direction.

        Parameters
        ----------
        alpha : float
            Step length.

        Returns
        -------
        tuple
            Point, value, gradient, and directional derivative.
        """
        budget[0] -= 1
        point, value, gradient = evaluate(x + alpha * direction)
        if not (math.isfinite(value) and bool(torch.isfinite(gradient).all())):
            return point, math.inf, gradient, math.nan
        return point, value, gradient, float(gradient @ direction)

    def zoom(
        lo: tuple[float, float, float], hi: tuple[float, float, float]
    ) -> tuple[torch.Tensor, float, torch.Tensor] | None:
        """Shrink a bracket to a strong Wolfe step.

        Parameters
        ----------
        lo : tuple
            Step, value and slope with the lower objective.
        hi : tuple
            The other end of the bracket.

        Returns
        -------
        tuple or None
            Accepted point, value, gradient; None on failure.
        """
        while budget[0] > 0:
            (a_lo, f_lo, d_lo), (a_hi, f_hi, d_hi) = lo, hi
            if abs(a_hi - a_lo) <= 1e-16 * max(abs(a_lo), abs(a_hi), 1.0):
                return None
            trial = (
                _cubic_minimizer(a_lo, f_lo, d_lo, a_hi, f_hi, d_hi)
                if math.isfinite(f_hi) and math.isfinite(d_hi)
                else math.nan
            )
            low, high = min(a_lo, a_hi), max(a_lo, a_hi)
            if not (math.isfinite(trial) and low < trial < high):
                trial = 0.5 * (a_lo + a_hi)
            point, value, gradient, derivative = probe(trial)
            if value > f0 + c1 * trial * slope or value >= f_lo:
                hi = (trial, value, derivative)
                continue
            if abs(derivative) <= -c2 * slope:
                return point, value, gradient
            if derivative * (a_hi - a_lo) >= 0:
                hi = lo
            lo = (trial, value, derivative)
        return None

    previous = (0.0, f0, slope)
    alpha = start
    first = True
    while budget[0] > 0:
        point, value, gradient, derivative = probe(alpha)
        if value > f0 + c1 * alpha * slope or (not first and value >= previous[1]):
            return zoom(previous, (alpha, value, derivative))
        if abs(derivative) <= -c2 * slope:
            return point, value, gradient
        if derivative >= 0:
            return zoom((alpha, value, derivative), previous)
        previous = (alpha, value, derivative)
        alpha *= 4.0
        first = False
    return None


def recover_diagonal(path: OptimizationPath) -> DiagonalRecovery:
    """Recover the diagonal inverse-Hessian estimates along a path.

    Algorithm 3 of [@zhang2022pathfinder]: starting from ``alpha = 1``, each
    step with ``s'z > 1e-12 |z|^2`` updates ``alpha`` elementwise by the
    diagonal BFGS formula of Gilbert and Lemarechal,
    ``1 / alpha_n = a / (b alpha_n) + z_n^2 / b - a s_n^2 / (b c alpha_n^2)``
    with ``a = z' diag(alpha) z``, ``b = z' s``, ``c = s' diag(alpha)^(-1) s``
    [@gilbert1989numerical], and is marked accepted. The paper's printed condition has its
    inequality reversed; the condition here is the one its text states,
    which keeps steps whose gradient change is not 10^12 times larger than
    the position change along it.

    Parameters
    ----------
    path : OptimizationPath
        L-BFGS iterates and gradients.

    Returns
    -------
    DiagonalRecovery
        Per-step diagonals, acceptance marks, and updates; O(L N).
    """
    positions, gradients = path.positions, path.gradients
    steps = positions[1:] - positions[:-1]
    changes = gradients[:-1] - gradients[1:]
    dim = positions.shape[1]
    alpha = torch.ones(dim, dtype=torch.float64)
    alphas, accepted = [], []
    for s, z in zip(steps, changes, strict=True):
        b = float(s @ z)
        if b > 1e-12 * float(z @ z):
            a = float(z @ (alpha * z))
            c = float(s @ (s / alpha))
            alpha = 1.0 / (a / (b * alpha) + z * z / b - a * s * s / (b * c * alpha**2))
            accepted.append(True)
        else:
            accepted.append(False)
        alphas.append(alpha)
    return DiagonalRecovery(
        alphas=torch.stack(alphas) if alphas else torch.empty(0, dim),
        accepted=torch.tensor(accepted, dtype=torch.bool),
        steps=steps,
        gradient_steps=changes,
    )


def compact_factors(
    alpha: torch.Tensor, s: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact BFGS representation of the inverse Hessian.

    Equation (2) of [@zhang2022pathfinder], after Byrd, Nocedal and Schnabel
    (1994, eq. 2.6): ``beta = [diag(alpha) Z, S]`` and
    ``gamma = [[0, -E^(-1)], [-E^(-T), E^(-T) (diag(eta) + Z' diag(alpha) Z) E^(-1)]]``
    with ``E`` the upper triangle of ``S'Z`` and ``eta`` its diagonal, so that
    ``diag(alpha) + beta gamma beta'`` equals the BFGS updates of
    ``diag(alpha)`` by the pairs in order.

    Parameters
    ----------
    alpha : torch.Tensor
        Initial diagonal, shape ``(N,)``.
    s : torch.Tensor
        Position updates as columns, shape ``(N, m)``.
    z : torch.Tensor
        Gradient updates of ``-log p`` as columns, shape ``(N, m)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``beta`` of shape ``(N, 2m)`` and ``gamma`` of shape ``(2m, 2m)``;
        O(N m^2 + m^3).
    """
    m = s.shape[1]
    inner = s.T @ z
    upper = torch.triu(inner)
    eta = torch.diagonal(inner)
    eye = torch.eye(m, dtype=torch.float64)
    inverse = torch.linalg.solve_triangular(upper, eye, upper=True)
    beta = torch.cat([alpha[:, None] * z, s], dim=1)
    lower_right = inverse.T @ (torch.diag(eta) + z.T @ (alpha[:, None] * z)) @ inverse
    gamma = torch.cat(
        [
            torch.cat([torch.zeros(m, m, dtype=torch.float64), -inverse], dim=1),
            torch.cat([-inverse.T, lower_right], dim=1),
        ],
        dim=0,
    )
    return beta, gamma


def normal_approximation(
    position: torch.Tensor,
    gradient: torch.Tensor,
    alpha: torch.Tensor,
    s: torch.Tensor,
    z: torch.Tensor,
) -> NormalApproximation | None:
    """Factor the normal approximation at one iterate (Algorithm 4).

    Parameters
    ----------
    position : torch.Tensor
        Iterate ``theta``, shape ``(N,)``.
    gradient : torch.Tensor
        ``grad log p(theta)``, shape ``(N,)``.
    alpha : torch.Tensor
        Diagonal estimate, shape ``(N,)``.
    s, z : torch.Tensor
        Selected update pairs as columns, shape ``(N, m)``, oldest first.

    Returns
    -------
    NormalApproximation or None
        The factored approximation, or None when the estimate is not
        positive definite. Cost O(N m^2 + m^3).
    """
    if not bool((alpha > 0).all() and torch.isfinite(alpha).all()):
        return None
    dim = position.shape[0]
    if s.shape[1] == 0:
        empty = torch.zeros(dim, 0, dtype=torch.float64)
        return NormalApproximation(
            mean=position + alpha * gradient,
            alpha=alpha,
            q=empty,
            cholesky=torch.zeros(0, 0, dtype=torch.float64),
            beta=empty,
            gamma=torch.zeros(0, 0, dtype=torch.float64),
            log_det=float(alpha.log().sum()),
        )
    beta, gamma = compact_factors(alpha, s, z)
    q, r = torch.linalg.qr(beta / alpha.sqrt()[:, None], mode="reduced")
    inner = torch.eye(q.shape[1], dtype=torch.float64) + r @ gamma @ r.T
    cholesky, info = torch.linalg.cholesky_ex(0.5 * (inner + inner.T))
    if int(info) != 0 or not bool(torch.isfinite(cholesky).all()):
        return None
    mean = position + alpha * gradient + beta @ (gamma @ (beta.T @ gradient))
    log_det = float(alpha.log().sum() + 2 * torch.diagonal(cholesky).log().sum())
    if not (math.isfinite(log_det) and bool(torch.isfinite(mean).all())):
        return None
    return NormalApproximation(
        mean=mean,
        alpha=alpha,
        q=q,
        cholesky=cholesky,
        beta=beta,
        gamma=gamma,
        log_det=log_det,
    )


def bfgs_approximation(
    path: OptimizationPath,
    recovery: DiagonalRecovery,
    iteration: int,
    history_size: int,
) -> NormalApproximation | None:
    """The normal approximation at iterate ``iteration`` of a path.

    It uses the last ``history_size`` accepted pairs among steps
    ``1..iteration`` and the diagonal recovered at that iterate.

    Parameters
    ----------
    path : OptimizationPath
        The L-BFGS iterates.
    recovery : DiagonalRecovery
        α-RECOVER output for the path.
    iteration : int
        Iterate ``l`` in ``1..L``.
    history_size : int
        Number ``J`` of pairs.

    Returns
    -------
    NormalApproximation or None
        The approximation, or None when it is not positive definite.
    """
    chosen = [index for index in range(iteration) if bool(recovery.accepted[index])][
        -history_size:
    ]
    return normal_approximation(
        path.positions[iteration],
        path.gradients[iteration],
        recovery.alphas[iteration - 1],
        recovery.steps[chosen].T,
        recovery.gradient_steps[chosen].T,
    )


def stream(seed: int, index: int) -> torch.Generator:
    """Derive an independent generator from a seed by SplitMix64.

    The mixing is the one `MCMC` uses to give each chain its stream.

    Parameters
    ----------
    seed : int
        Base seed.
    index : int
        Stream index.

    Returns
    -------
    torch.Generator
        A CPU generator seeded with the mixed value.
    """
    value = (seed + (index + 1) * 0x9E3779B97F4A7C15) & _MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK
    value ^= value >> 31
    return torch.Generator().manual_seed(value)


def _score(target: PathfinderTarget, draws: torch.Tensor) -> torch.Tensor:
    """Evaluate the target log density at each draw.

    Parameters
    ----------
    target : PathfinderTarget
        The log density.
    draws : torch.Tensor
        Points in the target's dtype as float64, shape ``(M, N)``.

    Returns
    -------
    torch.Tensor
        Float64 log densities, ``-inf`` where not finite.
    """
    values = torch.tensor(
        [float(target.log_density(draw.to(target.dtype))) for draw in draws],
        dtype=torch.float64,
    )
    return torch.where(torch.isfinite(values), values, -math.inf)


def _draw(
    approximation: NormalApproximation,
    count: int,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw from an approximation at the target's precision.

    Parameters
    ----------
    approximation : NormalApproximation
        The normal approximation.
    count : int
        Number of draws.
    generator : torch.Generator
        Stream of the standard normal noise.
    dtype : torch.dtype
        Precision the target evaluates in; draws are rounded to it so that
        the approximation scores the evaluated points.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Draws ``(count, N)`` and their log densities under the approximation.
    """
    dim = approximation.mean.shape[0]
    noise = torch.randn(count, dim, generator=generator, dtype=torch.float64)
    draws = approximation.sample(noise).to(dtype).to(torch.float64)
    return draws, approximation.log_prob(draws)


class SinglePath(ValueModel):
    """One Pathfinder run (Algorithm 1).

    Parameters
    ----------
    path : OptimizationPath
        The L-BFGS iterates.
    elbo : torch.Tensor
        ELBO estimate at iterates ``1..L``, ``-inf`` where the approximation
        is not positive definite or a draw has a nonfinite log density.
    best_iteration : int
        Iterate ``l*`` of the largest ELBO, zero when the path failed.
    draws : torch.Tensor
        Unconstrained draws from the selected approximation, shape ``(M, N)``.
    log_density : torch.Tensor
        Target log density at each draw (NaN without ``calculate_lp``).
    log_approx : torch.Tensor
        Approximation log density at each draw.
    failed : bool
        Whether no iterate gave a usable approximation. A failed path keeps
        its last iterate as its draws with ``log_approx = inf``, so that it
        receives zero importance weight.
    """

    path: OptimizationPath = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    elbo: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    best_iteration: int
    draws: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_density: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_approx: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    failed: bool


def single_path(
    target: PathfinderTarget,
    initial: torch.Tensor,
    *,
    seed: int,
    settings: LBFGSSettings,
    num_elbo_draws: int,
    num_draws: int,
    calculate_lp: bool = True,
) -> SinglePath:
    """Run single-path Pathfinder from one initial point.

    The ELBO at iterate ``l`` averages ``log p(phi) - log q(phi)`` over
    ``num_elbo_draws`` draws of stream ``stream(seed, l)``; the returned
    draws come from stream ``stream(seed, -1)``.

    Parameters
    ----------
    target : PathfinderTarget
        The log density.
    initial : torch.Tensor
        Starting position, shape ``(N,)``.
    seed : int
        Base of the path's random streams.
    settings : LBFGSSettings
        L-BFGS settings, including the history size.
    num_elbo_draws : int
        Draws per ELBO estimate ``K``.
    num_draws : int
        Draws returned ``M``.
    calculate_lp : bool
        Whether to evaluate the target at the returned draws.

    Returns
    -------
    SinglePath
        The path, its ELBO profile, and the draws. Cost is the L-BFGS
        evaluations plus ``L K + M`` log-density evaluations and
        O(L (N J^2 + J^3 + K (N J + J^2))) arithmetic.
    """
    path = lbfgs_path(target, initial, settings)
    recovery = recover_diagonal(path)
    elbo = torch.full((path.num_iterations,), -math.inf, dtype=torch.float64)
    best: NormalApproximation | None = None
    best_iteration = 0
    for iteration in range(1, path.num_iterations + 1):
        approximation = bfgs_approximation(
            path, recovery, iteration, settings.history_size
        )
        if approximation is None:
            continue
        draws, log_q = _draw(
            approximation, num_elbo_draws, stream(seed, iteration), target.dtype
        )
        log_p = _score(target, draws)
        estimate = float((log_p - log_q).mean())
        if not math.isfinite(estimate):
            continue
        elbo[iteration - 1] = estimate
        if best is None or estimate > float(elbo[best_iteration - 1]):
            best, best_iteration = approximation, iteration
    if best is None:
        last = path.positions[-1:]
        return SinglePath(
            path=path,
            elbo=elbo,
            best_iteration=0,
            draws=last,
            log_density=path.log_densities[-1:],
            log_approx=torch.full((1,), math.inf, dtype=torch.float64),
            failed=True,
        )
    draws, log_q = _draw(best, num_draws, stream(seed, -1), target.dtype)
    log_p = (
        _score(target, draws)
        if calculate_lp
        else torch.full((num_draws,), math.nan, dtype=torch.float64)
    )
    return SinglePath(
        path=path,
        elbo=elbo,
        best_iteration=best_iteration,
        draws=draws,
        log_density=log_p,
        log_approx=log_q,
        failed=False,
    )


def _gpd_fit(exceedances: torch.Tensor) -> tuple[float, float]:
    """Fit a generalized Pareto distribution as loo's ``gpdfit``.

    The empirical Bayes estimate of Zhang and Stephens with loo's prior of
    3, grid of ``30 + floor(sqrt(n))`` points, and weakly informative
    shrinkage ``(n k + 5) / (n + 10)``.

    Parameters
    ----------
    exceedances : torch.Tensor
        Sorted positive exceedances, float64.

    Returns
    -------
    tuple[float, float]
        Shape ``k`` (``inf`` when not estimable) and scale ``sigma``.
    """
    n = exceedances.shape[0]
    grid = 30 + math.floor(math.sqrt(n))
    j = torch.arange(1, grid + 1, dtype=torch.float64)
    quartile = float(exceedances[math.floor(n / 4 + 0.5) - 1])
    theta = (
        1 / float(exceedances[-1]) + (1 - torch.sqrt(grid / (j - 0.5))) / 3 / quartile
    )
    k_profile = torch.log1p(-theta[:, None] * exceedances[None, :]).mean(1)
    profile = n * (torch.log(-theta / k_profile) - k_profile - 1)
    weights = torch.exp(profile - torch.logsumexp(profile, 0))
    theta_hat = float((theta * weights).sum())
    k = float(torch.log1p(-theta_hat * exceedances).mean())
    sigma = -k / theta_hat
    k = (k * n + 10 * 0.5) / (n + 10)
    if math.isnan(k):
        k = math.inf
    return k, sigma


def psis_log_weights(log_ratios: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Pareto-smooth importance log ratios as loo's ``psis`` with ``r_eff = 1``.

    The largest ``M = ceil(min(0.2 S, 3 sqrt(S)))`` ratios are replaced by
    the quantiles ``(i - 1/2) / M`` of a generalized Pareto distribution fit
    to their exceedances over the next largest ratio, and every ratio is
    truncated at the largest raw ratio [@vehtari2024psis]. Ratios of
    ``-inf`` keep zero weight.

    Parameters
    ----------
    log_ratios : torch.Tensor
        Log importance ratios ``log p - log q``, shape ``(S,)``.

    Returns
    -------
    tuple[torch.Tensor, float]
        Normalized smoothed log weights and the Pareto shape ``k`` (``inf``
        when the tail cannot be fit). Cost O(S log S).

    Raises
    ------
    ValueError
        If no ratio is finite.
    """
    ratios = log_ratios.detach().to(torch.float64)
    finite = torch.isfinite(ratios)
    if not bool(finite.any()):
        raise ValueError("Pathfinder: no draw has a finite importance ratio")
    count = int(finite.sum())
    peak = float(ratios[finite].max())
    weights = torch.where(finite, ratios - peak, -math.inf)
    tail = math.ceil(min(0.2 * count, 3 * math.sqrt(count)))
    k = math.inf
    if tail >= 5:
        values, order = torch.sort(weights[finite])
        indices = torch.nonzero(finite).flatten()[order]
        tail_values = values[-tail:]
        if abs(float(tail_values[-1] - tail_values[0])) >= _EPSILON / 100:
            cutoff = float(values[-tail - 1])
            k, sigma = _gpd_fit(torch.exp(tail_values) - math.exp(cutoff))
            if math.isfinite(k):
                p = (torch.arange(1, tail + 1, dtype=torch.float64) - 0.5) / tail
                quantiles = sigma * torch.expm1(-k * torch.log1p(-p)) / k
                weights[indices[-tail:]] = torch.log(quantiles + math.exp(cutoff))
    weights = torch.clamp(weights, max=0.0)
    return weights - torch.logsumexp(weights, 0), k


class PathfinderDraws(PriorDraws):
    """Pathfinder draws, read by the fit machinery as independent draws.

    The samples share `PriorDraws`'s layout ``(1, draw, *event_shape)``:
    importance-resampled draws are not a Markov chain and carry no chain
    diagnostics. `draw_provenance` names them variational.

    Parameters
    ----------
    samples : Mapping[str, torch.Tensor]
        Constrained draws ``(1, draw, *event_shape)``.
    num_samples : int
        Number of draws.
    log_density : torch.Tensor
        Unconstrained log joint (Stan's ``lp__``) at each draw, shape
        ``(1, draw)``; NaN without ``calculate_lp``.
    log_approx : torch.Tensor
        Approximation log density (``lp_approx__``), shape ``(1, draw)``.
    path_index : torch.Tensor
        Path each draw came from, shape ``(draw,)``.
    path_elbo : torch.Tensor
        Largest ELBO estimate of each path, ``-inf`` for a failed path.
    path_iterations : torch.Tensor
        L-BFGS iterations of each path.
    path_best_iteration : torch.Tensor
        Selected iterate of each path.
    path_termination : tuple[str, ...]
        Why each path's L-BFGS run stopped.
    num_evaluations : torch.Tensor
        Log-density evaluations of each path.
    pareto_k : float
        Pareto shape of the pooled importance ratios; NaN without
        resampling.
    resampled : bool
        Whether the draws were importance resampled.
    single_paths : Mapping[str, torch.Tensor]
        With ``save_single_paths``, each path's draws ``(path, draw, ...)``
        before resampling; empty otherwise.
    """

    log_density: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_approx: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    path_index: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    path_elbo: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    path_iterations: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    path_best_iteration: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    path_termination: tuple[str, ...] = ()
    num_evaluations: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    pareto_k: float = math.nan
    resampled: bool = False
    single_paths: Mapping[str, torch.Tensor] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @property
    def num_paths(self) -> int:
        """Number of Pathfinder runs."""
        return int(self.path_elbo.shape[0])


def draw_provenance(
    posterior: MCMCResult | PriorDraws | Guide,
) -> Literal["mcmc", "variational", "prior"]:
    """Name where retained draws come from.

    Parameters
    ----------
    posterior : MCMCResult or PriorDraws or Guide
        A fit's posterior.

    Returns
    -------
    {"mcmc", "variational", "prior"}
        ``"mcmc"`` for recorded chains, ``"variational"`` for a guide or
        Pathfinder draws, ``"prior"`` for independent prior draws.
    """
    if isinstance(posterior, MCMCResult):
        return "mcmc"
    if isinstance(posterior, PathfinderDraws):
        return "variational"
    return "prior" if isinstance(posterior, PriorDraws) else "variational"


type InitialValues = Mapping[str, torch.Tensor | float]
"""Natural-scale starting values keyed by latent site."""


class Pathfinder(ValueModel):
    """Single- and multi-path Pathfinder over a compiled program.

    Parameter names follow CmdStan's ``pathfinder`` method; brms maps
    ``chains`` to ``num_paths``. With one path the draws are that path's
    ``num_draws`` draws. With several, the ``num_paths * num_draws`` pooled
    draws are resampled with replacement to ``num_psis_draws`` draws with
    Pareto-smoothed importance weights ``p / q`` (Algorithm 2), unless
    ``psis_resample`` or ``calculate_lp`` is False, in which case every
    pooled draw is returned. Paths run one after another, each from its own
    stream derived from ``seed``.

    Parameters
    ----------
    num_paths : int
        Independent runs ``I`` (CmdStan ``num_paths``, default 4).
    num_draws : int
        Draws per path ``M`` (``num_draws``; cmdstanr ``single_path_draws``).
    num_psis_draws : int
        Draws after resampling ``R`` (``num_psis_draws``; cmdstanr ``draws``).
    num_elbo_draws : int
        Draws per ELBO estimate ``K`` (``num_elbo_draws``, default 25).
    psis_resample : bool
        Whether to importance resample across paths.
    calculate_lp : bool
        Whether to evaluate the target at the returned draws.
    save_single_paths : bool
        Whether to keep each path's draws in the result.
    lbfgs : LBFGSSettings
        L-BFGS settings (``history_size``, ``max_lbfgs_iters``,
        ``init_alpha``, and the tolerances).
    init_radius : float
        Initial points are uniform on ``(-init_radius, init_radius)`` in
        every unconstrained coordinate (CmdStan's ``init`` radius).
    init_strategy : {"uniform", "zero"}
        Uniform initial points, or the origin.
    initial_values : Mapping or Sequence of Mapping or None
        Natural-scale values replacing the named sites' initial coordinates,
        shared or one mapping per path.
    seed : int
        Base seed of every path's streams.
    """

    num_paths: int = 4
    num_draws: int = 1000
    num_psis_draws: int = 1000
    num_elbo_draws: int = 25
    psis_resample: bool = True
    calculate_lp: bool = True
    save_single_paths: bool = False
    lbfgs: LBFGSSettings = dx.field(default_factory=LBFGSSettings)
    init_radius: float = 2.0
    init_strategy: Literal["uniform", "zero"] = "uniform"
    initial_values: InitialValues | tuple[InitialValues, ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    seed: int = 0

    @dx.model_validator()
    def _validate(self) -> None:
        for name in ("num_paths", "num_draws", "num_psis_draws", "num_elbo_draws"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Pathfinder: {name} must be a positive integer")
        if not (math.isfinite(self.init_radius) and self.init_radius > 0):
            raise ValueError("Pathfinder: init_radius must be positive and finite")
        if self.init_strategy not in ("uniform", "zero"):
            raise ValueError("Pathfinder: init_strategy must be 'uniform' or 'zero'")
        if (
            self.initial_values is not None
            and not isinstance(self.initial_values, Mapping)
            and len(self.initial_values) != self.num_paths
        ):
            raise ValueError(
                "Pathfinder: per-path initial values need one mapping per path "
                f"({self.num_paths})"
            )

    def run(
        self,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> PathfinderDraws:
        """Run Pathfinder on a program's unconstrained log joint.

        The latents are those one recorded run of the program reaches, as
        for `MCMC`, and the target is the same Jacobian-corrected
        `PotentialFn` the Hamiltonian kernels use.

        Parameters
        ----------
        model : MonadicProgram
            The program.
        x : torch.Tensor
            One-row program input.
        observations : dict[str, torch.Tensor]
            Observed values and data.

        Returns
        -------
        PathfinderDraws
            Constrained draws and per-path diagnostics.

        Raises
        ------
        ValueError
            If the input has several rows, the data lie outside their
            support, an initial value is invalid, an initial point has a
            nonfinite log density, or every path fails.
        """
        _require_one_run(x, "Pathfinder.run")
        _check_observed_support(model, observations, "Pathfinder.run")
        with (
            torch.random.fork_rng(devices=[]),
            latent_structure_recording(),
        ):
            torch.random.set_rng_state(stream(self.seed, -2).get_state())
            recorded = trace(model, x, observations=observations)
        registry = LatentRegistry.from_trace(model, recorded)
        potential = PotentialFn(model, registry, x, observations)
        unconstrained = self.run_target(
            _PotentialTarget(potential),
            registry.total_unconstrained_dim,
            registry=registry,
        )
        return unconstrained

    def _initial_point(
        self, dim: int, path: int, registry: LatentRegistry | None
    ) -> torch.Tensor:
        """Draw a path's initial point.

        Parameters
        ----------
        dim : int
            Unconstrained dimension.
        path : int
            Path index.
        registry : LatentRegistry or None
            Registry naming the sites of ``initial_values``.

        Returns
        -------
        torch.Tensor
            Float64 starting position.

        Raises
        ------
        ValueError
            If initial values are given without a registry.
        """
        if self.init_strategy == "zero":
            point = torch.zeros(dim, dtype=torch.float64)
        else:
            uniform = torch.rand(
                dim, generator=stream(self.seed, 2 * path), dtype=torch.float64
            )
            point = self.init_radius * (2 * uniform - 1)
        named = (
            self.initial_values
            if self.initial_values is None or isinstance(self.initial_values, Mapping)
            else self.initial_values[path]
        )
        if named:
            if registry is None:
                raise ValueError(
                    "Pathfinder: named initial values need a program's registry"
                )
            for name, value in unconstrained_initial_values(registry, named).items():
                site = registry[name]
                point[site.flat_offset : site.flat_offset + site.flat_length] = (
                    value.reshape(-1).to(torch.float64)
                )
        return point

    def run_target(
        self,
        target: PathfinderTarget,
        dim: int,
        *,
        registry: LatentRegistry | None = None,
    ) -> PathfinderDraws:
        """Run Pathfinder on a log density over ``R^dim``.

        Parameters
        ----------
        target : PathfinderTarget
            The log density.
        dim : int
            Dimension ``N``.
        registry : LatentRegistry or None
            The program's registry; its sites name the result's samples.
            Without one, the result holds the unconstrained draws as site
            ``"theta"``.

        Returns
        -------
        PathfinderDraws
            Draws and per-path diagnostics.

        Raises
        ------
        ValueError
            If every path fails.
        """
        paths = [
            single_path(
                target,
                self._initial_point(dim, path, registry),
                seed=int(stream(self.seed, 2 * path + 1).initial_seed()),
                settings=self.lbfgs,
                num_elbo_draws=self.num_elbo_draws,
                num_draws=self.num_draws,
                calculate_lp=self.calculate_lp,
            )
            for path in range(self.num_paths)
        ]
        usable = [index for index, path in enumerate(paths) if not path.failed]
        if not usable:
            raise ValueError(
                "Pathfinder: every path failed to produce a positive definite "
                "approximation with a finite ELBO; reasons: "
                + ", ".join(path.path.termination for path in paths)
            )
        pooled = torch.cat([paths[index].draws for index in usable])
        log_p = torch.cat([paths[index].log_density for index in usable])
        log_q = torch.cat([paths[index].log_approx for index in usable])
        origin = torch.cat(
            [
                torch.full((paths[index].draws.shape[0],), index, dtype=torch.long)
                for index in usable
            ]
        )
        resample = self.num_paths > 1 and self.psis_resample and self.calculate_lp
        pareto_k = math.nan
        if resample:
            log_weights, pareto_k = psis_log_weights(log_p - log_q)
            chosen = torch.multinomial(
                log_weights.exp(),
                self.num_psis_draws,
                replacement=True,
                generator=stream(self.seed, -3),
            )
            pooled, log_p, log_q, origin = (
                pooled[chosen],
                log_p[chosen],
                log_q[chosen],
                origin[chosen],
            )
        count = int(pooled.shape[0])
        samples = _constrained_samples(pooled, registry)
        single: dict[str, torch.Tensor] = {}
        if self.save_single_paths:
            per_path = [
                None if path.failed else _constrained_samples(path.draws, registry)
                for path in paths
            ]
            for name, value in samples.items():
                filler = torch.full(
                    (self.num_draws, *value.shape[2:]), math.nan, dtype=value.dtype
                )
                single[name] = torch.stack(
                    [filler if drawn is None else drawn[name][0] for drawn in per_path]
                )
        return PathfinderDraws(
            samples=samples,
            num_samples=count,
            log_density=log_p.reshape(1, count),
            log_approx=log_q.reshape(1, count),
            path_index=origin,
            path_elbo=torch.tensor(
                [
                    -math.inf if path.failed else float(path.elbo.max())
                    for path in paths
                ],
                dtype=torch.float64,
            ),
            path_iterations=torch.tensor(
                [path.path.num_iterations for path in paths], dtype=torch.long
            ),
            path_best_iteration=torch.tensor(
                [path.best_iteration for path in paths], dtype=torch.long
            ),
            path_termination=tuple(path.path.termination for path in paths),
            num_evaluations=torch.tensor(
                [
                    path.path.num_evaluations
                    + path.path.num_iterations * self.num_elbo_draws
                    + (0 if path.failed or not self.calculate_lp else self.num_draws)
                    for path in paths
                ],
                dtype=torch.long,
            ),
            pareto_k=pareto_k,
            resampled=resample,
            single_paths=single,
        )


def _constrained_samples(
    draws: torch.Tensor, registry: LatentRegistry | None
) -> dict[str, torch.Tensor]:
    """Map unconstrained draws to constrained site samples.

    Parameters
    ----------
    draws : torch.Tensor
        Float64 draws, shape ``(S, N)``.
    registry : LatentRegistry or None
        Registry of the program; None keeps the draws as site ``"theta"``.

    Returns
    -------
    dict[str, torch.Tensor]
        Site samples ``(1, S, *constrained_shape)`` in the default dtype.
    """
    if registry is None:
        return {"theta": draws.unsqueeze(0)}
    dtype = torch.get_default_dtype()
    per_draw = [
        registry.to_constrained(registry.unflatten_unconstrained(draw.to(dtype)))[0]
        for draw in draws
    ]
    return {
        name: torch.stack([values[name] for values in per_draw]).unsqueeze(0)
        for name in registry.names
    }


__all__ = [
    "DiagonalRecovery",
    "LBFGSSettings",
    "NormalApproximation",
    "OptimizationPath",
    "Pathfinder",
    "PathfinderDraws",
    "PathfinderTarget",
    "SinglePath",
    "bfgs_approximation",
    "compact_factors",
    "draw_provenance",
    "lbfgs_path",
    "normal_approximation",
    "psis_log_weights",
    "recover_diagonal",
    "single_path",
    "stream",
]
