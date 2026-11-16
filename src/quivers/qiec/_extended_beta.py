"""Extended-Beta reference scores and paired-coordinate sampling."""

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass

from quivers.qiec._beta_tails import log_beta_front, log_beta_tails
from quivers.qiec._sampling import _dirichlet_log_weights
from quivers.qiec.reference_support import DistributionError, real_parameter, finite_value


@dataclass(frozen=True)
class _ExtendedBeta:
    """Retain concentration and extension metadata for scalar measure queries.

    Parameters
    ----------
    a, b : float
        Strictly positive Beta concentrations.
    kappa : float
        Nonnegative symmetric extension.
    """

    a: float
    b: float
    kappa: float

    def _tails(self, value):
        """Evaluate CDF and survival with complementary endpoint coordinates.

        Parameters
        ----------
        value : float
            Scalar response coordinate.

        Returns
        -------
        tuple[float, float]
            Complementary log cumulative and survival probabilities.

        Raises
        ------
        DistributionError
            If the evaluation or sample exceeds the documented numerical capacity.
        """
        y = float(value)
        if math.isnan(y):
            raise DistributionError("ExtendedBeta CDF point must not be NaN")
        if y < 0:
            return -math.inf, 0.
        if y >= 1:
            return 0., -math.inf
        scale = 1+2*self.kappa
        lower, upper = (y+self.kappa)/scale, (1-y+self.kappa)/scale
        if self.kappa > 1:
            lower, upper = .5+(y-.5)/scale, .5+(.5-y)/scale
        if upper < lower:
            cdf, sf = log_beta_tails(self.b,self.a,upper)
            return sf,cdf
        return log_beta_tails(self.a,self.b,lower)

    def log_cdf(self, value):
        """Evaluate the inclusive CDF, retaining the lower censoring atom.

        Parameters
        ----------
        value : float
            Scalar response coordinate.

        Returns
        -------
        float
            Scalar distribution result.
        """
        return self._tails(value)[0]

    def log_survival_function(self, value):
        """Evaluate strict survival, excluding the upper censoring atom.

        Parameters
        ----------
        value : float
            Scalar response coordinate.

        Returns
        -------
        float
            Scalar distribution result.
        """
        return self._tails(value)[1]

    def log_cdf_left(self, value):
        """Evaluate the left CDF, excluding either censoring atom.

        Parameters
        ----------
        value : float
            Scalar response coordinate.

        Returns
        -------
        float
            Scalar distribution result.
        """
        if value <= 0:
            return -math.inf
        if value == 1:
            return log_beta_tails(self.b,self.a,self.kappa/(1+2*self.kappa))[1]
        return self.log_cdf(value)

    def log_survival_function_left(self, value):
        """Evaluate inclusive survival, retaining the upper censoring atom.

        Parameters
        ----------
        value : float
            Scalar response coordinate.

        Returns
        -------
        float
            Scalar distribution result.
        """
        if value <= 0:
            return 0.
        if value == 1:
            return log_beta_tails(self.b,self.a,self.kappa/(1+2*self.kappa))[0]
        return self.log_survival_function(value)

    def log_prob(self, value):
        """Score the two atom masses and the interior affine density.

        Parameters
        ----------
        value : float
            Scalar response coordinate.

        Returns
        -------
        float
            Scalar distribution result.
        """
        y = finite_value(value,"ExtendedBeta")
        if not 0 <= y <= 1:
            return -math.inf
        scale = 1+2*self.kappa
        if y == 0:
            return log_beta_tails(self.a,self.b,self.kappa/scale)[0]
        if y == 1:
            return log_beta_tails(self.b,self.a,self.kappa/scale)[0]
        lower, upper = (y+self.kappa)/scale, (1-y+self.kappa)/scale
        if self.kappa > 1:
            lower, upper = .5+(y-.5)/scale, .5+(.5-y)/scale
        a,b=self.a,self.b
        if upper < lower:
            lower,a,b=upper,b,a
        return log_beta_front(a,b,lower)-math.log(lower)-math.log1p(-lower)-math.log(scale)

    def sample(self, rng):
        """Classify censoring atoms before rounding either Beta coordinate.

        Parameters
        ----------
        rng : random.Random
            Caller-owned random generator.

        Returns
        -------
        float
            Scalar distribution result.

        Raises
        ------
        DistributionError
            If the evaluation or sample exceeds the documented numerical capacity.
        """
        log_x,log_q=_dirichlet_log_weights((self.a,self.b),rng)
        threshold=math.log(self.kappa)-math.log1p(2*self.kappa) if self.kappa else -math.inf
        if log_x <= threshold:
            return 0.
        if log_q <= threshold:
            return 1.
        if log_x <= log_q:
            x=math.exp(log_x)
            result=x+2*self.kappa*(x-.5)
        else:
            q=math.exp(log_q)
            result=1-(q+2*self.kappa*(q-.5))
        if not 0 < result < 1:
            raise DistributionError("ExtendedBeta interior sample exceeds Float64 endpoint resolution")
        return result


def distribution(arguments: Mapping[str, object]) -> _ExtendedBeta:
    """Construct a retained reference kernel for density, sampling, and tails.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Beta mean mu, precision phi, and extension kappa.

    Returns
    -------
    _ExtendedBeta
        Validated scalar kernel with explicit atom-aware left tails.

    Raises
    ------
    DistributionError
        If arguments violate the law or Float64 interval capacity.
    """
    mu,phi,kappa=(real_parameter(arguments,name,"ExtendedBeta") for name in ("mu","phi","kappa"))
    if not 0 < mu < 1 or phi <= 0 or kappa < 0 or not math.isfinite(mu+phi+kappa):
        raise DistributionError("ExtendedBeta requires mu in(0,1), phi>0 and kappa>=0")
    scale=1+2*kappa
    if not math.isfinite(scale) or not kappa/scale < .5 < .5+.5/scale:
        raise DistributionError("ExtendedBeta extension exceeds Float64 interval resolution")
    a,b=mu*phi,(1-mu)*phi
    if min(a,b) <= 0:
        raise DistributionError("ExtendedBeta concentrations exceed positive Float64 capacity")
    return _ExtendedBeta(a,b,kappa)


def density(arguments: Mapping[str, object], value: object) -> float:
    """Evaluate an ExtendedBeta event through the retained reference kernel.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Named family parameters.
    value : object
        Scalar response.

    Returns
    -------
    float
        Log atom mass or interior log density.
    """
    return distribution(arguments).log_prob(value)


def sample(arguments: Mapping[str, object], rng: random.Random) -> float:
    """Draw one ExtendedBeta response through the shared log-Gamma sampler.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Named family parameters.
    rng : random.Random
        Caller-owned host generator.

    Returns
    -------
    float
        Response in the closed unit interval.
    """
    return distribution(arguments).sample(rng)


__all__: list[str]=[]
