"""brms evidence, effects and predictive-check methods shared by formula fits.

`EvidenceEffectsMethods` is a base of `BayesianFit` and `MultivariateFit`.
It collects the marginal-likelihood methods of `quivers.formulas._evidence`
(``bridge_sampler``, ``bayes_factor``), brms's automatic smooth effects
(``conditional_smooths``), predictive checks (``pp_check``), and the MCMC
plots brms draws with bayesplot (``mcmc_plot``, ``plot``, ``pairs``),
drawn with ArviZ.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, cast

from narwhals.typing import IntoDataFrame

from quivers.formulas._brms_effects import (
    BrmsConditionalEffects,
    IntCondition,
    brms_conditional_smooths,
)
from quivers.formulas._evidence import EvidenceMethods
from quivers.formulas._mcmc_plot import PlotResult, mcmc_plot
from quivers.formulas._pp_check import PPCheck, Prefix, StatisticSpec, pp_check

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit
    from quivers.formulas._joint_fit import MultivariateFit


class EvidenceEffectsMethods(EvidenceMethods):
    """brms evidence, effects and predictive-check methods of formula fits."""

    def conditional_smooths(
        self,
        smooths: str | Sequence[str] | None = None,
        *,
        resp: str | Sequence[str] | None = None,
        int_conditions: Mapping[str, IntCondition] | None = None,
        prob: float = 0.95,
        spaghetti: bool = False,
        surface: bool = True,
        resolution: int = 100,
        too_far: float = 0.0,
        ndraws: int | None = None,
        draw_ids: Sequence[int] | None = None,
        seed: int = 0,
    ) -> BrmsConditionalEffects:
        """Compute every smooth's contribution on a grid, as brms ``conditional_smooths``.

        Parameters
        ----------
        smooths : str or Sequence[str] or None
            Smooth term labels; None for every ``s()`` and ``t2()`` term.
        resp : str or Sequence[str] or None
            Responses of a joint fit, as brms's ``resp``; None for all. Must
            be None for a single-response fit.
        int_conditions : Mapping or None
            Values of covariates or ``by`` variables.
        prob : float
            Interval probability.
        spaghetti : bool
            Also return per-draw curves.
        surface : bool
            Grid a numeric second covariate over its range.
        resolution : int
            Grid size per numeric covariate.
        too_far : float
            Drop surface points farther than this from the data.
        ndraws : int or None
            Random subset of draws.
        draw_ids : Sequence[int] or None
            Explicit 0-based draws.
        seed : int
            Seed of draw selection.

        Returns
        -------
        BrmsConditionalEffects
            One frame per smooth term.

        Raises
        ------
        ValueError
            If the fit has no matching smooth term, or ``resp`` is given for
            a single-response fit.
        """
        return brms_conditional_smooths(
            cast("BayesianFit | MultivariateFit", self),
            smooths,
            resp=resp,
            int_conditions=int_conditions,
            prob=prob,
            spaghetti=spaghetti,
            surface=surface,
            resolution=resolution,
            too_far=too_far,
            ndraws=ndraws,
            draw_ids=draw_ids,
            seed=seed,
        )

    def pp_check(
        self,
        type: str = "dens_overlay",
        *,
        ndraws: int | None = None,
        prefix: Prefix = "ppc",
        group: str | None = None,
        x: str | None = None,
        newdata: IntoDataFrame | None = None,
        resp: str | None = None,
        draw_ids: Sequence[int] | None = None,
        stat: StatisticSpec | None = None,
        prob: float = 0.5,
        prob_outer: float = 0.9,
        bins: int | None = None,
        freq: bool = True,
        style: Literal["standing", "hanging", "suspended"] = "standing",
        seed: int = 0,
    ) -> PPCheck:
        """Prepare a predictive check, as brms ``pp_check``.

        Parameters
        ----------
        type : str
            bayesplot type without prefix.
        ndraws : int or None
            Random draws; brms's default by type.
        prefix : {"ppc", "ppd"}
            Compare with observed data, or show the predictive distribution.
        group, x : str or None
            Grouping variable and covariate.
        newdata : IntoDataFrame or None
            Prediction rows.
        resp : str or None
            Joint response.
        draw_ids : Sequence[int] or None
            Explicit 0-based draws.
        stat : str or Callable or tuple or None
            Test statistic(s).
        prob, prob_outer : float
            Interval probabilities.
        bins : int or None
            Bins of ``error_binned``.
        freq : bool
            Counts instead of proportions for ``bars``.
        style : {"standing", "hanging", "suspended"}
            Rootogram style.
        seed : int
            Seed of draw selection and predictions.

        Returns
        -------
        PPCheck
            The data behind the plot.

        Raises
        ------
        ValueError
            If the type or its required arguments are invalid.
        """
        return pp_check(
            cast("BayesianFit | MultivariateFit", self),
            type,
            ndraws=ndraws,
            prefix=prefix,
            group=group,
            x=x,
            newdata=newdata,
            resp=resp,
            draw_ids=draw_ids,
            stat=stat,
            prob=prob,
            prob_outer=prob_outer,
            bins=bins,
            freq=freq,
            style=style,
            seed=seed,
        )

    def mcmc_plot(
        self,
        type: str = "intervals",
        *,
        variable: str | Sequence[str] | None = None,
        backend: str,
        **kwargs: str | int | float | bool,
    ) -> PlotResult:
        """Draw an MCMC diagnostic with ArviZ, as brms ``mcmc_plot``.

        Parameters
        ----------
        type : str
            bayesplot ``mcmc_*`` type without prefix.
        variable : str or Sequence[str] or None
            brms variable names or regular expressions.
        backend : str
            ArviZ backend.
        **kwargs : str or int or float or bool
            ArviZ keyword arguments.

        Returns
        -------
        arviz_plots.PlotCollection or arviz_plots.PlotMatrix
            The plot.

        Raises
        ------
        ValueError
            If the type is unknown or has no ArviZ counterpart.
        """
        return mcmc_plot(
            cast("BayesianFit | MultivariateFit", self),
            type,
            variable=variable,
            backend=backend,
            **kwargs,
        )

    def plot(
        self,
        *,
        variable: str | Sequence[str] | None = None,
        backend: str,
        **kwargs: str | int | float | bool,
    ) -> PlotResult:
        """Draw marginal densities and trace plots, as brms ``plot``.

        Parameters
        ----------
        variable : str or Sequence[str] or None
            brms variable names or regular expressions.
        backend : str
            ArviZ backend.
        **kwargs : str or int or float or bool
            ArviZ keyword arguments.

        Returns
        -------
        arviz_plots.PlotCollection
            ArviZ ``plot_trace_dist``.
        """
        return self.mcmc_plot("combo", variable=variable, backend=backend, **kwargs)

    def pairs(
        self,
        *,
        variable: str | Sequence[str] | None = None,
        backend: str,
        **kwargs: str | int | float | bool,
    ) -> PlotResult:
        """Draw pairwise scatter plots of draws, as brms ``pairs``.

        Parameters
        ----------
        variable : str or Sequence[str] or None
            brms variable names or regular expressions.
        backend : str
            ArviZ backend.
        **kwargs : str or int or float or bool
            ArviZ keyword arguments.

        Returns
        -------
        arviz_plots.PlotMatrix
            ArviZ ``plot_pair``.
        """
        return self.mcmc_plot("pairs", variable=variable, backend=backend, **kwargs)


__all__ = ["EvidenceEffectsMethods"]
