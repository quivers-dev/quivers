"""Extended-Beta formulas retain atom densities, gradients, and response means."""

import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from quivers.dsl import loads
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.formulas import FormulaToQVRModule, families, formula_from_data, fit
from quivers.inference import trace
from quivers.inference.mcmc.driver import MCMCResult
from quivers.qiec.program_runtime import run_program


def _oracle(eta, phi, kappa, values):
    mu = 1/(1+np.exp(-eta))
    a, b, scale = mu*phi, (1-mu)*phi, 1+2*kappa
    y = np.asarray(values)
    result = stats.beta.logpdf((y+kappa)/scale, a, b)-np.log(scale)
    result = np.where(y == 0, stats.beta.logcdf(kappa/scale, a, b), result)
    return np.where(y == 1, stats.beta.logcdf(kappa/scale, b, a), result)


@pytest.mark.parametrize("distributional", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_formula_native_reference_and_all_head_gradients_match_independent_law(distributional, weighted):
    frame = pd.DataFrame({"y": [0., .3, 1., .7], "x": [-1., 0., 1., 2.], "w": [0., .5, 2., 1.]})
    formula = formula_from_data("y ~ x", frame, weights="w" if weighted else None)
    lens = FormulaToQVRModule(
        families["xbeta"],
        fixed_parameters={} if distributional else {"phi": 4., "kappa": .2},
        distributional={"phi": formula, "kappa": formula} if distributional else None,
    )
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    source = module_to_source(module)
    native = loads(source).morphism
    checked = lower_qvr_to_qiec(parse(source))
    values = {"intercept": -.4, "beta_x": .2}
    if distributional:
        values |= {"dpar_phi_intercept": math.log(4.), "dpar_phi_beta_x": .1,
                   "dpar_kappa_intercept": math.log(.2), "dpar_kappa_beta_x": -.1}
    parameters = {name: torch.tensor(value, dtype=torch.float64, requires_grad=True) for name, value in values.items()}
    columns = lens.fixed_column_observations(formula)
    observations = columns | {"y": torch.tensor(frame.y.to_numpy(), dtype=torch.float64)} | parameters
    execution = trace(native, torch.zeros(1, 1), observations)
    actual = execution.sites["y"].log_prob.sum()

    def expected(arguments):
        x = frame.x.to_numpy()
        eta = arguments["intercept"]+arguments["beta_x"]*x
        phi = np.exp(arguments["dpar_phi_intercept"]+arguments["dpar_phi_beta_x"]*x) if distributional else 4.
        kappa = np.exp(arguments["dpar_kappa_intercept"]+arguments["dpar_kappa_beta_x"]*x) if distributional else .2
        return (_oracle(eta, phi, kappa, frame.y)*(frame.w.to_numpy() if weighted else 1)).sum()

    assert actual.item() == pytest.approx(expected(values), abs=3e-7)
    gradients = torch.autograd.grad(actual, tuple(parameters.values()))
    for name, gradient in zip(parameters, gradients, strict=True):
        high, low = dict(values), dict(values)
        high[name] += 1e-5
        low[name] -= 1e-5
        assert gradient.item() == pytest.approx((expected(high)-expected(low))/2e-5, rel=2e-5, abs=3e-6)
    data = {name: tuple(tensor.double().tolist()) for name, tensor in columns.items()} | {"y": tuple(frame.y)}
    reference = run_program(checked, "model", sites=values, data=data)
    assert reference.log_joint == pytest.approx(execution.log_joint.sum().item(), abs=3e-6)


def test_xbeta_predictions_use_censored_response_mean_on_new_rows():
    frame = pd.DataFrame({"y": [0., .3, 1., .7], "x": [-1., 0., 1., 2.]})
    result = fit("y~x", data=frame, family="xbeta", fixed_parameters={"phi": 4., "kappa": .2}, method="svi", num_steps=1, seed=9)
    result = result.with_(posterior=MCMCResult(
        samples={"intercept": torch.full((2, 3), -.4), "beta_x": torch.full((2, 3), .2)},
        log_densities=torch.zeros(2, 3), acceptance_rates=torch.ones(2), divergence_counts=torch.zeros(2, dtype=torch.long),
        num_warmup=0, num_samples=3,
    ))
    fresh = pd.DataFrame({"x": [-1., 2.]})
    predicted = result.posterior_epred(new_data=fresh)
    from scipy.integrate import quad
    expected = []
    for eta in [-.6, 0.]:
        mu, kappa, phi = 1/(1+math.exp(-eta)), .2, 4.
        a, b, scale = mu*phi, (1-mu)*phi, 1+2*kappa
        expected.append(stats.beta.sf((1+kappa)/scale,a,b)+quad(lambda y:y*stats.beta.pdf((y+kappa)/scale,a,b)/scale,0,1)[0])
    np.testing.assert_allclose(predicted.detach().numpy(), np.broadcast_to(expected, (2,3,2)), atol=3e-7)
    assert result.posterior_predict(new_data=fresh, seed=14).shape == (2,3,2)
