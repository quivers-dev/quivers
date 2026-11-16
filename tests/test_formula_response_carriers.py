"""Formula response carriers preserve exact counts through fitting and replay."""

import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from quivers.formulas import families, fit, formula_from_data
from quivers.formulas._response import response_tensor
from quivers.formulas._prediction import prepare_prediction
from quivers.inference.trace import trace
from quivers.qiec.families import family as semantic_family
from quivers.qiec.types import BOOL, INT


@pytest.mark.parametrize('name',sorted(families))
def test_every_formula_family_uses_its_semantic_response_element(name):
    family=families[name]
    record=semantic_family(family.observe_family)
    frame=pd.DataFrame({'y':[0,1,2],'a':[1,2,3],'b':[2,1,2]})
    vector=record.event_rank==1 and family.response is not None and family.response.kind!='categorical'
    parsed=formula_from_data('cbind(a,b)~1' if vector else 'y~1',frame)
    if record.element==BOOL:
        parsed=formula_from_data('y~1',frame.assign(y=[0,1,0]))
    tensor=response_tensor(parsed,family)
    assert tensor.shape==parsed.response_values.shape
    if record.element==INT: assert tensor.dtype==torch.int64
    elif record.element==BOOL: assert tensor.dtype==torch.bool
    else: assert tensor.dtype==(torch.float64 if vector else torch.float32)


@pytest.mark.parametrize('name',['poisson','binomial','negative_binomial'])
@pytest.mark.parametrize('count',[1_000_001,33_554_433])
def test_large_scalar_count_is_exact_in_fit_source_and_new_data_replay(name,count):
    frame=pd.DataFrame({'y':np.array([count,count+2],dtype=np.int64)})
    if name=='binomial':
        trials=2*count+5
        intercept=.1
        kwargs={'binomial_trials':trials}
        oracle=stats.binom(trials,1/(1+math.exp(-intercept)))
    elif name=='negative_binomial':
        intercept=math.log(count)
        kwargs={'fixed_parameters':{'disp':2.}}
        oracle=stats.nbinom(2.,2/(2+math.exp(intercept)))
    else:
        intercept=math.log(count)
        kwargs={}
        oracle=stats.poisson(math.exp(intercept))
    result=fit('y~1',data=frame,family=name,method='svi',num_steps=0,initial_values={'intercept':intercept},**kwargs)
    assert result.formula.response_values.dtype==np.int64
    assert result.observations['y'].dtype==torch.int64
    assert result.observations['y'].tolist()==[count,count+2]
    coefficient=torch.tensor(intercept,dtype=torch.float64,requires_grad=True)
    recorded=trace(result.program,torch.zeros(1,1),dict(result.observations)|{'intercept':coefficient})
    actual=recorded.sites['y'].sampleable.raw_log_prob_pointwise(result.observations['y'])
    torch.testing.assert_close(actual,torch.tensor(oracle.logpmf(np.array([count,count+2]))),atol=5e-7,rtol=1e-8)
    context=prepare_prediction(result,pd.DataFrame(index=range(3)),None,trials=kwargs.get('binomial_trials'))
    assert context.observations['y'].dtype==torch.int64
    assert context.observations['y'].tolist()==([0]*3 if name=='binomial' else [count]*3)


@pytest.mark.parametrize('name',['poisson','binomial','negative_binomial','categorical'])
def test_fractional_integer_response_is_rejected_before_conversion(name):
    with pytest.raises(ValueError,match='integer responses'):
        response_tensor(formula_from_data('y~1',pd.DataFrame({'y':[0.,1.5]})),families[name])


@pytest.mark.parametrize('count',[2**53+1,np.iinfo(np.int64).max])
def test_integer_response_capacity_validation_runs_before_any_float_cast(count):
    parsed=formula_from_data('y~1',pd.DataFrame({'y':np.array([count],dtype=np.int64)}))
    assert int(parsed.response_values[0])==count
    with pytest.raises(ValueError,match='scoring capacity'):
        response_tensor(parsed,families['poisson'])


def test_boolean_binary_codes_and_recorded_count_endpoints_keep_their_carriers():
    binary=fit('y~1',data=pd.DataFrame({'y':[0,1,0,1]}),family='bernoulli',method='svi',num_steps=0)
    assert binary.observations['y'].dtype==torch.bool
    assert binary.posterior_predict(num_samples=2).shape==(1,2,4)
    frame=pd.DataFrame({'y':np.array([-1,0,33_554_433],dtype=np.int64),'c':[1,0,1]})
    count=fit('y|cens(c)~1',data=frame,family='poisson',method='svi',num_steps=0)
    assert count.observations['y'].dtype==torch.int64
    assert count.observations['y'].tolist()==[-1,0,33_554_433]
