"""Shared probability primitives for standalone pymc runtime generation."""

# ruff: noqa: F821
# pyright: reportUndefinedVariable=false


def _qvr_select(condition, then, otherwise):
    """Select with a host Boolean or a symbolic PyTensor condition."""
    pt = pymc.pytensorf.pt
    if isinstance(condition, bool):
        condition = pt.constant(condition, dtype="bool")
    return pt.switch(condition, then, otherwise)


def _qvr_probability_check(value, valid, name, integer=False):
    from pytensor.raise_op import Assert

    valid_type = (
        value.dtype.startswith(("int", "uint"))
        if integer
        else value.dtype.startswith("float")
    )
    if not valid_type:
        raise ValueError(
            name + (" requires integer values" if integer else " requires real values")
        )
    return Assert(name + " received a value outside its domain")(value, valid.all())


def _qvr_log_probability(value):
    pt = pymc.pytensorf.pt
    value = pt.as_tensor_variable(value)
    value = _qvr_probability_check(
        value,
        ~pt.isnan(value) & ~pt.isinf(value) & (value >= 0) & (value <= 1),
        "log_probability",
    )
    safe = pt.switch(value > 0, value, pt.ones_like(value))
    return pt.switch(value > 0, pt.log(safe), -float("inf"))


def _qvr_log1mexp(value):
    pt = pymc.pytensorf.pt
    value = pt.as_tensor_variable(value)
    value = _qvr_probability_check(value, ~pt.isnan(value) & (value <= 0), "log1mexp")
    safe = pt.switch(~pt.isinf(value) & (value < 0), value, -pt.ones_like(value))
    near = safe > -0.6931471805599453
    a = pt.switch(near, safe, -pt.ones_like(value))
    b = pt.switch(near, -pt.ones_like(value), safe)
    result = pt.switch(near, pt.log(-pt.expm1(a)), pt.log1p(-pt.exp(b)))
    return pt.switch(
        pt.eq(value, 0), -float("inf"), pt.switch(pt.isinf(value), 0.0, result)
    )


def _qvr_log_ndtr(value):
    pt = pymc.pytensorf.pt
    value = pt.as_tensor_variable(value)
    value = _qvr_probability_check(value, ~pt.isnan(value), "log_ndtr")
    safe = pt.switch(pt.isinf(value), pt.zeros_like(value), value)
    sqrt_two = pt.constant(1.4142135623730951, dtype=value.dtype)
    log_two = pt.constant(0.6931471805599453, dtype=value.dtype)
    negative = pt.switch(safe < 0, -safe, pt.zeros_like(safe))
    positive = pt.switch(safe < 0, pt.zeros_like(safe), safe)
    lower = pt.log(pt.erfcx(negative / sqrt_two)) - 0.5 * negative * negative - log_two
    upper = pt.log1p(-0.5 * pt.erfc(positive / sqrt_two))
    result = pt.switch(safe < 0, lower, upper)
    return pt.switch(pt.isinf(value), pt.switch(value < 0, -float("inf"), 0.0), result)


def _qvr_positive_parameter(value):
    pt = pymc.pytensorf.pt
    value = pt.as_tensor_variable(value)
    return _qvr_probability_check(
        value, ~pt.isnan(value) & ~pt.isinf(value) & (value > 0), "positive_parameter"
    )


def _qvr_observation_status(value):
    """Validate integer recorded-event codes without changing their carrier.

    Parameters
    ----------
    value
        Scalar or lifted integer code tensor.

    Returns
    -------
    integer tensor
        Codes minus one through two, unchanged.

    Raises
    ------
    ValueError
        If the carrier is not integer or an active code is invalid.
    """
    if isinstance(value, bool):
        raise ValueError("observation_status requires integer codes -1, 0, 1, or 2")
    pt = pymc.pytensorf.pt
    value = pt.as_tensor_variable(value)
    return _qvr_probability_check(
        value, (value >= -1) & (value <= 2), "observation_status", integer=True
    )


def _qvr_interval_upper(lower, upper):
    """Check an active interval and retain its upper-endpoint derivatives.

    Parameters
    ----------
    lower, upper
        Real scalar or lifted endpoints; lower may be infinite.

    Returns
    -------
    real tensor
        Finite upper endpoint, strictly above lower.

    Raises
    ------
    ValueError
        If the endpoint carrier or active interval is invalid.
    """
    pt = pymc.pytensorf.pt
    lower, upper = pt.as_tensor_variable(lower), pt.as_tensor_variable(upper)
    if not lower.dtype.startswith("float"):
        raise ValueError("interval_upper requires real endpoints")
    upper = pt.broadcast_arrays(lower, upper)[1]
    return _qvr_probability_check(
        upper,
        ~pt.isnan(lower) & ~pt.isnan(upper) & ~pt.isinf(upper) & (upper > lower),
        "interval_upper",
    )
