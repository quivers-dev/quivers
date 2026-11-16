"""Shared positive-tail coefficients for Gaussian divided differences."""

# Differentiating the integral representation of erfcx gives an alternating
# expansion for -erfcx'(x). At x>=8 its next omitted term bounds the error;
# 32 terms leave a relative remainder below 3e-22 at the boundary.
_coefficients = [1.0]
for _degree in range(1, 32):
    _coefficients.append(-_coefficients[-1] * (2 * _degree + 1))
_ERFCX_DERIVATIVE_COEFFICIENTS = tuple(_coefficients)


def erfcx_derivative_polynomial(value):
    """Evaluate the common asymptotic derivative polynomial by Horner's rule.

    Parameters
    ----------
    value : float or Tensor
        One over twice a squared argument at least eight.

    Returns
    -------
    float or Tensor
        Polynomial multiplying 1/(sqrt(pi)*x**2).
    """
    result = _ERFCX_DERIVATIVE_COEFFICIENTS[-1]
    for coefficient in reversed(_ERFCX_DERIVATIVE_COEFFICIENTS[:-1]):
        result = coefficient + value * result
    return result


def erfcx_denominator(value):
    """Evaluate the positive erfcx continued-fraction denominator.

    Parameters
    ----------
    value : float or Tensor
        Argument at least five.

    Returns
    -------
    float or Tensor
        Denominator multiplying sqrt(pi) in the scaled error function.
    """
    fraction = 0.0
    for index in range(40, 0, -1):
        fraction = (index / 2) / (value + fraction)
    return value + fraction


__all__ = []
