"""Independently asserted constructor-alternative capabilities for matrix tests."""

PARAMETERIZATION_REFUSALS: dict[tuple[str, str], str] = {}

# These laws have native and checked-reference fixtures above. Their
# external cells are actual measured capability refusals, not numeric claims.
for _fixture, _family, _rendered in (
    ("steps/observe_bernstein_hazard_log_rate", "BernsteinHazard", ()),
    *(
        ("steps/observe_multivariate_student_" + schema, "MultivariateStudentT", ())
        for schema in ("factor", "covariance", "precision")
    ),
    ("steps/observe_multinomial_logits", "Multinomial", ()),
    ("steps/observe_dirichlet_multinomial", "DirichletMultinomial", ()),
    ("steps/observe_com_poisson_log_rate", "COMPoisson", ()),
    ("steps/observe_alr_reference", "AdditiveLogRatioNormal", ()),
    (
        "steps/observe_relaxed_bernoulli_logits",
        "RelaxedBernoulli",
        ("edward2", "numpyro", "pyro"),
    ),
    (
        "steps/observe_onehot_categorical_logits",
        "OneHotCategorical",
        ("edward2", "numpyro", "pyro"),
    ),
    (
        "steps/observe_relaxed_onehot_logits",
        "RelaxedOneHotCategorical",
        ("edward2", "pyro"),
    ),
):
    for _target, _kind in (
        ("bugs", f"family:{_family}:"),
        ("church", f"family:{_family}:no-church-target"),
        ("edward2", f"family:{_family}:no-edward2-target"),
        ("gen", f"family:{_family}:no-gen-target"),
        ("jags", f"family:no-target-name:{_family}"),
        ("numpyro", f"family:no-target-name:{_family}"),
        ("pymc", f"family:{_family}"),
        ("pyro", f"family:{_family}"),
        ("stan", f"family:no-stan-target:{_family}"),
        ("turing", f"family:{_family}:"),
        ("webppl", f"family:no-webppl-target:{_family}"),
    ):
        if _target not in _rendered:
            PARAMETERIZATION_REFUSALS[(_fixture, _target)] = _kind


# Matrix factors are multiplied into covariances by the shared plan helper.
# The remaining refusals are the existing MatrixNormal/literal capabilities.
for _axis in ("row", "col"):
    for _target, _kind in (
        ("bugs", "arg:matrix-literal"),
        ("church", "arg:matrix-literal"),
        ("jags", "family:no-target-name:MatrixNormal"),
    ):
        PARAMETERIZATION_REFUSALS[
            ("steps/observe_matrix_normal_" + _axis + "_factor", _target)
        ] = _kind
