# In-container Turing.jl probe.
#
# Reads /io/source.jl + /io/points.json; evals the @model source
# and writes /io/result.json with `Turing.logjoint(model, theta)`
# at each point.
#
# When /io/export_names.json is present the probe also reports the
# program's exported value at each point. Turing's export surface is
# the `@model` function's own `return`, which DynamicPPL hands back
# when the model instance is called; conditioning on the point's
# latents first makes that value a deterministic function of the
# point rather than of the sampler's generator.
#
# A simplex-valued latent needs one marshalling step beyond the
# reshape, which `_reshape.jl` supplies: every name the emitted model
# draws from a `Dirichlet` is rescaled by its own row sum on the way
# in, and a row too far from one to be float32 rounding raises. All
# this probe adds is the Turing-specific reading of which sites those
# are.
#
# Julia world-age note: `Base.eval` introduces a new method (the
# `@model`-expanded `model` factory) at a world age newer than the
# enclosing `main`'s. Calling `model_factory(args...)` directly from
# `main` would dispatch in `main`'s captured world, where the new
# method is not visible ("method too new to be called from this
# world context"). `Base.invokelatest` re-resolves dispatch in the
# latest world so the freshly-eval'd model is callable; we apply it
# to every cross-eval boundary call (factory construction, JSON
# coercions on params/data, `Turing.logjoint`).
using Turing, Distributions, LinearAlgebra, JSON3

const DPPL = Turing.DynamicPPL

include("/io/_reshape.jl")

# `native_array` (from `_reshape.jl`) coerces a reshaped value into
# the native Julia container Distributions.jl / Turing.logjoint expect:
# `Vector{Float64}` for rank 1, `Matrix{Float64}` for rank 2, and an
# `Array{Float64,N}` for anything deeper. `reshape_value` hands back a
# nested vector rather than a multi-dimensional array, whose `ndims` is
# 1 whatever its true rank, so the projection reads the leaves rather
# than dispatching on `ndims`.
function _coerce_nt(d)
    pairs = Tuple((Symbol(k), native_array(v)) for (k, v) in d)
    return NamedTuple{Tuple(p[1] for p in pairs)}(Tuple(p[2] for p in pairs))
end

# The renderer spells a simplex-valued draw as `<name> ~ Dirichlet(...)`
# for an unplated one and `<name>[<subscript>] ~ Dirichlet(...)` for a
# plated one, so the site name is all this pattern needs.
const _DIRICHLET_SITE_RE =
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*~\s*Dirichlet\s*\("

# Names the emitted model draws from a `Dirichlet`, whose value is a
# point of the simplex and reaches Distributions.jl through a
# constraint check.
function simplex_site_names(source::AbstractString)
    return Set{String}(
        String(m.captures[1])
        for m in eachmatch(_DIRICHLET_SITE_RE, source)
    )
end

function main()
    source = read("/io/source.jl", String)
    points = JSON3.read(read("/io/points.json", String))
    shapes, dtypes = load_tables("/io")
    export_names = load_export_names("/io")
    # The point payload is in QVR's zero-based coding, and the emitted
    # program reads an index covariate or a class code one higher where
    # Julia counts from one, so the payload reaches the model as it is.
    simplex_names = simplex_site_names(source)

    # Eval the @model declaration in Main; the macro produces a
    # callable `model` symbol.
    # The source is a whole file: the model, and the QIEC runtime and
    # the helpers it calls when the program calls any.
    Base.include_string(Main, source)

    log_densities = Float64[]
    exports = []
    for pt in points
        reshaped = reshape_point(pt, shapes, dtypes)
        data = reshaped["data"]
        params = renormalise_simplex_params(
            reshaped["params"], simplex_names,
        )
        # Pass observed values as positional args (sorted by name to
        # match the python harness's convention).
        sorted_keys = sort(collect(keys(data)))
        args = Tuple(native_array(data[k]) for k in sorted_keys)
        model_instance = Base.invokelatest(Main.model, args...)
        theta = _coerce_nt(params)
        lp = Base.invokelatest(Turing.logjoint, model_instance, theta)
        push!(log_densities, Float64(lp))
        if !isempty(export_names)
            conditioned = Base.invokelatest(
                DPPL.condition, model_instance, theta,
            )
            returned = Base.invokelatest(conditioned)
            push!(exports, export_payload(export_names, returned))
        end
    end

    open("/io/result.json", "w") do io
        if isempty(export_names)
            JSON3.write(io, (log_densities = log_densities,))
        else
            JSON3.write(
                io, (log_densities = log_densities, exports = exports),
            )
        end
    end
end

main()
