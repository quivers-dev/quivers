# In-container Gen.jl probe.
#
# Evals the transpiled `@gen function model(...) ... end` source,
# extracts each trace address (its name and how many indices it
# carries) by scanning the source for the `@trace ... :NAME` /
# `@trace ... (:NAME, m_AXIS...)` shapes the Gen renderer emits, then
# builds a `Gen.choicemap` that constrains every `params` value and
# every `data` value whose key matches a trace-site name (so
# covariate-only function args are skipped), generates a trace under
# those constraints, and reads its score as the log-density at each
# test point. A scored weight (a `score` step, or a helper's scored
# weight) is traced as a factor choice under the `:qvr_factor`
# address, a one-point distribution whose log density is the weight;
# such a choice has nothing to pin, and it is the only choice the
# probe leaves unconstrained: any other unconstrained address would
# be a fresh draw, and the probe fails rather than score it.
#
# Every payload is reshaped to its declared shape first. The address
# arity decides how far into that shape the constraint walk descends:
# a rank-2 latent traced at `(:Z_mat, m_Item, m_LatentDim)` gets one
# scalar per cell, a simplex-valued latent traced at `(:theta, m_Doc)`
# gets the whole row at one address.
#
# A simplex-valued latent needs one marshalling step beyond the
# reshape, which `_reshape.jl` supplies: every name the emitted model
# draws from `dirichlet` is rescaled by its own row sum on the way in.
# `Gen.dirichlet` scores through `Distributions.Dirichlet`, whose
# support check rejects a row that misses a sum of one by the float32
# rounding the wire form carries, and it reports that rejection as a
# `-Inf` log-density rather than as an error. An unrescaled row
# therefore leaves the probe as a wrong answer at some points of a set
# and a correct one at others, which reads as a non-constant offset
# rather than as the marshalling defect it is.
using Gen, Distributions, JSON3

include("/io/_reshape.jl")

function _trace_site_arities(source::String)
    # The Gen renderer emits trace sites in two forms (and two
    # macro-call shapes, space-separated and parenthesised):
    #
    #   space, scalar:    `@trace <dist> :name`
    #   space, batched:   `@trace <dist> (:name, m_<axis>...)`
    #   parens, scalar:   `@trace(<dist>, :name)`
    #   parens, batched:  `@trace(<dist>, (:name, m_<axis>...))`
    #
    # The number of index components in the address is the number of
    # plate axes the renderer wrapped a `for` loop around, and it is a
    # contract between the emitted model and the choicemap the probe
    # builds: a rank-2 latent traced at `(:Z_mat, m_Item, m_LatentDim)`
    # needs one scalar constraint per cell, while a simplex-valued
    # latent traced at `(:theta, m_Doc)` needs the whole row at one
    # address. Record the arity per site so the constraint walk knows
    # how far to descend into the reshaped value.
    arities = Dict{Symbol,Int}()
    # Scalar address: `:name` not followed by a trailing word
    # character (handled by Julia's regex word-boundary), and not
    # part of a tuple (so we exclude addresses preceded by `(`).
    scalar_re = r"@trace[\s(][^\n]*?[\s,]:([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*,)"
    for m in eachmatch(scalar_re, source)
        arities[Symbol(m.captures[1])] = 0
    end
    # Batched address: `(:name, i_1, ..., i_k)`, the `:` preceded by
    # `(` (possibly with whitespace). The index components the renderer
    # emits are loop variables, so a plain identifier list suffices.
    batched_re = r"@trace[\s(][^\n]*?\(\s*:([A-Za-z_][A-Za-z0-9_]*)\s*((?:,\s*[A-Za-z_][A-Za-z0-9_]*\s*)+)\)"
    for m in eachmatch(batched_re, source)
        arities[Symbol(m.captures[1])] = count(==(','), m.captures[2])
    end
    return arities
end

# The Gen renderer spells a simplex-valued draw as a `dirichlet` call
# inside a `@trace`, in the same four macro-call shapes
# `_trace_site_arities` reads, and the site name lives in the address
# rather than on the left of the assignment. Both address forms carry
# it as the first symbol after the distribution's own argument list.
const _DIRICHLET_SCALAR_RE =
    r"@trace[\s(]\s*dirichlet\b[^\n]*?[\s,]:([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*,)"
const _DIRICHLET_BATCHED_RE =
    r"@trace[\s(]\s*dirichlet\b[^\n]*?\(\s*:([A-Za-z_][A-Za-z0-9_]*)\s*(?:,\s*[A-Za-z_][A-Za-z0-9_]*\s*)+\)"

# Names the emitted model draws from `dirichlet`, whose value is a
# point of the simplex and reaches Distributions.jl through a support
# check.
function simplex_site_names(source::AbstractString)
    names = Set{String}()
    for re in (_DIRICHLET_SCALAR_RE, _DIRICHLET_BATCHED_RE)
        for m in eachmatch(re, source)
            push!(names, String(m.captures[1]))
        end
    end
    return names
end

function _set_constraint_at!(constraints, name::Symbol, prefix, value, depth::Int)
    # Descend exactly `depth` axes, one per `for` loop the renderer
    # wrapped around the trace site, accumulating the index prefix.
    # Whatever remains at the bottom is the value the site draws, be it
    # a scalar or a simplex row, and it goes in at that one address.
    if depth > 0
        if !(value isa AbstractArray)
            error(
                "gen probe: `$name` is traced under a $depth-index " *
                "address but its value runs out of axes at prefix " *
                "$prefix"
            )
        end
        for (i, elem) in enumerate(value)
            _set_constraint_at!(
                constraints, name, (prefix..., i), elem, depth - 1,
            )
        end
        return constraints
    end
    address = isempty(prefix) ? name : (name, prefix...)
    Gen.set_value!(constraints, address, native_array(value))
    return constraints
end

function _set_constraint!(constraints, name::Symbol, value, arity::Int)
    return _set_constraint_at!(constraints, name, (), value, arity)
end

# The top-level addresses of `choices` that `constraints` does not
# pin, the factor namespace excepted.
function _unclamped_addresses(choices, constraints)
    unclamped = Any[]
    for (address, _) in Gen.get_values_shallow(choices)
        Gen.has_value(constraints, address) || push!(unclamped, address)
    end
    for (address, _) in Gen.get_submaps_shallow(choices)
        address == :qvr_factor && continue
        Gen.has_submap(constraints, address) || push!(unclamped, address)
    end
    return unclamped
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

    # Use `include_string` (whole-file evaluation) instead of
    # `Meta.parse` (single expression) so source files with multiple
    # top-level statements -- runtime helper grafts (e.g. the
    # `TruncatedNormalDist` Gen.Distribution lift) followed by the
    # `@gen function model() ... end` -- load correctly. `Meta.parse`
    # only consumes one expression and rejects the rest with
    # "extra token after end of expression".
    Base.include_string(Main, source)
    arities = _trace_site_arities(source)

    log_densities = Float64[]
    exports = []
    for pt in points
        # Rebuild each flat row-major payload at its declared shape
        # before it reaches either the model signature or the
        # choicemap: a rank-2 latent is traced under one address per
        # cell, so the constraint builder needs the nesting the flat
        # list erases.
        reshaped = reshape_point(pt, shapes, dtypes)
        data = reshaped["data"]
        params = renormalise_simplex_params(
            reshaped["params"], simplex_names,
        )
        # Pass observed values as positional args sorted by name to
        # match the renderer's alphabetical signature ordering.
        args = Tuple(
            native_array(data[k]) for k in sort(collect(keys(data)))
        )
        constraints = Gen.choicemap()
        for (k, v) in pairs(params)
            sym = Symbol(k)
            # A site the model body traces carries the arity its
            # address was scanned with; a helper's site is traced under
            # its label alone, so a param the scan did not see is a
            # scalar address.
            _set_constraint!(constraints, sym, v, get(arities, sym, 0))
        end
        for (k, v) in pairs(data)
            sym = Symbol(k)
            haskey(arities, sym) && _set_constraint!(
                constraints, sym, v, arities[sym],
            )
        end
        # `Base.invokelatest` is required because the `@gen` macro
        # registered the model definition in a newer world age than
        # the one this main() began executing in; without it Julia
        # rejects the call with a "method too new" MethodError.
        trace, _ = Base.invokelatest(
            Gen.generate, Main.model, args, constraints,
        )
        unclamped = _unclamped_addresses(Gen.get_choices(trace), constraints)
        if !isempty(unclamped)
            error(
                "gen probe: unclamped choice(s) $(unclamped); every " *
                "traced address must be supplied through the point's " *
                "params or data"
            )
        end
        push!(log_densities, Float64(Gen.get_score(trace)))
        returned = Gen.get_retval(trace)
        if !isempty(export_names)
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
