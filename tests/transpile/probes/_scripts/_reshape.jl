# Shared in-container reshape helper for Julia probes.
#
# Loads `/io/shapes.json` and `/io/dtypes.json` (each absent file
# resolves to an empty dict so legacy probe runs are a no-op),
# rebuilds nested arrays from row-major flat lists, and casts
# leaves to Int / Float64 per dtype.
#
# `reshape_value` is the per-name entry point; `reshape_point`
# takes a JSON3.Object with `params` / `data` sections and returns
# a `Dict{Symbol,Any}` ready for the runtime-specific coercion in
# the calling probe (turing.jl, gen.jl).

using JSON3

function _load_json_dict(path::AbstractString)
    if !isfile(path)
        return Dict{String,Any}()
    end
    raw = JSON3.read(read(path, String))
    # JSON3.Object iterates as (Symbol, value); rebuild with
    # explicitly stringified keys so the downstream
    # `Dict{String,Vector{Int}}` / `Dict{String,String}` typed
    # conversions accept the entries.
    return Dict{String,Any}(String(k) => v for (k, v) in pairs(raw))
end

function load_tables(io::AbstractString)
    shapes_raw = _load_json_dict(joinpath(io, "shapes.json"))
    dtypes_raw = _load_json_dict(joinpath(io, "dtypes.json"))
    shapes = Dict{String,Vector{Int}}(
        k => [Int(x) for x in v] for (k, v) in shapes_raw
    )
    dtypes = Dict{String,String}(k => String(v) for (k, v) in dtypes_raw)
    return shapes, dtypes
end

function _flat_to_nested(flat::AbstractVector, shape::Vector{Int})
    if isempty(shape)
        length(flat) == 1 || error(
            "scalar shape but length(flat)=$(length(flat))"
        )
        return flat[1]
    end
    expected = prod(shape)
    length(flat) == expected || error(
        "flat length $(length(flat)) does not match shape $shape "
        * "(expected $expected)"
    )
    if length(shape) == 1
        return collect(flat)
    end
    stride_ = div(expected, shape[1])
    return [
        _flat_to_nested(flat[(i - 1) * stride_ + 1:i * stride_], shape[2:end])
        for i in 1:shape[1]
    ]
end

function _cast_leaves(value, dtype::String)
    if value isa AbstractArray
        return [_cast_leaves(v, dtype) for v in value]
    end
    return dtype == "int" ? Int(value) : Float64(value)
end

function reshape_value(
    name::String, value, shapes::Dict, dtypes::Dict,
)
    out = value
    if haskey(shapes, name)
        if out isa Real
            out = _flat_to_nested([out], shapes[name])
        elseif out isa AbstractArray
            out = _flat_to_nested(collect(out), shapes[name])
        elseif out isa JSON3.Array
            out = _flat_to_nested([x for x in out], shapes[name])
        end
    end
    if haskey(dtypes, name)
        out = _cast_leaves(out, dtypes[name])
    end
    return out
end

function reshape_point(pt, shapes::Dict, dtypes::Dict)
    out = Dict{String,Dict{Symbol,Any}}()
    for section in ("params", "data")
        sec = haskey(pt, section) ? pt[section] : Dict{Symbol,Any}()
        out[section] = Dict{Symbol,Any}(
            Symbol(k) => reshape_value(String(k), v, shapes, dtypes)
            for (k, v) in sec
        )
    end
    return out
end

# `reshape_value` returns a nested `Vector` of `Vector`s for a rank >= 2
# name, which is the shape a JSON payload naturally rebuilds into but
# not one Distributions.jl or Gen accepts. `native_array` projects that
# nesting onto a dense `Array{T,N}` with the same axis order. Julia is
# column-major and the wire payload is row-major, so the leaves reshape
# against the reversed dimension list and then permute back.
function nested_shape(value)
    dims = Int[]
    cur = value
    while cur isa AbstractArray
        push!(dims, length(cur))
        isempty(cur) && break
        cur = first(cur)
    end
    return dims
end

function _collect_leaves!(out, value)
    if value isa AbstractArray
        for x in value
            _collect_leaves!(out, x)
        end
    else
        push!(out, value)
    end
    return out
end

function nested_leaves(value)
    return _collect_leaves!(Any[], value)
end

function native_array(value)
    if !(value isa AbstractArray)
        if value isa Integer
            return Int(value)
        elseif value isa Real
            return Float64(value)
        end
        return value
    end
    if ndims(value) > 1
        # Already a dense multi-dimensional array; only the element
        # type needs projecting.
        elt = all(x -> x isa Integer, value) ? Int : Float64
        return Array{elt}(value)
    end
    dims = nested_shape(value)
    leaves = nested_leaves(value)
    elt = all(x -> x isa Integer, leaves) ? Int : Float64
    typed = elt[elt(x) for x in leaves]
    length(dims) <= 1 && return typed
    return permutedims(
        reshape(typed, reverse(dims)...), collect(length(dims):-1:1),
    )
end

# ---------------------------------------------------------------------
# Simplex marshalling.
#
# A simplex-valued latent needs one step beyond the reshape before it
# reaches a Julia target. `Distributions.Dirichlet` and
# `Distributions.Categorical` both screen their probability vector
# through a support check, and the point payload carries each row as
# float32-rounded doubles, which miss a sum of one by roughly
# `eps(Float32)`. A row that is a simplex in the reference's arithmetic
# is therefore not one in Distributions.jl's: `Categorical` aborts the
# run outright and `Dirichlet` quietly returns `-Inf`, which is a
# finite-looking wrong answer rather than an error. Rescaling each row
# by its own sum is the smallest correction that restores the
# constraint, moving every component by at most the float32 rounding
# already present in the wire form.
#
# Each Julia probe supplies its own set of simplex-valued site names,
# read off its renderer's own draw syntax (`~ Dirichlet(` for Turing,
# `@trace dirichlet(...) :name` for Gen); the rescaling and its
# tolerance live here so both probes police the same boundary.
# ---------------------------------------------------------------------

# How far a wire-form simplex row may sit from summing to one before
# the probe calls it a defect rather than a rounding artefact. Anything
# above this floor is a wrong value rather than a wrong representation,
# and rescaling it would turn a mis-shipped parameter into a silently
# valid one.
const SIMPLEX_SUM_TOLERANCE = 1e-5

# Scale every innermost row of `value` to sum to exactly one.
function renormalise_rows(name::String, value)
    if value isa AbstractArray && !isempty(value) &&
            first(value) isa AbstractArray
        return [renormalise_rows(name, row) for row in value]
    end
    row = [Float64(x) for x in value]
    total = sum(row)
    abs(total - 1.0) <= SIMPLEX_SUM_TOLERANCE || error(
        "probe: the point's $(name) row sums to $(total), " *
        "which is further from one than the float32 rounding of the " *
        "wire form can explain (tolerance " *
        "$(SIMPLEX_SUM_TOLERANCE)). " *
        "The emitted model draws $(name) from a Dirichlet, so this " *
        "row is not the value the reference scored."
    )
    return row ./ total
end

# Return `params` with every simplex-typed entry rescaled and every
# other entry passed through untouched.
function renormalise_simplex_params(params, names)
    return Dict{Symbol,Any}(
        k => (
            String(k) in names && v isa AbstractArray ?
                renormalise_rows(String(k), v) : v
        )
        for (k, v) in params
    )
end

# ---------------------------------------------------------------------
# The export channel.
#
# `/io/export_names.json` holds the QVR program's return-variable
# names, in declaration order. A probe that finds the file reads the
# exported value out of the emitted program's own return surface (the
# `@model` return under `condition`, the second element of
# `Gen.assess`) and reports one entry per name per point.
# ---------------------------------------------------------------------

function load_export_names(io::AbstractString)
    path = joinpath(io, "export_names.json")
    isfile(path) || return String[]
    return [String(x) for x in JSON3.read(read(path, String))]
end

# Nest a returned value row-major so JSON3 emits the same layout the
# host reference produced. Writing a Julia `Matrix` directly would
# serialise its column-major storage as one flat array, which compares
# element-for-element against a transposed reference.
function export_nested(v)
    if v isa AbstractArray
        if ndims(v) == 0
            return export_nested(v[])
        elseif ndims(v) == 1
            return [export_nested(x) for x in v]
        end
        rest = ntuple(_ -> Colon(), ndims(v) - 1)
        return [export_nested(v[i, rest...]) for i in axes(v, 1)]
    elseif v isa Tuple
        return [export_nested(x) for x in v]
    elseif v isa Bool
        return Int(v)
    elseif v isa Integer
        return Int(v)
    end
    return Float64(v)
end

function export_payload(names::Vector{String}, returned)
    isempty(names) && error(
        "export_payload called with no export names; the caller did " *
        "not ship /io/export_names.json and the probe must not " *
        "report an export channel."
    )
    returned === nothing && error(
        "the emitted model returns nothing where the QVR program " *
        "exports $(names). A transpilation that drops the return " *
        "clause emits a program denoting the right joint and the " *
        "wrong kernel."
    )
    if length(names) == 1
        return [export_nested(returned)]
    end
    returned isa Tuple || error(
        "the emitted model returns a single value where the QVR " *
        "program exports $(length(names)) ($(names)). The renderer " *
        "dropped part of the program's return clause."
    )
    length(returned) == length(names) || error(
        "the emitted model returns $(length(returned)) value(s) " *
        "where the QVR program exports $(length(names)) ($(names))."
    )
    return [export_nested(x) for x in returned]
end
