# QIEC distribution bridge for generated Turing modules. A QIEC tensor is
# a nested tuple until it reaches a distribution, where it becomes an
# array with the tuple's outermost axis first; a log density is
# Distributions.jl's `logpdf` at the converted value. A family whose
# Distributions.jl support is shifted against the QIEC convention is
# wrapped with its offset, which is added to the scored value.
struct _QvrQiecShifted
    distribution
    offset
end
_qvr_qiec_shifted(distribution, offset) = _QvrQiecShifted(distribution, offset)
function _qvr_qiec_array(value)
    value = _qvr_qiec_value(value)
    if value isa AbstractArray
        return value
    end
    if value isa Tuple
        entries = [_qvr_qiec_array(item) for item in value]
        if !isempty(entries) && all(item -> item isa AbstractArray, entries)
            return stack(entries; dims=1)
        end
        return collect(promote(entries...))
    end
    return value
end
function _qvr_qiec_log_density(distribution, value)
    point = _qvr_qiec_array(value)
    if distribution isa _QvrQiecShifted
        return logpdf(distribution.distribution, point .+ distribution.offset)
    end
    return logpdf(distribution, point)
end
function _qvr_qiec_log_mass(distribution, retained=0, shape=())
    distribution = distribution isa _QvrQiecShifted ? distribution.distribution : distribution
    # Every mapped Distributions.jl constructor is a probability law.
    # Raw measures require an explicit normalizer protocol and family mapping.
    mass = hasproperty(distribution, :log_normalizer) ? distribution.log_normalizer() : 0.0
    0 <= retained <= length(shape) || error("log_mass retained axes must be within the batch rank")
    isempty(shape) && return mass
    values = mass isa Number ? fill(mass, shape) : reshape(mass, (ntuple(_ -> 1, length(shape)-ndims(mass))..., size(mass)...)) .+ zeros(shape)
    retained == length(shape) && return values
    retained == 0 && return sum(values)
    return dropdims(sum(values; dims=Tuple(retained+1:length(shape))); dims=Tuple(retained+1:length(shape)))
end
function _qvr_qiec_site_names()
    # Site labels a helper's draws and scores carry, counted so the n-th
    # occurrence of a label in one run of the model is "<label>@<n>", as
    # the reference machine replays them.
    occurrences = Dict{String,Int}()
    return label -> begin
        count = get(occurrences, label, 0)
        occurrences[label] = count + 1
        count == 0 ? label : label * "@" * string(count)
    end
end
function _qvr_qiec_native_operations(random_instance, sample_operation, score_instance, add_operation, draw, add)
    # The program's canonical instances handled by the model's own
    # closures: `draw` traces a distribution under a site name through
    # DynamicPPL, and `add` accumulates a scored weight. A shifted
    # family is drawn on the host's support and read back on QIEC's.
    name = _qvr_qiec_site_names()
    sample = request -> begin
        label, distribution = request["arguments"]
        if distribution isa _QvrQiecShifted
            return draw(name(label), distribution.distribution) - distribution.offset
        end
        return draw(name(label), distribution)
    end
    score = request -> begin
        (weight,) = request["arguments"]
        add(name("score"), _qvr_qiec_array(weight))
        return nothing
    end
    return Dict{Any,Any}(
        (random_instance, sample_operation) => sample,
        (score_instance, add_operation) => score,
    )
end
