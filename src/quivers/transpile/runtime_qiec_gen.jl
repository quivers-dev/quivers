# QIEC distribution bridge for generated Gen modules. A Gen distribution
# takes its parameters at each draw, so a first-class distribution is the
# distribution object paired with its argument list; a log density
# unpacks the list into `Gen.logpdf`. A QIEC tensor is a nested tuple
# until it reaches a distribution, where it becomes an array with the
# tuple's outermost axis first. A family whose Gen support is shifted
# against the QIEC convention is wrapped with its offset, which is added
# to the scored value.
struct _QvrQiecDistribution
    distribution
    arguments
end
struct _QvrQiecShifted
    distribution
    offset
end
_qvr_qiec_distribution(distribution, arguments...) = _QvrQiecDistribution(distribution, Any[arguments...])
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
        return _qvr_qiec_log_density(distribution.distribution, point .+ distribution.offset)
    end
    return Gen.logpdf(distribution.distribution, point, distribution.arguments...)
end
function _qvr_qiec_log_mass(distribution, retained=0, shape=())
    distribution = distribution isa _QvrQiecShifted ? distribution.distribution : distribution
    # Mapped Gen distributions are normalized; raw measures must supply
    # an explicit normalizer rather than a density-at-a-point surrogate.
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
    # closures: `draw` traces a distribution at a site address through
    # Gen's trace, and `add` traces a scored weight as a factor. A
    # shifted family is drawn on the host's support and read back on
    # QIEC's.
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
