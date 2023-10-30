Base.@kwdef struct Target
    xy = [5,5]
    radius = 1
end
Base.broadcastable(target::Target) = Ref(target)

Base.@kwdef mutable struct TrajectoryParams
	σθ = 0.5 # next heading angle std
end
Base.broadcastable(τθ::TrajectoryParams) = Ref(τθ)

Base.@kwdef struct InitialState
    ranges::Vector{Vector}
	target::Target
	D_xy = Product([Uniform(ranges[1]...), Uniform(ranges[2]...)])
	D_speed = TruncatedNormal(1, 1, 0, Inf) # speed
    D_heading = Uniform(0, 2π)
	buffer = 1
end
Base.broadcastable(initialstate::InitialState) = Ref(initialstate)

function Base.rand(ds::InitialState)
	xy = rand(ds.D_xy)
	# rejection sampling outside of target region
	while is_violation(xy, Target(ds.target.xy, ds.buffer * ds.target.radius))
		xy = rand(ds.D_xy)
	end
    xy_dot = rand(ds.D_speed)
    h₀ = rand(ds.D_heading)
	return [xy..., xy_dot, h₀]
end

function generate_trajectory(τθ::TrajectoryParams; s=[0,0,0,0], h=90, d=20)
	τ = Vector{Real}[]
	θ(h) = Normal(h, τθ.σθ) # next heading angle
    r = s[3]
	h = s[4]
	for t in 1:d
		push!(τ, s)
		h = rand(θ(h))
		s = [s[1] + r*sin(h), s[2] + r*cos(h), r, h]
	end
	return τ
end

function generate_trajectories(τθ::TrajectoryParams, initialstate::InitialState; m=500, d=20, seed=missing)
	if !ismissing(seed)
		Random.seed!(seed)
	end
	return [generate_trajectory(τθ; s=rand(initialstate), h=rand(-90:90), d) for _ in 1:m]
end

# function extract_features(τₜ::Vector{Real}, τₜ₊₁::Vector{Real}, target::Target)
function extract_features(τₜ::Vector{Real}, target::Target)
    xyₜ = τₜ[1:2]
    speed = τₜ[3]
    h = τₜ[4] # heading of the threat
	α = angle_to_target(xyₜ, target.xy)
    dvₜ = norm(xyₜ - target.xy) - target.radius
	dtₜ = norm(xyₜ - target.xy)
	xyₜ₊₁ = [xyₜ[1] + speed*sin(h), xyₜ[2] + speed*cos(h)]
    dtₜ₊₁ = norm(xyₜ₊₁ - target.xy)
    # dvₜ₊₁ = norm(xyₜ₊₁ - target.xy) - target.radius
    # rate = dvₜ - dvₜ₊₁ # same as dt because target.radius cancels
    rate = dtₜ - dtₜ₊₁ # closure rate to target
	return [α, dvₜ, rate]
	# return [dvₜ]
	# return [α, dvₜ]
	# return [α, dvₜ, dtₜ, rate]
	# return [h, dvₜ, dtₜ]
end

function extract_features(τ::Vector{<:Vector{Real}}, target::Target)
	# inputs:
	# - headings to target
	# - distances to violation (negative if inside violation)
	# - distances to target
    # - rates to target
	X = Vector{Real}[]
	for t in eachindex(τ)
		# push!(X, extract_features(τ[t], t == length(τ) ? τ[t] : τ[t+1], target))
		push!(X, extract_features(τ[t], target))
	end

	# outputs (i.e., ML target)
	# - is_violated (any point)
	Y = fill(Int(any(map(τₜ->is_violation(τₜ, target), τ))), length(τ))
	return X, Y
end

function extract_features(trajs::Vector{<:Vector{<:Vector{<:Real}}}, target::Target)
	𝐗 = []
	𝐘 = []
	for τ in trajs
		X, Y = extract_features(τ, target)
		push!(𝐗, X...)
		push!(𝐘, Y...)
	end
	return 𝐗, 𝐘
end

function get_dataset(τθ::TrajectoryParams, target::Target, initialstate::InitialState; m=500)
    trajs = generate_trajectories(τθ, initialstate; m)
    return get_dataset(trajs, target)
end

get_dataset(trajs::Vector, target::Target) = extract_features(trajs, target)

is_violation(τₜ::Vector{<:Real}, target::Target) = norm(τₜ[1:2] - target.xy) < target.radius
is_violation(τ::Vector{<:Vector{<:Real}}, target::Target) = any(is_violation.(τ, target))

function find_violations(trajs, target::Target; negate=false)
	violated_trajs = Vector{Vector{Real}}[]
	for τ in trajs
		for t in eachindex(τ)
			violated = is_violation(τ[t], target)
            if violated
				push!(violated_trajs, τ)
				break
			end
		end
	end
    if negate
        return setdiff(trajs, violated_trajs)
    else
    	return violated_trajs
    end
end

function is_predicted_violation(τₜ::Vector{<:Real}, target::Target, f::Chain; thresh=0.5)
    ŷ = lookup(f, τₜ, target)
    return ŷ > thresh
end

function is_predicted_violation(τ::Vector{<:Vector{<:Real}}, target::Target, f::Chain; thresh=0.5)
    Ŷ = lookup(f, τ, target)
    return any(Ŷ .> thresh)
end

function compute_classification_stats(trajs, target::Target, thresholds, f)
	stats = Dict()
	for thresh in thresholds
		num_tp = 0
		num_fp = 0
		num_tn = 0
		num_fn = 0
		for τ in trajs
            # truth over entire trajectory
			truth = is_violation(τ, target)
			for t in eachindex(τ)
                # prediction at each step of the trajectory
				pred = is_predicted_violation(τ[t], target, f; thresh)
				if truth && pred
					num_tp += 1
				elseif truth && !pred
					num_fn += 1
				elseif !truth && pred
					num_fp += 1
				elseif !truth && !pred
					num_tn += 1
				end
			end
		end
		stats[thresh] = (
			tp=num_tp,
			fp=num_fp,
			tn=num_tn,
			fn=num_fn,
		)
	end
	return stats
end

function accuracy(stats::Dict, thresh::Number)
	s = stats[thresh]
	return (s.tp + s.tn) / (s.tp + s.tn + s.fp + s.fn)
end

function precision(stats::Dict, thresh::Number)
	s = stats[thresh]
	return s.tp / (s.tp + s.fp)
end

function recall(stats::Dict, thresh::Number)
	s = stats[thresh]
	return s.tp / (s.tp + s.fn)
end
