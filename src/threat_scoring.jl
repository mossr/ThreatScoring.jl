Base.@kwdef struct Target
    xy = [5,5]
    radius = 1
end

Base.@kwdef mutable struct TrajectoryParams
	σθ = 0.5 # next heading angle std
	xbar = TruncatedNormal(1, 1, 0, Inf) # speed
end

Base.@kwdef struct InitialState
    ranges::Vector{Vector}
	target::Target
	distribution = Product([Uniform(ranges[1]...), Uniform(ranges[2]...)])
	buffer = 1
end

function Base.rand(ds::InitialState)
	# rejection sampling outside of target region
	s = rand(ds.distribution)
	while is_violation(s, Target(ds.target.xy, ds.buffer * ds.target.radius))
		s = rand(ds.distribution)
	end
	return s
end

function generate_trajectory(τθ::TrajectoryParams; s=[0,0], h=90, d=20)
	τ = Vector{Real}[]
	θ(h) = Normal(h, τθ.σθ) # next heading angle
	h = rand(θ(h))
	r = rand(τθ.xbar)
	for t in 1:d
		push!(τ, s)
		h = rand(θ(h))
		s = [s[1] + r*sin(h), s[2] + r*cos(h)]
	end
	return τ
end

function generate_trajectories(τθ::TrajectoryParams, initialstate::InitialState; m=500, d=20, seed=missing)
	if !ismissing(seed)
		Random.seed!(seed)
	end
	return [generate_trajectory(τθ; s=rand(initialstate), h=rand(-90:90), d) for _ in 1:m]
end

function extract_features(τₜ::Vector{Real}, target::Target)
	h = angle_to_target(τₜ, target.xy)
	dv = norm(τₜ - target.xy) - target.radius
	dt = norm(τₜ - target.xy)
	return [h, dv, dt]
end

function extract_features(τ::Vector{<:Vector{Real}}, target::Target)
	# inputs:
	# - headings to target
	# - distances to violation (negative if inside violation)
	# - distances to target
	X = Vector{Real}[]
	for t in eachindex(τ)
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
    return extract_features(trajs, target)
end

is_violation(τₜ, target::Target) = norm(τₜ - target.xy) < target.radius

function find_violations(trajs, target::Target)
	violated_trajs = Vector{Vector{Real}}[]
	for τ in trajs
		for t in eachindex(τ)
			if is_violation(τ[t], target)
				push!(violated_trajs, τ)
				break
			end
		end
	end
	return violated_trajs
end
