const PLOT_DEFAULTS = (fontfamily="Computer Modern", framestyle=:box)

function plot_trajectories(trajs::Vector, target::Target, initialstate::InitialState; t=missing, f=missing, thresh=0, cmap=cgrad([:black, :red]))
	default(; PLOT_DEFAULTS...)

    ranges = initialstate.ranges
    plot()
	for τ in trajs
		if ismissing(t)
			t = length(τ)
		end
		τx = map(τₜ->τₜ[1], τ[1:t])
		τy = map(τₜ->τₜ[2], τ[1:t])
		if ismissing(f)
			c = :crimson
		else
			Ỹ = lookup(f, τ, target; return_mean=isa(f, EnsembleNetwork))
			g(ỹ) = thresh == 0 ? get(cmap, ỹ) : get(cmap, ỹ > thresh)
			c = map(ỹ->g(ỹ), Ỹ)[1:t]
		end
		plot!(τx, τy,
			c=c,
			mark=true,
			ms=2,
			msw=0,
			α=0.5,
			label=false)
		scatter!([τ[1][1]], [τ[1][2]], label=false, c=:white, ms=2, α=0.5)
	end	
	
	plot!(circle(target), seriestype=[:shape], lw=0.5,
		c=:yellow, linecolor=:black, label=false,
		fillalpha=0.2)
	plot!([target.xy[1], target.xy[1]-target.radius], [target.xy[2], target.xy[2]],
		c=:gray, label=false)
	scatter!([target.xy[1]], [target.xy[2]], c=:yellow, label=false)

	xlims!(ranges[1]...)
	ylims!(ranges[2]...)
	plot!(ratio=1)
end

function plot_training(e, training_epochs, losses_train, losses_valid)
	default(; PLOT_DEFAULTS...)

    learning_curve = plot(xlims=(1, training_epochs), title="learning curve")
    plot!(1:e, losses_train, label="training", c=1)
    plot!(1:e, losses_valid, label="validation", c=2)
    # ylims!(0, ylims()[2])
    return learning_curve
end

function plot_classification_metrics(stats::Dict, thresholds)
	accuracies = map(thresh->accuracy(stats, thresh), thresholds)
	precisions = map(thresh->ThreatScoring.precision(stats, thresh), thresholds)
	recalls = map(thresh->recall(stats, thresh), thresholds)

	args = (mark=false, ms=3, msw=0, lw=2)
	plot(thresholds, recalls; c=:red, label="recall", args...)
	plot!(thresholds, precisions; c=:blue, label="precision", args...)
	plot!(thresholds, accuracies; c=:green, label="accuracies", args...)
	plot!(size=(500,300), ylims=(0,1.05), yticks=0:0.1:1, xticks=0:0.1:1)
end

function plot_traj_and_prediction(τ, target, initialstate; f, cmap, t)
	plt_pred = plot_trajectories([τ], target, initialstate; f, cmap, t)
	𝒩 = TruncatedNormal(lookup(f, τ[t], target)..., 0, 1)
	plt_ensemble = plot(0:0.001:1, x->pdf(𝒩, x), xlabel="probability", label=false, c=:crimson, yticks=false)
	return plot(plt_pred, plt_ensemble, size=(600,300), margin=2Plots.mm)
end

function create_gif(trajs, target, initialstate; filename="traj.gif", callback=(trajs, target, initialstate; kwargs...)->plot_trajectories(trajs, target, initialstate; kwargs...), kwargs...)
	frames = Frames(MIME("image/png"), fps=2)
	for t in 1:length(trajs[1])
		frame = callback(trajs, target, initialstate; t, kwargs...)
		# plot_trajectories(trajs, target, initialstate; t, kwargs...)
		push!(frames, frame)
	end
	# [push!(frames, frame) for _ in 1:10] # duplicate last frame
	write(filename, frames)
    # LocalResource("./$filename")
    return frames
end
