const PLOT_DEFAULTS = (fontfamily="Computer Modern", framestyle=:box)

function plot_trajectories(trajs::Vector, target::Target, initialstate::InitialState; t=missing, f=missing, thresh=0)
	default(; PLOT_DEFAULTS...)

    ranges = initialstate.ranges
    plot()
	for τ in trajs
		if ismissing(t)
			t = length(τ)
		end
		τx = first.(τ)[1:t]
		τy = last.(τ)[1:t]
		if ismissing(f)
			c = :crimson
		else
			cmap = cgrad([:black, :red])
			Ỹ = lookup(f, τ, target)
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

function create_gif(trajs, target, initialstate; f=missing)
	frames = Frames(MIME("image/png"), fps=2)
	for t in 1:length(trajs[1])
		frame = plot_trajectories(trajs, target, initialstate; t, f)
		push!(frames, frame)
	end
	# [push!(frames, frame) for _ in 1:10] # duplicate last frame
	write("traj.gif", frames)
    # LocalResource("./traj.gif")
    return frames
end

function plot_training(e, training_epochs, losses_train, losses_valid)
	default(; PLOT_DEFAULTS...)

    learning_curve = plot(xlims=(1, training_epochs), title="learning curve")
    plot!(1:e, losses_train, label="training", c=1)
    plot!(1:e, losses_valid, label="validation", c=2)
    ylims!(0, ylims()[2])
    return learning_curve
end
