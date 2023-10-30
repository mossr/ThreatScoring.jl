### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ eb77a7f8-412c-4fb5-a0a1-d777292730cb
begin
	using Pkg
	Pkg.develop(path="..")
end

# ╔═╡ b42402a7-d52d-4647-8657-9a97311f5ed2
begin
	using Revise
	using ThreatScoring
end

# ╔═╡ da8594de-742a-11ee-2c88-818d5c5cd958
using PlutoUI; TableOfContents()

# ╔═╡ 5684532b-2c77-4571-85f4-2d27ffbafee1
md"""
# Threat scoring using neural networks
"""

# ╔═╡ 88963fbd-066a-43bd-9656-d472180d465c
target = Target([5,4], 3)

# ╔═╡ c8d3b5d0-21f2-489c-b5d5-5f8230eacaaf
initialstate = InitialState(; ranges=[[-10,10], [-10,10]], target, buffer=2)

# ╔═╡ 9eed3d0d-6046-43ef-8852-12716e71e0ee
md"""
## Trajectories
"""

# ╔═╡ 2e354f0b-99c4-4fd0-9ccb-b2a0465c6d48
τθ = TrajectoryParams()

# ╔═╡ 25484c05-3fae-42f8-9504-151bafbc976f
trajs = generate_trajectories(τθ, initialstate; m=500)

# ╔═╡ 4de81db4-d196-495a-9273-69e865bfa263
md"""
# Features

$$\mathbf{x} = [\alpha, d, r]$$
where

$\alpha = \text{bearing to target}$

$d = \text{distance to violation}$

$r = \text{closure rate to target}$

These features are all relative, thus they should generalize to different targets.
"""

# ╔═╡ 99a39a6b-303f-4378-824f-cdc8bc07c403
extract_features(trajs[1][1], target)

# ╔═╡ 5e23ff6d-f322-4323-ad46-0abf06e2f462
md"""
# Neural network training
"""

# ╔═╡ 3d6c6322-d838-4c53-a289-c2edc2ca633d
input_size = get_input_size(τθ, initialstate, target)

# ╔═╡ 5e15b7c0-0c7a-4030-ab37-37cc015bb015
nn_params = NNParams(; input_size, lr=1e-4)

# ╔═╡ 1231bea9-9ca5-4b4a-bd1a-43152d66ff41
begin
	f = initialize_network(nn_params)
	data = get_dataset(τθ, target, initialstate; m=500) # trigger NN reinitialization
end

# ╔═╡ b0952f18-eb86-43bc-ad71-6c6965a07a8d
@bind run_training CheckBox(false)

# ╔═╡ abae61b8-3b6e-42ff-8533-34100f8e093f
f′ = run_training ? train(f, nn_params, data; epochs=200) : missing

# ╔═╡ 588df368-815f-4ce0-8818-07fbba60c9b4
md"""
## All trajectories
"""

# ╔═╡ da10d661-a4a2-4a37-96d5-53f22fe71f14
md"""
### Thresholding prediction (all trajectories)
"""

# ╔═╡ ee36059c-15b9-41aa-8f8c-b284f7d095fc
md"""
## Trajectories that violated the target
"""

# ╔═╡ 4c2b53af-25dd-438b-9f1f-5761d0dda8b9
violated_trajs = find_violations(trajs, target)

# ╔═╡ 148ac0b2-fc1c-4643-b54a-033372da3bed
md"""
### Thresholding prediction (only violations)
"""

# ╔═╡ a0e2fae1-e6ee-4002-b52c-08fd82f6afcd
@bind i Slider(eachindex(violated_trajs), show_value=true)

# ╔═╡ fe49d27d-ab4a-4c6a-8c3a-c325b7c674e7
LocalResource("./traj.gif")

# ╔═╡ 2f4b1f12-c72b-44cf-886a-4ec27b72532c
md"""
# Non-violations
"""

# ╔═╡ 7dca2cde-979d-4124-9f10-45014d69847a
non_violated_trajs = find_violations(trajs, target; negate=true)

# ╔═╡ f461b609-4894-4fc8-9da1-bef0c3fa42b1
md"""
### Thresholding prediction (only non-violations)
"""

# ╔═╡ 3d1d8a50-5f0c-478e-8d7f-4386a674b389
md"""
# Trade-off

**Precision**: Of all _positive predictions_ (demoninator), how many are _really positive_ (numerator)?

$$\text{precision} = \frac{\text{true positives}}{\text{true positives} + \text{false positives}}$$


**Recall**: Of all _real positive cases_ (denominator), how many are _predicted positive_ (numerator)?

$$\text{recall} = \frac{\text{true positives}}{\text{true positives} + \text{false negatives}}$$


**Accuracy**: Fraction of correct predictions.

$$\text{accuracy} = \frac{\text{true posititives} + \text{true negatives}}{\text{total}}$$

"""

# ╔═╡ 6fcab9ad-b3e4-49d5-b4f5-5150f4148408
thresholds = 0.95:-0.05:0.05 # threshold for NN output to mark as threat

# ╔═╡ e39af021-acb9-4880-8c8b-094cb82e5add
stats = compute_classification_stats(trajs, target, thresholds, f′)

# ╔═╡ 219676b7-bacf-4f02-8bae-a63b4f4fce33
plot_classification_metrics(stats, thresholds)

# ╔═╡ 4bc9a61e-ac48-4668-b59b-b94539193a91
stats

# ╔═╡ 722f9ea9-5d2c-4dcc-a90e-825ab01bf58b
md"""
# Step a single trajectory
"""

# ╔═╡ a47a8e51-5801-4d20-a34a-74c7bb4dab0c
@bind vi Slider(1:length(violated_trajs))

# ╔═╡ 6d9a19d0-592c-4dac-bf85-0d66d925f46c
@bind t Slider(1:length(violated_trajs[vi]), default=length(violated_trajs[vi]))

# ╔═╡ 02d9bab2-a3c3-465e-94ea-4818d9b751a5
plot_trajectories([violated_trajs[vi]], target, initialstate; t)

# ╔═╡ e8354489-a561-4d21-8a83-f040ab636a39
extract_features(violated_trajs[vi][t], target)

# ╔═╡ 5575a20b-aa9f-450e-b701-bd2bf7d376f3
cmap = cgrad(["#eeeeee", :red])

# ╔═╡ 547f04e4-3590-4082-93af-cb8c2c5ccc85
plt = plot_trajectories(trajs, target, initialstate; f=f′, cmap)

# ╔═╡ c2889e26-805f-4ec5-b87d-4aa3b45e3c31
begin
	plt_thresholds = []
	for thresh in [0.5:-0.1:0.1; 0.05]
		plt = plot_trajectories(trajs, target, initialstate; f=f′, cmap, thresh)
		title!("$thresh")
		push!(plt_thresholds, plt)
	end
	plot(plt_thresholds...)
end

# ╔═╡ 567b5027-1c84-4eb1-93f6-ca86fd215d0c
plt_vio = plot_trajectories(violated_trajs, target, initialstate; f=f′, cmap)

# ╔═╡ 00ea40a7-8a83-44f3-b81b-0e0b9876d931
plot(plt, plt_vio)

# ╔═╡ dacdfffd-74d8-4082-80f4-88cf942151af
begin
	plt_vio_thresholds = []
	for thresh in [0.5:-0.1:0.1; 0.05]
		plt = plot_trajectories(violated_trajs, target, initialstate; f=f′, cmap, thresh)
		title!("$thresh")
		push!(plt_vio_thresholds, plt)
	end
	plot(plt_vio_thresholds...)
end

# ╔═╡ 7b396a45-b0b6-4c25-b0b0-57231bc6e2e3
plot_trajectories([violated_trajs[i]], target, initialstate; f=f′, cmap)

# ╔═╡ 00a69227-4ee3-4419-9a60-50bad6a4a6f3
create_gif(violated_trajs, target, initialstate; f=f′, cmap)

# ╔═╡ 9fe66d59-b6ad-47e2-94de-0a0609c00fed
begin
	plt_non_vio_thresholds = []
	for thresh in [0.5:-0.1:0.1; 0.05]
		plt = plot_trajectories(non_violated_trajs, target, initialstate; f=f′, cmap, thresh)
		title!("$thresh")
		push!(plt_non_vio_thresholds, plt)
	end
	plot(plt_non_vio_thresholds...)
end

# ╔═╡ Cell order:
# ╟─5684532b-2c77-4571-85f4-2d27ffbafee1
# ╠═da8594de-742a-11ee-2c88-818d5c5cd958
# ╠═eb77a7f8-412c-4fb5-a0a1-d777292730cb
# ╠═b42402a7-d52d-4647-8657-9a97311f5ed2
# ╠═88963fbd-066a-43bd-9656-d472180d465c
# ╠═c8d3b5d0-21f2-489c-b5d5-5f8230eacaaf
# ╟─9eed3d0d-6046-43ef-8852-12716e71e0ee
# ╠═2e354f0b-99c4-4fd0-9ccb-b2a0465c6d48
# ╠═25484c05-3fae-42f8-9504-151bafbc976f
# ╟─4de81db4-d196-495a-9273-69e865bfa263
# ╠═99a39a6b-303f-4378-824f-cdc8bc07c403
# ╟─5e23ff6d-f322-4323-ad46-0abf06e2f462
# ╠═3d6c6322-d838-4c53-a289-c2edc2ca633d
# ╠═5e15b7c0-0c7a-4030-ab37-37cc015bb015
# ╠═1231bea9-9ca5-4b4a-bd1a-43152d66ff41
# ╠═b0952f18-eb86-43bc-ad71-6c6965a07a8d
# ╠═00ea40a7-8a83-44f3-b81b-0e0b9876d931
# ╠═abae61b8-3b6e-42ff-8533-34100f8e093f
# ╟─588df368-815f-4ce0-8818-07fbba60c9b4
# ╠═547f04e4-3590-4082-93af-cb8c2c5ccc85
# ╟─da10d661-a4a2-4a37-96d5-53f22fe71f14
# ╠═c2889e26-805f-4ec5-b87d-4aa3b45e3c31
# ╟─ee36059c-15b9-41aa-8f8c-b284f7d095fc
# ╠═4c2b53af-25dd-438b-9f1f-5761d0dda8b9
# ╠═567b5027-1c84-4eb1-93f6-ca86fd215d0c
# ╟─148ac0b2-fc1c-4643-b54a-033372da3bed
# ╠═dacdfffd-74d8-4082-80f4-88cf942151af
# ╠═a0e2fae1-e6ee-4002-b52c-08fd82f6afcd
# ╠═7b396a45-b0b6-4c25-b0b0-57231bc6e2e3
# ╠═00a69227-4ee3-4419-9a60-50bad6a4a6f3
# ╠═fe49d27d-ab4a-4c6a-8c3a-c325b7c674e7
# ╟─2f4b1f12-c72b-44cf-886a-4ec27b72532c
# ╠═7dca2cde-979d-4124-9f10-45014d69847a
# ╟─f461b609-4894-4fc8-9da1-bef0c3fa42b1
# ╠═9fe66d59-b6ad-47e2-94de-0a0609c00fed
# ╟─3d1d8a50-5f0c-478e-8d7f-4386a674b389
# ╠═6fcab9ad-b3e4-49d5-b4f5-5150f4148408
# ╠═e39af021-acb9-4880-8c8b-094cb82e5add
# ╠═219676b7-bacf-4f02-8bae-a63b4f4fce33
# ╠═4bc9a61e-ac48-4668-b59b-b94539193a91
# ╟─722f9ea9-5d2c-4dcc-a90e-825ab01bf58b
# ╠═a47a8e51-5801-4d20-a34a-74c7bb4dab0c
# ╠═6d9a19d0-592c-4dac-bf85-0d66d925f46c
# ╠═02d9bab2-a3c3-465e-94ea-4818d9b751a5
# ╠═e8354489-a561-4d21-8a83-f040ab636a39
# ╠═5575a20b-aa9f-450e-b701-bd2bf7d376f3
