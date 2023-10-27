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
using PlutoUI

# ╔═╡ 5684532b-2c77-4571-85f4-2d27ffbafee1
md"""
# Threat scoring: Trajectory generation
"""

# ╔═╡ 88963fbd-066a-43bd-9656-d472180d465c
target = Target([5,4], 3)

# ╔═╡ c8d3b5d0-21f2-489c-b5d5-5f8230eacaaf
initialstate = InitialState(; ranges=[[-10,10], [-10,10]], target, buffer=2)

# ╔═╡ 588df368-815f-4ce0-8818-07fbba60c9b4
md"""
## All trajectories
"""

# ╔═╡ 2e354f0b-99c4-4fd0-9ccb-b2a0465c6d48
τθ = TrajectoryParams()

# ╔═╡ 25484c05-3fae-42f8-9504-151bafbc976f
trajs = generate_trajectories(τθ, initialstate; m=500)

# ╔═╡ 7b396a45-b0b6-4c25-b0b0-57231bc6e2e3
plot_trajectories(trajs, target, initialstate)

# ╔═╡ ee36059c-15b9-41aa-8f8c-b284f7d095fc
md"""
## Trajectories that violated the target
"""

# ╔═╡ 4c2b53af-25dd-438b-9f1f-5761d0dda8b9
violated_trajs = find_violations(trajs, target)

# ╔═╡ 567b5027-1c84-4eb1-93f6-ca86fd215d0c
plot_trajectories(violated_trajs, target, initialstate)

# ╔═╡ 00a69227-4ee3-4419-9a60-50bad6a4a6f3
create_gif(violated_trajs, target, initialstate)

# ╔═╡ fe49d27d-ab4a-4c6a-8c3a-c325b7c674e7
LocalResource("./traj.gif")

# ╔═╡ 722f9ea9-5d2c-4dcc-a90e-825ab01bf58b
md"""
# Pluto UI
"""

# ╔═╡ 6d9a19d0-592c-4dac-bf85-0d66d925f46c
@bind t Slider(1:length(violated_trajs[1]))

# ╔═╡ 02d9bab2-a3c3-465e-94ea-4818d9b751a5
plot_trajectories([violated_trajs[1]], target, initialstate; t)

# ╔═╡ 4de81db4-d196-495a-9273-69e865bfa263
md"""
# Features
"""

# ╔═╡ 86ef9af3-e831-4f11-84e3-bab6edc5396d
τ = violated_trajs[1]

# ╔═╡ e8354489-a561-4d21-8a83-f040ab636a39
extract_features(τ[t], target)

# ╔═╡ 683a221f-feff-4e4e-b01d-eccb349a8d6a
𝐗, 𝐘 = extract_features(trajs, target)

# ╔═╡ Cell order:
# ╟─5684532b-2c77-4571-85f4-2d27ffbafee1
# ╠═da8594de-742a-11ee-2c88-818d5c5cd958
# ╠═eb77a7f8-412c-4fb5-a0a1-d777292730cb
# ╠═b42402a7-d52d-4647-8657-9a97311f5ed2
# ╠═88963fbd-066a-43bd-9656-d472180d465c
# ╠═c8d3b5d0-21f2-489c-b5d5-5f8230eacaaf
# ╟─588df368-815f-4ce0-8818-07fbba60c9b4
# ╠═2e354f0b-99c4-4fd0-9ccb-b2a0465c6d48
# ╠═25484c05-3fae-42f8-9504-151bafbc976f
# ╠═7b396a45-b0b6-4c25-b0b0-57231bc6e2e3
# ╟─ee36059c-15b9-41aa-8f8c-b284f7d095fc
# ╠═4c2b53af-25dd-438b-9f1f-5761d0dda8b9
# ╠═567b5027-1c84-4eb1-93f6-ca86fd215d0c
# ╠═00a69227-4ee3-4419-9a60-50bad6a4a6f3
# ╠═fe49d27d-ab4a-4c6a-8c3a-c325b7c674e7
# ╟─722f9ea9-5d2c-4dcc-a90e-825ab01bf58b
# ╠═6d9a19d0-592c-4dac-bf85-0d66d925f46c
# ╠═02d9bab2-a3c3-465e-94ea-4818d9b751a5
# ╠═e8354489-a561-4d21-8a83-f040ab636a39
# ╟─4de81db4-d196-495a-9273-69e865bfa263
# ╠═86ef9af3-e831-4f11-84e3-bab6edc5396d
# ╠═683a221f-feff-4e4e-b01d-eccb349a8d6a
