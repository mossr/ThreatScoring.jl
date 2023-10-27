module ThreatScoring

using Reexport
@reexport using Plots; default(fontfamily="Computer Modern", framestyle=:box)
@reexport using Distributions
@reexport using Random
@reexport using LinearAlgebra
@reexport using Reel

export
    InitialState,
    TrajectoryParams,
    generate_trajectory,
    generate_trajectories,
    Target,
    heading,
    headings,
    angle_to_target,
    extract_features,
    is_violation,
    find_violations,
    plot_trajectories,
    create_gif

include("threat_scoring.jl")
include("utils.jl")
include("plots.jl")


end # module ThreatScoring
