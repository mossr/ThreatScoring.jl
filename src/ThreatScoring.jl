module ThreatScoring

using Reexport
@reexport using Plots
@reexport using Distributions
@reexport using Random
@reexport using LinearAlgebra
@reexport using Reel
@reexport using Flux
@reexport using Flux.NNlib
@reexport using Flux.MLUtils

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
    get_dataset,
    is_violation,
    find_violations,
    is_predicted_violation,
    compute_classification_stats,
    accuracy,
    precision,
    recall,
    NNParams,
    initialize_network,
    train,
    lookup,
    get_input_size,
    plot_training,
    plot_trajectories,
    plot_classification_metrics,
    create_gif

include("threat_scoring.jl")
include("utils.jl")
include("training.jl")
include("plots.jl")


end # module ThreatScoring
