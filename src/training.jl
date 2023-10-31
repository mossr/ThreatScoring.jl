Base.@kwdef struct NNParams
    input_size
    layer_size = 64
    activation = relu
    p_dropout = 0.0
    device = cpu
    batchsize = 32
    training_split = 0.8
    lr = 1e-3
    λ = 1e-3
end

mutable struct EnsembleNetwork
    networks
end

function initialize_network(nn_params::NNParams; is_ensemble=false, n=is_ensemble ? 3 : 1)
    input_size = nn_params.input_size
    activation = nn_params.activation
    ℓs = nn_params.layer_size

    p_dropout = nn_params.p_dropout
    use_dropout = p_dropout > 0

    function DenseRegularizedLayer(in_out::Pair)
        input, output = in_out
        if use_dropout
            return [Dense(input => output, activation), Dropout(p_dropout)]
        else
            return [Dense(input => output, activation)]
        end
    end

    create_network() = Chain(
        DenseRegularizedLayer(input_size => ℓs)...,
        DenseRegularizedLayer(ℓs => ℓs)...,
        DenseRegularizedLayer(ℓs => ℓs)...,
        DenseRegularizedLayer(ℓs => ℓs)...,
        Dense(ℓs => 1),
        # sigmoid done via logitbinarycrossentropy, handled in lookup
        # sigmoid, # if not using logitbinarycrossentropy
    )

    if is_ensemble
        return EnsembleNetwork([create_network() for _ in 1:n])
    else
        return create_network()
    end
end

function train(f::Chain, nn_params::NNParams, data; epochs=100, verbose=true)
    X = Float32.(hcat(data[1]...))
    Y = Float32.(reshape(data[2], 1, :))
    # Vector{Vector} -> Matrix
    n = length(Y)
    n_train = Int(n ÷ (1/nn_params.training_split))
    n_valid = n - n_train

    perm = randperm(length(Y))
    x_train_idx = perm[1:n_train]
    x_train, y_train = X[:, x_train_idx], Y[:, x_train_idx] # note MxN matrices

    x_valid_idx = perm[n_train+1:end]
    x_valid, y_valid = X[:, x_valid_idx], Y[:, x_valid_idx] # note MxN matrices
    @assert length(x_valid_idx) == n_valid

    train_data = Flux.Data.DataLoader((x_train, y_train), batchsize=nn_params.batchsize, shuffle=true)
    valid_data = Flux.Data.DataLoader((x_valid, y_valid), batchsize=nn_params.batchsize, shuffle=true)

    device = nn_params.device
    f = device(f)

    sqnorm(x) = sum(abs2, x)
    penalty() = nn_params.λ*sum(sqnorm, Flux.params(f))

    loss(x, y) = begin
        local ỹ = f(x)
        l = Flux.Losses.logitbinarycrossentropy(ỹ, y)
        # l = Flux.Losses.binarycrossentropy(ỹ, y)
        # l = Flux.Losses.tversky_loss(ỹ, y; beta=0.3)
        regularization = penalty()

        return l + regularization
    end

    opt = Adam(nn_params.lr)
    θ = Flux.params(f)

    # Batch calculate the loss, placing data on device
    function calc_loss(data)
        ℓ = 0
        total = 0
        for (x, y) in data
            ℓ += loss(device(x), device(y))
            total += size(y, 2)
        end
        return ℓ/total
    end

    logging_fn(epoch, loss_train, loss_valid; digits=5) = string("Epoch: ", epoch, "\t Loss Train: ", round(loss_train; digits), "\t Loss Val: ", round(loss_valid; digits), "\t")

    losses_train = []
    losses_valid = []

    for e in 1:epochs
        for (x, y) in train_data
            # Only put batches on device
            x = device(x)
            y = device(y)
            _, back = Flux.pullback(() -> loss(x, y), θ)
            Flux.update!(opt, θ, back(1.0f0))
        end
        loss_train = calc_loss(train_data)
        loss_valid = calc_loss(valid_data)
        push!(losses_train, loss_train)
        push!(losses_valid, loss_valid)

        if verbose # && e % nn_params.verbose_update_frequency == 0
            println(logging_fn(e, loss_train, loss_valid))
        end

        learning_curve = plot_training(e, epochs, losses_train, losses_valid)
        Plots.savefig(learning_curve, "training_curve.png")
        # display(learning_curve)
    end

    f = cpu(f)

    # Clean GPU memory explicitly
    if device == gpu
        x_train = y_train = x_valid = y_valid = nothing
        GC.gc()
        Flux.CUDA.reclaim()
    end

    return f
end

function train(en::EnsembleNetwork, nn_params::NNParams, data; kwargs...)
    for (i,f) in enumerate(en.networks)
        en.networks[i] = train(f, nn_params, data; kwargs...)
    end
    return en
end

lookup(f::Union{Chain,EnsembleNetwork}, τ::Vector{<:Vector{<:Real}}, target::Target; kwargs...) = map(τₜ->lookup(f, τₜ, target; kwargs...), τ)

function lookup(f::Chain, τₜ::Vector{<:Real}, target::Target; kwargs...)
	x = Float32.(extract_features(τₜ, target))
	return sigmoid(f(x)[1]) # 1-element vector
	# return f(x)[1] # 1-element vector, if not using logitbinarycrossentropy
end

function lookup(en::EnsembleNetwork, τₜ::Vector{<:Real}, target::Target; return_mean=false, return_raw=false)
    𝐲̃ = map(f->lookup(f, τₜ, target), en.networks)
    if return_raw
        return 𝐲̃
    else
        if return_mean
            return mean(𝐲̃)
        else
            return mean(𝐲̃), std(𝐲̃)
        end
    end
end

function get_input_size(τθ::TrajectoryParams, initialstate::InitialState, target::Target)
    trajs = generate_trajectories(τθ, initialstate; m=1)
    τ = trajs[1]
    τₜ = τ[1]
    return length(extract_features(τₜ, target))
end
