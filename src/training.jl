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

function initialize_network(nn_params::NNParams)
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

    return Chain(
        DenseRegularizedLayer(input_size => ℓs)...,
        DenseRegularizedLayer(ℓs => ℓs)...,
        DenseRegularizedLayer(ℓs => ℓs)...,
        DenseRegularizedLayer(ℓs => ℓs)...,
        Dense(ℓs => 1),
        sigmoid
    )
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
        l = Flux.Losses.binarycrossentropy(ỹ, y)
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

function plot_training(e, training_epochs, losses_train, losses_valid)
    learning_curve = plot(xlims=(1, training_epochs), title="learning curve")
    plot!(1:e, losses_train, label="training", c=1)
    plot!(1:e, losses_valid, label="validation", c=2)
    ylims!(0, ylims()[2])
    return learning_curve
end

lookup(f::Chain, τ::Vector{<:Vector{<:Real}}, target::Target) = map(τₜ->lookup(f, τₜ, target), τ)

function lookup(f::Chain, τₜ::Vector{<:Real}, target::Target)
	x = Float32.(extract_features(τₜ, target))
	return f(x)[1] # 1-element vector
end
