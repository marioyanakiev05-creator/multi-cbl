############################################################
# TRAINING LOOP
############################################################

const CHECKPOINT_PATH = "checkpoint.jld2"

function save_checkpoint(model, opt_state, epoch, best_val_loss)
    println("  Saving checkpoint at epoch $epoch (val loss = $(round(best_val_loss, digits=4)))")
    JLD2.save(CHECKPOINT_PATH,
        "model_state",    Flux.state(MLDataDevices.cpu_device()(model)),
        "opt_state",      Flux.state(MLDataDevices.cpu_device()(opt_state)),
        "epoch",          epoch,
        "best_val_loss",  best_val_loss,
        "arch_version",   ARCH_VERSION,
    )
end

function load_checkpoint!(model, opt_state)
    !isfile(CHECKPOINT_PATH) && return 0, Inf32

    saved_arch = JLD2.load(CHECKPOINT_PATH, "arch_version")
    if saved_arch != ARCH_VERSION
        println("Checkpoint arch '$saved_arch' ≠ current '$ARCH_VERSION' — ignoring checkpoint.")
        return 0, Inf32
    end

    println("Resuming from checkpoint at $CHECKPOINT_PATH ...")
    cpu_model_state = JLD2.load(CHECKPOINT_PATH, "model_state")
    cpu_opt_state   = JLD2.load(CHECKPOINT_PATH, "opt_state")
    epoch           = JLD2.load(CHECKPOINT_PATH, "epoch")
    best_val_loss   = JLD2.load(CHECKPOINT_PATH, "best_val_loss")

    Flux.loadmodel!(model, cpu_model_state)

    try
        Flux.loadmodel!(opt_state, cpu_opt_state)
        println("  Restored optimizer state.")
    catch e
        println("  Could not restore optimizer state (old checkpoint?), resetting Adam. Loss may spike briefly.")
    end

    println("  Resuming from epoch $(epoch+1), best val loss = $(round(best_val_loss, digits=4))")
    return epoch, Float32(best_val_loss)
end

# Weights per class — order matches sorted FG_NAMES:
# [Aromatic Ring, Chlorine, Ester Linkage, Ethylene Backbone, Methyl Branch]
# Chlorine (index 2) is rare so its errors are penalised more heavily

#added weights
const CLASS_WEIGHTS = reshape(Float32[1f0, 1f0, 1f0, 1f0, 1f0], N_FG, 1)

function weighted_focal_loss(logits, y; gamma=0f0)
    p = clamp.(sigmoid.(logits), 1f-6, 1f0 - 1f-6)
    w = MLDataDevices.gpu_device()(CLASS_WEIGHTS)
    pos = -y        .* (1f0 .- p).^gamma .* log.(p)
    neg = -(1f0.-y) .*         p .^gamma .* log.(1f0 .- p)
    return mean(w .* (pos .+ neg))
end

function train_model!(model, chunk_paths::Vector{String},
                      norm::Normalizer,
                      Xv::Matrix{Float32}, Yv::Matrix{Float32};
                      epochs=30,
                      lr_start=1f-3,
                      lr_min=1f-6,
                      patience=5,
                      resume=true)

    dev = MLDataDevices.gpu_device()

    opt_state = Flux.setup(Adam(lr_start), model)

    start_epoch, best_val_loss = resume ? load_checkpoint!(model, opt_state) : (0, Inf32)
    epochs_no_improve = 0

    val_loader = DataLoader((Xv, Yv), batchsize=256,
                            partial=false, parallel=true) |> dev

    loss_fn(m, x, y) = Flux.binary_focal_loss(
         clamp.(sigmoid(m(x)), 1f-6, 1f0 - 1f-6), y, gamma=0.6f0)
    #loss_fn(m, x, y) = weighted_focal_loss(m(x), y; gamma=2f0)

    @showprogress desc="Epochs" for e in (start_epoch+1):epochs

        lr = lr_min + 0.5f0 * (lr_start - lr_min) *
             (1f0 + cos(Float32(π) * (e - 1f0) / (epochs - 1f0)))
        Flux.adjust!(opt_state, lr)

        Flux.trainmode!(model)

        @showprogress desc="Chunks" offset=1 for path in shuffle(chunk_paths)
            X, Y, s = cached_load_chunk(path)
            tr_idx = findall(sv -> sv < 8, s)
            Xtr = apply_normalizer(norm, X[:, tr_idx])
            Ytr = Y[:, tr_idx]
            perm = randperm(size(Xtr, 2))
            train_loader = DataLoader((Xtr[:, perm], Ytr[:, perm]),
                                      batchsize=256, shuffle=false,
                                      partial=false, parallel=true) |> dev
            for (x, y) in train_loader
                loss, grads = Flux.withgradient(m -> loss_fn(m, x, y), model)
                Flux.update!(opt_state, model, grads[1])
            end
        end

        Flux.testmode!(model)
        val_loss = 0f0
        n_val    = 0
        for (Xb, Yb) in val_loader
            val_loss += loss_fn(model, Xb, Yb)
            n_val    += 1
        end
        val_loss /= n_val

        @printf("Epoch %2d | val loss = %.4f | lr = %.2e\n", e, val_loss, lr)

        if val_loss < best_val_loss
            best_val_loss      = val_loss
            epochs_no_improve  = 0
            save_checkpoint(model, opt_state, e, best_val_loss)
        else
            epochs_no_improve += 1
            println("  No improvement ($epochs_no_improve/$patience)")
        end

        if epochs_no_improve >= patience
            println("\nEarly stopping at epoch $e (no improvement for $patience epochs)")
            break
        end
    end

    if isfile(CHECKPOINT_PATH)
        println("\nRestoring best model weights from checkpoint...")
        best_state = JLD2.load(CHECKPOINT_PATH, "model_state")
        Flux.loadmodel!(model, best_state)
    end
end