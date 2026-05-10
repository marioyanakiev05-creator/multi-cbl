############################################################
# MODEL
############################################################

# --- Type-stable custom layers (no anonymous functions) ---

"""
    ReshapeLayer(spec_len)

Replaces `x -> reshape(x, spec_len, 1, :)`. A named struct avoids
the boxing that happens when a closure captures `spec_len` from an
enclosing function, which is critical for Zygote gradient specialization.
"""
struct ReshapeLayer
    spec_len::Int
end

(m::ReshapeLayer)(x) = reshape(x, m.spec_len, 1, :)

"""
    GlobalMeanPool

Replaces `x -> dropdims(mean(x; dims=1); dims=1)`. A named singleton
type lets Zygote fully specialize the gradient, avoiding the generic
(alloc-heavy) AD path that anonymous functions can trigger.
"""
struct GlobalMaxPool end
(m::GlobalMaxPool)(x) = dropdims(maximum(x; dims=1); dims=1)
# No @layer needed — these types have no trainable parameters or
# sub-layers. Flux's generic fallback handles them correctly.

# --- ResBlock ---

struct ResBlock
    conv1
    bn1
    conv2
    bn2
    skip
end

Flux.@layer ResBlock

function ResBlock(ch_in::Int, ch_out::Int; stride::Int=1)
    conv1 = Conv((7,), ch_in => ch_out; stride=stride, pad=3)
    bn1 = BatchNorm(ch_out)
    conv2 = Conv((7,), ch_out => ch_out; pad=3)
    bn2 = BatchNorm(ch_out)

    if stride == 1 && ch_in == ch_out
        skip = identity
    else
        skip = Chain(
            Conv((1,), ch_in => ch_out; stride=stride),
            BatchNorm(ch_out),
        )
    end

    return ResBlock(conv1, bn1, conv2, bn2, skip)
end

function (b::ResBlock)(x)
    h = relu.(b.bn1(b.conv1(x)))
    h = b.bn2(b.conv2(h))
    return relu.(h .+ b.skip(x))
end

function build_model(spec_len::Int, n_fg::Int)
    m = Chain(
        ReshapeLayer(spec_len),
        # Stride 4 instead of 2 halves the first intermediate (98→49 MB at batch 128)
        # Kernel 31 maintains enough receptive field for IR band detection
        Conv((31,), 1 => 32; stride=4, pad=15),
        BatchNorm(32),
        relu,
        ResBlock(32, 32; stride=2),
        ResBlock(32, 64; stride=2),
        ResBlock(64, 64; stride=2),
        ResBlock(64, 128; stride=2),
        ResBlock(128, 128; stride=2),

        #--- Change: replaced Flatten + Dense(128, 64) with GlobalMaxPool + Dense(1920, 256) ---"
        #Flux.flatten,          # add this
        #Dense(1920, 256, relu),  # was Dense(128, 64, relu)
         
        
        # NEW: GlobalMaxPool then small dense
        GlobalMaxPool(),          
        Dense(128, 256, relu),    


        # --- END OF CHANGE ---
        Dropout(0.3),
        Dense(256, n_fg),

    )

    dev = MLDataDevices.gpu_device()
    return dev(m)
end
