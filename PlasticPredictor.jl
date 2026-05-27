# Predicts plastic types based on IR measurement.
# Preprocessing: transmittance -> absorbance -> normalizer fitted on lab batch.

using Pkg
Pkg.activate(".")

using Flux, MLDataDevices, JLD2, Statistics, Printf, DelimitedFiles, PythonCall

include("src/Featurization.jl")
include("src/ModelNext.jl")

# Wavenumber range of IR spectra (cm⁻¹)
const TRAIN_WAVENUMBERS = 400:2:3998

# Plastic types
const PLASTIC_TYPES = ["PET", "PE", "PVC", "PP", "PS", "Other"]

############################################################
# STRUCT
############################################################

struct LoadedModel
    model        ::Chain
    norm_mu      ::Vector{Float32}
    norm_sigma   ::Vector{Float32}
    spec_len     ::Int
    fg_names     ::Vector{String}
    arch_version ::String
end

############################################################
# MODEL LOADING
############################################################

function load_model(path::String = "model.jld2")::Union{LoadedModel, Nothing}
    if !isfile(path)
        println("No model found at $path")
        return nothing
    end

    println("Loading model from $path ...")
    norm_mu      = JLD2.load(path, "norm_mu")
    norm_sigma   = JLD2.load(path, "norm_sigma")
    spec_len     = JLD2.load(path, "spec_len")
    fg_names     = JLD2.load(path, "fg_names")
    arch_version = JLD2.load(path, "arch_version")

    model = build_model(spec_len, length(fg_names))
    Flux.loadmodel!(model, JLD2.load(path, "model_state"))
    Flux.testmode!(model)

    println("Model loaded successfully (arch version: $arch_version)")
    return LoadedModel(model, norm_mu, norm_sigma, spec_len, fg_names, arch_version)
end

############################################################
# PREPROCESSING
############################################################

# Interpolate a measured spectrum onto the training wavenumber grid.
function interpolate_spectrum(wn_meas, spectrum)
    ascending_indices = sortperm(wn_meas)
    wn_meas_sorted    = wn_meas[ascending_indices]
    spectrum_sorted   = spectrum[ascending_indices]

    interpolated = zeros(Float32, length(TRAIN_WAVENUMBERS))
    for (i, wn) in enumerate(TRAIN_WAVENUMBERS)
        idx_right = findfirst(x -> x >= wn, wn_meas_sorted)
        if idx_right === nothing
            interpolated[i] = spectrum_sorted[end]
        elseif idx_right == 1
            interpolated[i] = spectrum_sorted[1]
        else
            idx_left = idx_right - 1
            x0, y0  = wn_meas_sorted[idx_left],  spectrum_sorted[idx_left]
            x1, y1  = wn_meas_sorted[idx_right], spectrum_sorted[idx_right]
            interpolated[i] = y0 + (y1 - y0) * (wn - x0) / (x1 - x0)
        end
    end
    return interpolated
end

# Convert %Transmittance to Absorbance.
# Values above 100 are clamped to 100. Computation in Float64 for precision.
function transmittance_to_absorbance(spectrum)::Vector{Float32}
    t = min.(Float64.(spectrum), 100.0)
    t = max.(t, 1e-9)
    return Float32.(-log10.(t ./ 100.0))
end

# Apply z-score normalisation.
function normalize_spectrum(spectrum  ::Vector{Float32},
                             norm_mu  ::Vector{Float32},
                             norm_sigma::Vector{Float32})::Vector{Float32}
    return (spectrum .- norm_mu) ./ norm_sigma
end

# Fit a normalizer from a batch of absorbance spectra.
# Returns (mu, sigma) as Float32 vectors.
# Must be called on a batch — a single spectrum has no meaningful statistics.
function get_lab_normalizer(spectra_abs::Vector{Vector{Float32}})
    mat   = hcat(spectra_abs...)
    mu    = Float32.(vec(mean(mat, dims=2)))
    sigma = Float32.(vec(std(mat,  dims=2))) .+ 1f-6
    return mu, sigma
end

############################################################
# FUNCTIONAL GROUP PREDICTION
############################################################

function predict_functional_groups(loaded_model::LoadedModel,
                                    spectrum    ::Vector{Float32},
                                    threshold   ::Float32 = 0.5f0)
    x      = MLDataDevices.gpu_device()(reshape(spectrum, loaded_model.spec_len, 1))
    output = MLDataDevices.cpu_device()(loaded_model.model(x))
    probs  = vec(sigmoid.(output))
    binary = vec(probs .> threshold)
    return probs, binary
end

############################################################
# POLYMER IDENTIFICATION
############################################################

function functional_groups_to_plastic(binary  ::AbstractVector{Bool},
                                       fg_names::Vector{String})::String
    idx = Dict(name => i for (i, name) in enumerate(fg_names))

    ar = binary[idx["Aromatic Ring"]]
    cl = binary[idx["Chlorine"]]
    el = binary[idx["Ester Linkage"]]
    eb = binary[idx["Ethylene Backbone"]]
    mb = binary[idx["Methyl Branch"]]

    if cl
        return "PVC"
    elseif ar && el
        return "PET"
    elseif ar && !el
        return "PS"
    elseif mb && !ar && !el
        return "PP"
    elseif eb && !ar && !el && !mb
        return "PE"
    else
        return "Other"
    end
end

############################################################
# SINGLE MEASUREMENT PREDICTION
#
# Requires lab_mu and lab_sigma computed from a batch via
# get_lab_normalizer. Normalising a single spectrum in
# isolation is not meaningful — obtain these statistics from
# prepare_data_lab_session first.
############################################################

function predict_plastic_type(wn_meas    ,
                               spectrum  ,
                               loaded_model::LoadedModel,
                               lab_mu    ::Vector{Float32},
                               lab_sigma ::Vector{Float32};
                               threshold ::Float32 = 0.5f0,
                               verbose   ::Bool    = true)

    interp     = interpolate_spectrum(wn_meas, spectrum)
    absorbance = transmittance_to_absorbance(interp)
    normalised = normalize_spectrum(absorbance, lab_mu, lab_sigma)

    probs, binary = predict_functional_groups(loaded_model, normalised, threshold)
    plastic_type  = functional_groups_to_plastic(binary, loaded_model.fg_names)

    if verbose
        println("Predicted plastic type: $plastic_type")
        println("Functional group probabilities:")
        for (i, name) in enumerate(loaded_model.fg_names)
            @printf("  %-22s %.1f%%\n", name, 100 * probs[i])
        end
    end
    return plastic_type, probs, binary
end

############################################################
# BATCH EVALUATION
# Expects spectra already fully preprocessed by
# prepare_data_lab_session.
############################################################

function evaluate_batch(spectra     ::Vector{Vector{Float32}},
                         loaded_model::LoadedModel,
                         threshold   ::Float32 = 0.5f0)::Vector{String}
    predictions = Vector{String}(undef, length(spectra))
    for (i, spectrum) in enumerate(spectra)
        _, binary      = predict_functional_groups(loaded_model, spectrum, threshold)
        predictions[i] = functional_groups_to_plastic(binary, loaded_model.fg_names)
    end
    return predictions
end

############################################################
# PERFORMANCE EVALUATION
############################################################

function evaluate_performance_batch(true_labels     ::Vector{String},
                                     predicted_labels::Vector{String},
                                     classes         ::Vector{String} = PLASTIC_TYPES)
    N           = length(classes)
    idx_classes = Dict(c => i for (i, c) in enumerate(classes))

    matrix_confusion = zeros(Int, N, N)
    for (t, p) in zip(true_labels, predicted_labels)
        matrix_confusion[idx_classes[t], idx_classes[p]] += 1
    end

    tp       = [matrix_confusion[i, i]     for i in 1:N]
    row_sums = [sum(matrix_confusion[i, :]) for i in 1:N]
    col_sums = [sum(matrix_confusion[:, j]) for j in 1:N]

    precision = [tp[i] / (col_sums[i] + eps()) for i in 1:N]
    recall    = [tp[i] / (row_sums[i] + eps()) for i in 1:N]
    f1_scores = [2 * precision[i] * recall[i] / (precision[i] + recall[i] + eps()) for i in 1:N]
    overall_accuracy = sum(tp) / sum(matrix_confusion)
    macro_f1         = mean(f1_scores)

    max_len   = max(maximum(length, classes), 7)
    col_width = max(maximum(length, classes), 5) + 1

    println("="^45)
    println("          MODEL PERFORMANCE SUMMARY")
    println("="^45)
    @printf(" Overall Accuracy: %.2f%%\n", overall_accuracy * 100)
    @printf(" Macro F1-Score:   %.4f\n",   macro_f1)
    println("-"^45)
    println()

    println("PER-CLASS METRICS:")
    println(rpad("Plastic", max_len), " | Precision |  Recall  | F1-Score")
    println("-"^max_len, "-+-----------+----------+---------")
    for i in 1:N
        @printf("%s |   %.4f  |  %.4f  |  %.4f\n",
                rpad(classes[i], max_len), precision[i], recall[i], f1_scores[i])
    end
    println()

    println("CONFUSION MATRIX (Rows: True / Cols: Predicted):")
    print(rpad("", max_len + 3))
    for c in classes; print(lpad(c, col_width)); end
    println()
    println("   ", "-"^(max_len + 1 + col_width * N))
    for i in 1:N
        print("   ", rpad(classes[i], max_len), " |")
        for j in 1:N
            print(lpad(string(matrix_confusion[i, j]), col_width))
        end
        println()
    end
    println("="^45)

    return overall_accuracy, macro_f1, matrix_confusion
end

############################################################
# LAB SESSION DATA LOADING
############################################################

const LABEL_MAP = Dict(
    "HDPE"    => "PE",
    "LDPE"    => "PE",
    "PET"     => "PET",
    "PP"      => "PP",
    "PS"      => "PS",
    "PVC"     => "PVC",
    "Nitrile" => "Other"
)

# Load all CSV files from a lab session folder.
# Pipeline per spectrum:
#   1. Interpolate onto training grid
#   2. Convert transmittance -> absorbance
# Then fit a normalizer across the whole batch and apply it.
#
# Returns preprocessed spectra, true labels, and the fitted
# (lab_mu, lab_sigma) so they can be reused for single predictions.
function prepare_data_lab_session(folder_session::String = "experimental_data/lab_session_1")
    csv_files = sort(filter(f -> endswith(f, ".csv"), readdir(folder_session, join=true)))

    spectra_abs = Vector{Vector{Float32}}()
    true_labels = Vector{String}()

    for filepath in csv_files
        data     = readdlm(filepath, ',', Float64)
        wn_meas  = data[:, 1]
        spectrum = Float32.(data[:, 2])

        interp     = interpolate_spectrum(wn_meas, spectrum)
        absorbance = transmittance_to_absorbance(interp)
        push!(spectra_abs, absorbance)

        filename  = basename(filepath)
        label_key = match(r"^([A-Za-z]+)\d*_", filename).captures[1]
        push!(true_labels, get(LABEL_MAP, label_key, "Other"))
    end

    # Fit normalizer on the full lab batch, then apply to every spectrum
    lab_mu, lab_sigma = get_lab_normalizer(spectra_abs)
    spectra_norm = [normalize_spectrum(s, lab_mu, lab_sigma) for s in spectra_abs]

    println("Prepared $(length(spectra_norm)) spectra from: $folder_session")
    return spectra_norm, true_labels, lab_mu, lab_sigma
end

############################################################
# PIPELINE ENTRY POINTS
############################################################

function run_evaluation_session_pipeline_from_loaded_model(
        loaded_model  ::LoadedModel,
        folder_session::String  = "experimental_data/lab_session_1",
        threshold     ::Float32 = 0.5f0)

    spectra, true_labels, _, _ = prepare_data_lab_session(folder_session)
    predicted_labels = evaluate_batch(spectra, loaded_model, threshold)
    overall_acc, macro_f1, conf_matrix = evaluate_performance_batch(true_labels, predicted_labels)
    return overall_acc, macro_f1, conf_matrix
end

function run_evaluation_session_from_model_path(
        model_path    ::String,
        folder_session::String  = "experimental_data/lab_session_1",
        threshold     ::Float32 = 0.5f0)

    loaded_model = load_model(model_path)
    loaded_model === nothing && return
    return run_evaluation_session_pipeline_from_loaded_model(loaded_model, folder_session, threshold)
end
