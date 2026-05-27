#Predicts plastic types based on IR measurement.

using Pkg
Pkg.activate(".")

using Flux, MLDataDevices, JLD2, Statistics, Printf, DelimitedFiles, PythonCall


include("src/Featurization.jl")
include("src/ModelNext.jl")


# Wavenumber range of IR spectra (cm⁻¹)
const TRAIN_WAVENUMBERS = 400:2:3998

# Plastic types
const PLASTIC_TYPES = ["PET", "PE", "PVC", "PP", "PS", "Other"]

# Extract LoadedModel into a single object

struct LoadedModel
    model ::Chain
    norm_mu :: Vector{Float32}
    norm_sigma :: Vector{Float32}
    spec_len :: Int
    fg_names :: Vector{String}
    arch_version :: String
end

# Load Model

function load_model(path::String = "model.jld2")::Union{LoadedModel, Nothing}
    if !isfile(path)
        println("No model found at $path")
        return nothing
    end

    println("Loading model from $path ...")
    norm_mu     = JLD2.load(path, "norm_mu")
    norm_sigma  = JLD2.load(path, "norm_sigma")
    spec_len    = JLD2.load(path, "spec_len")
    fg_names    = JLD2.load(path, "fg_names")
    arch_version= JLD2.load(path, "arch_version")

    model = build_model(spec_len, length(fg_names))
    Flux.loadmodel!(model, JLD2.load(path, "model_state"))
    Flux.testmode!(model)

    println("Model loaded successfully (arch version: $arch_version)")
    return LoadedModel(model, norm_mu, norm_sigma, spec_len, fg_names, arch_version)
end

# Preprocess obtained IR spectra to fit model dimensions through interpolation

function interpolate_spectrum(wn_meas, spectrum)
    
    #sort measured wavenumbers to be in ascending order
    ascending_indices = sortperm(wn_meas)
    wn_meas_sorted = wn_meas[ascending_indices]
    spectrum_sorted = spectrum[ascending_indices]

    interpolated_spectrum = zeros(Float32, length(TRAIN_WAVENUMBERS))
    for (i, wn) in enumerate(TRAIN_WAVENUMBERS)
        # Find the indices of the two measured wavenumbers that bracket the target wavenumber
        idx_right = findfirst(x -> x >= wn, wn_meas_sorted)
        if idx_right === nothing
            # If the target wavenumber is above the max measured wavenumber, use the last measured value
            interpolated_spectrum[i] = spectrum_sorted[end]
        elseif idx_right == 1
            # If the target wavenumber is below the min measured wavenumber, use the first measured value
            interpolated_spectrum[i] = spectrum_sorted[1]
        else
            idx_left = idx_right - 1
            # Perform linear interpolation between the two bracketing points
            x0, y0 = wn_meas_sorted[idx_left], spectrum_sorted[idx_left]
            x1, y1 = wn_meas_sorted[idx_right], spectrum_sorted[idx_right]
            interpolated_spectrum[i] = y0 + (y1 - y0) * (wn - x0) / (x1 - x0)
        end
    end

    return interpolated_spectrum
end

# Apply normalization to the interpolated spectrum using the loaded model's parameters
function normalize_spectrum(spectrum::Vector{Float32}, norm_mu::Vector{Float32}, norm_sigma::Vector{Float32}) :: Vector{Float32}
    return (spectrum .- norm_mu) ./ norm_sigma
end

# Combine everything into a single preprocessing function
function preprocess_spectrum(wn_meas, spectrum, loaded_model::LoadedModel) :: Vector{Float32}
    interpolated_spectrum = interpolate_spectrum(wn_meas, spectrum)
    normalized_spectrum = normalize_spectrum(interpolated_spectrum, loaded_model.norm_mu, loaded_model.norm_sigma)
    return normalized_spectrum
end

# Predict functional groups from the preprocessed spectrum
function predict_functional_groups(loaded_model::LoadedModel, preprocessed_spectrum::Vector{Float32}, threshold :: Float32 = 0.5f0)
    reshaped_spectrum = MLDataDevices.gpu_device()(reshape(preprocessed_spectrum, loaded_model.spec_len, 1))
    outputsize = MLDataDevices.cpu_device()(loaded_model.model(reshaped_spectrum)) 
    probabilities = vec(sigmoid.(outputsize))
    binary_predictions = vec(probabilities .> threshold)
    return probabilities, binary_predictions
end

# Identify polimer based on predicted functional groups
function functional_groups_to_plastic(binary_predictions ::AbstractVector{Bool}, fg_names :: Vector{String}) :: String
    indexes = Dict(name => idx for (idx, name) in enumerate(fg_names))

    ar = binary_predictions[indexes["Aromatic Ring"]]
    cl = binary_predictions[indexes["Chlorine"]]
    el = binary_predictions[indexes["Ester Linkage"]]
    eb = binary_predictions[indexes["Ethylene Backbone"]]
    mb = binary_predictions[indexes["Methyl Branch"]]

    if cl
        return "PVC"
    elseif ar
        if el
            return "PET"
        else
            return "PS"
        end
    elseif mb && !ar && !el
        return "PP"
    elseif eb && !ar && !el && !mb
        return "PE"
    else
        return "Other"
    end
end

# Single Spectrum Prediction Pipeline
function predict_plastic_type(wn_meas, spectrum, loaded_model::LoadedModel, threshold :: Float32 = 0.5f0, verbose :: Bool = true)
    preprocessed_spectrum = preprocess_spectrum(wn_meas, spectrum, loaded_model)
    probabilities, binary_predictions = predict_functional_groups(loaded_model, preprocessed_spectrum, threshold)
    plastic_type = functional_groups_to_plastic(binary_predictions, loaded_model.fg_names)

    if verbose
        println("Predicted plastic type: $plastic_type")
        println("Functional group probabilities:")
        for (i, name) in enumerate(loaded_model.fg_names)
            println("  $name: $(round(100 * probabilities[i], digits=2))%")
        end
    end
    return plastic_type, probabilities, binary_predictions
end

#Evaluate a batch of measurements

function evaluate_batch(spectra :: Vector{Vector{Float32}}, wn_meas, loaded_model :: LoadedModel, threshold :: Float32 = 0.5f0)
    shared_wn = isa(wn_meas, Vector{<:Real})
    predictions = Vector{String}(undef, length(spectra))
    for (i, spectrum) in enumerate(spectra)
        if shared_wn
            predictions[i], _, _ = predict_plastic_type(wn_meas, spectrum, loaded_model, threshold, false)
        else
            predictions[i], _, _ = predict_plastic_type(wn_meas[i], spectrum, loaded_model, threshold, false)
        end
    end
    return predictions
end

# Evaluate batch performance against true labels
function evaluate_performance_batch(true_labels :: Vector{String}, predicted_labels :: Vector{String}, classes :: Vector{String} = PLASTIC_TYPES)
    N = length(classes)
    idx_classes = Dict(c => i for (i, c) in enumerate(classes))

    matrix_confusion = zeros(Int, N, N)
    for (true_label, pred_label) in zip(true_labels, predicted_labels)
        i = idx_classes[true_label]
        j = idx_classes[pred_label]
        matrix_confusion[i, j] += 1
    end

    tp = [matrix_confusion[i, i] for i in 1:N]
    row_sums = [sum(matrix_confusion[i, :]) for i in 1:N]
    col_sums = [sum(matrix_confusion[:, j]) for j in 1:N]
    
    precision = [tp[i] / (col_sums[i] + eps()) for i in 1:N]
    recall = [tp[i] / (row_sums[i] + eps()) for i in 1:N]
    f1_scores = [2 * precision[i] * recall[i] / (precision[i] + recall[i] + eps()) for i in 1:N]
    overall_accuracy = sum(tp) / sum(matrix_confusion)
    macro_f1 = mean(f1_scores)

    # ---------------------------------------------------------
    # PRINTING RESULTS
    # ---------------------------------------------------------
    
    # 1. Overall Summary
    println("="^45)
    println("             MODEL PERFORMANCE SUMMARY         ")
    println("="^45)
    @printf(" Overall Accuracy: %.2f%%\n", overall_accuracy * 100)
    @printf(" Macro F1-Score:   %.4f\n", macro_f1)
    println("-"^45)
    println()

    # 2. Per-Class Metrics Table (Plastic | Precision | F1)
    # Find longest class name to dynamically adjust padding
    max_len = max(maximum(length, classes), 7) 
    
    println("PER-CLASS METRICS:")
    # Header
    println(rpad("Plastic", max_len), " | Precision | F1-Score")
    println("-"^max_len, "-+-----------+---------")
    # Rows
    for i in 1:N
        @printf("%s |   %.4f  |  %.4f\n", rpad(classes[i], max_len), precision[i], f1_scores[i])
    end
    println()

    # 3. Confusion Matrix Table
    println("CONFUSION MATRIX (Rows: True / Cols: Predicted):")
    # Header row with abbreviated or padded class names
    col_width = max(maximum(length, classes), 5) + 1
    print(rpad("", max_len + 3)) # space for the side header row
    for c in classes
        print(lpad(c, col_width))
    end
    println()
    println("   ", "-"^(max_len + 1 + (col_width * N)))

    # Matrix rows
    for i in 1:N
        print("   ", rpad(classes[i], max_len), " |")
        for j in 1:N
            print(lpad(string(matrix_confusion[i, j]), col_width))
        end
        println()
    end
    println("="^45)

    # Return metrics in case you want to use them programmatically later
    return overall_accuracy, macro_f1, matrix_confusion
end


# Match lables from file names

const LABEL_MAP = Dict(
    "HDPE"    => "PE",
    "LDPE"    => "PE",
    "PET"     => "PET",
    "PP"      => "PP",
    "PS"      => "PS",
    "PVC"     => "PVC",
    "Nitrile" => "Other"
)

#Return parallel vectors of vectors of spectra, wavenumber, true label (each csv file has two columns - first one for wavenumbers, second one for spectra, no titles, separated by commas)
function prepare_data_lab_session(folder_session::String = "experimental_data/lab_session_1")
    csv_files = sort(filter(f -> endswith(f, ".csv"), readdir(folder_session, join=true)))

    spectra = Vector{Vector{Float32}}()
    wavenumbers = Vector{Vector{Float64}}()
    true_labels = Vector{String}()

    for filepath in csv_files
        data = readdlm(filepath, ',', Float64)
        wn_meas = data[:, 1]
        spectrum = Float32.(data[:, 2])

        push!(wavenumbers, wn_meas)
        push!(spectra, spectrum)

        filename = basename(filepath)
        # File names are of the form "HDPE3_trn.csv", "PET1_tst.csv", etc.
        label_key = match(r"^([A-Za-z]+)\d*_", filename).captures[1]
        true_label = get(LABEL_MAP, label_key, "Other")

        push!(true_labels, true_label)
    end

    println("Prepared data from $folder_session: $(length(spectra)) samples.")
    return spectra, wavenumbers, true_labels

end

#Run full evaluation pipeline on lab session data
function run_evaluation_session_pipeline_from_loaded_model(loaded_model :: LoadedModel, folder_session::String = "experimental_data/lab_session_1", threshold :: Float32 = 0.5f0)
    spectra, wavenumbers, true_labels = prepare_data_lab_session(folder_session)
    predicted_labels = evaluate_batch(spectra, wavenumbers, loaded_model, threshold)
    overall_acc, macro_f1, conf_matrix = evaluate_performance_batch(true_labels, predicted_labels)
    return overall_acc, macro_f1, conf_matrix
end

function run_evaluation_session_from_model_path(model_path::String, folder_session::String = "experimental_data/lab_session_1", threshold :: Float32 = 0.5f0)
    loaded_model = load_model(model_path)
    if loaded_model === nothing
        println("No model loaded. Cannot run evaluation pipeline.")
        return
    end
    return run_evaluation_session_pipeline_from_loaded_model(loaded_model, folder_session, threshold)
end

