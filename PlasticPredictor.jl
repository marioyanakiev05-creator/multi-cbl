using Pkg
Pkg.activate(".")

using Flux, MLDataDevices, JLD2, Statistics, Printf, DelimitedFiles, PythonCall
using DuckDB, DBInterface, DataFrames
using Interpolations

include("src/Featurization.jl")
include("src/ModelNext.jl")

# Plastic types
const PLASTIC_TYPES = ["PET", "PE", "PVC", "PP", "PS", "Other"]

# Create a LoadedModel object for easier handling

struct LoadedModel
    model             ::Chain
    norm_mu           ::Vector{Float32}
    norm_sigma        ::Vector{Float32}
    spec_len          ::Int
    fg_names          ::Vector{String}
    arch_version      ::String
    train_wavenumbers ::Vector{Float64}
end

#Extract the wavenumber axis from a parquet chunk.

function load_train_wavenumbers(parquet_path::String)::Vector{Float64}
    con    = DBInterface.connect(DuckDB.DB, ":memory:")
    schema = DBInterface.execute(con,
        "DESCRIBE SELECT * FROM read_parquet('$parquet_path') LIMIT 1") |> DataFrame

    freq_col = nothing
    for col in schema.column_name
        lc = lowercase(col)
        if occursin("cm", lc)
            freq_col = col
            break
        end
    end

    
    df = DBInterface.execute(con, "SELECT \"$freq_col\" FROM read_parquet('$parquet_path') LIMIT 1") |> DataFrame
    wn = sort(Float64.(collect(df[1, 1])))
    DBInterface.close!(con)
    @printf("Training grid: %.2f – %.2f cm^-1  (%d pts)\n", wn[1], wn[end], length(wn))
    return wn
    
end


# Load the model and associated metadata from a JLD2 file.

function load_model(path        ::String = "model.jld2",
                    parquet_path::String = "src/parquet-files/data/IR_data_chunk001_of_009.parquet"
                    )::Union{LoadedModel, Nothing}
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

    train_wn = load_train_wavenumbers(parquet_path)

    println("Model loaded successfully (arch version: $arch_version, spec_len: $spec_len)")
    return LoadedModel(model, norm_mu, norm_sigma, spec_len, fg_names, arch_version, train_wn)
end



# Interpolate a measured spectrum onto the training wavenumber grid

function interpolate_spectrum(wn_meas  ::Vector{Float64},
                               spectrum ,
                               train_wn ::Vector{Float64})::Vector{Float32}
    order     = sortperm(wn_meas)
    wn_sorted = wn_meas[order]
    sp_sorted = Float64.(spectrum[order])

    itp = linear_interpolation(wn_sorted, sp_sorted, extrapolation_bc=Flat())
    return Float32.(itp.(train_wn))
end

# Convert Transmittance to Absorbance.
function transmittance_to_absorbance(spectrum)::Vector{Float32}
    t = min.(Float64.(spectrum), 100.0)
    #t = max.(t, 1e-9)
    #return Float32.(-log10.(t ./ 100.0))
    return Float32.(t./ 100.0)
end

#Normalize a single spectrum so its peak reaches 1 (this is to match the training data)
function peak_normalize(spectrum::Vector{Float32})::Vector{Float32}
    max_val = maximum(spectrum)
    if max_val <= 0
        return spectrum
    else
        return spectrum ./ max_val
    end
end

# Apply z-score normalisation.
function normalize_spectrum(spectrum  ::Vector{Float32},
                             norm_mu  ::Vector{Float32},
                             norm_sigma::Vector{Float32})::Vector{Float32}
    return (spectrum .- norm_mu) ./ norm_sigma
end

# Fit a normalizer from a batch of absorbance spectra.
# Must be called on a batch — fitting on a single spectrum is meaningless.
function get_lab_normalizer(spectra_abs::Vector{Vector{Float32}})
    mat   = hcat(spectra_abs...)
    mu    = Float32.(vec(mean(mat, dims=2)))
    sigma = Float32.(vec(std(mat,  dims=2))) .+ 1f-6
    return mu, sigma
end



# Predict functional groups and their probabilities from a single spectrum. 

function predict_functional_groups(loaded_model::LoadedModel,
                                    spectrum    ::Vector{Float32},
                                    threshold   ::Float32 = 0.5f0)
    x      = MLDataDevices.gpu_device()(reshape(spectrum, loaded_model.spec_len, 1))
    output = MLDataDevices.cpu_device()(loaded_model.model(x))
    probs  = vec(sigmoid.(output))
    binary = vec(probs .> threshold)
    return probs, binary
end

# Map functional group presence/absence to plastic type.

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

# Predict the plastic type from a single measured spectrum, given a loaded model and lab normalizer.

function predict_plastic_type(wn_meas     ::Vector{Float64},
                               spectrum   ,
                               loaded_model::LoadedModel,
                               lab_mu     ::Vector{Float32},
                               lab_sigma  ::Vector{Float32};
                               threshold  ::Float32 = 0.5f0,
                               verbose    ::Bool    = true)

    interp     = interpolate_spectrum(wn_meas, spectrum, loaded_model.train_wavenumbers)
    absorbance = transmittance_to_absorbance(interp)
    peak_normalised = peak_normalize(absorbance)
    normalised = normalize_spectrum(peak_normalised, lab_mu, lab_sigma)

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

# Predict plastic types from a batch of spectra, given a loaded model.

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

# Load data from Lab session

const LABEL_MAP = Dict(
    "HDPE"    => "PE",
    "LDPE"    => "PE",
    "PET"     => "PET",
    "PP"      => "PP",
    "PS"      => "PS",
    "PVC"     => "PVC",
    "Nitrile" => "Other"
)

# Load and preprocess spectra from a lab session folder. Returns preprocessed spectra, true labels, and normalizer parameters.

function prepare_data_lab_session(loaded_model  ::LoadedModel,
                                   folder_session::String = "experimental_data/lab_session_1")
    csv_files = sort(filter(f -> endswith(f, ".csv"), readdir(folder_session, join=true)))

    spectra_abs = Vector{Vector{Float32}}()
    true_labels = Vector{String}()

    for filepath in csv_files
        data     = readdlm(filepath, ',', Float64)
        wn_meas  = data[:, 1]
        spectrum = Float32.(data[:, 2])

        interp     = interpolate_spectrum(wn_meas, spectrum, loaded_model.train_wavenumbers)
        absorbance = transmittance_to_absorbance(interp)
        peak_normalised = peak_normalize(absorbance)
        push!(spectra_abs, peak_normalised)

        filename  = basename(filepath)
        label_key = match(r"^([A-Za-z]+)\d*_", filename).captures[1]
        push!(true_labels, get(LABEL_MAP, label_key, "Other"))
    end

    lab_mu, lab_sigma = get_lab_normalizer(spectra_abs)
    spectra_norm = [normalize_spectrum(s, lab_mu, lab_sigma) for s in spectra_abs]

    println("Prepared $(length(spectra_norm)) spectra from: $folder_session")
    return spectra_norm, true_labels, lab_mu, lab_sigma
end

# Run the full evaluation pipeline on a lab session dataset, given a loaded model.

function run_evaluation_session_pipeline_from_loaded_model(
        loaded_model  ::LoadedModel,
        folder_session::String  = "experimental_data/lab_session_1",
        threshold     ::Float32 = 0.5f0)

    spectra, true_labels, _, _ = prepare_data_lab_session(loaded_model, folder_session)
    predicted_labels = evaluate_batch(spectra, loaded_model, threshold)
    overall_acc, macro_f1, conf_matrix = evaluate_performance_batch(true_labels, predicted_labels)
    return overall_acc, macro_f1, conf_matrix
end

function run_evaluation_session_from_model_path(
        model_path    ::String,
        parquet_path  ::String  = "src/parquet-files/data/IR_data_chunk001_of_009.parquet",
        folder_session::String  = "experimental_data/lab_session_1",
        threshold     ::Float32 = 0.5f0)

    loaded_model = load_model(model_path, parquet_path)
    loaded_model === nothing && return
    return run_evaluation_session_pipeline_from_loaded_model(loaded_model, folder_session, threshold)
end

############################################################
# ZENODO PLASTICS DATASET
############################################################

const ZENODO_LABEL_MAP = Dict(
    "LDPE_c4" => "PE",
    "HDPE_c4" => "PE",
    "PET_c4"  => "PET",
    "PP_c4"   => "PP",
    "PS_c4"   => "PS",
    "PVC_c4"  => "PVC"
)

const ZENODO_HEADER_LINES = 15

# Load a single zenodo CSV: skip the 15 metadata lines, parse wavenumber + transmittance.
function load_zenodo_spectrum(filepath::String)
    wn   = Float64[]
    spec = Float32[]
    open(filepath, "r") do f
        for _ in 1:ZENODO_HEADER_LINES
            readline(f)
        end
        for line in eachline(f)
            parts = split(strip(line), ',')
            length(parts) == 2 || continue
            push!(wn,   parse(Float64, parts[1]))
            push!(spec, parse(Float32, parts[2]))
        end
    end
    return wn, spec
end

# Load all spectra from all plastic-type subfolders, intrepolate onto training grid, conver to absorbance and normalize
function prepare_data_zenodo_dataset(loaded_model::LoadedModel,
                                      folder      ::String = "experimental_data/zenodo_plastics_dataset")

    subfolders  = sort(filter(d -> isdir(joinpath(folder, d)), readdir(folder)))
    spectra_abs = Vector{Vector{Float32}}()
    true_labels = Vector{String}()

    for subfolder in subfolders
        label          = get(ZENODO_LABEL_MAP, subfolder, "Other")
        subfolder_path = joinpath(folder, subfolder)
        csv_files      = sort(filter(f -> endswith(f, ".csv"),
                                     readdir(subfolder_path, join=true)))

        for filepath in csv_files
            wn, spectrum = load_zenodo_spectrum(filepath)
            interp       = interpolate_spectrum(wn, spectrum, loaded_model.train_wavenumbers)
            absorbance   = transmittance_to_absorbance(interp)
            peak_normalized = peak_normalize(absorbance)
            push!(spectra_abs, peak_normalized)
            push!(true_labels, label)
        end

        println("  $(rpad(subfolder, 10)) → $label  ($(length(csv_files)) spectra)")
    end

    lab_mu, lab_sigma = get_lab_normalizer(spectra_abs)
    spectra_norm = [normalize_spectrum(s, lab_mu, lab_sigma) for s in spectra_abs]

    println("Total: $(length(spectra_norm)) spectra prepared from $folder")
    return spectra_norm, true_labels, lab_mu, lab_sigma
end

# Run the full evaluation pipeline on the zenodo dataset.
function run_evaluation_zenodo_dataset_from_loaded_model(
        loaded_model::LoadedModel,
        folder      ::String  = "experimental_data/zenodo_plastics_dataset",
        threshold   ::Float32 = 0.5f0)

    spectra, true_labels, _, _ = prepare_data_zenodo_dataset(loaded_model, folder)
    predicted_labels = evaluate_batch(spectra, loaded_model, threshold)
    overall_acc, macro_f1, conf_matrix = evaluate_performance_batch(true_labels, predicted_labels)
    return overall_acc, macro_f1, conf_matrix
end

function run_evaluation_zenodo_dataset_from_model_path(
        model_path  ::String,
        parquet_path::String  = "src/parquet-files/data/IR_data_chunk001_of_020.parquet",
        folder      ::String  = "experimental_data/zenodo_plastics_dataset",
        threshold   ::Float32 = 0.5f0)

    loaded_model = load_model(model_path, parquet_path)
    loaded_model === nothing && return
    return run_evaluation_zenodo_dataset_from_loaded_model(loaded_model, folder, threshold)
end