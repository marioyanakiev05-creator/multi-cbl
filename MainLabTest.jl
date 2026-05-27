include("PlasticPredictor.jl")
lm = load_model("model.jld2")
run_evaluation_zenodo_dataset_from_loaded_model(lm)