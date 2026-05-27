#Runs full evaluation pipeline on la session measurements.

include("PlasticPredictor.jl")

loaded_model = load_model("model.jld2")

if loaded_model !== nothing
    run_evaluation_session_pipeline_from_loaded_model(loaded_model, "experimental_data/lab_session_1")
end