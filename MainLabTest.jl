include("PlasticPredictor.jl")
lm = load_model("model.jld2")
run_evaluation_session_pipeline_from_loaded_model(lm)