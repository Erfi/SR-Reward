from pathlib import Path
from eval_utils import (
    find_all_unevaluated_models,
    find_all_evaluated_models,
    get_all_evaluations,
    evaluate_model,
    remove_incomplete_runs,
)


if __name__ == "__main__":
    # Remove all the incomplete runs (BECAREFUL WITH THIS, ARE MODELS STILL TRAINING?)
    # remove_incomplete_runs(Path("models"))

    # Evaluate all models that have not been evaluated yet
    model_paths = find_all_unevaluated_models(Path("models"), only_completed_runs=True)
    for model_path in model_paths:
        evaluate_model(model_path, n_eval_episodes=30)

    # Get all the evaluated models
    model_paths = find_all_evaluated_models(Path("models"))
    get_all_evaluations(model_paths, save_path=Path("all_evaluations.txt"))
