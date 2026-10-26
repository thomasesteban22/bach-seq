"""
Selecciona el modelo final por menor perdida de validacion (E1-E4) y lo evalua
UNA sola vez en test. El resultado se registra en el run elegido y se guarda en
results/ para el README.
"""

import json
import os

from _common import load_data

import mlflow
import torch

from bach.config import RESULTS_DIR, setup_mlflow
from bach.data import make_loader
from bach.runs import best_run, comparison_table, experiment_runs, load_run_checkpoint
from bach.training import evaluate


def main():
    setup_mlflow()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = experiment_runs()
    table = comparison_table(df)
    table.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)
    print(table.drop(columns="run_id").to_string(index=False))

    best = best_run(df)
    run_id, lab_id = best["run_id"], best["tags.lab_id"]
    print(f"\nModelo elegido: {lab_id} (run {run_id}), val_loss {best['metrics.best_val_loss']:.4f}")

    _, windows = load_data()
    Xte, Yte, _ = windows["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_run_checkpoint(run_id, device)
    test = evaluate(model, make_loader(Xte, Yte, 64, shuffle=False), device)
    print(f"Test: loss {test['loss']:.4f}  note_acc {test['note_acc']:.4f}  step_acc {test['step_acc']:.4f}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({"test_loss": test["loss"], "test_note_acc": test["note_acc"],
                            "test_step_acc": test["step_acc"]})
        mlflow.set_tags({"selected": "true", "test_windows": str(len(Xte))})

    out = {"selected_run": lab_id, "run_id": run_id,
           "val_loss": float(best["metrics.best_val_loss"]),
           "test_windows": len(Xte), **{f"test_{k}": v for k, v in test.items()}}
    with open(os.path.join(RESULTS_DIR, "selection.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
