"""
Entrenamiento, evaluacion y registro en MLflow.

Protocolo (identico para E1-E4):
  - Semilla 42 reiniciada al comenzar cada ejecucion.
  - Adam, lr 1e-3, lotes de 64, maximo 20 epocas.
  - Early stopping: se detiene tras 4 epocas sin bajar la perdida de validacion.
  - Se guarda el checkpoint de la epoca con menor perdida de validacion (pesos +
    configuracion + epoca), y las exactitudes se calculan recargando ESE checkpoint.
  - Perdida: entropia cruzada promedio sobre 16 pasos x 4 voces y sobre los ejemplos.
    Los logits se reorganizan a (batch*16*4, 128) sin softmax previo.
  - Validacion en modo eval, sin gradientes y con teacher forcing.
"""

from __future__ import annotations

import json
import os
import platform
import random
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn

from bach import CONTEXT, HORIZON, SEED, VOCAB
from bach.data import STRIDES, make_loader
from bach.models import build_model, count_params

DEFAULTS = {
    "cell": "gru", "attention": False, "hidden": 128, "emb_dim": 16, "layers": 1,
    "lr": 1e-3, "batch_size": 64, "max_epochs": 20, "patience": 4, "seed": SEED,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_name() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    return torch.device("cpu"), f"CPU {platform.processor() or platform.machine()} ({os.cpu_count()} nucleos)"


def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Entropia cruzada promedio sobre todas las notas del lote (16 x 4 x batch)."""
    return nn.functional.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Perdida y exactitudes con teacher forcing, acumuladas sobre todos los ejemplos."""
    model.eval()
    total_loss, n_examples = 0.0, 0
    notes_ok, notes_total, steps_ok, steps_total = 0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, y)
        total_loss += loss_fn(logits, y).item() * x.size(0)
        n_examples += x.size(0)
        pred = logits.argmax(dim=-1)                       # (B, 16, 4)
        hit = pred == y
        notes_ok += hit.sum().item()
        notes_total += hit.numel()
        steps_ok += hit.all(dim=-1).sum().item()           # las 4 voces aciertan
        steps_total += hit.shape[0] * hit.shape[1]
    return {"loss": total_loss / n_examples,
            "note_acc": notes_ok / notes_total,
            "step_acc": steps_ok / steps_total}


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x, y)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


def plot_losses(history: dict, best_epoch: int, title: str, path: str) -> None:
    epochs = range(1, len(history["train"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train"], marker="o", label="train")
    ax.plot(epochs, history["valid"], marker="o", label="valid")
    ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"mejor epoca ({best_epoch})")
    ax.set_xlabel("Epoca"); ax.set_ylabel("Entropia cruzada")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def load_checkpoint(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def run_experiment(run_name: str, cfg: dict, windows: dict, description: str = "",
                   extra_tags: dict | None = None, attention_fn=None) -> str:
    """
    Entrena una configuracion y la registra como un run de MLflow. Devuelve el run_id.

    attention_fn: funcion opcional (modelo, carpeta) -> ruta de imagen; se usa en E3
    para guardar el mapa de atencion como artefacto.
    """
    cfg = {**DEFAULTS, **cfg}
    set_seed(cfg["seed"])
    device, device_label = device_name()

    Xtr, Ytr, _ = windows["train"]
    Xva, Yva, _ = windows["valid"]
    train_loader = make_loader(Xtr, Ytr, cfg["batch_size"], shuffle=True, seed=cfg["seed"])
    valid_loader = make_loader(Xva, Yva, cfg["batch_size"], shuffle=False)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    n_params = count_params(model)

    with mlflow.start_run(run_name=run_name, description=description) as run, \
            tempfile.TemporaryDirectory() as tmp:
        mlflow.set_tags({"lab_id": run_name, **(extra_tags or {})})
        mlflow.log_params({
            "architecture": f"encoder-decoder {cfg['cell'].upper()}",
            "cell": cfg["cell"], "attention": cfg["attention"],
            "attention_type": "aditiva (Bahdanau)" if cfg["attention"] else "ninguna",
            "layers": cfg["layers"], "hidden_dim": cfg["hidden"], "embedding_dim": cfg["emb_dim"],
            "embedding_combination": "concatenacion de 4 voces (tabla compartida)",
            "learning_rate": cfg["lr"], "batch_size": cfg["batch_size"],
            "max_epochs": cfg["max_epochs"], "patience": cfg["patience"], "seed": cfg["seed"],
            "optimizer": "Adam", "context": CONTEXT, "horizon": HORIZON,
            "stride_train": STRIDES["train"], "stride_valid_test": STRIDES["valid"],
            "train_windows": len(Xtr), "valid_windows": len(Xva),
        })
        mlflow.log_metric("trainable_params", n_params)
        mlflow.set_tag("device", device_label)

        ckpt_path = os.path.join(tmp, "best_checkpoint.pt")
        history = {"train": [], "valid": []}
        best_loss, best_epoch, bad_epochs = float("inf"), 0, 0
        t0 = time.time()

        for epoch in range(1, cfg["max_epochs"] + 1):
            tr = train_one_epoch(model, train_loader, optimizer, device)
            va = evaluate(model, valid_loader, device)["loss"]
            history["train"].append(tr)
            history["valid"].append(va)
            mlflow.log_metrics({"train_loss": tr, "val_loss": va}, step=epoch)
            print(f"[{run_name}] epoca {epoch:2d}  train {tr:.4f}  valid {va:.4f}", flush=True)

            if va < best_loss:
                best_loss, best_epoch, bad_epochs = va, epoch, 0
                torch.save({"state_dict": model.state_dict(), "config": cfg,
                            "epoch": epoch, "val_loss": va}, ckpt_path)
            else:
                bad_epochs += 1
                if bad_epochs >= cfg["patience"]:
                    print(f"[{run_name}] early stopping en la epoca {epoch}", flush=True)
                    break

        train_time = time.time() - t0

        # Exactitudes del MEJOR checkpoint, no de la ultima epoca.
        best_model, _ = load_checkpoint(ckpt_path, device)
        val = evaluate(best_model, valid_loader, device)

        mlflow.log_metrics({
            "best_epoch": best_epoch, "best_val_loss": best_loss,
            "val_loss_best_ckpt": val["loss"],
            "val_note_acc": val["note_acc"], "val_step_acc": val["step_acc"],
            "epochs_run": len(history["train"]), "train_time_s": train_time,
        })

        cfg_path = os.path.join(tmp, "config.json")
        with open(cfg_path, "w") as f:
            json.dump({**cfg, "best_epoch": best_epoch, "trainable_params": n_params,
                       "device": device_label, "train_time_s": round(train_time, 1)}, f, indent=2)
        curve = os.path.join(tmp, "loss_curve.png")
        plot_losses(history, best_epoch, f"{run_name}: perdida por epoca", curve)
        resources = os.path.join(tmp, "resources.json")
        with open(resources, "w") as f:
            json.dump({"trainable_params": n_params, "train_time_s": round(train_time, 1),
                       "device": device_label, "torch": torch.__version__}, f, indent=2)

        mlflow.log_artifact(cfg_path)
        mlflow.log_artifact(curve)
        mlflow.log_artifact(resources)
        mlflow.log_artifact(ckpt_path, artifact_path="checkpoint")

        if attention_fn is not None:
            img = attention_fn(best_model, tmp)
            mlflow.log_artifact(img, artifact_path="attention")

        print(f"[{run_name}] mejor epoca {best_epoch}  val_loss {best_loss:.4f}  "
              f"note_acc {val['note_acc']:.4f}  step_acc {val['step_acc']:.4f}  "
              f"params {n_params}  tiempo {train_time:.0f}s", flush=True)
        return run.info.run_id
