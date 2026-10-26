"""
Generacion libre con el modelo elegido y mapa de atencion de E3.

  - Contexto: primeros 32 pasos de dos corales distintos de valid.
  - 32 pasos nuevos por coral (2 bloques de 16), sin teacher forcing,
    re-codificando los ultimos 32 pasos tras cada bloque.
  - Muestreo de softmax(logits / T) con T = 1.0 y semilla fija.
  - MIDI = contexto + continuacion, tempo documentado en bach.music.TEMPO.
  - Mapa de atencion de E3 (16 pasos generados x 32 de contexto).

Todo queda en results/ y como artefactos del run elegido (carpeta generation/).
"""

import json
import os

from _common import load_data

import mlflow
import numpy as np

from bach.config import RESULTS_DIR, setup_mlflow
from bach.music import TEMPO, attention_map, continuation_stats, generate, plot_attention, to_midi
from bach.runs import load_run_checkpoint, run_by_lab_id

CHORALES = [0, 40]      # indices dentro de la particion valid
STEPS, TEMPERATURE, SAMPLING_SEED = 32, 1.0, 42


def main():
    setup_mlflow()
    out_dir = os.path.join(RESULTS_DIR, "generation")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "selection.json")) as f:
        sel = json.load(f)
    splits, _ = load_data()
    model, _ = load_run_checkpoint(sel["run_id"])

    report = {"model_run": sel["selected_run"], "run_id": sel["run_id"], "steps": STEPS,
              "temperature": TEMPERATURE, "sampling_seed": SAMPLING_SEED,
              "tempo_bpm": TEMPO, "step_duration_s": 60 / (4 * TEMPO), "chorales": []}

    for i, idx in enumerate(CHORALES):
        ch = splits["valid"][idx]
        context = ch.notes[:32]
        cont = generate(model, context, STEPS, TEMPERATURE, seed=SAMPLING_SEED + i)
        full = np.concatenate([context, cont])
        midi = os.path.join(out_dir, f"{ch.name}_continuacion.mid")
        to_midi(full, midi)
        np.savetxt(os.path.join(out_dir, f"{ch.name}_continuacion.csv"), full, fmt="%d",
                   delimiter=",", header="note0,note1,note2,note3", comments="")
        stats = continuation_stats(cont)
        ref = continuation_stats(ch.notes[32:32 + STEPS])
        report["chorales"].append({"valid_index": idx, "chorale": ch.name,
                                   "sampling_seed": SAMPLING_SEED + i, "midi": os.path.basename(midi),
                                   "stats_generado": stats, "stats_original_mismo_tramo": ref})
        print(ch.name, stats)

    # Mapa de atencion de E3 (siempre E3, aunque el elegido sea otro modelo)
    e3 = run_by_lab_id("E3")
    e3_model, _ = load_run_checkpoint(e3["run_id"])
    ch = splits["valid"][CHORALES[0]]
    w = attention_map(e3_model, ch.notes[:32], TEMPERATURE, seed=SAMPLING_SEED)
    att_png = os.path.join(out_dir, "attention_E3.png")
    plot_attention(w, att_png, f"E3: atencion al generar 16 pasos ({ch.name}, T={TEMPERATURE})")
    np.savetxt(os.path.join(out_dir, "attention_E3.csv"), w, fmt="%.4f", delimiter=",")
    top = w.argmax(axis=1) + 1
    report["attention_E3"] = {"chorale": ch.name, "argmax_context_step_por_paso": top.tolist(),
                              "peso_ultimos_8_pasos_%": round(100 * float(w[:, -8:].sum(1).mean()), 1),
                              "peso_max_medio": round(float(w.max(1).mean()), 3)}
    print("Atencion E3:", report["attention_E3"])

    with open(os.path.join(out_dir, "generation_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with mlflow.start_run(run_id=sel["run_id"]):
        mlflow.log_artifacts(out_dir, artifact_path="generation")
        mlflow.log_params({"gen_temperature": TEMPERATURE, "gen_seed": SAMPLING_SEED,
                           "gen_steps": STEPS, "gen_tempo_bpm": TEMPO,
                           "gen_chorales": ",".join(c["chorale"] for c in report["chorales"])})
    with mlflow.start_run(run_id=e3["run_id"]):
        mlflow.log_artifact(att_png, artifact_path="attention")


if __name__ == "__main__":
    main()
