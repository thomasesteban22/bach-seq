"""
Pruebas unitarias de datos, modelos, perdida, metricas y MIDI.
No necesitan MLflow ni red: usan corales sinteticos y modelos sin entrenar.
Las pruebas sobre el dataset real se saltan si data/jsb_chorales no existe.
  pytest -v tests/test_unit.py
"""

import os
import sys

import numpy as np
import pandas as pd
import pretty_midi
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bach import CONTEXT, HORIZON, N_VOICES, VOCAB, VOICES  # noqa: E402
from bach.data import STRIDES, Chorale, load_all, load_split, make_windows  # noqa: E402
from bach.models import Seq2Seq, count_params  # noqa: E402
from bach.music import continuation_stats, to_midi  # noqa: E402
from bach.training import evaluate, loss_fn  # noqa: E402

DATA_ROOT = os.path.join(ROOT, "data", "jsb_chorales")
CONFIGS = {
    "E1": dict(cell="lstm", attention=False),
    "E2": dict(cell="gru", attention=False),
    "E3": dict(cell="gru", attention=True),
}


def chorale(name, length, offset=0):
    """Coral sintetico: el valor de cada nota codifica su paso, para rastrear ventanas."""
    t = np.arange(length) + offset
    notes = np.stack([(t % 100) + 1 + v for v in range(N_VOICES)], axis=1).astype(np.int64)
    return Chorale(name, notes)


def random_batch(batch=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, VOCAB, (batch, CONTEXT, N_VOICES), generator=g)
    y = torch.randint(0, VOCAB, (batch, HORIZON, N_VOICES), generator=g)
    return x, y


# ---------------------------------------------------------------- datos

def test_strides_de_la_guia():
    assert STRIDES == {"train": 8, "valid": 16, "test": 16}


@pytest.mark.parametrize("length,stride", [(48, 8), (100, 8), (100, 16), (47, 8), (64, 16)])
def test_ventanas_forma_y_cantidad(length, stride):
    if length < CONTEXT + HORIZON:
        with pytest.raises(ValueError):        # np.stack de una lista vacia
            make_windows([chorale("c", length)], stride)
        return
    X, Y, origin = make_windows([chorale("c", length)], stride)
    esperadas = (length - (CONTEXT + HORIZON)) // stride + 1
    assert X.shape == (esperadas, CONTEXT, N_VOICES)
    assert Y.shape == (esperadas, HORIZON, N_VOICES)
    assert [s for _, s in origin] == list(range(0, esperadas * stride, stride))


def test_ventanas_son_tramos_contiguos_del_coral():
    c = chorale("c", 100)
    X, Y, origin = make_windows([c], 8)
    for x, y, (_, start) in zip(X, Y, origin):
        np.testing.assert_array_equal(x, c.notes[start:start + CONTEXT])
        np.testing.assert_array_equal(y, c.notes[start + CONTEXT:start + CONTEXT + HORIZON])


def test_ninguna_ventana_cruza_dos_corales():
    a, b = chorale("a", 70), chorale("b", 90, offset=500)
    X, Y, origin = make_windows([a, b], 8)
    largos = {"a": 70, "b": 90}
    for name, start in origin:
        assert start + CONTEXT + HORIZON <= largos[name]
    # 70 pasos -> 3 ventanas (0, 8, 16); 90 pasos -> 6 ventanas (0..40)
    assert [n for n, _ in origin] == ["a"] * 3 + ["b"] * 6


def test_load_split_valida_columnas_y_rango(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    pd.DataFrame([[60, 55, 52, 0]] * 5, columns=VOICES).to_csv(split / "chorale_000.csv", index=False)
    corales = load_split(str(tmp_path), "train")
    assert len(corales) == 1 and corales[0].notes.shape == (5, 4) and corales[0].notes[0, 3] == 0

    pd.DataFrame([[60, 55, 52, 200]], columns=VOICES).to_csv(split / "chorale_001.csv", index=False)
    with pytest.raises(ValueError, match="fuera de 0..127"):
        load_split(str(tmp_path), "train")

    (split / "chorale_001.csv").unlink()
    pd.DataFrame([[60, 55, 52, 48]], columns=["a", "b", "c", "d"]).to_csv(split / "chorale_002.csv", index=False)
    with pytest.raises(ValueError, match="columnas"):
        load_split(str(tmp_path), "train")


@pytest.mark.skipif(not os.path.isdir(os.path.join(DATA_ROOT, "train")), reason="dataset no descargado")
def test_dataset_real_particiones_y_ventanas():
    splits = load_all(DATA_ROOT)
    assert {s: len(c) for s, c in splits.items()} == {"train": 229, "valid": 76, "test": 77}
    ventanas = {s: len(make_windows(splits[s], STRIDES[s])[0]) for s in splits}
    assert ventanas == {"train": 5681, "valid": 982, "test": 1011}
    nombres = {s: {c.name for c in cs} for s, cs in splits.items()}
    assert not (nombres["train"] & nombres["valid"]) and not (nombres["valid"] & nombres["test"])


# ---------------------------------------------------------------- modelos

@pytest.mark.parametrize("run", ["E1", "E2", "E3"])
def test_forward_forma_de_logits(run):
    model = Seq2Seq(**CONFIGS[run])
    x, y = random_batch(batch=3)
    logits, attn = model(x, y)
    assert logits.shape == (3, HORIZON, N_VOICES, VOCAB)
    if run == "E3":
        assert attn.shape == (3, HORIZON, CONTEXT)
        torch.testing.assert_close(attn.sum(-1), torch.ones(3, HORIZON))
    else:
        assert attn is None


@pytest.mark.parametrize("run", ["E1", "E2", "E3"])
def test_teacher_forcing_usa_el_paso_real_anterior(run):
    """La salida del paso k depende de y_1..y_(k-1), nunca de y_k ni de lo que sigue."""
    torch.manual_seed(0)
    model = Seq2Seq(**CONFIGS[run]).eval()
    x, y = random_batch(batch=2)
    y2 = y.clone()
    k = 5
    y2[:, k] = (y2[:, k] + 1) % VOCAB            # cambia solo el objetivo del paso k (indice 0)
    with torch.no_grad():
        a, _ = model(x, y)
        b, _ = model(x, y2)
    torch.testing.assert_close(a[:, :k + 1], b[:, :k + 1])   # pasos 0..k no ven y_k
    assert not torch.allclose(a[:, k + 1], b[:, k + 1])      # el paso k+1 recibe y_k


def test_atencion_solo_con_gru():
    with pytest.raises(ValueError):
        Seq2Seq(cell="lstm", attention=True)


@pytest.mark.parametrize("cfg,esperados", [
    (dict(cell="lstm", attention=False), 266_752),
    (dict(cell="gru", attention=False), 217_088),
    (dict(cell="gru", attention=True), 299_393),
    (dict(cell="gru", attention=False, hidden=256), 628_224),
])
def test_parametros_coinciden_con_la_tabla(cfg, esperados):
    assert count_params(Seq2Seq(**cfg)) == esperados


# ---------------------------------------------------------------- perdida y metricas

def test_perdida_es_entropia_cruzada_promedio_con_silencio():
    torch.manual_seed(0)
    logits = torch.randn(2, HORIZON, N_VOICES, VOCAB)
    y = torch.randint(0, VOCAB, (2, HORIZON, N_VOICES))
    y[0, 0, 0] = 0                                               # el silencio es una clase mas
    manual = torch.stack([F.cross_entropy(logits[b, k, v][None], y[b, k, v][None])
                          for b in range(2) for k in range(HORIZON) for v in range(N_VOICES)]).mean()
    torch.testing.assert_close(loss_fn(logits, y), manual)
    solo_silencio = loss_fn(logits, torch.zeros_like(y))
    assert torch.isfinite(solo_silencio) and solo_silencio > 0


class Oraculo(nn.Module):
    """Predice exactamente y, salvo la voz `falla` en los pasos pares, que se equivoca."""

    def __init__(self, falla=None):
        super().__init__()
        self.falla = falla
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        pred = y.clone()
        if self.falla is not None:
            pred[:, ::2, self.falla] = (pred[:, ::2, self.falla] + 1) % VOCAB
        return F.one_hot(pred, VOCAB).float() * 50, None


def test_exactitudes_por_nota_y_por_paso():
    x, y = random_batch(batch=5)
    loader = [(x, y)]
    perfecto = evaluate(Oraculo(), loader, "cpu")
    assert perfecto["note_acc"] == 1.0 and perfecto["step_acc"] == 1.0
    # Una voz falla en 8 de 16 pasos: 8 de 64 notas mal, y la mitad de los pasos mal.
    parcial = evaluate(Oraculo(falla=2), loader, "cpu")
    assert parcial["note_acc"] == pytest.approx(56 / 64)
    assert parcial["step_acc"] == pytest.approx(0.5)


# ---------------------------------------------------------------- generacion

CTX = [[72, 67, 60, 48]] * 16 + [[69, 65, 60, 50]] * 16


@pytest.mark.parametrize("steps,bloques", [(1, [1]), (16, [16]), (33, [16, 16, 1]), (64, [16, 16, 16, 16])])
def test_generacion_por_bloques(steps, bloques):
    torch.manual_seed(0)
    model = Seq2Seq(**CONFIGS["E3"])
    out, attn = model.generate(CTX, steps, generator=torch.Generator().manual_seed(1), return_attention=True)
    assert out.shape == (steps, N_VOICES) and out.dtype == torch.long
    assert int(out.min()) >= 0 and int(out.max()) < VOCAB
    assert [a.shape[0] for a in attn] == bloques                # un mapa por bloque
    assert all(a.shape[1] == CONTEXT for a in attn)


def test_generacion_reproducible_con_la_misma_semilla():
    torch.manual_seed(0)
    model = Seq2Seq(**CONFIGS["E2"])
    a = model.generate(CTX, 20, generator=torch.Generator().manual_seed(42))
    b = model.generate(CTX, 20, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(a, b)


def test_generacion_rechaza_contexto_de_otra_forma():
    model = Seq2Seq(**CONFIGS["E2"])
    with pytest.raises(ValueError, match=r"\(32, 4\)"):
        model.generate(CTX[:31], 16)


# ---------------------------------------------------------------- MIDI y estadisticas

def test_midi_fusiona_notas_iguales_y_omite_silencios(tmp_path):
    notas = np.array([[72, 67, 60, 0],
                      [72, 67, 62, 0],
                      [72, 65, 62, 48],
                      [74, 65, 62, 48]])
    path = str(tmp_path / "prueba.mid")
    to_midi(notas, path, tempo=80)
    pm = pretty_midi.PrettyMIDI(path)
    assert [i.name for i in pm.instruments] == VOICES
    paso = 60 / (4 * 80)                                        # 0.1875 s por semicorchea
    soprano = [(n.pitch, round(n.start / paso), round(n.end / paso)) for n in pm.instruments[0].notes]
    assert soprano == [(72, 0, 3), (74, 3, 4)]                  # tres 72 seguidos = una nota larga
    bajo = [(n.pitch, round(n.start / paso), round(n.end / paso)) for n in pm.instruments[3].notes]
    assert bajo == [(48, 2, 4)]                                 # los 0 no producen nota


def test_estadisticas_de_continuacion():
    g = np.array([[72, 67, 60, 48],
                  [72, 67, 60, 48],
                  [72, 67, 60, 0],
                  [74, 67, 62, 50]])
    s = continuation_stats(g)
    assert s["silencios_%"] == pytest.approx(100 / 16, abs=0.1)
    assert s["repeticion_%"] == pytest.approx(100 / 3, abs=0.1)
    assert s["pasos_distintos"] == 3
    assert s["notas_distintas_por_voz"] == {"note0": 2, "note1": 1, "note2": 2, "note3": 3}
