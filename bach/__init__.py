"""Generacion de corales de Bach con modelos encoder-decoder recurrentes."""

CONTEXT = 32       # pasos de contexto (X)
HORIZON = 16       # pasos a predecir (Y)
N_VOICES = 4       # note0..note3
VOCAB = 128        # notas MIDI 0..127; 0 = silencio
SEED = 42
VOICES = ["note0", "note1", "note2", "note3"]
