"""
Modelos encoder-decoder para continuar corales.

E1: encoder-decoder LSTM.
E2: encoder-decoder GRU.
E3: encoder-decoder GRU con atencion aditiva (Bahdanau) sobre los 32 estados del encoder.

Decisiones comunes (iguales en E1-E3):
  - Cada paso musical son 4 notas. Cada nota pasa por una tabla de embeddings de
    tamano `emb_dim` y los 4 vectores se concatenan en el orden note0..note3,
    de modo que la entrada de la red en cada paso mide 4 * emb_dim.
    La tabla es compartida entre voces: la nota 60 es el mismo tono en cualquier voz.
  - El decoder se inicializa con el estado final del encoder (en LSTM, h y c).
  - La salida por paso son 4 vectores de 128 logits: forma (batch, 16, 4, 128).

La clase Seq2Seq incluye los metodos de generacion, para que el objeto registrado
en MLflow pueda generar musica sin codigo adicional.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bach import CONTEXT, HORIZON, N_VOICES, VOCAB


class StepEmbedding(nn.Module):
    """(batch, T, 4) enteros -> (batch, T, 4 * emb_dim) concatenando las 4 voces."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, emb_dim)

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        e = self.emb(steps)                         # (B, T, 4, emb)
        return e.flatten(start_dim=2)               # (B, T, 4*emb)


def make_rnn(cell: str, input_size: int, hidden: int, layers: int) -> nn.Module:
    rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
    return rnn_cls(input_size, hidden, num_layers=layers, batch_first=True)


class AdditiveAttention(nn.Module):
    """
    Atencion aditiva (Bahdanau):
        score(s, h_i) = v^T tanh(W_q s + W_k h_i)
        alpha = softmax(score)          sobre los 32 pasos del contexto
        contexto = sum_i alpha_i * h_i
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.W_q = nn.Linear(hidden, hidden)
        self.W_k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        # query: (B, 1, H)  estado actual del decoder | keys: (B, 32, H) salidas del encoder
        scores = self.v(torch.tanh(self.W_q(query) + self.W_k(keys)))   # (B, 32, 1)
        weights = F.softmax(scores.squeeze(-1), dim=-1)                 # (B, 32)
        context = torch.bmm(weights.unsqueeze(1), keys)                 # (B, 1, H)
        return context, weights


class Seq2Seq(nn.Module):
    def __init__(self, cell: str = "gru", attention: bool = False, hidden: int = 128,
                 emb_dim: int = 16, layers: int = 1):
        super().__init__()
        if attention and cell != "gru":
            raise ValueError("La guia define atencion solo sobre GRU (E3).")
        self.config = {"cell": cell, "attention": attention, "hidden": hidden,
                       "emb_dim": emb_dim, "layers": layers}
        self.cell, self.attention, self.layers = cell, attention, layers

        in_size = N_VOICES * emb_dim
        self.embed = StepEmbedding(emb_dim)
        self.encoder = make_rnn(cell, in_size, hidden, layers)
        if attention:
            self.attn = AdditiveAttention(hidden)
            # El decoder recibe [embedding del paso ; vector de contexto]
            self.decoder = make_rnn(cell, in_size + hidden, hidden, layers)
        else:
            self.decoder = make_rnn(cell, in_size, hidden, layers)
        self.out = nn.Linear(hidden, N_VOICES * VOCAB)

    # ------------------------------------------------------------------ partes
    def encode(self, x: torch.Tensor):
        """x: (B, 32, 4) -> salidas (B, 32, H) y estado final (h o (h, c))."""
        enc_out, state = self.encoder(self.embed(x))
        return enc_out, state

    def _top_hidden(self, state):
        h = state[0] if self.cell == "lstm" else state
        return h[-1].unsqueeze(1)                    # (B, 1, H), capa superior

    def decode_step(self, step: torch.Tensor, state, enc_out: torch.Tensor):
        """
        Un paso del decoder.
        step: (B, 1, 4) paso musical de entrada -> logits (B, 1, 4, 128), nuevo estado,
        pesos de atencion (B, 32) o None.
        """
        emb = self.embed(step)                       # (B, 1, 4*emb)
        weights = None
        if self.attention:
            context, weights = self.attn(self._top_hidden(state), enc_out)
            emb = torch.cat([emb, context], dim=-1)
        dec_out, state = self.decoder(emb, state)    # (B, 1, H)
        logits = self.out(dec_out).view(-1, 1, N_VOICES, VOCAB)
        return logits, state, weights

    # ------------------------------------------------------------------ entrenamiento
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Teacher forcing. Entradas del decoder: ultimo paso de X y luego y_1..y_15
        (el paso real anterior). Devuelve logits (B, 16, 4, 128) y atenciones (B, 16, 32) o None.
        """
        enc_out, state = self.encode(x)
        dec_in = torch.cat([x[:, -1:], y[:, :-1]], dim=1)   # (B, 16, 4)
        logits, attns = [], []
        for t in range(dec_in.size(1)):
            lg, state, w = self.decode_step(dec_in[:, t:t + 1], state, enc_out)
            logits.append(lg)
            if w is not None:
                attns.append(w)
        logits = torch.cat(logits, dim=1)
        return logits, (torch.stack(attns, dim=1) if attns else None)

    # ------------------------------------------------------------------ generacion
    @torch.no_grad()
    def generate(self, context, steps: int, temperature: float = 1.0,
                 generator: torch.Generator | None = None, return_attention: bool = False):
        """
        Generacion libre (sin teacher forcing) por bloques de hasta 16 pasos.

        context: (32, 4) enteros (lista, ndarray o tensor).
        Al terminar cada bloque, los ultimos 32 pasos disponibles son el nuevo contexto
        y el decoder se reinicializa con el estado del encoder de ese contexto.
        Cada nota se muestrea de softmax(logits / T).
        Devuelve (steps, 4) LongTensor con solo los pasos nuevos; si return_attention,
        tambien la lista de matrices de atencion (pasos_del_bloque, 32) por bloque.
        """
        self.eval()
        device = next(self.parameters()).device
        seq = torch.as_tensor(context, dtype=torch.long, device=device)
        if seq.shape != (CONTEXT, N_VOICES):
            raise ValueError(f"El contexto debe tener forma (32, 4), no {tuple(seq.shape)}")

        generated, attn_blocks = [], []
        remaining = steps
        while remaining > 0:
            block = min(HORIZON, remaining)
            ctx = seq[-CONTEXT:].unsqueeze(0)                    # (1, 32, 4)
            enc_out, state = self.encode(ctx)
            inp = ctx[:, -1:]                                    # ultimo paso del contexto
            block_attn = []
            for _ in range(block):
                logits, state, w = self.decode_step(inp, state, enc_out)
                probs = F.softmax(logits[0, 0] / temperature, dim=-1)        # (4, 128)
                nxt = torch.multinomial(probs, 1, generator=generator).view(1, 1, N_VOICES)
                generated.append(nxt[0, 0])
                seq = torch.cat([seq, nxt[0]], dim=0)
                inp = nxt
                if w is not None:
                    block_attn.append(w[0])
            if block_attn:
                attn_blocks.append(torch.stack(block_attn).cpu())
            remaining -= block

        out = torch.stack(generated).cpu()
        return (out, attn_blocks) if return_attention else out


def build_model(cfg: dict) -> Seq2Seq:
    return Seq2Seq(cell=cfg["cell"], attention=cfg["attention"], hidden=cfg["hidden"],
                   emb_dim=cfg["emb_dim"], layers=cfg["layers"])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
