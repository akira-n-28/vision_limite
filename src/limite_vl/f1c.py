"""Dati di F1c: domande visive brevi con risposte verificabili e testi senza informazioni non visibili.

- Domande e risposte: una richiesta con 1-5 domande numerate e una risposta breve per riga (`qa_messages`). Le
  domande vengono da fatti certi: i diagrammi sintetici (limite_vl.synth) e le didascalie di MAVIS, dove
  `caption.jpg` disegna tutte le misure citate (`mavis_questions`). Ogni domanda ha anche le opzioni per la
  valutazione a scelta forzata.
- MAVIS: didascalia preceduta dal numero e dall'elenco delle figure (`mavis_count_caption`), così il conteggio
  viene scritto prima della descrizione; le domande "Is there a shape named X?" con un nome inventato nello stile
  di MAVIS (FEGH dopo ADEF) allenano a non aggiungere figure.
- STEM: `visible_caption` tiene solo le sezioni sul contenuto visibile delle descrizioni di Qwen3-VL e toglie le
  frasi che parlano della domanda, del problema o di proprietà dedotte.
- Formule su più righe: `array_rows` divide un ambiente array nelle sue righe, `split_lines` taglia l'immagine nei
  n-1 spazi bianchi più alti se sono nettamente più alti degli altri (scripts/shard_unimer.py --lines).
"""
from __future__ import annotations

import hashlib
import random
import re

from .facts import mavis_facts
from .f2 import _IMAGE_TITLE
from .latex import visible_tokens

QA_ONE = "Answer the question with a short answer.\n"
QA_MANY = "Answer each question with a short answer.\n"
YES_NO = ["yes", "no"]


def key_rng(key: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little"))


def qa_messages(questions: list[dict]) -> tuple[str, str]:
    """(richiesta, risposta) per una o più domande {"q", "answer"}."""
    if len(questions) == 1:
        return QA_ONE + questions[0]["q"], questions[0]["answer"]
    prompt = QA_MANY + "\n".join(f"{i}. {q['q']}" for i, q in enumerate(questions, 1))
    return prompt, "\n".join(f"{i}. {q['answer']}" for i, q in enumerate(questions, 1))


def pick_questions(key: str, items: list[dict]) -> list[dict]:
    """Da 1 a 5 domande diverse, scelte in modo deterministico dalla chiave."""
    if not items:
        raise ValueError(f"{key}: nessuna domanda")
    rng = key_rng(key + "/pick")
    k = min(len(items), rng.choice([1, 2, 3, 3, 4, 5]))
    return rng.sample(items, k)


def _qa(qtype: str, q: str, options: list[str], answer: str) -> dict:
    if answer not in options:
        raise ValueError(f"{q}: risposta {answer!r} fuori dalle opzioni {options}")
    return {"type": qtype, "q": q, "options": options, "answer": answer}


# ---------------------------------------------------------------- MAVIS

MAVIS_KINDS = ("square", "rectangle", "parallelogram", "rhombus", "trapezoid", "kite", "triangle", "right triangle",
               "isosceles triangle", "equilateral triangle", "sector", "circle", "semicircle")
LENGTH_POOL = ("3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "2*√(2)", "3*√(2)", "2*√(3)", "4*√(3)")
ANGLE_POOL = ("30°", "45°", "60°", "90°", "120°", "135°")


def _main_shapes(f: dict) -> list[str]:
    return [k for k in f["shape_order"] if not k.startswith("ext:")]


def _options(rng: random.Random, answer: str, pool, n: int = 4) -> list[str]:
    opts = [answer] + rng.sample([x for x in dict.fromkeys(pool) if x != answer], n - 1)
    rng.shuffle(opts)
    return opts


def _fake_name(f: dict, main: list[str]) -> str:
    """Nome di una figura che non c'è, come la continuerebbe MAVIS: le ultime due lettere dell'ultima figura al
    contrario più le due lettere successive non usate (ADEF -> FEGH)."""
    used = set("".join(f["display"].values()))
    last = f["display"][f"shape:{main[-1]}"]
    nxt = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in used and c > max(used)]
    return last[-1] + last[-2] + "".join(nxt[:2]) if len(nxt) >= 2 else ""


def mavis_questions(caption: str, depth: int, key: str) -> list[dict]:
    """Domande verificabili su un diagramma MAVIS (caption.jpg). Figura inventata solo se le regole trovano tante
    figure principali quante ne dice la profondità (depth + 1); conteggio solo se in più nessun lato è prolungato in
    un'altra figura (a vista sarebbe una figura in più)."""
    f = mavis_facts(caption)
    rng = key_rng(key + "/mavis-qa")
    main = _main_shapes(f)
    consistent = bool(main) and len(main) == depth + 1
    out = []
    # un lato prolungato in un triangolo o in un semicerchio a vista è una figura in più: conteggio ambiguo
    if consistent and len(main) == len(f["shape"]):
        out.append(_qa("count", "How many shapes are in the figure?", ["1", "2", "3", "4", "5"], str(len(main))))
    extended = [set(k[4:]) for k in f["shape"] if k.startswith("ext:")]
    for k in main:
        if any(e <= set(k) for e in extended):  # un lato prolungato: a vista non è più "rectangle" (audit di F1c)
            continue
        out.append(_qa("shape_type", f"What type of shape is {f['display'][f'shape:{k}']}?",
                       _options(rng, f["shape"][k], MAVIS_KINDS), f["shape"][k]))
    raw_sides = [f["raw"][f"side:{k}"] for k in f["side"]]
    for k in f["side"]:
        raw = f["raw"][f"side:{k}"]
        out.append(_qa("side_len", f"What is the length of side {f['display'][f'side:{k}']}?",
                       _options(rng, raw, raw_sides + list(LENGTH_POOL)), raw))
    for k in f["angle"]:
        ans = f["raw"][f"angle:{k}"] + "°"
        out.append(_qa("angle_val", f"What is the measure of angle {f['display'][f'angle:{k}']}?",
                       _options(rng, ans, ANGLE_POOL + (ans,)), ans))
    for a, b in zip(main, main[1:]):
        na, nb = f["display"][f"shape:{a}"], f["display"][f"shape:{b}"]
        sides = [na[i] + na[(i + 1) % len(na)] for i in range(len(na))]
        common = {c for c in na if c in nb}
        shared = [s for s in sides if set(s) == common]
        if len(common) == 2 and len(shared) == 1:
            out.append(_qa("shared_side", f"Which side do {na} and {nb} share?", sides[:4] if shared[0] in sides[:4]
                           else sides[:3] + shared, shared[0]))
    if consistent:
        out.append(_qa("exists", f"Is there a shape named {f['display'][f'shape:{rng.choice(main)}']}?", YES_NO, "yes"))
        fake = _fake_name(f, main)
        if fake:
            out.append(_qa("exists", f"Is there a shape named {fake}?", YES_NO, "no"))
    return out


def mavis_count_caption(caption: str, depth: int) -> str:
    """Didascalia preceduta da "The figure contains N shapes: ...", se le figure trovate sono depth + 1 e nessun lato
    è prolungato in un'altra figura."""
    f = mavis_facts(caption)
    main = _main_shapes(f)
    if not main or len(main) != depth + 1 or len(main) != len(f["shape"]):  # niente prolungamenti (conteggio ambiguo)
        return caption
    items = [f"{f['display'][f'shape:{k}']} ({f['shape'][k]})" for k in main]
    listing = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    noun = "shape" if len(main) == 1 else "shapes"
    return f"The figure contains {len(main)} {noun}: {listing}. {caption}"


# ---------------------------------------------------------------- STEM

_ANY_SECTION = re.compile(r"^\s*\*\*\s*([^*:]{2,60}?)\s*:?\s*\*\*\s*:?\s*")
KEEP_SECTIONS = {"scene summary", "explicit component breakdown", "interactions and relationships"}
NON_VISUAL = re.compile(r"\b(question|problem|solv\w*|theorem|infer\w*|implie[sd]|imply|implicit\w*|derived?|assum\w*|"
                        r"likely|suggest\w*|intended|probably|presumabl\w*|answers?|asked|unknown|determine|calculat\w*)\b",
                        re.IGNORECASE)
_PROBLEM_FIGURE = re.compile(r"\b(?:problem|question)\s+(diagram|figure|illustration|image|graph)\b", re.IGNORECASE)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z*`$(])")
_TAIL = re.compile(r",?\s+(?:used to|accompanied by|for (?:a|the) (?:geometry |math |physics )?(?:problem|question|task)|"
                   r"to (?:determine|find|solve|calculate)|illustrating (?:a|the) (?:problem|question)).*$", re.IGNORECASE)


def visible_caption(caption: str) -> str:
    """Descrizione STEM senza la riga del tipo di immagine, senza titoli, senza le sezioni non visive (proprietà
    dedotte, ambiguità, contesto della domanda) e senza le frasi che citano domanda, problema o deduzioni."""
    out, keep = [], True
    for line in caption.splitlines():
        if _IMAGE_TITLE.match(line):
            keep = True
            continue
        m = _ANY_SECTION.match(line)
        if m and line.strip().startswith("**") and m.group(1).strip().lower() in KEEP_SECTIONS | {
                "implicit and inferred properties", "identified ambiguities", "inferred properties", "question context",
                "problem context", "answer inference", "correct inference", "answer choices"}:
            keep = m.group(1).strip().lower() in KEEP_SECTIONS
            line = line[m.end():]
            if not line.strip():
                continue
        if not keep:
            continue
        sentences = []
        line = _PROBLEM_FIGURE.sub(r"\1", line)  # "a geometry problem diagram" descrive comunque la figura
        for s in _SENT.split(line.rstrip()):
            s = _TAIL.sub(".", s) if _TAIL.search(s) else s
            if not NON_VISUAL.search(s):
                sentences.append(s)
        text = " ".join(sentences).rstrip()
        if text.strip() in ("*", "-", "* .", "."):
            continue
        if text or not line.strip():
            out.append(text)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


# ---------------------------------------------------------------- formule su più righe

_ARRAY = re.compile(r"^\s*\\begin\{array\}\{[^{}]*\}(.*)\\end\{array\}\s*$", re.DOTALL)


def array_rows(label: str) -> list[str] | None:
    """Righe (senza &) di una formula che è un solo ambiente array con almeno 2 righe visibili, altrimenti None."""
    m = _ARRAY.match(label)
    if not m:
        return None
    body, rows, cur, depth, i = m.group(1), [], "", 0, 0
    while i < len(body):
        if body.startswith("\\\\", i) and depth == 0:
            rows.append(cur)
            cur, i = "", i + 2
            continue
        if body[i] == "\\" and i + 1 < len(body):
            cur += body[i:i + 2]
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(body[i], 0)
        cur += body[i]
        i += 1
    rows.append(cur)
    rows = [r.replace("&", "").strip() for r in rows]
    rows = [r for r in rows if visible_tokens(r)]
    return rows if len(rows) >= 2 else None


def split_lines(ink_rows: list[bool], n: int, ratio: float = 1.5) -> list[list[int]] | None:
    """Fasce [y0, y1) di n righe di testo da un profilo di righe di pixel con inchiostro: si taglia nei n-1 spazi
    bianchi più alti, solo se il più basso dei tagli è alto almeno `ratio` volte il più alto degli altri spazi
    (quelli fra numeratore e linea di frazione, fra apici e base) e nessuna riga è più bassa del 30% della più alta.
    Altrimenti None: il numero di righe del LaTeX dà il numero di tagli, l'immagine deve confermarlo."""
    ys = [i for i, v in enumerate(ink_rows) if v]
    if not ys or n < 1:
        return None
    top, bottom = ys[0], ys[-1] + 1
    gaps, y = [], top
    while y < bottom:
        if ink_rows[y]:
            y += 1
            continue
        s0 = y
        while y < bottom and not ink_rows[y]:
            y += 1
        gaps.append((y - s0, s0, y))
    if len(gaps) < n - 1:
        return None
    gaps.sort(reverse=True)
    chosen, rest = gaps[: n - 1], gaps[n - 1:]
    if chosen and rest and chosen[-1][0] < ratio * rest[0][0]:
        return None
    bands, start = [], top
    for _, s0, e0 in sorted(chosen, key=lambda g: g[1]):
        bands.append([start, s0])
        start = e0
    bands.append([start, bottom])
    heights = [b - a for a, b in bands]
    return bands if min(heights) >= 0.3 * max(heights) else None
