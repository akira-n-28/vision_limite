"""Audit automatico dei dati STEM di MMFineReason: campionamento, affermazioni delle descrizioni, controlli deterministici,
calibrazione degli agenti, risoluzione delle divergenze e intervalli. Solo libreria standard (tests/test_stem_audit.py).

Protocollo (richieste dell'utente del 2026-10-19; scripts/stem_audit_export.py e scripts/stem_audit_round.py):
- controlli deterministici prima degli agenti: integrità e risoluzione delle immagini, duplicati, metadati (scelte
  multiple), contaminazione; la risposta di riferimento si verifica con codice quando l'agente ne scrive il calcolo;
- ogni problema ha due compiti, ciascuno dato a due agenti indipendenti alla cieca: S (risolvere dalla sola immagine
  e domanda, senza la descrizione di Qwen3-VL; leggibilità; coerenza) e D (ogni affermazione della descrizione divisa
  in fatti, ciascuno visibile o deduzione, verificato / falso / non verificabile dall'immagine);
- risposta di riferimento confermata dagli agenti se entrambi i solutori la ritrovano da soli; altrimenti due
  revisioni con la risposta in vista (G1, G2) e, se discordano, un arbitro;
- altri campi: terzo controllo alla cieca se A e B discordano, poi un arbitro (R1) che legge le motivazioni, e un
  secondo (R2) se R1 non conferma la maggioranza; al più cinque giudizi per campo, poi quarantena. Il 10% dei
  problemi approvati riceve comunque un controllo completo in più;
- elementi di calibrazione con verità nota (MAVIS, diagrammi sintetici) o con errori introdotti apposta, mescolati
  alla cieca nei lotti: misurano quanti errori gli agenti trovano e quanti ne perdono.
"""
from __future__ import annotations

import ast
import math
import random
import re

# ---------------------------------------------------------------- affermazioni della descrizione

_IMAGE_TITLE = re.compile(r"^\s*\**\s*#{1,6}\s*Image\b.*$", re.IGNORECASE)
_HEADER = re.compile(r"^\s*\*\*\s*([^*:]{2,60}?)\s*:?\s*\*\*\s*:?\s*")
_BULLET = re.compile(r"^\s*(?:[*\-\u2022]|\d+[.)])\s+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z*`$(\"'])|;\s+")
SECTIONS = {
    "scene summary": "summary", "explicit component breakdown": "components",
    "interactions and relationships": "relations", "implicit and inferred properties": "inferred",
    "inferred properties": "inferred", "identified ambiguities": "ambiguities", "question context": "context",
    "problem context": "context", "answer inference": "context", "correct inference": "context",
    "answer choices": "context",
}
VISIBLE_SECTIONS = {"summary", "components", "relations"}
_EMPTY = re.compile(r"^\W*(?:none|n/?a|no ambiguities?(?: identified)?)\W*$", re.IGNORECASE)


def claim_units(caption: str) -> list[dict]:
    """Affermazioni numerate di una descrizione: una per frase (o parte separata da punto e virgola), con la sezione
    di provenienza. I titoli in grassetto di sezioni note cambiano la sezione; quelli dei punti di elenco
    ("**Asse x:** ...") restano nel testo. Gli agenti dividono poi ogni affermazione nei suoi fatti."""
    units, section = [], "summary"
    for line in caption.splitlines():
        if not line.strip() or _IMAGE_TITLE.match(line):
            continue
        m = _HEADER.match(line)
        if m and line.strip().startswith("**") and m.group(1).strip().lower() in SECTIONS:
            section = SECTIONS[m.group(1).strip().lower()]
            line = line[m.end():]
        text = _BULLET.sub("", line).strip()
        if not text:
            continue
        for s in _SENT.split(text):
            s = s.strip()
            if len(re.sub(r"\W", "", s)) < 3 or _EMPTY.match(s):
                continue
            units.append({"cid": f"c{len(units) + 1:02d}", "section": section,
                          "visible": section in VISIBLE_SECTIONS, "text": s})
    return units


# ---------------------------------------------------------------- metadati: scelte multiple

_LETTER = re.compile(r"^\(?([A-H])\)?[.:]?$")


def answer_letter(answer: str) -> str | None:
    """Lettera di una risposta a scelta multipla ("B", "(B)", "\\text{B}"), o None."""
    a = re.sub(r"\\text\{([^}]*)\}|\\mathrm\{([^}]*)\}", lambda m: m.group(1) or m.group(2), answer or "").strip()
    a = re.sub(r"^\\boxed\{(.*)\}$", r"\1", a).strip()
    m = _LETTER.match(a)
    return m.group(1) if m else None


def mc_options(question: str) -> list[str]:
    """Lettere delle opzioni elencate nella domanda (A, B, ... consecutive da A), [] se non è a scelta multipla."""
    found = []
    for letter in "ABCDEFGH":
        if re.search(rf"(?:^|[\s(\[（])\(?{letter}[.):：、．]\s*\S", question, re.MULTILINE):
            found.append(letter)
        else:
            break
    return found if len(found) >= 2 else []


def option_text(question: str, letter: str) -> str | None:
    """Testo dell'opzione `letter` elencata nella domanda ("B. 12 cm" -> "12 cm"), o None."""
    m = re.search(rf"(?:^|[\s(\[（])\(?{letter}[.):：、．]\s*(.+?)(?=\s\(?[A-H][.):：、．]\s|\n|$)", question, re.MULTILINE)
    return m.group(1).strip() if m else None


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9π]+", " ", text.lower()).strip()


def match_answer(own: str, gold: str, question: str, base) -> bool:
    """La risposta di un solutore coincide con quella di riferimento? `base` è il confronto matematico (f2.answers_match);
    in più, con codice: lettera con o senza maiuscola e con il contenuto dopo ("c (cornea)" per "C"), contenuto
    dell'opzione al posto della lettera, primo numero con l'unità ("14 mi/h" per "14"), pi greco scritto in testo."""
    own, gold = str(own).strip(), str(gold).strip()
    if base(own, gold):
        return True
    letter = answer_letter(gold)
    lead = re.match(r"^\(?([A-Ha-h])\)?(?=$|[\s.):,;])", own)
    if letter:
        if lead and lead.group(1).upper() == letter:
            return True
        content = option_text(question, letter)
        if content and not lead:
            return base(own, content) or _norm_text(own) == _norm_text(content)
        return False
    if re.fullmatch(r"-?\d+(?:\.\d+)?", gold):
        nums = re.findall(r"-?\d+(?:\.\d+)?", own)
        if nums and abs(float(nums[0]) - float(gold)) < 1e-9 and len(nums) == 1:
            return True
    fix = lambda t: re.sub(r"(?<![\\a-zA-Z])pi\b", r"\\pi", t.replace("π", r"\pi")).replace("$", "")  # noqa: E731
    return base(fix(own), fix(gold))


def norm_question(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").replace("<image>", " ")).strip().lower()


# ---------------------------------------------------------------- calibrazione: errori introdotti apposta

_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.]*\d)")
SWAPS = [("left", "right"), ("above", "below"), ("horizontal", "vertical"), ("increasing", "decreasing"),
         ("upward", "downward"), ("clockwise", "counterclockwise"), ("inside", "outside"), ("solid", "dashed")]


def mutate_text(text: str, rng: random.Random) -> tuple[str, str] | None:
    """Un errore introdotto in una frase: un numero cambiato (in modo che resti plausibile) o una parola di direzione
    sostituita con il suo opposto. Ritorna (nuovo testo, descrizione del cambio) o None se non c'è niente da cambiare."""
    numbers = [m for m in _NUMBER.finditer(text) if m.group(1) not in ("0", "1")]
    words = [(m, b) for a, b in SWAPS for m in re.finditer(rf"\b{a}\b", text, re.IGNORECASE)] + \
            [(m, a) for a, b in SWAPS for m in re.finditer(rf"\b{b}\b", text, re.IGNORECASE)]
    words = [(m, r) for m, r in words if not (r == "clockwise" and m.group(0).lower() == "counterclockwise")]
    if not numbers and not words:
        return None
    if numbers and (not words or rng.random() < 0.7):
        m = rng.choice(numbers)
        v = m.group(1)
        if "." in v:
            new = f"{float(v) + rng.choice([0.5, 1.5, 2.5]):.{len(v.split('.')[1])}f}"
        else:
            iv = int(v)
            new = str(iv + rng.choice([2, 3, 4, 5]) if iv < 20 or rng.random() < 0.5 else iv * 2)
        return text[:m.start(1)] + new + text[m.end(1):], f"{v} -> {new}"
    m, repl = rng.choice(words)
    repl = repl.capitalize() if m.group(0)[0].isupper() else repl
    return text[:m.start()] + repl + text[m.end():], f"{m.group(0)} -> {repl}"


def wrong_answer(answer: str, options: list[str], rng: random.Random) -> str | None:
    """Una risposta di riferimento sbagliata apposta: un'altra lettera se a scelta multipla, un numero cambiato
    se numerica; None se non si sa cambiarla in modo plausibile."""
    letter = answer_letter(answer)
    if letter:
        others = [o for o in (options or list("ABCD")) if o != letter]
        return rng.choice(others) if others else None
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(.*)", answer or "")
    if not m:
        return None
    v, unit = m.group(1), m.group(2)
    if "." in v:
        new = f"{float(v) * rng.choice([2, 0.5, 1.5]):.{len(v.split('.')[1])}f}"
    else:
        iv = int(v)
        new = str(iv + rng.choice([1, 2, 3]) if abs(iv) < 10 else iv * 2)
    return (new + (" " + unit if unit else "")).strip()


# ---------------------------------------------------------------- verifica con codice

ALLOWED_IMPORTS = {"math", "sympy", "fractions", "itertools"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals",
                   "vars", "input", "breakpoint", "help", "exit", "quit", "memoryview", "os", "sys", "subprocess",
                   "socket", "shutil", "pathlib", "builtins", "importlib"}
SAFE_LITERALS = {0, 1, 2, 3, 4, 10, 100, 180, 360, 90, 60, 45, 30, 0.5}


def program_problems(code: str) -> list[str]:
    """Motivi per cui un programma dell'agente non si esegue: sintassi, import non ammessi, nomi pericolosi,
    attributi privati, nessuna variabile `result`."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"sintassi: {exc.msg}"]
    out, assigns = [], False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            out += [f"import {m}" for m in mods if m.split(".")[0] not in ALLOWED_IMPORTS]
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            out.append(f"nome {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            out.append(f"attributo {node.attr}")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assigns |= any(isinstance(t, ast.Name) and t.id == "result" for t in targets)
    if not assigns:
        out.append("manca result")
    return sorted(set(out))


def undeclared_literals(code: str) -> list[float]:
    """Numeri scritti nel codice fuori da un piccolo insieme di costanti: valori letti ma non dichiarati fra gli input."""
    out = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if node.value not in SAFE_LITERALS:
                out.append(node.value)
    return out


def numbers_in(text: str) -> set[str]:
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+(?:\.\d+)?", text or "")}


def input_in_question(value: str, question: str) -> bool:
    """Un input dichiarato "dalla domanda" è credibile se tutti i suoi numeri compaiono nel testo della domanda."""
    nums = numbers_in(str(value))
    return bool(nums) and nums <= numbers_in(question)


# ---------------------------------------------------------------- campionamento stratificato

def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """n campioni divisi in parti uguali fra le sorgenti, senza superare la dimensione di nessuna: quello che una
    sorgente piccola non usa va alle altre (le più grandi per prime a parità di resto)."""
    alloc = {s: 0 for s in sizes}
    left = min(n, sum(sizes.values()))
    open_ = sorted((s for s in sizes if sizes[s] > 0), key=lambda s: (sizes[s], s))
    while left > 0 and open_:
        share = max(1, left // len(open_))
        for s in sorted(open_, key=lambda s: (-sizes[s], s)):
            if left == 0:
                break
            take = min(share, sizes[s] - alloc[s], left)
            alloc[s] += take
            left -= take
        open_ = [s for s in open_ if alloc[s] < sizes[s]]
    return alloc


# ---------------------------------------------------------------- intervalli di confidenza

def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
                        for i in range(0, k + 1)))


def _bisect(f, target: float) -> float:
    """p in [0, 1] con f(p) = target, per f decrescente."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Intervallo esatto bilaterale per k successi su n prove."""
    if n <= 0:
        return 0.0, 1.0
    a = (1 - conf) / 2
    lower = 0.0 if k == 0 else _bisect(lambda p: _binom_cdf(k - 1, n, p), 1 - a)
    upper = 1.0 if k >= n else _bisect(lambda p: _binom_cdf(k, n, p), a)
    return lower, upper


def clustered_rate(clusters: list[tuple[int, int]], conf: float = 0.95, min_errors: int = 5) -> dict:
    """Tasso di errori di affermazioni raggruppate per problema (errori, affermazioni), con un intervallo che tiene
    conto della correlazione dentro il problema: Clopper-Pearson su una dimensione efficace n / deff. L'effetto del
    disegno deff si stima dalla varianza fra problemi (stimatore a rapporto) se gli errori sono almeno min_errors;
    con meno errori la stima non regge e si usa il caso peggiore, deff = affermazioni medie per problema (come se
    ogni problema fosse una sola osservazione)."""
    clusters = [(e, t) for e, t in clusters if t > 0]
    n, k = sum(t for _, t in clusters), sum(e for e, _ in clusters)
    m = len(clusters)
    if n == 0:
        return {"errors": 0, "total": 0, "rate": None, "ci": (0.0, 1.0), "deff": None, "deff_rule": None}
    p = k / n
    if k >= min_errors and m > 1 and 0 < p < 1:
        var_cluster = m / (m - 1) * sum((e - p * t) ** 2 for e, t in clusters) / n ** 2
        deff, rule = max(1.0, var_cluster / (p * (1 - p) / n)), "stimato"
    else:
        deff, rule = max(1.0, n / m), "caso peggiore"
    n_eff = max(1, math.floor(n / deff))
    k_eff = min(n_eff, math.ceil(k / deff))
    lo, hi = clopper_pearson(k_eff, n_eff, conf)
    return {"errors": k, "total": n, "rate": p, "ci": (min(lo, p), max(hi, p)), "deff": round(deff, 2), "deff_rule": rule}


def corrected_upper(upper: float, sensitivity_lower: float | None) -> float:
    """Limite superiore del tasso vero quando gli agenti trovano solo una parte degli errori: il tasso osservato
    sottostima quello vero di un fattore pari alla sensibilità; si divide per il suo limite inferiore (prudente)."""
    if not sensitivity_lower or sensitivity_lower <= 0:
        return 1.0
    return min(1.0, upper / sensitivity_lower)


# ---------------------------------------------------------------- risoluzione delle divergenze

MAX_JUDGMENTS = 5
UNSURE = {"uncertain", "unclear", None}


def legibility_class(v: str | None) -> str | None:
    return {"ok": "readable", "partial": "readable", "illegible": "illegible"}.get(v)


def coherence_class(v: str | None) -> str | None:
    return {"ok": "ok", "incoherent": "problem", "corrupt": "problem"}.get(v)


def unit_classes(facts: list[dict]) -> dict[str, str]:
    """Classi di decisione di un'affermazione dai fatti etichettati da un agente: per i fatti visibili e per le
    deduzioni separatamente, E se almeno un fatto è falso, K altrimenti (verificato o non verificabile non sono
    errori)."""
    out = {"vis": "K", "ded": "K"}
    for f in facts or []:
        if not isinstance(f, dict):
            continue
        kind = "ded" if str(f.get("kind", "")).lower().startswith("ded") else "vis"
        if str(f.get("label", "")).lower() in ("false", "falso"):
            out[kind] = "E"
    return out


def resolve_field(blind: list, arbiters: list) -> dict:
    """Stato di un campo dai giudizi alla cieca (classi, nell'ordine A, B, C) e degli arbitri (R1, R2).

    - due o più giudizi alla cieca uguali e non incerti, senza nessun altro diverso: risolto (consenso);
    - altrimenti serve un terzo giudizio alla cieca, poi l'arbitro R1: risolto se R1 conferma, motivando, la
      maggioranza dei giudizi alla cieca; altrimenti R2, e risolto solo se R2 dà la stessa decisione di R1;
    - incerto ovunque o divergenza che resta dopo cinque giudizi: quarantena.
    Ritorna {"status": "resolved"|"need_blind"|"need_arbiter"|"quarantine", "value", "how"}."""
    sure = [v for v in blind if v not in UNSURE]
    if len(blind) >= 2 and len(sure) == len(blind) and len(set(sure)) == 1 and not arbiters:
        return {"status": "resolved", "value": sure[0], "how": "consenso" if len(blind) == 2 else "consenso+controllo"}
    if len(blind) < 3:
        return {"status": "need_blind", "value": None, "how": None}
    majority = next((v for v in set(sure) if sure.count(v) >= 2), None)
    if not arbiters:
        return {"status": "need_arbiter", "value": None, "how": None}
    v1 = arbiters[0]
    if v1 not in UNSURE and v1 == majority:
        return {"status": "resolved", "value": v1, "how": "maggioranza+arbitro"}
    if len(arbiters) < 2:
        if len(blind) + 1 < MAX_JUDGMENTS:
            return {"status": "need_arbiter", "value": None, "how": None}
        return {"status": "quarantine", "value": None, "how": "limite di giudizi"}
    if v1 not in UNSURE and arbiters[1] == v1:
        return {"status": "resolved", "value": v1, "how": "due arbitri"}
    return {"status": "quarantine", "value": None, "how": "arbitri discordi o incerti"}


def resolve_gold(solves: list[str], reviews: list[str], force_review: bool = False) -> dict:
    """Stato della risposta di riferimento. solves: esito delle soluzioni alla cieca ("match", "mismatch",
    "unsolved"); reviews: verdetti con la risposta in vista ("correct", "wrong", "uncertain"), nell'ordine G1, G2,
    arbitro. Confermata senza revisioni se almeno due solutori la ritrovano da soli e nessuno trova altro (a meno
    di force_review, es. un duplicato con risposta diversa); con le revisioni, risolta quando due verdetti non
    incerti coincidono."""
    if not reviews:
        if not force_review and solves.count("match") >= 2 and "mismatch" not in solves and "unsolved" not in solves:
            return {"status": "resolved", "value": "correct", "how": "ritrovata alla cieca"}
        return {"status": "need_review", "value": None, "how": None}
    if len(reviews) < 2:
        return {"status": "need_review", "value": None, "how": None}
    sure = [r for r in reviews if r not in UNSURE]
    for v in ("correct", "wrong"):
        if sure.count(v) >= 2:
            return {"status": "resolved", "value": v, "how": "revisioni concordi" if len(reviews) == 2 else "revisioni+arbitro"}
    if len(reviews) == 2 and len(solves) + 2 < MAX_JUDGMENTS:
        return {"status": "need_arbiter", "value": None, "how": None}
    return {"status": "quarantine", "value": None, "how": "revisioni discordi o incerte"}


def recheck_sample(keys_by_group: dict[str, list[str]], frac: float, seed: int) -> set[str]:
    """Il frac (arrotondato per eccesso, almeno uno) dei problemi approvati di ogni gruppo, scelto con il seed."""
    out = set()
    for group, keys in sorted(keys_by_group.items()):
        keys = sorted(keys)
        if keys:
            out.update(random.Random(f"{seed}/{group}").sample(keys, max(1, math.ceil(frac * len(keys)))))
    return out
