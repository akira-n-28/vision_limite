"""Ponte testuale di F2: prompt per Violetto, estrazione e verifica delle risposte.

Il piano: per ogni problema visivo con risposta nota, Violetto riceve una descrizione densa della figura e la
domanda, campiona più soluzioni e si tengono quelle con la risposta giusta. L'esempio di training è poi
immagine + domanda -> soluzione di Violetto, senza la descrizione. Le descrizioni vengono da MMFineReason
(Qwen3-VL-235B), invece che da un teacher servito da noi.

Formato: lo stesso istruzione del valutatore ufficiale per MATH-500 (limite-evals, suites.MATH500_INSTRUCTION),
aggiunta dopo la domanda. Violetto risponde con `<think>...</think>` seguito dalla risposta con `\\boxed{}`.
"""
from __future__ import annotations

import re

INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
# Prima prova (job 29476): con "Here is a detailed description of the figure" il 97% delle soluzioni corrette
# parlava della descrizione ("The user describes a figure..."). Ora la descrizione entra nel testo del problema
# come figura, senza la parola "description".
BRIDGE_HEADER = "Figure:\n"
# Solo per registrarlo nei piani di generazione: le graffe di \boxed{} vieterebbero str.format (test_f2).
BRIDGE_TEMPLATE = BRIDGE_HEADER + "<caption>\n\n<question>" + INSTRUCTION
# Riferimenti al testo del ponte: nelle soluzioni corrette si riscrivono come riferimenti alla figura, che il
# modello in training vede davvero (riscrittura di poche parole, il ragionamento resta quello di Violetto).
_MENTIONS_BRIDGE = re.compile(r"\b(description|described|describes|caption)\b", re.IGNORECASE)
_REWRITES = [
    # aperture frequenti (job 29500): prima quelle mirate, poi le sostituzioni parola per parola
    (re.compile(r"\b(?:has |had )?(?:provided|provides|gave|gives|given)(?: us)? (?:a|an|the) "
                r"(?:detailed |textual |text |long )?description of\b", re.IGNORECASE), "shows"),
    (re.compile(r"\b(There's|There is) (?:a|an) (?:detailed |textual )?description of\b", re.IGNORECASE), r"\1"),
    (re.compile(r"\b(?:a|the) problem description\b", re.IGNORECASE), "the problem"),
    (re.compile(r"\b(?:described|given|provided) (?:in|by) (?:the )?text\b", re.IGNORECASE), "shown"),
    (re.compile(r"\btextual description\b", re.IGNORECASE), "figure"),
    (re.compile(r"\b(figure|image|diagram) (description|caption)\b", re.IGNORECASE), r"\1"),
    (re.compile(r"\bdescriptions\b", re.IGNORECASE), "figures"),
    (re.compile(r"\b(description|caption)\b", re.IGNORECASE), "figure"),
    (re.compile(r"\bdescribed\b", re.IGNORECASE), "shown"),
    (re.compile(r"\bdescribes\b", re.IGNORECASE), "shows"),
    (re.compile(r"\bdescribing\b", re.IGNORECASE), "showing"),
    (re.compile(r"\bdescribe\b", re.IGNORECASE), "show"),
]
# Una soluzione che dichiara di non vedere l'immagine contraddice l'esempio di training (che la contiene).
_NO_IMAGE = re.compile(r"(not provided|isn't provided|is not shown|can(?:no|')t see|cannot view|no image|"
                       r"without (?:the|an) (?:image|figure|diagram)|image is missing|figure is missing)", re.IGNORECASE)
_LETTER = re.compile(r"^\(?([A-H])\)?[.:]?$")
# Violetto non sa il cinese: caratteri CJK, kana o hangul in una soluzione vengono in genere copiati dalla descrizione
# (trascrizioni delle scritte nelle figure) e insegnerebbero a produrre token che il modello non sa usare. Il 4.4%
# delle soluzioni corrette su domande in inglese (job 29592).
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


def bridge_prompt(question: str, caption: str) -> str:
    return BRIDGE_HEADER + caption.strip() + "\n\n" + question.strip() + INSTRUCTION


def training_question(question: str) -> str:
    """Testo della domanda negli esempi di F2 (con l'immagine al posto della descrizione)."""
    return question.strip() + INSTRUCTION


def last_boxed(text: str) -> str | None:
    """Contenuto dell'ultimo \\boxed{...}, con le graffe annidate."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j].strip()
    return None


def split_completion(text: str) -> tuple[str | None, str]:
    """(ragionamento, risposta finale) di una generazione di Violetto; ragionamento None se manca </think>."""
    if "</think>" not in text:
        return None, text
    head, _, tail = text.rpartition("</think>")
    return head.removeprefix("<think>"), tail


def _letter(answer: str) -> str | None:
    a = re.sub(r"\\text\{([^}]*)\}|\\mathrm\{([^}]*)\}", lambda m: m.group(1) or m.group(2), answer).strip()
    m = _LETTER.match(a)
    return m.group(1) if m else None


def answers_match(prediction: str, gold: str, timeout: int = 5) -> bool:
    """Lettere per le scelte multiple; altrimenti math_verify (come il valutatore), poi confronto di stringhe."""
    gold_letter, pred_letter = _letter(gold), _letter(prediction)
    if gold_letter is not None:
        return pred_letter == gold_letter
    norm = lambda s: re.sub(r"[\s$]", "", s).rstrip(".")  # noqa: E731
    if norm(prediction) == norm(gold):
        return True
    try:
        from math_verify import parse, verify

        return bool(verify(parse("\\boxed{" + gold + "}", parsing_timeout=timeout),
                           parse("\\boxed{" + prediction + "}", parsing_timeout=timeout), timeout_seconds=timeout))
    except Exception:  # noqa: BLE001 - parsing fallito o timeout: non corretta
        return False


def _same_case(word: str, repl: str) -> str:
    return repl[:1].upper() + repl[1:] if word[:1].isupper() else repl


def clean_solution(text: str) -> tuple[str, str | None]:
    """(testo riscritto, motivo di scarto o None): riferimenti al ponte -> riferimenti alla figura."""
    if _NO_IMAGE.search(text):
        return text, "no_image"
    if CJK.search(text):
        return text, "non_english"
    out = text
    for pattern, repl in _REWRITES:
        out = pattern.sub(lambda m: _same_case(m.group(0), m.expand(repl)), out)
    if _MENTIONS_BRIDGE.search(out):
        return out, "mentions_bridge"
    return out, None


def grade(text: str, gold: str, finish_reason: str) -> dict:
    """Esito di una generazione: formato, risposta estratta, correttezza; per le corrette, soluzione ripulita
    dai riferimenti al ponte (clean_solution) e motivo di scarto."""
    reasoning, final = split_completion(text)
    answer = last_boxed(final) if reasoning is not None else None
    result = {"stopped": finish_reason == "stop", "has_think": reasoning is not None, "answer": answer,
              "mentions_bridge": bool(_MENTIONS_BRIDGE.search(text))}
    result["correct"] = bool(result["stopped"] and answer is not None and answers_match(answer, gold))
    result["cleaned"], result["reject"] = clean_solution(text) if result["correct"] else (None, None)
    result["usable"] = result["correct"] and result["reject"] is None
    return result
