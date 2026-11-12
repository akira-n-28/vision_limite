#!/usr/bin/env python3
"""Sonda CPU del contratto di risposta (D3; S3 §2; docs/F0_IMPLEMENTATION_PLAN.md punto 1): proprietà meccaniche della
tokenizzazione di tre alternative e della baseline matematica, col tokenizer e il template dello snapshot di Violetto.

Diagnostica isolata: non modifica il processor operativo, il tokenizer salvato, il template dello snapshot, i loader o
le configurazioni. Alternative, regole del contratto e template della sonda (`limite-vl-sonda/1`, la copia del template
dello snapshot che ammette anche il system VL) esistono solo qui; il processor operativo si riusa per sottoclasse,
cambiando solo il template passato a apply_chat_template. Solo in un job SLURM CPU: nessun peso, nessun dataset.

Alternative (S3 §2.2):
- A, storico: template dello snapshot (system canonico); risposta diretta dopo il prefill F1 `<think></think>` senza
  label; risposta ragionata `<think>…</think>` con label (formato del replay e di F2 storici);
- B: system neutro `You are a helpful assistant.`; risposta diretta `<think></think>` + risposta, tutto con label;
  ragionata `<think>…</think>` + risposta;
- B-F1: i token di B, col `<think></think>` della risposta diretta dato come prefill dichiarato senza label (decoder
  congelato); un solo turno, nessun target ragionato;
- C: system neutro; risposta diretta senza tag; ragionata `<think>…</think>` + risposta;
- M, baseline matematica: template dello snapshot, istruzione di MATH-500 del valutatore ufficiale nel turno utente,
  sempre `<think>…</think>` e `\\boxed{}`.

Per ogni caso scritto a mano e ogni alternativa: messaggi, template, token di prompt, prefill e target, intervalli con e
senza label, EOS e chiusure di turno, conteggi. Gli esiti attesi sono stringhe scritte qui (i pezzi del formato
concatenati, non il template) e id espliciti dei tag e dei terminatori; poi confronti fra alternative, identità della
baseline canonica col template dello snapshot, rifiuti delle combinazioni incoerenti, comportamento del processor
operativo e fatti misurati. Dimostra proprietà del formato, non quale alternativa il modello seguirà meglio.

    srun $RUN_PYTHON -B scripts/contract_probe.py --snapshot DIR --out DIR --docs PREFISSO
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.processing import (EOS_ID, IGNORE, IM_END_ID, IMAGE_PAD_ID, IMAGE_PLACEHOLDER,  # noqa: E402
                                  LimiteVLProcessor)
from limite_vl.provenance import code_identity, refuse_existing, sha256_file, sha256_text, write_new_json  # noqa: E402
from limite_vl.replay import MATH500_INSTRUCTION  # noqa: E402

SCHEMA = "contract-probe/1"
EOT = "<|endoftext|>"
THINK, END_THINK = "<think>", "</think>"
EMPTY = THINK + END_THINK
THINK_ID, END_THINK_ID, IM_START_ID = 151665, 151666, 151644
SYS_CANON = "You are a helpful assistant.\nPlease reason step by step, and put your final answer within \\boxed{}."
SYS_VL = "You are a helpful assistant."
OPEN = "<|im_start|>assistant\n"
GRID = (16, 16)  # segnaposto visivo della sonda: 64 token immagine; il numero reale dipende dalla risoluzione
SNAPSHOT_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                  "generation_config.json")
SUITES = ROOT / "third_party/limite-violetto/limite-evals/src/limite_evals/suites.py"

# Template della sonda: quello dello snapshot, con una sola differenza. Il primo messaggio può essere un system canonico
# o VL, che sostituisce il canonico nell'intestazione; senza system resta il canonico. Un system in un'altra posizione è
# rifiutato (lo snapshot accetta un canonico ovunque e non lo scrive). Il resto, reasoning_content compreso, è invariato.
PROBE_TEMPLATE = r"""{%- set canonical_prompt = 'You are a helpful assistant.\nPlease reason step by step, and put your final answer within \\boxed{}.' %}
{%- set vl_prompt = 'You are a helpful assistant.' %}
{%- if tools %}
    {{- raise_exception('Limite-VL probe supports text-only chat; tools are not supported.') }}
{%- endif %}
{%- set ns = namespace(system_prompt=canonical_prompt, first=0) %}
{%- if messages and messages[0].role == 'system' %}
    {%- if messages[0].content == canonical_prompt or messages[0].content == vl_prompt %}
        {%- set ns.system_prompt = messages[0].content %}
        {%- set ns.first = 1 %}
    {%- else %}
        {{- raise_exception('Limite-VL probe admits only the canonical and the VL system prompts.') }}
    {%- endif %}
{%- endif %}
{{- '<|endoftext|><|im_start|>system\n' + ns.system_prompt + '<|im_end|>\n' }}
{%- for message in messages[ns.first:] %}
    {%- if (message.tool_calls is defined and message.tool_calls is not none) or (message.function_call is defined and message.function_call is not none) %}
        {{- raise_exception('Limite-VL probe supports text-only chat; tool calls are not supported.') }}
    {%- endif %}
    {%- if message.content is not string %}
        {{- raise_exception('Limite-VL probe supports text-only chat; message content must be a string.') }}
    {%- endif %}
    {%- if message.role == 'system' %}
        {{- raise_exception('Limite-VL probe admits a system message only as the first message.') }}
    {%- elif message.role == 'user' %}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == 'assistant' %}
        {{- '<|im_start|>assistant\n' }}
        {%- if message.reasoning_content is string and message.reasoning_content %}
            {{- '<think>\n' + message.reasoning_content + '\n</think>\n\n' }}
        {%- endif %}
        {{- message.content + '<|im_end|>\n' }}
    {%- else %}
        {{- raise_exception('Limite-VL probe supports only user, assistant and one leading system message.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""

ALTERNATIVES = {
    "A": {"system": None, "template": "snapshot", "diretta": "prefill", "ragionata": True, "un_turno": False,
          "fase": "F1 storico (decoder congelato); il ragionato è il formato del replay e di F2 storici",
          "descrizione": "template dello snapshot (system canonico); diretta: prefill <think></think> senza label, poi "
                         "risposta ed EOS con label; ragionata: <think>…</think> e risposta con label"},
    "B": {"system": SYS_VL, "template": "sonda", "diretta": "think_vuoto", "ragionata": True, "un_turno": False,
          "fase": "uso finale, decoder addestrabile",
          "descrizione": "system VL; diretta: <think></think> generato e con label, poi risposta ed EOS; ragionata: "
                         "<think>…</think> e risposta con label"},
    "B-F1": {"system": SYS_VL, "template": "sonda", "diretta": "prefill", "ragionata": False, "un_turno": True,
             "fase": "F1, decoder congelato",
             "descrizione": "token di B; il <think></think> della diretta è un prefill dichiarato senza label; un "
                            "turno, nessun target ragionato"},
    "C": {"system": SYS_VL, "template": "sonda", "diretta": "senza_tag", "ragionata": True, "un_turno": False,
          "fase": "uso finale, decoder addestrabile",
          "descrizione": "system VL; diretta: risposta ed EOS senza tag; ragionata: <think>…</think> e risposta"},
    "M": {"system": None, "template": "snapshot", "diretta": None, "ragionata": True, "un_turno": True,
          "istruzione": MATH500_INSTRUCTION, "fase": "baseline matematica e replay (modalità canonica)",
          "descrizione": "template dello snapshot; istruzione di MATH-500 dopo il problema; sempre <think>…</think> e "
                         "\\boxed{} nella risposta finale"},
}

LATEX = r"\frac{a+b}{2}\geq\sqrt{ab}"
CAPTION = "A red bicycle leaning against a white brick wall."
NUM_Q = "Answer the question with a short answer.\nWhat is the length of side AB?"
MC_Q = ("Which shape has the largest area?\nA. the square\nB. the circle\nC. the triangle\n"
        "Answer with the letter of the correct option.")
PROBLEM = "A rectangle has a perimeter of 20 cm and one side of length 3 cm. What is its area?"
REASON = "Half the perimeter is 10 cm, so the other side is 10 - 3 = 7 cm. The area is 3 \\times 7 = 21."
FINAL = "The other side is 7 cm, so the area is \\boxed{21} square centimetres."
D_Q1, D_A1 = "What shape is shown in the image?", "A right triangle."
D_Q2 = "Its legs are 3 and 4. What is its area?"
D_R2 = "The area of a right triangle is half the product of its legs: 3 \\times 4 / 2 = 6."
D_F2 = "The area is \\boxed{6}."


def direct(user: str, answer: str, image: bool = True) -> list[dict]:
    return [{"utente": user, "immagine": image, "ragionamento": None, "risposta": answer}]


CASES = {
    "latex": {"descrizione": "trascrizione LaTeX, con immagine",
              "turni": direct("Transcribe the formula in the image into LaTeX.", LATEX)},
    "didascalia": {"descrizione": "didascalia breve, con immagine",
                   "turni": direct("Describe this image briefly.", CAPTION)},
    "numerica": {"descrizione": "risposta numerica, con immagine", "turni": direct(NUM_Q, "12")},
    "scelta_multipla": {"descrizione": "scelta multipla, con immagine", "turni": direct(MC_Q, "B")},
    "spiegazione": {"descrizione": "problema con spiegazione e risposta finale, solo testo",
                    "turni": [{"utente": PROBLEM, "immagine": False, "ragionamento": REASON, "risposta": FINAL}]},
    "dialogo": {"descrizione": "conversazione con un turno precedente dell'assistente (diretto), poi un turno ragionato",
                "turni": [{"utente": D_Q1, "immagine": True, "ragionamento": None, "risposta": D_A1},
                          {"utente": D_Q2, "immagine": False, "ragionamento": D_R2, "risposta": D_F2}]},
}

# Esiti attesi, scritti a mano: per alternativa, i turni precedenti dell'assistente come stanno nella cronologia (con
# label), il prefill dell'ultimo turno (senza label) e il target (con label). Le alternative assenti non si applicano.
EXPECTED = {
    "latex": {"A": {"prefill": "<think></think>", "target": LATEX + "<|endoftext|>"},
              "B": {"prefill": "", "target": "<think></think>" + LATEX + "<|endoftext|>"},
              "B-F1": {"prefill": "<think></think>", "target": LATEX + "<|endoftext|>"},
              "C": {"prefill": "", "target": LATEX + "<|endoftext|>"}},
    "didascalia": {"A": {"prefill": "<think></think>", "target": CAPTION + "<|endoftext|>"},
                   "B": {"prefill": "", "target": "<think></think>" + CAPTION + "<|endoftext|>"},
                   "B-F1": {"prefill": "<think></think>", "target": CAPTION + "<|endoftext|>"},
                   "C": {"prefill": "", "target": CAPTION + "<|endoftext|>"}},
    "numerica": {"A": {"prefill": "<think></think>", "target": "12<|endoftext|>"},
                 "B": {"prefill": "", "target": "<think></think>12<|endoftext|>"},
                 "B-F1": {"prefill": "<think></think>", "target": "12<|endoftext|>"},
                 "C": {"prefill": "", "target": "12<|endoftext|>"}},
    "scelta_multipla": {"A": {"prefill": "<think></think>", "target": "B<|endoftext|>"},
                        "B": {"prefill": "", "target": "<think></think>B<|endoftext|>"},
                        "B-F1": {"prefill": "<think></think>", "target": "B<|endoftext|>"},
                        "C": {"prefill": "", "target": "B<|endoftext|>"}},
    "spiegazione": {k: {"prefill": "", "target": "<think>" + REASON + "</think>" + FINAL + "<|endoftext|>"}
                    for k in ("A", "B", "C", "M")},
    "dialogo": {"A": {"cronologia": ["<think></think>" + D_A1], "prefill": "",
                      "target": "<think>" + D_R2 + "</think>" + D_F2 + "<|endoftext|>"},
                "B": {"cronologia": ["<think></think>" + D_A1], "prefill": "",
                      "target": "<think>" + D_R2 + "</think>" + D_F2 + "<|endoftext|>"},
                "C": {"cronologia": [D_A1], "prefill": "",
                      "target": "<think>" + D_R2 + "</think>" + D_F2 + "<|endoftext|>"}},
}


class ContractError(ValueError):
    """Combinazione incoerente col contratto della sonda."""


def header(system: str) -> str:
    return EOT + "<|im_start|>system\n" + system + "<|im_end|>\n"


def user_block(text: str, image: bool) -> str:
    return "<|im_start|>user\n" + (IMAGE_PLACEHOLDER if image else "") + text + "<|im_end|>\n"


# ---------------------------------------------------------------- regole del contratto (solo nella sonda)

def tag_form(content: str) -> str:
    """`senza_tag` o `think` (un blocco in apertura, chiuso); altrimenti ContractError (tag malformati)."""
    n_open, n_close = content.count(THINK), content.count(END_THINK)
    if n_open == n_close == 0:
        return "senza_tag"
    if n_open == n_close == 1 and content.startswith(THINK) and content.index(END_THINK) > 0:
        return "think"
    raise ContractError(f"tag malformati: {n_open} <think>, {n_close} </think>, inizio {content[:12]!r}")


def check_system(alt: str, messages: list[dict]) -> None:
    first = messages[0] if messages and messages[0]["role"] == "system" else None
    want = ALTERNATIVES[alt]["system"]
    got = None if first is None else first["content"]
    if want is None and got not in (None, SYS_CANON):
        raise ContractError(f"{alt}: modalità canonica con un system diverso dal canonico")
    if want is not None and got != want:
        raise ContractError(f"{alt}: modalità VL senza il system VL ({got!r})")


def check_turn(alt: str, prefill: str, content: str, final: bool) -> str:
    """Forma di un turno dell'assistente (`diretta` o `ragionata`) o ContractError."""
    a = ALTERNATIVES[alt]
    if prefill:
        if not final:
            raise ContractError("prefill fuori dall'ultimo turno")
        if a["diretta"] != "prefill":
            raise ContractError(f"{alt}: prefill non ammesso")
        if prefill != EMPTY:
            raise ContractError(f"prefill {prefill!r}: ammesso solo <think></think>")
        if tag_form(content) != "senza_tag":
            raise ContractError("prefill <think></think> e tag nel contenuto: tag duplicati")
        return "diretta"
    if tag_form(content) == "senza_tag":
        if a["diretta"] != "senza_tag":
            raise ContractError(f"{alt}: risposta senza <think>")
        return "diretta"
    reasoning, final_text = content[len(THINK):content.index(END_THINK)], content[content.index(END_THINK) + 8:]
    if not reasoning:
        if a["diretta"] == "think_vuoto" or (a["diretta"] == "prefill" and not final):
            return "diretta"
        raise ContractError(f"{alt}: <think></think> vuoto con label non ammesso")
    if not a["ragionata"]:
        raise ContractError(f"{alt}: target ragionato non ammesso")
    if alt == "M" and "\\boxed{" not in final_text:
        raise ContractError("M: risposta finale senza \\boxed{}")
    return "ragionata"


def check_prompt(alt: str, prefill: str) -> None:
    if prefill and ALTERNATIVES[alt]["diretta"] != "prefill":
        raise ContractError(f"{alt}: prefill nel prompt d'inferenza della modalità d'uso finale")


def turn_text(alt: str, reasoning: str | None, answer: str, final: bool) -> tuple[str, str]:
    """(prefill, contenuto) di un turno dell'assistente secondo l'alternativa."""
    a = ALTERNATIVES[alt]
    for part in (answer, reasoning or ""):
        if THINK in part or END_THINK in part:
            raise ContractError("tag nel testo della risposta o del ragionamento: li scrive il contratto")
    if reasoning is None:
        if a["diretta"] is None:
            raise ContractError(f"{alt}: risposta diretta non ammessa, serve il ragionamento")
        if a["diretta"] == "prefill":
            return (EMPTY, answer) if final else ("", EMPTY + answer)
        return ("", EMPTY + answer) if a["diretta"] == "think_vuoto" else ("", answer)
    if not reasoning.strip():
        raise ContractError("ragionamento vuoto")
    return "", THINK + reasoning + END_THINK + answer


def compose(alt: str, turns: list[dict]) -> tuple[list[dict], list, str]:
    """Messaggi per il processor, immagini (segnaposto) e prefill del prompt d'inferenza dell'ultimo turno."""
    a = ALTERNATIVES[alt]
    if a["un_turno"] and len(turns) > 1:
        raise ContractError(f"{alt}: un solo turno")
    messages = [] if a["system"] is None else [{"role": "system", "content": a["system"]}]
    images, prefill = [], ""
    for i, t in enumerate(turns):
        final = i == len(turns) - 1
        text = t["utente"] + a.get("istruzione", "")
        if t["immagine"]:
            messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]})
            images.append((np.zeros((GRID[0] * GRID[1], 768), dtype=np.uint8), GRID))
        else:
            messages.append({"role": "user", "content": text})
        p, c = turn_text(alt, t["ragionamento"], t["risposta"], final)
        check_turn(alt, p, c, final)
        messages.append({"role": "assistant", "content": c, **({"prefill": p} if p else {})})
        if final:
            prefill = p
    check_system(alt, messages)
    check_prompt(alt, prefill)
    return messages, images, prefill


class TemplateProcessor(LimiteVLProcessor):
    """Il processor operativo con un template esplicito: esiste solo nella sonda."""

    def __init__(self, tokenizer, template: str):
        super().__init__(tokenizer)
        self.template = template

    def _ids(self, messages, add_generation_prompt=False):
        text = self.tokenizer.apply_chat_template(messages, chat_template=self.template, tokenize=False,
                                                  add_generation_prompt=add_generation_prompt)
        return self._text_ids(text)


# ---------------------------------------------------------------- strumenti

class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def __call__(self, group: str, name: str, cond, detail=None) -> bool:
        self.items.append({"gruppo": group, "controllo": name, "esito": bool(cond), "dettaglio": detail})
        print(("ok   " if cond else "FAIL ") + f"[{group}] {name}" + (f"  ({str(detail)[:300]})" if detail else ""),
              flush=True)
        return bool(cond)

    def failed(self) -> list[str]:
        return [f"[{c['gruppo']}] {c['controllo']}" for c in self.items if not c["esito"]]


def collapse(ids) -> list[int]:
    out = []
    for t in ids:
        if t == IMAGE_PAD_ID and out and out[-1] == IMAGE_PAD_ID:
            continue
        out.append(int(t))
    return out


def spans(ids, labels) -> list[tuple[int, int]]:
    out, start = [], None
    for i, lab in enumerate(labels):
        if lab != IGNORE and start is None:
            start = i
        if lab == IGNORE and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(labels)))
    return out


def find_all(ids: list[int], sub: list[int]) -> list[int]:
    return [i for i in range(len(ids) - len(sub) + 1) if ids[i:i + len(sub)] == sub]


def show(text: str) -> str:
    return text.replace("\n", "\\n")


# ---------------------------------------------------------------- analisi di un esempio

def analyse(tok, procs, alt: str, case_id: str, checks: Checks) -> dict:
    case, exp, a = CASES[case_id], EXPECTED[case_id][alt], ALTERNATIVES[alt]
    proc = procs[a["template"]]
    ttok = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    dec = lambda ids: tok.decode(ids, skip_special_tokens=False)  # noqa: E731
    g = f"{case_id}/{alt}"
    messages, images, prefill = compose(alt, case["turni"])
    train = proc.encode(messages, images)
    infer = proc.encode(messages[:-1], images, add_generation_prompt=True, prefill=prefill)
    bare = proc.encode(messages[:-1], images, add_generation_prompt=True)
    ids, labels, pids = train.input_ids.tolist(), train.labels.tolist(), infer.input_ids.tolist()
    system = SYS_CANON if a["system"] is None else SYS_VL
    text, turns = header(system), case["turni"]
    for i, t in enumerate(turns):
        text += user_block(t["utente"] + a.get("istruzione", ""), t["immagine"])
        if i < len(turns) - 1:
            text += OPEN + exp["cronologia"][i] + "<|im_end|>\n"
    exp_prompt = text + OPEN + exp["prefill"]
    exp_train = exp_prompt + exp["target"]
    exp_spans = list(exp.get("cronologia", [])) + [exp["target"]]
    got_spans = [dec(ids[s:e]) for s, e in spans(ids, labels)]
    checks(g, "testo dell'esempio di training uguale all'atteso", dec(collapse(ids)) == exp_train,
           None if dec(collapse(ids)) == exp_train else show(dec(collapse(ids))[-200:]))
    checks(g, "prompt d'inferenza uguale all'atteso", dec(collapse(pids)) == exp_prompt,
           None if dec(collapse(pids)) == exp_prompt else show(dec(collapse(pids))[-200:]))
    checks(g, "parità del prefisso: il prompt d'inferenza è un prefisso dell'esempio di training", ids[:len(pids)] == pids)
    for j in range(len(turns) - 1):  # prompt di ogni turno precedente, con la cronologia fino a lì
        m_j, i_j, p_j = compose(alt, turns[:j + 1])
        pj = proc.encode(m_j[:-1], i_j, add_generation_prompt=True, prefill=p_j).input_ids.tolist()
        checks(g, f"parità del prefisso al turno {j + 1}", ids[:len(pj)] == pj)
    checks(g, "intervalli con label uguali agli attesi", got_spans == exp_spans, None if got_spans == exp_spans
           else [show(s) for s in got_spans])
    checks(g, "label uguali agli input dove ci sono", all(lab in (IGNORE, t) for t, lab in zip(ids, labels)))
    target_ids = ttok(exp["target"][:-len(EOT)]) + [EOS_ID]
    prefill_ids = ttok(exp["prefill"]) if exp["prefill"] else []
    tail_ok = ids[len(ids) - len(target_ids):] == target_ids and all(lab != IGNORE for lab in labels[-len(target_ids):])
    checks(g, "target: token del testo atteso più EOS (151643), tutti con label", tail_ok)
    pre = ids[len(bare.input_ids):len(ids) - len(target_ids)]
    pre_labels = labels[len(bare.input_ids):len(ids) - len(target_ids)]
    want_pre = [THINK_ID, END_THINK_ID] if exp["prefill"] == "<think></think>" else []
    checks(g, "prefill: [151665, 151666] senza label se c'è, altrimenti nulla", pre == prefill_ids == want_pre
           and all(lab == IGNORE for lab in pre_labels), pre)
    if a["diretta"] == "think_vuoto" and turns[-1]["ragionamento"] is None:
        checks(g, "B: i primi due token con label sono <think> e </think>", target_ids[:2] == [THINK_ID, END_THINK_ID])
    if a["diretta"] == "senza_tag" and turns[-1]["ragionamento"] is None:
        checks(g, "C: nessun tag nel target diretto", THINK_ID not in target_ids and END_THINK_ID not in target_ids)
    counts = {"<think>": (ids.count(THINK_ID), exp_train.count(THINK)),
              "</think>": (ids.count(END_THINK_ID), exp_train.count(END_THINK)),
              "<|endoftext|>": (ids.count(EOS_ID), exp_train.count(EOT)),
              "<|im_end|>": (ids.count(IM_END_ID), exp_train.count("<|im_end|>")),
              "<|im_start|>": (ids.count(IM_START_ID), exp_train.count("<|im_start|>"))}
    checks(g, "nessun tag o terminatore duplicato: conteggi uguali all'atteso", all(x == y for x, y in counts.values()),
           counts)
    checks(g, "EOS solo in apertura (senza label) e alla fine (con label); nel prompt solo in apertura",
           ids.count(EOS_ID) == 2 and ids[0] == EOS_ID and labels[0] == IGNORE and ids[-1] == EOS_ID
           and labels[-1] == EOS_ID and pids.count(EOS_ID) == 1)
    checks(g, "<|im_end|> e <|im_start|> mai con label",
           all(lab == IGNORE for t, lab in zip(ids, labels) if t in (IM_END_ID, IM_START_ID)))
    open_ids = ttok(OPEN)
    starts = find_all(ids, open_ids)
    history = []
    for k, content in enumerate(exp.get("cronologia", [])):
        s = starts[k] + len(open_ids)
        e = ids.index(IM_END_ID, s)
        seg, seg_labels = ids[s:e], labels[s:e]
        same = seg == ttok(content)
        history.append({"contenuto": content, "token": seg, "ritokenizzati_uguali": same})
        checks(g, f"cronologia {k + 1}: i token nel contesto sono quelli del testo generato, con label, e "
                  "<|im_end|>\\n senza label", same and all(lab != IGNORE for lab in seg_labels)
               and labels[e] == IGNORE and labels[e + 1] == IGNORE)
    n_header = len(ttok(header(system)))
    n_image = ids.count(IMAGE_PAD_ID)
    return {
        "alternativa": alt, "caso": case_id, "template": a["template"],
        "messaggi": [{k: v for k, v in m.items()} for m in messages],
        "testo_del_template": proc.tokenizer.apply_chat_template(
            proc.render(messages[:-1])[0], chat_template=getattr(proc, "template", None), tokenize=False,
            add_generation_prompt=True),
        "atteso": {"prompt": exp_prompt, "training": exp_train, "intervalli_con_label": exp_spans},
        "input_ids": ids, "labels_mask": [int(lab != IGNORE) for lab in labels], "prompt_ids": pids,
        "token": [dec([t]) for t in ids],
        "cronologia": history,
        "conteggi": {"intestazione": n_header, "immagine": n_image,
                     "resto_del_prompt": len(bare.input_ids) - n_header - n_image, "prefill": len(prefill_ids),
                     "target": len(target_ids),
                     "cronologia_con_label": sum(len(h["token"]) for h in history),
                     "con_label": sum(lab != IGNORE for lab in labels), "prompt_d_inferenza": len(pids),
                     "totale": len(ids)},
    }


# ---------------------------------------------------------------- controlli trasversali

def cross_checks(results: dict, procs: dict, tok, checks: Checks) -> None:
    def ex(case, alt):
        return results.get((case, alt))
    for case in ("latex", "didascalia", "numerica", "scelta_multipla"):
        b, f, c, a = ex(case, "B"), ex(case, "B-F1"), ex(case, "C"), ex(case, "A")
        diff = [i for i, (x, y) in enumerate(zip(b["labels_mask"], f["labels_mask"])) if x != y]
        checks(f"{case}/B~B-F1", "stessi input; le label differiscono solo su <think> e </think>",
               b["input_ids"] == f["input_ids"] and len(diff) == 2
               and [b["input_ids"][i] for i in diff] == [THINK_ID, END_THINK_ID], diff)
        k = len(b["input_ids"]) - len(c["input_ids"])
        pos = len(b["input_ids"]) - b["conteggi"]["target"]
        checks(f"{case}/B~C", "C = B senza i due tag del target (2 token in meno)",
               k == 2 and b["input_ids"][:pos] == c["input_ids"][:pos]
               and b["input_ids"][pos + 2:] == c["input_ids"][pos:] and b["input_ids"][pos:pos + 2] == [THINK_ID,
                                                                                                        END_THINK_ID])
        na, nf = a["conteggi"]["intestazione"], f["conteggi"]["intestazione"]
        checks(f"{case}/A~B-F1", "stessi token e label dopo l'intestazione: differisce solo il system",
               a["input_ids"][na:] == f["input_ids"][nf:] and a["labels_mask"][na:] == f["labels_mask"][nf:],
               {"intestazione A": na, "intestazione B-F1": nf})
    for case in ("spiegazione",):
        b, c = ex(case, "B"), ex(case, "C")
        checks(f"{case}/B~C", "risposta ragionata: B e C identiche", b["input_ids"] == c["input_ids"]
               and b["labels_mask"] == c["labels_mask"])
    b, c = ex("dialogo", "B"), ex("dialogo", "C")
    tb, tc = b["conteggi"]["target"], c["conteggi"]["target"]
    checks("dialogo/B~C", "stesso ultimo turno ragionato; la cronologia diretta differisce per i due tag",
           b["input_ids"][-tb:] == c["input_ids"][-tc:] and len(b["input_ids"]) - len(c["input_ids"]) == 2)
    # baseline canonica: template della sonda (senza system o col canonico) = template dello snapshot, token per token
    snap, probe = procs["snapshot"], procs["sonda"]
    for (case, alt), r in results.items():
        if ALTERNATIVES[alt]["template"] != "snapshot":
            continue
        messages, images, prefill = compose(alt, CASES[case]["turni"])
        variants = {"sonda senza system": messages,
                    "sonda col system canonico": [{"role": "system", "content": SYS_CANON}] + messages}
        for name, msgs in variants.items():
            same = (probe.encode(msgs, images).input_ids.tolist() == r["input_ids"]
                    and probe.encode(msgs, images).labels.tolist() == snap.encode(messages, images).labels.tolist()
                    and probe.encode(msgs[:-1], images, add_generation_prompt=True, prefill=prefill)
                    .input_ids.tolist() == r["prompt_ids"])
            text_same = (tok.apply_chat_template(probe.render(msgs[:-1])[0], chat_template=PROBE_TEMPLATE,
                                                 tokenize=False, add_generation_prompt=True)
                         == tok.apply_chat_template(snap.render(messages[:-1])[0], tokenize=False,
                                                    add_generation_prompt=True))
            checks(f"{case}/{alt}", f"baseline canonica: {name} = snapshot (testo, token e label)", same and text_same)
    m = ex("spiegazione", "M")
    official = tok.apply_chat_template([{"role": "user", "content": PROBLEM + MATH500_INSTRUCTION}], tokenize=False,
                                       add_generation_prompt=True)
    literal = header(SYS_CANON) + user_block(PROBLEM + MATH500_INSTRUCTION, False) + OPEN
    checks("spiegazione/M", "prompt di M = resa del valutatore ufficiale (solo il messaggio utente, template dello "
           "snapshot) = stringa attesa", official == literal == tok.decode(m["prompt_ids"], skip_special_tokens=False))


def rejections(tok, procs, checks: Checks) -> list[dict]:
    import jinja2

    out = []
    img = (np.zeros((GRID[0] * GRID[1], 768), dtype=np.uint8), GRID)
    u = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q"}]}

    def stage_of(exc):
        if isinstance(exc, ContractError):
            return "contratto"
        if isinstance(exc, jinja2.exceptions.TemplateError):
            return "template"
        return "processor"

    def expect(name, stage, fn):
        try:
            fn()
            got, msg = "accettato", ""
        except Exception as exc:  # noqa: BLE001
            got, msg = stage_of(exc), f"{type(exc).__name__}: {exc}"
        out.append({"caso": name, "atteso": f"rifiuto ({stage})", "esito": got, "messaggio": msg})
        checks("rifiuti", name, got == stage, msg or got)

    sys_msg = lambda s: {"role": "system", "content": s}  # noqa: E731
    expect("sonda: system non ammesso", "template",
           lambda: procs["sonda"].encode([sys_msg("You are a geometry tutor."), u], [img], add_generation_prompt=True))
    expect("snapshot: system VL", "template",
           lambda: procs["snapshot"].encode([sys_msg(SYS_VL), u], [img], add_generation_prompt=True))
    expect("sonda: system non in prima posizione", "template",
           lambda: procs["sonda"].encode([u, sys_msg(SYS_VL)], [img], add_generation_prompt=True))
    expect("A: prefill <think></think> e tag nel contenuto", "contratto",
           lambda: check_turn("A", EMPTY, EMPTY + "12", True))
    rendered = procs["snapshot"].render([{"role": "user", "content": "Q"},
                                         {"role": "assistant", "reasoning_content": "r", "content": THINK + "x" +
                                          END_THINK + "12"}])[0][-1]["content"]
    expect("reasoning_content e <think> nel contenuto", "contratto", lambda: check_turn("B", "", rendered, True))
    expect("B: prefill nell'ultimo turno", "contratto", lambda: check_turn("B", EMPTY, "12", True))
    expect("C: prefill nell'ultimo turno", "contratto", lambda: check_turn("C", EMPTY, "12", True))
    expect("B: prefill nel prompt d'inferenza", "contratto", lambda: check_prompt("B", EMPTY))
    expect("C: prefill nel prompt d'inferenza", "contratto", lambda: check_prompt("C", EMPTY))
    expect("C: <think></think> vuoto", "contratto", lambda: check_turn("C", "", EMPTY + "12", True))
    expect("B: risposta diretta senza <think>", "contratto", lambda: check_turn("B", "", "12", True))
    expect("A: <think></think> vuoto con label nell'ultimo turno", "contratto",
           lambda: check_turn("A", "", EMPTY + "12", True))
    expect("M: risposta diretta con <think></think>", "contratto",
           lambda: check_turn("M", "", EMPTY + "\\boxed{6}", True))
    expect("M: risposta senza tag", "contratto", lambda: check_turn("M", "", "\\boxed{6}", True))
    expect("M: risposta finale senza \\boxed{}", "contratto", lambda: check_turn("M", "", THINK + "r" + END_THINK + "6",
                                                                                   True))
    expect("M: system VL", "contratto", lambda: check_system("M", [sys_msg(SYS_VL)]))
    expect("B: system canonico", "contratto", lambda: check_system("B", [sys_msg(SYS_CANON)]))
    expect("B-F1: target ragionato", "contratto", lambda: compose("B-F1", CASES["spiegazione"]["turni"]))
    expect("B-F1: più turni", "contratto", lambda: compose("B-F1", CASES["dialogo"]["turni"]))
    expect("tag nel testo dato al contratto", "contratto", lambda: turn_text("B", None, THINK + "12", True))
    for bad in (END_THINK + "12", THINK + "a" + END_THINK + "b" + THINK + "c" + END_THINK + "d",
                "x" + THINK + "a" + END_THINK + "b", THINK + "a"):
        for alt in ALTERNATIVES:
            expect(f"{alt}: tag malformati {bad!r}", "contratto", lambda alt=alt, bad=bad: check_turn(alt, "", bad, True))
    expect("processor: token speciale nel testo dell'assistente", "processor",
           lambda: procs["snapshot"].encode([u, {"role": "assistant", "content": "12<|im_end|>"}], [img]))
    expect("processor: prefill in un turno intermedio", "processor",
           lambda: procs["sonda"].encode([sys_msg(SYS_VL), u, {"role": "assistant", "content": "A", "prefill": EMPTY},
                                          {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "B"}],
                                         [img]))
    return out


def operational(tok, procs, checks: Checks) -> list[dict]:
    """Combinazioni incoerenti che il processor operativo accetta (fatti per C3) e che il contratto rifiuta."""
    out = []
    proc = procs["snapshot"]
    cases = {  # messaggio dell'assistente, alternativa, prefill e contenuto da dare al contratto
        "prefill <think></think> e <think></think> nel contenuto": ({"content": EMPTY + "12", "prefill": EMPTY},
                                                                     "A", EMPTY, EMPTY + "12"),
        "reasoning_content e <think> nel contenuto": ({"content": THINK + "x" + END_THINK + "12",
                                                       "reasoning_content": "r"}, "B", "",
                                                      THINK + "r" + END_THINK + THINK + "x" + END_THINK + "12"),
        "</think> senza <think>": ({"content": END_THINK + "12"}, "B", "", END_THINK + "12"),
        "due blocchi <think>": ({"content": THINK + "a" + END_THINK + THINK + "b" + END_THINK + "12"}, "B", "",
                                THINK + "a" + END_THINK + THINK + "b" + END_THINK + "12"),
    }
    for name, (msg, alt, prefill, content) in cases.items():
        try:
            ex = proc.encode([{"role": "user", "content": "Q"}, {"role": "assistant", **msg}])
            tail = tok.decode(ex.input_ids.tolist()[-12:], skip_special_tokens=False)
            accepted = True
        except Exception as exc:  # noqa: BLE001
            tail, accepted = f"{type(exc).__name__}: {exc}", False
        try:
            check_turn(alt, prefill, content, True)
            refused = False
        except ContractError:
            refused = True
        out.append({"caso": name, "processor_operativo": "accettato" if accepted else "rifiutato",
                    "coda": tail, "contratto_della_sonda": "rifiutato" if refused else "accettato"})
        checks("processor operativo", f"{name}: accettato dal processor, rifiutato dal contratto", accepted and refused,
               show(tail))
    return out


def boundaries(tok, procs, checks: Checks) -> list[dict]:
    """Confine fra `assistant\\n` del template e il contenuto di un turno precedente."""
    ttok = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    out = []
    specs = (("C", "\n12", "rifiuto del processor: il prefisso non si conserva (\\n\\n diventa un token)"),
             ("C", "12", "uguali"), ("B", EMPTY + "\n12", "uguali"))
    for alt, content, want in specs:
        msgs = [{"role": "system", "content": SYS_VL}, {"role": "user", "content": "Q"},
                {"role": "assistant", "content": content}, {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "ok"}]
        try:
            ids = procs["sonda"].encode(msgs).input_ids.tolist()
            open_ids = ttok(OPEN)
            s = find_all(ids, open_ids)[0] + len(open_ids)
            seg = ids[s:ids.index(IM_END_ID, s)]
            got = "uguali" if seg == ttok(content) else f"diversi: {seg} contro {ttok(content)}"
        except ValueError as exc:
            got = f"rifiuto del processor: {exc}"
        ok = got == want if want == "uguali" else got.startswith("rifiuto del processor") and "prefisso" in got
        out.append({"alternativa": alt, "contenuto": content, "atteso": want, "esito": got})
        checks("confine della cronologia", f"{alt}, turno precedente {content!r}: {want}", ok, got)
    joined = tok.apply_chat_template([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "\n12"}],
                                     chat_template=PROBE_TEMPLATE, tokenize=False)
    out.append({"nota": "ritokenizzazione di 'assistant\\n\\n12'", "token": ttok(joined.split("<|im_start|>")[-1])})
    return out


def self_checks(results: dict, tok, checks: Checks) -> None:
    """I confronti sanno fallire: un carattere in più nell'atteso e una label in più vengono visti."""
    r = results[("latex", "B")]
    text = tok.decode(collapse(r["input_ids"]), skip_special_tokens=False)
    checks("controlli dei controlli", "un carattere in più nel testo atteso è visto", text != r["atteso"]["training"] + "x")
    mask = list(r["labels_mask"])
    first = mask.index(1)
    mask[first - 1] = 1
    got = [tok.decode(r["input_ids"][s:e], skip_special_tokens=False)
           for s, e in spans(r["input_ids"], [t if m else IGNORE for t, m in zip(r["input_ids"], mask)])]
    checks("controlli dei controlli", "una label in più prima del target è vista", got != r["atteso"]["intervalli_con_label"])


# ---------------------------------------------------------------- identità e fatti

def identity(snapshot: Path, tok) -> dict:
    import tokenizers
    import transformers

    files = {name: sha256_file(snapshot / name) for name in SNAPSHOT_FILES}
    template_file = (snapshot / "chat_template.jinja").read_text()
    gen = json.loads((snapshot / "generation_config.json").read_text())
    return {"snapshot": str(snapshot.relative_to(ROOT)), "file_sha256": files,
            "chat_template_del_tokenizer_sha256": sha256_text(tok.chat_template),
            "chat_template_del_tokenizer_uguale_al_file": tok.chat_template == template_file,
            "template_della_sonda": {"nome": "limite-vl-sonda/1", "sha256": sha256_text(PROBE_TEMPLATE),
                                     "testo": PROBE_TEMPLATE},
            "tokenizer": {"classe": type(tok).__name__, "voci": len(tok), "transformers": transformers.__version__,
                          "tokenizers": tokenizers.__version__},
            "generation_config_eos": gen.get("eos_token_id")}


def facts(tok, procs, checks: Checks, ident: dict) -> dict:
    ttok = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    ids = {t: tok.convert_tokens_to_ids(t) for t in (THINK, END_THINK, "<|im_start|>", "<|im_end|>", EOT)}
    checks("fatti", "id: <think> 151665, </think> 151666, <|im_start|> 151644, <|im_end|> 151645, <|endoftext|> 151643",
           list(ids.values()) == [151665, 151666, 151644, 151645, 151643], ids)
    checks("fatti", "generation_config: si ferma solo su <|endoftext|> (151643)", ident["generation_config_eos"] in
           (151643, [151643]), ident["generation_config_eos"])
    checks("fatti", "il tokenizer usa il chat_template.jinja dello snapshot", ident["chat_template_del_tokenizer_uguale_al_file"])
    suites = SUITES.read_text()
    literal = 'MATH500_INSTRUCTION = "' + MATH500_INSTRUCTION.replace("\\", "\\\\").replace("\n", "\\n") + '"'
    checks("fatti", "istruzione di MATH-500 di replay.py uguale a quella del valutatore (suites.py)", literal in suites)
    rc = tok.apply_chat_template([{"role": "user", "content": "Q"},
                                  {"role": "assistant", "content": "A", "reasoning_content": "r"}], tokenize=False)
    inline = procs["snapshot"].render([{"role": "user", "content": "Q"},
                                       {"role": "assistant", "content": "A", "reasoning_content": "r"}])[0][-1]["content"]
    checks("fatti", "reasoning_content nel template: <think>\\n…\\n</think>\\n\\n; il processor lo scrive invece inline "
           "come lo genera Violetto", rc.endswith("<|im_start|>assistant\n<think>\nr\n</think>\n\nA<|im_end|>\n")
           and inline == "<think>r</think>A", show(rc[-60:]))
    canon, vl = ttok(SYS_CANON), ttok(SYS_VL)
    common = next((i for i, (x, y) in enumerate(zip(canon, vl)) if x != y), min(len(canon), len(vl)))
    return {"id_dei_token": ids,
            "system": {"canonico": {"token": len(canon), "intestazione": len(ttok(header(SYS_CANON)))},
                       "vl": {"token": len(vl), "intestazione": len(ttok(header(SYS_VL)))},
                       "prefisso_comune_in_token": common,
                       "token_divergenti": {"canonico": [tok.decode([t]) for t in canon[common:common + 2]],
                                            "vl": [tok.decode([t]) for t in vl[common:common + 2]]}},
            "reasoning_content_nel_template": rc, "reasoning_content_nel_processor": inline}


# ---------------------------------------------------------------- rapporto leggibile

def token_table(r: dict, start: int) -> list[str]:
    rows = ["pos    id      label  token"]
    for i in range(start, len(r["input_ids"])):
        t = r["input_ids"][i]
        if t == IMAGE_PAD_ID and i > 0 and r["input_ids"][i - 1] == IMAGE_PAD_ID:
            continue
        rows.append(f"{i:<6} {t:<7} {'sì' if r['labels_mask'][i] else 'no':<6} {r['token'][i]!r}")
    return rows


def markdown(report: dict, results: dict) -> str:
    ident, f = report["identita"], report["fatti_misurati"]
    lines = [f"# Sonda del contratto di risposta (D3): {report['nome']}", "",
             f"Job {report['job_id']}, codice `{report['codice'].get('commit', '?')}`"
             f"{' con modifiche' if report['codice'].get('dirty') else ''}, schema `{SCHEMA}`. Generato da "
             "`scripts/contract_probe.py`: diagnostica CPU col tokenizer dello snapshot, senza pesi. Dimostra "
             "proprietà meccaniche del formato, non quale alternativa il modello seguirà meglio. Le alternative e il "
             "template della sonda esistono solo nella sonda; nessun contratto è attivato.", "",
             "## Identità", "", "| file dello snapshot | sha256 |", "| --- | --- |"]
    lines += [f"| `{k}` | `{v}` |" for k, v in ident["file_sha256"].items()]
    tk = ident["tokenizer"]
    lines += ["", f"- Tokenizer `{tk['classe']}`, {tk['voci']} voci, transformers {tk['transformers']}, tokenizers "
              f"{tk['tokenizers']}; il tokenizer usa il `chat_template.jinja` dello snapshot: "
              f"{'sì' if ident['chat_template_del_tokenizer_uguale_al_file'] else 'no'}.",
              f"- Template della sonda `limite-vl-sonda/1`, sha256 `{ident['template_della_sonda']['sha256']}`: "
              "quello dello snapshot con un system iniziale canonico o VL; testo in `report.json`.",
              f"- `generation_config.json`: `eos_token_id` = {ident['generation_config_eos']}.", "",
              "## Intestazioni e system", "",
              f"- Canonica: `{show(header(SYS_CANON))}`, {f['system']['canonico']['intestazione']} token.",
              f"- VL: `{show(header(SYS_VL))}`, {f['system']['vl']['intestazione']} token.",
              f"- Prefisso comune dei due system: {f['system']['prefisso_comune_in_token']} token; poi "
              f"{f['system']['token_divergenti']['canonico']} nel canonico e {f['system']['token_divergenti']['vl']} "
              "nel VL.", "", "## Alternative", "", "| id | fase | regola |", "| --- | --- | --- |"]
    lines += [f"| {k} | {a['fase']} | {a['descrizione']} |" for k, a in ALTERNATIVES.items()]
    lines += ["", "## Esempi", "",
              "Testi decodificati con `<|image_pad|>` contratto (×N token immagine); `⟨canonica⟩` e `⟨VL⟩` sono le "
              "intestazioni sopra. `prompt` è ciò che riceve il modello in inferenza (prefill compreso), `target` le "
              "label dell'ultimo turno; con la cronologia, anche i turni precedenti hanno label."]
    for case_id, case in CASES.items():
        lines += ["", f"### {case_id}: {case['descrizione']}", ""]
        for t in case["turni"]:
            lines.append(f"- utente: `{show(t['utente'])}`" + (" (con immagine)" if t["immagine"] else ""))
        lines += ["", "```"]
        for alt in ALTERNATIVES:
            r = results.get((case_id, alt))
            if r is None:
                continue
            c = r["conteggi"]
            sysname = "⟨canonica⟩" if ALTERNATIVES[alt]["system"] is None else "⟨VL⟩"
            hdr = header(SYS_CANON if ALTERNATIVES[alt]["system"] is None else SYS_VL)
            prompt = r["atteso"]["prompt"][len(hdr):].replace(IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER.replace(
                "<|image_pad|>", f"<|image_pad|>×{c['immagine']}"))
            lines.append(f"{alt:<5} prompt  {sysname}{show(prompt)}")
            for h in r["atteso"]["intervalli_con_label"][:-1]:
                lines.append(f"      cronologia con label  {show(h)}")
            lines.append(f"      target  {show(r['atteso']['intervalli_con_label'][-1])}")
            lines.append(f"      token   intestazione {c['intestazione']}, resto del prompt {c['resto_del_prompt']}"
                         f" + immagine {c['immagine']}, prefill {c['prefill']} (senza label), target {c['target']}"
                         f" (con label), cronologia con label {c['cronologia_con_label']}, totale {c['totale']}")
        lines.append("```")
    lines += ["", "## Token per token", "", "Dall'apertura dell'ultimo turno dell'assistente; per il dialogo dal primo "
              "turno dell'assistente. `label` = il token è supervisionato."]
    for case_id, alt, first in (("latex", "A", False), ("latex", "B", False), ("latex", "B-F1", False),
                                ("latex", "C", False), ("dialogo", "B", True), ("dialogo", "C", True)):
        r = results[(case_id, alt)]
        starts = [i for i, t in enumerate(r["input_ids"][:-1]) if t == IM_START_ID and r["token"][i + 1] == "assistant"]
        start = starts[0] if first else starts[-1]
        lines += ["", f"### {case_id}, {alt}", "", "```", *token_table(r, start), "```"]
    lines += ["", "## Controlli", ""]
    groups: dict[str, list] = {}
    for c in report["controlli"]:
        groups.setdefault(c["gruppo"].split("/")[0] if "/" in c["gruppo"] else c["gruppo"], []).append(c)
    lines += ["| gruppo | superati | falliti |", "| --- | --- | --- |"]
    lines += [f"| {g} | {sum(c['esito'] for c in cs)} | {sum(not c['esito'] for c in cs)} |" for g, cs in groups.items()]
    lines += ["", f"In tutto {sum(c['esito'] for c in report['controlli'])} su {len(report['controlli'])}; falliti: "
              f"{', '.join(report['falliti']) or 'nessuno'}. Elenco completo in `report.json` e nel riepilogo JSON.", "",
              "## Rifiuti delle combinazioni incoerenti", "", "| combinazione | atteso | esito |", "| --- | --- | --- |"]
    lines += [f"| {show(x['caso'])} | {x['atteso']} | {x['esito']} |" for x in report["rifiuti"]]
    lines += ["", "## Processor operativo: combinazioni accettate oggi", "",
              "| combinazione | processor operativo | contratto della sonda | coda dell'esempio |", "| --- | --- | --- | --- |"]
    lines += [f"| {x['caso']} | {x['processor_operativo']} | {x['contratto_della_sonda']} | "
              f"`{show(x['coda']).replace('|', chr(92) + '|')}` |" for x in report["processor_operativo"]]
    lines += ["", "## Confine fra template e cronologia", "", "| alternativa | turno precedente | atteso | esito |",
              "| --- | --- | --- | --- |"]
    lines += [f"| {x['alternativa']} | `{show(x['contenuto'])}` | {x['atteso']} | {show(x['esito'])[:120]} |"
              for x in report["confini"] if "alternativa" in x]
    lines += ["", "## Costi", "", f"- {report['costi']['secondi']} s, RSS massima {report['costi']['rss_max_mib']} MiB; "
              "nessun peso, nessun dataset, solo i file del tokenizer e del template.", "", "## Limiti", ""]
    lines += [f"- {x}" for x in LIMITS]
    return "\n".join(lines) + "\n"


LIMITS = [
    "proprietà meccaniche del formato (testo, token, label, prefisso, terminatori, rifiuti), non il comportamento del "
    "modello: quale alternativa Violetto o il modello VL seguiranno meglio richiede un esperimento GPU autorizzato",
    "pochi esempi scritti a mano; il segnaposto visivo ha 64 token, il numero reale dipende dalla risoluzione",
    "il template della sonda e le regole del contratto esistono solo qui: il processor, i loader e le configurazioni "
    "operative restano quelli storici (A)",
    "la baseline canonica è verificata sul dominio condiviso: senza system o col system canonico in prima posizione; "
    "un system canonico in un'altra posizione, che lo snapshot accetta e non scrive, la sonda lo rifiuta",
    "la ritokenizzazione dei turni generati è provata sul testo; i token generati davvero da un modello possono avere "
    "una segmentazione diversa da quella canonica del tokenizer",
]


def run(args) -> int:
    from transformers import Qwen2Tokenizer

    t0 = time.time()
    refuse_existing(args.out, Path(f"{args.docs}.json"), Path(f"{args.docs}.md"))
    snapshot = args.snapshot.resolve()
    tok = Qwen2Tokenizer.from_pretrained(str(snapshot), local_files_only=True)
    procs = {"snapshot": LimiteVLProcessor(tok), "sonda": TemplateProcessor(tok, PROBE_TEMPLATE)}
    checks = Checks()
    ident = identity(snapshot, tok)
    measured = facts(tok, procs, checks, ident)
    results = {}
    for case_id, per_alt in EXPECTED.items():
        for alt in per_alt:
            results[(case_id, alt)] = analyse(tok, procs, alt, case_id, checks)
    cross_checks(results, procs, tok, checks)
    rej = rejections(tok, procs, checks)
    oper = operational(tok, procs, checks)
    bound = boundaries(tok, procs, checks)
    self_checks(results, tok, checks)
    rss = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    report = {"schema": SCHEMA, "nome": args.out.name, "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
              "perimetro": "sonda CPU del contratto D3: tokenizer e template dello snapshot, esempi scritti a mano; "
                           "nessun peso, nessun dataset, nessun contratto attivato",
              "identita": ident, "alternative": ALTERNATIVES, "fatti_misurati": measured,
              "controlli": checks.items, "falliti": checks.failed(), "rifiuti": rej, "processor_operativo": oper,
              "confini": bound, "limiti": LIMITS,
              "costi": {"secondi": round(time.time() - t0, 1), "rss_max_mib": rss}}
    examples = {f"{c}/{a}": r for (c, a), r in results.items()}
    args.out.mkdir(parents=True)
    write_new_json(args.out / "report.json", {**report, "esempi": examples})
    compact = {k: {kk: vv for kk, vv in r.items() if kk not in ("input_ids", "labels_mask", "prompt_ids", "token",
                                                                  "testo_del_template")}
               for k, r in examples.items()}
    ident_compact = {**ident, "template_della_sonda": {k: v for k, v in ident["template_della_sonda"].items()
                                                       if k != "testo"}}
    write_new_json(Path(f"{args.docs}.json"), {**report, "identita": ident_compact, "esempi": compact,
                                               "rapporto_completo": str((args.out / "report.json").relative_to(ROOT))})
    Path(f"{args.docs}.md").write_text(markdown(report, results))
    print(json.dumps({"controlli": len(checks.items), "falliti": checks.failed(), **report["costi"]},
                     ensure_ascii=False), flush=True)
    return 1 if checks.failed() else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs", required=True, help="prefisso del riepilogo versionato (PREFISSO.json e PREFISSO.md)")
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
