"""Record delle soluzioni F2 (schema f2-sol/2): identità, hash, controlli deterministici, certificati.

Una soluzione F2 è una generazione precisa di Violetto, non un problema: due soluzioni dello stesso problema hanno
ID diversi. Contratto (revisione tecnica del 2026-10-21, alternativa C approvata il 2026-10-22):

- `sid` = "<gen_run>/<chiave>/<indice>": nome della cartella della generazione (scripts/f2_generate.py), chiave del
  problema, indice della generazione nella sua riga (`completions[indice]`). Si ricava dagli artefatti, quindi
  vale anche per le generazioni storiche (scripts/f2_backfill.py).
- `raw_sha256` = sha256 del testo generato; `text_sha256` = sha256 del testo selezionato (dopo f2.clean_solution),
  cioè quello che andrebbe in training. Un testo cambiato dopo un audit cambia l'hash e l'associazione non vale più.
- `checks`: controlli deterministici sul testo selezionato. `failed` elenca quelli che scartano (REJECTING),
  `flags` quelli solo rilevati (FLAGGED), su cui la decisione non è ancora presa.
- Nessun campo del record certifica la soluzione: una risposta finale giusta (`answer_correct`) non basta.
  L'ammissione al training passa solo da admit() con un certificato del lotto (schema f2-cert/1), che in questa
  fase nessuno script scrive.

I record della selezione v1 (selected.jsonl senza `schema`) restano leggibili con read_selected().
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from .provenance import sha256_text

SOLUTION_SCHEMA = "f2-sol/2"
CERT_SCHEMA = "f2-cert/1"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Intestazione del ponte (f2.BRIDGE_HEADER) citata nella soluzione: `It says: "Figure: ABC is ...`. Nell'esempio di
# training il prompt non c'è, quindi la citazione si riferisce a un testo che il modello non riceve. "Figure 1:"
# (titoli delle descrizioni di MMFineReason) è già coperto da f2._CAPTION_TRACES e qui non conta.
PROMPT_QUOTE = re.compile(r"(?<![\w-])Figure:(?=\s)")

# Controlli che scartano. think_tags: il formato di Violetto ha un solo blocco di ragionamento <think>...</think>
# (CLAUDE.md, formato delle risposte); f2.split_completion usa l'ultimo </think>, quindi un secondo </think> lascia
# nel ragionamento un pezzo troncato (f011 di mavis-f2sol).
REJECTING = ("think_tags",)
# Controlli solo rilevati: la scelta (scartare, riscrivere o tenere) va approvata dall'utente (revisione, sezione 10).
FLAGGED = ("prompt_quote", "text_before_think")


# ---------------------------------------------------------------- identità

def gen_run_name(gen_dir: Path | str) -> str:
    """Nome della generazione nel sid: nome della cartella di f2_generate.py, risolti i link (stessa regola per
    f2_select.py e f2_backfill.py)."""
    name = Path(gen_dir).resolve().name
    if not name or "/" in name:
        raise ValueError(f"nome di generazione non valido: {gen_dir!r}")
    return name


def make_sid(gen_run: str, key: str, completion: int) -> str:
    if not gen_run or "/" in gen_run:
        raise ValueError(f"gen_run non valido: {gen_run!r}")
    if not key:
        raise ValueError("chiave vuota")
    if not isinstance(completion, int) or isinstance(completion, bool) or completion < 0:
        raise ValueError(f"indice di generazione non valido: {completion!r}")
    return f"{gen_run}/{key}/{completion}"


def parse_sid(sid: str) -> tuple[str, str, int]:
    """(gen_run, chiave, indice). gen_run non contiene "/", l'indice è l'ultima parte: la chiave può contenere "/"."""
    gen_run, sep, rest = sid.partition("/")
    key, sep2, idx = rest.rpartition("/")
    if not (sep and sep2 and gen_run and key and idx.isdigit()):
        raise ValueError(f"sid non valido: {sid!r}")
    return gen_run, key, int(idx)


# ---------------------------------------------------------------- controlli deterministici

def solution_checks(text: str) -> dict:
    """Controlli sul testo selezionato (gli stessi tag e le stesse frasi della generazione: clean_solution riscrive
    solo parole)."""
    n_open, n_close = text.count(THINK_OPEN), text.count(THINK_CLOSE)
    order_ok = n_open == 1 and n_close == 1 and text.index(THINK_OPEN) < text.index(THINK_CLOSE)
    before = text.split(THINK_OPEN, 1)[0] if n_open else text
    return {"think_open": n_open, "think_close": n_close, "think_order_ok": order_ok,
            "text_before_think": bool(before.strip()) if n_open else False,
            "prompt_quotes": len(PROMPT_QUOTE.findall(text))}


def check_failures(checks: dict) -> list[str]:
    """Controlli di REJECTING non superati."""
    out = []
    if not (checks["think_open"] == 1 and checks["think_close"] == 1 and checks["think_order_ok"]):
        out.append("think_tags")
    return out


def check_flags(checks: dict) -> list[str]:
    """Controlli di FLAGGED che segnalano qualcosa (nessuno scarto)."""
    out = []
    if checks["prompt_quotes"] > 0:
        out.append("prompt_quote")
    if checks["text_before_think"]:
        out.append("text_before_think")
    return out


# ---------------------------------------------------------------- record

def solution_record(*, gen_run: str, key: str, completion: int, solution: str, raw_text: str | None = None,
                    raw_sha256: str | None = None, finish_reason: str, tokens: int | None, source_dataset: str | None,
                    answer: str, rewritten: bool, violetto_pass_rate: float | None, origin: str,
                    answer_letter: str | None = None, match: str | None = None) -> dict:
    """Record v2 di una soluzione selezionata. I campi della selezione v1 (key, source_dataset, answer, solution,
    rewritten, tokens, violetto_pass_rate) restano con lo stesso nome: gli script che leggono selected.jsonl leggono
    anche i record v2. `question_sha256` è None finché f2_generate.py non salva la domanda (passo 4 del piano)."""
    if (raw_text is None) == (raw_sha256 is None):
        raise ValueError("serve raw_text oppure raw_sha256")
    checks = solution_checks(solution)
    rec = {"schema": SOLUTION_SCHEMA, "sid": make_sid(gen_run, key, completion), "gen_run": gen_run, "key": key,
           "completion": completion, "source_dataset": source_dataset, "answer": answer}
    if answer_letter is not None:
        rec["answer_letter"] = answer_letter
    rec.update({"solution": solution, "raw_sha256": raw_sha256 or sha256_text(raw_text),
                "text_sha256": sha256_text(solution), "question_sha256": None, "rewritten": rewritten,
                "tokens": tokens, "finish_reason": finish_reason, "violetto_pass_rate": violetto_pass_rate,
                "answer_correct": True, "checks": checks, "failed": check_failures(checks),
                "flags": check_flags(checks), "origin": origin})
    if match is not None:
        rec["match"] = match
    return rec


def record_version(rec: dict) -> int:
    """2 per i record f2-sol/2, 1 per le righe di selected.jsonl delle selezioni storiche."""
    if rec.get("schema") == SOLUTION_SCHEMA:
        return 2
    if "schema" not in rec and {"key", "solution", "answer"} <= rec.keys():
        return 1
    raise ValueError(f"record di selezione sconosciuto: schema {rec.get('schema')!r}, campi {sorted(rec)[:8]}")


def read_selected(path: Path | str) -> Iterator[dict]:
    """Righe di una selezione, v1 o v2 (record_version dice quale)."""
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            try:
                record_version(rec)
            except ValueError as e:
                raise ValueError(f"{path}:{n}: {e}") from None
            yield rec


def validate_record(rec: dict) -> list[str]:
    """Problemi di un record v2 (lista vuota se è coerente): sid, hash del testo, controlli ricalcolati."""
    problems = []
    if record_version(rec) != 2:
        return ["record v1: niente sid né hash"]
    try:
        gen_run, key, idx = parse_sid(rec["sid"])
        if (gen_run, key, idx) != (rec["gen_run"], rec["key"], rec["completion"]):
            problems.append("sid diverso da gen_run/chiave/indice")
    except (KeyError, ValueError) as e:
        problems.append(f"sid: {e}")
    if not _HEX64.match(str(rec.get("raw_sha256", ""))):
        problems.append("raw_sha256 mancante o non valido")
    if rec.get("text_sha256") != sha256_text(rec.get("solution", "")):
        problems.append("testo diverso dal suo text_sha256")
    checks = solution_checks(rec.get("solution", ""))
    if rec.get("checks") != checks or rec.get("failed") != check_failures(checks) or rec.get("flags") != check_flags(checks):
        problems.append("controlli diversi da quelli ricalcolati")
    return problems


def association_ok(rec: dict, sid: str, text_sha256: str) -> bool:
    """Un esito (di audit o certificato) dato a (sid, text_sha256) vale per questo record solo se il sid è lo stesso
    e il testo attuale ha ancora quell'hash."""
    return rec.get("sid") == sid and sha256_text(rec.get("solution", "")) == text_sha256 == rec.get("text_sha256")


def duplicate_sids(records) -> list[str]:
    seen, dup = set(), set()
    for rec in records:
        (dup if rec["sid"] in seen else seen).add(rec["sid"])
    return sorted(dup)


def compare_selections(new: dict[str, str], old: dict[str, str], rejected: set[str], show: int = 50) -> dict:
    """Confronto fra due selezioni v2 ({sid: text_sha256}): una nuova (f2_select.py) e una di riferimento (di solito
    il backfill della selezione storica). `rejected` = sid scartati dai nuovi controlli nella selezione nuova: spiegano
    le soluzioni che mancano rispetto al riferimento."""
    same = [s for s in new if s in old and new[s] == old[s]]
    changed = sorted(s for s in new if s in old and new[s] != old[s])
    only_new = sorted(s for s in new if s not in old)
    only_old = sorted(s for s in old if s not in new)
    explained = [s for s in only_old if s in rejected]
    unexplained = [s for s in only_old if s not in rejected]
    return {"new_solutions": len(new), "reference_solutions": len(old), "same_sid_and_text": len(same),
            "same_sid_other_text": len(changed), "only_new": len(only_new), "only_reference": len(only_old),
            "only_reference_rejected_by_new_checks": len(explained), "only_reference_unexplained": len(unexplained),
            "identical_except_new_checks": not changed and not only_new and not unexplained,
            "examples": {"same_sid_other_text": changed[:show], "only_new": only_new[:show],
                         "only_reference_unexplained": unexplained[:show]}}


# ---------------------------------------------------------------- certificati e ammissione

def validate_certificate(cert: dict) -> list[str]:
    """Problemi strutturali di un certificato f2-cert/1 (vuoto = ben formato). Un certificato `certified` deve
    dichiarare le soglie e i tassi su cui è stato deciso: un certificato senza numeri non vale."""
    problems = []
    if not isinstance(cert, dict):
        return ["non è un oggetto JSON"]
    if cert.get("schema") != CERT_SCHEMA:
        problems.append(f"schema {cert.get('schema')!r}, atteso {CERT_SCHEMA}")
    pop = cert.get("population") or {}
    if not _HEX64.match(str(pop.get("selection_sha256", ""))):
        problems.append("population.selection_sha256 mancante")
    if cert.get("decision") not in ("certified", "rejected"):
        problems.append(f"decision {cert.get('decision')!r}")
    for field in ("audited", "excluded", "quarantined"):
        entries = cert.get(field)
        if not isinstance(entries, list):
            problems.append(f"{field} mancante")
            continue
        for e in entries:
            if not (isinstance(e, list) and len(e) >= 2 and isinstance(e[0], str) and _HEX64.match(str(e[1]))):
                problems.append(f"{field}: voce non valida {e!r}")
                break
    audit = cert.get("audit") or {}
    if not (audit.get("name") and _HEX64.match(str(audit.get("manifest_sha256", "")))):
        problems.append("audit.name o audit.manifest_sha256 mancante")
    if cert.get("decision") == "certified":
        thresholds, rates = cert.get("thresholds") or {}, cert.get("rates") or {}
        if not thresholds or any(v is None for v in thresholds.values()):
            problems.append("certificato senza soglie")
        if not rates or any(rates.get(k) is None for k in thresholds):
            problems.append("certificato senza i tassi delle sue soglie")
    return problems


def admit(rec: dict, cert: dict | None, selection_sha256: str | None) -> tuple[bool, str]:
    """(ammessa, motivo). Ammessa solo se: record v2 coerente, controlli che scartano superati, risposta finale
    verificata, certificato ben formato e `certified` per questa stessa selezione, soluzione non esclusa né in
    quarantena, testo uguale a quello auditato. La risposta giusta è necessaria, ma da sola non ammette nulla. Qui non si controlla l'ammissione del problema (decontaminazione,
    audit dei problemi): la farà il gate del build (passo 6)."""
    if record_version(rec) != 2:
        return False, "record v1 senza sid né hash"
    problems = validate_record(rec)
    if problems:
        return False, "record incoerente: " + "; ".join(problems)
    if rec["failed"]:
        return False, "controlli non superati: " + ", ".join(rec["failed"])
    if rec.get("answer_correct") is not True:
        return False, "risposta finale non verificata"
    if cert is None:
        return False, "nessun certificato"
    problems = validate_certificate(cert)
    if problems:
        return False, "certificato non valido: " + "; ".join(problems)
    if cert["decision"] != "certified":
        return False, "lotto non certificato"
    if not selection_sha256 or cert["population"]["selection_sha256"] != selection_sha256:
        return False, "certificato di un'altra selezione"
    sid, sha = rec["sid"], rec["text_sha256"]
    audited = {e[0]: e[1] for e in cert["audited"]}
    if sid in audited and audited[sid] != sha:
        return False, "testo cambiato dopo l'audit"
    if any(e[0] == sid for e in cert["excluded"]):
        return False, "esclusa dall'audit"
    if any(e[0] == sid for e in cert["quarantined"]):
        return False, "in quarantena"
    return True, "ammessa"
