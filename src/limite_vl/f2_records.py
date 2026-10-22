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
  L'ammissione al training passa solo da admit() con un certificato del lotto (schema f2-cert/1, scritto da
  scripts/f2_certify.py con la politica di config/audit/f2_policy.json). Il certificato porta le regole per record
  (record_policy: controlli che scartano, decisione su ogni segnalazione) e admit() le applica.

I record della selezione v1 (selected.jsonl senza `schema`) restano leggibili con read_selected().
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from .provenance import ROOT, sha256_file, sha256_text

SOLUTION_SCHEMA = "f2-sol/2"
BUILD_SCHEMA = "f2-build/2"
CERT_SCHEMA = "f2-cert/1"
POLICY_SCHEMA = "f2-policy/1"
FLAG_DECISIONS = ("reject", "keep")
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
                    answer_letter: str | None = None, match: str | None = None, question: str | None = None,
                    question_sha256: str | None = None, prompt_sha256: str | None = None) -> dict:
    """Record v2 di una soluzione selezionata. I campi della selezione v1 (key, source_dataset, answer, solution,
    rewritten, tokens, violetto_pass_rate) restano con lo stesso nome: gli script che leggono selected.jsonl leggono
    anche i record v2. `question` (domanda dell'esempio di training) e `prompt_sha256` vengono dalla riga della
    generazione; sono None per le generazioni precedenti al passo 4, che quindi non arrivano al build (admit)."""
    if (raw_text is None) == (raw_sha256 is None):
        raise ValueError("serve raw_text oppure raw_sha256")
    if question is not None and question_sha256 is not None and sha256_text(question) != question_sha256:
        raise ValueError(f"{key}: domanda diversa dal suo question_sha256 nella generazione")
    checks = solution_checks(solution)
    rec = {"schema": SOLUTION_SCHEMA, "sid": make_sid(gen_run, key, completion), "gen_run": gen_run, "key": key,
           "completion": completion, "source_dataset": source_dataset, "answer": answer}
    if answer_letter is not None:
        rec["answer_letter"] = answer_letter
    rec.update({"question": question, "solution": solution, "raw_sha256": raw_sha256 or sha256_text(raw_text),
                "text_sha256": sha256_text(solution), "question_sha256": None if question is None else sha256_text(question),
                "prompt_sha256": prompt_sha256, "rewritten": rewritten,
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
    question = rec.get("question")
    if (question is None) != (rec.get("question_sha256") is None):
        problems.append("question_sha256 senza domanda o domanda senza hash")
    elif question is not None and rec["question_sha256"] != sha256_text(question):
        problems.append("domanda diversa dal suo question_sha256")
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


# ---------------------------------------------------------------- politica, certificati e ammissione

def policy_open(policy: dict) -> list[str]:
    """Scelte metodologiche ancora aperte nella politica (valori null o assenti). Finché ce n'è una, f2_certify.py non
    certifica nessun lotto: le decide l'utente (revisione tecnica, sezione 10), non il codice."""
    if policy.get("schema") != POLICY_SCHEMA:
        return [f"schema {policy.get('schema')!r}, atteso {POLICY_SCHEMA}"]
    thresholds = policy.get("thresholds") or {}
    out = [f"thresholds.{k}" for k, v in thresholds.items() if v is None] or ([] if thresholds else ["thresholds"])
    flag_policy = policy.get("flag_policy") or {}
    out += [f"flag_policy.{f}" for f in FLAGGED if flag_policy.get(f) not in FLAG_DECISIONS]
    if policy.get("frame_counts_as_error") not in (True, False):
        out.append("frame_counts_as_error")
    if not (policy.get("calibration") or {}).get("required_kinds"):
        out.append("calibration.required_kinds")
    return out


def record_policy(policy: dict) -> dict:
    """Regole per record del certificato: controlli che scartano e decisione su ogni segnalazione ("reject", "keep" o
    None se ancora aperta)."""
    flag_policy = policy.get("flag_policy") or {}
    return {"rejecting": list(REJECTING),
            "flags": {f: flag_policy.get(f) if flag_policy.get(f) in FLAG_DECISIONS else None for f in FLAGGED}}


def eligible(rec: dict, rules: dict) -> tuple[bool, str]:
    """(nella popolazione, motivo): il record supera i controlli che scartano e ogni sua segnalazione è decisa "keep"."""
    if rec["failed"]:
        return False, "controlli non superati: " + ", ".join(rec["failed"])
    rejected = [f for f in rec["flags"] if rules["flags"].get(f) == "reject"]
    if rejected:
        return False, "segnalazione scartata dalla politica: " + ", ".join(rejected)
    undecided = [f for f in rec["flags"] if rules["flags"].get(f) != "keep"]
    if undecided:
        return False, "segnalazione senza decisione: " + ", ".join(undecided)
    return True, "ok"


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
        rules = cert.get("record_policy") or {}
        if not set(REJECTING) <= set(rules.get("rejecting") or []):
            problems.append("record_policy senza i controlli che scartano")
        if any((rules.get("flags") or {}).get(f) not in FLAG_DECISIONS for f in FLAGGED):
            problems.append("record_policy con segnalazioni non decise")
    return problems


def admit(rec: dict, cert: dict | None, selection_sha256: str | None) -> tuple[bool, str]:
    """(ammessa, motivo). Ammessa solo se: record v2 coerente, controlli che scartano superati, risposta finale
    verificata, domanda salvata nella generazione, certificato ben formato e `certified` per questa stessa selezione,
    segnalazioni ammesse dalle regole per record del certificato, soluzione non esclusa né in quarantena, testo uguale
    a quello auditato. La risposta giusta è necessaria, ma da sola non ammette nulla. L'ammissione del problema
    (decontaminazione, audit dei problemi) la fa f2_build.py con --exclude."""
    if record_version(rec) != 2:
        return False, "record v1 senza sid né hash"
    problems = validate_record(rec)
    if problems:
        return False, "record incoerente: " + "; ".join(problems)
    if rec["failed"]:
        return False, "controlli non superati: " + ", ".join(rec["failed"])
    if rec.get("answer_correct") is not True:
        return False, "risposta finale non verificata"
    if rec.get("question") is None:
        return False, "domanda non salvata nella generazione"
    if cert is None:
        return False, "nessun certificato"
    problems = validate_certificate(cert)
    if problems:
        return False, "certificato non valido: " + "; ".join(problems)
    if cert["decision"] != "certified":
        return False, "lotto non certificato"
    if not selection_sha256 or cert["population"]["selection_sha256"] != selection_sha256:
        return False, "certificato di un'altra selezione"
    ok, why = eligible(rec, cert["record_policy"])
    if not ok:
        return False, why
    sid, sha = rec["sid"], rec["text_sha256"]
    audited = {e[0]: e[1] for e in cert["audited"]}
    if sid in audited and audited[sid] != sha:
        return False, "testo cambiato dopo l'audit"
    if any(e[0] == sid for e in cert["excluded"]):
        return False, "esclusa dall'audit"
    if any(e[0] == sid for e in cert["quarantined"]):
        return False, "in quarantena"
    return True, "ammessa"


# ---------------------------------------------------------------- verifica nel training

def _project_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def check_f2_shards(shards: list[str]) -> dict:
    """Catena del gate per i tar di una sorgente f2_solution: manifest f2-build/2 nella loro cartella (scripts/f2_build.py),
    certificato e samples.jsonl ancora con lo sha256 registrato, certificato ben formato, `certified` e della stessa
    selezione, tar della sorgente uguali a quelli del build. {"certified": False, "why": ...} se un anello manca."""
    folder = Path(shards[0]).parent
    man_path = folder / "manifest.json"
    if not man_path.is_file():
        return {"certified": False, "why": f"{folder}: manifest.json mancante"}
    man = json.loads(man_path.read_text())
    if man.get("schema") != BUILD_SCHEMA:
        return {"certified": False, "why": f"{folder}: shard costruiti senza gate (manifest senza schema {BUILD_SCHEMA})"}
    cert_path = _project_path(man["certificate"]["path"])
    if not cert_path.is_file():
        return {"certified": False, "why": f"certificato mancante: {cert_path}"}
    if sha256_file(cert_path) != man["certificate"]["sha256"]:
        return {"certified": False, "why": f"{cert_path}: certificato cambiato dopo il build"}
    cert = json.loads(cert_path.read_text())
    problems = validate_certificate(cert)
    if problems:
        return {"certified": False, "why": f"certificato non valido: {problems}"}
    if cert["decision"] != "certified":
        return {"certified": False, "why": "lotto non certificato"}
    if cert["population"]["selection_sha256"] != man["selection_sha256"]:
        return {"certified": False, "why": "certificato di un'altra selezione"}
    samples = folder / "samples.jsonl"
    if not samples.is_file() or sha256_file(samples) != man["samples_file"]["sha256"]:
        return {"certified": False, "why": f"{samples}: mancante o diverso da quello del build"}
    names, built = sorted(Path(s).name for s in shards), sorted(r["name"] for r in man["shards"])
    if names != built:
        return {"certified": False, "why": f"tar della sorgente diversi da quelli del build ({len(names)} contro {len(built)})"}
    return {"certified": True, "why": None, "manifest_sha256": sha256_file(man_path),
            "certificate_sha256": man["certificate"]["sha256"], "selection_sha256": man["selection_sha256"],
            "samples_sha256": man["samples_file"]["sha256"], "samples": man["samples"]}


def f2_data_certification(specs: list[dict], shards: list[list[str]]) -> list[dict]:
    """Per ogni sorgente f2_solution della config (nell'ordine di make_sources): catena del gate verificata oppure un
    `uncertified_reason` esplicito (ablation di ricerca su dati non certificati, registrata nel manifest del run).
    ValueError se manca l'una e l'altro: un training F2 non usa dati non certificati in silenzio."""
    out = []
    for spec, files in zip(specs, shards, strict=True):
        if spec["kind"] != "f2_solution":
            continue
        chain = check_f2_shards(files)
        reason = spec.get("uncertified_reason")
        if not chain["certified"] and not (isinstance(reason, str) and reason.strip()):
            raise ValueError(f"sorgente {spec['name']}: shard F2 non certificati ({chain['why']}). Per un'ablation di ricerca "
                             "su questi dati serve un uncertified_reason esplicito nella sorgente, in una copia della config")
        out.append({"name": spec["name"], **chain, "uncertified_reason": reason})
    return out
