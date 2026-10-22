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
# Origine della domanda di training di un record: "generation" = salvata da f2_generate.py nella riga del problema;
# "reconstructed" = ricostruita da scripts/f2_questions.py con il codice storico dimostrato (commit del job, file puliti)
# e i dati verificati per hash. Una domanda solo inferita non entra nei record (resta None).
QUESTION_ORIGINS = ("generation", "reconstructed")
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
                "question_origin": None if question is None else "generation", "prompt_sha256": prompt_sha256,
                "rewritten": rewritten,
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
    if question is not None and rec.get("question_origin") not in QUESTION_ORIGINS:
        problems.append(f"origine della domanda {rec.get('question_origin')!r} non ammessa")
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


def _rate_problems(name: str, r, k_key: str = "k", n_key: str = "n") -> list[str]:
    """Un tasso {k, n, rate, lo, hi} (o {errors, claims, ...}) coerente: interi, 0 <= k <= n, n > 0, rate = k/n,
    0 <= lo <= rate <= hi <= 1."""
    if not isinstance(r, dict):
        return [f"{name}: tasso mancante"]
    k, n, rate, lo, hi = r.get(k_key), r.get(n_key), r.get("rate"), r.get("lo"), r.get("hi")
    if not (isinstance(k, int) and isinstance(n, int) and 0 <= k <= n and n > 0):
        return [f"{name}: conteggi non validi ({k_key}={k!r}, {n_key}={n!r})"]
    if not all(isinstance(x, (int, float)) for x in (rate, lo, hi)):
        return [f"{name}: rate, lo o hi mancanti"]
    if abs(rate - k / n) > 1e-9 or not (0 <= lo <= rate + 1e-12 and rate <= hi + 1e-12 and hi <= 1):
        return [f"{name}: rate {rate} e intervallo [{lo}, {hi}] incoerenti con {k}/{n}"]
    return []


def _certified_problems(cert: dict) -> list[str]:
    """Coerenza numerica di un certificato `certified`, verificabile dal solo certificato: politica chiusa e uguale a
    quella incorporata, soglie e regole per record ricavate dalla politica, tassi e calibrazione validi, limiti
    corretti ricalcolati e sotto le soglie, esiti per soluzione coerenti fra loro. Che selezione, audit e politica
    citati siano davvero quelli lo verifica f2_certify.py --verify rileggendo gli artefatti."""
    from .stem_audit import corrected_upper

    problems = []
    if cert.get("reasons"):
        problems.append(f"certified con motivi di rifiuto: {cert['reasons'][:3]}")
    if cert.get("stale"):
        problems.append("certified con item stale")
    if cert.get("retroactive") is not False or (cert.get("audit") or {}).get("version") != 2:
        problems.append("solo un audit v2 non retroattivo può certificare")
    policy = (cert.get("policy") or {}).get("content")
    if not isinstance(policy, dict):
        return problems + ["politica incorporata mancante"]
    if policy_open(policy) or (cert.get("policy") or {}).get("open"):
        problems.append(f"scelte della politica aperte: {policy_open(policy) or cert['policy']['open']}")
    thresholds, rates, corrected = cert.get("thresholds") or {}, cert.get("rates") or {}, cert.get("corrected_upper") or {}
    if not thresholds or any(v is None for v in thresholds.values()):
        problems.append("certificato senza soglie")
    if thresholds != policy.get("thresholds"):
        problems.append("soglie diverse da quelle della politica incorporata")
    if cert.get("record_policy") != record_policy(policy):
        problems.append("record_policy diversa da quella della politica incorporata")
    required = (policy.get("calibration") or {}).get("required_kinds") or []
    sens = cert.get("sensitivity") or {}
    for kind in required:
        problems += _rate_problems(f"sensitivity.{kind}", sens.get(kind))
    if problems:
        return problems
    s_lo = min(sens[k]["lo"] for k in required)
    if abs(corrected.get("sensitivity_lower", -1) - s_lo) > 1e-12:
        problems.append("corrected_upper.sensitivity_lower diverso dal minimo della calibrazione richiesta")
    for key, thr in thresholds.items():
        r = rates.get(key)
        problems += _rate_problems(f"rates.{key}", r, *(("errors", "claims") if key == "claim_visual" else ("k", "n")))
        if problems:
            continue
        expect = corrected_upper(r["hi"], s_lo) if key in ("solution_visual", "claim_visual") else r["hi"]
        got = corrected.get(key)
        if not isinstance(got, (int, float)) or abs(got - expect) > 1e-9:
            problems.append(f"corrected_upper.{key} = {got!r}, ricalcolato {expect:.6g}")
        elif got > thr:
            problems.append(f"{key}: limite corretto {got:.4g} oltre la soglia {thr}")
    audited = cert.get("audited") or []
    sids = [e[0] for e in audited]
    if len(set(sids)) != len(sids):
        problems.append("sid ripetuti in audited")
    if any(len(e) < 3 or e[2] not in ("include", "exclude", "quarantine") for e in audited):
        problems.append("audited senza verdetto valido")
    else:
        by = {v: {(e[0], e[1]) for e in audited if e[2] == v} for v in ("exclude", "quarantine")}
        if {tuple(e[:2]) for e in cert.get("excluded") or []} != by["exclude"]:
            problems.append("excluded diverso dagli item auditati con verdetto exclude")
        if {tuple(e[:2]) for e in cert.get("quarantined") or []} != by["quarantine"]:
            problems.append("quarantined diverso dagli item auditati con verdetto quarantine")
        counts = {v: sum(e[2] == v for e in audited) for v in ("include", "exclude", "quarantine")}
        if {k: v for k, v in counts.items() if v} != {k: v for k, v in (rates.get("verdicts") or {}).items() if v}:
            problems.append("rates.verdicts diverso dagli esiti per soluzione")
        if rates.get("sample_in_population") != len(audited):
            problems.append("rates.sample_in_population diverso dagli item auditati")
    for sid in sids:
        try:
            parse_sid(sid)
        except ValueError:
            problems.append(f"sid non valido in audited: {sid!r}")
            break
    solutions = (cert.get("population") or {}).get("solutions")
    if not (isinstance(solutions, int) and solutions >= len(audited) and solutions > 0):
        problems.append("population.solutions non valido")
    return problems


def validate_certificate(cert: dict) -> list[str]:
    """Problemi di un certificato f2-cert/1 (vuoto = valido). Per ogni certificato: struttura, riferimenti e liste di
    (sid, text_sha256). Per un certificato `certified` anche la coerenza numerica della decisione (_certified_problems):
    un certificato che dichiara certified con un limite sopra la soglia, una metrica mancante, una calibrazione non
    eseguita o scelte aperte non vale."""
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
    sel = pop.get("selection") or {}
    if sel and sel.get("sha256") != pop.get("selection_sha256"):
        problems.append("population.selection.sha256 diverso da population.selection_sha256")
    if cert.get("decision") == "certified":
        rules = cert.get("record_policy") or {}
        if not set(REJECTING) <= set(rules.get("rejecting") or []):
            problems.append("record_policy senza i controlli che scartano")
        if any((rules.get("flags") or {}).get(f) not in FLAG_DECISIONS for f in FLAGGED):
            problems.append("record_policy con segnalazioni non decise")
        problems += _certified_problems(cert)
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


def _samples_problems(folder: Path, man: dict, shards: list[str]) -> str | None:
    """Coerenza fra samples.jsonl, manifest e tar: righe, sample_id e sid unici, campioni per tar uguali a quelli del
    manifest, nomi dei membri di ogni tar (solo le intestazioni, senza leggere le immagini) uguali ai campioni attesi."""
    import tarfile

    per_tar: dict[str, set[str]] = {}
    ids, sids = set(), set()
    with open(folder / "samples.jsonl", encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            if s["sample_id"] in ids or s["sid"] in sids:
                return f"samples.jsonl: sample_id o sid ripetuto ({s['sample_id']})"
            if not all(_HEX64.match(str(s.get(k, ""))) for k in ("text_sha256", "question_sha256", "image_sha256")):
                return f"samples.jsonl: hash mancanti per {s['sample_id']}"
            ids.add(s["sample_id"])
            sids.add(s["sid"])
            per_tar.setdefault(s["tar"], set()).add(s["sample_id"])
    if len(ids) != man["samples"] or sum(r["samples"] for r in man["shards"]) != man["samples"]:
        return f"samples.jsonl ({len(ids)} righe), manifest ({man['samples']}) e tar del manifest non tornano"
    for path in shards:
        name = Path(path).name
        row = next(r for r in man["shards"] if r["name"] == name)
        want = per_tar.get(name, set())
        if len(want) != row["samples"]:
            return f"{name}: {len(want)} campioni in samples.jsonl, {row['samples']} nel manifest"
        with tarfile.open(path) as tar:
            files = [m.name for m in tar.getmembers() if m.isfile()]
        members, n_json = {f.rsplit(".", 1)[0] for f in files}, sum(f.endswith(".json") for f in files)
        if members != want or n_json != len(want):
            return f"{name}: membri del tar diversi dai campioni di samples.jsonl"
    return None


def check_f2_shards(shards: list[str], verify_tars: bool = True) -> dict:
    """Catena del gate per i tar di una sorgente f2_solution: manifest f2-build/2 nella loro cartella (scripts/f2_build.py),
    certificato e samples.jsonl ancora con lo sha256 registrato, certificato valido (validate_certificate, coerenza
    numerica compresa), `certified` e della stessa selezione, tar della sorgente uguali a quelli del build per nome e
    per sha256 (lettura completa: qualche secondo per GB, una volta per job), samples.jsonl coerente con manifest e
    membri dei tar. {"certified": False, "why": ...} se un anello manca. verify_tars=False salta solo gli sha256 dei tar
    (per i test di velocità, mai nel training)."""
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
    if verify_tars:
        by_name = {r["name"]: r for r in man["shards"]}
        for path in shards:
            row = by_name[Path(path).name]
            if Path(path).stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
                return {"certified": False, "why": f"{Path(path).name}: contenuto diverso da quello del build (sha256)"}
    why = _samples_problems(folder, man, shards)
    if why:
        return {"certified": False, "why": why}
    return {"certified": True, "why": None, "manifest_sha256": sha256_file(man_path),
            "certificate_sha256": man["certificate"]["sha256"], "selection_sha256": man["selection_sha256"],
            "samples_sha256": man["samples_file"]["sha256"], "samples": man["samples"], "tars_verified": verify_tars}


MIN_REASON = 20


def f2_data_certification(specs: list[dict], shards: list[list[str]]) -> list[dict]:
    """Per ogni sorgente f2_solution della config (nell'ordine di make_sources): catena del gate verificata oppure un
    `uncertified_reason` esplicito (ablation di ricerca su dati non certificati, registrata nel manifest del run).
    Fail-closed: ValueError se manca l'una e l'altro, se il motivo è troppo corto per dire qualcosa (meno di MIN_REASON
    caratteri) e anche se una sorgente certificata porta un uncertified_reason, così una config non può dichiarare
    insieme le due cose. Il motivo non rende mai certificata una sorgente: `certified` resta quello della catena."""
    out = []
    for spec, files in zip(specs, shards, strict=True):
        if spec["kind"] != "f2_solution":
            continue
        chain = check_f2_shards(files)
        reason = spec.get("uncertified_reason")
        if reason is not None and not (isinstance(reason, str) and len(reason.strip()) >= MIN_REASON):
            raise ValueError(f"sorgente {spec['name']}: uncertified_reason deve spiegare l'ablation (almeno {MIN_REASON} caratteri)")
        if chain["certified"] and reason is not None:
            raise ValueError(f"sorgente {spec['name']}: shard certificati con un uncertified_reason: togliere il motivo")
        if not chain["certified"] and reason is None:
            raise ValueError(f"sorgente {spec['name']}: shard F2 non certificati ({chain['why']}). Per un'ablation di ricerca "
                             "su questi dati serve un uncertified_reason esplicito nella sorgente, in una copia della config")
        out.append({"name": spec["name"], **chain, "uncertified_reason": reason})
    return out


def run_certification_summary(entries: list[dict]) -> dict:
    """Riepilogo per il manifest del run: certificato solo se ogni sorgente f2_solution lo è; altrimenti il run è
    un'ablation su dati non certificati, con l'elenco delle sorgenti e dei motivi."""
    uncertified = [{"name": e["name"], "why": e["why"], "uncertified_reason": e["uncertified_reason"]}
                   for e in entries if not e["certified"]]
    return {"f2_data_certified": bool(entries) and not uncertified, "f2_sources": len(entries), "uncertified_sources": uncertified}
