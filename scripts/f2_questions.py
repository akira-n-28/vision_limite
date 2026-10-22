#!/usr/bin/env python3
"""F2: ricostruzione non distruttiva della domanda di training nei record di generazioni precedenti al passo 4.

Da eseguire in un job CPU (slurm/f2_questions.sbatch). Le generazioni storiche non salvavano la domanda vista da Violetto,
quindi i loro record v2 (f2_backfill.py) hanno question None e non arrivano al build. La domanda è una funzione
deterministica di dati a revisione fissa e del codice del job che ha scritto la parte:
- MAVIS: limite_vl.mavis.f2_problem(meta)["question"] sugli shard caption;
- MMFineReason: limite_vl.mmfine.load_texts (domanda normalizzata) sui parquet dello snapshot.
Lo script la ricalcola con il codice storico (git archive del commit del job, eseguito in un processo a parte), mai con
quello attuale e mai per somiglianza, e classifica ogni record:
- exact: commit del job noto, file del codice della domanda puliti in quel job (log di provenienza), dati verificati
  per sha256 contro il loro manifest e, con --tokenizer, numero di token del prompt ricostruito uguale a prompt_tokens
  salvato nella parte. Solo questi ricevono question, question_sha256 e question_origin "reconstructed";
- inferred: ricalcolata con il commit più vicino, ma un anello manca (file della domanda modificati o non tracciati nel
  job, dati non verificati): la domanda va solo in questions_inferred.jsonl, il record resta senza;
- conflict: il prompt ricostruito non ha il numero di token salvato (la ricostruzione è smentita);
- missing: problema non trovato nei dati o codice storico non eseguibile.
Uscite in --out-dir (nuova o vuota): selected.v2q.jsonl (tutti i record, nello stesso ordine, domanda solo per gli
exact), questions_inferred.jsonl, questions_issues.jsonl, questions.json (conteggi per stato e per job, commit e blob
del codice usato, verifica dei dati, sha256 di input e uscite, identità del codice). Nessun certificato: la nuova
selezione ha un altro sha256 e va auditata da capo.

    srun $RUN_PYTHON -u -B scripts/f2_questions.py --selected outputs/f2_backfill/f2-mavis2k/selected.v2.jsonl \
        --gen outputs/f2_gen/f2-mavis2k --gen-logs outputs/logs/lvl-f2-gen-29832.out \
        --mavis-shards 'data/shards/mavis-geometry/9c80fff3-caption/*.tar' --out-dir outputs/f2_questions/f2-mavis2k \
        [--tokenizer MODELLO] [--data-manifest downloads/manifests/p3.tsv]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# file da cui dipende la domanda di training (e il prompt, per il controllo dei token)
QUESTION_CODE = {"mavis": ("src/limite_vl/mavis.py", "scripts/f2_generate.py"),
                 "mmfine": ("src/limite_vl/mmfine.py", "scripts/f2_generate.py")}
PROMPT_CODE = ("src/limite_vl/f2.py",)
_HEADER = re.compile(r"^=== provenienza job (\d+) ===$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_STATUS = re.compile(r"^(?: M|M |MM|A |AM| D|D |\?\?) (.+)$")
_PART = re.compile(r"^parte (\d+)/(\d+)")
_KEY = re.compile(rb'^\{"key": ("(?:[^"\\]|\\.)*")')
_TOKENS = re.compile(rb'"prompt_tokens": (\d+)')


# ---------------------------------------------------------------- aiutante (processo con il codice storico)

def helper(args) -> None:
    """Gira con sys.path sul codice estratto da git: stampa una riga JSON per chiave (domanda, prompt del ponte)."""
    sys.path.insert(0, str(Path(args.code_root) / "src"))
    from limite_vl import f2 as hf2  # noqa: E402  (versione storica)

    keys = set(Path(args.keys).read_text().split())
    clean = bool(args.clean_caption)

    def prompt(question: str, caption: str) -> str:
        try:
            return hf2.bridge_prompt(question, caption, clean=clean)
        except TypeError:  # versioni senza clean_caption
            if clean:
                raise
            return hf2.bridge_prompt(question, caption)

    if args.kind == "mavis":
        from limite_vl.mavis import f2_problem  # noqa: E402
        for shard in sorted(glob.glob(args.source)):
            with tarfile.open(shard, "r|") as tar:
                for member in tar:
                    if not (member.isfile() and member.name.endswith(".json")):
                        continue
                    meta = json.loads(tar.extractfile(member).read())
                    key = f"mavis-{meta['id']}"
                    if key in keys:
                        p = f2_problem(meta)
                        out = {"key": key, "question": None, "prompt": None} if p is None else \
                            {"key": key, "question": p["question"], "prompt": prompt(p["question"], p["caption"])}
                        print(json.dumps(out, ensure_ascii=False), flush=True)
    else:
        from limite_vl.mmfine import load_texts, scan_problems  # noqa: E402
        rows = [r for r in scan_problems([Path(p) for p in args.source.split(os.pathsep)]) if r["key"] in keys]
        for p in load_texts(rows):
            print(json.dumps({"key": p["key"], "question": p["question"], "prompt": prompt(p["question"], p["caption"])},
                             ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- provenienza delle parti

def parse_log(path: Path) -> dict:
    """Job, commit, file modificati o non tracciati e parti scritte, dal log di un job di f2_generate.py."""
    job, commit, dirty, parts, in_prov = None, None, [], [], False
    for line in path.read_text(errors="replace").splitlines():
        if m := _HEADER.match(line):
            job, in_prov = m.group(1), True
            continue
        if line.startswith("=== fine provenienza"):
            in_prov = False
        elif in_prov and commit is None and _COMMIT.match(line):
            commit = line
        elif in_prov and (m := _STATUS.match(line)):
            dirty.append(m.group(1).strip())
        elif m := _PART.match(line):
            parts.append(int(m.group(1)) - 1)
    if not (job and commit):
        raise SystemExit(f"{path}: provenienza del job (commit) non trovata")
    return {"log": str(path), "job": job, "commit": commit, "dirty": sorted(dirty), "parts": parts}


def git(*cmd: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *cmd], capture_output=True, text=True, check=True).stdout.strip()


def blob(commit: str, path: str) -> str | None:
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{commit}:{path}"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def code_commit(job: dict, files: tuple[str, ...]) -> tuple[str | None, str]:
    """(commit da usare, motivo): quello del job se contiene i file; se mancano (file non tracciati quando il job è
    partito), il primo commit successivo che li contiene, come ipotesi."""
    if all(blob(job["commit"], f) for f in files):
        return job["commit"], "commit del job"
    later = [c for c in git("rev-list", "--reverse", "--ancestry-path", f"{job['commit']}..HEAD").split() if all(blob(c, f) for f in files)]
    return (later[0], "primo commit successivo con i file (ipotesi)") if later else (None, "file assenti dalla storia")


def extract(commit: str, dest: Path) -> Path:
    """Albero src/ e scripts/ del commit in dest/<commit> (git archive letto con tarfile, senza shell)."""
    root = dest / f"code-{commit[:12]}"
    if root.exists():
        return root
    data = subprocess.run(["git", "-C", str(ROOT), "archive", commit, "src", "scripts"], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(root, filter="data")
    return root


def verify_data(kind: str, source: list[str], data_manifest: Path | None) -> dict:
    """sha256 dei dati contro il loro manifest: per MAVIS manifest.json degli shard, per MMFineReason il manifest di
    download (colonne url, dest, size_bytes, hash, label: il nome del file è l'ultima parte dell'url)."""
    from limite_vl.provenance import sha256_file

    if kind == "mavis":
        man_path = Path(source[0]).parent / "manifest.json"
        if not man_path.is_file():
            return {"verified": False, "why": f"manca {man_path}"}
        expected = {r["name"]: r["sha256"] for r in json.loads(man_path.read_text()).get("shards", [])}
        manifest = {"path": str(man_path), "sha256": sha256_file(man_path)}
    else:
        if not data_manifest:
            return {"verified": False, "why": "nessun --data-manifest per i parquet"}
        expected = {}
        for line in data_manifest.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            url, _, _, h, *_ = line.split("\t")
            expected[url.rsplit("/", 1)[-1]] = h.removeprefix("sha256:")
        manifest = {"path": str(data_manifest), "sha256": sha256_file(data_manifest)}
    bad = [Path(f).name for f in source if expected.get(Path(f).name) != sha256_file(f)]
    return {"verified": not bad, "why": f"{len(bad)} file diversi dal manifest (es. {bad[:3]})" if bad else None,
            "files": len(source), "manifest": manifest}


def scan_parts(gen: Path, keys: set[str]) -> tuple[dict, dict, dict]:
    """Per le chiavi della selezione: parte e prompt_tokens (letti senza decodificare le generazioni), sha256 delle parti."""
    where, tokens, digests = {}, {}, {}
    for part in sorted(gen.glob("part-*.jsonl")):
        idx = int(part.stem.split("-")[1])
        h = hashlib.sha256()
        with open(part, "rb") as fh:
            for line in fh:
                h.update(line)
                m = _KEY.match(line)
                key = json.loads(m.group(1)) if m else None
                if key in keys:
                    where[key] = idx
                    head = line[:line.find(b'"completions"')]
                    t = _TOKENS.search(head)
                    tokens[key] = int(t.group(1)) if t else None
        digests[part.name] = h.hexdigest()
    return where, tokens, digests


# ---------------------------------------------------------------- principale

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--helper", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--code-root", help=argparse.SUPPRESS)
    ap.add_argument("--kind", choices=("mavis", "mmfine"), help=argparse.SUPPRESS)
    ap.add_argument("--keys", help=argparse.SUPPRESS)
    ap.add_argument("--source", help=argparse.SUPPRESS)
    ap.add_argument("--clean-caption", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--selected", type=Path, help="selected.v2.jsonl di f2_backfill.py (o di f2_select.py)")
    ap.add_argument("--gen", type=Path, help="cartella della generazione (plan.json e parti)")
    ap.add_argument("--gen-logs", nargs="+", type=Path, help="log dei job di f2_generate.py che hanno scritto le parti")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--mavis-shards", help="glob degli shard caption di MAVIS-Geometry")
    src.add_argument("--parquet", nargs="+", help="parquet sft dello snapshot di MMFineReason (quelli di plan.json)")
    ap.add_argument("--data-manifest", type=Path, help="manifest di download con gli sha256 dei parquet")
    ap.add_argument("--no-data-check", action="store_true", help="non verificare i dati (nessun record risulta exact)")
    ap.add_argument("--tokenizer", type=Path, help="snapshot di Violetto: controllo del numero di token del prompt")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    if args.helper:
        helper(args)
        return
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f2_questions.py va eseguito in un job SLURM")
    if not (args.selected and args.gen and args.gen_logs and args.out_dir and (args.mavis_shards or args.parquet)):
        sys.exit("servono --selected, --gen, --gen-logs, --out-dir e --mavis-shards o --parquet")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        sys.exit(f"{args.out_dir} non è vuota: le uscite non si sovrascrivono")
    from limite_vl.f2_records import gen_run_name, read_selected, record_version, validate_record
    from limite_vl.provenance import HashingTextWriter, code_identity, file_entry, sha256_text, write_new_json

    t0 = time.time()
    kind = "mavis" if args.mavis_shards else "mmfine"
    files = QUESTION_CODE[kind]
    plan = json.loads((args.gen / "plan.json").read_text())
    recs = list(read_selected(args.selected))
    if any(record_version(r) != 2 or r["gen_run"] != gen_run_name(args.gen) for r in recs):
        sys.exit("--selected deve contenere record v2 della generazione --gen")
    keys = {r["key"] for r in recs}
    where, saved_tokens, part_digests = scan_parts(args.gen, keys)
    print(f"{len(recs)} record, {len(keys)} problemi, {len(where)} ritrovati nelle parti ({time.time() - t0:.0f} s)", flush=True)

    # provenienza di ogni parte
    jobs, part_job = [parse_log(p) for p in args.gen_logs], {}
    for j in jobs:
        dirty_code = [f for f in j["dirty"] if f in files]
        commit, why = code_commit(j, files)
        j.update({"dirty_question_code": dirty_code, "code_commit": commit, "code_commit_reason": why,
                  "code_blobs": {f: blob(commit, f) for f in files + PROMPT_CODE} if commit else {},
                  "code_proven": commit == j["commit"] and not dirty_code})
        for i in j["parts"]:
            if i in part_job:
                sys.exit(f"parte {i} scritta da due job ({part_job[i]['job']} e {j['job']})")
            part_job[i] = j
    data = {"verified": False, "why": "--no-data-check"} if args.no_data_check else \
        verify_data(kind, sorted(glob.glob(args.mavis_shards)) if args.mavis_shards else args.parquet, args.data_manifest)
    print(f"dati: {data} ({time.time() - t0:.0f} s)", flush=True)

    # domande con il codice storico di ogni commit
    tok = None
    if args.tokenizer:
        from transformers import Qwen2TokenizerFast  # come Qwen2Tokenizer, senza l'auto_map della config (CLAUDE.md)
        tok = Qwen2TokenizerFast.from_pretrained(str(args.tokenizer))
    args.out_dir.mkdir(parents=True)
    work = args.out_dir / "_code"
    found: dict[str, dict] = {}
    for commit in sorted({j["code_commit"] for j in jobs if j["code_commit"]}):
        want = sorted(k for k in keys if k in where and part_job.get(where[k], {}).get("code_commit") == commit)
        if not want:
            continue
        root = extract(commit, work)
        keys_file = work / f"keys-{commit[:12]}.txt"
        keys_file.write_text("\n".join(want) + "\n")
        source = args.mavis_shards or os.pathsep.join(args.parquet)
        r = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--helper", "--code-root", str(root), "--kind", kind,
                            "--keys", str(keys_file), "--source", source, "--clean-caption", str(int(bool(plan.get("clean_caption"))))],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ""})
        if r.returncode:
            print(f"codice {commit[:12]} non eseguibile: {r.stderr[-800:]}", flush=True)
            continue
        for line in r.stdout.splitlines():
            d = json.loads(line)
            if tok is not None and d["prompt"] is not None:
                text = tok.apply_chat_template([{"role": "user", "content": d["prompt"]}], tokenize=False, add_generation_prompt=True)
                d["prompt_tokens"] = len(tok(text, add_special_tokens=False)["input_ids"])
            found[d["key"]] = {**d, "commit": commit}
        print(f"codice {commit[:12]}: {len(want)} problemi chiesti, {sum(found.get(k, {}).get('commit') == commit for k in want)} "
              f"ricostruiti ({time.time() - t0:.0f} s)", flush=True)

    # stato per record e scrittura
    sel_w = HashingTextWriter(args.out_dir / "selected.v2q.jsonl")
    inf_w = HashingTextWriter(args.out_dir / "questions_inferred.jsonl")
    iss_w = HashingTextWriter(args.out_dir / "questions_issues.jsonl")
    status, by_job = Counter(), defaultdict(Counter)
    for rec in recs:
        key = rec["key"]
        j = part_job.get(where.get(key))
        f = found.get(key)
        reasons = []
        if rec.get("question") is not None:
            st = "already_saved"
        elif key not in where:
            st, reasons = "missing", ["problema non trovato nelle parti"]
        elif j is None:
            st, reasons = "missing", [f"parte {where[key]} senza log del job"]
        elif f is None or f["question"] is None:
            st, reasons = "missing", ["domanda non ricostruita (problema assente dai dati o codice non eseguibile)"]
        else:
            tokens_match = None if "prompt_tokens" not in f or saved_tokens.get(key) is None else f["prompt_tokens"] == saved_tokens[key]
            if not j["code_proven"]:
                reasons.append(f"codice della domanda non dimostrato nel job {j['job']} ({j['code_commit_reason']}; "
                               f"modificati {j['dirty_question_code']})")
            if not data["verified"]:
                reasons.append(f"dati non verificati ({data['why']})")
            st = "conflict" if tokens_match is False else ("inferred" if reasons else "exact")
            if st == "conflict":
                reasons.append(f"prompt ricostruito di {f['prompt_tokens']} token, salvati {saved_tokens[key]}")
        status[st] += 1
        by_job[j["job"] if j else "?"][st] += 1
        out = dict(rec)
        if st == "exact":
            out.update({"question": f["question"], "question_sha256": sha256_text(f["question"]), "question_origin": "reconstructed",
                        "question_provenance": {"job": j["job"], "commit": f["commit"], "code_blobs": j["code_blobs"],
                                                "data": data.get("manifest"), "prompt_tokens": saved_tokens.get(key),
                                                "prompt_tokens_match": tokens_match}})
            problems = validate_record(out)
            if problems:
                raise RuntimeError(f"{rec['sid']}: record ricostruito incoerente: {problems}")
        elif st in ("inferred", "conflict") and f:
            inf_w.write_json_line({"sid": rec["sid"], "key": key, "status": st, "question": f["question"],
                                   "question_sha256": sha256_text(f["question"]), "commit": f["commit"], "reasons": reasons})
        if st not in ("exact", "already_saved"):
            iss_w.write_json_line({"sid": rec["sid"], "key": key, "status": st, "reasons": reasons})
        sel_w.write_json_line(out)
    outputs = {"selected_v2q": sel_w.close(), "inferred": inf_w.close(), "issues": iss_w.close()}
    write_new_json(args.out_dir / "questions.json", {
        "schema": "f2-questions/1", "job_id": os.environ.get("SLURM_JOB_ID"), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kind": kind, "selected": file_entry(args.selected), "gen": {"dir": str(args.gen), "parts": part_digests},
        "jobs": jobs, "data": data, "tokenizer": str(args.tokenizer) if args.tokenizer else None,
        "counts": dict(status), "counts_by_job": {k: dict(v) for k, v in sorted(by_job.items())}, "outputs": outputs,
        "code": code_identity(),
        "notes": ["exact: codice del job dimostrato, dati verificati, token del prompt non smentiti; solo questi hanno la domanda",
                  "inferred e conflict: domanda solo in questions_inferred.jsonl, mai nei record",
                  "nessun certificato: selected.v2q.jsonl ha un altro sha256 e va auditata da capo"]})
    print(json.dumps({"counts": dict(status), "by_job": {k: dict(v) for k, v in by_job.items()}}, ensure_ascii=False))
    print(f"fine in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
