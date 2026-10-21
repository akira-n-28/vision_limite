#!/usr/bin/env python3
"""F2, ponte testuale: Violetto risolve i problemi di MMFineReason leggendo la descrizione della figura.

Da eseguire SOLO in un job SLURM con GPU (slurm/f2_generate.sbatch; VLLM_PLUGINS=limite). Legge i problemi
dagli shard di MMFineReason (scripts/shard_parquet.py) oppure direttamente dai parquet dello snapshot
(--parquet, solo le colonne di testo: non servono le immagini), con domanda, risposta verificata e descrizione
di Qwen3-VL-235B. Salta quelli senza domanda, risposta o descrizione, le chiavi di --exclude (decontaminazione,
domande non in inglese), i problemi già generati nelle cartelle di --exclude-gen e le sorgenti di
--exclude-sources (quelle che la verifica non sa giudicare o che il ponte non risolve quasi mai, job 29598).
Le chiavi di --include si generano comunque, oltre alle --select scelte, anche se stanno in --exclude-gen (problemi
del primo lotto risolti ma con tutte le soluzioni scartate dai filtri: f2_select.py --keys-only); --exclude e
--exclude-sources valgono anche per loro.
Con --select N sceglie N problemi, --easy-frac fra quelli che Qwen3-VL-4B risolve sempre (Violetto ne risolve il
57% contro il 20% degli altri, job 29498) e il resto fra gli altri, in ordine pseudo-casuale fissato dal seed,
e per ognuno campiona --samples soluzioni con vLLM (temperature e top_p dai default del checkpoint, come il
valutatore). Prompt: limite_vl.f2.bridge_prompt; con --clean-caption la descrizione perde la riga del tipo di
immagine, i titoli delle sezioni e la sezione delle ambiguità (limite_vl.f2.clean_caption, job 29614 e 29626).

Con --mavis-shards i problemi sono di MAVIS-Geometry (limite_vl.mavis): domanda con la struttura della figura (frasi
senza misure della didascalia riscritta) e la richiesta di vision_dominant, opzioni A-D, risposta esatta, ponte con la
didascalia del motore riscritta dove era letteralmente falsa; --select in parti uguali fra le profondità 0-3.

Con --no-caption il prompt è solo la domanda con l'istruzione (lo stesso testo degli esempi di training, senza
descrizione): le soluzioni non possono copiare la descrizione, e servono a misurare quanti problemi si risolvono
senza (prova solo domanda del 2026-10-19). --select 0 con --include genera solo le chiavi di --include.

Uscita in --out, a parti da --part-size problemi: part-NNNNN.jsonl con, per problema, chiave, sorgente,
risposta, pass_rate di Qwen3-VL-4B e le generazioni (testo, finish_reason, token). Ogni parte si scrive in
`.tmp` e si rinomina; un nuovo avvio con la stessa --out salta le parti già scritte (il piano dei problemi
dipende solo da shard, esclusioni e --limit, e si controlla con plan.json). Non inizia una parte che, con la
durata della più lunga già fatta nel job, finirebbe a meno di --margin-min minuti dalla scadenza: una parte
interrotta dal limite di tempo andrebbe persa (job 29514). Con il piano scrive keys.txt, le chiavi dei problemi,
per le esclusioni dei lotti successivi. Le soluzioni si scelgono poi con scripts/f2_select.py.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.f2 import BRIDGE_TEMPLATE, INSTRUCTION, bridge_prompt, generated_keys, training_question  # noqa: E402
from limite_vl.mavis import f2_problem as mavis_problem  # noqa: E402
from limite_vl.mmfine import choose, load_texts, rank, scan_problems  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from train_f1 import seconds_left_in_job  # noqa: E402


def load_problems(shards: list[str], exclude: set[str], exclude_sources: set[str],
                  limit: int | None) -> tuple[list[dict], dict]:
    problems, skipped = [], {"missing": 0, "excluded": 0, "excluded_source": 0}
    for shard in shards:
        for key, files in iter_shard(shard):
            meta = json.loads(files["json"])
            if not (meta.get("question") and meta.get("answer") and meta.get("caption")):
                skipped["missing"] += 1
                continue
            if key in exclude:
                skipped["excluded"] += 1
                continue
            if meta.get("source_dataset") in exclude_sources:
                skipped["excluded_source"] += 1
                continue
            problems.append({"key": key, "source_dataset": meta.get("source_dataset"), "question": meta["question"],
                             "answer": meta["answer"], "caption": meta["caption"], "qwen4b_pass_rate": meta.get("pass_rate")})
            if limit and len(problems) >= limit:
                return problems, skipped
    return problems, skipped


def load_mavis(shards: list[str], exclude: set[str], select: int, seed: int) -> tuple[list[dict], dict]:
    """Problemi MAVIS-Geometry (limite_vl.mavis.f2_problem: domanda con la struttura della figura e le opzioni, ponte
    con la didascalia riscritta), select in parti uguali fra le profondità in ordine pseudo-casuale fissato dal seed."""
    import tarfile

    by_depth, skipped = {}, {"not_rewritable": 0, "excluded": 0}
    for shard in shards:
        with tarfile.open(shard, "r|") as tar:
            for member in tar:
                if not (member.isfile() and member.name.endswith(".json")):
                    continue
                p = mavis_problem(json.loads(tar.extractfile(member).read()))
                if p is None:
                    skipped["not_rewritable"] += 1
                elif p["key"] in exclude:
                    skipped["excluded"] += 1
                else:
                    by_depth.setdefault(p["depth"], []).append(p)
    per = select // max(1, len(by_depth))
    problems = []
    for depth in sorted(by_depth):
        for p in sorted(by_depth[depth], key=lambda p: rank(p["key"], seed))[:per]:
            problems.append({"key": p["key"], "source_dataset": f"MAVIS-Geometry d{depth}", "question": p["question"],
                             "answer": p["answer_text"], "answer_letter": p["answer_letter"], "caption": p["caption"],
                             "qwen4b_pass_rate": None})
    skipped["available"] = sum(len(v) for v in by_depth.values())
    return problems, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="snapshot di Violetto")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--shards", help="glob degli shard di MMFineReason")
    src.add_argument("--parquet", nargs="+", type=Path, help="parquet dello snapshot di MMFineReason")
    src.add_argument("--mavis-shards", help="glob degli shard caption di MAVIS-Geometry (con --select)")
    ap.add_argument("--select", type=int, help="con --parquet: quanti problemi scegliere")
    ap.add_argument("--easy-frac", type=float, default=0.5, help="frazione con pass_rate di Qwen3-VL-4B = 1.0")
    ap.add_argument("--exclude", nargs="*", default=[], help="file di chiavi da saltare (decontaminazione)")
    ap.add_argument("--exclude-gen", nargs="*", type=Path, default=[],
                    help="cartelle di generazioni precedenti: i loro problemi non si rigenerano")
    ap.add_argument("--exclude-sources", nargs="*", default=[], help="sorgenti (colonna source) da saltare")
    ap.add_argument("--include", type=Path, help="con --parquet: file di chiavi da generare comunque, oltre a --select")
    ap.add_argument("--plan-only", action="store_true",
                    help="scrive plan.json e keys.txt e si ferma (job CPU: controllo del piano prima della generazione)")
    ap.add_argument("--clean-caption", action="store_true", help="descrizioni senza titoli delle sezioni")
    ap.add_argument("--no-caption", action="store_true",
                    help="solo la domanda, senza descrizione della figura (prova solo domanda, regola sulla qualità dei dati)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--part-size", type=int, default=2048)
    ap.add_argument("--limit", type=int, help="solo i primi N problemi (prova)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin-min", type=float, default=20,
                    help="minuti di scarto fra la fine prevista di una parte e la scadenza del job")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f2_generate.py va eseguito in un job SLURM")
    if os.environ.get("VLLM_PLUGINS") != "limite":
        sys.exit("impostare VLLM_PLUGINS=limite (il plugin registra LimiteForCausalLM)")
    t_start = time.time()
    left = seconds_left_in_job()
    deadline = t_start + left - args.margin_min * 60 if left else None

    exclude: set[str] = set()
    for path in args.exclude:
        exclude.update(line.strip() for line in open(path) if line.strip())
    n_gen_excluded = 0
    for run_dir in args.exclude_gen:
        if run_dir.resolve() == args.out.resolve():
            sys.exit("--exclude-gen non può contenere la cartella --out")
        keys = generated_keys(run_dir)
        n_gen_excluded += len(keys)
        exclude |= keys
    exclude_sources = set(args.exclude_sources)
    include: set[str] = set()
    if args.include:
        if not args.parquet:
            sys.exit("--include richiede --parquet")
        include = {line.strip() for line in open(args.include) if line.strip()}
        include -= {k for path in args.exclude for k in (line.strip() for line in open(path)) if k}
    if args.parquet:
        missing = [str(p) for p in args.parquet if not p.is_file()]
        if missing:
            sys.exit(f"parquet mancanti: {missing[:5]}")
        inputs = [p.name for p in args.parquet]
        scanned = scan_problems(args.parquet)
        chosen = choose(scanned, len(scanned) if args.select is None else args.select, args.easy_frac, args.seed, exclude | include,
                        exclude_sources)
        forced = [p for p in scanned if p["key"] in include and p["source_dataset"] not in exclude_sources]
        chosen = sorted(chosen + forced, key=lambda p: (p["file"], p["row"]))
        problems = load_texts(chosen)
        skipped = {"scanned": len(scanned), "excluded": sum(p["key"] in exclude for p in scanned),
                   "excluded_source": sum(p["source_dataset"] in exclude_sources for p in scanned),
                   "chosen": len(chosen), "included": len(forced), "include_not_found": len(include) - len(
                       {p["key"] for p in scanned if p["key"] in include}), "empty_question": len(chosen) - len(problems),
                   "easy": sum(p["qwen4b_pass_rate"] == 1.0 for p in problems)}
    elif args.mavis_shards:
        shards = sorted(glob.glob(args.mavis_shards))
        if not shards or not args.select:
            sys.exit(f"--mavis-shards vuole shard esistenti e --select (shard: {len(shards)})")
        inputs = [Path(s).name for s in shards]
        problems, skipped = load_mavis(shards, exclude, args.select, args.seed)
    else:
        shards = sorted(glob.glob(args.shards))
        if not shards:
            sys.exit(f"nessuno shard per {args.shards}")
        inputs = [Path(s).name for s in shards]
        problems, skipped = load_problems(shards, exclude, exclude_sources, args.limit)
    skipped["exclude_gen_keys"] = n_gen_excluded
    parts = [problems[i:i + args.part_size] for i in range(0, len(problems), args.part_size)]
    args.out.mkdir(parents=True, exist_ok=True)
    plan = {"inputs": inputs, "select": args.select, "easy_frac": args.easy_frac,
            "exclude": [str(p) for p in args.exclude], "limit": args.limit,
            "part_size": args.part_size, "samples": args.samples, "max_tokens": args.max_tokens, "seed": args.seed,
            "template": "<question>" + INSTRUCTION if args.no_caption else BRIDGE_TEMPLATE, "keys_digest": hashlib.sha256("\n".join(p["key"] for p in problems).encode()).hexdigest()}
    if args.exclude_gen or exclude_sources:  # solo se usate: i piani dei lotti precedenti restano validi
        plan["exclude_gen"] = [str(p) for p in args.exclude_gen]
        plan["exclude_sources"] = sorted(exclude_sources)
    if args.clean_caption:
        plan["clean_caption"] = True
    if args.no_caption:
        plan["no_caption"] = True
    if args.include:
        plan["include"] = str(args.include)
    plan_path = args.out / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if {k: old.get(k) for k in plan} != plan:
            sys.exit(f"piano diverso da quello in {plan_path}: usare un'altra cartella --out")
    else:
        plan_path.write_text(json.dumps({**plan, "problems": len(problems), "parts": len(parts), "skipped": skipped}, indent=1))
    keys_path = args.out / "keys.txt"
    if not keys_path.exists():
        keys_path.write_text("".join(f"{k}\n" for k in sorted(p["key"] for p in problems)))
    todo = [i for i in range(len(parts)) if not (args.out / f"part-{i:05d}.jsonl").exists()]
    print(f"{len(problems)} problemi in {len(parts)} parti, da fare {len(todo)}; saltati {skipped} "
          f"({time.time() - t_start:.0f} s)", flush=True)
    if not todo or args.plan_only:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", seed=args.seed, gpu_memory_utilization=0.90)
    tok = llm.get_tokenizer()
    default = llm.get_default_sampling_params()
    params = SamplingParams(n=args.samples, temperature=default.temperature, top_p=default.top_p, top_k=default.top_k,
                            max_tokens=args.max_tokens, seed=args.seed)
    print(f"engine pronto in {time.time() - t_start:.0f} s; temperature {params.temperature}, top_p {params.top_p}, "
          f"max_tokens {params.max_tokens}, n {params.n}", flush=True)

    durations: list[float] = []
    for i in todo:
        expected = max(durations, default=0.0)
        if deadline and time.time() + expected > deadline:
            print(f"fermo prima della scadenza del job: parti rimaste {len([j for j in todo if j >= i])}, "
                  f"una parte richiede circa {expected / 60:.0f} min", flush=True)
            break
        t0 = time.time()
        part = parts[i]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": training_question(p["question"]) if args.no_caption
              else bridge_prompt(p["question"], p["caption"], clean=args.clean_caption)}],
            tokenize=False, add_generation_prompt=True) for p in part]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        n_out = 0
        path = args.out / f"part-{i:05d}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as fh:
            for p, out in zip(part, outputs):
                completions = [{"text": c.text, "finish_reason": c.finish_reason, "tokens": len(c.token_ids)}
                               for c in out.outputs]
                n_out += sum(c["tokens"] for c in completions)
                row = {"key": p["key"], "source_dataset": p["source_dataset"], "answer": p["answer"],
                       "qwen4b_pass_rate": p["qwen4b_pass_rate"], "prompt_tokens": len(out.prompt_token_ids),
                       "completions": completions}
                if "answer_letter" in p:  # MAVIS: scelta multipla, vale la lettera o il contenuto dell'opzione
                    row["answer_letter"] = p["answer_letter"]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        dt = time.time() - t0
        durations.append(dt)
        print(f"parte {i + 1}/{len(parts)}: {len(part)} problemi, {n_out} token generati in {dt:.0f} s "
              f"({n_out / dt:.0f} token/s, {n_out / (len(part) * args.samples):.0f} token per soluzione)", flush=True)
    print(f"fine in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
