#!/usr/bin/env python3
"""Test CPU dei passi 4-7 del refactoring F2 (contratto + gate): provenienza della generazione (f2_generate.py), audit v2
delle soluzioni (stem_audit_export.py --f2-v2, stem_audit_round.py), certificato (f2_certify.py), gate del build
(f2_build.py) e verifica nel training (f2_records.f2_data_certification, stream, data.f2_solution_sample).

Giro completo su fixture minuscole in stile MAVIS (shard caption finti, generazione finta con la domanda salvata, risposte
degli agenti simulate): niente GPU, niente agenti, niente dati veri. Gira in slurm/tests_cpu.sbatch."""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from PIL import Image, ImageDraw  # noqa: E402

from limite_vl.f2_records import check_f2_shards, f2_data_certification, read_selected  # noqa: E402
from limite_vl.mavis import f2_problem  # noqa: E402
from limite_vl.provenance import sha256_file, sha256_text  # noqa: E402
from limite_vl.stem_audit import note_covers, quote_covers  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID", "test"), VLLM_PLUGINS="limite")
    return subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / script), *map(str, args)], env=env,
                          capture_output=True, text=True)


def png(i: int) -> bytes:
    im = Image.new("RGB", (96, 64), "white")
    ImageDraw.Draw(im).rectangle([10 + i, 10, 80, 50], outline="black")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def write_tar(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def mavis_meta(i: int) -> dict:
    return {"id": f"d0-{i}", "depth": 0, "choices": ["25", "20", "30", "35"], "answer_index": 0,
            "text_en": {"caption": "ABCD is a square. Side AB is 5 units long.", "diagram_generation": "",
                        "vision_dominant_question": "What is the area of ABCD?",
                        "CoT_reasoning": "Side AB of the square is 5, so the area is 5 * 5 = 25."}}


def test_quote_covers():
    text = "<think>Place \\(D\\) at the origin. The side \\(DF\\) is vertical, so DF = 12.</think>"
    start = text.index("vertical")
    end = start + len("vertical")
    check("citazione: delimitatori LaTeX e spazi ignorati", quote_covers(text, "the side DF is  vertical.", start, end))
    check("citazione: altrove non copre il punto", not quote_covers(text, "Place D at the origin", start, end))
    check("citazione: pezzi separati da '...'", quote_covers(text, "Place D ... DF is vertical", start, end))
    check("citazione: parafrasi non conta", not quote_covers(text, "DF is a vertical side", start, end))
    check("citazione: troppo corta non conta", not quote_covers(text, "ve", start, end))
    sol = "\\begin{aligned} P &= 12+15+16+12+15 \\\\ &= 66 \\end{aligned}"
    s = sol.index("16")
    check("nota sul ragionamento che riproduce il punto alterato", note_covers(
        sol, "The displayed sum 12+15+16+12+15 = 66 is wrong: the 16 should be 12.", s, s + 2))
    check("nota che cita solo il numero: non basta", not note_covers(sol, "the 16 should be 12", s, s + 2))


def test_generate(tmp: Path, shards: str):
    import f2_generate as fg

    p = {"question": "What is the area?"}
    f = fg.problem_fields(p, "<|im_start|>user\nFigure: ...")
    check("generazione: domanda e hash del prompt nella riga", f["question"] == p["question"]
          and f["question_sha256"] == sha256_text(p["question"]) and f["prompt_sha256"] == sha256_text("<|im_start|>user\nFigure: ..."))
    prov = tmp / "provenance.json"
    inv = {"model_files": {"model.safetensors": "a" * 64}, "sampling": {"temperature": 0.6}, "vllm": "x"}
    fg.start_provenance(prov, inv, {"job_id": "1", "parts": []}, ["part-00000.jsonl"])
    fg.start_provenance(prov, inv, {"job_id": "2", "parts": []}, [])
    got = json.loads(prov.read_text())
    check("provenienza: un job per avvio, parti precedenti senza provenienza", [j["job_id"] for j in got["jobs"]] == ["1", "2"]
          and got["parts_without_provenance"] == ["part-00000.jsonl"])
    try:
        fg.start_provenance(prov, {**inv, "sampling": {"temperature": 1.0}}, {"job_id": "3", "parts": []}, [])
        stopped = False
    except SystemExit:
        stopped = True
    check("provenienza: ripresa con campionamento diverso fermata", stopped and len(json.loads(prov.read_text())["jobs"]) == 2)

    out = tmp / "gen-plan"
    base = ["--model", tmp / "nessun-modello", "--mavis-shards", shards, "--select", "4", "--samples", "2", "--out", out]
    r1 = run("f2_generate.py", *base, "--plan-only")
    plan = (out / "plan.json").read_bytes() if r1.returncode == 0 else b""
    r2 = run("f2_generate.py", *base, "--plan-only")
    check("--plan-only su fixture e ripresa della stessa cartella", r1.returncode == 0 and r2.returncode == 0
          and (out / "plan.json").read_bytes() == plan and (out / "keys.txt").is_file(), (r1.stderr + r2.stderr)[-600:])
    check("--plan-only: niente provenance.json (nessuna parte generata)", not (out / "provenance.json").exists())
    r3 = run("f2_generate.py", *base[:-4], "--samples", "3", "--out", out, "--plan-only")
    check("ripresa con un piano diverso rifiutata", r3.returncode != 0 and "piano diverso" in r3.stderr, r3.stderr[-300:])


def make_generation(gen: Path, metas: list[dict]) -> None:
    gen.mkdir()
    rows = []
    for m in metas:
        p = f2_problem(m)
        i = int(m["id"].split("-")[1])
        quote = 'It says: "Figure: ABCD is a square." ' if i == 8 else ""
        texts = [f"<think>{quote}Side AB is 5 so the area is 5*5 = 25. The side AB is horizontal. Variante {c}.</think> "
                 "The area is \\boxed{25}." for c in "abc"]
        comps = [{"text": t, "finish_reason": "stop", "tokens": 30} for t in texts]
        comps.append({"text": "<think>no</think> \\boxed{20}", "finish_reason": "stop", "tokens": 5})
        rows.append({"key": p["key"], "source_dataset": "MAVIS-Geometry d0", "answer": p["answer_text"], "qwen4b_pass_rate": None,
                     "prompt_tokens": 50, "question": p["question"], "question_sha256": sha256_text(p["question"]),
                     "prompt_sha256": sha256_text("prompt " + p["key"]), "completions": comps, "answer_letter": p["answer_letter"]})
    (gen / "part-00000.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def answer_batches(audit: Path, prefix: str, judge) -> None:
    for b in sorted((audit / "batches").glob(f"{prefix}-*.json")):
        tasks = json.loads(b.read_text())
        (audit / "answers" / b.name).write_text(json.dumps([judge(t["id"]) for t in tasks]))


def test_pipeline(tmp: Path, shards: str, metas: list[dict]):
    gen = tmp / "gen-gate"
    make_generation(gen, metas)
    sel = tmp / "sel"
    r = run("f2_select.py", "--gen", gen, "--out", sel, "--keep", "2", "--workers", "1")
    check("selezione v2 con la domanda salvata", r.returncode == 0, r.stderr[-600:])
    if r.returncode:
        return
    recs = list(read_selected(sel / "selected.v2.jsonl"))
    check("selezione: domanda e hash del prompt copiati nei record", all(x["question"] and x["question_sha256"] == sha256_text(x["question"])
                                                                        and x["prompt_sha256"] for x in recs) and len(recs) == 16)

    audit = tmp / "audit-v2"
    r = run("stem_audit_export.py", "--out", audit, "--seed", "7", "--mavis-shard", "x", "--synth-test", "x", "--f2-v2",
            "--f2-selected", sel / "selected.v2.jsonl", "--mavis-shards", shards, "--f2-n", "6", "--f2-cal-measure", "2",
            "--f2-cal-layout", "2", "--f2-negatives", "1", "--f2-recheck-frac", "0.5", "--f2-batch", "4")
    check("export dell'audit v2", r.returncode == 0, r.stderr[-800:])
    if r.returncode:
        return
    items = {it["fid"]: it for it in json.loads((audit / "_gold" / "items_f2.json").read_text())}
    truth = json.loads((audit / "_gold" / "calibration_f2.json").read_text())
    man = json.loads((audit / "manifest.json").read_text())
    kinds = sorted(t["kind"] for t in truth.values())
    check("audit v2: unità soluzione con sid e hash, calibrazione per tipo", man["audit_version"] == 2
          and kinds == ["layout", "layout", "measure", "measure", "negative"]
          and all(it["sid"] and it["text_sha256"] == sha256_text(it["solution"]) for it in items.values() if not it["calibration"])
          and man["f2_v2"]["selection"]["sha256"] == sha256_file(sel / "selected.v2.jsonl"), (kinds, man.get("f2_v2", {}).get("design")))
    check("audit v2: istruzioni v2 registrate con lo sha256", man["f2_v2"]["instructions"]["FA"]["path"].endswith("f2_solution_v2.md"))
    spans_ok = all(items[f]["solution"][t["start"]:t["end"]] == t["changed"] for f, t in truth.items() if t["kind"] != "negative")
    check("audit v2: punto alterato registrato esattamente", spans_ok)
    check("audit v2: domande salvate usate, nessuna differenza da quelle ricostruite", man["f2_v2"]["design"]["question_mismatch"] == 0)

    real = sorted(f for f, it in items.items() if not it["calibration"])
    bad_real = next(f for f in real if items[f]["key"] != "mavis-d0-8")  # nella popolazione anche con prompt_quote: reject
    layout = sorted(f for f, t in truth.items() if t["kind"] == "layout")

    def judge(fid: str, who: str = "A") -> dict:
        it, t = items[fid], truth.get(fid, {})
        ans = {"id": fid, "figure_claims": 4, "visual_ok": True, "frame_ok": True, "issues": [], "reasoning_valid": True,
               "reasoning_note": "ok", "relies_on_unseen": False}
        if fid == bad_real:
            ans |= {"visual_ok": False, "issues": [{"quote": "The side AB is horizontal", "type": "false", "kind": "layout"}]}
        elif t.get("kind") in ("measure", "layout"):
            s, e = t["start"], t["end"]
            quote = it["solution"][max(0, s - 12):e + 2]
            if fid == layout[-1]:  # esclusa, ma la citazione non indica il punto alterato: non attribuibile
                quote = "so the area is"
            ans |= {"visual_ok": False, "issues": [{"quote": quote, "type": "false", "kind": t["kind"]}]}
        return ans

    (audit / "answers").mkdir(exist_ok=True)
    answer_batches(audit, "FA", judge)
    answer_batches(audit, "FB", lambda f: judge(f, "B"))
    r = run("stem_audit_round.py", "--dir", audit)
    fc = [json.loads(b.read_text()) for b in (audit / "batches").glob("FC-*.json")]
    rechecked = [t["id"] for b in fc for t in b]
    # approvate all'unanimità: le 5 soluzioni vere senza errori e il controllo negativo; la metà, per eccesso
    check("turni v2: ricontrollo da FC di una parte delle approvazioni unanimi", r.returncode == 0 and len(rechecked) == 3
          and bad_real not in rechecked, (rechecked, r.stderr[-400:]))
    answer_batches(audit, "FC", judge)
    r = run("stem_audit_round.py", "--dir", audit)
    r = run("stem_audit_round.py", "--dir", audit, "--final")
    rows = {json.loads(x)["fid"]: json.loads(x) for x in (audit / "audit_f2.jsonl").read_text().splitlines()} if r.returncode == 0 else {}
    check("turni v2: finale con sid, frame_ok e consenso+controllo", r.returncode == 0 and rows[bad_real]["verdict"] == "exclude"
          and rows[bad_real]["sid"] == items[bad_real]["sid"] and "frame_ok" in rows[real[1]]
          and any(rows[f]["visual_ok"]["how"] == "consenso+controllo" for f in rechecked), r.stderr[-500:])

    policy = json.loads((ROOT / "config" / "audit" / "f2_policy.json").read_text())
    decided = {**policy, "thresholds": {k: 1.0 for k in policy["thresholds"]}, "frame_counts_as_error": True,
               "flag_policy": {"prompt_quote": "reject", "text_before_think": "keep"}}
    pol = tmp / "policy-test.json"
    pol.write_text(json.dumps(decided))
    cert_dir = tmp / "cert"
    r = run("f2_certify.py", "--audit", audit, "--selection", sel / "selected.v2.jsonl", "--policy", pol, "--out", cert_dir)
    cert = json.loads((cert_dir / "certificate.json").read_text()) if r.returncode == 0 else {}
    check("certificato: politica decisa, audit completo -> certified", cert.get("decision") == "certified", (r.stderr[-600:], cert.get("reasons")))
    if not cert:
        return
    sens = cert["sensitivity"]
    check("calibrazione attribuibile: misure 2/2, posizioni 1/2 (citazione fuori dal punto alterato non conta)",
          (sens["measure"]["k"], sens["measure"]["n"], sens["layout"]["k"], sens["layout"]["n"]) == (2, 2, 1, 2)
          and sens["layout"]["esclusa_per_qualunque_motivo"] == 2, sens)
    check("controlli negativi: nessun falso allarme", (cert["false_alarms"]["k"], cert["false_alarms"]["n"]) == (0, 1))
    check("certificato: esclusa per (sid, hash), popolazione senza le citazioni del prompt (reject)",
          [items[bad_real]["sid"], items[bad_real]["text_sha256"]] in cert["excluded"] and cert["population"]["solutions"] == 14
          and cert["rates"]["claim_visual"] is not None)
    r = run("f2_certify.py", "--audit", audit, "--selection", sel / "selected.v2.jsonl", "--policy", pol, "--out", cert_dir)
    check("certificato: cartella non vuota rifiutata", r.returncode != 0 and "non è vuota" in r.stderr)
    repo_cert = tmp / "cert-repo"
    r = run("f2_certify.py", "--audit", audit, "--selection", sel / "selected.v2.jsonl", "--policy",
            ROOT / "config" / "audit" / "f2_policy.json", "--out", repo_cert)
    rc = json.loads((repo_cert / "certificate.json").read_text()) if r.returncode == 0 else {}
    check("certificato con la politica del repository: rejected per le scelte aperte", rc.get("decision") == "rejected"
          and any("scelte della politica ancora aperte" in x for x in rc.get("reasons", [])), r.stderr[-400:])

    # ---- gate del build ----
    built = tmp / "shards-ok"
    r = run("f2_build.py", "--selected", sel / "selected.v2.jsonl", "--certificate", cert_dir / "certificate.json",
            "--mavis-shards", shards, "--out", built, "--per-shard", "5")
    bm = json.loads((built / "manifest.json").read_text()) if r.returncode == 0 else {}
    lines = [json.loads(x) for x in (built / "samples.jsonl").read_text().splitlines()] if bm else []
    check("build: solo le soluzioni ammesse dal gate", bm.get("schema") == "f2-build/2" and bm["samples"] == 13 == len(lines)
          and bm["not_admitted"] == {"segnalazione scartata dalla politica: prompt_quote": 2, "esclusa dall'audit": 1}
          and items[bad_real]["sid"] not in {x["sid"] for x in lines}, (r.stderr[-600:], bm.get("not_admitted")))
    check("build: sample_id dall'indice della generazione, testo e domanda con hash", lines and all(
        x["sample_id"] == f"{x['sid'].split('/')[1]}-g{x['sid'].split('/')[2]}" for x in lines))
    again = tmp / "shards-again"
    run("f2_build.py", "--selected", sel / "selected.v2.jsonl", "--certificate", cert_dir / "certificate.json",
        "--mavis-shards", shards, "--out", again, "--per-shard", "5")
    check("build: a parità di input stessi sample_id e text_sha256",
          (again / "samples.jsonl").is_file() and (again / "samples.jsonl").read_bytes() == (built / "samples.jsonl").read_bytes())
    r = run("f2_build.py", "--selected", sel / "selected.v2.jsonl", "--certificate", repo_cert / "certificate.json",
            "--mavis-shards", shards, "--out", tmp / "shards-rej")
    check("build: lotto non certificato rifiutato", r.returncode != 0 and "lotto non certificato" in r.stderr
          and not (tmp / "shards-rej").exists())
    tampered = tmp / "tampered.v2.jsonl"
    rows_t = (sel / "selected.v2.jsonl").read_text().splitlines()
    first = json.loads(rows_t[0])
    first["solution"] = first["solution"].replace("5*5", "5 * 5")
    tampered.write_text("\n".join([json.dumps(first, ensure_ascii=False)] + rows_t[1:]) + "\n")
    r = run("f2_build.py", "--selected", tampered, "--certificate", cert_dir / "certificate.json", "--mavis-shards", shards,
            "--out", tmp / "shards-tampered")
    check("build: testo manomesso dopo il certificato rifiutato", r.returncode != 0 and "un'altra selezione" in r.stderr)
    dup_dir = tmp / "mavis-dup"
    dup_dir.mkdir()
    shutil.copy(sorted(Path(shards).parent.glob("*.tar"))[0], dup_dir / "a.tar")
    shutil.copy(sorted(Path(shards).parent.glob("*.tar"))[0], dup_dir / "b.tar")
    r = run("f2_build.py", "--selected", sel / "selected.v2.jsonl", "--certificate", cert_dir / "certificate.json",
            "--mavis-shards", dup_dir / "*.tar", "--out", tmp / "shards-dup")
    check("build: chiave ripetuta negli shard rifiutata", r.returncode != 0 and "ripetuta" in r.stderr, r.stderr[-300:])

    # ---- verifica nel training ----
    tars = sorted(str(p) for p in built.glob("*.tar"))
    chain = check_f2_shards(tars)
    check("training: catena del certificato verificata", chain["certified"] and chain["samples"] == 13, chain)
    old = tmp / "shards-old"
    old.mkdir()
    shutil.copy(tars[0], old / Path(tars[0]).name)
    (old / "manifest.json").write_text(json.dumps({"selected": "x", "shards": [{"name": Path(tars[0]).name}]}))
    spec = {"name": "bridge", "kind": "f2_solution", "shards": "x", "weight": 1.0}
    try:
        f2_data_certification([spec], [[str(old / Path(tars[0]).name)]])
        refused = False
    except ValueError as e:
        refused = "uncertified_reason" in str(e)
    check("training: shard senza gate e senza motivo rifiutati", refused)
    entry = f2_data_certification([{**spec, "uncertified_reason": "riproduzione dell'ablation del 2026-10-16"}],
                                  [[str(old / Path(tars[0]).name)]])
    check("training: con uncertified_reason accettati e annotati", entry[0]["certified"] is False
          and entry[0]["uncertified_reason"].startswith("riproduzione"))
    ok = f2_data_certification([spec], [tars])
    check("training: shard del gate accettati senza motivo", ok[0]["certified"] and ok[0]["uncertified_reason"] is None)
    other = tmp / "cert-copy.json"
    shutil.copy(cert_dir / "certificate.json", other)
    b2 = tmp / "shards-copy"
    run("f2_build.py", "--selected", sel / "selected.v2.jsonl", "--certificate", other, "--mavis-shards", shards, "--out", b2)
    other.write_text(other.read_text().replace('"certified"', '"certified" ', 1))
    check("training: certificato cambiato dopo il build", not check_f2_shards(sorted(str(p) for p in b2.glob("*.tar")))["certified"])

    from limite_vl.data import f2_solution_sample, iter_shard
    from limite_vl.stream import Source, SourceReader

    key, files = next(iter_shard(tars[0]))
    sample = f2_solution_sample(key, files)
    check("campione F2 del gate: domanda salvata nel messaggio", "What is the area" in json.dumps(sample.messages[0]["content"]))
    meta = json.loads(files["json"])
    bad = dict(files, json=json.dumps({**meta, "solution": meta["solution"] + " "}).encode())
    try:
        f2_solution_sample(key, bad)
        caught = False
    except ValueError:
        caught = True
    check("campione F2 con testo alterato dopo il build: errore", caught)
    excluded_problem = meta["problem"]
    src = Source("bridge", "f2_solution", tars, 1.0, frozenset({excluded_problem}))
    reader = SourceReader(src, 0, 0, 1, 0)
    seen = [reader.next()[0] for _ in range(13 - sum(x["sample_id"].startswith(excluded_problem + "-g") for x in lines))]
    check("stream: lista per problema applicata ai campioni F2", seen and not any(k.startswith(excluded_problem + "-g") for k in seen)
          and len(set(seen)) == len(seen), seen[:3])


if __name__ == "__main__":
    tmp = ROOT / "outputs" / "tmp" / f"test-f2-gate-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        metas = [mavis_meta(i) for i in range(1, 9)]
        sdir = tmp / "mavis-shards"
        sdir.mkdir()
        write_tar(sdir / "mavis-geometry-caption-00000.tar",
                  [(f"{m['id']}.{ext}", data) for i, m in enumerate(metas, 1)
                   for ext, data in (("json", json.dumps(m).encode()), ("png", png(i)))])
        shards = str(sdir / "*.tar")
        test_quote_covers()
        test_generate(tmp, shards)
        test_pipeline(tmp, shards, metas)
    finally:
        shutil.rmtree(tmp)
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
