#!/usr/bin/env python3
"""Test CPU del manifest candidato di UniMER-1M (scripts/unimer_manifest.py; F0_CAMPIONE_CANDIDATO §6 punto 3), su shard,
liste storiche e snapshot fittizi costruiti qui: nessun dato reale. Solo in un job (slurm/tests_cpu.sbatch o
slurm/unimer_manifest.sbatch).

Il fixture copre: rietichettatura di un indice isolato e di graffe codificate, un errore certo (comando non reso), un
conflitto (stessa immagine, formule diverse), un duplicato (stessa immagine e formula), una formula lunga del test con
un'immagine diversa, due formule con l'immagine di un campione del test, una formula breve del test, righe ritagliate di
una formula e di una formula rietichettata, e 40 formule semplici. Controlla:
- categorie e stati di unimer-check/4 uguali a quelli di scripts/unimer_check.py;
- costruzione, scrittura e verifica col percorso del job (unimer_manifest.run): stati, gruppi, split, trasformazioni e
  genitori, evidenze, ranghi, conteggi, verifica dopo la scrittura;
- riproducibilità: due costruzioni danno righe identiche, e gli split dipendono dal seme;
- rifiuti prima della scrittura: chiave storica senza record o ripetuta, chiave in due shard, etichetta attesa diversa,
  stati incoerenti fra le liste, run4 o run2 non riprodotti, riga ritagliata senza formula o con righe diverse, shard
  con sha256 diverso, snapshot diverso."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import unimer_manifest as um  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.f1c import array_rows  # noqa: E402
from limite_vl.latex import effective_label, visible_tokens  # noqa: E402

FAILS = []
REV = "2343ddd963290469da36ca83e3a56c66e068add9"
SRC = f"wanderkid/UniMER_Dataset@{REV}"
LONG = "a+b+c+d+e+f+g+h+i+j+k"  # 21 token visibili
ARRAY = "\\begin{array}{l}a\\\\b\\end{array}"
ARRAY_CODED = "\\begin{array}{l}aḊ1Ḍ\\\\b\\end{array}"


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def png(n: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (32 + n % 50, 16 + n // 50), color=(n * 37) % 256).save(buf, format="PNG")
    return buf.getvalue()


def sample(key: str, label: str, image: bytes, subset: str = "train", **extra) -> tuple[str, bytes, dict]:
    with Image.open(io.BytesIO(image)) as im:
        w, h = im.size
    n_vis = max(1, len(visible_tokens(effective_label(label)[0])))
    meta = {"id": key, "subset": subset, "index": int(key.split("-")[1]), "label": label, "label_spaced": label,
            "orig_width": 10 * n_vis, "orig_height": 10, "width": w, "height": h, "max_patches": 1024,
            "format": "PNG", "source": SRC, **extra}
    return key, image, meta


def spec() -> dict:
    """Campioni per shard e liste storiche coerenti (a mano) con le regole di run4 e run2."""
    p_conf, p_dup, p_test = png(901), png(902), png(903)
    s0 = [sample("train-0000001", "x^{2}+1", png(1)), sample("train-0000002", "^{3}", png(2)),
          sample("train-0000003", "a\\footnote{b}", png(3)), sample("train-0000004", "a+b", p_conf),
          sample("train-0000005", "a-b", p_conf), sample("train-0000006", "y=mx", p_dup),
          sample("train-0000007", "y=mx", p_dup)]
    s1 = [sample("train-0000008", LONG, png(8)), sample("train-0000009", "w", p_test),
          sample("train-0000010", "w", p_test), sample("train-0000011", "z", png(11)),
          sample("train-0000012", ARRAY, png(12)), sample("train-0000013", ARRAY_CODED, png(13))]
    s1 += [sample(f"train-{100 + n:07d}", f"x_{{{n}}}+y", png(100 + n)) for n in range(40)]
    lines = [sample("train-0000012", ARRAY, png(12), rows=array_rows(ARRAY), lines=[[0, 5], [6, 10]]),
             sample("train-0000013", ARRAY_CODED, png(13), rows=array_rows(ARRAY_CODED), lines=[[0, 5], [6, 10]])]
    test = [sample("spe-0000001", LONG, png(951), "spe"), sample("spe-0000002", "q", p_test, "spe"),
            sample("spe-0000003", "z", png(953), "spe")]
    relabel = [{"key": "train-0000002", "label": "^{3}", "new_label": "3", "rule": "indice_isolato"},
               {"key": "train-0000013", "label": ARRAY_CODED, "new_label": effective_label(ARRAY_CODED)[0],
                "rule": "graffe_codificate"}]
    reasons = [{"key": "train-0000002", "state": "accepted", "categories": ["rietichettato_indice_isolato"],
                "label": "^{3}"},
               {"key": "train-0000003", "state": "excluded", "categories": ["comando_non_reso"], "label": "a\\footnote{b}"},
               {"key": "train-0000004", "state": "uncertain", "categories": ["conflitto_irrisolto"], "label": "a+b"},
               {"key": "train-0000005", "state": "uncertain", "categories": ["conflitto_irrisolto"], "label": "a-b"},
               {"key": "train-0000013", "state": "accepted", "categories": ["rietichettato_graffe_codificate"],
                "label": ARRAY_CODED}]
    return {"train": [("unimer-train-00000.tar", s0), ("unimer-train-00001.tar", s1)],
            "righe": [("unimer-lines-00000.tar", lines)], "test": [("unimer-test-spe.tar", test)],
            "excluded": ["train-0000003"], "uncertain": ["train-0000004", "train-0000005"], "relabel": relabel,
            "reasons": reasons, "matches": ["train-0000008", "train-0000009", "train-0000010", "train-0000011"],
            "contamination": ["train-0000008", "train-0000009", "train-0000010"],
            "duplicates": [{"val_key": "train-0000006", "immagine": 1}, {"val_key": "train-0000011", "target": 1}],
            "report4": {"conflict_groups": 1, "conflict_samples": 2}}


def write_tar(path: Path, items) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for key, image, meta in items:
            for name, data in ((f"{key}.png", image), (f"{key}.json", json.dumps(meta).encode())):
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))
    data = path.read_bytes()
    return {"name": path.name, "samples": len(items), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_fixture(tmp: Path, s: dict, tamper=None) -> argparse.Namespace:
    """Shard, manifest dello sharding, snapshot e liste storiche su disco; tamper(cartelle) altera i file dopo."""
    shards, check, overlap, dups = tmp / "shards", tmp / "run4", tmp / "run2", tmp / "dup"
    blobs, snap = tmp / "blobs", tmp / "snapshot"
    for d in (check, overlap, dups, blobs, snap):
        d.mkdir(parents=True)
    zips = {}
    for name, content in (("UniMER-1M.zip", b"zip finto"), ("UniMER-Test.zip", b"zip di test finto")):
        sha = hashlib.sha256(content).hexdigest()
        (blobs / sha).write_bytes(content)
        (snap / name).symlink_to(blobs / sha)
        zips[name] = (str(snap / name), sha)
    train = [write_tar(shards / n, items) for n, items in s["train"]]
    righe = [write_tar(shards / "lines" / n, items) for n, items in s["righe"]]
    test = [write_tar(shards / "test" / n, items) for n, items in s["test"]]
    common = {"source": SRC, "scale": 4.0, "max_patches": 1024, "tokens_per_symbol": 4.0, "dense_cap": 2304}
    (shards / "manifest.json").write_text(json.dumps({**common, "job_id": "prova", "shards": train,
                                                      "zip": zips["UniMER-1M.zip"][0],
                                                      "zip_sha256_expected": zips["UniMER-1M.zip"][1],
                                                      "test_zip": zips["UniMER-Test.zip"][0],
                                                      "test_zip_sha256_expected": zips["UniMER-Test.zip"][1]}))
    (shards / "lines" / "manifest.json").write_text(json.dumps({**common, "shards": righe}))
    (shards / "test" / "manifest.json").write_text(json.dumps({**common, "shards": test}))
    n_train = sum(len(items) for _, items in s["train"])
    (check / "report.json").write_text(json.dumps({
        "schema": "unimer-check/4", "job_id": "prova4", "shards": [{"sha256": t["sha256"]} for t in train],
        "samples": n_train, "quality_excluded": len(s["excluded"]), "uncertain": len(s["uncertain"]),
        "relabeled": len(s["relabel"]), "render_used": False, **s["report4"]}))
    (check / "quality_excluded.txt").write_text("".join(k + "\n" for k in s["excluded"]))
    (check / "uncertain.txt").write_text("".join(k + "\n" for k in s["uncertain"]))
    (check / "relabel.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in s["relabel"]))
    (check / "reasons.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in s["reasons"]))
    (overlap / "report.json").write_text(json.dumps({
        "schema": "unimer-test-overlap/2", "job_id": "prova2", "long": 20,
        "test_shards": [{"sha256": t["sha256"]} for t in test], "train_shards": [{"sha256": t["sha256"]} for t in train],
        "train_samples_matching": len(s["matches"]), "train_contamination_long": len(s["contamination"])}))
    (overlap / "train_matches_test.txt").write_text("".join(k + "\n" for k in sorted(s["matches"])))
    (overlap / "train_contamination_long.txt").write_text("".join(k + "\n" for k in sorted(s["contamination"])))
    (dups / "val_with_duplicates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in s["duplicates"]))
    if tamper:
        tamper({"shards": shards, "check": check, "overlap": overlap, "dup": dups, "snapshot": snap})
    return argparse.Namespace(shards=shards, check=check, overlap=overlap, duplicates=dups, out=tmp / "manifest",
                              report=tmp / "verifica", summary=tmp / "riepilogo.json", name="unimer-prova",
                              version="1", dev_fraction=0.25, seed=0, workers=1, probe=False)


def refused(s: dict, tamper=None, run: bool = False) -> str:
    """Il motivo del rifiuto della costruzione (o di run, con lo snapshot), vuoto se passa."""
    with tempfile.TemporaryDirectory() as tmp:
        args = write_fixture(Path(tmp), s, tamper)
        try:
            if run:
                um.run(args)
            else:
                um.build(args, um.shard_plan(args.shards))
        except um.Problem as exc:
            return str(exc) or "rifiutato"
    return ""


def test_sets():
    import unimer_check

    check("categorie certe e incerte uguali a quelle di unimer-check/4 (scripts/unimer_check.py)",
          um.CERTAIN == unimer_check.CERTAIN and um.UNCERTAIN == unimer_check.UNCERTAIN)


def test_build_and_verify():
    with tempfile.TemporaryDirectory() as tmp:
        args = write_fixture(Path(tmp), spec())
        summary = um.run(args)
        m = mf.load(args.out)
        recs = [r for rows in m.by_shard.values() for r in rows.values()]
        verification = json.loads((args.report / "verifica.json").read_text())
        saved = json.loads(args.summary.read_text())
    base = {r["record"]: r for r in recs if r["insieme"] == "train"}
    righe = {r["record"]: r for r in recs if r["insieme"] == "righe"}
    st = {k: r["stato"] for k, r in base.items()}
    check("costruzione: tutte le formule e le righe ritagliate, verifica dopo la scrittura superata",
          len(base) == 53 and len(righe) == 2 and verification["esito"] and summary["esito_verifica"]
          and saved["manifest_sha256"] == verification["manifest_sha256"],
          f"{verification['superati']}/{verification['totale']}: "
          f"{[c['controllo'] for c in verification['controlli'] if not c['esito']]}")
    check("stati da run4: errore certo escluso, conflitto in quarantena, rietichettature ammesse",
          st["train-0000003"] == "escluso" and st["train-0000004"] == st["train-0000005"] == "quarantena"
          and st["train-0000002"] == st["train-0000013"] == "ammesso" and st["train-0000001"] == "ammesso", st)
    check("stati da run2: stessa immagine del test esclusa per gruppo, formula lunga in quarantena, formula breve "
          "solo evidenza",
          st["train-0000009"] == st["train-0000010"] == "escluso" and st["train-0000008"] == "quarantena"
          and st["train-0000011"] == "ammesso"
          and {"lista": "unimer_overlap_run2", "esito": "formula_breve_del_test"} in base["train-0000011"]["evidenze"]
          and any("stessa immagine di UniMER-Test (spe-0000002)" in x for x in base["train-0000009"]["motivo"])
          and any(x.startswith("formula di UniMER-Test") for x in base["train-0000008"]["motivo"]), st)
    g = {k: r["gruppo"] for k, r in base.items()}
    check("gruppi per stessi byte del PNG: conflitto, duplicato e immagine del test insieme; formule diverse separate",
          g["train-0000004"] == g["train-0000005"] and g["train-0000006"] == g["train-0000007"]
          and g["train-0000009"] == g["train-0000010"] and g["train-0000001"] != g["train-0000002"]
          and len(set(g.values())) == 50)
    r12, r13 = righe["train-0000012"], righe["train-0000013"]
    check("righe ritagliate: stesso record_id, gruppo e split della formula, genitore con gli hash della formula",
          all(r["record_id"] == base[k]["record_id"] and r["gruppo"] == base[k]["gruppo"]
              and r["split"] == base[k]["split"] and r["genitori"][0]["json_sha256"] == base[k]["json_sha256"]
              and r["genitori"][0]["png_sha256"] == base[k]["immagini"][0]["sha256"]
              and r["trasformazione"]["nome"] == "righe" and r["trasformazione"]["stessa_immagine_del_genitore"]
              for k, r in righe.items()))
    check("righe ritagliate: ammesse con la formula ammessa, in quarantena se la formula è rietichettata",
          r12["stato"] == "ammesso" and r13["stato"] == "quarantena"
          and any("rietichettata" in x for x in r13["motivo"]))
    t2, t13 = base["train-0000002"]["trasformazione"], base["train-0000013"]["trasformazione"]
    check("rietichettature tracciate: regola, nuova etichetta e genitore con l'etichetta originale",
          t2["nome"] == "rietichettatura" and t2["regola"] == "indice_isolato"
          and t2["etichetta_sha256"] == hashlib.sha256(b"3").hexdigest()
          and base["train-0000002"]["genitori"][0]["etichetta_sha256"] == hashlib.sha256(b"^{3}").hexdigest()
          and t13["regola"] == "graffe_codificate" and base["train-0000001"]["trasformazione"] is None)
    splits = {}
    for r in recs:
        splits.setdefault(r["gruppo"], set()).add(r["split"])
    dev = [r for r in recs if r["split"] == "dev"]
    check("split: nessun gruppo in due split, dev presente, gruppi non ammessi in train",
          all(len(v) == 1 for v in splits.values()) and dev
          and all(base[k]["split"] == "train" for k in ("train-0000003", "train-0000009", "train-0000010")))
    check("ranghi del campione candidato: rango_f0 per tutti, rango_f0_sviluppo solo nel dev",
          all(r["rango_f0"] == um.f0_rank(0, r["gruppo"]) for r in recs)
          and all((r["rango_f0_sviluppo"] is not None) == (r["split"] == "dev") for r in recs))
    check("gold assente e compiti per insieme", all(r["gold"] is None for r in recs)
          and all(r["compiti"] == um.TASKS[r["insieme"]] for r in recs))
    check("riproduzione di run4 e run2 registrata nel riepilogo",
          saved["riproduzione_run4"]["reasons_uguali"] and saved["riproduzione_run4"]["relabel_uguali"]
          and saved["riproduzione_run4"]["run2"]["uguali_ai_file"]
          and saved["riproduzione_run4"]["gruppi_in_conflitto"] == 1)


def test_reproducible():
    with tempfile.TemporaryDirectory() as tmp:
        args = write_fixture(Path(tmp), spec())
        plan = um.shard_plan(args.shards)
        lines = [[json.dumps(r, sort_keys=True) for r in um.records(um.build(args, plan))] for _ in range(2)]
        args.seed = 1
        other = [json.dumps(r, sort_keys=True) for r in um.records(um.build(args, plan))]
    check("riproducibilità: due costruzioni danno righe identiche", lines[0] == lines[1] and len(lines[0]) == 55)
    split0 = [json.loads(x)["split"] for x in lines[0]]
    split1 = [json.loads(x)["split"] for x in other]
    check("con un altro seme cambiano gli split, non gli stati né i gruppi", split0 != split1
          and [json.loads(x)["stato"] for x in lines[0]] == [json.loads(x)["stato"] for x in other]
          and [json.loads(x)["gruppo"] for x in lines[0]] == [json.loads(x)["gruppo"] for x in other])


def test_refusals():
    cases = {}
    s = spec()
    s["excluded"] = s["excluded"] + ["train-9999999"]
    cases["chiave storica senza record"] = refused(s)
    s = spec()
    s["excluded"] = s["excluded"] * 2
    s["reasons"] = s["reasons"]
    cases["chiave storica ripetuta (evidenza ambigua)"] = refused(s)
    s = spec()
    s["train"][1][1].append(sample("train-0000001", "x^{2}+1", png(1)))
    cases["stessa chiave in due shard di train"] = refused(s)
    s = spec()
    s["relabel"][0] = {**s["relabel"][0], "label": "^{4}"}
    cases["rietichettatura con un'etichetta attesa diversa da quella dello shard"] = refused(s)
    s = spec()
    s["reasons"][1] = {**s["reasons"][1], "state": "uncertain"}
    cases["stati di reasons.jsonl incoerenti con le liste"] = refused(s)
    s = spec()
    s["excluded"] = []
    s["reasons"] = [r for r in s["reasons"] if r["key"] != "train-0000003"]
    cases["run4 non riprodotto (errore certo tolto dalle liste)"] = refused(s)
    s = spec()
    s["contamination"] = ["train-0000008", "train-0000009"]
    cases["run2 non riprodotto (contaminazione tolta)"] = refused(s)
    s = spec()
    s["righe"][0][1].append(sample("train-0000777", ARRAY, png(777), rows=array_rows(ARRAY), lines=[[0, 5], [6, 10]]))
    cases["riga ritagliata senza la formula"] = refused(s)
    s = spec()
    s["righe"][0][1][0] = sample("train-0000012", ARRAY, png(12), rows=["a"], lines=[[0, 5]])
    cases["riga ritagliata con righe diverse da array_rows"] = refused(s)

    def wrong_sha(dirs):
        path = dirs["shards"] / "manifest.json"
        meta = json.loads(path.read_text())
        meta["shards"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(meta))

    cases["shard con sha256 diverso dal manifest dello sharding"] = refused(spec(), wrong_sha)

    def wrong_blob(dirs):
        path = dirs["shards"] / "manifest.json"
        meta = json.loads(path.read_text())
        meta["zip_sha256_expected"] = "1" * 64
        path.write_text(json.dumps(meta))

    cases["snapshot con un blob diverso da quello dello sharding"] = refused(spec(), wrong_blob, run=True)
    missed = [k for k, v in cases.items() if not v]
    check(f"rifiuti prima della scrittura: {len(cases)} casi, ciascuno con un motivo", not missed, missed)
    for k, v in cases.items():
        print(f"     {k}: {v[:150]}")


if __name__ == "__main__":
    test_sets()
    test_build_and_verify()
    test_reproducible()
    test_refusals()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
