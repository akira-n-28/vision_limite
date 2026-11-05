#!/usr/bin/env python3
"""Test CPU della riverifica in sola lettura (scripts/manifest_recheck.py), su manifest fittizi costruiti coi fixture di
tests/test_unimer_manifest.py e tests/test_mavis_manifest.py: nessun dato reale. Solo in un job (slurm/tests_cpu.sbatch o
slurm/manifest_recheck.sbatch).

Controlla:
- manifest corretti: tutti i controlli superati, nessuna anomalia, ogni riga ritagliata controllata contro lo shard e
  la formula, rapporti nuovi legati allo sha256 del manifest, nessun controllo sul cambio di seme; manifest, shard,
  evidenze e rapporti della costruzione con gli stessi byte di prima;
- rifiuti prima di scrivere: sha256 atteso diverso, cartella dei rapporti già scritta, evidenza cambiata dopo la
  costruzione, sorgente sbagliata;
- anomalie riportate e nulla corretto: manifest con il PNG di una riga ritagliata scambiato; shard delle righe con
  fasce sovrapposte (JSON diverso da quello del manifest)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import manifest_recheck as mr  # noqa: E402
import mavis_manifest as mm  # noqa: E402
import test_mavis_manifest as tm  # noqa: E402
import test_unimer_manifest as tu  # noqa: E402
import unimer_manifest as um  # noqa: E402
from limite_vl.provenance import sha256_file  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def snapshot(*folders: Path) -> dict[str, str]:
    """sha256 di ogni file sotto le cartelle: la riverifica non deve scrivere negli ingressi."""
    return {str(p): sha256_file(p) for d in folders for p in sorted(Path(d).rglob("*")) if p.is_file()}


def ns(source: str, manifest: Path, sha: str, report: Path, summary: Path, **inputs) -> argparse.Namespace:
    return argparse.Namespace(source=source, manifest=manifest, sha256=sha, report=report, summary=summary,
                              check=inputs.get("check"), overlap=inputs.get("overlap"), evidence=inputs.get("evidence"))


def attempt(args) -> tuple[dict | None, str]:
    try:
        return mr.run(args), ""
    except mr.Problem as exc:
        return None, str(exc) or "rifiutato"


def failed(report: dict | None) -> set[str]:
    return {c["controllo"] for c in report["controlli"] if not c["esito"]} if report else {"(nessun rapporto)"}


def test_unimer():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        args = tu.write_fixture(tmp, tu.spec())
        um.run(args)
        sha = sha256_file(args.out / "manifest.json")
        inputs = {"check": args.check, "overlap": args.overlap}
        before = snapshot(args.out, args.report, args.shards, args.check, args.overlap) | {
            str(args.summary): sha256_file(args.summary)}
        report, why = attempt(ns("unimer", args.out, sha, tmp / "riv" / "a", tmp / "riv-a.json", **inputs))
        after = snapshot(args.out, args.report, args.shards, args.check, args.overlap) | {
            str(args.summary): sha256_file(args.summary)}
        saved = json.loads((tmp / "riv-a.json").read_text()) if report else {}
        lines = report["parti"]["righe_ritagliate"] if report else {}
        check("unimer, manifest corretto: tutti i controlli superati, nessuna anomalia, ogni riga ritagliata controllata",
              report is not None and report["esito"] and report["anomalie"] == 0
              and lines["controllati"] == lines["record_righe"] == 4 and lines["fasce_per_record"] == {2: 4}
              and not (tmp / "riv" / "a" / "anomalie.jsonl").exists(),
              why or sorted(failed(report)))
        check("unimer: rapporti legati allo sha256 del manifest, nessun controllo sul cambio di seme",
              saved.get("manifest_sha256") == sha == report["manifest_sha256"] == report["manifest_sha256_atteso"]
              and saved.get("rapporto_sha256") == sha256_file(tmp / "riv" / "a" / "verifica.json")
              and not any("altro seme" in c["controllo"] for c in report["controlli"]))
        check("unimer, sola lettura: manifest, shard, evidenze e rapporti della costruzione con gli stessi byte",
              before == after)

        cases = {"sha256 atteso diverso": attempt(ns("unimer", args.out, "0" * 64, tmp / "riv" / "b",
                                                     tmp / "riv-b.json", **inputs))[1],
                 "cartella dei rapporti già scritta": attempt(ns("unimer", args.out, sha, tmp / "riv" / "a",
                                                                 tmp / "riv-c.json", **inputs))[1],
                 "sorgente sbagliata": attempt(ns("mavis", args.out, sha, tmp / "riv" / "d", tmp / "riv-d.json",
                                                  evidence=tmp))[1]}
        changed = tmp / "run4-alterato"
        shutil.copytree(args.check, changed)
        with open(changed / "uncertain.txt", "a") as fh:
            fh.write("train-0000001\n")
        cases["evidenza cambiata dopo la costruzione"] = attempt(ns(
            "unimer", args.out, sha, tmp / "riv" / "e", tmp / "riv-e.json", check=changed, overlap=args.overlap))[1]
        missed = [k for k, v in cases.items() if not v]
        check(f"unimer, rifiuti prima di scrivere: {len(cases)} casi, ciascuno con un motivo e senza rapporti",
              not missed and not any((tmp / "riv" / x).exists() for x in "bde")
              and not any((tmp / f"riv-{x}.json").exists() for x in "bcde"), missed)
        for k, v in cases.items():
            print(f"     {k}: {v[:150]}")

        # manifest con il PNG di una riga ritagliata scambiato con quello di una formula estranea
        base = {r["record"]: r for r in tu.records_of(args.out) if r["insieme"] == "train"}
        foreign = base["train-0000001"]["immagini"][0]["sha256"]

        def swap(r):
            if r["insieme"] == "righe" and r["record"] == "train-0000012":
                r["immagini"][0]["sha256"] = foreign

        swapped = tu.altered(args.out, tmp / "scambio", swap)
        sha_swapped = sha256_file(swapped / "manifest.json")
        frozen = snapshot(swapped)
        report, why = attempt(ns("unimer", swapped, sha_swapped, tmp / "riv" / "f", tmp / "riv-f.json", **inputs))
        found = {a["record"]: a for a in report["anomalie_in_questo_file"]} if report else {}
        check("unimer, PNG di una riga ritagliata scambiato nel manifest: riportato dalla verifica e dal confronto con "
              "lo shard, nulla corretto",
              report is not None and not report["esito"] and set(found) == {"train-0000012"}
              and found["train-0000012"]["problemi"] == {"shard": ["PNG con byte diversi da quelli del manifest"]}
              and any("lo stesso PNG della formula" in c for c in failed(report))
              and (tmp / "riv" / "f" / "anomalie.jsonl").is_file() and snapshot(swapped) == frozen,
              why or sorted(failed(report)))

        # shard delle righe con fasce sovrapposte (il JSON non è più quello del manifest)
        items = tu.spec()["righe"][0][1]
        items[0] = tu.line("train-0000012", tu.ARRAY, tu.png(12), bands=[[0, 7], [6, 10]])
        tu.write_tar(args.shards / "lines" / "unimer-lines-00000.tar", items)
        report, why = attempt(ns("unimer", args.out, sha, tmp / "riv" / "g", tmp / "riv-g.json", **inputs))
        found = {a["record"]: a for a in report["anomalie_in_questo_file"]} if report else {}
        problems = found.get("train-0000012", {}).get("problemi", {})
        check("unimer, fasce sovrapposte nello shard delle righe: riportate con l'identificativo, le fasce e la causa",
              report is not None and not report["esito"] and set(found) == {"train-0000012"}
              and any("sovrapposta alla precedente" in x for x in problems.get("fasce", []))
              and "JSON con byte diversi da quelli del manifest" in problems.get("shard", [])
              and found["train-0000012"]["fasce"] == [[0, 7], [6, 10]]
              and report["parti"]["righe_ritagliate"]["anomalie_per_causa"]
              and sha256_file(args.out / "manifest.json") == sha,
              why or sorted(failed(report)))


def test_mavis():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        args = tm.write_fixture(tmp, tm.spec())
        mm.run(args)
        sha = sha256_file(args.out / "manifest.json")
        before = snapshot(args.out, args.report, args.shards, args.evidence)
        report, why = attempt(ns("mavis", args.out, sha, tmp / "riv" / "a", tmp / "riv-a.json",
                                 evidence=args.evidence))
        check("mavis, manifest corretto: tutti i controlli superati, nessun controllo sul cambio di seme, sola lettura",
              report is not None and report["esito"] and report["anomalie"] == 0
              and not any("altro seme" in c["controllo"] for c in report["controlli"])
              and snapshot(args.out, args.report, args.shards, args.evidence) == before
              and json.loads((tmp / "riv-a.json").read_text())["manifest_sha256"] == sha,
              why or sorted(failed(report)))
        changed = tmp / "outputs-alterato"
        shutil.copytree(args.evidence, changed)
        with open(changed / "mavis_caption_check" / "run1" / "issues.jsonl", "a") as fh:
            fh.write(json.dumps({"key": "d0-9323", "issues": ["misura_diversa"]}) + "\n")
        cases = {"sha256 atteso diverso": attempt(ns("mavis", args.out, "f" * 64, tmp / "riv" / "b",
                                                     tmp / "riv-b.json", evidence=args.evidence))[1],
                 "evidenza cambiata dopo la costruzione": attempt(ns("mavis", args.out, sha, tmp / "riv" / "c",
                                                                     tmp / "riv-c.json", evidence=changed))[1]}
        missed = [k for k, v in cases.items() if not v]
        check(f"mavis, rifiuti prima di scrivere: {len(cases)} casi, ciascuno con un motivo", not missed, missed)
        for k, v in cases.items():
            print(f"     {k}: {v[:150]}")


if __name__ == "__main__":
    test_unimer()
    test_mavis()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
