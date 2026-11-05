#!/usr/bin/env python3
"""Test CPU della riverifica in sola lettura (scripts/manifest_recheck.py), su manifest fittizi costruiti coi fixture di
tests/test_unimer_manifest.py e tests/test_mavis_manifest.py: nessun dato reale. Solo in un job (slurm/tests_cpu.sbatch o
slurm/manifest_recheck.sbatch).

Controlla:
- manifest corretti: tutti i controlli superati, nessuna anomalia, ogni riga ritagliata controllata contro lo shard e
  la formula, rapporti nuovi legati allo sha256 del manifest, nessun controllo sul cambio di seme; manifest, shard,
  evidenze e rapporti della costruzione con gli stessi byte di prima; tutte le evidenze obbligatorie verificate (8 file
  per UniMER, 10 per MAVIS), uguali a quelle che la costruzione registra; un manifest UniMER senza righe ritagliate
  superato con zero righe controllate;
- rifiuti prima di scrivere: sha256 atteso diverso, cartella dei rapporti già scritta, evidenza cambiata dopo la
  costruzione, sorgente sbagliata;
- classificazione degli shard: lo shard delle righe dichiarato `train` in manifest.json, coi record invariati e lo
  sha256 del manifest alterato come atteso, rifiutato per l'incoerenza;
- evidenze vincolate al manifest, per entrambe le sorgenti, su copie di manifest.json passate col loro sha256:
  liste_importate assente o vuota, una lista obbligatoria assente, un file obbligatorio non registrato, digest
  mancante o non valido, tutti rifiutati prima di usare le evidenze;
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


def meta_edited(folder: Path, dst: Path, edit) -> tuple[Path, str]:
    """Copia del manifest scritto con manifest.json alterato da edit(meta) e record invariati; restituisce la cartella e
    lo sha256 del nuovo manifest.json, da passare come atteso: il rifiuto non può venire dallo sha256."""
    shutil.copytree(folder, dst)
    meta = json.loads((dst / "manifest.json").read_text())
    edit(meta)
    (dst / "manifest.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    return dst, sha256_file(dst / "manifest.json")


def loadable(folder: Path, sha: str) -> bool:
    """La copia alterata passa load_expected (sha256 atteso e manifest.load): il rifiuto viene dopo, dal controllo nuovo."""
    try:
        mr.load_expected(folder, sha)
        return True
    except mr.Problem:
        return False


def refusals(tmp: Path, source: str, folder: Path, cases: list, **inputs) -> None:
    """Ogni caso (nome, edit di manifest.json, frase attesa nel motivo) rifiutato prima di scrivere, col motivo atteso e
    non per lo sha256."""
    for n, (name, edit, phrase) in enumerate(cases):
        dst, sha = meta_edited(folder, tmp / f"{source}-ev-{n}", edit)
        report_dir, summary = tmp / "riv" / f"{source}-ev-{n}", tmp / f"riv-{source}-ev-{n}.json"
        report, why = attempt(ns(source, dst, sha, report_dir, summary, **inputs))
        check(f"{source}, {name}: rifiutato prima di scrivere, col motivo",
              report is None and phrase in why and "non è il manifest da riverificare" not in why
              and loadable(dst, sha) and not report_dir.exists() and not summary.exists(), why[:200] or "accettato")


def drop(path: tuple, key: str):
    """edit che toglie meta[path...][key]."""
    def edit(meta):
        node = meta
        for k in path:
            node = node[k]
        del node[key]
    return edit


def set_digest(path: tuple, value):
    def edit(meta):
        node = meta
        for k in path:
            node = node[k]
        node["sha256"] = value(node["sha256"])
    return edit


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
        recorded = json.loads((args.out / "manifest.json").read_text())["liste_importate"]
        check("unimer: evidenze obbligatorie uguali ai file di run4 e run2 registrati dalla costruzione, tutte verificate; "
              "le cartelle del fixture hanno solo quei file, quindi la verifica non ne rilegge altri; split_duplicates_run1 "
              "registrata e non riletta, elencata a parte",
              report is not None
              and {k: set(v) for k, v in mr.UNIMER_EVIDENCE.items()} == {k: set(recorded[k]["file"])
                                                                         for k in mr.UNIMER_EVIDENCE}
              and sorted(p.name for p in args.check.iterdir()) == sorted(mr.UNIMER_EVIDENCE["unimer_check_run4"])
              and sorted(p.name for p in args.overlap.iterdir()) == sorted(mr.UNIMER_EVIDENCE["unimer_overlap_run2"])
              and len(report["evidenze"]) == 8 and all(x["uguale"] for x in report["evidenze"])
              and saved.get("evidenze_uguali_a_quelle_registrate") is True and saved.get("evidenze_obbligatorie") == 8
              and set(report["evidenze_registrate_non_rilette"]) == {"split_duplicates_run1/val_with_duplicates.jsonl"},
              why or sorted(failed(report)))
        check("unimer: classificazione degli shard coerente coi record, riportata (2 shard train con 56 formule, 1 shard "
              "righe con 4 righe)",
              lines.get("classificazione_degli_shard") == {"righe": {"shard": 1, "record": 4},
                                                           "train": {"shard": 2, "record": 56}}
              and saved.get("righe_ritagliate", {}).get("classificazione_degli_shard")
              == lines.get("classificazione_degli_shard"), lines.get("classificazione_degli_shard"))

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

        # lo shard delle righe classificato train in manifest.json, coi record invariati: prima di questo controllo la
        # riverifica non lo leggeva e dichiarava successo con zero righe attese e zero controllate
        def as_train(meta):
            for s in meta["shards"]:
                if s["insieme"] == "righe":
                    s["insieme"] = "train"

        lines_shard = next(s["percorso"] for s in json.loads((args.out / "manifest.json").read_text())["shards"]
                           if s["insieme"] == "righe")
        moved, sha_moved = meta_edited(args.out, tmp / "classificazione", as_train)
        report, why = attempt(ns("unimer", moved, sha_moved, tmp / "riv" / "h", tmp / "riv-h.json", **inputs))
        check("unimer, shard delle righe classificato train (record invariati, sha256 del manifest alterato come atteso): "
              "rifiutato per l'incoerenza, prima di scrivere",
              report is None and "classificazione degli shard incoerente con quella dei record" in why
              and lines_shard in why and "4 record dell'insieme 'righe'" in why
              and "non è il manifest da riverificare" not in why and loadable(moved, sha_moved)
              and sha256_file(moved / "records.jsonl.gz") == sha256_file(args.out / "records.jsonl.gz")
              and not (tmp / "riv" / "h").exists() and not (tmp / "riv-h.json").exists(), why[:220] or "accettato")

        # evidenze vincolate al manifest: copie di manifest.json passate col loro sha256
        lists = "liste_importate"
        refusals(tmp, "unimer", args.out, [
            ("liste_importate assente", drop((), lists), "liste_importate assente o vuota"),
            ("liste_importate vuota", lambda meta: meta.update({lists: {}}), "liste_importate assente o vuota"),
            ("lista obbligatoria assente (unimer_overlap_run2)", drop((lists,), "unimer_overlap_run2"),
             "unimer_overlap_run2: lista obbligatoria non registrata"),
            ("file obbligatorio non registrato (relabel.jsonl di run4)",
             drop((lists, "unimer_check_run4", "file"), "relabel.jsonl"),
             "unimer_check_run4/relabel.jsonl: file obbligatorio non registrato"),
            ("digest mancante (train_matches_test.txt di run2)",
             drop((lists, "unimer_overlap_run2", "file", "train_matches_test.txt"), "sha256"),
             "unimer_overlap_run2/train_matches_test.txt: digest sha256 mancante o non valido"),
            ("digest non valido (prefisso di 12 cifre, quality_excluded.txt di run4)",
             set_digest((lists, "unimer_check_run4", "file", "quality_excluded.txt"), lambda d: d[:12]),
             "unimer_check_run4/quality_excluded.txt: digest sha256 mancante o non valido"),
        ], **inputs)

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


def test_unimer_without_lines():
    """Zero righe ritagliate è valido quando il manifest non ne ha: nessun obbligo di ritagli."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        s = tu.spec()
        s["righe"] = []
        (tmp / "shards" / "lines").mkdir(parents=True)  # il manifest dello sharding delle righe, senza tar
        args = tu.write_fixture(tmp, s)
        um.run(args)
        sha = sha256_file(args.out / "manifest.json")
        report, why = attempt(ns("unimer", args.out, sha, tmp / "riv", tmp / "riv.json", check=args.check,
                                 overlap=args.overlap))
        lines = report["parti"]["righe_ritagliate"] if report else {}
        check("unimer senza righe ritagliate: superato, zero righe attese e zero controllate, nessuno shard delle righe",
              report is not None and report["esito"] and report["anomalie"] == 0
              and lines["record_righe"] == lines["controllati"] == 0 and lines["shard"] == []
              and lines["classificazione_degli_shard"] == {"train": {"shard": 2, "record": 56}},
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
        saved = json.loads((tmp / "riv-a.json").read_text()) if report else {}
        recorded = json.loads((args.out / "manifest.json").read_text())["liste_importate"]
        required = mm.evidence_files(args.evidence)
        check("mavis: evidenze obbligatorie (mavis_manifest.evidence_files) uguali a quelle registrate dalla costruzione, "
              "tutte verificate; nessuna lista registrata fuori da quelle rilette",
              report is not None
              and {k: set(v) for k, v in required.items()} == {k: set(v["file"]) for k, v in recorded.items()}
              and len(report["evidenze"]) == 10 and all(x["uguale"] for x in report["evidenze"])
              and saved.get("evidenze_uguali_a_quelle_registrate") is True and saved.get("evidenze_obbligatorie") == 10
              and report["evidenze_registrate_non_rilette"] == {}, why or sorted(failed(report)))
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

        # evidenze vincolate al manifest: copie di manifest.json passate col loro sha256
        lists = "liste_importate"
        refusals(tmp, "mavis", args.out, [
            ("liste_importate assente", drop((), lists), "liste_importate assente o vuota"),
            ("liste_importate vuota", lambda meta: meta.update({lists: {}}), "liste_importate assente o vuota"),
            ("lista obbligatoria assente (mavis_caption_check_run1)", drop((lists,), "mavis_caption_check_run1"),
             "mavis_caption_check_run1: lista obbligatoria non registrata"),
            ("file obbligatorio non registrato (failures.jsonl di f1_mavis_run2)",
             drop((lists, "f1_mavis_run2", "file"), "failures.jsonl"),
             "f1_mavis_run2/failures.jsonl: file obbligatorio non registrato"),
            ("digest mancante (violations.jsonl del geometry check)",
             drop((lists, "mavis_geometry_run1", "file", "violations.jsonl"), "sha256"),
             "mavis_geometry_run1/violations.jsonl: digest sha256 mancante o non valido"),
            ("digest non valido (maiuscole, issues.jsonl del caption check)",
             set_digest((lists, "mavis_caption_check_run1", "file", "issues.jsonl"), str.upper),
             "mavis_caption_check_run1/issues.jsonl: digest sha256 mancante o non valido"),
        ], evidence=args.evidence)


if __name__ == "__main__":
    test_unimer()
    test_unimer_without_lines()
    test_mavis()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
