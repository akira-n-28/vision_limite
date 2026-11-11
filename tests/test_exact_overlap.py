"""Identità esatte fra sorgenti (scripts/exact_overlap.py, primo sottoinsieme di C6). Fixture costruita qui: quattro
manifest piccoli nel formato di synth-v4-candidato-2, unimer-candidato-2, mavis-geometry-candidato-3 e
pixmo-cap-candidato-1, con le loro scansioni nel formato di decontam.py scan, panel_decontam.py scan e
pixmo_inventory.py. Nessun dato reale, nessuna immagine: i digest sono sha256 di etichette. Solo in un job SLURM
(slurm/exact_overlap.sbatch o slurm/tests_cpu.sbatch).

Controlla:
- copia identica in due sorgenti (byte e pixel dell'originale), con conflitto train/dev fra record ammessi;
- catena transitiva che unisce tre gruppi esistenti di tre sorgenti, con un membro senza legame diretto;
- immagine finale (shard) identica con originali diversi;
- digest mancanti (immagine illeggibile), pixel RGBA del manifest dei sintetici mai confrontato, convenzione
  incompatibile di una scansione (i pixel non si confrontano, i byte sì);
- formula e riga ritagliata di UniMER con la stessa chiave e lo stesso record_id: due voci, un record, nessuna
  sovrascrittura; legami fra le due dentro il gruppo, non nuovi;
- legame nuovo dentro una sorgente (train/dev di UniMER), record riservati, quarantene distinte dai conflitti fra
  dati caricabili, con la portata dei motivi;
- arresti e controlli falliti: sha256 del manifest, riga della scansione alterata, records.jsonl.gz diverso, riga
  ritagliata in un altro gruppo, catena di C9; prova limitata con la stima."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("SLURM_JOB_ID", "test")
import exact_overlap as eo  # noqa: E402
import unimer_manifest as um  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.provenance import sha256_file  # noqa: E402

FAILS: list[str] = []
DONE: list[str] = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({str(detail)[:400]})" if detail != "" and not cond else ""))
    DONE.append(name)
    if not cond:
        FAILS.append(name)


def H(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


PX = {k: H("px-" + k) for k in ("copy", "chain-1", "final-same", "reserved", "blocked-synth", "line-shared",
                                 "u-internal")}
BY = {k: H("by-" + k) for k in ("copy", "chain-2", "blocked-pair")}


def rid(src: str, rev: str, key: str) -> str:
    return mf.sha256_text(mf.canonical({"sorgente": src, "revisione": rev, "record": key}))


def gid(label: str) -> str:
    return H("gruppo-" + label)


def write_gz(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps(r, sort_keys=True) + "\n").encode())
    return sha256_file(path)


def entry(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


# ---------------------------------------------------------------- sorgenti della fixture

SREV, UREV, MREV, PREV = "v4@" + "1" * 40, "2" * 40, "3" * 40, "4" * 40
T0, TE = "data/shards/synth/v4/train/synth-train-00000.tar", "data/shards/synth/v4/test/synth-test.tar"
UT = "data/shards/unimer/2343ddd9-dense/unimer-train-00000.tar"
UL = "data/shards/unimer/2343ddd9-dense/lines/unimer-lines-00000.tar"
MS = "data/shards/mavis-geometry/9c80fff3-caption/mavis-geometry-caption-00000.tar"
PS = "data/shards/pixmo-cap/3e42775e/pixmo-cap-00000.tar"
CONV = "rgb-su-bianco/1"


def synth_source(variant: set) -> dict:
    def rec(key, ins, shard, idx, g, split, state, png, rgba, motivo=None):
        return {"record_id": rid("synth", SREV, key), "sorgente": "synth", "revisione": SREV, "record": key,
                "insieme": ins, "shard": shard, "indice": idx, "famiglia": "coord", "coppia": None, "gruppo": gid(g),
                "split": split, "stato": state, "motivo": motivo or {}, "gold": None,
                "immagini": [{"sha256": png, "pixel_sha256": rgba, "ordine": 0, "griglia": [4, 4]}],
                "json_sha256": H(key + ".json"), "qa": [{"tipo": "count", "sha256": H(key + ".qa")}],
                "evidenze": [], "trasformazione": None, "genitori": []}

    def row(key, shard, idx, png, px, read="letta"):
        return {"key": key, "shard": str(mf.ROOT / shard), "posizione": idx, "immagine": read, "byte_sha256": png,
                "pixel_sha256": px if read == "letta" else None, "phash": None, "corrispondenze": {}}

    recs = [rec("synth-0000001", "train", T0, 0, "s1", "train", "ammesso", H("s1.png"), H("s1.rgba")),
            # pixel RGBA del manifest uguale, come stringa, al pixel rgb-su-bianco/1 dell'originale di p1: mai confrontato
            rec("synth-0000002", "train", T0, 1, "s2", "dev", "ammesso", H("s2.png"), PX["copy"]),
            rec("synth-0000004", "train", T0, 2, "s4", "train", "quarantena", H("s4.png"), H("s4.rgba"),
                {"incerto": ["testi_sovrapposti: etichette"]}),
            rec("synth-0000005", "train", T0, 3, "s5", "train", "ammesso", H("s5.png"), H("s5.rgba")),
            rec("synth-test-0000003", "test", TE, 0, "s3", "test", "ammesso", H("s3.png"), H("s3.rgba"))]
    rows = [row("synth-test-0000003", TE, 0, H("s3.png"), PX["reserved"]),  # ordine dello shard come il glob ordinato
            row("synth-0000001", T0, 0, H("s1.png"), PX["chain-1"]),
            row("synth-0000002", T0, 1, H("s2.png"), H("s2.px")),
            row("synth-0000004", T0, 2, H("s4.png"), PX["blocked-synth"]),
            row("synth-0000005", T0, 3, H("s5.png"), None, "illeggibile")]
    return {"records": recs, "rows": rows,
            "shards": [{"insieme": "train", "nome": "synth-train-00000.tar", "percorso": T0, "record": 4},
                       {"insieme": "test", "nome": "synth-test.tar", "percorso": TE, "record": 1}],
            "summary": {"kind": "synth", "samples": len(rows), "convenzione_immagini": CONV}, "meta": {}}


def unimer_source(variant: set) -> dict:
    def rec(key, ins, shard, idx, g, split, png):
        return {"record_id": rid("unimer", UREV, key), "sorgente": "unimer", "revisione": UREV, "record": key,
                "insieme": ins, "shard": shard, "indice": idx,
                "compiti": ["transcribe"] if ins == "train" else ["transcribe_line", "transcribe_line_select"],
                "gruppo": gid(g), "split": split, "stato": "ammesso", "motivo": [], "gold": None,
                "immagini": [{"sha256": png, "ordine": 0, "griglia": [2, 8]}], "json_sha256": H(f"{ins}/{key}.json"),
                "etichetta_sha256": H(key + ".tex"), "evidenze": [], "trasformazione": None, "genitori": []}

    def row(key, ins, shard, idx, png, ob, op, pp, crops=None):
        out = {"insieme": ins, "shard": shard, "indice": idx, "record": key, "json_sha256": H(f"{ins}/{key}.json"),
               "png_sha256": png, "problemi": [], "controlli": {},
               "originale": {"lettura": "letta", "byte_sha256": ob, "pixel_sha256": op, "dimensioni": [10, 10]},
               "png": {"lettura": "letta", "byte_sha256": png, "pixel_sha256": pp, "dimensioni": [20, 20]}}
        if crops is not None:
            out["ritagli"] = [{"riga": k, "fascia": [5 * k, 5 * k + 4], "pixel_sha256": c, "dimensioni": [20, 8]}
                              for k, c in enumerate(crops)]
        return out

    k1 = "train-0000001"
    recs = [rec(k1, "train", UT, 0, "u1", "train", H("u1.png")),
            rec("train-0000002", "train", UT, 1, "u2", "dev", H("u2.png")),
            rec("train-0000003", "train", UT, 2, "u3", "train", H("u3.png")),
            rec("train-0000004", "train", UT, 3, "u4", "train", H("u4.png")),
            rec("train-0000005", "train", UT, 4, "u5", "train", H("u5.png")),
            rec("train-0000006", "train", UT, 5, "u6", "dev", H("u6.png")),
            # riga ritagliata: stessa chiave e stesso record_id della formula, stesso PNG, gruppo della formula
            rec(k1, "righe", UL, 0, "altro" if "righe-gruppo" in variant else "u1", "train", H("u1.png"))]
    rows = [row(k1, "train", UT, 0, H("u1.png"), H("u1.ob"), H("u1.op"), H("u1.pp")),
            row("train-0000002", "train", UT, 1, H("u2.png"), H("u2.ob"), H("u2.op"), PX["final-same"]),
            row("train-0000003", "train", UT, 2, H("u3.png"), BY["chain-2"], H("u3.same"), H("u3.same")),
            row("train-0000004", "train", UT, 3, H("u4.png"), H("u4.ob"), PX["blocked-synth"], H("u4.pp")),
            row("train-0000005", "train", UT, 4, H("u5.png"), H("u5.ob"), PX["u-internal"], H("u5.pp")),
            row("train-0000006", "train", UT, 5, H("u6.png"), H("u6.ob"), PX["u-internal"], H("u6.pp")),
            row(k1, "righe", UL, 0, H("u1.png"), H("u1.ob"), H("u1.op"), H("u1.pp"),
                [PX["line-shared"], H("u1.line1")])]
    scanned = H("unimer-candidato-1")
    return {"records": recs, "rows": rows,
            "shards": [{"insieme": "train", "nome": "unimer-train-00000.tar", "percorso": UT, "record": 6},
                       {"insieme": "righe", "nome": "unimer-lines-00000.tar", "percorso": UL, "record": 1}],
            "summary": {"schema": "panel-decontam-scan/1", "parziale": False, "convenzione_immagini": CONV,
                        "manifest": {"manifest_sha256": scanned, "righe": len(rows)}},
            "meta": {"precedente": {"versione": "1", "manifest_sha256": scanned}}}


def mavis_source(variant: set) -> dict:
    tasks = [{"compito": "describe", "stato": "ammesso"}, {"compito": "side_len", "stato": "ammesso", "domande": 2},
             {"compito": "shape_type", "stato": "quarantena", "domande": 1}]

    def rec(key, idx, g, split, state="ammesso", motivo=None, evid=None):
        return {"record_id": rid("mavis-geometry", MREV, key), "sorgente": "mavis-geometry", "revisione": MREV,
                "record": key, "insieme": "train", "shard": MS, "indice": idx, "famiglia": "d0",
                "compiti": [dict(t, stato=um.worst(t["stato"], state)) for t in tasks], "gruppo": gid(g),
                "split": split, "stato": state, "motivo": motivo or [], "gold": None,
                "immagini": [{"sha256": H(key + ".png"), "ordine": 0, "griglia": [4, 4]}],
                "json_sha256": H(key + ".json"), "qa": [], "evidenze": evid or [], "trasformazione": None,
                "genitori": []}

    def row(key, idx, ob, op, pp, json_altered=False):
        return {"insieme": "train", "shard": MS, "indice": idx, "record": key,
                "json_sha256": H("altro") if json_altered else H(key + ".json"), "png_sha256": H(key + ".png"),
                "problemi": [], "controlli": {},
                "originale": {"lettura": "letta", "byte_sha256": ob, "pixel_sha256": op, "phash": "0" * 16},
                "png": {"lettura": "letta", "byte_sha256": H(key + ".png"), "pixel_sha256": pp, "phash": "0" * 16}}

    dec = ("decontaminazione_pannello: candidato da verificare (pHash entro 4 bit o 10-gram in comune; gruppo in "
           "quarantena)")
    recs = [rec("d0-1", 0, "m1", "dev"),
            rec("d0-2", 1, "m2", "train", "quarantena", [dec],
                [{"controllo": "decontaminazione_pannello", "corrispondenze": {"phash": 1}}]),
            rec("d0-3", 2, "m3", "train"), rec("d0-3b", 3, "m3", "train"), rec("d0-4", 4, "m4", "train"),
            rec("d0-3c", 5, "m3", "train")]
    rows = [row("d0-1", 0, BY["copy"], PX["copy"], H("m1.pp"), "mavis-json-alterato" in variant),
            row("d0-2", 1, BY["blocked-pair"], H("m2.op"), H("m2.pp")),
            row("d0-3", 2, H("m3.ob"), H("m3.op"), PX["chain-1"]),
            row("d0-3b", 3, BY["chain-2"], H("m3b.op"), H("m3b.pp")),
            row("d0-4", 4, H("m4.ob"), H("m4.op"), PX["reserved"]),
            row("d0-3c", 5, H("m3c.ob"), H("m3c.op"), H("m3c.pp"))]
    scanned = H("mavis-candidato-1")
    return {"records": recs, "rows": rows,
            "shards": [{"insieme": "train", "nome": "mavis-geometry-caption-00000.tar", "percorso": MS, "record": 6}],
            "summary": {"schema": "panel-decontam-scan/1", "parziale": False, "convenzione_immagini": CONV,
                        "manifest": {"manifest_sha256": scanned, "righe": len(rows)}},
            "meta": {"catena_delle_versioni": [{"versione": "1", "manifest_sha256": scanned},
                                               {"versione": "2", "manifest_sha256": H("mavis-candidato-2")}]}}


def pixmo_source(variant: set) -> dict:
    def rec(key, idx, g, split, jpg, jpx, opx, state="ammesso", direct=()):
        return {"record_id": rid("pixmo-cap", PREV, key), "sorgente": "pixmo-cap", "revisione": PREV, "record": key,
                "insieme": "train", "shard": PS, "indice": idx, "compiti": ["pixmo_caption"], "gruppo": gid(g),
                "split": split, "stato": state, "motivo": [f"motivo: {d}" for d in direct], "gold": None,
                "blocchi": {"diretti": list(direct), "ereditati": []},
                "immagini": [{"sha256": jpg, "ordine": 0, "griglia": [4, 4], "pixel_sha256": jpx}],
                "json_sha256": H(key + ".json"), "didascalia_sha256": H(key + ".cap"), "url_sha256": None,
                "richiesta": 0, "originale": {"file": "data/train-00000-of-00075.parquet", "riga": idx,
                                              "pixel_sha256": opx, "orientamento_exif": None},
                "evidenze": [{"lista": "decontaminazione_pannello_pixmo", "esito": "nessuna_corrispondenza",
                              "corrispondenze": {}}], "trasformazione": None, "genitori": []}

    def row(key, idx, ob, opx, jpg, jpx):
        return {"chiave": key, "file": "data/train-00000-of-00075.parquet", "riga": idx, "shard": "pixmo-cap-00000.tar",
                "posizione": idx, "originale": {"lettura": "letta", "byte_sha256": ob, "pixel_sha256": opx},
                "immagine_shard": {"lettura": "letta", "byte_sha256": jpg, "pixel_sha256": jpx},
                "didascalia_sha256": H(key + ".cap"), "corrispondenze": {}}

    recs = [rec("pixmo-00000-000000", 0, "p1", "train", H("p1.jpg"), H("p1.jpx"), PX["copy"]),
            rec("pixmo-00000-000001", 1, "p2", "train", H("p2.jpg"), H("p2.jpx"), H("p2.opx"), "quarantena",
                ("possibile_segnaposto",)),
            rec("pixmo-00000-000002", 2, "p3", "dev", H("p3.jpg"), PX["final-same"], H("p3.opx")),
            rec("pixmo-00000-000003", 3, "p5", "train", H("p5.jpg"), PX["line-shared"], H("p5.opx"))]
    rows = [row("pixmo-00000-000000", 0, BY["copy"], PX["copy"], H("p1.jpg"), H("p1.jpx")),
            row("pixmo-00000-000001", 1, BY["blocked-pair"], H("p2.opx"), H("p2.jpg"), H("p2.jpx")),
            row("pixmo-00000-000002", 2, H("p3.ob"), H("p3.opx"), H("p3.jpg"), PX["final-same"]),
            row("pixmo-00000-000003", 3, H("p5.ob"), H("p5.opx"), H("p5.jpg"), PX["line-shared"]),
            {"chiave": "pixmo-00000-000099", "file": "data/train-00000-of-00075.parquet", "riga": 99, "shard": None,
             "posizione": None, "originale": {"lettura": "illeggibile", "byte_sha256": H("p99.ob")},
             "immagine_shard": None, "didascalia_sha256": H("p99.cap"), "corrispondenze": {}}]
    return {"records": recs, "rows": rows,
            "shards": [{"insieme": "train", "nome": "pixmo-cap-00000.tar", "percorso": PS, "record": 4}],
            "summary": {"schema": "pixmo-panel-scan/1", "righe": len(rows), "parziale": False,
                        "convenzione_immagini": "grigio-diretto/0" if "pixmo-grigio" in variant else CONV},
            "meta": {"scarti_dello_sharding": [{"chiave": "pixmo-00000-000099", "esito_sharding": "scartata:illeggibile"}]}}


BUILDERS = {"synth": (synth_source, "synth", SREV), "unimer": (unimer_source, "unimer", UREV),
            "mavis": (mavis_source, "mavis-geometry", MREV), "pixmo": (pixmo_source, "pixmo-cap", PREV)}


def build(tmp: Path, variant: set = frozenset()) -> dict:
    """Le quattro sorgenti in tmp: scansione, records.jsonl.gz e manifest.json che cita la scansione con gli sha256.
    Restituisce {nome: (cartella del manifest, sha256 di manifest.json)}."""
    out = {}
    for name, (fn, source, rev) in BUILDERS.items():
        src = fn(set(variant))
        scan = tmp / name / "scan"
        write_gz(scan / "records.jsonl.gz", src["rows"])
        (scan / "summary.json").write_text(json.dumps(src["summary"]))
        scan_ref = {"records": entry(scan / "records.jsonl.gz"), "summary": entry(scan / "summary.json")}
        folder = tmp / name / "manifest"
        sha = write_gz(folder / "records.jsonl.gz", src["records"])
        if name == "unimer" and "records-sha-errato" in variant:
            sha = H("altro")
        meta = {"schema": mf.SCHEMA, "nome": f"{source}-candidato", "versione": "9", "sorgente": source,
                "revisione": rev, "shards": src["shards"],
                "records": {"file": "records.jsonl.gz", "sha256": sha, "righe": len(src["records"])}, **src["meta"]}
        if name == "pixmo":
            meta["ingressi"] = {"file": {"scansione": {"records.jsonl.gz": scan_ref["records"],
                                                       "summary.json": scan_ref["summary"]}}}
        else:
            meta["decontaminazione"] = {"scansione": scan_ref}
        (folder / "manifest.json").write_text(json.dumps(meta, indent=1))
        out[name] = (folder, sha256_file(folder / "manifest.json"))
    return out


def c9_manifest(tmp: Path, parent_sha: str, in_chain: bool) -> tuple[Path, str]:
    folder = tmp / ("c9-ok" if in_chain else "c9-altro")
    folder.mkdir(parents=True)
    chain = [{"versione": "3", "manifest_sha256": parent_sha if in_chain else H("altro")},
             {"versione": "5", "manifest_sha256": None}]
    (folder / "manifest.json").write_text(json.dumps({"schema": mf.SCHEMA, "sorgente": "mavis-geometry",
                                                      "nome": "mavis-geometry-candidato", "versione": "5",
                                                      "catena_delle_versioni": chain, "records": {"righe": 6}}))
    return folder, sha256_file(folder / "manifest.json")


def args_for(srcs: dict, out: Path, summary: Path | None = None, limit=None, c9=None, override=None):
    manifest = [[n, str(f), (override or {}).get(n, s)] for n, (f, s) in srcs.items()]
    return argparse.Namespace(manifest=manifest, c9=c9, out=out, summary=summary, limit=limit)


def load_components(out: Path) -> list[dict]:
    with gzip.open(out / "componenti.jsonl.gz", "rt") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def by_groups(comps: list[dict]) -> dict[frozenset, dict]:
    return {frozenset(g["gruppo"] for g in c["gruppi"]): c for c in comps}


def fails(fn) -> str:
    try:
        fn()
    except eo.Failed as exc:
        return str(exc)
    return ""


# ---------------------------------------------------------------- test

def test_pure() -> None:
    m = lambda g, split, ok: {"gruppo": g, "split": split, "caricabile": ok}  # noqa: E731
    check("classificazione: train e dev caricabili -> train_dev", eo.classify([m("a", "train", True),
                                                                             m("b", "dev", True)]) == "train_dev")
    check("classificazione: riservato legato a un caricabile -> riservati",
          eo.classify([m("a", "test", False), m("b", "train", True)]) == "riservati")
    check("classificazione: due gruppi caricabili di train -> solo_train",
          eo.classify([m("a", "train", True), m("b", "train", True)]) == "solo_train")
    check("classificazione: un solo gruppo caricabile -> un_gruppo_caricabile (la quarantena non è un conflitto)",
          eo.classify([m("a", "train", True), m("b", "dev", False)]) == "un_gruppo_caricabile")
    check("classificazione: nessun caricabile -> non_caricabili",
          eo.classify([m("a", "train", False), m("b", "dev", False)]) == "non_caricabili")
    check("caricabile: ammesso con un compito ammesso; shape_type in quarantena non basta",
          eo.loadable("ammesso", "train", (("describe", "ammesso", True), ("shape_type", "quarantena", True)))
          and not eo.loadable("ammesso", "train", (("shape_type", "quarantena", True),))
          and not eo.loadable("ammesso", "test", (("x", "ammesso", True),))
          and not eo.loadable("quarantena", "train", (("x", "quarantena", True),)))
    check("portata: decontaminazione per pHash -> immagine, per testo -> testo, senza corrispondenze proprie -> gruppo",
          eo.scope(["decontaminazione_pannello: candidato"], ("phash",)) == ["immagine"]
          and eo.scope(["decontaminazione_pannello: candidato"], ("testo",)) == ["testo_o_risposta"]
          and eo.scope(["decontaminazione_pannello: candidato"], ()) == ["gruppo"])
    check("portata: etichetta di UniMER -> testo; segnaposto ed EXIF -> immagine; motivo ignoto -> da esaminare",
          eo.scope(["unimer_check/4: comando_non_reso"], ()) == ["testo_o_risposta"]
          and eo.scope(["possibile_segnaposto", "orientamento_exif"], ()) == ["immagine"]
          and eo.scope(["qualcosa di nuovo"], ()) == ["da_esaminare"])
    check("portata: la contaminazione dei sintetici (stato della scena) è di gruppo, non confusa con la "
          "decontaminazione", eo.scope(["contaminazione: stato della scena di test"], ()) == ["gruppo"])
    st = eo.Store()
    st.add_digest(b"P", H("x"), 1, 0)
    st.add_digest(b"P", H("x"), 2, 1)
    st.add_digest(b"B", H("x"), 3, 0)
    check("classi: stesso digest e stesso tipo uniti, tipi diversi separati",
          st.dups == {b"P" + bytes.fromhex(H("x")): [1 << eo.SHIFT, 2 << eo.SHIFT | 1]} and len(st.first) == 2)


def test_main(tmp: Path) -> None:
    srcs = build(tmp / "base")
    out, summary = tmp / "out-base", tmp / "riepilogo.json"
    c9 = c9_manifest(tmp, srcs["mavis"][1], True)
    code = eo.run(args_for(srcs, out, summary, c9=[str(c9[0]), c9[1]]))
    rep = json.loads((out / "report.json").read_text())
    check("percorso completo: nessun controllo fallito, uscita 0", code == 0 and not rep["falliti"], rep["falliti"])
    check("C9: candidato-5 della stessa catena controllato, non una sorgente in più",
          any(c["controllo"].startswith("C9:") and c["esito"] for c in rep["controlli"])
          and set(rep["ingressi"]) == {"synth", "unimer", "mavis", "pixmo"})
    comps = load_components(out)
    g = by_groups(comps)
    check("componenti: otto con almeno due gruppi (sette fra sorgenti, una dentro UniMER)",
          len(comps) == 8 and rep["categorie"]["fra_sorgenti"] == {"train_dev": 1, "solo_train": 2, "solo_dev": 1,
                                                                    "non_caricabili": 1, "un_gruppo_caricabile": 1,
                                                                    "riservati": 1}
          and rep["categorie"]["dentro_una_sorgente"] == {"train_dev": 1}, rep["categorie"])
    a = g.get(frozenset({gid("p1"), gid("m1")}))
    check("copia identica in due sorgenti: byte e pixel dell'originale, conflitto train/dev fra ammessi",
          a is not None and a["categoria"] == "train_dev" and a["fra_sorgenti"]
          and set(a["rappresentazioni"]) == {"byte: mavis-geometry/originale = pixmo-cap/originale",
                                             "pixel: mavis-geometry/originale = pixmo-cap/originale"}
          and a["caricabili"] == {"train": 1, "dev": 1}, a and a["rappresentazioni"])
    check("compiti per voce: shape_type in quarantena visibile accanto ai compiti ammessi del record di dev",
          a is not None and a["compiti"].get("mavis-geometry/shape_type/dev/quarantena") == 1
          and a["compiti"].get("mavis-geometry/side_len/dev/ammesso") == 1)
    b = g.get(frozenset({gid("s1"), gid("m3"), gid("u3")}))
    check("catena transitiva: tre gruppi di tre sorgenti uniti senza un digest comune fra sintetici e UniMER",
          b is not None and b["categoria"] == "solo_train" and b["sorgenti"] == ["mavis-geometry", "synth", "unimer"]
          and b["voci"] == 5 and b["voci_con_legame_diretto"] == 4
          and not any({"synth", "unimer"} <= {v["sorgente"] for v in cl["voci"]} for cl in b["classi"]),
          b and {k: b[k] for k in ("categoria", "voci", "voci_con_legame_diretto")})
    c = g.get(frozenset({gid("p3"), gid("u2")}))
    check("immagine finale identica con originali diversi: legame solo fra gli shard, nessun originale",
          c is not None and c["categoria"] == "solo_dev"
          and list(c["rappresentazioni"]) == ["pixel: pixmo-cap/shard = unimer/shard"], c and c["rappresentazioni"])
    check("pixel RGBA del manifest dei sintetici mai confrontato (stringa uguale a un pixel rgb di PixMo)",
          not any(gid("s2") in k for k in g))
    e = g.get(frozenset({gid("p2"), gid("m2")}))
    check("solo record in quarantena: non_caricabili, motivi con la portata (pHash e segnaposto -> immagine)",
          e is not None and e["categoria"] == "non_caricabili"
          and all(x["portata"] == ["immagine"] for x in e["non_caricabili"]), e and e["non_caricabili"])
    f = g.get(frozenset({gid("s4"), gid("u4")}))
    check("quarantena legata a un gruppo caricabile: un_gruppo_caricabile, non un conflitto fra dati caricabili",
          f is not None and f["categoria"] == "un_gruppo_caricabile" and f["caricabili"] == {"train": 1, "dev": 0}
          and f["non_caricabili"][0]["portata"] == ["immagine"], f and f["non_caricabili"])
    r = g.get(frozenset({gid("s3"), gid("m4")}))
    check("record di test dei sintetici legato a un record di train: riservati",
          r is not None and r["categoria"] == "riservati" and r["con_riservati"])
    h = g.get(frozenset({gid("u1"), gid("p5")}))
    uni = [m for m in (h or {}).get("membri", []) if m["sorgente"] == "unimer"]
    check("formula e riga ritagliata di UniMER: due voci (train e righe), un record, nessuna sovrascrittura",
          h is not None and h["voci_per_sorgente"]["unimer"] == 2 and h["record_per_sorgente"]["unimer"] == 1
          and sorted(m["insieme"] for m in uni) == ["righe", "train"] and len({m["record_id"] for m in uni}) == 1,
          h and h["voci_per_sorgente"])
    check("ritaglio della riga legato allo shard di PixMo, con la rappresentazione esatta",
          h is not None and h["classi"][0]["firma"] == "pixel: pixmo-cap/shard = unimer/ritaglio"
          and {(v["insieme"], v["rappresentazione"]) for v in h["classi"][0]["voci"]}
          == {("righe", "ritaglio 0"), ("train", "shard")})
    i = g.get(frozenset({gid("u5"), gid("u6")}))
    check("legame nuovo dentro UniMER (stessi pixel dell'originale, PNG diversi): train_dev dentro la sorgente",
          i is not None and i["categoria"] == "train_dev" and not i["fra_sorgenti"])
    cl = rep["classi"]
    check("formula e riga nello stesso gruppo: legami già coperti, non nuovi; originale e shard uguali nella stessa voce",
          cl["legami_nei_gruppi"].get("byte: unimer/shard") == 1 and cl["legami_nei_gruppi"].get("pixel: unimer/originale") == 1
          and cl["stessa_voce"] == {"pixel: unimer/originale = unimer/shard": 1}, cl)
    cov = rep["copertura"]
    check("copertura: immagine illeggibile contata come pixel non eseguibile, byte disponibili",
          cov["synth"]["train/shard/pixel"] == {"disponibile": 3, "illeggibile": 1}
          and cov["synth"]["train/shard/byte"] == {"disponibile": 4}, cov["synth"])
    check("copertura: ritagli di UniMER solo pixel, scarto dello sharding di PixMo fuori dal confronto",
          cov["unimer"]["righe/ritaglio/pixel"] == {"disponibile": 2} and "righe/ritaglio/byte" not in cov["unimer"]
          and rep["ingressi"]["pixmo"]["scansione"]["righe_fuori_confronto"]
          == {"scarto dello sharding (nessuna voce del manifest)": 1})
    pairs = rep["coppie_di_sorgenti"]
    check("coppie di sorgenti: componenti, classi dirette e record con legame diretto",
          pairs["mavis-geometry + pixmo-cap"]["componenti"] == 2 and pairs["mavis-geometry + pixmo-cap"]["classi_dirette"] == 3
          and pairs["synth + unimer"]["classi_dirette"] == 1
          and pairs["synth + unimer"]["record_nelle_componenti"] == {"synth": 2, "unimer": 2}
          and pairs["unimer (dentro la sorgente)"]["componenti"] == 1, pairs)
    check("conflitti train/dev: due componenti, una fra sorgenti", rep["conflitti_train_dev"]["componenti"] == 2
          and rep["conflitti_train_dev"]["fra_sorgenti"] == 1, rep["conflitti_train_dev"])
    s = json.loads(summary.read_text())
    check("riepilogo per git: stesse categorie, componenti per categoria, evidenze complete citate per sha256",
          s["categorie"] == rep["categorie"] and "train_dev/fra_sorgenti" in s["componenti"]
          and s["files"]["componenti.jsonl.gz"]["sha256"] == sha256_file(out / "componenti.jsonl.gz"))
    check("uscite esistenti rifiutate", "esiste già" in fails(lambda: eo.run(args_for(srcs, out))))
    check("sha256 atteso di un manifest diverso: arresto prima di leggere",
          "atteso" in fails(lambda: eo.run(args_for(srcs, tmp / "out-x", override={"mavis": H("altro")}))))


def test_convention(tmp: Path) -> None:
    srcs = build(tmp / "grigio", {"pixmo-grigio"})
    out = tmp / "out-grigio"
    code = eo.run(args_for(srcs, out))
    rep = json.loads((out / "report.json").read_text())
    g = by_groups(load_components(out))
    check("convenzione incompatibile: i pixel di PixMo non si confrontano, i byte sì",
          code == 0 and rep["ingressi"]["pixmo"]["scansione"]["pixel_confrontati"] is False
          and frozenset({gid("p1"), gid("m1")}) in g and frozenset({gid("p3"), gid("u2")}) not in g
          and frozenset({gid("u1"), gid("p5")}) not in g
          and list(g[frozenset({gid("p1"), gid("m1")})]["rappresentazioni"])
          == ["byte: mavis-geometry/originale = pixmo-cap/originale"], sorted(map(len, g)))
    check("convenzione incompatibile: copertura dei pixel di PixMo non eseguibile",
          rep["copertura"]["pixmo"]["train/originale/pixel"] == {"convenzione_incompatibile": 4}
          and rep["copertura"]["pixmo"]["train/shard/pixel"] == {"convenzione_incompatibile": 4})


def test_failures(tmp: Path) -> None:
    srcs = build(tmp / "guasti", {"mavis-json-alterato", "records-sha-errato", "righe-gruppo"})
    c9 = c9_manifest(tmp, srcs["mavis"][1], False)
    out = tmp / "out-guasti"
    code = eo.run(args_for(srcs, out, c9=[str(c9[0]), c9[1]]))
    rep = json.loads((out / "report.json").read_text())
    failed = rep["falliti"]
    check("controlli falliti: uscita 1", code == 1)
    check("riga della scansione con un JSON diverso da quello del manifest: collegamento fallito",
          any(x.startswith("mavis: ogni riga della scansione ricollegata") for x in failed), failed)
    check("records.jsonl.gz con uno sha256 diverso da manifest.json: controllo fallito",
          any(x.startswith("unimer: records.jsonl.gz") for x in failed), failed)
    check("riga ritagliata in un gruppo diverso da quello della formula: controllo fallito",
          any(x.startswith("unimer: record_id ricalcolati") for x in failed), failed)
    check("C9 con una catena che non contiene candidato-3: controllo fallito", any(x.startswith("C9:") for x in failed))
    check("altri controlli non toccati dai guasti", not any(x.startswith(("synth:", "pixmo:")) for x in failed), failed)


def test_limit(tmp: Path) -> None:
    srcs = build(tmp / "prova")
    out = tmp / "out-prova"
    code = eo.run(args_for(srcs, out, limit=2))
    rep = json.loads((out / "report.json").read_text())
    est = rep.get("stima_completa") or {}
    check("prova limitata: parziale, nessun controllo fallito, sha256 completi",
          code == 0 and rep["parziale"] and not rep["falliti"]
          and rep["ingressi"]["unimer"]["records"]["sha256"] == sha256_file(srcs["unimer"][0] / "records.jsonl.gz"),
          rep["falliti"])
    check("prova limitata: righe fuori dalla prova contate, non errori; stima per file",
          rep["ingressi"]["synth"]["scansione"]["righe_fuori_confronto"].get("fuori dalla prova limitata", 0) >= 1
          and len(est.get("file", [])) == 8 and est.get("voci_lette") == 8 and est.get("voci_totali") == 22, est)


def main() -> None:
    test_pure()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_main(tmp)
        test_convention(tmp)
        test_failures(tmp)
        test_limit(tmp)
    print(f"{len(DONE) - len(FAILS)} controlli su {len(DONE)}")
    if FAILS:
        print("falliti: " + "; ".join(FAILS))
        sys.exit(1)


if __name__ == "__main__":
    main()
