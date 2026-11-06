#!/usr/bin/env python3
"""Test CPU della decontaminazione esterna di UniMER e MAVIS-Geometry contro il pannello (scripts/panel_decontam.py), su
manifest costruiti coi fixture di tests/test_unimer_manifest.py e tests/test_mavis_manifest.py, con zip dello snapshot,
indice, pannello e parquet di OCRBench e MathVerse fittizi: nessun dato reale. Solo in un job (slurm/tests_cpu.sbatch o
slurm/panel_decontam.sbatch).

Controlla:
- decisioni pure: pixel identici confermati; pHash, testo e formula lunga candidati; formula breve solo evidenza;
  controllo mancante non verificabile; stato per gruppo senza riammissioni, anche fra formula e riga ritagliata con la
  stessa chiave; compiti di MAVIS che seguono il record con shape_type almeno in quarantena; ritaglio uguale a quello
  del loader (data.unimer_line_sample);
- UniMER, percorso completo (scansione, verifica, manifest, verifica della versione nuova): originale con gli stessi
  pixel di un elemento del pannello (anche RGBA con trasparenze, riconosciuto solo nella convenzione rgb-su-bianco/1),
  originale con un pixel diverso non promosso a prova, ritaglio di una riga identico a un elemento, formula lunga uguale
  a una di OCRBench, riga ritagliata con la formula lunga di UniMER-Test (identità distinta dalla sua formula, stato
  propagato al gruppo), formula breve solo evidenza, riconciliazione con run2, nessuna riammissione;
- MAVIS, percorso completo: originale identico a un elemento di MathVerse (gruppo escluso, duplicato compreso), copia con
  un pixel diverso solo candidato pHash, 10-gram della didascalia candidato testuale, stati dei compiti conservati
  (shape_type in quarantena), provenienze di MathVerse, candidati storici solo diagnostici;
- regole testuali esplicite (panel-decontam/2): la frequenza nella sorgente non decide nulla. Un problema specifico del
  benchmark ripetuto in più di 25 record, senza corrispondenze visive, resta candidato, anche con duplicati e varianti
  nello stesso gruppo; lo stesso test fallisce col filtro di f163a81 (letto da git). L'enumerazione delle lettere dei
  punti è ignorata solo nella descrizione del loader (riga `Points:`, testo legato al manifest), non negli altri campi;
  una corrispondenza coperta accanto a una specifica lascia il record candidato; propagazione al gruppo con stati
  precedenti e compiti conservati;
- riesame di una versione decisa col filtro precedente (candidato-2 costruito dal codice di f163a81 sul fixture): solo
  JSON letti, versione nuova derivata dal genitore senza riammissioni, catena verificata (il verificatore della
  costruzione non si applica a una versione derivata), versioni o descrizioni alterate rifiutate;
- arresti e incompatibilità: sha256 del manifest o dell'indice diversi da quelli attesi, indice nella convenzione storica,
  shard alterato dopo la costruzione, prova limitata usata come scansione completa, righe della scansione con un'identità
  scambiata, versione nuova alterata (stato riammesso, evidenza tolta, shape_type ammessa)."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import types
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import mavis_manifest as mm  # noqa: E402
import panel_decontam as pd  # noqa: E402
import test_mavis_manifest as tm  # noqa: E402
import test_unimer_manifest as tu  # noqa: E402
import unimer_manifest as um  # noqa: E402
from limite_vl import decontam as dc  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.data import _unimer_line, unimer_line_sample  # noqa: E402
from limite_vl.f1_mavis import mavis_truth  # noqa: E402
from limite_vl.f1_schema import render_geometry  # noqa: E402
from limite_vl.processing import to_rgb  # noqa: E402
from limite_vl.provenance import sha256_file  # noqa: E402
from test_mavis_gen import MULTI_CAP, SEMI_GEN  # noqa: E402

FAILS = []
DONE = []
LONG2 = "p+q+r+s+t+u+v+w+x+y+z"  # 21 token visibili, solo in OCRBench
ARRAY2 = "\\begin{array}{l}a+b+c+d+e+f+g+h+i+j+k\\\\z\\end{array}"  # prima riga = formula lunga di UniMER-Test
HME = "Handwritten Mathematical Expression Recognition"
OLD_COMMIT = "f163a81"  # panel_decontam.py col filtro di frequenza (panel-decontam/1, mavis-geometry-candidato-2)
# un problema specifico di un benchmark, ripetuto nelle domande di 27 record MAVIS del fixture (più di 25)
SPECIFIC = ("Two equal squares ABCD and EFGH overlap so that the shaded region has area equal to 1. What is the area "
            "of the square ABCD?")
SPECIFIC_GRAMS = [" ".join(g) for g in [tuple(dc.normalize_words(SPECIFIC)[k:k + 10])
                                        for k in range(len(dc.normalize_words(SPECIFIC)) - 9)]]
# enumerazione di lettere in un problema (come le dodici lettere di AIME 2025 I) e in una domanda MAVIS
LETTERS = "The twelve letters A, B, C, D, E, F, G, H, I, J, K, and L are randomly grouped into six pairs of letters."
LETTER_QUESTION = "The points A, B, C, D, E, F, G, H, I, J are marked on the figure; which one is the apex?"
ENUM = "a b c d e f g h i j"
# figura a quattro forme con dieci punti A-J (sector, rettangolo con triangolo equilatero, triangolo rettangolo,
# rettangolo con semicerchio): la riga Points della descrizione canonica ha dieci etichette
CAP10 = ("ABC takes the form of a sector. CBDFE is a rectangle, although its side DE extends into an equilateral triangle "
         "beyond the rectangle. FDG is a right triangle. HGIJ is a rectangle, featuring an extension of the side IJ into "
         "a semi-circle. Angle BAD equals 90 degrees.")
GEN10 = SEMI_GEN
os.environ.setdefault("SLURM_JOB_ID", "test")


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    DONE.append(name)
    if not cond:
        FAILS.append(name)


def encode(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def drawing(seed: int, size: tuple[int, int]) -> Image.Image:
    """Disegno a rettangoli diverso per ogni seme: pHash informativo (le immagini piatte hanno tutte lo stesso)."""
    rng = random.Random(seed)
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    w, h = size
    for _ in range(10):
        x0, y0 = rng.randrange(0, max(1, w - 8)), rng.randrange(0, max(1, h - 8))
        d.rectangle((x0, y0, x0 + rng.randrange(4, max(5, w // 2)), y0 + rng.randrange(4, max(5, h // 2))),
                    fill=tuple(rng.randrange(0, 256) for _ in range(3)))
    return im


def one_pixel_changed(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.putpixel((0, 0), tuple((v + 128) % 256 for v in im.getpixel((0, 0))))
        return encode(im)


def real_zip(tmp: Path, snapshot: Path, name: str, members: dict[str, bytes], shards: Path) -> None:
    """Zip vero dello snapshot (blob col nome del suo sha256) al posto di quello finto del fixture, e manifest dello
    sharding aggiornato prima della costruzione."""
    blobs = tmp / "blobs-veri"
    blobs.mkdir(exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for member, data in members.items():
            zf.writestr(member, data)
    data = buf.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    (blobs / sha).write_bytes(data)
    (snapshot / name).unlink()
    (snapshot / name).symlink_to(blobs / sha)
    meta = json.loads((shards / "manifest.json").read_text())
    meta["zip_sha256_expected"] = sha
    (shards / "manifest.json").write_text(json.dumps(meta))


def panel_fixture(tmp: Path, entries: list[dict], convention: str = dc.IMAGE_CONVENTION) -> tuple[Path, Path, str]:
    """Indice e pannello: entries con benchmark, file, item, image (byte o None), text, hints, ruolo, metrica."""
    items, elements = [], {}
    for e in entries:
        h, pixels, size = dc.hash_image_bytes(e["image"]) if e.get("image") else (None, None, None)
        items.append({"benchmark": e["benchmark"], "file": e["file"], "item": e["item"], "id": e["item"],
                      "pos": 0 if e.get("image") else None, "phash": None if h is None else f"{h:016x}",
                      "pixel_sha256": pixels, "size": size, "text": e.get("text", ""), "hints": e.get("hints", {})})
        key = f"{e['benchmark']}/{e['item']}"
        elements[key] = {"chiave": key, "benchmark": e["benchmark"], "ruolo": e["ruolo"],
                         "gruppo": hashlib.sha256(key.encode()).hexdigest(), "metrica_primaria": e.get("metrica")}
    index_dir, panel_dir = tmp / "indice", tmp / "pannello"
    index_dir.mkdir()
    panel_dir.mkdir()
    doc = {"schema": "panel-index/2", "items": items}
    if convention:
        doc["convenzione_immagini"] = convention
    (index_dir / "index.json").write_text(json.dumps(doc))
    (index_dir / "verifica.json").write_text(json.dumps({
        "verificato": True, "errori": 0, "voci_con_immagine": sum(it["pos"] is not None for it in items),
        "convenzione_immagini": convention}))
    (panel_dir / "elements.jsonl.gz").write_bytes(gzip.compress(
        "".join(json.dumps(x) + "\n" for x in elements.values()).encode(), mtime=0))
    sha = sha256_file(index_dir / "index.json")
    (panel_dir / "summary.json").write_text(json.dumps({"indice": {"sha256": sha}, "elementi": len(elements),
                                                        "job_id": "test"}))
    return index_dir, panel_dir, sha


def fillers() -> list[dict]:
    """Un elemento per ruolo che manca nei fixture: il perimetro deve averli tutti."""
    return [{"benchmark": "mathvista", "file": "data/test.parquet", "item": "test:900", "image": encode(drawing(900, (40, 30))),
             "text": "What is the value of the largest bar in the chart shown?", "ruolo": "solo_decontaminazione"},
            {"benchmark": "chartqa", "file": "data/test.parquet", "item": "test:901", "image": encode(drawing(901, (40, 30))),
             "text": "How many bars?", "ruolo": "test", "metrica": False},
            {"benchmark": "dynamath", "file": "data/s.parquet", "item": "s:902", "image": encode(drawing(902, (40, 30))),
             "text": "Find x.", "ruolo": "test", "metrica": True},
            {"benchmark": "gsm8k", "file": "main/test.parquet", "item": "test:903", "image": None,
             "text": "A long text-only question about apples and oranges for the panel fixture.", "ruolo": "sviluppo"}]


def parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def attempt(fn, args) -> tuple[object, str]:
    try:
        return fn(args), ""
    except (pd.Failed, um.Problem) as exc:
        return None, str(exc) or "rifiutato"


def args_of(*argv) -> object:
    return pd.parser().parse_args([str(a) for a in argv])


def scan_rows(folder: Path) -> dict[tuple, dict]:
    return {pd.ident(r): r for r in pd.read_rows(folder / "records.jsonl.gz")}


def rewrite_scan(src: Path, dst: Path, edit) -> Path:
    """Copia della scansione con le righe alterate da edit(lista) e lo sha256 aggiornato in summary.json: l'unica
    incoerenza è quella introdotta."""
    shutil.copytree(src, dst)
    rows = pd.read_rows(dst / "records.jsonl.gz")
    edit(rows)
    data = gzip.compress("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode(), mtime=0)
    (dst / "records.jsonl.gz").write_bytes(data)
    s = json.loads((dst / "summary.json").read_text())
    s["files"]["records.jsonl.gz"]["sha256"] = hashlib.sha256(data).hexdigest()
    (dst / "summary.json").write_text(json.dumps(s))
    return dst


def altered_manifest(folder: Path, dst: Path, source: str, edit) -> Path:
    """Copia della versione nuova coi record alterati da edit(record), sha256 e conteggi aggiornati."""
    shutil.copytree(folder, dst)
    recs = [json.loads(x) for x in gzip.decompress((dst / "records.jsonl.gz").read_bytes()).decode().splitlines()]
    for r in recs:
        edit(r)
    (dst / "records.jsonl.gz").unlink()
    sha, _ = um.write_records_gz(dst / "records.jsonl.gz", (json.dumps(r, sort_keys=True, ensure_ascii=False,
                                                                       separators=(",", ":")) for r in recs))
    meta = json.loads((dst / "manifest.json").read_text())
    meta["records"]["sha256"] = sha
    meta["conteggi"] = pd.summarize(source, recs)
    (dst / "manifest.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return dst


def failing(result: dict) -> set[str]:
    return {c["controllo"] for c in result["controlli"] if not c["esito"]}


def old_module():
    """scripts/panel_decontam.py di OLD_COMMIT (filtro di frequenza), letto da git e caricato come modulo
    `panel_decontam_v1`: i processi del pool lo ritrovano col fork."""
    if "panel_decontam_v1" in sys.modules:
        return sys.modules["panel_decontam_v1"]
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{OLD_COMMIT}:scripts/panel_decontam.py"], check=True,
                         capture_output=True, text=True).stdout
    mod = types.ModuleType("panel_decontam_v1")
    mod.__file__ = str(ROOT / "scripts" / "panel_decontam.py")
    sys.modules["panel_decontam_v1"] = mod
    exec(compile(src, f"{OLD_COMMIT}:scripts/panel_decontam.py", "exec"), mod.__dict__)
    return mod


def attempt_any(fn, args) -> tuple[object, str]:
    """Come attempt, anche per le eccezioni del modulo di OLD_COMMIT."""
    old = old_module()
    try:
        return fn(args), ""
    except (pd.Failed, old.Failed, um.Problem, pd.mr.Problem) as exc:
        return None, str(exc) or "rifiutato"


def text_detail(field: str, grams: list[str], element: str = "mathvision/test:753") -> dict:
    return {"tipo": "testo", "campo": field, "ngram": grams[0], "ngram_tutti": list(grams), "elemento": element,
            "benchmark": element.split("/")[0], "posizione": None}


def text_row(key: str, details: list[dict], **found) -> dict:
    """Riga di scansione di MAVIS con le corrispondenze testuali date (e gli altri tipi in found)."""
    r = row(testo=len(details), **found) if details else row(**found)
    r.update(insieme="train", record=key, dettagli=list(details),
             ngram_in_comune=sorted({g for d in details for g in d["ngram_tutti"]}))
    return r


# ---------------------------------------------------------------- 1. decisioni pure

def row(insieme="train", **found) -> dict:
    checks = {c: True for c in ("pixel_originale", "pixel_png", "pixel_ritagli", "formula", "phash_originale",
                                "phash_png", "testo")}
    checks.update(found.pop("checks", {}))
    return {"insieme": insieme, "corrispondenze": found, "controlli": checks, "problemi": []}


def repeated_problem_stays_candidate(outcome_of) -> bool:
    """Il caso della correzione: lo stesso problema specifico di un benchmark nelle domande di 30 record MAVIS (più di
    25), senza corrispondenze visive. Dieci record stanno in un gruppo, con duplicati (stesso testo) e varianti (testo
    con un prefisso diverso), e il gruppo ha un membro senza testi in comune; gli altri venti sono gruppi da soli. Il
    test passa se tutti i trenta restano candidati e i gruppi vanno in quarantena, membro senza testo compreso, senza
    toccare lo stato escluso del genitore. outcome_of(righe) -> {identità: esito}."""
    detail = text_detail("text_lite_question", SPECIFIC_GRAMS)
    rows = [text_row(f"k{n:02d}", [detail]) for n in range(30)] + [text_row("k99", [])]
    recs = [{"insieme": "train", "record": f"k{n:02d}", "gruppo": "GA" if n < 10 else f"G{n}",
             "stato": "escluso" if n == 3 else "ammesso"} for n in range(30)]
    recs.append({"insieme": "train", "record": "k99", "gruppo": "GA", "stato": "ammesso"})
    out = outcome_of(rows)
    d = pd.decide(recs, out)
    return (all(out[("train", f"k{n:02d}")] == "candidato" for n in range(30))
            and out[("train", "k99")] == "nessuna_corrispondenza"
            and all(d[("train", r["record"])]["stato"] == ("escluso" if r["record"] == "k03" else "quarantena")
                    for r in recs))


def test_rules():
    """Regola enumerazione_punti_loader e legame delle descrizioni col manifest."""
    desc12 = render_geometry({"points": list("ABCDEFGHIJKL"), "shapes": [], "sides": {}, "angles": {}})
    windows = {"a b c d e f g h i j", "b c d e f g h i j k", "c d e f g h i j k l"}
    cov = pd.rule_coverage("descrizione_del_loader", desc12)
    check("regola: nella descrizione del loader copre le finestre di 10 etichette della riga Points, e solo quelle",
          set(cov) == windows and set(cov.values()) == {"enumerazione_punti_loader"}
          and "points a b c d e f g h i" not in cov, sorted(cov))
    check("regola: nessun altro campo è coperto, nemmeno con lo stesso testo (didascalia, domande, CoT, domande del "
          "loader)", all(pd.rule_coverage(f, desc12) == {} for f in ("caption", "text_lite_question", "CoT_reasoning",
                                                                     "domande_del_loader", "diagram_generation")))
    desc7 = render_geometry({"points": list("ABCDEFG"), "shapes": [], "sides": {}, "angles": {}})
    bad = [desc12.replace("Points: A, B,", "Points: AB, B,"), desc12.replace("Points: A, B,", "Points: B, A,"),
           desc12.replace("Figure: geometry diagram.", "Figure: a diagram."), desc12.replace("Points: ", "Vertices: ")]
    check("regola: meno di dieci punti, etichette non di una lettera, non ordinate o formato diverso: nulla coperto",
          pd.rule_coverage("descrizione_del_loader", desc7) == {}
          and all(pd.rule_coverage("descrizione_del_loader", t) == {} for t in bad))
    outside = desc12 + "\nNote: a b c d e f g h i j."
    check("regola: un 10-gram che compare anche fuori dalla riga Points non è coperto, gli altri sì",
          set(pd.rule_coverage("descrizione_del_loader", outside)) == windows - {"a b c d e f g h i j"})
    check("regole: un solo campo autorizzato e sha256 dichiarato delle regole",
          pd.RULE_FIELDS == {"descrizione_del_loader"}
          and pd.TEXT_RULES_SHA256 == mf.sha256_text(mf.canonical(pd.TEXT_RULES)))
    truth, why = mavis_truth(CAP10, GEN10)
    real = render_geometry(truth) if truth else ""
    check("preparazione: la figura a quattro forme del fixture ha i dieci punti A-J e la descrizione canonica ne copre "
          "solo l'enumerazione", truth is not None and truth["points"] == list("ABCDEFGHIJ")
          and set(pd.rule_coverage("descrizione_del_loader", real)) == {ENUM}, (why, real[:200]))
    # legame col manifest: testo presente, sha256 di descrizione_sha256, 10-gram della corrispondenza presenti
    enum = text_detail("descrizione_del_loader", sorted(windows), "aime25/aime2025-I:6")
    rows = {("train", k): text_row(k, [enum]) for k in ("e1", "e2", "e3")}
    rows[("train", "e4")] = text_row("e4", [text_detail("descrizione_del_loader", ["x " * 9 + "y"], "aime25/x")])
    records = {("train", "e1"): {"descrizione_sha256": mf.sha256_text(desc12)},
               ("train", "e2"): {"descrizione_sha256": "0" * 64}, ("train", "e3"): {"descrizione_sha256": None},
               ("train", "e4"): {"descrizione_sha256": mf.sha256_text(desc12)}}
    texts = {i: {"descrizione_del_loader": desc12} for i in rows}
    texts.pop(("train", "e3"))
    cov, problems = pd.bind_rule_texts(rows, records, texts)
    check("legame: copre solo la descrizione con lo sha256 del manifest; sha256 diverso, testo mancante o 10-gram "
          "assenti dal testo sono problemi e non coprono nulla", set(cov) == {("train", "e1")} and len(problems) == 3
          and pd.record_outcome(rows[("train", "e1")], "mavis", cov.get(("train", "e1"))) == "nessuna_corrispondenza"
          and pd.record_outcome(rows[("train", "e2")], "mavis", cov.get(("train", "e2"))) == "candidato", problems)


def test_decisions():
    o = pd.record_outcome
    t_info = text_detail("caption", ["a b", "x y"])
    enum = text_detail("descrizione_del_loader", [ENUM], "aime25/aime2025-I:6")
    specific = text_detail("text_lite_question", SPECIFIC_GRAMS)
    covered = {"descrizione_del_loader": {ENUM: "enumerazione_punti_loader"}}
    check("esito: pixel identici confermati, anche con altre somiglianze",
          o(row(pixel=1, phash=3), "mavis") == "corrispondenza_confermata")
    check("esito: pHash, testo e formula lunga sono candidati, non prove",
          o(row(phash=1), "mavis") == o(dict(row(testo=1), dettagli=[t_info]), "mavis")
          == o(row(formula_lunga=1), "unimer") == "candidato")
    check("esito: formula breve uguale, solo evidenza", o(row(formula_breve=4), "unimer") == "nessuna_corrispondenza")
    check("esito: l'enumerazione dei punti coperta nella descrizione del loader è solo evidenza; nello stesso record "
          "una corrispondenza specifica, o un pHash, lascia il record candidato",
          o(text_row("e", [enum]), "mavis", covered) == "nessuna_corrispondenza"
          and o(text_row("e", [enum, specific]), "mavis", covered) == "candidato"
          and o(text_row("e", [enum], phash=1), "mavis", covered) == "candidato"
          and o(text_row("e", [enum]), "mavis") == "candidato")
    check("esito: la stessa enumerazione in un altro campo resta candidata (la copertura vale per il suo campo)",
          o(text_row("e", [text_detail("text_lite_question", [ENUM])]), "mavis", covered) == "candidato")
    check("problema specifico ripetuto in più di 25 record, senza corrispondenze visive, con duplicati e varianti nello "
          "stesso gruppo: resta candidato e il gruppo va in quarantena (filtro corretto)",
          repeated_problem_stays_candidate(lambda rows: {pd.ident(r): o(r, "mavis") for r in rows}))
    old = old_module()

    def old_filter(rows):
        rito = old.rito_of(old.source_ngrams(rows))
        return {pd.ident(r): old.record_outcome(r, "mavis", rito) for r in rows}

    check(f"regressione: lo stesso test fallisce col filtro di {OLD_COMMIT} (frequenza nella sorgente)",
          not repeated_problem_stays_candidate(old_filter))
    rows = [text_row(f"k{n:02d}", [specific]) for n in range(30)]
    check("regola storica (solo per riverificare candidato-2): ricalcolata dalle righe come in "
          f"{OLD_COMMIT}, e come copertura riproduce i suoi esiti",
          pd.frequency_rule_v1(rows) == old.rito_of(old.source_ngrams(rows)) == frozenset(SPECIFIC_GRAMS)
          and all(o(r, "mavis", pd.frequency_coverage_v1(r, pd.frequency_rule_v1(rows))) == "nessuna_corrispondenza"
                  for r in rows))
    check("motivi del riesame: uno per una transizione più grave e uno per una meno grave, nessuno senza cambio",
          pd.transition_reason("nessuna_corrispondenza", "candidato") == pd.REASON_RIESAME["piu_grave"]
          and pd.transition_reason("candidato", "nessuna_corrispondenza") == pd.REASON_RIESAME["meno_grave"]
          and pd.transition_reason("candidato", "candidato") is None)
    check("esito: un controllo richiesto non eseguito rende il record non verificabile",
          o(row(checks={"pixel_originale": False}), "unimer") == "non_verificabile"
          and o(row("righe", checks={"pixel_ritagli": False}), "unimer") == "non_verificabile"
          and o(row(checks={"phash_png": False}), "mavis") == "non_verificabile"
          and o(row(checks={"pixel_ritagli": False}), "unimer") == "nessuna_corrispondenza")
    recs = [{"insieme": "train", "record": "k1", "gruppo": "G1", "stato": "ammesso"},
            {"insieme": "righe", "record": "k1", "gruppo": "G1", "stato": "ammesso"},
            {"insieme": "train", "record": "k2", "gruppo": "G1", "stato": "escluso"},
            {"insieme": "train", "record": "k3", "gruppo": "G2", "stato": "quarantena"},
            {"insieme": "train", "record": "k4", "gruppo": "G2", "stato": "ammesso"},
            {"insieme": "train", "record": "k5", "gruppo": "G3", "stato": "quarantena"}]
    out = {("righe", "k1"): "candidato", ("train", "k1"): "nessuna_corrispondenza",
           ("train", "k2"): "nessuna_corrispondenza", ("train", "k3"): "corrispondenza_confermata",
           ("train", "k4"): "nessuna_corrispondenza", ("train", "k5"): "nessuna_corrispondenza"}
    d = pd.decide(recs, out)
    check("gruppo: la riga ritagliata candidata mette in quarantena la sua formula (stessa chiave, identità distinte)",
          d[("train", "k1")]["stato"] == d[("righe", "k1")]["stato"] == "quarantena"
          and d[("train", "k1")]["esito"] == "nessuna_corrispondenza" and d[("righe", "k1")]["esito"] == "candidato"
          and d[("train", "k1")]["cause"] == ["righe/k1"])
    check("gruppo: nessuna riammissione, l'escluso resta escluso e la quarantena non diventa ammesso",
          d[("train", "k2")]["stato"] == "escluso" and d[("train", "k5")]["stato"] == "quarantena")
    check("gruppo: una conferma esclude tutto il gruppo, compreso il membro già in quarantena",
          d[("train", "k3")]["stato"] == d[("train", "k4")]["stato"] == "escluso")
    tasks = [{"compito": "describe", "stato": "ammesso"}, {"compito": "shape_type", "stato": "quarantena", "domande": 2},
             {"compito": "side_len", "stato": "ammesso", "domande": 3}]
    st = {s: {t["compito"]: t["stato"] for t in pd.task_states(tasks, s)} for s in ("ammesso", "quarantena", "escluso")}
    check("compiti MAVIS: seguono il record, shape_type almeno in quarantena",
          st["ammesso"] == {"describe": "ammesso", "shape_type": "quarantena", "side_len": "ammesso"}
          and set(st["quarantena"].values()) == {"quarantena"} and set(st["escluso"].values()) == {"escluso"}
          and pd.task_states(tasks, "ammesso")[2]["domande"] == 3)
    check("OCRBench: domande HME riconosciute dal tipo o dal dataset",
          pd.is_hme({"question_type": HME}) and pd.is_hme({"dataset": "HME100k"})
          and not pd.is_hme({"question_type": "Regular Text Recognition", "dataset": "IIIT5K"}))
    # ritaglio: uguale a quello del loader per la riga che il loader sceglie
    key, image, meta = tu.line("train-0000012", tu.ARRAY, tu.png(12))
    files = {"png": image, "json": json.dumps(meta).encode()}
    loader = unimer_line_sample(key, files)
    k = _unimer_line(key, meta)
    ours = pd.line_crop(Image.open(io.BytesIO(image)), meta["lines"], k)
    check("ritaglio: gli stessi pixel di data.unimer_line_sample per la riga scelta dal loader",
          dc.pixel_sha256(to_rgb(ours)) == dc.pixel_sha256(to_rgb(loader.image)) and ours.size == loader.image.size)


# ---------------------------------------------------------------- 2. UniMER

def unimer_spec() -> dict:
    s = tu.spec()
    s["train"][1][1].extend([tu.sample("train-0000017", LONG2, tu.png(17)), tu.sample("train-0000018", ARRAY2, tu.png(18))])
    s["righe"][0][1].append(tu.line("train-0000018", ARRAY2, tu.png(18)))
    return s


def unimer_originals(s: dict) -> dict[str, bytes]:
    """Originale di ogni formula con le dimensioni originali del JSON; PNG uguali negli shard, originali uguali."""
    out = {}
    for _, items in s["train"]:
        for key, image, meta in items:
            color = tuple(hashlib.sha256(image).digest()[:3])
            out[key] = encode(Image.new("RGB", (meta["orig_width"], meta["orig_height"]), color))
    w, h = next(meta for k, _, meta in s["train"][1][1] if k == "train-0000102")["orig_width"], 10
    rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # trasparente a sinistra, colorato a destra
    ImageDraw.Draw(rgba).rectangle((w // 2, 0, w - 1, h - 1), fill=(200, 30, 60, 255))
    out["train-0000102"] = encode(rgba)
    return out


def unimer_fixture(tmp: Path) -> dict:
    s = unimer_spec()
    originals = unimer_originals(s)
    test_items = {key: (image, meta) for key, image, meta in s["test"][0][1]}

    def tamper(dirs):
        real_zip(tmp, dirs["snapshot"], "UniMER-1M.zip",
                 {f"UniMER-1M/images/{int(k.split('-')[1])}.png": v for k, v in originals.items()}, dirs["shards"])
        (dirs["overlap"] / "test_seen.jsonl").write_text("".join(json.dumps({"key": k}) + "\n" for k in test_items))
        (dirs["overlap"] / "test_unseen_spe.txt").write_text("")

    args = tu.write_fixture(tmp, s, tamper)
    um.run(args)
    # immagini del pannello: UniMER-Test con gli originali (quelle di run2 uguali agli originali del training)
    test_image = {"spe-0000002": originals["train-0000009"], "spe-0000004": originals["train-0000014"]}
    entries = []
    roles = {"spe-0000001": ("test", True), "spe-0000002": ("test", True), "spe-0000003": ("sviluppo", None),
             "spe-0000004": ("test", False), "spe-0000005": ("diagnostica", None), "spe-0000006": ("test", True)}
    for key, (image, meta) in test_items.items():
        role, metric = roles[key]
        entries.append({"benchmark": "unimer_test", "file": "spe", "item": key, "text": meta["label"],
                        "image": test_image.get(key, encode(drawing(int(key[-1]), (30, 12)))),
                        "hints": {"subset": "spe", "formula": meta["label_spaced"]}, "ruolo": role, "metrica": metric})
    line_key, line_png, line_meta = next(x for x in s["righe"][0][1] if x[0] == "train-0000012")
    crop = to_rgb(pd.line_crop(Image.open(io.BytesIO(line_png)), line_meta["lines"], 0))
    flat = to_rgb(Image.open(io.BytesIO(originals["train-0000102"])))
    ocr = [{"dataset": "HME100k", "question_type": HME, "answer": [" ".join(LONG2)]},
           {"dataset": "HME100k", "question_type": HME, "answer": ["y = m x"]},
           {"dataset": "IIIT5K", "question_type": "Regular Text Recognition", "answer": ["hello"]}]
    for n, r in enumerate(ocr):
        entries.append({"benchmark": "ocrbench", "file": "data/test-00000-of-00001.parquet",
                        "item": f"test-00000-of-00001:{n}", "image": encode(drawing(50 + n, (24, 24))),
                        "text": "Please write out the expression of the formula in the image using LaTeX format.",
                        "hints": {"dataset": r["dataset"], "question_type": r["question_type"]}, "ruolo": "sviluppo"})
    entries += [{"benchmark": "mathvista", "file": "data/testmini.parquet", "item": "testmini:1",
                 "image": originals["train-0000101"], "text": "Which formula is shown?", "ruolo": "sviluppo"},
                {"benchmark": "ocrbench", "file": "data/test-00000-of-00001.parquet", "item": "test-00000-of-00001:9",
                 "image": encode(flat), "text": "What is written?", "hints": {"dataset": "x", "question_type": "y"},
                 "ruolo": "test", "metrica": True},
                {"benchmark": "mmstar", "file": "mmstar.parquet", "item": "mmstar:7", "image": encode(crop),
                 "text": "Read the line.", "ruolo": "sviluppo"},
                {"benchmark": "mathvision", "file": "data/test.parquet", "item": "test:8",
                 "image": one_pixel_changed(originals["train-0000103"]), "text": "Compute the sum.", "ruolo": "test",
                 "metrica": True}]
    index_dir, panel_dir, isha = panel_fixture(tmp, entries + fillers())
    parquet(tmp / "ocrbench" / "test-00000-of-00001.parquet", ocr)
    return {"args": args, "index": index_dir, "panel": panel_dir, "isha": isha, "originals": originals,
            "ocrbench": tmp / "ocrbench" / "test-00000-of-00001.parquet", "flat": flat, "s": s,
            "msha": sha256_file(args.out / "manifest.json")}


def unimer_common(fx: dict) -> list:
    return ["--manifest", fx["args"].out, "--manifest-sha256", fx["msha"], "--index", fx["index"],
            "--index-sha256", fx["isha"]]


def test_unimer(tmp: Path) -> None:
    fx = unimer_fixture(tmp)
    a = fx["args"]
    common = unimer_common(fx)
    rgba = fx["originals"]["train-0000102"]
    with Image.open(io.BytesIO(rgba)) as im:
        naive = hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
    check("preparazione: l'originale RGBA ha i pixel dell'immagine del pannello solo con le trasparenze su bianco",
          dc.hash_image_bytes(rgba)[1] == dc.pixel_sha256(fx["flat"])
          and naive != hashlib.sha256(fx["flat"].tobytes()).hexdigest())
    scan = tmp / "scan"
    summary, why = attempt(pd.cmd_scan, args_of("scan", "unimer", *common, "--out", scan, "--workers", 2,
                                                "--ocrbench", fx["ocrbench"]))
    check("unimer, scansione completa eseguita", summary is not None, why)
    if summary is None:
        return
    rows = scan_rows(scan)
    st = summary["statistiche"]
    check("unimer, scansione: tutte le formule e le righe ritagliate, shard e zip con gli sha256 attesi",
          st["righe"] == {"righe": 5, "train": 58} and all(s["uguale"] for s in summary["shard"])
          and summary["zip"]["sha256_uguale"] is True and st["problemi"] == {}, (st["righe"], summary["zip"]))
    check("unimer, formule dei benchmark: UniMER-Test dal pannello e le due risposte HME di OCRBench",
          summary["ocrbench"]["domande_hme"] == 2 and summary["ocrbench"]["con_risposta"] == 2
          and summary["formule_dei_benchmark"]["unimer_test"] == {"elementi": 6, "con_formula": 6})
    r101, r102, r103 = rows[("train", "train-0000101")], rows[("train", "train-0000102")], rows[("train", "train-0000103")]
    check("unimer: originale con gli stessi pixel di un elemento del pannello, ricondotto all'elemento e alla posizione",
          any(d["tipo"] == "pixel" and d["immagine"] == "originale" and d["elemento"] == "mathvista/testmini:1"
              and d["posizione"] == 0 for d in r101["dettagli"]), r101["dettagli"])
    check("unimer: originale RGBA riconosciuto con la convenzione dell'indice (rgb-su-bianco/1)",
          r102["corrispondenze"].get("pixel") == 1, r102["corrispondenze"])
    check("unimer: un pixel diverso non è una prova e il pHash delle formule non si usa",
          r103["corrispondenze"] == {} and pd.record_outcome(r103, "unimer") == "nessuna_corrispondenza"
          and "phash" not in r103["originale"])
    line12, line18 = rows[("righe", "train-0000012")], rows[("righe", "train-0000018")]
    check("unimer: ritaglio di una riga identico a un elemento del pannello (confermato sul ritaglio, non sulla formula)",
          any(d["tipo"] == "pixel" and d["immagine"] == "ritaglio 0" for d in line12["dettagli"])
          and not rows[("train", "train-0000012")]["corrispondenze"].get("pixel") and len(line12["ritagli"]) == 2)
    check("unimer: formula e riga ritagliata con la stessa chiave sono righe distinte della scansione",
          ("train", "train-0000018") in rows and ("righe", "train-0000018") in rows
          and rows[("train", "train-0000018")]["corrispondenze"] == {}
          and any(d["tipo"] == "formula_lunga" and d["campo"] == "riga 0" and d["benchmark"] == "unimer_test"
                  for d in line18["dettagli"]))
    r17, r6 = rows[("train", "train-0000017")], rows[("train", "train-0000006")]
    check("unimer: formula lunga uguale a una risposta HME di OCRBench è un candidato",
          any(d["tipo"] == "formula_lunga" and d["benchmark"] == "ocrbench" for d in r17["dettagli"])
          and pd.record_outcome(r17, "unimer") == "candidato")
    check("unimer: formula breve uguale a una di OCRBench registrata, senza effetto",
          r6["corrispondenze"] == {"formula_breve": 1} and pd.record_outcome(r6, "unimer") == "nessuna_corrispondenza")
    ver = ["--panel", fx["panel"], "--overlap", a.overlap]
    code, why = attempt(pd.cmd_verifica, args_of("verifica", "unimer", *common, *ver, "--scan", scan, "--report",
                                                 tmp / "pd-verifica"))
    path = tmp / "pd-verifica/report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    check("unimer, verifica di compatibilità superata (indice, perimetro, shard, zip, identificativi, run2)",
          code == 0 and not report.get("falliti"), (why, report.get("falliti")))
    check("unimer, verifica: perimetro con tutti i ruoli e riconciliazione con run2",
          set(report.get("perimetro_pannello", {})) == {"sviluppo", "test_ammesso", "test_non_ammesso", "diagnostica",
                                                       "solo_decontaminazione"}
          and report.get("riconciliazione_run2", {}).get("immagine_del_test_ritrovata_dai_pixel") == 3,
          report.get("riconciliazione_run2"))
    # ---- manifest nuovo
    new, rep, summ = tmp / "manifest-2", tmp / "verifica-2", tmp / "riepilogo-2.json"
    man = [*common, *ver, "--scan", scan, "--new", new, "--report", rep, "--summary", summ, "--version", "2",
           "--check", a.check]
    code, why = attempt(pd.cmd_manifest, args_of("manifest", "unimer", *man))
    result = json.loads((rep / "verifica.json").read_text()) if (rep / "verifica.json").exists() else {}
    check("unimer, versione 2 scritta e verificata (genitore con la sua verifica, transizioni dalla scansione)",
          code == 0 and result.get("esito"), (why, failing(result) if result else None))
    if code != 0:
        return
    m1, m2 = mf.load(a.out), mf.load(new)
    old = {pd.ident(r): r for rows_ in m1.by_shard.values() for r in rows_.values()}
    cur = {pd.ident(r): r for rows_ in m2.by_shard.values() for r in rows_.values()}
    s = {i: r["stato"] for i, r in cur.items()}
    T, R = "train", "righe"
    check("unimer, stati: pixel identici escludono il gruppo (originale, RGBA, ritaglio con la sua formula)",
          s[(T, "train-0000101")] == s[(T, "train-0000102")] == s[(T, "train-0000012")] == s[(R, "train-0000012")]
          == "escluso" and old[(T, "train-0000101")]["stato"] == "ammesso")
    check("unimer, stati: formula lunga di OCRBench e riga con la formula lunga di UniMER-Test in quarantena, col "
          "gruppo", s[(T, "train-0000017")] == s[(T, "train-0000018")] == s[(R, "train-0000018")] == "quarantena")
    check("unimer, stati: somiglianza e formula breve senza effetto",
          s[(T, "train-0000103")] == s[(T, "train-0000006")] == s[(T, "train-0000007")] == "ammesso")
    check("unimer, stati: quarantene ed esclusioni del genitore conservate (formule lunghe del test, immagini del "
          "test, errori certi, conflitti), nessuna riammissione",
          s[(T, "train-0000008")] == s[(T, "train-0000015")] == s[(T, "train-0000004")] == "quarantena"
          and s[(T, "train-0000009")] == s[(T, "train-0000003")] == s[(T, "train-0000016")] == "escluso"
          and all(pd.SEVERITY[cur[i]["stato"]] >= pd.SEVERITY[r["stato"]] for i, r in old.items()))
    ev = [e for e in cur[(T, "train-0000008")]["evidenze"] if e.get("controllo") == pd.CHECK]
    check("unimer, evidenze: formula lunga del test distinta dall'immagine identica, motivi del genitore conservati",
          len(ev) == 1 and ev[0]["esito"] == "candidato" and ev[0]["elementi"][0]["tipo"] == "formula_lunga"
          and cur[(T, "train-0000008")]["motivo"][:len(old[(T, "train-0000008")]["motivo"])]
          == old[(T, "train-0000008")]["motivo"]
          and [e["esito"] for e in cur[(T, "train-0000009")]["evidenze"] if e.get("controllo") == pd.CHECK]
          == ["corrispondenza_confermata"])
    check("unimer, versione 2: split, gruppi e identità invariati, una evidenza nuova per record, scansione citata",
          all((cur[i]["split"], cur[i]["gruppo"], cur[i]["record_id"]) == (r["split"], r["gruppo"], r["record_id"])
              for i, r in old.items())
          and all(sum(e.get("controllo") == pd.CHECK for e in r["evidenze"]) == 1 for r in cur.values())
          and m2.meta["split"]["ereditato_da"]["manifest_sha256"] == m1.manifest_sha256
          and m2.meta["decontaminazione"]["scansione"]["records"]["sha256"] == sha256_file(scan / "records.jsonl.gz"))
    # ---- versioni nuove alterate: la verifica le rifiuta
    vm = [*common, "--scan", scan, "--check", a.check, "--overlap", a.overlap]

    def readmit(r):  # train-17: quarantena data dalla decontaminazione; train-8: quarantena del genitore (run2)
        if (r["insieme"], r["record"]) in ((T, "train-0000017"), (T, "train-0000008")):
            r["stato"] = "ammesso"

    bad = altered_manifest(new, tmp / "m2-riammesso", "unimer", readmit)
    code, _ = attempt(pd.cmd_verifica_manifest, args_of("verifica-manifest", "unimer", *vm, "--new", bad, "--report",
                                                        tmp / "vm-1"))
    res = json.loads((tmp / "vm-1/verifica.json").read_text())
    wrong = next((c["dettaglio"] for c in res["controlli"] if "il più grave" in c["controllo"]), {})
    check("unimer, versione alterata: quarantene riammesse (della decontaminazione e del genitore) rifiutate dalla "
          "verifica delle transizioni", code == 1 and wrong.get("casi") == 2
          and any("meno grave" in c for c in failing(res)), sorted(failing(res)))

    def drop_evidence(r):
        if (r["insieme"], r["record"]) == (R, "train-0000018"):
            r["evidenze"] = [e for e in r["evidenze"] if e.get("controllo") != pd.CHECK]

    bad = altered_manifest(new, tmp / "m2-senza-evidenza", "unimer", drop_evidence)
    code, _ = attempt(pd.cmd_verifica_manifest, args_of("verifica-manifest", "unimer", *vm, "--new", bad, "--report",
                                                        tmp / "vm-2"))
    res = json.loads((tmp / "vm-2/verifica.json").read_text())
    check("unimer, versione alterata: una riga ritagliata senza la sua evidenza è rifiutata",
          code == 1 and any("una e una sola evidenza" in c for c in failing(res)), sorted(failing(res)))
    # ---- arresti e incompatibilità
    wrong = [x if x != fx["msha"] else "0" * 64 for x in common]
    _, why = attempt(pd.cmd_scan, args_of("scan", "unimer", *wrong, "--out", tmp / "s-x", "--ocrbench", fx["ocrbench"]))
    check("unimer, arresto: manifest con uno sha256 diverso da quello atteso", "non è il manifest" in why, why[:120])
    wrong = [x if x != fx["isha"] else "f" * 64 for x in common]
    _, why = attempt(pd.cmd_scan, args_of("scan", "unimer", *wrong, "--out", tmp / "s-y", "--ocrbench", fx["ocrbench"]))
    check("unimer, arresto: indice con uno sha256 diverso da quello atteso", "indice con sha256" in why, why[:120])
    legacy = tmp / "legacy"
    legacy.mkdir()
    li, _, lsha = panel_fixture(legacy, fillers(), convention="grigio-diretto/0")
    _, why = attempt(pd.cmd_scan, args_of("scan", "unimer", "--manifest", a.out, "--manifest-sha256", fx["msha"],
                                          "--index", li, "--index-sha256", lsha, "--out", tmp / "s-z",
                                          "--ocrbench", fx["ocrbench"]))
    check("unimer, arresto: indice nella convenzione storica (senza sha256 dei pixel)", "rgb-su-bianco/1" in why,
          why[:120])
    # prova limitata: va come prova, non come scansione completa
    part = tmp / "scan-prova"
    attempt(pd.cmd_scan, args_of("scan", "unimer", *common, "--out", part, "--limit", 3, "--ocrbench", fx["ocrbench"]))
    code_p, _ = attempt(pd.cmd_verifica, args_of("verifica", "unimer", *common, *ver, "--scan", part, "--report",
                                                 tmp / "v-prova", "--partial"))
    _, why = attempt(pd.cmd_manifest, args_of("manifest", "unimer", *common, *ver, "--scan", part, "--new",
                                              tmp / "m-prova", "--report", tmp / "r-prova", "--summary",
                                              tmp / "p.json", "--version", "2", "--check", a.check))
    check("unimer, prova limitata: verificata come prova, rifiutata per scrivere un manifest (copertura incompleta)",
          code_p == 0 and "copertura" in why and not (tmp / "m-prova" / "manifest.json").exists(), why[:200])

    def swap(rows_):
        for r in rows_:
            if (r["insieme"], r["record"]) == (R, "train-0000012"):
                r["insieme"] = T

    swapped = rewrite_scan(scan, tmp / "scan-scambiata", swap)
    _, why = attempt(pd.cmd_manifest, args_of("manifest", "unimer", *common, *ver, "--scan", swapped, "--new",
                                              tmp / "m-sc", "--report", tmp / "r-sc", "--summary", tmp / "sc.json",
                                              "--version", "2", "--check", a.check))
    check("unimer, identità: una riga ritagliata spacciata per la sua formula è rifiutata",
          "identificativi" in why and not (tmp / "m-sc" / "manifest.json").exists(), why[:200])
    # shard alterato dopo la costruzione: JSON di una riga ritagliata cambiato
    lines_tar = a.shards / "lines" / "unimer-lines-00000.tar"
    items = [(k, img, dict(meta, lines=[[0, 4], [6, 10]]) if k == "train-0000013" else meta)
             for k, img, meta in fx["s"]["righe"][0][1]]
    lines_tar.unlink()
    tu.write_tar(lines_tar, items)
    altered = tmp / "scan-shard-alterato"
    attempt(pd.cmd_scan, args_of("scan", "unimer", *common, "--out", altered, "--ocrbench", fx["ocrbench"]))
    _, why = attempt(pd.cmd_manifest, args_of("manifest", "unimer", *common, *ver, "--scan", altered, "--new",
                                              tmp / "m-al", "--report", tmp / "r-al", "--summary", tmp / "al.json",
                                              "--version", "2", "--check", a.check))
    check("unimer, shard alterato dopo la costruzione: sha256 e digest diversi, nessun manifest",
          "shard" in why and "identificativi" in why and not (tmp / "m-al" / "manifest.json").exists(), why[:240])


# ---------------------------------------------------------------- 3. MAVIS

CARRIERS = {**{f"d0-{91000 + n}": f"Variant {n}. {SPECIFIC}" for n in range(6, 26)},
            "d0-9323": SPECIFIC, "d0-90001": SPECIFIC, "d0-90002": SPECIFIC,  # duplicati; immagine in conflitto
            "d1-95001": SPECIFIC, "d1-95002": SPECIFIC, "d1-95003": f"Variant 95. {SPECIFIC}",
            "d3-97003": f"Variant 97. {SPECIFIC}"}
GROUP5 = ("d1-95001", "d1-95002", "d1-95003", "d1-95004")  # stesso diagram_generation: un gruppo


def mavis_spec() -> dict:
    """Il fixture di test_mavis_manifest con le domande (text_lite_question) dei casi della correzione:
    - SPECIFIC, problema specifico di un benchmark, in 27 record (più di 25): le 20 varianti dei rettangoli, i duplicati
      d0-9323 e d0-90001 (pixel identici a MathVerse), d0-90002 (in conflitto con d0-48357), d3-97003 e il gruppo
      GROUP5 con un duplicato (d1-95002, stesso PNG), una variante (d1-95003) e un membro senza la domanda (d1-95004);
    - d3-97000: figura con i punti A-J, enumerazione nella descrizione del loader (coperta dalla regola);
    - d0-97002: la stessa enumerazione in una domanda (non coperta);
    - d3-97003: enumerazione nella descrizione e SPECIFIC in una domanda (candidato)."""
    s = tm.spec()
    cap, gen = tm.REAL["d0-9323"]

    def asking(item, text):
        key, image, meta = item
        meta = json.loads(json.dumps(meta))
        meta["text_en"]["text_lite_question"] = text
        return key, image, meta

    def variant(n):
        return cap.replace("Side AB is 3", f"Side AB is {n}"), gen.replace("side ab is 3", f"side ab is {n}")

    c40, g40 = variant(40)
    cap60 = CAP10.replace("Angle BAD equals 90 degrees.", "Angle BAD equals 60 degrees.")
    gen60 = GEN10.replace("angle bad equals 90 degrees.", "angle bad equals 60 degrees.")
    extra = [tm.sample("d1-95001", c40, g40, tm.png(80)), tm.sample("d1-95002", c40, g40, tm.png(80)),
             tm.sample("d1-95003", c40, g40, tm.png(81)), tm.sample("d1-95004", c40, g40, tm.png(82)),
             tm.sample("d3-97000", CAP10, GEN10, tm.png(70)),
             asking(tm.sample("d0-97002", *variant(41), tm.png(72)), LETTER_QUESTION),
             tm.sample("d3-97003", cap60, gen60, tm.png(73))]
    shards = []
    for n, (name, items) in enumerate(s["shards"]):
        items = items + (extra if n == len(s["shards"]) - 1 else [])
        shards.append((name, [asking(it, CARRIERS[it[0]]) if it[0] in CARRIERS else it for it in items]))
    s["shards"] = shards
    return s


def mavis_fixture(tmp: Path) -> dict:
    s = mavis_spec()
    originals = {}
    for _, items in s["shards"]:
        for key, image, meta in items:
            seed = int.from_bytes(hashlib.sha256(image).digest()[:4], "little")  # PNG uguali, originali uguali
            originals[key] = (meta["zip_dir"], encode(drawing(seed, (meta["orig_width"], meta["orig_height"]))))

    def tamper(dirs):
        real_zip(tmp, tmp / "snapshot", "RuleBaseGeo.zip",
                 {f"{d}/{d.rsplit('/', 1)[1]}_caption.jpg": data for d, data in originals.values()}, dirs["shards"])

    args = tm.write_fixture(tmp, s, tamper)
    mm.run(args)
    near = one_pixel_changed(originals["d0-91006"][1])
    h1, p1, _ = dc.hash_image_bytes(originals["d0-91006"][1])
    h2, p2, _ = dc.hash_image_bytes(near)
    words = " ".join(MULTI_CAP.split()[-16:])  # misure e lati propri della figura a più forme
    entries = [{"benchmark": "mathverse", "file": "testmini.parquet", "item": "testmini:1",
                "image": originals["d0-9323"][1], "text": "As shown in the figure, find the area.",
                "hints": {"problem_index": "1", "version": "Vision Only"}, "ruolo": "sviluppo"},
               {"benchmark": "mathvision", "file": "data/test.parquet", "item": "test:2", "image": near,
                "text": "What is the perimeter of the rectangle?", "ruolo": "test", "metrica": True},
               {"benchmark": "geoperception", "file": "data/train.parquet", "item": "train:3",
                "image": encode(drawing(7777, (50, 50))), "text": f"Consider the figure. {words} What is x?",
                "ruolo": "diagnostica"},
               # problema specifico, nelle domande di 27 record MAVIS; nessuna immagine in comune
               {"benchmark": "mathvision", "file": "data/test.parquet", "item": "test:753", "image": None,
                "text": SPECIFIC + "\n<image1>", "ruolo": "test", "metrica": True},
               # dodici lettere in fila: la stessa sequenza della riga Points di una figura con dieci punti
               {"benchmark": "aime25", "file": "aime2025-I.jsonl", "item": "aime2025-I:6", "image": None,
                "text": LETTERS, "ruolo": "sviluppo"}]
    index_dir, panel_dir, isha = panel_fixture(tmp, entries + fillers())
    mv = [tmp / "mathverse" / "testmini.parquet", tmp / "mathverse" / "testmini_text_only.parquet"]
    parquet(mv[0], [{"sample_index": "1", "problem_index": "1", "metadata": {"source": "GeoQA", "split": "testmini"}},
                    {"sample_index": "2", "problem_index": "2", "metadata": {"source": "MAVIS-prova", "split": "x"}}])
    parquet(mv[1], [{"sample_index": "3", "problem_index": "3", "metadata": {"source": "Geometry3K", "split": "y"}}])
    hist = tmp / "storiche" / "p6-mavis"
    hist.mkdir(parents=True)
    (hist / "exclude.txt").write_text("d0-91006\nd0-91007\n")
    (hist / "exclude_final.txt").write_text("")
    return {"args": args, "index": index_dir, "panel": panel_dir, "isha": isha, "mathverse": mv, "hist": hist,
            "msha": sha256_file(args.out / "manifest.json"), "near": (dc.hamming(h1, h2), p1 != p2)}


def last(rec: dict) -> dict:
    return pd.last_check(rec)


def by_key(m: mf.Manifest) -> dict[str, dict]:
    return {k: rec for rows_ in m.by_shard.values() for k, rec in rows_.items()}


SPECIFIC_ONLY = [k for k in CARRIERS if k not in ("d0-9323", "d0-90001")]  # senza pixel identici


def test_mavis(tmp: Path) -> None:
    fx = mavis_fixture(tmp)
    a = fx["args"]
    check("preparazione MAVIS: la copia con un pixel diverso ha pixel diversi e pHash entro 4 bit",
          fx["near"][1] and fx["near"][0] <= 4, fx["near"])
    v1 = by_key(mf.load(a.out))
    check("preparazione MAVIS: GROUP5 è un solo gruppo coi quattro record ammessi; d3-97000 ha la verità e la descrizione "
          "canonica", len({v1[k]["gruppo"] for k in GROUP5}) == 1 and all(v1[k]["stato"] == "ammesso" for k in GROUP5)
          and v1["d3-97000"]["descrizione_sha256"] is not None,
          {k: (v1[k]["gruppo"][:8], v1[k]["stato"]) for k in (*GROUP5, "d3-97000")})
    mavis_new_rules(tmp, fx, v1)
    mavis_riesame(tmp, fx)


def mavis_new_rules(tmp: Path, fx: dict, v1: dict) -> None:
    """Percorso completo col codice nuovo: scansione panel-decontam-scan/2, verifica, versione 2 con le regole esplicite."""
    a = fx["args"]
    common = ["--manifest", a.out, "--manifest-sha256", fx["msha"], "--index", fx["index"], "--index-sha256", fx["isha"]]
    scan = tmp / "scan"
    summary, why = attempt(pd.cmd_scan, args_of("scan", "mavis", *common, "--out", scan, "--workers", 2,
                                                "--mathverse", *fx["mathverse"]))
    check("mavis, scansione completa eseguita", summary is not None, why)
    if summary is None:
        return
    rows = scan_rows(scan)
    r = {k: rows[("train", k)] for k in ("d0-9323", "d0-90001", "d0-91006", "d2-90010", "d0-91007")}
    check("mavis: caption.jpg originale identico a un elemento di MathVerse (anche il duplicato), non il PNG dello shard",
          all(any(d["tipo"] == "pixel" and d["immagine"] == "originale" and d["elemento"] == "mathverse/testmini:1"
                  for d in r[k]["dettagli"]) for k in ("d0-9323", "d0-90001"))
          and not any(d["tipo"] == "pixel" and d["immagine"] == "png" for d in r["d0-9323"]["dettagli"]))
    check("mavis: la copia con un pixel diverso è solo un candidato pHash, non una prova",
          r["d0-91006"]["corrispondenze"].get("phash", 0) >= 1 and not r["d0-91006"]["corrispondenze"].get("pixel")
          and pd.record_outcome(r["d0-91006"], "mavis") == "candidato")
    check("mavis: 10-gram della didascalia in comune con un testo del pannello, col campo e l'n-gram",
          any(d["tipo"] == "testo" and d["campo"] == "caption" and d["elemento"] == "geoperception/train:3"
              and len(d["ngram"].split()) == 10 for d in r["d2-90010"]["dettagli"]))
    tf = summary["statistiche"]["testi_per_campo"]
    check("mavis: campi confrontati dichiarati (caption e diagram_generation in ogni record, domande e testi del loader)",
          tf["caption"]["testi"] == tf["diagram_generation"]["testi"] == len(v1) and tf["domande_del_loader"]["testi"] > 0
          and tf["text_lite_question"]["testi"] == len(CARRIERS) + 1 and tf["text_dominant_question"]["testi"] == 0, tf)
    prov = summary["mathverse_provenienze"]
    check("mavis: provenienze dichiarate di MathVerse lette, legame con MAVIS segnalato",
          prov["con_mavis_o_rulebasegeo"] == ["2"] and prov["fonti"].get("GeoQA") == 1, prov)

    def texts_of(k):
        return [d for d in rows[("train", k)]["dettagli"] if d["tipo"] == "testo"]

    check("mavis, scansione: il problema specifico ritrovato nella domanda di tutti i 27 record, e solo lì",
          all(any(d["campo"] == "text_lite_question" and d["elemento"] == "mathvision/test:753" for d in texts_of(k))
              for k in CARRIERS)
          and sum(any(d["elemento"] == "mathvision/test:753" for d in texts_of(k)) for k in v1) == len(CARRIERS) == 27)
    desc = rows[("train", "d3-97000")].get("testi_delle_regole") or {}
    check("mavis, scansione: enumerazione A-J nella descrizione del loader (d3-97000, d3-97003) e in una domanda "
          "(d0-97002); il testo della regola registrato solo per la descrizione, con lo sha256 del manifest",
          [d["campo"] for d in texts_of("d3-97000")] == ["descrizione_del_loader"]
          and texts_of("d3-97000")[0]["ngram_tutti"] == [ENUM]
          and any(d["campo"] == "text_lite_question" and ENUM in d["ngram_tutti"] for d in texts_of("d0-97002"))
          and set(desc) == {"descrizione_del_loader"} and rows[("train", "d0-97002")]["testi_delle_regole"] == {}
          and mf.sha256_text(desc.get("descrizione_del_loader", "")) == v1["d3-97000"]["descrizione_sha256"],
          [(d["campo"], d["ngram_tutti"][:1]) for d in texts_of("d3-97000")])
    ti = summary["testi_in_comune"]
    freq = {x["ngram"]: x for x in ti["ngram_piu_frequenti"]}
    top = freq.get(SPECIFIC_GRAMS[0]) or {}
    check("mavis, riepilogo della scansione: frequenze solo per ordinare i casi (record, campi, coperture), regole "
          "dichiarate, nessuna soglia di frequenza", summary["schema"] == pd.SCAN_SCHEMA
          and ti["regole_testuali_sha256"] == pd.TEXT_RULES_SHA256 and top.get("record") == 27
          and top.get("corrispondenze_per_regola") == {"non_coperta": 27}
          and (freq.get(ENUM) or {}).get("corrispondenze_per_regola") == {"enumerazione_punti_loader": 2, "non_coperta": 1}
          and "max_df_sorgente" not in ti, (top, freq.get(ENUM)))
    ver = ["--panel", fx["panel"], "--historical", fx["hist"]]
    code, why = attempt(pd.cmd_verifica, args_of("verifica", "mavis", *common, *ver, "--scan", scan, "--report",
                                                 tmp / "pd-verifica"))
    path = tmp / "pd-verifica/report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    hist = (report.get("candidati_storici") or [{}])[0]
    check("mavis, verifica superata (descrizioni del loader legate al manifest); candidati storici solo confrontati",
          code == 0 and not report.get("falliti") and hist.get("candidati_storici") == 2
          and hist.get("storici_anche_candidati_nuovi") == 2, (why, report.get("falliti"), hist))
    new, rep, summ = tmp / "manifest-2", tmp / "verifica-2", tmp / "riepilogo-2.json"
    code, why = attempt(pd.cmd_manifest, args_of("manifest", "mavis", *common, *ver, "--scan", scan, "--new", new,
                                                 "--report", rep, "--summary", summ, "--version", "2",
                                                 "--evidence", a.evidence))
    result = json.loads((rep / "verifica.json").read_text()) if (rep / "verifica.json").exists() else {}
    check("mavis, versione 2 con le regole esplicite scritta e verificata", code == 0 and result.get("esito"),
          (why, failing(result) if result else None))
    if code != 0:
        return
    m2 = mf.load(new)
    cur = by_key(m2)
    s = {k: rec["stato"] for k, rec in cur.items()}
    tasks = {k: {t["compito"]: t["stato"] for t in rec["compiti"]} for k, rec in cur.items()}
    check("mavis, stati: pixel identici escludono il gruppo; pHash, didascalia e problema specifico in quarantena",
          s["d0-9323"] == s["d0-90001"] == "escluso" and s["d0-91006"] == s["d2-90010"] == "quarantena"
          and all(s[k] == "quarantena" for k in SPECIFIC_ONLY))
    e95 = last(cur["d1-95004"])
    check("mavis: problema specifico ripetuto in 27 record (più di 25), senza corrispondenze visive: ogni record resta "
          "candidato, anche duplicati e varianti dello stesso gruppo, e il gruppo va in quarantena col membro senza la "
          "domanda", all(last(cur[k])["esito"] == "candidato" for k in SPECIFIC_ONLY)
          and s["d1-95004"] == "quarantena" and e95["esito"] == "nessuna_corrispondenza"
          and e95["esito_gruppo"] == "candidato"
          and set(e95["cause_nel_gruppo"]) == {"train/d1-95001", "train/d1-95002", "train/d1-95003"}, e95)
    e0, e2, e3 = last(cur["d3-97000"]), last(cur["d0-97002"]), last(cur["d3-97003"])
    check("mavis, enumerazione: senza effetto nella descrizione del loader (stato del candidato), candidata nella "
          "domanda; una corrispondenza specifica accanto a quella coperta lascia il record candidato",
          e0["esito"] == "nessuna_corrispondenza" and s["d3-97000"] == v1["d3-97000"]["stato"] and e0["testo_coperto"] == 1
          and any(x["tipo"] == "testo_coperto" and x["regole"] == ["enumerazione_punti_loader"] for x in e0["elementi"])
          and e2["esito"] == "candidato" and s["d0-97002"] == "quarantena" and e2["testo_coperto"] == 0
          and e3["esito"] == "candidato" and e3["testo_coperto"] == 1 and s["d3-97003"] == "quarantena", (e0, e3))
    check("mavis, stati precedenti conservati: quarantene del candidato (d0-90003, conflitto d0-48357 e d0-90002), "
          "esclusioni, nessuna riammissione", s["d0-90003"] == v1["d0-90003"]["stato"] == "quarantena"
          and s["d0-48357"] == s["d0-90002"] == "quarantena"
          and all(pd.SEVERITY[cur[k]["stato"]] >= pd.SEVERITY[o["stato"]] for k, o in v1.items()))
    check("mavis, compiti: seguono il record (anche nel gruppo propagato), shape_type mai ammessa",
          set(tasks["d0-9323"].values()) == {"escluso"} and set(tasks["d1-95004"].values()) == {"quarantena"}
          and all(t["stato"] != "ammesso" for rec in cur.values() for t in rec["compiti"] if t["compito"] == "shape_type")
          and all(cur[k]["compiti"] == pd.task_states(v1[k]["compiti"], s[k]) for k in v1))
    dec = m2.meta["decontaminazione"]
    check("mavis, manifest: decisioni panel-decontam/2, regole testuali registrate, nessuna regola di frequenza",
          dec["versione"] == pd.SCHEMA and dec["regole_testuali"] == {"regole": pd.TEXT_RULES, "sha256": pd.TEXT_RULES_SHA256}
          and "testo_di_rito" not in dec and all(last(rec)["versione"] == pd.SCHEMA for rec in cur.values()))
    vm = [*common, "--scan", scan, "--evidence", a.evidence]
    target = next((k for k, rec in cur.items() if rec["stato"] == "ammesso"
                   and any(t["compito"] == "shape_type" for t in rec["compiti"])), None)
    check("preparazione: un record ammesso con il compito shape_type da alterare", target is not None)

    def shape_admitted(rec):
        if rec["record"] == target:
            for t in rec["compiti"]:
                if t["compito"] == "shape_type":
                    t["stato"] = "ammesso"

    bad = altered_manifest(new, tmp / "m2-shape", "mavis", shape_admitted)
    code, _ = attempt(pd.cmd_verifica_manifest, args_of("verifica-manifest", "mavis", *vm, "--new", bad, "--report",
                                                        tmp / "vm-1"))
    res = json.loads((tmp / "vm-1/verifica.json").read_text())
    check("mavis, versione alterata: shape_type ammessa è rifiutata", code == 1
          and any(c.startswith("compiti") for c in failing(res)), sorted(failing(res)))

    def forged(rows_):  # descrizione della scansione diversa da quella del manifest
        for x in rows_:
            if x["record"] == "d3-97003":
                x["testi_delle_regole"]["descrizione_del_loader"] += "\n"

    forged_scan = rewrite_scan(scan, tmp / "scan-descrizione", forged)
    _, why = attempt(pd.cmd_manifest, args_of("manifest", "mavis", *common, *ver, "--scan", forged_scan, "--new",
                                              tmp / "m-fd", "--report", tmp / "r-fd", "--summary", tmp / "fd.json",
                                              "--version", "2", "--evidence", a.evidence))
    check("mavis, legame: una descrizione della scansione senza lo sha256 del manifest non copre nulla e ferma il "
          "manifest", "testo: per ogni corrispondenza" in why and not (tmp / "m-fd" / "manifest.json").exists(),
          why[:200])


def mavis_riesame(tmp: Path, fx: dict) -> None:
    """Riesame di un candidato-2 costruito sul fixture dal codice di OLD_COMMIT (filtro di frequenza)."""
    old = old_module()
    a = fx["args"]
    common = ["--manifest", a.out, "--manifest-sha256", fx["msha"], "--index", fx["index"], "--index-sha256", fx["isha"]]
    scan1 = tmp / "scan-v1"
    argv = ["scan", "mavis", *common, "--out", scan1, "--workers", 2, "--mathverse", *fx["mathverse"]]
    summary, why = attempt_any(old.cmd_scan, old.parser().parse_args([str(x) for x in argv]))
    check(f"riesame, preparazione: scansione col codice di {OLD_COMMIT} (panel-decontam-scan/1, senza testi della regola)",
          summary is not None and summary["schema"] == "panel-decontam-scan/1", why)
    if summary is None:
        return
    v2dir = tmp / "manifest-2-v1"
    argv = ["manifest", "mavis", *common, "--panel", fx["panel"], "--scan", scan1, "--new", v2dir, "--report",
            tmp / "verifica-2-v1", "--summary", tmp / "riepilogo-2-v1.json", "--version", "2", "--evidence", a.evidence]
    code, why = attempt_any(old.cmd_manifest, old.parser().parse_args([str(x) for x in argv]))
    check(f"riesame, preparazione: candidato-2 costruito e verificato col filtro di {OLD_COMMIT}", code == 0, why)
    if code != 0:
        return
    m2 = mf.load(v2dir)
    old2 = by_key(m2)
    s2 = {k: rec["stato"] for k, rec in old2.items()}
    check(f"regressione: col filtro di {OLD_COMMIT} il problema ripetuto in 27 record non è candidato e i suoi record "
          "restano ammessi (il test del problema ripetuto fallisce); l'enumerazione, in soli 3 record, è candidata",
          all(last(old2[k])["esito"] == "nessuna_corrispondenza" for k in SPECIFIC_ONLY
              if k not in ("d0-91006", "d3-97003"))  # pHash; enumerazione nella descrizione
          and s2["d0-91007"] == s2["d1-95001"] == s2["d1-95004"] == "ammesso"
          and last(old2["d3-97000"])["esito"] == "candidato" and s2["d3-97000"] == "quarantena",
          {k: s2[k] for k in ("d0-91007", "d1-95001", "d1-95004", "d3-97000")})
    _, why = attempt(pd.cmd_manifest, args_of("manifest", "mavis", *common, "--panel", fx["panel"], "--scan", scan1,
                                              "--new", tmp / "m-x", "--report", tmp / "r-x", "--summary", tmp / "x.json",
                                              "--version", "2", "--evidence", a.evidence))
    check("riesame: una scansione panel-decontam-scan/1 non decide un manifest col codice nuovo (si riesamina)",
          "panel-decontam-scan/2" in why and not (tmp / "m-x" / "manifest.json").exists(), why[:200])
    v2sha = sha256_file(v2dir / "manifest.json")
    rargs = ["--origine", a.out, "--origine-sha256", fx["msha"], "--manifest", v2dir, "--manifest-sha256", v2sha,
             "--index", fx["index"], "--index-sha256", fx["isha"], "--panel", fx["panel"], "--scan", scan1,
             "--evidence", a.evidence]
    new2 = tmp / "manifest-2"  # deciso col codice nuovo: non si riesamina
    if (new2 / "manifest.json").exists():
        alt = [x if x not in (v2dir, v2sha, scan1) else {v2dir: new2, v2sha: sha256_file(new2 / "manifest.json"),
                                                         scan1: tmp / "scan"}[x] for x in rargs]
        _, why = attempt(pd.cmd_riesame, args_of("riesame", "mavis", *alt, "--out", tmp / "rs-x", "--new", tmp / "m3-x",
                                                 "--version", "3", "--report", tmp / "r3-x", "--summary", tmp / "s3x"))
        check("riesame: una versione già decisa con panel-decontam/2 non si riesamina", "panel-decontam/1" in why
              and not (tmp / "m3-x" / "manifest.json").exists(), why[:200])
    v3dir, rep3, out = tmp / "manifest-3", tmp / "verifica-3", tmp / "riesame"
    code, why = attempt(pd.cmd_riesame, args_of("riesame", "mavis", *rargs, "--out", out, "--new", v3dir, "--version",
                                                "3", "--report", rep3, "--summary", tmp / "riepilogo-3.json"))
    result = json.loads((rep3 / "verifica.json").read_text()) if (rep3 / "verifica.json").exists() else {}
    check("riesame: versione 3 scritta e verificata (genitore con la sua regola storica e candidato iniziale, catena, "
          "decisioni ricalcolate)", code == 0 and result.get("esito"), (why, failing(result) if result else None))
    if code != 0:
        return
    m3 = mf.load(v3dir)
    cur = by_key(m3)
    s3 = {k: rec["stato"] for k, rec in cur.items()}
    rs = json.loads((out / "summary.json").read_text())
    with_text = sum(any(d["tipo"] == "testo" for d in r_["dettagli"]) for r_ in scan_rows(scan1).values())
    check("riesame: letti solo i JSON dei record con corrispondenze testuali, saltando le immagini",
          rs["letture"]["json_letti"] == rs["conteggi"]["record"]["con_corrispondenze_testuali"] == with_text
          and rs["letture"]["byte_letti"] < rs["letture"]["byte_degli_shard_aperti"], rs["letture"])
    check("riesame: il problema ripetuto, che candidato-2 ignorava per frequenza, è candidato; i record ammessi e il "
          "loro gruppo vanno in quarantena (lo stesso test, che falliva col filtro precedente, passa)",
          all(last(cur[k])["esito"] == "candidato" for k in SPECIFIC_ONLY)
          and all(s3[k] == "quarantena" for k in SPECIFIC_ONLY) and s3["d1-95004"] == "quarantena")
    e0 = last(cur["d3-97000"])
    check("riesame: nessuna riammissione; d3-97000 (enumerazione coperta, prima candidato) resta in quarantena col motivo "
          "della transizione meno grave; esclusioni conservate", e0["esito"] == "nessuna_corrispondenza"
          and e0["esito_precedente"] == "candidato" and s3["d3-97000"] == "quarantena"
          and cur["d3-97000"]["motivo"][-1] == pd.REASON_RIESAME["meno_grave"]
          and "train/d3-97000" in rs["stati_mantenuti_dalla_sola_decisione_precedente"]
          and s3["d0-9323"] == s3["d0-90001"] == "escluso"
          and all(pd.SEVERITY[cur[k]["stato"]] >= pd.SEVERITY[o["stato"]] for k, o in old2.items()), e0)
    check("riesame: evidenze e motivi del genitore conservati; un motivo nuovo solo dove l'esito del gruppo cambia, anche "
          "senza cambio di stato (d0-48357, già in quarantena)",
          all(cur[k]["evidenze"][:len(o["evidenze"])] == o["evidenze"] and cur[k]["motivo"][:len(o["motivo"])] == o["motivo"]
              for k, o in old2.items())
          and cur["d0-48357"]["motivo"][-1] == pd.REASON_RIESAME["piu_grave"] and s3["d0-48357"] == "quarantena"
          and cur["d0-91006"]["motivo"] == old2["d0-91006"]["motivo"])
    check("riesame: stessi gruppi e split del genitore; compiti dal nuovo stato, shape_type mai ammessa",
          all((cur[k]["gruppo"], cur[k]["split"]) == (o["gruppo"], o["split"]) for k, o in old2.items())
          and all(cur[k]["compiti"] == pd.task_states(o["compiti"], s3[k]) for k, o in old2.items())
          and all(t["stato"] != "ammesso" for rec in cur.values() for t in rec["compiti"] if t["compito"] == "shape_type"))
    changes = {c["record"]: c for c in rs["cambi"]}
    expected = {k for k, o in old2.items() if (last(o)["esito"], last(o)["esito_gruppo"], o["stato"])
                != (last(cur[k])["esito"], last(cur[k])["esito_gruppo"], cur[k]["stato"])}
    check("riesame: elenco completo dei cambi, con la causa e la decisione precedente accanto",
          set(changes) == expected and changes["d1-95004"]["causa"].startswith("esito del gruppo")
          and changes["d0-91007"]["causa"].startswith("esito proprio")
          and changes["d0-91007"]["corrispondenze_cambiate"][0]["prima"] == "ignorata_per_frequenza"
          and changes["d0-91007"]["stato"] == ["ammesso", "quarantena"], sorted(set(changes) ^ expected))
    row_ = next((x for x in rs["ngram"] if x["ngram"] == SPECIFIC_GRAMS[0]), {})
    case = next((c for c in rs["casi_testuali_informativi"] if c["elemento"] == "mathvision/test:753"), {})
    check("riesame: frequenze solo per ordinare (27 record, 24 gruppi, 23 testi distinti) e casi coi contesti",
          (row_.get("record"), row_.get("gruppi"), row_.get("testi_distinti")) == (27, 24, 23)
          and row_.get("ignorato_dalla_regola_precedente") is True and case.get("record") == 27
          and all(x["parole_comuni_consecutive"] == 25 for x in case.get("casi") or [{}])
          and rs["elementi_dei_casi"]["mathvision/test:753"]["testi"][0].startswith("Two equal squares"),
          (row_, {k: v for k, v in case.items() if k != "casi"}))
    check("riesame: parentela documentata (candidato iniziale, candidato-2, versione 3) e regola precedente citata",
          [c["percorso"] for c in m3.meta["catena_delle_versioni"]] == [pd.rel(a.out), pd.rel(v2dir), pd.rel(v3dir)]
          and m3.meta["split"]["ereditato_da"]["manifest_sha256"] == m2.manifest_sha256
          and m3.meta["decontaminazione"]["decisione_precedente"]["regola"] == m2.meta["decontaminazione"]["testo_di_rito"])
    # ---- verifiche che devono rifiutare
    _, why = attempt(pd.cmd_verifica_manifest, args_of("verifica-manifest", "mavis", "--manifest", v2dir,
                                                        "--manifest-sha256", v2sha, "--scan", scan1, "--evidence",
                                                        a.evidence, "--new", v3dir, "--report", tmp / "vm-catena"))
    res = json.loads((tmp / "vm-catena/verifica.json").read_text())
    check("riesame: il verificatore della costruzione non si applica a un genitore derivato (stati della "
          "decontaminazione): verifica-manifest rifiuta e rimanda alla catena",
          not res["esito"] and any(c.startswith("genitore: candidato iniziale") for c in failing(res)), sorted(failing(res)))
    vr = [*rargs, "--riesame", out]

    def readmit(rec):  # d3-97000: quarantena del genitore che il riesame non toglie; d0-91008: quarantena nuova
        if rec["record"] in ("d3-97000", "d0-91008"):
            rec["stato"] = "ammesso"

    bad = altered_manifest(v3dir, tmp / "m3-riammesso", "mavis", readmit)
    code, _ = attempt(pd.cmd_verifica_riesame, args_of("verifica-riesame", "mavis", *vr, "--new", bad, "--report",
                                                       tmp / "vr-1"))
    res = json.loads((tmp / "vr-1/verifica.json").read_text())
    wrong = next((c["dettaglio"] for c in res["controlli"] if "il più grave" in c["controllo"]), {})
    check("riesame, versione alterata: la quarantena del genitore riammessa e quella nuova tolta sono rifiutate (stato, "
          "nessuna riammissione, elenco dei cambi)", code == 1 and wrong.get("casi") == 2
          and any(c.startswith("nessuna riammissione") for c in failing(res))
          and any("elenco dei cambi" in c for c in failing(res)), sorted(failing(res)))
    out_bad, v3_bad = tmp / "riesame-alterato", tmp / "m3-descrizione"
    shutil.copytree(out, out_bad)
    lines = [json.loads(x) for x in gzip.decompress((out_bad / "decisioni.jsonl.gz").read_bytes()).decode().splitlines()]
    for x in lines:
        if x["record"] == "d3-97000":
            x["descrizione_del_loader"] = x["descrizione_del_loader"].replace("Points: ", "Points: A, ") + " "
    (out_bad / "decisioni.jsonl.gz").unlink()
    dsha, _ = um.write_records_gz(out_bad / "decisioni.jsonl.gz", (json.dumps(x, sort_keys=True) for x in lines))
    shutil.copytree(v3dir, v3_bad)
    meta = json.loads((v3_bad / "manifest.json").read_text())
    meta["decontaminazione"]["riesame"]["decisioni"]["sha256"] = dsha
    (v3_bad / "manifest.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    code, _ = attempt(pd.cmd_verifica_riesame, args_of("verifica-riesame", "mavis", *rargs, "--riesame", out_bad,
                                                       "--new", v3_bad, "--report", tmp / "vr-2"))
    res = json.loads((tmp / "vr-2/verifica.json").read_text())
    check("riesame, descrizione alterata: senza lo sha256 del manifest non copre nulla e la versione è rifiutata",
          code == 1 and any(c.startswith("testo: una descrizione del riesame") for c in failing(res)), sorted(failing(res)))


def main() -> None:
    test_rules()
    test_decisions()
    for test in (test_unimer, test_mavis):
        with tempfile.TemporaryDirectory(prefix="panel_decontam_") as tmp:
            test(Path(tmp))
    print(f"\ntest_panel_decontam: {len(DONE) - len(FAILS)} controlli superati su {len(DONE)}"
          + (f", FALLITI: {len(FAILS)}" if FAILS else ""))
    if FAILS:
        for f in FAILS:
            print(" -", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
