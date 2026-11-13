"""Backup (scripts/backup.py): inventario, piano, creazione, verifica e prova di ripristino su un albero costruito qui.
Nessun dato reale. Solo in un job SLURM (slurm/tests_cpu.sbatch tests/test_backup.py): usa git e GNU tar.

Controlla:
- inventario: file nascosti e ignorati, cartelle vuote, `trasferimento/` esclusa, link relativi, assoluti, verso
  l'esterno e interrotti, hard link fra cache/uv e venvs, nomi con spazi e accenti, segreti senza testo;
- piano: categorie (codice, metadati, shard, HF e outputs grandi, ambienti uniti dagli hard link), nucleo, ordine,
  volumi entro il limite, file grande in un volume a sé, arresti per secrets/ non vuota e segreti forti;
- creazione: parti con nomi e hash, rilettura completa, elenco di GNU tar, ripresa senza rifare i volumi pronti,
  staging limitato, avviso per un file cambiato dopo l'inventario;
- ripristino: cat | tar -xp ricostruisce file, permessi, date, link e hard link; sha256sum -c; una parte alterata
  fa fallire la prova;
- aggiornamento: con --precedente le categorie invariate si riusano e solo le cambiate hanno volumi nuovi.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("SLURM_JOB_ID", "test")
import backup as bk  # noqa: E402

FAILS: list[str] = []
DONE: list[str] = []
KIB = 1024


def check(name: str, cond, detail="") -> None:
    DONE.append(name)
    if not cond:
        FAILS.append(f"{name}: {detail}")
        print(f"FALLITO {name}: {detail}")


def write(p: Path, data: bytes | str, mode: int | None = None) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    if mode is not None:
        os.chmod(p, mode)
    return p


def build_tree(root: Path) -> None:
    write(root / "CLAUDE.md", "istruzioni\n")
    write(root / ".gitignore", "/data/\n/hf/\n/outputs/\n/venvs/\n/cache/\n/wheels/\n/tools/\n/trasferimento/\n")
    write(root / "scripts" / "a.py", "print('ciao')\n" * 50)
    write(root / "docs" / "nota con spazi è.md", "accenti\n")
    write(root / "runs" / "ledger.tsv", "data\tjob\n")
    (root / "secrets").mkdir(mode=0o700)
    write(root / "data" / "download-log.tsv", "time\tstatus\n")
    write(root / "data" / "manifests" / "m1" / "manifest.json", json.dumps({"a": 1}) * 100)
    write(root / "data" / "shards" / "s" / "v1" / "x-00000.tar", os.urandom(120 * KIB))
    write(root / "data" / "shards" / "s" / "nota.txt", "file diretto in data/shards/s\n")
    (root / "data" / "vuota").mkdir(parents=True)
    blob = write(root / "hf" / "hub" / "models--o--m" / "blobs" / "abc", os.urandom(1_200_000))  # sopra 1 MiB
    snap = root / "hf" / "hub" / "models--o--m" / "snapshots" / "r1"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to("../../blobs/abc")
    write(root / "hf" / "hub" / "datasets--o--d" / "blobs" / "d1", b"piccolo\n")
    ck = root / "outputs" / "run" / "checkpoints" / "step-000001"
    write(ck / "optimizer.pt", os.urandom(200 * KIB))
    write(ck / "trainer_state.json", '{"step": 1}\n')
    write(ck / "decoder.safetensors", os.urandom(50 * KIB), mode=0o600)
    write(root / "outputs" / "logs" / "j.out", "log\n" * 1000)
    uvf = write(root / "cache" / "uv" / "archive-v0" / "k" / "pkg" / "x.py", "x = 1\n" * 300)
    lib = root / "venvs" / "build-1" / "lib" / "pkg"
    lib.mkdir(parents=True)
    os.link(uvf, lib / "x.py")
    (root / "venvs" / "current").symlink_to("build-1")
    (root / "venvs" / "build-1" / "bin").mkdir()
    (root / "venvs" / "build-1" / "bin" / "python").symlink_to("/usr/bin/python3")
    (root / "venvs" / "build-1" / "bin" / "rotto").symlink_to("non-esiste")
    write(root / "cache" / "vllm" / "c.json", "{}\n")
    write(root / "wheels" / "p-1.whl", os.urandom(10 * KIB))
    write(root / "tools" / "uv", os.urandom(5 * KIB), mode=0o755)
    write(root / "trasferimento" / "staging" / "non-va-nel-backup", "x")
    write(root / "third_party" / "lib" / "test_x.py", 'api_key = "segnaposto-di-prova"\n')
    write(root / "downloads" / "catalog_status.tsv", "# commento\ngruppo\trepo\trevisione_locale\tdisponibilita\t"
          "verifica_hash\tderivati_locali\truolo_pipeline\tammissione\tnota\np1\to/m\tr1\tscaricato\t1\t-\tmodello\t"
          "modello\t-\n")
    write(root / "data" / "snapshot-specs" / "models--o--m@r1.json", json.dumps({
        "cache_repo": str(root / "hf" / "hub" / "models--o--m"), "commit": "r1", "requested_revision": "main",
        "repo": "o/m", "type": "model", "files": {"model.safetensors": {"etag": "abc", "size": 1_200_000},
                                                   "README.md": {"etag": "f" * 40, "size": 10}}}))
    write(root / "data" / "shards" / "s" / "v1" / "manifest.json", json.dumps({"source": "o/d@r1", "job_id": "77"}))
    write(root / "runs" / "ledger.tsv", "data\tjob\n2026\t77\tlvl-shard\tsubmitted\tshard di prova\n")
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "CLAUDE.md", ".gitignore", "scripts", "docs", "runs", "third_party", "downloads"],
                   check=True)
    subprocess.run([*git, "commit", "-q", "-m", "c1"], check=True)
    write(root / "scripts" / "nuovo.py", "# non tracciato\n")
    os.utime(blob, ns=(1_700_000_000_000_000_000, 1_700_000_000_123_456_789))


def ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def inventory(root: Path, out: Path) -> dict:
    bk.cmd_inventario(ns(root=str(root), out=str(out), segreti=True))
    return json.loads((out / "riepilogo.json").read_text())


def plan(inv: Path, out: Path, gen: str, prev: str = "", **kw) -> dict:
    a = dict(inventario=str(inv), out=str(out), gen=gen, solo_categorie="", escludi_categorie="", precedente=prev,
             byte_volume=200 * KIB, dimensione_parte=256 * KIB, soglia_grandi=100 * KIB)
    a.update(kw)
    bk.cmd_piano(ns(**a))
    return json.loads((out / "piano.json").read_text())


def create(pdir: Path, staging: Path, state: Path, **kw) -> None:
    a = dict(piano=str(pdir), staging=str(staging), stato=str(state), volumi="", massimo=None,
             limite_staging_gib=1.0, attendi=False, solo_nucleo=False)
    a.update(kw)
    bk.cmd_crea(ns(**a))


def fails(fn) -> str:
    try:
        fn()
    except SystemExit as exc:
        return str(exc)
    return ""


def test_pure() -> None:
    cat = lambda p, t="f": bk.category(p, t, frozenset({"models--o--m"}), frozenset({"run"}))  # noqa: E731
    check("cat_codice", cat("CLAUDE.md") == "codice" and cat(".git/HEAD") == "codice" and cat("secrets", "d") == "codice"
          and cat("data/download-log.tsv") == "codice" and cat("third_party/x/y") == "codice")
    check("cat_dati", cat("data", "d") == "data-metadati" and cat("data/shards/s", "d") == "data-metadati"
          and cat("data/shards/s/v1", "d") == "data-shards-s-v1" and cat("data/shards/s/v1/x.tar") == "data-shards-s-v1"
          and cat("data/shards/s/nota.txt") == "data-metadati" and cat("data/manifests/m/x") == "data-metadati")
    check("cat_hf_outputs", cat("hf/hub/models--o--m", "d") == "hf-models--o--m"
          and cat("hf/hub/datasets--o--d/blobs/x") == "hf-piccoli" and cat("hf/hub/version.txt") == "hf-piccoli"
          and cat("outputs/run/a") == "outputs-run" and cat("outputs/logs/x") == "outputs-resto"
          and cat("outputs/x.txt") == "outputs-resto")
    check("cat_ambienti", cat("venvs/current", "l") == "ambienti" and cat("cache/uv/a") == "ambienti"
          and cat("cache/vllm/a") == "cache-altre" and cat("cache", "d") == "cache-altre"
          and cat("tools/uv") == "strumenti" and cat("wheels/a.whl") == "wheels" and cat("nuova/x") == "altro-nuova")
    check("nucleo", bk.is_nucleo("codice") and bk.is_nucleo("outputs-run") and bk.is_nucleo("hf-models--o--m")
          and bk.is_nucleo("data-shards-pixmo-cap-3e42775e") and not bk.is_nucleo("ambienti")
          and not bk.is_nucleo("hf-datasets--liuhaotian--LLaVA-Pretrain") and not bk.is_nucleo("data-shards-s-v1"))
    check("ordine", sorted(["ambienti", "outputs-run", "codice", "hf-models--o--m", "data-metadati"], key=bk.cat_rank)
          == ["codice", "data-metadati", "hf-models--o--m", "outputs-run", "ambienti"])
    check("sha256sum_escape", bk.sha256sum_line("ab", "a\\b") == "\\ab  a\\\\b" and bk.sha256sum_line("ab", "x y")
          == "ab  x y")
    with tempfile.TemporaryDirectory() as d:  # parti: confini esatti e nomi
        pw = bk.PartWriter(Path(d) / "v.tar", 10)
        pw.write(b"0123456789")
        pw.write(b"abcdefghij" * 2 + b"z")
        parts = pw.close()
        names = sorted(p.name for p in Path(d).iterdir())
        check("parti_confini", [p["byte"] for p in parts] == [10, 10, 10, 1]
              and names == [f"v.tar.part00{k}" for k in range(4)], (parts, names))
        cr = bk.ConcatReader([Path(d) / n for n in names])
        data = cr.read()
        check("parti_rilettura", data == b"0123456789abcdefghijabcdefghijz" and cr.sha.hexdigest() == pw.sha.hexdigest())
        pw1 = bk.PartWriter(Path(d) / "w.tar", 10)
        pw1.write(b"0123456789")
        check("parte_unica", [p["nome"] for p in pw1.close()] == ["w.tar"])


def test_flow(tmp: Path) -> None:
    root = tmp / "progetto"
    build_tree(root)
    s = inventory(root, tmp / "inv1")
    recs = list(bk.read_jsonl_gz(tmp / "inv1" / "inventario.jsonl.gz"))
    paths = {r["p"] for r in recs}
    check("inv_esclude_trasferimento", not any(p.startswith("trasferimento") for p in paths))
    check("inv_vuote_nascosti", {"data/vuota", ".gitignore", ".git/HEAD", "secrets"} <= paths)
    check("inv_hardlink", s["hardlink"]["gruppi"] == 1 and s["hardlink"]["gruppi_con_nomi_fuori_dalla_visita"] == 0,
          s["hardlink"])
    links = {x["p"]: x for x in bk.read_jsonl_gz(tmp / "inv1" / "collegamenti.jsonl.gz")}
    check("inv_link", links["venvs/build-1/bin/python"]["assoluto"] and not links["venvs/build-1/bin/python"][
        "reale_dentro"] and not links["venvs/build-1/bin/rotto"]["esiste"] and links["venvs/current"]["dentro"]
          and links["hf/hub/models--o--m/snapshots/r1/model.safetensors"]["esiste"], links)
    sec = json.loads((tmp / "inv1" / "segreti.json").read_text())
    check("inv_segreti_senza_testo", [h["p"] for h in sec["file"]] == ["third_party/lib/test_x.py"]
          and "segnaposto" not in json.dumps(sec), sec["file"])
    g = json.loads((tmp / "inv1" / "git.json").read_text())
    check("inv_git", g["stato_file"].get("non_tracciati") == 1 and g["radice"]["head"] == g["head_inizio"], g["stato_file"])

    pdir = tmp / "piani" / "g1"
    p = plan(tmp / "inv1", pdir, "g1")
    cats = p["categorie"]
    check("piano_categorie", {"codice", "data-metadati", "data-shards-s-v1", "hf-models--o--m", "hf-piccoli",
                              "outputs-run", "outputs-resto", "ambienti", "cache-altre", "wheels", "strumenti"}
          == set(cats), sorted(cats))
    check("piano_ordine", [v["categoria"] for v in p["volumi"]][0] == "codice"
          and p["volumi"][-1]["categoria"] in ("ambienti", "cache-altre"), [v["id"] for v in p["volumi"]])
    big = [v for v in p["volumi"] if v["grande"]]
    check("piano_grandi", {v["id"] for v in big} == {"hf-models--o--m-002", "outputs-run-002"}, big)
    check("piano_compressione", cats["codice"]["compressione"] == "gzip"
          and cats["hf-models--o--m"]["compressione"] == "nessuna", cats["codice"])
    members = list(bk.read_jsonl_gz(pdir / "membri.jsonl.gz"))
    check("piano_copertura", sorted(m["p"] for m in members) == sorted(paths))
    amb = [m for m in members if m["c"] == "ambienti"]
    check("piano_hardlink_insieme", len({m["v"] for m in amb if m["p"].endswith("pkg/x.py")}) == 1)
    ck = (pdir / "checkpoint.tsv").read_text()
    check("piano_checkpoint", "outputs/run/checkpoints/step-000001\tsì\tsì\tsì" in ck, ck)

    staging, state = tmp / "staging", tmp / "stato"
    write(root / "scripts" / "a.py", "print('cambiato')\n")  # dopo l'inventario: il volume lo segnala
    create(pdir, staging, state, massimo=2)
    ready = sorted(x.name for x in (state / "g1" / "vol").glob("*.pronto.json"))
    check("crea_massimo", len(ready) == 2, ready)
    create(pdir, staging, state)
    ready = sorted(x.name for x in (state / "g1" / "vol").glob("*.pronto.json"))
    check("crea_tutti", len(ready) == len(p["volumi"]), (len(ready), len(p["volumi"])))
    info = {v["id"]: json.loads((state / "g1" / "vol" / f"{v['id']}.pronto.json").read_text()) for v in p["volumi"]}
    warns = [w for v in info.values() for w in v["avvisi"]]
    check("crea_avviso_cambiato", any(w["p"] == "scripts/a.py" and "cambiato" in w["avviso"] for w in warns), warns)
    hfv = info["hf-models--o--m-002"]
    check("crea_parti", len(hfv["parti"]) == 5 and hfv["parti"][0]["nome"].endswith(".tar.part000")
          and all((staging / "g1" / "hf-models--o--m" / x["nome"]).is_file() for x in hfv["parti"]), hfv["parti"])
    rows = (state / "g1" / "vol" / "hf-models--o--m-002.parti.tsv").read_text().splitlines()
    check("crea_parti_tsv", rows[0].startswith("tipo\tfile") and len(rows) == 1 + 5 + 2
          and rows[1].split("\t")[6] == "volumi/hf-models--o--m/hf-models--o--m-002.tar.part000", rows[:3])

    # ripristino di ogni volume con GNU tar
    ok = True
    for v in p["volumi"]:
        vid, c = v["id"], v["categoria"]
        rep = tmp / f"rip-{vid}.json"
        try:
            bk.cmd_ripristino(ns(pronto=str(state / "g1" / "vol" / f"{vid}.pronto.json"),
                                 cartella=str(staging / "g1" / c), indice=str(state / "g1" / "indici" / f"{vid}.jsonl.gz"),
                                 sha256sum=str(state / "g1" / "indici" / f"{vid}.sha256"), dest=str(tmp / f"d-{vid}"),
                                 report=str(rep), rimuovi=True, esistente=False))
        except SystemExit as exc:
            ok = False
            print(vid, exc, rep.read_text() if rep.exists() else "")
    check("ripristino_tutti", ok)

    # ripristino vero: tutti i volumi, nell'ordine del piano, nella stessa radice nuova; poi confronto con l'originale
    rebuilt = tmp / "ricostruito"
    for v in p["volumi"]:
        vid = v["id"]
        bk.cmd_ripristino(ns(pronto=str(state / "g1" / "vol" / f"{vid}.pronto.json"),
                             cartella=str(staging / "g1" / v["categoria"]),
                             indice=str(state / "g1" / "indici" / f"{vid}.jsonl.gz"),
                             sha256sum=str(state / "g1" / "indici" / f"{vid}.sha256"), dest=str(rebuilt),
                             report=str(tmp / f"vero-{vid}.json"), rimuovi=False, esistente=True))
    diff = []
    for r in recs:
        a, b = root / r["p"], rebuilt / r["p"]
        if not os.path.lexists(b):
            diff.append(f"manca {r['p']}")
        elif r["t"] == "l" and os.readlink(a) != os.readlink(b):
            diff.append(f"link {r['p']}")
        elif r["t"] == "f" and bk.file_sha256(a) != bk.file_sha256(b):
            diff.append(f"contenuto {r['p']}")
        elif r["t"] in ("f", "d") and (os.lstat(a).st_mode & 0o7777) != (os.lstat(b).st_mode & 0o7777):
            diff.append(f"permessi {r['p']}")
    hl = [rebuilt / "cache/uv/archive-v0/k/pkg/x.py", rebuilt / "venvs/build-1/lib/pkg/x.py"]
    check("ripristino_completo", not diff and os.lstat(hl[0]).st_ino == os.lstat(hl[1]).st_ino
          and not (rebuilt / "trasferimento").exists(), diff[:5])
    check("rifiuta_rimozione_radice", "non si rimuove" in fails(lambda: bk.cmd_ripristino(ns(
        pronto=str(state / "g1" / "vol" / "codice-001.pronto.json"), cartella=str(staging / "g1" / "codice"),
        indice="-", sha256sum="-", dest=str(rebuilt), report=str(tmp / "x.json"), rimuovi=True, esistente=True))))
    amb_vol = next(v["id"] for v in p["volumi"] if v["categoria"] == "ambienti")
    rep = json.loads((tmp / f"rip-{amb_vol}.json").read_text())
    check("ripristino_hardlink_link", rep["voci_controllate"].get("h") == 1 and rep["voci_controllate"].get("l") == 3,
          rep)
    idx = list(bk.read_jsonl_gz(state / "g1" / "indici" / "outputs-run-001.jsonl.gz"))
    dec = next(e for e in idx if e["p"].endswith("decoder.safetensors"))
    check("indice_permessi", dec["m"] == 0o600, dec)

    part = staging / "g1" / "hf-models--o--m" / hfv["parti"][2]["nome"]
    data = bytearray(part.read_bytes())
    data[10] ^= 0xFF
    part.write_bytes(bytes(data))
    err = fails(lambda: bk.cmd_ripristino(ns(
        pronto=str(state / "g1" / "vol" / "hf-models--o--m-002.pronto.json"), cartella=str(part.parent),
        indice=str(state / "g1" / "indici" / "hf-models--o--m-002.jsonl.gz"),
        sha256sum=str(state / "g1" / "indici" / "hf-models--o--m-002.sha256"), dest=str(tmp / "d-alterata"),
        report=str(tmp / "rip-alterata.json"), rimuovi=True, esistente=False)))
    rep = json.loads((tmp / "rip-alterata.json").read_text())
    check("ripristino_parte_alterata", err != "" and any("part002" in x for x in rep["problemi"]), rep["problemi"][:3])
    data[10] ^= 0xFF  # parte di nuovo integra per le prove successive
    part.write_bytes(bytes(data))

    # staging limitato: con lo staging occupato e senza --attendi non si crea nulla di nuovo
    pdir2 = tmp / "piani" / "g1b"
    plan(tmp / "inv1", pdir2, "g1b")
    write(staging / "g1b" / "occupato", os.urandom(300 * KIB))
    create(pdir2, staging, state, limite_staging_gib=200 * KIB / bk.GIB)
    check("staging_limitato", not list((state / "g1b" / "vol").glob("*.pronto.json")))

    # catalogo di recupero
    cat_out = tmp / "catalogo"
    bk.cmd_catalogo(ns(root=str(root), inventario=str(tmp / "inv1"), stato=str(state / "g1"), out=str(cat_out)))
    cat_rows = bk.read_tsv(cat_out / "catalogo_recupero.tsv")
    mod = next(r for r in cat_rows if r["file"] == "model.safetensors")
    rdm = next(r for r in cat_rows if r["file"] == "README.md")
    check("catalogo_presenza", mod["presente"] == "sì" and len(mod["sha256_locale"]) == 64
          and mod["volume"] == "hf-models--o--m-002" and rdm["presente"] == "no" and rdm["tipo_hash"] == "gitsha1"
          and mod["ruolo"] == "modello", (mod, rdm))
    ref = bk.read_tsv(cat_out / "hash_riferimento_dati.tsv")
    check("catalogo_riferimento", {r["percorso"] for r in ref} >= {"hf/hub/models--o--m/blobs/abc",
                                                                   "data/shards/s/v1/x-00000.tar"}, ref)
    der = bk.read_tsv(cat_out / "derivati.tsv")
    check("catalogo_derivati", der and der[0]["sorgente"] == "o/d@r1" and der[0]["launcher"] == "lvl-shard", der)
    abs_rows = bk.read_tsv(cat_out / "percorsi_assoluti.tsv")
    check("catalogo_percorsi", any(r["file"] == "data/snapshot-specs/models--o--m@r1.json" for r in abs_rows), abs_rows)
    ext = bk.read_tsv(cat_out / "collegamenti_assoluti_o_esterni.tsv")
    check("catalogo_link", any(r["link"] == "venvs/build-1/bin/python" and r["risolto"] == "fuori dalla radice"
                               for r in ext), ext)

    # completezza: con ogni parte e indice verificati (come li scrive backup_upload.sh) passa; senza una, no
    def fake_drive(skip: str = "") -> None:
        idx = ["remoto\tbyte\tmd5_drive\tsha256_drive\tid_drive\tverificato"]
        for v in p["volumi"]:
            rows_v = ["remoto\tbyte\tmd5_drive\tsha256_drive\tid_drive\tverificato"]
            for r in bk.read_tsv(state / "g1" / "vol" / f"{v['id']}.parti.tsv"):
                line = f"LimiteVL-backup/g1/{r['remoto']}\t{r['byte']}\t{r['md5']}\t{r['sha256']}\tid\tora"
                (rows_v if r["tipo"] == "volume" else idx).append(line)
            f = state / "g1" / "vol" / f"{v['id']}.verificato.tsv"
            if v["id"] == skip:
                f.unlink(missing_ok=True)
            else:
                f.write_text("\n".join(rows_v) + "\n")
        (state / "g1" / "indici-verificati.tsv").write_text("\n".join(idx) + "\n")

    fake_drive()
    comp = dict(stato=str(state / "g1"), piani=[str(pdir)], inventari=[str(tmp / "inv1")],
                categorie_inventari=["tranne:nessuna"], out=str(tmp / "comp1"), solo_nucleo=False)
    err = fails(lambda: bk.cmd_completezza(ns(**comp)))
    rep = json.loads((tmp / "comp1" / "completezza.json").read_text())
    check("completezza_ok", err == "" and rep["n_problemi"] == 0 and rep["totali"]["volumi"] == len(p["volumi"]),
          rep["problemi"][:3])
    fake_drive(skip="wheels-001")
    err = fails(lambda: bk.cmd_completezza(ns(**{**comp, "out": str(tmp / "comp2")})))
    check("completezza_manca_volume", err != "" and any("wheels-001" in x for x in json.loads(
        (tmp / "comp2" / "completezza.json").read_text())["problemi"]))
    fake_drive()

    # backup selezionato: si creano e si verificano solo i volumi del nucleo
    pdir_s = tmp / "piani" / "gs"
    ps = plan(tmp / "inv1", pdir_s, "gs")
    create(pdir_s, staging, state, solo_nucleo=True)
    made = {x.name[: -len(".pronto.json")] for x in (state / "gs" / "vol").glob("*.pronto.json")}
    core = {v["id"] for v in ps["volumi"] if v["nucleo"]}
    check("selezionato_crea", made == core and len(core) < len(ps["volumi"]), (sorted(made), sorted(core)))
    idx = ["remoto\tbyte\tmd5_drive\tsha256_drive\tid_drive\tverificato"]
    for vid in made:
        rows_v = ["remoto\tbyte\tmd5_drive\tsha256_drive\tid_drive\tverificato"]
        for r in bk.read_tsv(state / "gs" / "vol" / f"{vid}.parti.tsv"):
            line = f"LimiteVL-backup/gs/{r['remoto']}\t{r['byte']}\t{r['md5']}\t{r['sha256']}\tid\tora"
            (rows_v if r["tipo"] == "volume" else idx).append(line)
        (state / "gs" / "vol" / f"{vid}.verificato.tsv").write_text("\n".join(rows_v) + "\n")
    (state / "gs" / "indici-verificati.tsv").write_text("\n".join(idx) + "\n")
    comp_s = dict(stato=str(state / "gs"), piani=[str(pdir_s)], inventari=[str(tmp / "inv1")],
                  categorie_inventari=["tranne:nessuna"], out=str(tmp / "comp-s1"), solo_nucleo=True)
    err = fails(lambda: bk.cmd_completezza(ns(**comp_s)))
    rep = json.loads((tmp / "comp-s1" / "completezza.json").read_text())
    check("selezionato_completezza", err == "" and rep["totali"]["volumi"] == len(core), rep["problemi"][:3])
    err = fails(lambda: bk.cmd_completezza(ns(**{**comp_s, "solo_nucleo": False, "out": str(tmp / "comp-s2")})))
    check("selezionato_completezza_tutto_manca", err != "")

    # script portabili (Mac o Linux): ripristino con sha256sum e con shasum, confronto per contenuto
    sh_ok = True
    for forza in ("", "1"):
        if forza and subprocess.run(["sh", "-c", "command -v shasum"], capture_output=True).returncode:
            print("shasum assente: ramo macOS non provato")
            continue
        dest = tmp / f"rip-sh{forza}"
        env = {**os.environ, "FORZA_SHASUM": forza}
        r = subprocess.run(["sh", str(ROOT / "scripts/ripristina_volumi.sh"), str(tmp / "comp1/volumi.tsv"),
                            str(tmp / "comp1/parti.tsv"), str(staging / "g1"), str(state / "g1" / "indici"), str(dest),
                            "tutti"], capture_output=True, text=True, env=env)
        done = (tmp / f"rip-sh{forza}.ripristino-fatti").read_text().split() if r.returncode == 0 else []
        same = all(bk.file_sha256(root / x["p"]) == bk.file_sha256(dest / x["p"]) for x in recs if x["t"] == "f")
        if r.returncode or len(done) != len(p["volumi"]) or not same:
            sh_ok = False
            print(r.stdout[-500:], r.stderr[-500:])
    check("script_ripristino", sh_ok)
    nuc = subprocess.run(["sh", str(ROOT / "scripts/ripristina_volumi.sh"), str(tmp / "comp1/volumi.tsv"),
                          str(tmp / "comp1/parti.tsv"), str(staging / "g1"), str(state / "g1" / "indici"),
                          str(tmp / "rip-nucleo"), "nucleo"], capture_output=True, text=True)
    done = (tmp / "rip-nucleo.ripristino-fatti").read_text().split()
    check("script_ripristino_nucleo", nuc.returncode == 0 and done and all(bk.is_nucleo(d.rsplit("-", 1)[0])
                                                                         for d in done), (done, nuc.stderr[-300:]))
    hfp = staging / "g1" / "hf-models--o--m" / hfv["parti"][1]["nome"]
    moved = hfp.with_name(hfp.name + ".via")
    hfp.rename(moved)
    bad = subprocess.run(["sh", str(ROOT / "scripts/ripristina_volumi.sh"), str(tmp / "comp1/volumi.tsv"),
                          str(tmp / "comp1/parti.tsv"), str(staging / "g1"), str(state / "g1" / "indici"),
                          str(tmp / "rip-manca"), "tutti"], capture_output=True, text=True)
    moved.rename(hfp)
    check("script_ripristino_parte_mancante", bad.returncode != 0 and "manca la parte" in bad.stderr, bad.stderr[-300:])
    mac = tmp / "mac"
    write(mac / "copia con nome diverso.tar", (root / "data/shards/s/v1/x-00000.tar").read_bytes())
    write(mac / "altro.bin", b"niente in comune")
    cwd = tmp / "confronto"
    cwd.mkdir()
    for forza in ("", "1"):
        r = subprocess.run(["sh", str(ROOT / "scripts/mac_confronta_hash.sh"), str(mac),
                            str(cat_out / "hash_riferimento_dati.tsv"), str(cat_out / "catalogo_recupero.tsv")],
                           cwd=cwd, capture_output=True, text=True, env={**os.environ, "FORZA_SHASUM": forza})
        same = (cwd / "confronto-uguali.tsv").read_text().splitlines()
        only = (cwd / "confronto-solo-mac.tsv").read_text().splitlines()
        check(f"script_confronto{forza}", r.returncode == 0 and len(same) == 1
              and same[0].endswith("data/shards/s/v1/x-00000.tar") and len(only) == 1, (r.stderr[-300:], same, only))

    # aggiornamento: solo le categorie cambiate hanno volumi nuovi
    write(root / "outputs" / "logs" / "j2.out", "nuovo\n")
    inventory(root, tmp / "inv2")
    err = fails(lambda: bk.cmd_completezza(ns(**{**comp, "inventari": [str(tmp / "inv2")], "out": str(tmp / "comp3")})))
    rep = json.loads((tmp / "comp3" / "completezza.json").read_text())
    check("completezza_inventario_scoperto", err != "" and any("outputs/logs/j2.out" in x for x in rep["problemi"]),
          rep["problemi"][:3])
    p2 = plan(tmp / "inv2", tmp / "piani" / "g2", "g2", prev=str(pdir / "piano.json"))
    new_cats = {v["categoria"] for v in p2["volumi"]}
    check("aggiorna_solo_cambiate", "outputs-resto" in new_cats and "hf-models--o--m" not in new_cats
          and p2["categorie"]["hf-models--o--m"].get("riusata_da") == "g1", new_cats)


def test_refusals(tmp: Path) -> None:
    root = tmp / "progetto2"
    build_tree(root)
    write(root / "secrets" / "hf_token", "x")
    inventory(root, tmp / "inv-s")
    check("rifiuta_secrets_non_vuota", "secrets/" in fails(lambda: plan(tmp / "inv-s", tmp / "p-s", "x")))
    (root / "secrets" / "hf_token").unlink()
    write(root / "outputs" / "logs" / "t.log", "token hf_" + "A1" * 20 + "\n")
    inventory(root, tmp / "inv-t")
    sec = json.loads((tmp / "inv-t" / "segreti.json").read_text())
    check("segreto_senza_testo", "A1A1" not in json.dumps(sec))
    check("rifiuta_segreto_forte", "alta affidabilità" in fails(lambda: plan(tmp / "inv-t", tmp / "p-t", "x")))


def main() -> None:
    test_pure()
    with tempfile.TemporaryDirectory() as d:
        test_flow(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_refusals(Path(d))
    print(f"{len(DONE) - len(FAILS)} controlli su {len(DONE)}")
    if FAILS:
        print("falliti: " + "; ".join(FAILS))
        sys.exit(1)


if __name__ == "__main__":
    main()
