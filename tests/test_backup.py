"""Backup (scripts/backup.py): inventario, piano, creazione, verifica e prova di ripristino su un albero costruito qui.
Nessun dato reale. Solo in un job SLURM (slurm/tests_cpu.sbatch tests/test_backup.py): usa git e GNU tar.

Controlla:
- inventario: file nascosti e ignorati, cartelle vuote, `trasferimento/` esclusa, link relativi, assoluti, verso
  l'esterno e interrotti, hard link fra cache/uv e venvs, nomi con spazi e accenti, segreti senza testo;
- piano: categorie (codice, metadati, shard, HF e outputs grandi, ambienti uniti dagli hard link), nucleo, ordine,
  volumi entro il limite, file grande in un volume a sé, arresti per secrets/ non vuota e segreti forti;
- creazione: parti con nomi e hash, rilettura completa, elenco di GNU tar, ripresa senza rifare i volumi pronti,
  staging limitato; avviso motivato per un file volatile cambiato (ledger), blocco per un file eliminato o non
  volatile cambiato dopo l'inventario, blocco per un marcatore pronto di un altro piano con gli stessi nomi;
- completezza: esito e codice d'uscita, copertura dalle voci archiviate, voce saltata di un volume dello schema 1,
  inventario con errori, marcatore di un altro piano, volume non verificato, backup selezionato;
- piano: rifiuto di un inventario con errori di visita e del percorso incrementale (--precedente); sottoinsieme
  dichiarato con le esclusioni, prefisso e dipendenza di un supplemento;
- ripristino: cat | tar -xp ricostruisce file, permessi, date, link e hard link; sha256sum -c; una parte alterata
  fa fallire la prova; base più supplemento con gli stessi nomi di volume, percorsi relativi (anche "indici"),
  elenco dei file rimossi, ripresa senza rifare, --elimina-parti solo dopo il controllo; composizione dello stato
  finale (file vecchio della base rimasto, stato di Git con un file non volatile cambiato, file atteso per sha256);
- confronto Mac: catalogo mancante o vuoto e file illeggibili danno un confronto non valido;
- caricamento con rclone simulato: stop fra due parti e durante l'attesa, ripresa senza ricaricare le parti già su
  Drive, marcatori dello stesso piano;
- firme: confronto in sola lettura fra un inventario nuovo e il piano di base.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
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
                                                   "README.md": {"etag": "f" * 40, "size": 10},
                                                   "extra.bin": {"etag": "0123456789abcdef" * 4, "size": 10}}}))
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
             byte_volume=200 * KIB, dimensione_parte=256 * KIB, soglia_grandi=100 * KIB, sottoinsieme=[],
             prefisso_volumi="", dipende_da="")
    a.update(kw)
    bk.cmd_piano(ns(**a))
    return json.loads((out / "piano.json").read_text())


def completeness(state: Path, plans: list[Path], out: Path, solo_nucleo: bool = False) -> tuple[str, dict]:
    err = fails(lambda: bk.cmd_completezza(ns(stato=str(state), piani=[str(p) for p in plans], out=str(out),
                                              solo_nucleo=solo_nucleo)))
    return err, json.loads((out / "completezza.json").read_text())


def fake_verify(state: Path, gen: str, vids) -> None:
    """Scrive le verifiche come le scrive backup_upload.sh dopo un caricamento riuscito (Drive simulato)."""
    idx = ["remoto\tbyte\tmd5_drive\tsha256_drive\tid_drive\tverificato"]
    old = state / "indici-verificati.tsv"
    if old.is_file():
        idx = old.read_text().splitlines()
    for vid in vids:
        rows = ["remoto\tbyte\tmd5_drive\tsha256_drive\tid_drive\tverificato"]
        for r in bk.read_tsv(state / "vol" / f"{vid}.parti.tsv"):
            line = f"LimiteVL-backup/{gen}/{r['remoto']}\t{r['byte']}\t{r['md5']}\t{r['sha256']}\tid\tora"
            (rows if r["tipo"] == "volume" else idx).append(line)
        (state / "vol" / f"{vid}.verificato.tsv").write_text("\n".join(rows) + "\n")
    (state / "indici-verificati.tsv").write_text("\n".join(idx) + "\n")


def run_script(name: str, *args, cwd=None, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(ROOT / "scripts" / name), *map(str, args)], capture_output=True, text=True,
                          cwd=cwd, env={**os.environ, **(env or {})})


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
    reports, bad = sorted((ROOT / "docs" / "backup").rglob("*.json")), []  # rapporti del backup, anche quelli scritti a mano
    for f in reports:
        try:
            json.loads(f.read_text(encoding="utf-8"))
        except ValueError as exc:
            bad.append(f"{f.relative_to(ROOT)}: {exc}")
    check("rapporti_json_leggibili", reports and not bad, bad)


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
    with open(root / "runs" / "ledger.tsv", "a") as fh:  # file volatile cambiato dopo l'inventario: avviso motivato
        fh.write("2026\t78\tlvl-x\tnote\triga aggiunta dopo l'inventario\n")
    create(pdir, staging, state, massimo=2)
    ready = sorted(x.name for x in (state / "g1" / "vol").glob("*.pronto.json"))
    check("crea_massimo", len(ready) == 2, ready)
    create(pdir, staging, state)
    ready = sorted(x.name for x in (state / "g1" / "vol").glob("*.pronto.json"))
    check("crea_tutti", len(ready) == len(p["volumi"]), (len(ready), len(p["volumi"])))
    info = {v["id"]: json.loads((state / "g1" / "vol" / f"{v['id']}.pronto.json").read_text()) for v in p["volumi"]}
    warns = [w for v in info.values() for w in v["avvisi"]]
    check("crea_avviso_volatile", any(w["p"] == "runs/ledger.tsv" and "cambiato" in w["avviso"] and not w["bloccante"]
                                      for w in warns) and not any(w["bloccante"] for w in warns), warns)
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
    all_ids = [v["id"] for v in p["volumi"]]
    fake_verify(state / "g1", "g1", all_ids)
    err, rep = completeness(state / "g1", [pdir], tmp / "comp1")
    check("completezza_ok", err == "" and rep["esito"] == "completo" and rep["totali"]["volumi_verificati"] == len(all_ids)
          and rep["totali"]["voci_mancanti"] == 0 and rep["n_avvisi_motivati"] >= 1, rep["problemi"][:3])
    (state / "g1" / "vol" / "wheels-001.verificato.tsv").rename(state / "g1" / "vol" / "wheels-001.verificato.via")
    err, rep = completeness(state / "g1", [pdir], tmp / "comp2")
    vt = {r["volume"]: r["stato"] for r in bk.read_tsv(tmp / "comp2" / "volumi.tsv")}
    check("completezza_manca_volume", err != "" and rep["esito"] == "incompleto"
          and vt["wheels-001"].startswith("non verificato") and rep["totali"]["voci_mancanti"] > 0, vt["wheels-001"])
    (state / "g1" / "vol" / "wheels-001.verificato.via").rename(state / "g1" / "vol" / "wheels-001.verificato.tsv")

    # volume dello schema 1 con un membro saltato e dichiarato solo come avviso: prima contava come "mancante" senza
    # errore; ora blocca (avviso, indice diverso dai membri, voce attesa non archiviata)
    forged = tmp / "stato-schema1"
    shutil.copytree(state / "g1", forged)
    vid = next(v["id"] for v in p["volumi"] if v["categoria"] == "outputs-run")
    entries = list(bk.read_jsonl_gz(forged / "indici" / f"{vid}.jsonl.gz"))
    drop = next(e for e in entries if e["t"] == "f")
    with gzip.open(forged / "indici" / f"{vid}.jsonl.gz", "wt", encoding="ascii") as fh:
        for e in entries:
            if e is not drop:
                fh.write(bk.dump(e) + "\n")
    pr = json.loads((forged / "vol" / f"{vid}.pronto.json").read_text())
    pr["schema"] = "limite-vl-backup-volume/1"
    pr["avvisi"] = [{"p": drop["p"], "avviso": "mancante al momento dell'archiviazione"}]
    for x in pr["indice"]:
        x["sha256"] = bk.file_sha256(forged / "indici" / x["nome"])
    (forged / "vol" / f"{vid}.pronto.json").write_text(json.dumps(pr))
    err, rep = completeness(forged, [pdir], tmp / "comp-schema1")
    check("completezza_saltato_bloccante", err != "" and any("avviso bloccante" in x for x in rep["problemi"])
          and drop["p"] in rep["voci_mancanti_elenco"], rep["problemi"][:4])

    # inventario con errori di visita: la completezza non lo accetta
    rp = tmp / "inv1" / "riepilogo.json"
    saved = rp.read_text()
    rp.write_text(json.dumps({**json.loads(saved), "n_errori": 1, "errori": [{"p": "x", "errore": "scandir"}]}))
    err, rep = completeness(state / "g1", [pdir], tmp / "comp-inv-err")
    rp.write_text(saved)
    check("completezza_inventario_con_errori", err != "" and any("errori di visita" in x for x in rep["problemi"]),
          rep["problemi"][:3])

    # backup selezionato: si creano e si verificano solo i volumi del nucleo
    pdir_s = tmp / "piani" / "gs"
    ps = plan(tmp / "inv1", pdir_s, "gs")
    create(pdir_s, staging, state, solo_nucleo=True)
    made = {x.name[: -len(".pronto.json")] for x in (state / "gs" / "vol").glob("*.pronto.json")}
    core = {v["id"] for v in ps["volumi"] if v["nucleo"]}
    check("selezionato_crea", made == core and len(core) < len(ps["volumi"]), (sorted(made), sorted(core)))
    fake_verify(state / "gs", "gs", made)
    err, rep = completeness(state / "gs", [pdir_s], tmp / "comp-s1", solo_nucleo=True)
    check("selezionato_completezza", err == "" and rep["totali"]["volumi_verificati"] == len(core), rep["problemi"][:3])
    err, rep = completeness(state / "gs", [pdir_s], tmp / "comp-s2")
    check("selezionato_completezza_tutto_manca", err != "" and rep["totali"]["voci_mancanti"] > 0)

    # script portabili (Mac o Linux): ripristino con sha256sum e con shasum, confronto per contenuto
    sh_ok = True
    for forza in ("", "1"):
        if forza and subprocess.run(["bash", "-c", "command -v shasum"], capture_output=True).returncode:
            print("shasum assente: ramo macOS non provato")
            continue
        dest = tmp / f"rip-sh{forza}"
        r = run_script("ripristina_volumi.sh", tmp / "comp1/volumi.tsv", tmp / "comp1/parti.tsv", staging / "g1",
                       state / "g1" / "indici", dest, "tutti", env={"FORZA_SHASUM": forza})
        done = (tmp / f"rip-sh{forza}.ripristino-fatti").read_text().splitlines() if r.returncode == 0 else []
        same = all(bk.file_sha256(root / x["p"]) == bk.file_sha256(dest / x["p"]) for x in recs if x["t"] == "f")
        if r.returncode or len(done) != len(p["volumi"]) or not same:
            sh_ok = False
            print(r.stdout[-500:], r.stderr[-500:])
    check("script_ripristino", sh_ok)
    nuc = run_script("ripristina_volumi.sh", tmp / "comp1/volumi.tsv", tmp / "comp1/parti.tsv", staging / "g1",
                     state / "g1" / "indici", tmp / "rip-nucleo", "nucleo")
    done = [x.split("\t")[0] for x in (tmp / "rip-nucleo.ripristino-fatti").read_text().splitlines()]
    check("script_ripristino_nucleo", nuc.returncode == 0 and done and all(bk.is_nucleo(d.rsplit("-", 1)[0])
                                                                         for d in done), (done, nuc.stderr[-300:]))
    hfp = staging / "g1" / "hf-models--o--m" / hfv["parti"][1]["nome"]
    moved = hfp.with_name(hfp.name + ".via")
    hfp.rename(moved)
    bad = run_script("ripristina_volumi.sh", tmp / "comp1/volumi.tsv", tmp / "comp1/parti.tsv", staging / "g1",
                     state / "g1" / "indici", tmp / "rip-manca", "tutti")
    moved.rename(hfp)
    check("script_ripristino_parte_mancante", bad.returncode != 0 and "manca la parte" in bad.stderr, bad.stderr[-300:])
    mac = tmp / "mac"
    write(mac / "copia con nome diverso.tar", (root / "data/shards/s/v1/x-00000.tar").read_bytes())
    write(mac / "altro.bin", b"niente in comune")
    cwd = tmp / "confronto"
    cwd.mkdir()
    for forza in ("", "1"):
        r = run_script("mac_confronta_hash.sh", mac, cat_out / "hash_riferimento_dati.tsv",
                       cat_out / "catalogo_recupero.tsv", cwd=cwd, env={"FORZA_SHASUM": forza})
        same = (cwd / "confronto-uguali.tsv").read_text().splitlines()
        only = (cwd / "confronto-solo-mac.tsv").read_text().splitlines()
        check(f"script_confronto{forza}", r.returncode == 0 and len(same) == 1
              and same[0].endswith("data/shards/s/v1/x-00000.tar") and len(only) == 1, (r.stderr[-300:], same, only))

    # aggiornamento incrementale: rifiutato con una diagnosi
    err = fails(lambda: plan(tmp / "inv1", tmp / "piani" / "g2", "g2", prev=str(pdir / "piano.json")))
    check("rifiuta_incrementale", "non supportato" in err and not (tmp / "piani" / "g2").exists(), err)


FAKE_RCLONE = r'''#!/usr/bin/env bash
# rclone simulato per i test di backup_upload.sh: "lvl-ist:PERCORSO" -> $FAKE_REMOTE/PERCORSO; hash veri.
set -euo pipefail
label=$1; shift; [ "$1" = "--" ] && shift
cmd=$1; shift
map() { local x=$1; x=${x#lvl-ist:}; printf '%s/%s' "$FAKE_REMOTE" "$x"; }
case $cmd in
  copyto)
    dst=$(map "$2"); mkdir -p "$(dirname "$dst")"; cp "$1" "$dst"
    echo "$label $2" >> "$FAKE_REMOTE/.registro"
    if [ -n "${FAKE_STOP_DOPO:-}" ] && [ "$(grep -c . "$FAKE_REMOTE/.registro")" -ge "$FAKE_STOP_DOPO" ]; then
      touch "$FAKE_STOP_FILE"
    fi ;;
  copy) dst=$(map "$2"); mkdir -p "$dst"; cp -r "$1"/. "$dst"/ ;;
  lsjson)
    d=""; for a in "$@"; do case $a in lvl-ist:*) d=$(map "$a") ;; esac; done
    [ -d "$d" ] || { echo "ERROR : directory not found" >&2; exit 3; }
    echo "["
    for f in "$d"/*; do
      [ -f "$f" ] || continue
      n=$(basename "$f")
      printf '{"Path":"%s","Name":"%s","Size":%s,"Hashes":{"md5":"%s","sha256":"%s"},"ID":"id-%s"},\n' "$n" "$n" \
        "$(stat -c %s "$f")" "$(md5sum "$f" | cut -d' ' -f1)" "$(sha256sum "$f" | cut -d' ' -f1)" "$n"
    done
    echo "]" ;;
  *) echo "comando non simulato: $cmd" >&2; exit 2 ;;
esac
'''


def test_correzioni(tmp: Path) -> None:
    root = tmp / "progetto3"
    build_tree(root)
    staging, state = tmp / "staging", tmp / "stato"
    inventory(root, tmp / "inv-a")

    # file eliminato dopo l'inventario: quel volume non è pronto e la completezza fallisce con codice d'uscita
    pa = tmp / "piani" / "ga"
    plan(tmp / "inv-a", pa, "ga", solo_categorie="codice")
    gone = "docs/nota con spazi è.md"
    vbad = {m["p"]: m["v"] for m in bk.read_jsonl_gz(pa / "membri.jsonl.gz")}[gone]
    (root / gone).unlink()
    err = fails(lambda: create(pa, staging, state))
    check("crea_file_eliminato", "mancante al momento dell'archiviazione" in err
          and not (state / "ga" / "vol" / f"{vbad}.pronto.json").exists()
          and (state / "ga" / "vol" / f"{vbad}.errore.json").exists(), err)
    err, rep = completeness(state / "ga", [pa], tmp / "comp-ga")
    check("completezza_file_eliminato", err != "" and rep["esito"] == "incompleto" and gone in rep["voci_mancanti_elenco"],
          rep["problemi"][:3])

    # errore di visita dell'inventario: il piano lo rifiuta
    locked = root / "outputs" / "chiusa"
    locked.mkdir()
    locked.chmod(0)
    s = inventory(root, tmp / "inv-err")
    locked.chmod(0o755)
    err = fails(lambda: plan(tmp / "inv-err", tmp / "piani" / "gerr", "gerr"))
    check("piano_rifiuta_inventario_con_errori", s["n_errori"] >= 1 and "non accettabile" in err
          and not (tmp / "piani" / "gerr").exists(), (s["n_errori"], err))
    locked.rmdir()

    # file non volatile cambiato dopo l'inventario: blocca il volume
    inventory(root, tmp / "inv-b")
    pb = tmp / "piani" / "gb"
    plan(tmp / "inv-b", pb, "gb", solo_categorie="codice")
    write(root / "scripts" / "a.py", "print('cambiato')\n")
    err = fails(lambda: create(pb, staging, state))
    check("crea_cambiato_bloccante", "cambiato dopo l'inventario" in err, err)

    # marcatore pronto di un altro piano con gli stessi nomi: niente riuso, né in creazione né in completezza
    inventory(root, tmp / "inv-c")
    pc1, pc2 = tmp / "piani" / "gc-1", tmp / "piani" / "gc-2"
    plan(tmp / "inv-c", pc1, "gc", solo_categorie="codice")
    create(pc1, staging, state)
    ids1 = [v["id"] for v in json.loads((pc1 / "piano.json").read_text())["volumi"]]
    fake_verify(state / "gc", "gc", ids1)
    write(root / "scripts" / "a.py", "print('altro contenuto, stesso nome')\n")
    inventory(root, tmp / "inv-d")
    plan(tmp / "inv-d", pc2, "gc", solo_categorie="codice")
    err = fails(lambda: create(pc2, staging, state))
    check("crea_marcatore_altro_piano", "marcatore di un altro piano" in err, err)
    err2, rep = completeness(state / "gc", [pc2], tmp / "comp-gc2")
    check("completezza_marcatore_altro_piano", err2 != "" and any("piano_sha256" in x for x in rep["problemi"]),
          rep["problemi"][:3])
    err3, rep = completeness(state / "gc", [pc1], tmp / "comp-gc1")
    check("completezza_piano_giusto", err3 == "" and rep["esito"] == "completo", rep["problemi"][:3])

    # base completa, poi supplemento autonomo (codice e un sottoinsieme dichiarato) con gli stessi nomi di volume
    pbase = tmp / "piani" / "gbase"
    pb_plan = plan(tmp / "inv-d", pbase, "gbase")
    create(pbase, staging, state)
    fake_verify(state / "gbase", "gbase", [v["id"] for v in pb_plan["volumi"]])
    err, rep = completeness(state / "gbase", [pbase], tmp / "comp-base")
    check("base_completa", err == "", rep["problemi"][:3])
    write(root / "scripts" / "a.py", "print('codice aggiornato del supplemento')\n")
    (root / "scripts" / "nuovo.py").unlink()
    write(root / "revisioni" / "esiti.tsv", "unità\tesito\nu1\tok\n")
    inventory(root, tmp / "inv-e")
    subset = tmp / "sottoinsieme.txt"
    subset.write_text("hf/hub/models--o--m/blobs/abc\n")
    psup = tmp / "piani" / "gsup"
    ps_plan = plan(tmp / "inv-e", psup, "gsup", solo_categorie="codice,hf-models--o--m",
                   sottoinsieme=[f"hf-models--o--m={subset}"], dipende_da="gbase")
    excl = list(bk.read_jsonl_gz(psup / "esclusioni.jsonl.gz"))
    check("supplemento_esclusioni_dichiarate", ps_plan["dipende_da"] == "gbase" and excl and all(
        e["motivo"] == "fuori dal sottoinsieme dichiarato" and "snapshots" in e["p"] for e in excl), excl)
    create(psup, staging, state)
    fake_verify(state / "gsup", "gsup", [v["id"] for v in ps_plan["volumi"]])
    err, rep = completeness(state / "gsup", [psup], tmp / "comp-sup")
    check("supplemento_completo", err == "" and rep["totali"]["esclusioni_dichiarate"] == len(excl), rep["problemi"][:3])
    base_code = [str(state / "gbase" / "indici" / f"{v['id']}.jsonl.gz") for v in pb_plan["volumi"]
                 if v["categoria"] == "codice"]
    sup_code = [str(state / "gsup" / "indici" / f"{v['id']}.jsonl.gz") for v in ps_plan["volumi"]
                if v["categoria"] == "codice"]
    bk.cmd_rimossi(ns(base=base_code, nuovo=sup_code, out=str(tmp / "rimossi.txt")))
    check("rimossi_elenco", "scripts/nuovo.py" in (tmp / "rimossi.txt").read_text().splitlines())
    # ripristino in due passaggi, con percorsi relativi (anche la cartella degli indici) e la stessa radice
    (tmp / "indici-base").symlink_to(state / "gbase" / "indici")
    r1 = run_script("ripristina_volumi.sh", "comp-base/volumi.tsv", "comp-base/parti.tsv", "staging/gbase",
                    "indici-base", "ricostruito", "tutti", cwd=tmp)
    (tmp / "indici").symlink_to(state / "gsup" / "indici")
    r2 = run_script("ripristina_volumi.sh", "comp-sup/volumi.tsv", "comp-sup/parti.tsv", "staging/gsup", "indici",
                    "ricostruito", "tutti", "--rimuovi-elenco", "rimossi.txt", cwd=tmp)
    rb = tmp / "ricostruito"
    marks = (tmp / "ricostruito.ripristino-fatti").read_text().splitlines()
    code_marks = [m for m in marks if m.split("\t")[0] == "codice-001"]
    check("ripristino_base_e_supplemento", r1.returncode == 0 and r2.returncode == 0
          and (rb / "scripts" / "a.py").read_text() == "print('codice aggiornato del supplemento')\n"
          and not (rb / "scripts" / "nuovo.py").exists() and (rb / "revisioni" / "esiti.tsv").is_file()
          and (rb / "hf/hub/models--o--m/blobs/abc").is_file() and len(code_marks) == 2
          and code_marks[0] != code_marks[1], (r1.stderr[-300:], r2.stderr[-300:], code_marks))
    r3 = run_script("ripristina_volumi.sh", "comp-sup/volumi.tsv", "comp-sup/parti.tsv", "staging/gsup", "indici",
                    "ricostruito", "tutti", cwd=tmp)
    check("ripristino_ripresa_senza_rifare", r3.returncode == 0 and "== " not in r3.stdout, r3.stdout[-300:])
    # composizione: stato finale uguale agli indici in ordine meno i rimossi; un file vecchio della base la rompe
    all_idx = [str(state / g / "indici" / f"{v['id']}.jsonl.gz")
               for g, pl in (("gbase", pb_plan), ("gsup", ps_plan)) for v in pl["volumi"]]
    blob = "hf/hub/models--o--m/blobs/abc"
    comp = dict(radice=str(rb), indici=all_idx, rimossi=str(tmp / "rimossi.txt"), git_commit=None, progetto=None,
                confronta=[], atteso=[f"{blob}={bk.file_sha256(root / blob)}"])
    err = fails(lambda: bk.cmd_composizione(ns(**comp, report=str(tmp / "comp-ok.json"))))
    rep = json.loads((tmp / "comp-ok.json").read_text())
    check("composizione_conforme", err == "" and rep["esito"] == "conforme" and rep["totali"]["in_piu"] == 0
          and rep["attesi"][blob]["sha256"] == rep["attesi"][blob]["sha256_atteso"], rep["problemi"][:3])
    write(rb / "scripts" / "nuovo.py", "print('rimasto dalla base')\n")
    err = fails(lambda: bk.cmd_composizione(ns(**comp, report=str(tmp / "comp-vecchio.json"))))
    rep = json.loads((tmp / "comp-vecchio.json").read_text())
    check("composizione_file_vecchio", err != "" and rep["esito"] == "non conforme"
          and any(x.startswith("scripts/nuovo.py:") for x in rep["problemi"]), rep["problemi"][:3])
    (rb / "scripts" / "nuovo.py").unlink()
    # commit: scripts/a.py è cambiato dopo c1 e non è volatile, quindi lo stato di Git non è accettato
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    err = fails(lambda: bk.cmd_composizione(ns(**{**comp, "git_commit": head}, report=str(tmp / "comp-git.json"))))
    rep = json.loads((tmp / "comp-git.json").read_text())
    check("composizione_git_non_volatile", err != "" and rep["git"]["head"] == head
          and any("scripts/a.py diverso dal commit" in x for x in rep["problemi"]), rep["problemi"][:3])
    # --elimina-parti solo dopo il controllo: una parte alterata ferma tutto e le parti restano
    copy = tmp / "parti-copia"
    shutil.copytree(staging / "gsup", copy)
    victim = sorted((copy / "codice").glob("codice-001.tar*"))[0]  # il primo volume: nessuna parte va cancellata
    victim.write_bytes(victim.read_bytes() + b"x")
    r4 = run_script("ripristina_volumi.sh", "comp-sup/volumi.tsv", "comp-sup/parti.tsv", copy, "indici",
                    "ricostruito-2", "tutti", "--elimina-parti", cwd=tmp)
    check("elimina_parti_solo_dopo_verifica", r4.returncode != 0 and victim.exists() and len(list(copy.rglob("*.tar*"))) ==
          len(list((staging / "gsup").rglob("*.tar*"))), r4.stderr[-200:])

    # confronto Mac: ingressi mancanti, vuoti o illeggibili non danno un confronto riuscito
    cat_out = tmp / "catalogo3"
    bk.cmd_catalogo(ns(root=str(root), inventario=str(tmp / "inv-d"), stato=str(state / "gbase"), out=str(cat_out)))
    ref, catf = cat_out / "hash_riferimento_dati.tsv", cat_out / "catalogo_recupero.tsv"
    mac, cwd = tmp / "mac3", tmp / "confronto3"
    write(mac / "uno.bin", b"uno")
    cwd.mkdir()
    r = run_script("mac_confronta_hash.sh", mac, ref, tmp / "non-esiste.tsv", cwd=cwd)
    check("mac_catalogo_mancante", r.returncode != 0 and "NON VALIDO" in r.stderr, r.stderr[-200:])
    empty = tmp / "catalogo-vuoto.tsv"
    empty.write_text(catf.read_text().splitlines()[0] + "\n")
    r = run_script("mac_confronta_hash.sh", mac, ref, empty, cwd=cwd)
    check("mac_catalogo_vuoto", r.returncode != 0 and "nessun hash" in r.stderr, r.stderr[-200:])
    blocked = mac / "chiuso.bin"
    write(blocked, b"segreto")
    blocked.chmod(0)
    r = run_script("mac_confronta_hash.sh", mac, ref, catf, cwd=cwd)
    blocked.chmod(0o644)
    check("mac_file_illeggibile", r.returncode != 0 and "NON VALIDO" in r.stderr
          and not (cwd / "confronto-uguali.tsv").exists(), r.stderr[-200:])

    # caricamento con rclone simulato: stop fra due parti, ripresa senza ricaricare, stop durante l'attesa
    fake = tmp / "fake_rclone.sh"
    fake.write_text(FAKE_RCLONE)
    fake.chmod(0o755)
    remote = tmp / "remoto"
    remote.mkdir()
    pu = tmp / "piani" / "gu"
    pu_plan = plan(tmp / "inv-e", pu, "gu", solo_categorie="hf-models--o--m")
    create(pu, staging, state)
    env = {"BACKUP_TRASFERIMENTO": str(tmp), "BACKUP_RCLONE": str(fake), "FAKE_REMOTE": str(remote),
           "FAKE_STOP_FILE": str(state / "STOP-UPLOAD"), "BACKUP_ATTESA_SECONDI": "1", "BACKUP_STABILE_SECONDI": "1",
           "BACKUP_PASSO_MEMORIA": "1"}
    big = next(v["id"] for v in pu_plan["volumi"] if v["grande"])
    u1 = run_script("backup_upload.sh", "lvl-ist", "gu", pu, env={**env, "FAKE_STOP_DOPO": "2"})
    big_parts = sorted((staging / "gu" / "hf-models--o--m").glob(f"{big}.tar.part*"))
    check("upload_stop_fra_parti", u1.returncode == 0 and "prima della parte" in u1.stdout
          and not (state / "gu" / "vol" / f"{big}.verificato.tsv").exists() and len(big_parts) == 5
          and len(list((remote / "LimiteVL-backup/gu/volumi/hf-models--o--m").glob(f"{big}*"))) == 1,
          (u1.stdout[-400:], u1.stderr[-300:]))
    u2 = run_script("backup_upload.sh", "lvl-ist", "gu", pu, env=env)
    verified = sorted(x.name for x in (state / "gu" / "vol").glob("*.verificato.tsv"))
    check("upload_ripresa", u2.returncode == 0 and "già su Drive" in u2.stdout and len(verified) == len(pu_plan["volumi"])
          and not list((staging / "gu").rglob("*.tar*")), (u2.stdout[-400:], u2.stderr[-300:]))
    err, rep = completeness(state / "gu", [pu], tmp / "comp-gu")
    check("upload_completezza", err == "", rep["problemi"][:3])
    pw = tmp / "piani" / "gw"
    plan(tmp / "inv-e", pw, "gw", solo_categorie="hf-models--o--m")
    proc = subprocess.Popen(["bash", str(ROOT / "scripts" / "backup_upload.sh"), "lvl-ist", "gw", str(pw)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env={**os.environ, **env, "BACKUP_CREATORE_ATTIVO": "1"})
    time.sleep(2.5)
    (state / "STOP-UPLOAD").touch()
    out, errs = proc.communicate(timeout=30)
    check("upload_stop_in_attesa", proc.returncode == 0 and "durante l'attesa" in out
          and not (remote / "LimiteVL-backup/gw/volumi").exists(), (out[-300:], errs[-300:]))

    # confronto delle firme (sola lettura): il codice è cambiato, i modelli no
    bk.cmd_firme(ns(piano_base=str(pbase / "piano.json"), inventario=str(tmp / "inv-e"), out=str(tmp / "firme.json")))
    fr = json.loads((tmp / "firme.json").read_text())["categorie"]
    check("firme", fr["codice"] == "cambiata" and fr["hf-models--o--m"] == "uguale", fr)



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
    with tempfile.TemporaryDirectory() as d:
        test_correzioni(Path(d))
    print(f"{len(DONE) - len(FAILS)} controlli su {len(DONE)}")
    if FAILS:
        print("falliti: " + "; ".join(FAILS))
        sys.exit(1)


if __name__ == "__main__":
    main()
