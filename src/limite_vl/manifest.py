"""Manifest delle ammissioni (docs/specs/S1_RECORD_MANIFEST.md), nel sottoinsieme che serve a rappresentare e applicare
il candidato dei sintetici v4 (S1 §9). Solo libreria standard.

- Identità (S1 §1): digest sha256 completi di JSON canonico, con la mappa digest -> forma canonica che ferma la
  costruzione se lo stesso digest compare con un contenuto diverso (`Digests`). Un prefisso breve non vale come chiave
  (`require_digest`).
- Record e gruppi: `record_id` = {sorgente, revisione, record}; `group_id` = {indice, membri ordinati}.
- Split per gruppo (S1 §3, `assign_splits`): i gruppi con record di valutazione vanno in `test`; fra gli altri una quota
  di dev stratificata per famiglia, con un seme e gruppi indivisibili.
- Lettura verificata (`load`): sha256 di records.jsonl.gz, record_id ricalcolati, un solo split per gruppo, conteggi per
  shard. `Manifest.admission` lega il manifest agli shard di una sorgente (sha256 e dimensione su disco) e restituisce
  il filtro dei reader (`Admission`). Ogni incoerenza è un `ManifestError`: il caricamento si ferma, senza ripiegare
  sull'intero dataset.

Fuori da questo sottoinsieme (S1 §9): `variant_id` e `uid`, che dipendono dal contratto di risposta (S3, D3); hash dei
token di input e di target; indice globale fra sorgenti (S1 §4) e import delle esclusioni storiche (S1 §5); ammissione
della singola QA dentro un record ammesso.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "limite-vl-manifest/1"
SPLITS = ("train", "dev", "test")
STATES = ("ammesso", "quarantena", "escluso")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
# Cartelle di shard che si leggono solo attraverso un manifest: una sorgente senza manifest si ferma (stream.make_sources).
MANIFEST_REQUIRED = ("data/shards/synth/v4",)


class ManifestError(ValueError):
    """Manifest mancante, incoerente o riferito ad altri shard: il caricamento si ferma."""


# ---------------------------------------------------------------- identità

def canonical(obj) -> str:
    """JSON canonico (S1 §1): chiavi ordinate, nessuno spazio, UTF-8. Valori: stringhe, interi, booleani, null, liste e
    oggetti con chiavi stringa. I numeri non interi si rifiutano, perché la loro forma testuale non è univoca."""
    def check(x) -> None:
        if x is None or isinstance(x, (bool, int, str)):
            return
        if isinstance(x, float):
            raise TypeError(f"numero non intero nel JSON canonico: {x!r}")
        if isinstance(x, (list, tuple)):
            for y in x:
                check(y)
            return
        if isinstance(x, dict):
            for k, y in x.items():
                if not isinstance(k, str):
                    raise TypeError(f"chiave non stringa nel JSON canonico: {k!r}")
                check(y)
            return
        raise TypeError(f"tipo non ammesso nel JSON canonico: {type(x).__name__}")

    check(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Digests:
    """Digest sha256 completi di JSON canonici, con la mappa digest -> forma canonica (S1 §1). Lo stesso digest con un
    contenuto diverso ferma la costruzione: nessuna fusione, nessuna scelta. `fn` si sostituisce solo nei collaudi."""

    def __init__(self, fn=sha256_text):
        self.fn = fn
        self.seen: dict[str, str] = {}

    def __call__(self, obj) -> str:
        text = canonical(obj)
        digest = self.fn(text)
        if self.seen.setdefault(digest, text) != text:
            raise ManifestError(f"collisione di digest {digest}: due contenuti diversi, costruzione fermata")
        return digest


def require_digest(value, what: str) -> str:
    """Gli identificatori sono digest completi: un prefisso breve non vale come chiave (S1 §1)."""
    if not isinstance(value, str) or not HEX64.match(value):
        raise ManifestError(f"{what}: atteso un digest sha256 completo (64 caratteri esadecimali), non {value!r}")
    return value


def record_id(digests: Digests, source: str, revision: str, record: str) -> str:
    return digests({"sorgente": source, "revisione": revision, "record": record})


def group_id(digests: Digests, index, members: list[str]) -> str:
    for m in members:
        require_digest(m, "membro del gruppo")
    return digests({"indice": index, "membri": sorted(members)})


def qa_digest(digests: Digests, qa: dict) -> str:
    """Digest di una QA: tipo, testo, opzioni e risposta."""
    return digests({k: qa[k] for k in ("type", "q", "options", "answer")})


# ---------------------------------------------------------------- split

def split_rank(seed: int, group: str) -> str:
    """Chiave d'ordine di un gruppo per l'estrazione di dev: dipende solo da seme e identificatore del gruppo."""
    return sha256_text(canonical({"seme": seed, "gruppo": require_digest(group, "gruppo")}))


def assign_splits(groups: dict[str, dict], fraction: float, seed: int) -> dict[str, str]:
    """Split di ogni gruppo (S1 §3, regola del candidato v4).

    groups: identificatore -> {"famiglia", "valutazione": il gruppo ha record di test o delle coppie, "ammessi": scene
    ammesse}. I gruppi di valutazione vanno in `test`. Gli altri, per famiglia e in ordine di `split_rank`, vanno in
    `dev` finché le scene ammesse di dev della famiglia sono meno di round(fraction · ammesse della famiglia); il resto
    in `train`. I gruppi non si dividono: la quota è superata al più dalla dimensione dell'ultimo gruppo. Un gruppo
    senza scene ammesse resta in train, dove nessun suo record viene letto."""
    if not 0 <= fraction < 1:
        raise ValueError(f"quota di dev {fraction}: attesa in [0, 1)")
    out: dict[str, str] = {}
    by_family: dict[str, list[str]] = defaultdict(list)
    for g, info in groups.items():
        if info["valutazione"]:
            out[g] = "test"
        else:
            by_family[info["famiglia"]].append(g)
    for family, members in sorted(by_family.items()):
        target = round(fraction * sum(groups[g]["ammessi"] for g in members))
        taken = 0
        for g in sorted(members, key=lambda g: split_rank(seed, g)):
            if groups[g]["ammessi"] > 0 and taken < target:
                out[g] = "dev"
                taken += groups[g]["ammessi"]
            else:
                out[g] = "train"
    return out


# ---------------------------------------------------------------- lettura e filtro dei reader

def shard_path(rel: str) -> Path:
    """Percorso di uno shard del manifest, registrato relativo alla radice del progetto."""
    return (ROOT / rel).resolve()


def manifest_required(shards: list[str]) -> list[str]:
    """Gli shard che si leggono solo attraverso un manifest (MANIFEST_REQUIRED)."""
    roots = [(ROOT / r).resolve() for r in MANIFEST_REQUIRED]
    return [p for p in shards if any(root in Path(p).resolve().parents for root in roots)]


@dataclass
class Admission:
    """Filtro dei reader per una sorgente: i suoi shard, verificati contro il manifest, e per ogni chiave (split, stato,
    sha256 del JSON, sha256 del PNG). Si leggono solo i record `ammesso` dello split richiesto; la validazione usa quelli
    `ammesso` dello split dev (S1 §3)."""

    manifest: str  # percorso di manifest.json
    manifest_sha256: str
    split: str
    shards: list[str]
    entries: dict[str, tuple[str, str, str, str | None]] = field(repr=False)
    expected: dict[str, int] = field(default_factory=dict)  # "split/stato" -> record in questi shard

    def allows(self, key: str) -> bool:
        e = self.entries.get(key)
        return e is not None and e[0] == self.split and e[1] == "ammesso"

    def validation(self, key: str) -> bool:
        e = self.entries.get(key)
        return e is not None and e[0] == "dev" and e[1] == "ammesso"

    def check(self, key: str, files: dict[str, bytes]) -> str | None:
        """Motivo per cui il record letto non è quello ammesso (JSON o PNG con altri byte), oppure None."""
        e = self.entries.get(key)
        if e is None:
            return "chiave assente dal manifest"
        if hashlib.sha256(files.get("json", b"")).hexdigest() != e[2]:
            return "JSON diverso da quello del manifest"
        if e[3] is not None and hashlib.sha256(files.get("png", b"")).hexdigest() != e[3]:
            return "PNG diverso da quello del manifest"
        return None

    def describe(self) -> dict:
        return {"manifest": self.manifest, "manifest_sha256": self.manifest_sha256, "split": self.split,
                "shards": [Path(p).name for p in self.shards], "attesi": dict(self.expected),
                "ammessi_nello_split": sum(1 for k in self.entries if self.allows(k))}


@dataclass
class Manifest:
    folder: Path
    meta: dict
    manifest_sha256: str
    by_shard: dict[str, dict[str, dict]]  # percorso relativo dello shard -> chiave -> record

    def admission(self, shards: list[str], split: str) -> Admission:
        """Filtro per gli shard di una sorgente e uno split. Errore se uno shard non è nel manifest, se su disco ha
        sha256 o dimensione diversi, se una chiave compare in due shard o se lo split non ha record ammessi."""
        if split not in SPLITS:
            raise ManifestError(f"split {split!r}: atteso uno di {SPLITS}")
        known = {str(shard_path(s["percorso"])): s for s in self.meta["shards"]}
        unknown = [p for p in shards if str(Path(p).resolve()) not in known]
        if unknown:
            raise ManifestError(f"{self.folder}: shard non registrati nel manifest: {unknown[:3]}")
        entries: dict[str, tuple] = {}
        expected: Counter = Counter()
        for p in shards:
            s = known[str(Path(p).resolve())]
            path = Path(p)
            if not path.is_file():
                raise ManifestError(f"shard mancante: {p}")
            if path.stat().st_size != s["bytes"] or _sha256_file(path) != s["sha256"]:
                raise ManifestError(f"{p}: sha256 o dimensione diversi da quelli del manifest {self.folder}")
            for key, r in self.by_shard.get(s["percorso"], {}).items():
                if key in entries:
                    raise ManifestError(f"{key}: in due shard della stessa sorgente")
                images = r.get("immagini") or [{}]
                entries[key] = (r["split"], r["stato"], r["json_sha256"], images[0].get("sha256"))
                expected[f"{r['split']}/{r['stato']}"] += 1
        adm = Admission(str(self.folder / "manifest.json"), self.manifest_sha256, split, [str(p) for p in shards],
                        entries, dict(sorted(expected.items())))
        if not any(adm.allows(k) for k in entries):
            raise ManifestError(f"{self.folder}: nessun record ammesso nello split {split} di questi shard")
        return adm


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def load(folder) -> Manifest:
    """Manifest letto e verificato: schema, sha256 e righe di records.jsonl.gz, record_id ricalcolati dalla terna, un
    solo split per gruppo, chiavi uniche per shard, conteggi per shard uguali a quelli dichiarati. Ogni incoerenza è
    un ManifestError."""
    folder = Path(folder)
    path = folder / "manifest.json"
    if not path.is_file():
        raise ManifestError(f"manifest mancante: {path}")
    raw_meta = path.read_bytes()
    try:
        meta = json.loads(raw_meta)
    except ValueError as exc:
        raise ManifestError(f"{path}: JSON non valido ({exc})") from None
    if meta.get("schema") != SCHEMA:
        raise ManifestError(f"{path}: schema {meta.get('schema')!r}, atteso {SCHEMA}")
    rec = meta.get("records") or {}
    data_path = folder / str(rec.get("file", ""))
    if not rec.get("file") or not data_path.is_file():
        raise ManifestError(f"{path}: file dei record mancante ({rec.get('file')!r})")
    raw = data_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != rec.get("sha256"):
        raise ManifestError(f"{data_path}: sha256 diverso da quello registrato in manifest.json")
    try:
        lines = gzip.decompress(raw).decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ManifestError(f"{data_path}: non si decomprime ({exc})") from None
    if len(lines) != rec.get("righe"):
        raise ManifestError(f"{data_path}: {len(lines)} righe, il manifest ne dichiara {rec.get('righe')}")
    shards = {s["percorso"]: s for s in meta.get("shards", [])}
    digests = Digests()
    by_shard: dict[str, dict[str, dict]] = defaultdict(dict)
    group_split: dict[str, str] = {}
    for n, line in enumerate(lines, 1):
        r = json.loads(line)
        rid = record_id(digests, r["sorgente"], r["revisione"], r["record"])
        if require_digest(r.get("record_id"), f"riga {n}: record_id") != rid:
            raise ManifestError(f"riga {n} ({r['record']}): record_id diverso da quello ricalcolato")
        if r.get("shard") not in shards:
            raise ManifestError(f"riga {n} ({r['record']}): shard {r.get('shard')!r} non registrato")
        if r.get("split") not in SPLITS or r.get("stato") not in STATES:
            raise ManifestError(f"riga {n} ({r['record']}): split {r.get('split')!r} o stato {r.get('stato')!r}")
        g = require_digest(r.get("gruppo"), f"riga {n}: gruppo")
        if group_split.setdefault(g, r["split"]) != r["split"]:
            raise ManifestError(f"riga {n} ({r['record']}): il gruppo {g} è in due split")
        if r["record"] in by_shard[r["shard"]]:
            raise ManifestError(f"riga {n}: {r['record']} ripetuto nello shard {r['shard']}")
        by_shard[r["shard"]][r["record"]] = r
    for rel, s in shards.items():
        if len(by_shard.get(rel, {})) != s.get("record"):
            raise ManifestError(f"{rel}: {len(by_shard.get(rel, {}))} record nel file, il manifest ne dichiara "
                                f"{s.get('record')}")
    return Manifest(folder, meta, hashlib.sha256(raw_meta).hexdigest(), dict(by_shard))
