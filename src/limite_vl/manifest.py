"""Manifest delle ammissioni (docs/specs/S1_RECORD_MANIFEST.md), nel sottoinsieme che serve a rappresentare e applicare
il candidato dei sintetici v4 (S1 §9). Solo libreria standard.

- Identità (S1 §1): digest sha256 completi di JSON canonico, con la mappa digest -> forma canonica che ferma la
  costruzione se lo stesso digest compare con un contenuto diverso (`Digests`). Un prefisso breve non vale come chiave
  (`require_digest`).
- Record e gruppi: `record_id` = {sorgente, revisione, record}; `group_id` = {indice, membri ordinati}.
- Split per gruppo (S1 §3, `assign_splits`): i gruppi con record di valutazione vanno in `test`; fra gli altri una quota
  di dev stratificata per famiglia, con un seme e gruppi indivisibili.
- Lettura verificata (`load`): sha256 di records.jsonl.gz, record_id ricalcolati, un solo split per gruppo, conteggi per
  shard. `Manifest.admission` lega il manifest agli shard di una sorgente (sha256 e dimensione su disco, verificati una
  volta per manifest) e restituisce il filtro dei reader (`Admission`). Ogni incoerenza è un `ManifestError`: il
  caricamento si ferma, senza ripiegare sull'intero dataset.
- Ammissione per compito (C7, UniMER-1M e MAVIS-Geometry, `TASKED`): il tipo di loader legge un solo insieme di record
  e svolge compiti registrati nel manifest; il manifest deve essere della sorgente e della revisione degli shard. Si
  leggono solo i record ammessi dello split che hanno almeno un compito ammesso per quel tipo; il reader allega al
  campione la decisione del record (`ADMISSION_FILE`), che il loader applica prima di scegliere il compito. Le
  rietichettature vengono dalla lista registrata nel manifest (`Manifest.relabels`). "Ammesso" è la decisione operativa
  del candidato: non è gold e non certifica le risposte.

Fuori da questo sottoinsieme (S1 §9): `variant_id` e `uid`, che dipendono dal contratto di risposta (S3, D3); hash dei
token di input e di target; indice globale fra sorgenti (S1 §4) e import delle esclusioni storiche (S1 §5).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "limite-vl-manifest/1"
SPLITS = ("train", "dev", "test")
STATES = ("ammesso", "quarantena", "escluso")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
# Cartelle di shard che si leggono solo attraverso un manifest: una sorgente senza manifest si ferma (stream.make_sources).
# Per UniMER e MAVIS tutte le varianti: quelle non registrate in un manifest non si leggono affatto (UniMER-Test, in
# unimer/*/test, è valutazione e lo leggono gli strumenti del pannello, non make_sources).
MANIFEST_REQUIRED = ("data/shards/synth/v4", "data/shards/unimer", "data/shards/mavis-geometry")
# Ammissione per compito (C7): sorgente del manifest -> per ogni tipo di loader (data.SAMPLE_LOADERS) l'insieme dei
# record che legge e il suo compito; None = i compiti per domanda del record, scelti dal loader fra quelli ammessi.
TASKED = {
    "unimer": {"unimer_formula": ("train", "transcribe"), "unimer_line": ("righe", "transcribe_line"),
               "unimer_line_select": ("righe", "transcribe_line_select")},
    "mavis-geometry": {"mavis_f1": ("train", None)},
}
# File aggiunto dal reader ai file di un campione di una sorgente TASKED: la decisione del manifest per il record (JSON
# canonico). Non è un file dello shard.
ADMISSION_FILE = "ammissione"
MAVIS_QA_FIELDS = ("type", "q", "answer")  # campi del digest di una domanda di MAVIS (scripts/mavis_manifest.qa_sha)
HASH_THREADS = 4  # shard verificati in parallelo (hashlib rilascia il GIL)


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


def qa_item_sha256(q: dict, fields: tuple[str, ...] = MAVIS_QA_FIELDS) -> str:
    """Digest di una domanda come lo registra il manifest di MAVIS (tipo, testo e risposta)."""
    return sha256_text(canonical({k: q[k] for k in fields}))


def qa_list_sha256(pairs) -> str:
    """Digest dell'elenco ordinato [tipo, digest] delle domande di un record: lega il pool del loader al manifest."""
    return sha256_text(canonical([[t, s] for t, s in pairs]))


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
    sha256 del JSON, sha256 del PNG, compiti ammessi, compiti bloccati, dati per il loader). Si leggono solo i record
    `ammesso` dello split richiesto; la validazione usa quelli `ammesso` dello split dev (S1 §3). Per una sorgente
    TASKED (`tasked`) il record deve avere anche il compito del tipo di loader (`task`) o, se `task` è None, almeno un
    compito ammesso; il reader gli allega la decisione (`attach`)."""

    manifest: str  # percorso di manifest.json
    manifest_sha256: str
    split: str
    shards: list[str]
    entries: dict[str, tuple] = field(repr=False)
    expected: dict[str, int] = field(default_factory=dict)  # "split/stato" -> record in questi shard
    task: str | None = None
    tasked: bool = False
    identity: dict = field(default_factory=dict)  # sorgente, revisione, nome e versione del manifest, insieme, tipo
    expected_tasks: dict[str, int] = field(default_factory=dict)  # "split/stato del compito/compito" -> record
    percorsi: list[str] = field(default_factory=list, repr=False)  # gli shard come li registra il manifest

    def _admitted(self, e: tuple) -> bool:
        if e[1] != "ammesso":
            return False
        if not self.tasked:
            return True
        return self.task in e[4] if self.task else bool(e[4])

    def allows(self, key: str) -> bool:
        e = self.entries.get(key)
        return e is not None and e[0] == self.split and self._admitted(e)

    def validation(self, key: str) -> bool:
        e = self.entries.get(key)
        return e is not None and e[0] == "dev" and self._admitted(e)

    def refusal(self, key: str, split: str | None = None) -> str | None:
        """Motivo per cui il record non entra nello split (quello della sorgente se None): None se entra. Motivi:
        assente, split_<altro>, quarantena, escluso, compito_non_ammesso/<compito>, nessun_compito_ammesso."""
        e = self.entries.get(key)
        if e is None:
            return "assente"
        if e[0] != (split or self.split):
            return f"split_{e[0]}"
        if e[1] != "ammesso":
            return e[1]
        if not self._admitted(e):
            return f"compito_non_ammesso/{self.task}" if self.task else "nessun_compito_ammesso"
        return None

    def check(self, key: str, files: dict[str, bytes]) -> str | None:
        """Motivo per cui il record letto non è quello ammesso (JSON o PNG con altri byte), oppure None. Va fatto sui
        byte dello shard, prima di ogni trasformazione (rietichettatura)."""
        e = self.entries.get(key)
        if e is None:
            return "chiave assente dal manifest"
        if hashlib.sha256(files.get("json", b"")).hexdigest() != e[2]:
            return "JSON diverso da quello del manifest"
        if e[3] is not None and hashlib.sha256(files.get("png", b"")).hexdigest() != e[3]:
            return "PNG diverso da quello del manifest"
        return None

    def attach(self, key: str, files: dict[str, bytes]) -> dict[str, bytes]:
        """I file del campione con la decisione del manifest per il record (ADMISSION_FILE), per una sorgente TASKED;
        gli altri invariati."""
        if not self.tasked:
            return files
        e = self.entries[key]
        doc = {"manifest_sha256": self.manifest_sha256, "split": e[0], "stato": e[1], "compito": self.task,
               "compiti_ammessi": list(e[4]), "compiti_bloccati": list(e[5]), **(e[6] or {})}
        return {**files, ADMISSION_FILE: canonical(doc).encode()}

    def describe(self) -> dict:
        admitted = sum(1 for k in self.entries if self.allows(k))
        return {"manifest": self.manifest, "manifest_sha256": self.manifest_sha256, "split": self.split,
                **self.identity, "compito_del_tipo": self.task,
                "shards": [Path(p).name for p in self.shards], "attesi": dict(self.expected),
                "attesi_per_compito": dict(self.expected_tasks),
                "ammessi_nello_split": admitted, "record_in_questi_shard": len(self.entries),
                "decisione": "ammesso = decisione operativa del manifest candidato, non gold: non certifica le risposte"}


def _family(kind: str) -> str | None:
    return next((src for src, kinds in TASKED.items() if kind in kinds), None)


@dataclass
class Manifest:
    folder: Path
    meta: dict
    manifest_sha256: str
    by_shard: dict[str, dict[str, dict]]  # percorso relativo dello shard -> chiave -> record
    # shard già verificati su disco (sha256 e dimensione) per questo manifest: una volta sola per processo
    verified: set = field(default_factory=set, repr=False)

    def tasked_source(self, kind: str) -> str | None:
        """Sorgente TASKED del tipo di loader, se c'è. Errore se tipo e sorgente del manifest non vanno insieme: un tipo
        di UniMER o di MAVIS vuole il manifest della sua sorgente, e un manifest di UniMER o di MAVIS si legge solo coi
        tipi i cui compiti registra."""
        by_kind, mine = _family(kind), self.meta.get("sorgente")
        by_meta = mine if mine in TASKED else None
        if by_kind != by_meta:
            raise ManifestError(f"{self.folder}: il tipo di loader {kind!r} non legge il manifest della sorgente "
                                f"{mine!r} (tipi ammessi: {sorted(TASKED.get(mine, {}))})")
        return by_kind

    def _verify_shards(self, items: list[tuple[Path, dict]]) -> None:
        """sha256 e dimensione su disco degli shard non ancora verificati, in parallelo: ogni shard si legge una volta
        per manifest, anche se più sorgenti (tipi o split diversi) lo usano."""
        todo = []
        for path, s in items:
            if not path.is_file():
                raise ManifestError(f"shard mancante: {path}")
            if path.stat().st_size != s["bytes"]:
                raise ManifestError(f"{path}: dimensione diversa da quella del manifest {self.folder}")
            if str(path) not in self.verified:
                todo.append((path, s))
        with ThreadPoolExecutor(max_workers=max(1, min(HASH_THREADS, len(todo)))) as pool:
            digests = list(pool.map(lambda x: _sha256_file(x[0]), todo))
        for (path, s), digest in zip(todo, digests):
            if digest != s["sha256"]:
                raise ManifestError(f"{path}: sha256 diverso da quello del manifest {self.folder}")
            self.verified.add(str(path))

    def _check_revision(self, paths: list[Path]) -> None:
        """La sorgente dichiarata dal manifest dello sharding di ogni cartella di shard (`source`: repository@revisione)
        deve essere quella del manifest."""
        want = f"{(self.meta.get('identita') or {}).get('repository')}@{self.meta.get('revisione')}"
        for folder in sorted({p.parent for p in paths}):
            sharding = folder / "manifest.json"
            try:
                got = json.loads(sharding.read_text(encoding="utf-8")).get("source")
            except (OSError, ValueError) as exc:
                raise ManifestError(f"{sharding}: manifest dello sharding illeggibile ({exc})") from None
            if got != want:
                raise ManifestError(f"{folder}: shard di {got!r}, il manifest {self.folder} è di {want!r}")

    def admission(self, shards: list[str], split: str, kind: str | None = None) -> Admission:
        """Filtro per gli shard di una sorgente e uno split. Errore se uno shard non è nel manifest, se su disco ha
        sha256 o dimensione diversi, se una chiave compare in due shard o se lo split non ha record ammessi. Con un
        tipo di loader TASKED (`kind`): manifest della sua sorgente, revisione degli shard uguale, shard tutti
        dell'insieme che il tipo legge, compito del tipo registrato in ogni record; si contano compiti ammessi e
        bloccati per record."""
        if split not in SPLITS:
            raise ManifestError(f"split {split!r}: atteso uno di {SPLITS}")
        if kind is None and self.meta.get("sorgente") in TASKED:
            raise ManifestError(f"{self.folder}: manifest con ammissione per compito, serve il tipo di loader")
        source = self.tasked_source(kind) if kind is not None else None
        insieme, task = TASKED[source][kind] if source else (None, None)
        known = {str(shard_path(s["percorso"])): s for s in self.meta["shards"]}
        unknown = [p for p in shards if str(Path(p).resolve()) not in known]
        if unknown:
            raise ManifestError(f"{self.folder}: shard non registrati nel manifest: {unknown[:3]}")
        items = [(Path(p).resolve(), known[str(Path(p).resolve())]) for p in shards]
        if source:
            wrong = [s["nome"] for _, s in items if s.get("insieme") != insieme]
            if wrong:
                raise ManifestError(f"{self.folder}: il tipo {kind} legge l'insieme {insieme!r}, gli shard {wrong[:3]} "
                                    "sono di un altro insieme")
            self._check_revision([p for p, _ in items])
        self._verify_shards(items)
        entries: dict[str, tuple] = {}
        expected: Counter = Counter()
        expected_tasks: Counter = Counter()
        shared: dict = {}  # tuple di compiti e stringhe uguali condivise fra i record (memoria)
        for _, s in items:
            for key, r in self.by_shard.get(s["percorso"], {}).items():
                if key in entries:
                    raise ManifestError(f"{key}: in due shard della stessa sorgente")
                images = r.get("immagini") or [{}]
                split_r, state = sys.intern(r["split"]), sys.intern(r["stato"])
                admitted, blocked, extra = (), (), None
                if source:
                    if r.get("insieme") != insieme:
                        raise ManifestError(f"{key}: record dell'insieme {r.get('insieme')!r}, il tipo {kind} legge "
                                            f"{insieme!r}")
                    admitted, blocked, extra = _record_tasks(key, r, task, expected_tasks)
                    admitted = shared.setdefault(admitted, admitted)
                    blocked = shared.setdefault(blocked, blocked)
                entries[key] = (split_r, state, r["json_sha256"], images[0].get("sha256"), admitted, blocked, extra)
                expected[f"{r['split']}/{r['stato']}"] += 1
        identity = {"sorgente": self.meta.get("sorgente"), "revisione": self.meta.get("revisione"),
                    "nome": self.meta.get("nome"), "versione": self.meta.get("versione")}
        if source:
            identity.update(insieme=insieme, tipo=kind)
        adm = Admission(str(self.folder / "manifest.json"), self.manifest_sha256, split, [str(p) for p in shards],
                        entries, dict(sorted(expected.items())), task=task, tasked=bool(source), identity=identity,
                        expected_tasks=dict(sorted(expected_tasks.items())), percorsi=[s["percorso"] for _, s in items])
        if not any(adm.allows(k) for k in entries):
            raise ManifestError(f"{self.folder}: nessun record ammesso nello split {split} di questi shard")
        return adm

    def relabels(self, adm: Admission) -> dict[str, tuple[str, str]]:
        """Rietichettature dei record della sorgente (trasformazione `rietichettatura`), dalla lista registrata in
        liste_importate (sha256 ricontrollato su disco): chiave -> (etichetta dello shard, etichetta nuova). Ogni record
        rietichettato deve averla, con lo sha256 dell'etichetta dello shard (etichetta_sha256) e di quella nuova (della
        trasformazione); una riga della lista per un record della sorgente senza trasformazione è un errore."""
        records = {k: r for perc in adm.percorsi for k, r in self.by_shard.get(perc, {}).items()}
        wanted = {k: r["trasformazione"] for k, r in records.items()
                  if (r.get("trasformazione") or {}).get("nome") == "rietichettatura"}
        out: dict[str, tuple[str, str]] = {}
        loaded: dict[str, dict] = {}
        for lista in sorted({t.get("lista") for t in wanted.values()}):
            entry = ((self.meta.get("liste_importate") or {}).get(lista) or {}).get("file", {}).get("relabel.jsonl")
            if not entry or not entry.get("path") or not HEX64.match(str(entry.get("sha256", ""))):
                raise ManifestError(f"{self.folder}: rietichettature di {lista!r} senza relabel.jsonl registrato")
            path = Path(entry["path"])
            if not path.is_file() or _sha256_file(path) != entry["sha256"]:
                raise ManifestError(f"{path}: relabel.jsonl mancante o diverso da quello registrato nel manifest")
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        x = json.loads(line)
                        loaded[x["key"]] = x
        stray = sorted(k for k in loaded if k in records and k not in wanted)
        if stray:
            raise ManifestError(f"{self.folder}: rietichettature della lista per record senza trasformazione: {stray[:3]}")
        for k, t in wanted.items():
            x = loaded.get(k)
            if x is None:
                raise ManifestError(f"{k}: rietichettatura registrata nel manifest ma assente dalla lista")
            if sha256_text(x["label"]) != records[k]["etichetta_sha256"] \
                    or sha256_text(x["new_label"]) != t.get("etichetta_sha256"):
                raise ManifestError(f"{k}: etichette della lista diverse dagli sha256 del manifest")
            out[k] = (x["label"], x["new_label"])
        return out


def _record_tasks(key: str, r: dict, task: str | None, counter: Counter) -> tuple[tuple, tuple, dict | None]:
    """(compiti ammessi, compiti bloccati, dati per il loader) di un record TASKED. UniMER: compiti come nomi, con lo
    stato del record; MAVIS: compiti con il loro stato (shape_type in quarantena), digest delle domande e della
    descrizione. Un compito ammesso del record vuole un record ammesso; un compito di domande vuole almeno una domanda."""
    tasks = r.get("compiti") or []
    if task is not None:  # compiti come nomi: lo stato è quello del record
        names = [t if isinstance(t, str) else t.get("compito") for t in tasks]
        if task not in names:
            raise ManifestError(f"{key}: il record non registra il compito {task!r} ({names})")
        for t in names:
            counter[f"{r['split']}/{r['stato']}/{t}"] += 1
        ok = r["stato"] == "ammesso"
        admitted = tuple(sorted(names)) if ok else ()
        blocked = () if ok else tuple(sorted(names))
        t = r.get("trasformazione") or {}
        extra = {"etichetta_sha256": t["etichetta_sha256"]} if t.get("nome") == "rietichettatura" else None
        return admitted, blocked, extra
    admitted, blocked = [], []
    for t in tasks:
        name, state = t["compito"], t["stato"]
        counter[f"{r['split']}/{state}/{name}"] += 1
        usable = name == "describe" or t.get("domande", 0) > 0
        (admitted if r["stato"] == "ammesso" and state == "ammesso" and usable else blocked).append(name)
    qa = r.get("qa") or []
    extra = {"qa_sha256": qa_list_sha256((q["tipo"], q["sha256"]) for q in qa),
             "descrizione_sha256": r.get("descrizione_sha256")}
    if r.get("costruzione_qa"):  # C9: costruzione delle domande dichiarata dal record; assente = storica
        extra["costruzione_qa"] = r["costruzione_qa"]
    return tuple(sorted(admitted)), tuple(sorted(blocked)), extra


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
