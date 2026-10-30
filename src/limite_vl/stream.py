"""Flusso misto, deterministico e riprendibile di micro-batch impacchettati per il training.

Sorgenti: shard tar di un tipo (data.SAMPLE_LOADERS) con un peso. Ogni worker del DataLoader legge
tutte le sorgenti nello stesso ordine di shard (mescolato a ogni epoca con il seed) e tiene i campioni
con indice nello shard ≡ id del worker (mod numero di worker): il carico è bilanciato anche con
sorgenti di pochi shard, e nessun campione è letto da due worker. Senza manifest, i primi `val_count`
campioni dell'ultimo shard (in ordine di nome) di ogni sorgente sono di validazione e non entrano mai nel
training (`validation_samples`).

Con un manifest delle ammissioni (limite_vl.manifest, S1; campi `manifest` e `split` della sorgente) si leggono
solo i record `ammesso` dello split richiesto, e la validazione è lo split dev del manifest. Gli shard di
manifest.MANIFEST_REQUIRED si leggono solo così. Ogni record consegnato deve avere gli sha256 di JSON e PNG del
manifest; uno diverso è un errore contato (`manifest_errors/<sorgente>`), mai un passaggio silenzioso. I contatori
dei reader (letti, saltati per il manifest o per le liste, consegnati) e gli esempi costruiti per sorgente
(`loaded/<sorgente>`) sono nel `meta` di ogni micro-batch.

Per ogni esempio il worker sceglie la sorgente con probabilità proporzionale al peso, prende un
campione dal buffer di mescolamento della sorgente, costruisce l'Example, impacchetta le righe
(collator.iter_packed_rows) e consegna micro-batch collati con `meta`: id del worker, stato, esempi
per sorgente. Tutte le scelte casuali sono hash di (seed, worker, contatore), quindi dipendono solo
dallo stato.

Ripresa: lo stato di un worker è, per sorgente, (epoca, posizione nell'ordine degli shard, campioni
letti nello shard, estrazioni dal buffer) più il contatore delle estrazioni di sorgente. Il training
salva l'ultimo stato consegnato da ogni worker; alla ripresa (stesso numero di worker) ogni worker
riparte da lì. Gli esempi che al momento del checkpoint erano nel buffer di mescolamento o in quello
di packing si perdono (al più shuffle_buffer per sorgente + pack_buffer per worker); nessuno si ripete.
"""
from __future__ import annotations

import glob
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterator

from torch.utils.data import IterableDataset, get_worker_info

from .collator import PackedCollator, iter_packed_rows
from .data import SAMPLE_LOADERS, build_example, iter_shard
from .manifest import Admission, ManifestError, manifest_required
from .manifest import load as load_manifest
from .processing import LimiteVLProcessor


@dataclass
class Source:
    name: str
    kind: str  # chiave di data.SAMPLE_LOADERS
    shards: list[str]  # percorsi in ordine di nome
    weight: float
    exclude: frozenset[str] = frozenset()  # chiavi da saltare (decontaminazione, scripts/decontam.py scan)
    # esito di f2_records.f2_data_certification per le sorgenti f2_solution: senza, MixedRowStream non le usa
    certification: dict | None = None
    # rietichettature esatte (scripts/unimer_check.py, relabel.jsonl): chiave -> (etichetta attesa, nuova etichetta)
    relabel: dict[str, tuple[str, str]] | None = None
    # manifest delle ammissioni (S1): solo i record ammessi dello split richiesto, validazione dallo split dev
    admission: Admission | None = None

    @property
    def val_shard(self) -> str:
        return self.shards[-1]


def make_sources(specs: list[dict]) -> list[Source]:
    """[{name, kind, shards: glob, weight, exclude: [file], relabel: [file], manifest: cartella, split}] -> Source, con
    controlli su tipi, nomi e shard presenti. `exclude` (facoltativo) elenca file con una chiave per riga, da saltare
    sempre; per f2_solution vale anche come chiave del problema (campo `problem` del JSON). `relabel` (facoltativo)
    elenca file JSONL {key, label, new_label}: il campo `label` del JSON del campione diventa new_label, solo se
    coincide con `label` (altrimenti la lista è vecchia e il campione fa fallire il training). `uncertified_reason` lo
    legge train_f2.py. `manifest` e `split` (insieme) legano la sorgente a un manifest delle ammissioni
    (limite_vl.manifest.load, Manifest.admission): manifest mancante o incoerente, shard non registrati o diversi
    sono errori; gli shard di MANIFEST_REQUIRED senza manifest anche."""
    sources, names, manifests = [], set(), {}
    for spec in specs:
        if spec["kind"] not in SAMPLE_LOADERS:
            raise ValueError(f"sorgente {spec['name']}: tipo {spec['kind']!r} sconosciuto")
        if spec["name"] in names:
            raise ValueError(f"sorgente ripetuta: {spec['name']}")
        shards = sorted(glob.glob(spec["shards"]))
        if not shards:
            raise FileNotFoundError(f"sorgente {spec['name']}: nessuno shard per {spec['shards']}")
        if float(spec["weight"]) <= 0:
            raise ValueError(f"sorgente {spec['name']}: peso {spec['weight']}")
        names.add(spec["name"])
        exclude: set[str] = set()
        for path in spec.get("exclude", []):
            with open(path) as fh:
                exclude.update(line.strip() for line in fh if line.strip())
        relabel: dict[str, tuple[str, str]] = {}
        for path in spec.get("relabel", []):
            with open(path) as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        relabel[r["key"]] = (r["label"], r["new_label"])
        admission = None
        if "manifest" in spec or "split" in spec:
            if not spec.get("manifest") or not spec.get("split"):
                raise ManifestError(f"sorgente {spec['name']}: 'manifest' e 'split' vanno indicati insieme")
            folder = str(spec["manifest"])
            if folder not in manifests:
                manifests[folder] = load_manifest(folder)
            admission = manifests[folder].admission(shards, spec["split"])
        elif manifest_required(shards):
            raise ManifestError(f"sorgente {spec['name']}: gli shard {manifest_required(shards)[:2]} si leggono solo "
                                "con un manifest delle ammissioni (campi 'manifest' e 'split')")
        sources.append(Source(spec["name"], spec["kind"], shards, float(spec["weight"]), frozenset(exclude),
                              relabel=relabel or None, admission=admission))
    return sources


def apply_relabel(source: Source, key: str, files: dict[str, bytes]) -> dict[str, bytes]:
    """File del campione con l'etichetta sostituita se la chiave è in source.relabel; errore se l'etichetta nel JSON
    non è quella attesa dalla lista."""
    if not source.relabel or key not in source.relabel:
        return files
    old, new = source.relabel[key]
    meta = json.loads(files["json"])
    if meta.get("label") != old:
        raise RuntimeError(f"{source.name}: {key} ha etichetta {meta.get('label')!r}, la lista si aspetta {old!r}")
    meta["label"] = new
    return {**files, "json": json.dumps(meta, ensure_ascii=False).encode()}


def u01(*parts) -> float:
    """Numero in [0, 1) deterministico, funzione solo degli argomenti."""
    digest = hashlib.blake2b("\x1f".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2**64


def shard_order(source: Source, seed: int, epoch: int) -> list[str]:
    order = list(source.shards)
    random.Random(f"{seed}/{source.name}/{epoch}").shuffle(order)  # seed stringa: stabile fra processi
    return order


class SourceReader:
    """Campioni (chiave, file) di una sorgente per un worker, epoca dopo epoca, dalla posizione data."""

    def __init__(self, source: Source, seed: int, worker: int, n_workers: int, val_count: int,
                 epoch: int = 0, pos: int = 0, offset: int = 0):
        self.source, self.seed, self.worker, self.n_workers = source, seed, worker, n_workers
        self.val_count = val_count
        self.epoch, self.pos, self.offset = epoch, pos, offset
        self._it: Iterator | None = None
        self._path = ""
        # campioni di questo worker: letti, saltati (manifest: non ammessi o di un altro split; liste), consegnati
        self.counts: Counter = Counter()

    def state(self) -> list[int]:
        return [self.epoch, self.pos, self.offset]

    def _advance_shard(self) -> None:
        self._it, self.offset = None, 0
        self.pos += 1
        if self.pos == len(self.source.shards):
            self.pos, self.epoch = 0, self.epoch + 1

    def next(self) -> tuple[str, dict[str, bytes]]:
        empty_shards = 0  # shard attraversati senza consegnare nulla: protezione dai cicli infiniti
        while True:
            if self._it is None:
                if empty_shards > 2 * len(self.source.shards):
                    raise RuntimeError(f"{self.source.name}: nessun campione per il worker {self.worker}")
                self._path = shard_order(self.source, self.seed, self.epoch)[self.pos]
                self._it = iter_shard(self._path)
                skipped = sum(1 for _ in zip(range(self.offset), self._it))  # campioni già letti
                if skipped < self.offset:  # shard più corto della posizione salvata
                    self._advance_shard()
                    empty_shards += 1
                    continue
            try:
                key, files = next(self._it)
            except StopIteration:
                self._advance_shard()
                empty_shards += 1
                continue
            index = self.offset
            self.offset += 1
            if index % self.n_workers != self.worker:
                continue
            self.counts["letti"] += 1
            admission = self.source.admission
            if admission is None and self._path == self.source.val_shard and index < self.val_count:
                self.counts["saltati_validazione"] += 1
                continue
            if admission is not None and not admission.allows(key):  # non ammesso, o di un altro split (dev compreso)
                self.counts["saltati_manifest"] += 1
                continue
            if key in self.source.exclude:
                self.counts["saltati_lista"] += 1
                continue
            # i campioni F2 si chiamano <chiave>-s<k> o <chiave>-g<indice>: le liste per problema (decontaminazione,
            # esiti degli audit) valgono sulla chiave del problema scritta nel JSON
            if self.source.exclude and self.source.kind == "f2_solution" \
                    and json.loads(files["json"]).get("problem") in self.source.exclude:
                self.counts["saltati_lista"] += 1
                continue
            self.counts["consegnati"] += 1
            return key, apply_relabel(self.source, key, files)


class Shuffler:
    """Buffer di mescolamento: estrae a caso (hash di seed e contatore) fra `size` campioni letti."""

    def __init__(self, reader: SourceReader, size: int, salt: tuple, draws: int = 0):
        self.reader, self.size, self.salt, self.draws = reader, size, salt, draws
        self.buffer: list = []

    def next(self):
        while len(self.buffer) < self.size:
            self.buffer.append(self.reader.next())
        j = int(u01(*self.salt, self.draws) * len(self.buffer))
        self.draws += 1
        item, self.buffer[j] = self.buffer[j], self.buffer[-1]
        self.buffer.pop()
        return item


class MixedRowStream(IterableDataset):
    """Micro-batch collati (PackedCollator) da sorgenti miste; vedi la docstring del modulo."""

    def __init__(self, sources: list[Source], processor: LimiteVLProcessor, max_len: int, micro_rows: int,
                 seed: int = 0, shuffle_buffer: int = 1024, pack_buffer: int = 512, val_count: int = 512,
                 resume: dict | None = None):
        super().__init__()
        # fail-closed per ogni trainer (anche train_f1.py): le soluzioni F2 entrano nel flusso di training solo dopo la
        # verifica del gate (certificato o uncertified_reason esplicito), che train_f2.py registra in Source.certification
        unchecked = [s.name for s in sources if s.kind == "f2_solution" and not s.certification]
        if unchecked:
            raise ValueError(f"sorgenti f2_solution senza verifica del gate (f2_records.f2_data_certification): {unchecked}")
        self.sources, self.processor = sources, processor
        self.max_len, self.micro_rows, self.seed = max_len, micro_rows, seed
        self.shuffle_buffer, self.pack_buffer, self.val_count = shuffle_buffer, pack_buffer, val_count
        self.resume = resume  # {"<id worker>": stato} dal checkpoint
        self.collate = PackedCollator(max_len)
        total = sum(s.weight for s in sources)
        self._cumulative, acc = [], 0.0
        for s in sources:
            acc += s.weight / total
            self._cumulative.append(acc)

    def __iter__(self):
        info = get_worker_info()
        worker, n_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        return self.iter_worker(worker, n_workers)

    def _pick(self, r: float) -> Source:
        for source, edge in zip(self.sources, self._cumulative):
            if r < edge:
                return source
        return self.sources[-1]

    def iter_worker(self, worker: int, n_workers: int) -> Iterator[dict]:
        # Un worker senza stato salvato non aveva ancora consegnato nulla: riparte da capo senza ripetere esempi.
        state = (self.resume or {}).get(str(worker))
        if state is not None and state["n_workers"] != n_workers:
            raise ValueError(f"worker {worker}: stato salvato con {state['n_workers']} worker, ora {n_workers}")
        shufflers = {}
        for s in self.sources:
            epoch, pos, offset, draws = state["sources"][s.name] if state else (0, 0, 0, 0)
            reader = SourceReader(s, self.seed, worker, n_workers, self.val_count, epoch, pos, offset)
            shufflers[s.name] = Shuffler(reader, self.shuffle_buffer, (self.seed, "shuffle", s.name, worker), draws)
        mix_draws = state["mix_draws"] if state else 0
        stats: Counter = Counter()

        def current_state() -> dict:
            return {"n_workers": n_workers, "mix_draws": mix_draws,
                    "sources": {name: sh.reader.state() + [sh.draws] for name, sh in shufflers.items()}}

        def examples():
            nonlocal mix_draws
            while True:
                source = self._pick(u01(self.seed, "mix", worker, mix_draws))
                mix_draws += 1
                key, files = shufflers[source.name].next()
                if source.admission is not None and source.admission.check(key, files) is not None:
                    stats[f"manifest_errors/{source.name}"] += 1  # record diverso da quello ammesso: contato e saltato
                    continue
                try:
                    example = build_example(self.processor, source.kind, key, files)
                except Exception:  # noqa: BLE001 - campione illeggibile: contato e saltato
                    stats[f"errors/{source.name}"] += 1
                    continue
                stats[f"loaded/{source.name}"] += 1
                example.source = source.name
                yield example

        rows: list = []
        for row in iter_packed_rows(examples(), self.max_len, buffer_size=self.pack_buffer, stats=stats):
            rows.append(row)
            if len(rows) < self.micro_rows:
                continue
            batch = self.collate(rows)
            batch["meta"] = {"worker": worker, "state": current_state(),
                             "sources": dict(Counter(e.source for r in rows for e in r)),
                             "keys": [f"{e.source}/{e.key}" for r in rows for e in r], "stats": dict(stats),
                             "readers": {name: dict(sh.reader.counts) for name, sh in shufflers.items()}}
            rows = []
            yield batch


def validation_samples(source: Source, count: int) -> list[tuple[str, dict[str, bytes]]]:
    """I primi `count` campioni dell'ultimo shard: la parte esclusa dal training. Con un manifest, i primi `count`
    record ammessi dello split dev, nell'ordine degli shard della sorgente; un record diverso da quello ammesso ferma la
    validazione."""
    if source.admission is not None:
        picked: list[tuple[str, dict[str, bytes]]] = []
        for path in source.shards:
            for key, files in iter_shard(path):
                if len(picked) >= count:
                    return picked
                if not source.admission.validation(key) or key in source.exclude:
                    continue
                problem = source.admission.check(key, files)
                if problem:
                    raise ManifestError(f"{source.name}: {key} di validazione: {problem}")
                picked.append((key, apply_relabel(source, key, files)))
        return picked
    out = []
    for index, (key, files) in enumerate(iter_shard(source.val_shard)):
        if index >= count:
            break
        if key not in source.exclude:  # la parte di validazione resta la stessa, meno i contaminati
            out.append((key, apply_relabel(source, key, files)))
    return out
