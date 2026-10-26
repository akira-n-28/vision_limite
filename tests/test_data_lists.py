"""Liste di dati delle config (limite_vl.data_lists) e rietichettature nel flusso (stream.apply_relabel): una config che
dimentica una lista di decontaminazione obbligatoria viene rifiutata, la provenienza registra hash e righe, un'etichetta
diversa da quella attesa dalla lista fa fallire. Importa stream (torch): job SLURM."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.data_lists import REQUIRED, family, missing_required, provenance  # noqa: E402
from limite_vl.stream import Source, apply_relabel  # noqa: E402


def check(name, cond, detail=None):
    print(("ok   " if cond else "FAIL ") + name + ("" if cond or detail is None else f"  -> {detail}"))
    return cond


def main() -> int:
    ok = True
    ok &= check("famiglia dal glob", family("/x/data/shards/unimer/2343ddd9-dense/unimer-train-*.tar") == "unimer"
                and family("data/shards/synth/v3/train/*.tar") == "synth" and family("/tmp/a/*.tar") is None)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for fam, reqs in REQUIRED.items():
            for r in reqs:
                (root / r).parent.mkdir(parents=True, exist_ok=True)
                (root / r).write_text("k1\nk2\n")
        full = [str(root / r) for r in REQUIRED["unimer"]]
        spec = {"name": "unimer", "shards": str(root / "data/shards/unimer/x/*.tar"), "exclude": full}
        ok &= check("config con le liste obbligatorie: accettata", missing_required([spec], root) == [])
        spec_missing = {**spec, "exclude": full[:1]}
        errs = missing_required([spec_missing], root)
        ok &= check("config senza la lista f1eval: rifiutata", len(errs) == 1 and "f1eval" in errs[0], errs)
        ok &= check("famiglia senza obblighi: nessun errore",
                    missing_required([{"name": "r", "shards": str(root / "data/shards/replay/x/*.tar")}], root) == [])
        rel = root / "relabel.jsonl"
        rel.write_text(json.dumps({"key": "a", "label": "^{26}", "new_label": "26"}) + "\n")
        prov = provenance([{**spec, "relabel": [str(rel)]}])
        ok &= check("provenienza: righe e sha256", prov["unimer"]["exclude"][0]["lines"] == 2 and
                    len(prov["unimer"]["exclude"][0]["sha256"]) == 64 and prov["unimer"]["relabel"][0]["lines"] == 1, prov)
    src = Source("u", "unimer_formula", ["x.tar"], 1.0, relabel={"a": ("^{26}", "26")})
    files = {"json": json.dumps({"label": "^{26}", "max_patches": 64}).encode(), "png": b""}
    out = apply_relabel(src, "a", files)
    ok &= check("rietichettatura applicata", json.loads(out["json"])["label"] == "26" and out["png"] == b"")
    ok &= check("chiave senza rietichettatura: invariata", apply_relabel(src, "b", files) is files)
    try:
        apply_relabel(src, "a", {"json": json.dumps({"label": "_{26}"}).encode()})
        ok &= check("etichetta diversa da quella attesa: errore", False)
    except RuntimeError:
        ok &= check("etichetta diversa da quella attesa: errore", True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
