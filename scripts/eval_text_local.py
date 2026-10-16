#!/usr/bin/env python3
"""limite-eval su un decoder esportato dal progetto (scripts/export_decoder.py), con il valutatore invariato.

Il valutatore (third_party/limite-violetto/limite-evals) ammette solo i checkpoint in elenco (limite_models.py):
una cartella locale passa se il nome del repo e la revisione, letti dalla config o da un percorso in stile cache
Hugging Face (`models--ORG--NOME/snapshots/<40 hex>`), corrispondono a un contratto. Scelta dell'utente
(2026-10-16, opzione 1): questo script aggiunge, solo nel processo, un contratto con il nome vero del checkpoint
(`limite-vl/NOME`) e lo stesso profilo di Violetto (`chat`, stadio `posttrain`), poi chiama limite-eval. Rendering,
campionamento, budget e correzione restano quelli della baseline; il report registra il nome e lo SHA-256 dei pesi.
Niente viene modificato in third_party né nella cartella esportata.

La revisione è l'inizio dello SHA-256 di model.safetensors scritto da export_decoder.py in export.json; il percorso
in stile cache è un link in --links che punta alla cartella esportata.

Da eseguire con l'interprete del client del valutatore (slurm/eval_text_local.sbatch):

    $VENVS_DIR/evals-current/bin/python -I scripts/eval_text_local.py --checkpoint DIR --name NOME --links DIR \
        -- <argomenti di limite-eval, senza il checkpoint>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NAME = re.compile(r"^[a-z0-9][a-z0-9._]*(-[a-z0-9._]+)*$")  # niente "--": separa ORG e NOME nella cache
ORG = "limite-vl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="cartella scritta da export_decoder.py")
    ap.add_argument("--name", required=True, help="nome del checkpoint, es. f2-abl-normuon-step540")
    ap.add_argument("--links", required=True, help="cartella dei link in stile cache Hugging Face")
    ap.add_argument("--resolve-only", action="store_true",
                    help="solo il riconoscimento del checkpoint da parte del valutatore (job CPU, senza engine)")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- e gli argomenti di limite-eval")
    args = ap.parse_args()
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest

    if not NAME.fullmatch(args.name):
        sys.exit(f"nome non valido: {args.name!r} (minuscole, cifre, . _ -, niente '--')")
    checkpoint = Path(args.checkpoint).resolve()
    export = json.loads((checkpoint / "export.json").read_text())
    if export.get("weights") == "roundtrip":
        print("checkpoint di prova: Violetto esportato senza modifiche (roundtrip)", flush=True)
    revision = export["sha256"]["model.safetensors"][:40]
    repo_id = f"{ORG}/{args.name}"

    link = Path(args.links).resolve() / f"models--{ORG}--{args.name}" / "snapshots" / revision
    if link.is_symlink():
        if link.resolve() != checkpoint:
            sys.exit(f"{link} punta a {link.resolve()}, non a {checkpoint}")
    else:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(checkpoint, target_is_directory=True)

    from limite_evals import limite_models

    if repo_id.casefold() in limite_models.BY_REPO:
        sys.exit(f"{repo_id} è già nell'elenco del valutatore")
    contract = limite_models.LimiteModel(name=f"{ORG}-{args.name}", repo_id=repo_id, revision=revision,
                                         profile="chat", stage="posttrain")
    limite_models.BY_REPO[repo_id.casefold()] = contract
    print(f"contratto aggiunto nel processo: {contract.identity} (profilo chat, stadio posttrain) -> {checkpoint}",
          flush=True)

    if args.resolve_only:
        from limite_evals.resolve import resolve

        resolved = resolve(str(link))
        print(f"riconosciuto: formato {resolved.source_format}, stadio {resolved.stage}, profilo {resolved.profile}, "
              f"revisione {resolved.revision}, contesto {resolved.max_model_len}, sha256 {resolved.sha256[:12]}", flush=True)
        return 0

    from limite_evals.cli import main as eval_main

    return eval_main([str(link), *rest])


if __name__ == "__main__":
    sys.exit(main())
