"""Codice Transformers di Limite (modeling_limite.py e moduli collegati) dalla copia locale verificata.

Il training non usa trust_remote_code: i file remoti dello snapshot fissato sono copiati in
third_party/limite-modeling/ da scripts/vendor_limite_modeling.py e qui si importano come pacchetto
`limite_modeling` solo se lo SHA-256 di ogni file coincide con quello fissato sotto.

Questo modulo usa solo la libreria standard finché non si chiama load(): il vendoring gira sul
frontend.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO = "paradigma-inc/limite-1b-violetto"
SOURCE_REVISION = "b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
VENDOR_DIR = ROOT / "third_party" / "limite-modeling"
PACKAGE = "limite_modeling"
# SHA-256 dei file allo snapshot SOURCE_REVISION (verificati sul blob della cache HF).
PINNED_SHA256 = {
    "configuration_limite.py": "69f101ad280b14b3ab9e9b0076685d022191cc5abcf4981666867c0249522ddb",
    "contract.py": "253d7511b4c89f74eb7f4919bf466a841844a4699079d148deb80f8e4ed61fe6",
    "modeling_limite.py": "8d44f9e8f7e0c7d811ecab68e9e93abda197cba01f48514935d917c30953d0ea",
    "registration.py": "2c9bbd09280a57b68064b5853ce6208c37d0342a299abfd149764aa481cad8ca",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory: Path = VENDOR_DIR) -> None:
    errors = []
    for name, expected in PINNED_SHA256.items():
        path = directory / name
        if not path.is_file():
            errors.append(f"{path}: assente (scripts/vendor_limite_modeling.py)")
        elif (actual := sha256(path)) != expected:
            errors.append(f"{path}: sha256 {actual[:12]}, atteso {expected[:12]}")
    if errors:
        raise RuntimeError("codice di Limite non verificato:\n" + "\n".join(errors))


def load(directory: Path = VENDOR_DIR) -> types.ModuleType:
    """Restituisce modeling_limite importato dalla copia verificata (pacchetto `limite_modeling`)."""
    if f"{PACKAGE}.modeling_limite" in sys.modules:
        return sys.modules[f"{PACKAGE}.modeling_limite"]
    verify(directory)
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(directory)]
    sys.modules[PACKAGE] = package
    return importlib.import_module(f"{PACKAGE}.modeling_limite")
