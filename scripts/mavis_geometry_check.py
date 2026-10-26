#!/usr/bin/env python3
"""MAVIS-Geometry: la costruzione è geometricamente coerente? Controlli deterministici su tutta la popolazione.

La coerenza fra didascalia e diagram_generation (job 30025) non dice se la figura costruita è possibile: un errore del
motore finirebbe identico in tutti e due i testi. Qui si controllano i vincoli di ogni figura con i valori dati (verità
di f1_mavis.mavis_truth), con tolleranza relativa 1e-6:
- quadrato: lati dati uguali, angoli ai vertici 90°; rombo: lati uguali;
- rettangolo (anche con un lato sostituito): lati opposti uguali, angoli 90°; il triangolo equilatero che sostituisce
  un lato ha i lati uguali al lato opposto a quello tolto;
- parallelogramma: lati opposti uguali, angoli opposti uguali, angoli consecutivi che sommano 180°;
- triangolo: somma degli angoli 180° quando sono dati tutti e tre; isoscele: lati dichiarati uguali (da
  diagram_generation: "two congruent sides dc and ce", "where hf = fi", base, angolo al vertice);
- triangolo rettangolo: ipotesi angolo retto al vertice centrale del nome (ABC: in B); con dati ridondanti si
  controllano Pitagora e le relazioni fra lati e angoli. Il report dice quante volte l'ipotesi è confermata;
- settore ABC (centro B): raggi AB e BC uguali.
Un vincolo si controlla solo quando tutti i valori che coinvolge sono dati. Report: controlli fatti, violazioni con
esempi, campioni con almeno una violazione (violations.jsonl). Job CPU: slurm/mavis_geometry_check.sbatch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_vl.f1_mavis import mavis_truth  # noqa: E402
from limite_vl.f1_schema import angle_name  # noqa: E402
from limite_vl.mavis_gen import figures  # noqa: E402
from limite_vl.provenance import code_identity, file_entry  # noqa: E402

EXAMPLES = 15
_NUM = r"\d+(?:\.\d+)?"


def value(v: str) -> float | None:
    """Valore in LaTeX di f1_schema.latex_value (9\\sqrt{3}, 10\\sqrt{3}/3, 3/2) come numero."""
    expr = re.sub(rf"({_NUM})?\\sqrt\{{({_NUM})\}}", lambda m: f"({m.group(1) or 1}*math.sqrt({m.group(2)}))", v)
    if not re.fullmatch(rf"[\d.()*/+\- ]*(?:math\.sqrt\({_NUM}\)[\d.()*/+\- ]*)*", expr):
        return None
    try:
        return float(eval(expr, {"math": math, "__builtins__": {}}))  # noqa: S307 - solo cifre, operatori e sqrt
    except Exception:  # noqa: BLE001
        return None


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


class Checker:
    def __init__(self, truth: dict, generation: str):
        self.sides = {k: value(v) for k, v in truth["sides"].items()}
        self.angles = {k: value(v) for k, v in truth["angles"].items()}
        self.truth, self.generation = truth, generation
        self.results: list[tuple[str, bool, str]] = []

    def side(self, a: str, b: str) -> float | None:
        return self.sides.get("".join(sorted(a + b)))

    def angle(self, end1: str, vertex: str, end2: str) -> float | None:
        return self.angles.get(angle_name(vertex, end1 + end2))

    def record(self, check: str, ok: bool, detail: str) -> None:
        self.results.append((check, ok, detail))

    def equal(self, check: str, values: list[tuple[str, float | None]]) -> None:
        known = [(n, v) for n, v in values if v is not None]
        if len(known) >= 2:
            ok = all(close(known[0][1], v) for _, v in known[1:])
            self.record(check, ok, ", ".join(f"{n}={v:g}" for n, v in known))

    def polygon(self, name: str, kind: str, shape: dict) -> None:
        n = len(name)
        sides = [(name[i] + name[(i + 1) % n], self.side(name[i], name[(i + 1) % n])) for i in range(n)]
        angles = [(name[i - 1] + name[i] + name[(i + 1) % n], self.angle(name[i - 1], name[i], name[(i + 1) % n]))
                  for i in range(n)]
        if kind in ("square", "rhombus"):
            self.equal(f"{kind}_lati_uguali", sides)
        if kind in ("square", "rectangle"):
            for a, v in angles:
                if v is not None:
                    self.record(f"{kind}_angolo_retto", close(v, 90), f"{a}={v:g}")
        if n == 4 and kind in ("square", "rectangle", "parallelogram", "rhombus"):
            self.equal(f"{kind}_lati_opposti", [sides[0], sides[2]])
            self.equal(f"{kind}_lati_opposti", [sides[1], sides[3]])
        if n == 4 and kind == "parallelogram":
            self.equal("parallelogram_angoli_opposti", [angles[0], angles[2]])
            self.equal("parallelogram_angoli_opposti", [angles[1], angles[3]])
            for (a1, v1), (a2, v2) in zip(angles, angles[1:] + angles[:1]):
                if v1 is not None and v2 is not None:
                    self.record("parallelogram_angoli_consecutivi", close(v1 + v2, 180), f"{a1}+{a2}={v1 + v2:g}")
        if shape.get("modification") == "equilateral triangle":
            r = shape["removed"]
            opposite = [s for s in sides if not set(s[0]) & set(r)]
            t = shape["triangle"]
            self.equal("triangolo_sostitutivo_equilatero",
                       opposite[:1] + [(t[0] + t[1], self.side(t[0], t[1])), (t[1] + t[2], self.side(t[1], t[2]))])

    def triangle(self, name: str, kind: str, sentence: str) -> None:
        a, b, c = name
        angs = [self.angle(c, a, b), self.angle(a, b, c), self.angle(b, c, a)]
        if all(v is not None for v in angs):
            self.record("triangolo_somma_angoli", close(sum(angs), 180), f"{name}: {sum(angs):g}")
        if kind == "equilateral triangle":
            self.equal("equilatero_lati", [(a + b, self.side(a, b)), (b + c, self.side(b, c)), (a + c, self.side(a, c))])
        if kind == "isosceles triangle":
            low = sentence.lower()
            m = re.search(r"sides ([a-z]{2}) and ([a-z]{2})|where ([a-z]{2}) = ([a-z]{2})", low)
            pair = None
            if m:
                pair = [x.upper() for x in m.groups() if x]
            elif (mb := re.search(r"with ([a-z]{2}) as its base", low)):
                base = set(mb.group(1).upper())
                apex = (set(name) - base).pop() if len(set(name) - base) == 1 else None
                pair = [apex + x for x in base] if apex else None
            elif (mv := re.search(r"with angle ([a-z]{3}) as its vertex angle", low)):
                v = mv.group(1).upper()
                pair = [v[1] + v[0], v[1] + v[2]]
            if pair:
                self.equal("isoscele_lati_uguali", [(p, self.side(p[0], p[1])) for p in pair])
        if kind == "right triangle":
            right = self.angle(a, b, c)
            if right is not None:
                self.record("rettangolo_angolo_al_vertice_centrale", close(right, 90), f"{name}: angolo in {b}={right:g}")
            leg1, leg2, hyp = self.side(a, b), self.side(b, c), self.side(a, c)
            if None not in (leg1, leg2, hyp):
                self.record("rettangolo_pitagora", close(leg1 ** 2 + leg2 ** 2, hyp ** 2), f"{name}")
            for vertex, opp, adj in ((a, leg2, leg1), (c, leg1, leg2)):
                ang = self.angle(b, vertex, c if vertex == a else a)
                if ang is None:
                    continue
                t = math.radians(ang)
                if opp is not None and adj is not None:
                    self.record("rettangolo_tangente", close(math.tan(t), opp / adj), f"{name}: angolo in {vertex}={ang:g}")
                if opp is not None and hyp is not None:
                    self.record("rettangolo_seno", close(math.sin(t), opp / hyp), f"{name}: angolo in {vertex}={ang:g}")
                if adj is not None and hyp is not None:
                    self.record("rettangolo_coseno", close(math.cos(t), adj / hyp), f"{name}: angolo in {vertex}={ang:g}")

    def run(self) -> list[tuple[str, bool, str]]:
        sentences = {f.name.upper(): f.sentence for f in figures(self.generation)[0]}
        for s in self.truth["shapes"]:
            name, kind = s["name"], s["kind"]
            if kind == "sector":
                self.equal("settore_raggi", [(name[0] + name[1], self.side(name[0], name[1])),
                                             (name[1] + name[2], self.side(name[1], name[2]))])
            elif len(name) == 3 and kind.endswith("triangle"):
                self.triangle(name, kind, sentences.get(name, ""))
            elif len(name) >= 4:
                self.polygon(name, kind, s)
        return self.results


def iter_json(path: Path):
    with tarfile.open(path, "r|") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".json"):
                yield json.loads(tar.extractfile(member).read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("mavis_geometry_check.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} esiste e non è vuota")
    args.out.mkdir(parents=True, exist_ok=True)
    n = bad_samples = no_truth = unparsed_values = 0
    checks, violations, examples = Counter(), Counter(), defaultdict(list)
    by_depth = defaultdict(Counter)
    with open(args.out / "violations.jsonl", "w") as fh:
        for shard in args.shards:
            for meta in iter_json(shard):
                n += 1
                te, depth = meta["text_en"], meta.get("depth")
                truth, _ = mavis_truth(te["caption"], te["diagram_generation"])
                if truth is None:
                    no_truth += 1
                    continue
                chk = Checker(truth, te["diagram_generation"])
                unparsed_values += sum(v is None for v in list(chk.sides.values()) + list(chk.angles.values()))
                results = chk.run()
                bad = [(c, d) for c, ok, d in results if not ok]
                checks.update(c for c, _, _ in results)
                violations.update(c for c, _ in bad)
                by_depth[depth]["campioni"] += 1
                by_depth[depth]["controllati"] += bool(results)
                if bad:
                    bad_samples += 1
                    by_depth[depth]["con_violazioni"] += 1
                    fh.write(json.dumps({"id": meta["id"], "depth": depth, "violations": bad}, ensure_ascii=False) + "\n")
                    for c, d in bad:
                        if len(examples[c]) < EXAMPLES:
                            examples[c].append({"id": meta["id"], "detail": d, "generation": te["diagram_generation"]})
            print(f"{shard.name}: {n} campioni, {bad_samples} con violazioni", flush=True)
    report = {
        "schema": "mavis-geometry-check/1",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "code": code_identity(),
        "shards": [file_entry(s) for s in args.shards],
        "samples": n, "without_truth": no_truth, "values_not_numeric": unparsed_values,
        "samples_with_violations": bad_samples,
        "checks": dict(checks.most_common()), "violations": dict(violations.most_common()),
        "by_depth": {str(k): dict(v) for k, v in sorted(by_depth.items(), key=lambda kv: str(kv[0]))},
        "examples": examples,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("samples", "without_truth", "values_not_numeric", "samples_with_violations",
                                             "checks", "violations", "by_depth")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
