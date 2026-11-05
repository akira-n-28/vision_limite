#!/usr/bin/env python3
"""Test CPU di src/limite_vl/estimate.py (C8; docs/specs/S4_CAMPIONAMENTO_BUDGET.md §§3, 8, 9) su dati fittizi, con
calcoli esatti e senza simulazioni. Solo in un job (slurm/tests_cpu.sbatch).

Coprono:
- limite ipergeometrico uguale all'enumerazione completa dei sottoinsiemi su popolazioni piccole (N <= 9, ogni n, k e D,
  tre livelli) e copertura esatta >= 1 - alpha per ogni D (N <= 12);
- casi limite: k = n, censimento, N = n = 1, k = 0 contro la scansione diretta, valori di S2 §2.8, monotonia,
  convergenza a Clopper-Pearson per N grande;
- Clopper-Pearson: k = n, k = 0 contro 1 - alpha^(1/n), probabilità esatta al limite (per eccesso, entro 2^-40),
  confronto con stem_audit.clopper_pearson (storico, in sola lettura), copertura esatta su una griglia di p;
- input non validi;
- controesempio di S4 §4 con copertura esatta: col campione di gruppi clustered_rate (storico, in sola lettura) copre la
  quota di QA nel 30% dei casi, la stima della quota di QA da quel campione è rifiutata, il disegno a QA uniformi
  copre almeno al 95%, e dallo stesso campione di gruppi si stima la quota di gruppi;
- Bonferroni: senza correzione la copertura simultanea di due strati costruiti apposta scende sotto 0.95, con alpha/2
  resta sopra per ogni D, e il limite dell'unione copre almeno quanto la coppia;
- compatibilità fra disegno, quantità e popolazione; esiti contati senza perdite; tre livelli del risultato;
- l'esempio minimo dei documenti, con i suoi numeri."""
from __future__ import annotations

import itertools
import math
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import estimate as est  # noqa: E402
from limite_vl import stem_audit  # noqa: E402  (storico congelato: solo per i confronti di S4 §4)

FAILS = []
ALPHAS = (Fraction(1, 20), Fraction(1, 5), Fraction(1, 2))


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, *args, kind=ValueError, **kw) -> bool:
    try:
        fn(*args, **kw)
    except kind:
        return True
    return False


def refusal(fn, *args) -> str:
    """Il motivo del rifiuto, vuoto se il calcolo non è rifiutato."""
    try:
        fn(*args)
    except est.Refused as exc:
        return str(exc) or "rifiutato"
    return ""


def subset_counts(N: int, n: int, D: int) -> list[int]:
    """Numero di n-sottoinsiemi con x unità errate, x = 0..n, enumerandoli tutti (le prime D unità sono errate)."""
    counts = [0] * (n + 1)
    for s in itertools.combinations(range(N), n):
        counts[sum(1 for i in s if i < D)] += 1
    return counts


def pmf_numerators(N: int, n: int, D: int) -> list[int]:
    return [math.comb(D, x) * math.comb(N - D, n - x) for x in range(n + 1)]


def misses(N: int, n: int, a: Fraction) -> list[Fraction]:
    """Probabilità esatta che il limite al livello a non copra, per ogni D da 0 a N."""
    total = math.comb(N, n)
    bounds = [est.hypergeometric_upper(N, n, x, a) for x in range(n + 1)]
    return [Fraction(sum(p for x, p in enumerate(pmf_numerators(N, n, D)) if bounds[x] < D), total)
            for D in range(N + 1)]


def binom_cdf_exact(n: int, k: int, p: Fraction) -> Fraction:
    return sum(math.comb(n, x) * p ** x * (1 - p) ** (n - x) for x in range(k + 1))


# ---------------------------------------------------------------- ipergeometrico

def test_enumeration():
    bad_cdf, bad_bound, cases = [], [], 0
    for N in range(1, 10):
        for n in range(1, N + 1):
            total = math.comb(N, n)
            tables = [subset_counts(N, n, D) for D in range(N + 1)]
            for D, t in enumerate(tables):
                cum = list(itertools.accumulate(t))
                bad_cdf += [(N, D, n, k) for k in range(n + 1)
                            if est.hypergeometric_cdf(N, D, n, k) != Fraction(cum[k], total)]
            for a in ALPHAS:
                for k in range(n + 1):
                    brute = max(D for D in range(N + 1) if Fraction(sum(tables[D][:k + 1]), total) > a)
                    cases += 1
                    if est.hypergeometric_upper(N, n, k, a) != brute:
                        bad_bound.append((N, n, k, str(a)))
    check("ipergeometrico: P(X <= k) esatta uguale all'enumerazione dei sottoinsiemi (N <= 9)", not bad_cdf,
          str(bad_cdf[:5]))
    check(f"ipergeometrico: D_U uguale all'enumerazione completa di tutti i D, {cases} casi (N <= 9, alpha 1/20, 1/5, "
          "1/2)", not bad_bound, str(bad_bound[:5]))


def test_coverage_small():
    worst = {}
    for a in ALPHAS[:2]:
        low = Fraction(1)
        for N in range(1, 13):
            for n in range(1, N + 1):
                low = min(low, 1 - max(misses(N, n, a)))
        worst[a] = low
    check("ipergeometrico: copertura esatta >= 1 - alpha per ogni N <= 12, n e D (alpha 1/20 e 1/5)",
          all(w >= 1 - a for a, w in worst.items()), {str(a): round(float(w), 4) for a, w in worst.items()})


def test_special_cases():
    check("ipergeometrico: k = n dà D_U = N, limite 1",
          all(est.hypergeometric_upper(N, n, n) == N for N, n in ((1, 1), (10, 3), (1000, 300), (45061, 75))))
    check("ipergeometrico: censimento n = N dà D_U = k, la quota esatta; con k = 0 dà 0",
          all(est.hypergeometric_upper(N, N, k) == k for N, k in ((1, 0), (7, 0), (7, 3), (7, 7), (500, 12))))
    check("ipergeometrico: N = n = 1 con k = 0 e k = 1 dà 0 e 1",
          est.hypergeometric_upper(1, 1, 0) == 0 and est.hypergeometric_upper(1, 1, 1) == 1)
    N, n = 300, 30
    scan = max(D for D in range(N - n + 1) if Fraction(math.comb(N - D, n), math.comb(N, n)) > est.ALPHA)
    check("ipergeometrico: k = 0 uguale alla scansione di tutti i D con P(X = 0) = C(N - D, n) / C(N, n)",
          est.hypergeometric_upper(N, n, 0) == scan, f"D_U = {scan}")
    for N, n, want in ((45061, 75, 0.039), (4038, 50, 0.058), (300, 30, 0.090)):
        got = est.hypergeometric_upper(N, n, 0) / N
        check(f"ipergeometrico: con 0 errori su {n} di {N} circa {want:.1%}, come S2 §2.8", abs(got - want) < 0.0015,
              f"{got:.4f}")
    by_k = [est.hypergeometric_upper(1000, 100, k) for k in range(101)]
    by_n = [est.hypergeometric_upper(1000, n, 2) for n in range(10, 301, 10)]
    check("ipergeometrico: il limite cresce con k e cala con n a k fisso",
          by_k == sorted(by_k) and by_n == sorted(by_n, reverse=True))
    cp = {k: est.clopper_pearson_upper(300, k) for k in (0, 2, 5)}
    rel = [max(abs(est.hypergeometric_upper(N, 300, k) / N - cp[k]) / cp[k] for k in cp) for N in (10 ** 5, 10 ** 6,
                                                                                                   10 ** 7)]
    check("ipergeometrico: per N grande tende a Clopper-Pearson (n = 300, k = 0, 2, 5)",
          rel[0] > rel[1] > rel[2] and rel[2] < 1e-3, ", ".join(f"{x:.1e}" for x in rel))


# ---------------------------------------------------------------- Clopper-Pearson

def test_clopper_pearson():
    check("Clopper-Pearson: k = n dà 1", all(est.clopper_pearson_upper(n, n) == 1.0 for n in (1, 2, 30, 300)))
    closed = {n: -math.expm1(math.log(0.05) / n) for n in (1, 2, 10, 300, 3000)}
    got = {n: est.clopper_pearson_upper(n, 0) for n in closed}
    check("Clopper-Pearson: k = 0 dà 1 - alpha^(1/n)", all(abs(got[n] - closed[n]) <= 1e-12 * closed[n] for n in closed),
          ", ".join(f"{n}: {got[n]:.6g}" for n in closed))
    check("Clopper-Pearson: 0 errori su 300 danno circa l'1.0%, come S4 §3", abs(got[300] - 0.00994) < 5e-5,
          f"{got[300]:.5f}")
    eps, bad = Fraction(1, 2 ** 40), []
    for n in range(1, 31):
        for k in range(n):
            p = Fraction(est.clopper_pearson_upper(n, k))
            if not binom_cdf_exact(n, k, p) <= est.ALPHA < binom_cdf_exact(n, k, p - eps):
                bad.append((n, k))
    check("Clopper-Pearson: al limite P(X <= k) esatta <= alpha, poco sotto (2^-40) > alpha, per n <= 30", not bad,
          str(bad[:5]))
    pairs = sorted({(n, k) for n in (1, 5, 30, 300) for k in (0, 1, n // 2, n - 1) if k < n})
    diff = max(abs(est.clopper_pearson_upper(n, k) - stem_audit.clopper_pearson(k, n, conf=0.90)[1]) for n, k in pairs)
    check("Clopper-Pearson: uguale a stem_audit.clopper_pearson (storico, bilaterale al 90%) entro 1e-9", diff < 1e-9,
          f"{diff:.1e}")
    low = Fraction(1)
    for n in (1, 2, 3, 5, 10, 25):
        bounds = [Fraction(est.clopper_pearson_upper(n, x)) for x in range(n + 1)]
        for j in range(1, 100):
            p = Fraction(j, 100)
            low = min(low, sum(math.comb(n, x) * p ** x * (1 - p) ** (n - x) for x in range(n + 1) if bounds[x] >= p))
    check("Clopper-Pearson: copertura esatta >= 0.95 per p = 1/100 ... 99/100 e n fino a 25", low >= 1 - est.ALPHA,
          f"{float(low):.4f}")


# ---------------------------------------------------------------- input non validi

def test_invalid():
    bad_hyper = [(10, 0, 0), (10, 11, 0), (10, 5, -1), (10, 5, 6), (10.0, 5, 0), (10, 5, 1.0), (True, 1, 0), (0, 0, 0),
                 ("10", 5, 0)]
    check("ipergeometrico: N, n, k non interi, n < 1, n > N, k < 0, k > n danno errore",
          all(raises(est.hypergeometric_upper, *x) for x in bad_hyper))
    bad_alpha = [0, 1, -0.1, 1.5, float("nan"), float("inf"), "x", None, True, Fraction(1)]
    check("alpha fuori da (0, 1) o non numerico dà errore",
          all(raises(est.hypergeometric_upper, 10, 5, 0, a) and raises(est.clopper_pearson_upper, 10, 0, a)
              for a in bad_alpha))
    check("alpha 0.05, '1/20' e Fraction(1, 20) danno lo stesso limite",
          len({est.hypergeometric_upper(500, 40, 1, a) for a in (0.05, "1/20", Fraction(1, 20))}) == 1
          and len({est.clopper_pearson_upper(40, 1, a) for a in (0.05, "0.05", Fraction(1, 20))}) == 1)
    check("Clopper-Pearson: n < 1, k < 0, k > n o non interi danno errore",
          all(raises(est.clopper_pearson_upper, *x) for x in ((0, 0), (5, -1), (5, 6), (5.0, 1), (5, True))))
    check("Bonferroni: 0.05 su 2 affermazioni dà 1/40; un numero di affermazioni non valido dà errore",
          est.bonferroni(0.05, 2) == Fraction(1, 40) and raises(est.bonferroni, est.ALPHA, 0)
          and raises(est.bonferroni, est.ALPHA, 1.5))
    invalid = [lambda: est.Counts(approvato=-1), lambda: est.Counts(errato=1.5), lambda: est.Counts(approvato=True),
               lambda: est.Design("p", "s", "riga", est.UNIFORM, 10, 5),
               lambda: est.Design("p", "s", "qa", "casuale", 10, 5),
               lambda: est.Design("p", "s", "qa", est.UNIFORM, 10, 11),
               lambda: est.Design("p", "s", "qa", est.UNIFORM, None, 5),
               lambda: est.Design("", "s", "qa", est.UNIFORM, 10, 5),
               lambda: est.Claim("correttezza", "quota_qa_errate", "p", "s", 0),
               lambda: est.Claim("", "quota_qa_errate", "p", "s", 10),
               lambda: est.describe(est.Counts())]
    check("conteggi, disegno e affermazione non validi danno errore", all(raises(fn) for fn in invalid),
          str([i for i, fn in enumerate(invalid) if not raises(fn)]))


# ---------------------------------------------------------------- controesempio di S4 §4

def test_counterexample():
    groups, bad_size, n = 1000, 1000, 300  # 999 gruppi con una QA corretta, uno con 1.000 QA errate
    N_qa, D_qa = groups - 1 + bad_size, bad_size
    rate = Fraction(D_qa, N_qa)
    p_in = Fraction(math.comb(groups - 1, n - 1), math.comb(groups, n))  # il gruppo errato è estratto
    without = stem_audit.clustered_rate([(0, 1)] * n, conf=0.90)
    with_bad = stem_audit.clustered_rate([(0, 1)] * (n - 1) + [(bad_size, bad_size)], conf=0.90)
    two_sided = stem_audit.clustered_rate([(0, 1)] * n)
    cov = p_in * (with_bad["ci"][1] >= rate) + (1 - p_in) * (without["ci"][1] >= rate)
    check("controesempio di S4 §4: col campione di 300 gruppi clustered_rate copre la quota di QA errate (50.03%) nel "
          "30% dei casi, calcolo esatto",
          p_in == Fraction(3, 10) and cov == Fraction(3, 10) and abs(without["ci"][1] - 0.0099) < 1e-4
          and abs(two_sided["ci"][1] - 0.0122) < 1e-4,
          f"copertura {float(cov)}; senza il gruppo errato limite {without['ci'][1]:.4f}, bilaterale "
          f"{two_sided['ci'][1]:.4f}, deff {without['deff']}")
    pop = "controesempio-S4"
    by_groups = est.Design(pop, "unico", "gruppo", est.UNIFORM, groups, n)
    why = refusal(est.estimate, est.Claim("correttezza", "quota_qa_errate", pop, "unico", N_qa), by_groups,
                  est.Counts(approvato=n))
    check("controesempio: la stima della quota di QA da un campione di gruppi è rifiutata", "S4 §4" in why, why[:110])
    bounds = [est.hypergeometric_upper(N_qa, n, x) for x in range(n + 1)]
    cov_qa = Fraction(sum(p for x, p in enumerate(pmf_numerators(N_qa, n, D_qa)) if bounds[x] >= D_qa),
                      math.comb(N_qa, n))
    p0 = Fraction(math.comb(N_qa - D_qa, n), math.comb(N_qa, n))
    check("controesempio: 300 QA estratte uniformemente fra le 1.999, copertura esatta del limite ipergeometrico >= "
          "0.95", cov_qa >= 1 - est.ALPHA, f"{float(cov_qa):.6f}")
    check("controesempio: le 300 QA sono tutte corrette con probabilità 1.2·10^-102, come S4 §4",
          f"{float(p0):.1e}" == "1.2e-102", f"{float(p0):.3e}")
    g = [est.hypergeometric_upper(groups, n, x) for x in (0, 1)]
    cov_g = (1 - p_in) * (g[0] >= 1) + p_in * (g[1] >= 1)
    r = est.estimate(est.Claim("correttezza", "quota_gruppi_con_errore", pop, "unico", groups), by_groups,
                     est.Counts(approvato=n))
    check("controesempio: dallo stesso campione di gruppi si stima la quota di gruppi con errore, con copertura esatta 1",
          cov_g == 1 and r["limite"]["D_U"] == g[0], f"limite {r['limite']['valore']:.4f} sui gruppi, non sulle QA")


# ---------------------------------------------------------------- Bonferroni

def test_bonferroni():
    strata = {"A": (400, 40), "B": (300, 30)}  # popolazioni costruite apposta, campioni indipendenti
    a, half = est.ALPHA, est.bonferroni(est.ALPHA, 2)
    plain = {s: misses(N, n, a) for s, (N, n) in strata.items()}
    fixed = {s: misses(N, n, half) for s, (N, n) in strata.items()}
    worst = {s: max(range(len(m)), key=m.__getitem__) for s, m in plain.items()}
    joint_plain = math.prod(1 - plain[s][worst[s]] for s in strata)
    joint_fixed = math.prod(1 - max(fixed[s]) for s in strata)
    check("Bonferroni: senza correzione, due strati al 95% ciascuno coprono insieme meno di 0.95 per i D peggiori",
          joint_plain < 1 - a, f"{float(joint_plain):.4f}; D peggiori {worst}, mancata copertura "
          f"{[round(float(plain[s][worst[s]]), 4) for s in strata]}")
    check("Bonferroni: con alpha/2 per strato la copertura simultanea esatta è almeno 0.95 per ogni coppia di D",
          joint_fixed >= 1 - a, f"{float(joint_fixed):.4f}")
    (NA, nA), (NB, nB) = strata["A"], strata["B"]
    DA, DB = (max(range(len(fixed[s])), key=fixed[s].__getitem__) for s in strata)
    bA = [est.hypergeometric_upper(NA, nA, x, half) for x in range(nA + 1)]
    bB = [est.hypergeometric_upper(NB, nB, x, half) for x in range(nB + 1)]
    pA, pB = pmf_numerators(NA, nA, DA), pmf_numerators(NB, nB, DB)
    total = math.comb(NA, nA) * math.comb(NB, nB)
    union = Fraction(sum(pA[x] * pB[y] for x in range(nA + 1) for y in range(nB + 1) if bA[x] + bB[y] >= DA + DB), total)
    pair = Fraction(sum(pA[x] for x in range(nA + 1) if bA[x] >= DA) * sum(pB[y] for y in range(nB + 1) if bB[y] >= DB),
                    total)
    check("Bonferroni: il limite dell'unione (somma dei D_U a alpha/2) copre almeno quanto la coppia, e almeno 0.95",
          union >= pair >= 1 - a, f"unione {float(union):.4f}, coppia {float(pair):.4f}, D = {DA}, {DB}")
    pop = "esempio-bonferroni"
    items = [(est.Claim("correttezza", "quota_qa_errate", pop, "A", NA), est.Design(pop, "A", "qa", est.UNIFORM, NA, nA),
              est.Counts(approvato=nA - 1, errato=1)),
             (est.Claim("correttezza", "quota_coppie_con_errore", pop, "B", NB),
              est.Design(pop, "B", "coppia", est.UNIFORM, NB, nB), est.Counts(approvato=nB))]
    res = est.estimate_simultaneous(items)
    check("Bonferroni: due affermazioni simultanee a 1/40 ciascuna, con risultati separati per strato",
          [r["limite"]["alpha"] for r in res] == ["1/40", "1/40"] and [r["disegno"]["strato"] for r in res] == ["A", "B"]
          and [r["limite"]["D_U"] for r in res] == [est.hypergeometric_upper(NA, nA, 1, half),
                                                    est.hypergeometric_upper(NB, nB, 0, half)]
          and all(r["limite"]["alpha_famiglia"] == "1/20" for r in res))
    check("Bonferroni: la stessa affermazione ripetuta nella famiglia è rifiutata",
          bool(refusal(est.estimate_simultaneous, [items[0], items[0]])))


# ---------------------------------------------------------------- disegno, esiti e risultato

def test_design_checks():
    pop, N, n = "fittizia", 2000, 100
    d = est.Design(pop, "S", "qa", est.UNIFORM, N, n)
    c = est.Claim("correttezza", "quota_qa_errate", pop, "S", N)
    ok = est.Counts(approvato=n)
    why = {
        "QA da un campione di coppie": refusal(est.estimate, c, est.Design(pop, "S", "coppia", est.UNIFORM, N, n), ok),
        "quota di QA da estrazioni proporzionali ai token": refusal(est.estimate, c,
                                                                    est.Design(pop, "S", "qa", est.PPS, N, n), ok),
        "target-token da QA uniformi": refusal(est.estimate, est.Claim("correttezza", "quota_target_token_errati", pop,
                                                                       "S", N), d, ok),
        "altra popolazione": refusal(est.estimate, est.Claim("correttezza", "quota_qa_errate", "altra", "S", N), d, ok),
        "altro strato": refusal(est.estimate, est.Claim("correttezza", "quota_qa_errate", pop, "T", N), d, ok),
        "altro N": refusal(est.estimate, est.Claim("correttezza", "quota_qa_errate", pop, "S", N + 1), d, ok),
        "quantità sconosciuta": refusal(est.estimate, est.Claim("correttezza", "quota_errori", pop, "S", N), d, ok),
        "casi mirati": refusal(est.estimate, c, est.Design(pop, "S", "qa", est.TARGETED, None, 20),
                               est.Counts(approvato=20)),
        "esiti meno delle estrazioni": refusal(est.estimate, c, d, est.Counts(approvato=n - 3)),
    }
    check("compatibilità: unità, estrazione, popolazione, strato, N, quantità, casi mirati e numerosità rifiutati",
          all(why.values()), str([k for k, v in why.items() if not v]))
    desc = est.describe(est.Counts(approvato=17, errato=2, incerto=1))
    check("casi mirati: rifiutati con il motivo, e restano solo i conteggi descrittivi",
          "casi mirati" in why["casi mirati"] and desc["tipo"] == "descrittivo" and desc["n"] == 20
          and desc["quota_non_risolta"] == "1/20", why["casi mirati"][:80])
    draws = [f"u{i}" for i in range(n)]
    outcomes = {u: "approvato" for u in draws}
    wrong = {
        "unità senza esito": (d, draws, {u: v for u, v in outcomes.items() if u != "u7"}),
        "esito di un'unità non estratta": (d, draws, {**outcomes, "u999": "approvato"}),
        "esito ripetuto": (d, draws, list(outcomes.items()) + [("u3", "errato")]),
        "esito sconosciuto": (d, draws, {**outcomes, "u4": "ok"}),
        "estrazioni diverse da n": (d, draws[:-1], {u: "approvato" for u in draws[:-1]}),
        "unità ripetuta senza reinserimento": (d, draws[:-1] + ["u0"], {u: "approvato" for u in draws[:-1]}),
    }
    missed = [k for k, args in wrong.items() if not refusal(est.tally, *args)]
    check("esiti: nessuna unità persa o aggiunta in silenzio (senza esito, non estratta, ripetuta, sconosciuta, numero "
          "diverso)", not missed, str(missed))
    check("esiti: con tutti gli esiti, il conteggio torna", est.tally(d, draws, outcomes) == ok)
    pps = est.Design(pop, "S", "qa", est.PPS, N, 4)
    t = est.tally(pps, ["u1", "u2", "u1", "u3"], {"u1": "errato", "u2": "approvato", "u3": "mancante"})
    check("esiti con reinserimento: un'unità estratta due volte conta due volte",
          t == est.Counts(approvato=1, errato=2, mancante=1), str(t))
    mixed = est.Counts(approvato=n - 6, errato=2, incerto=1, non_giudicabile=1, mancante=1, approvato_non_indipendente=1)
    r = est.estimate(c, d, mixed)
    check("esiti: incerti, non giudicabili, mancanti e approvazioni non indipendenti contano come errori nel limite",
          r["limite"]["k"] == 6 and r["limite"]["D_U"] == est.hypergeometric_upper(N, n, 6)
          > est.hypergeometric_upper(N, n, 2) and r["conteggi"]["non_risolti"] == 3
          and r["conteggi"]["quota_non_risolta"] == f"3/{n}" and r["conteggi"]["approvato_non_indipendente"] == 1
          and r["conclusione"]["completa"] is False and r["conclusione"]["esiti_mancanti"] == 1)
    r = est.estimate(c, d, ok)
    check("risultato: conteggi descrittivi, limite valido per il disegno e conclusione condizionata al revisore, "
          "separati; nessuna certificazione",
          set(r) == {"affermazione", "disegno", "conteggi", "limite", "conclusione", "certificazione"}
          and r["conteggi"]["tipo"] == "descrittivo" and r["limite"]["tipo"] == "valido per il disegno dichiarato"
          and r["conclusione"]["tipo"].startswith("condizionata") and r["certificazione"].startswith("nessuna")
          and r["limite"]["metodo"] == "ipergeometrico" and r["limite"]["valore"] == r["limite"]["D_U"] / N
          and r["conclusione"]["completa"] is True)
    r = est.estimate(est.Claim("correttezza", "quota_qa_errate", pop, "C", 50), est.Design(pop, "C", "qa", est.UNIFORM,
                                                                                         50, 50),
                     est.Counts(approvato=47, errato=3))
    check("censimento: quota esatta k/N, senza incertezza di campionamento",
          r["limite"]["metodo"] == "censimento" and r["limite"]["D_U"] == 3 and r["limite"]["valore_esatto"] == "3/50"
          and r["limite"]["confidenza"] == 1.0)
    r = est.estimate(est.Claim("correttezza", "quota_target_token_errati", pop, "S", N),
                     est.Design(pop, "S", "qa", est.PPS, N, 300), est.Counts(approvato=298, errato=2))
    check("target-token: Clopper-Pearson col disegno con reinserimento e probabilità proporzionale ai token",
          r["limite"]["metodo"] == "clopper_pearson" and r["limite"]["valore"] == est.clopper_pearson_upper(300, 2)
          and r["limite"]["D_U"] is None)


# ---------------------------------------------------------------- esempio dei documenti

def test_example():
    """L'esempio minimo dei documenti (S4 §9, piano F0 §3), con dati fittizi: due strati, uno di QA e uno di coppie."""
    pop = "esempio-fittizio"
    qa = est.Design(pop, "A", "qa", est.UNIFORM, 10_000, 300)
    pairs = est.Design(pop, "B", "coppia", est.UNIFORM, 500, 30)
    items = [(est.Claim("correttezza", "quota_qa_errate", pop, "A", 10_000), qa,
              est.Counts(approvato=294, errato=2, incerto=1, non_giudicabile=1, mancante=1,
                         approvato_non_indipendente=1)),
             (est.Claim("correttezza", "quota_coppie_con_errore", pop, "B", 500), pairs, est.Counts(approvato=30))]
    single = [est.estimate(*x) for x in items]
    joint = est.estimate_simultaneous(items)
    numbers = [(r["limite"]["alpha"], r["limite"]["k"], r["limite"]["D_U"], round(r["limite"]["valore"], 6))
               for r in single + joint]
    why = refusal(est.estimate, est.Claim("correttezza", "quota_qa_errate", pop, "B", 500), pairs,
                  est.Counts(approvato=30))
    print("esempio:", numbers)
    print("esempio, enunciato:", single[0]["limite"]["enunciato"])
    print("esempio, conclusione:", single[0]["conclusione"]["enunciato"])
    print("esempio, rifiuto:", why)
    check("esempio dei documenti: k = 6 (2 errori e 4 esiti non approvati), D_U 388 su 10.000 e 46 su 500 al 95%, 426 "
          "e 56 con Bonferroni, quota di QA dalle coppie rifiutata",
          numbers == [("1/20", 6, 388, 0.0388), ("1/20", 0, 46, 0.092), ("1/40", 6, 426, 0.0426), ("1/40", 0, 56, 0.112)]
          and single[0]["conteggi"]["quota_non_risolta"] == "3/300" and "S4 §4" in why, str(numbers))


if __name__ == "__main__":
    test_enumeration()
    test_coverage_small()
    test_special_cases()
    test_clopper_pearson()
    test_invalid()
    test_counterexample()
    test_bonferroni()
    test_design_checks()
    test_example()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
