"""LaTeX delle formule di UniMER: dal formato a token separati da spazi (stile im2latex) a LaTeX naturale,
e confronto fra trascrizioni.

Nel formato im2latex ogni token è separato da uno spazio (`\\frac { 1 } { 2 }`, `0 . 5`); uno spazio di
controllo `\\ ` diventa il token `\\` seguito dal separatore. `despace` toglie i separatori e rimette uno
spazio solo dove serve: dopo una parola di controllo seguita da una lettera (`\\alpha b`), e nello spazio
di controllo. `latex_tokens` divide in token sia il formato spaziato sia quello naturale, così due
trascrizioni si confrontano a meno degli spazi. `visible_tokens` tiene solo i token che cambiano l'aspetto
della formula (niente spazi, graffe, comandi di font, \\left/\\right, ambienti array; sinonimi unificati), così
una notazione diversa per la stessa formula non conta come errore.
"""
from __future__ import annotations

import re

_CONTROL_WORD_END = re.compile(r"\\[A-Za-z]+$")
_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|\S")


def despace(spaced: str) -> str:
    out = ""
    for token in spaced.split():
        if token == "\\":  # spazio di controllo
            out += "\\ "
            continue
        if out and token[0].isalpha() and _CONTROL_WORD_END.search(out):
            out += " "
        out += token
    return out


def latex_tokens(text: str) -> list[str]:
    """Token LaTeX: parole di controllo, simboli di controllo (incluso `\\ `), singoli caratteri non spazio."""
    return _TOKEN.findall(text)


def same_latex(a: str, b: str) -> bool:
    return latex_tokens(a) == latex_tokens(b)


# ---- token visibili: quelli che cambiano l'aspetto della formula (analisi degli errori di F1, densità di F1c) ----
SPACING = {r"\,", r"\;", r"\:", r"\!", r"\quad", r"\qquad", "\\ ", "~", r"\enspace", r"\thinspace", r"\hspace",
           r"\hfill", r"\vspace"}
FONT = {r"\mathrm", r"\mathbf", r"\mathcal", r"\mathit", r"\mathsf", r"\mathtt", r"\mathbb", r"\mathfrak",
        r"\mathscr", r"\boldsymbol", r"\bm", r"\operatorname", r"\text", r"\textrm", r"\textbf", r"\textit",
        r"\rm", r"\bf", r"\it", r"\cal", r"\sf", r"\tt", r"\displaystyle", r"\textstyle", r"\scriptstyle",
        r"\scriptscriptstyle", r"\mathop", r"\mathbin", r"\mathrel", r"\mathord"}
SIZING = {r"\left", r"\right", r"\big", r"\Big", r"\bigg", r"\Bigg", r"\bigl", r"\bigr", r"\Bigl", r"\Bigr",
          r"\biggl", r"\biggr", r"\Biggl", r"\Biggr", r"\limits", r"\nolimits", r"\middle"}
SYNONYMS = {r"\ge": r"\geq", r"\le": r"\leq", r"\ne": r"\neq", r"\to": r"\rightarrow", r"\gets": r"\leftarrow",
            r"\lbrace": r"\{", r"\rbrace": r"\}", r"\vert": "|", r"\lvert": "|", r"\rvert": "|", r"\Vert": r"\|",
            r"\lVert": r"\|", r"\rVert": r"\|", r"\dots": r"\ldots", r"\dotsc": r"\ldots", r"\dotsb": r"\cdots",
            r"\land": r"\wedge", r"\lor": r"\vee", r"\lnot": r"\neg", r"\colon": ":", r"\vline": "|"}
GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta", "iota",
         "kappa", "lambda", "mu", "nu", "xi", "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon",
         "phi", "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon",
         "Phi", "Psi", "Omega"}
BRACKETS = {"(", ")", "[", "]", "|", r"\{", r"\}", r"\|", r"\langle", r"\rangle", r"\lfloor", r"\rfloor",
            r"\lceil", r"\rceil"}
OPERATOR_CMDS = {r"\cdot", r"\times", r"\pm", r"\mp", r"\div", r"\leq", r"\geq", r"\neq", r"\approx", r"\sim",
                 r"\simeq", r"\equiv", r"\cong", r"\rightarrow", r"\leftarrow", r"\Rightarrow", r"\Leftarrow",
                 r"\leftrightarrow", r"\Leftrightarrow", r"\mapsto", r"\in", r"\ni", r"\notin", r"\subset",
                 r"\subseteq", r"\supset", r"\supseteq", r"\cup", r"\cap", r"\wedge", r"\vee", r"\neg", r"\otimes",
                 r"\oplus", r"\ast", r"\star", r"\circ", r"\bullet", r"\propto", r"\ll", r"\gg", r"\mid",
                 r"\perp", r"\parallel", r"\cdots", r"\ldots", r"\vdots", r"\ddots", r"\prime", r"\dagger",
                 r"\setminus", r"\langle", r"\rangle"}
INVISIBLE = {"spazio", "font", "dimensione", "graffe", "allineamento"}


def tclass(tok: str) -> str:
    if tok in SPACING:
        return "spazio"
    if tok in FONT:
        return "font"
    if tok in SIZING:
        return "dimensione"
    if tok in ("{", "}"):
        return "graffe"
    if tok in ("^", "_"):
        return "indici"
    if tok in ("&", "\\\\"):
        return "allineamento"
    if tok.isdigit():
        return "cifra"
    if len(tok) == 1 and tok.isalpha():
        return "lettera"
    if tok.startswith("\\") and tok[1:] in GREEK:
        return "greca"
    if tok in BRACKETS:
        return "parentesi"
    if tok in OPERATOR_CMDS or len(tok) == 1:
        return "operatore"
    return "comando"


def visible_tokens(text: str) -> list[str]:
    """Token che cambiano l'aspetto della formula, con i sinonimi unificati e x^{\\prime} = x'."""
    text = re.sub(r"\\begin\{array\}\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:begin|end)\{[A-Za-z*]+\}", " ", text)
    toks = [SYNONYMS.get(t, t) for t in latex_tokens(text)]
    toks = [t for t in toks if tclass(t) not in INVISIBLE]
    out, i = [], 0
    while i < len(toks):
        if toks[i] == "^" and i + 1 < len(toks) and toks[i + 1] == r"\prime":
            i += 1
        out.append("'" if toks[i] == r"\prime" else toks[i])
        i += 1
    return out


def structure_problems(text: str) -> list[str]:
    """Problemi di struttura che rendono la formula non compilabile o ambigua (controllo di UniMER senza LLM):
    "graffe" (sbilanciate), "left_right" (numero di \\left diverso da \\right), "ambiente" (\\begin/\\end non
    corrispondenti). Lista vuota se non ce ne sono."""
    toks = latex_tokens(text)
    problems, depth = set(), 0
    for t in toks:
        depth += (t == "{") - (t == "}")
        if depth < 0:
            problems.add("graffe")
            depth = 0
    if depth:
        problems.add("graffe")
    if toks.count(r"\left") != toks.count(r"\right"):
        problems.add("left_right")
    envs, i = [], 0
    while i < len(toks):
        if toks[i] in (r"\begin", r"\end") and i + 1 < len(toks) and toks[i + 1] == "{":
            j = i + 2
            while j < len(toks) and toks[j] != "}":
                j += 1
            name = "".join(toks[i + 2:j])
            if toks[i] == r"\begin":
                envs.append(name)
            elif not envs or envs.pop() != name:
                problems.add("ambiente")
            i = j
        i += 1
    if envs:
        problems.add("ambiente")
    return sorted(problems)
