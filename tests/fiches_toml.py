"""Lire les articles d'une fiche et le code TikZ de leurs figures.

Les tests de figures lisent les nombres dans le TOML tel qu'il est, pour
les comparer a ce que `polytech` calcule : rien n'est recopie ici.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, Final

import sympy as sp
import pytest

_LOCAL_FICHES = Path(__file__).resolve().parent.parent / "cursus" / "fiches"
_CORPUS_FICHES = Path(__file__).resolve().parent.parent.parent / "corpus" / "cursus" / "fiches"
FICHES: Final = _LOCAL_FICHES if _LOCAL_FICHES.exists() else _CORPUS_FICHES
NATURES: Final = ("schema", "calculee")


def articles(matiere: str) -> dict[str, dict[str, Any]]:
    """Les articles de la fiche, par notion."""
    chemin = FICHES / f"{matiere}.toml"
    if not chemin.exists():
        pytest.skip(f"Fiche {matiere}.toml absente ({chemin})")
    texte = chemin.read_text(encoding="utf-8")
    return {a["notion"]: a for a in tomllib.loads(texte)["article"]}


def figure(matiere: str, notion: str) -> str:
    """Le code TikZ de la figure de l'article."""
    return str(articles(matiere)[notion]["figure"])


def nombres(texte: str) -> list[float]:
    """Les nombres d'un texte, `-0.5` et `1e5` compris."""
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", texte)]


def expression(texte: str, variable: str = "x") -> sp.Expr:
    """Une expression pgfplots (`^`, `deg`, `abs`) lue comme sympy."""
    nette = (
        texte.replace("\\", "")
        .replace("^", "**")
        .replace("deg(", "(")
        .replace("abs(", "Abs(")
    )
    return sp.sympify(nette, locals={variable: sp.Symbol(variable, real=True)})


def courbes(figure_tikz: str) -> list[str]:
    r"""Le corps `{...}` de chaque `\\addplot[...] ... {expression};`."""
    return re.findall(
        r"\\addplot\[[^\]]*\](?:\s*coordinates)?\s*\{([^}]*)\}", figure_tikz
    )


def couples(corps: str) -> list[tuple[float, float]]:
    """Les `(x,y)` d'un `coordinates {...}`."""
    trouves = re.findall(r"\(([^,()]+),([^,()]+)\)", corps)
    return [(float(a), float(b)) for a, b in trouves]


def cherche(motif: str, texte: str, groupe: int = 0) -> str:
    """Le groupe du premier `motif` ; leve si le TikZ ne le porte pas."""
    trouve = re.search(motif, texte)
    if trouve is None:
        msg = f"motif absent de la figure : {motif}"
        raise AssertionError(msg)
    return trouve.group(groupe)


def texte(matiere: str, notion: str) -> str:
    """Tout le texte d'un article : formule, raison, piege et cas."""
    article = articles(matiere)[notion]
    morceaux = [article.get(c, "") for c in ("formule", "raison", "piege")]
    morceaux += [c["texte"] for c in article.get("cas", [])]
    return "\n".join(morceaux)


def latex_vers_sympy(fragment: str) -> sp.Expr:
    r"""Un fragment LaTeX de la fiche (`\frac`, `\sqrt`, `^{}`, `\ln`...) en sympy."""
    s = re.sub(r"\\[,;]|\\ ", "", fragment)
    s = re.sub(r"\\[td]frac", r"\\frac", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    for _ in range(4):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("\\ln", "log").replace("\\arctan", "atan").replace("\\pi", "pi")
    s = re.sub(r"(atan|log) ?(\d+)", r"\1(\2)", s)
    s = re.sub(r"\^(\w)", r"**\1", s)
    s = re.sub(r"(?<![A-Za-z])e\*\*", "E**", s).replace("{", "(").replace("}", ")")
    from sympy.parsing.sympy_parser import (  # noqa: PLC0415 -- import local : sympy.parsing est lent
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    transformations = (*standard_transformations, implicit_multiplication_application)
    return sp.sympify(parse_expr(s, transformations=transformations))
