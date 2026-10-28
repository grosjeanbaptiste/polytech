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

FICHES: Final = Path(__file__).resolve().parent.parent / "cursus" / "fiches"
NATURES: Final = ("schema", "calculee")


def articles(matiere: str) -> dict[str, dict[str, Any]]:
    """Les articles de la fiche, par notion."""
    texte = (FICHES / f"{matiere}.toml").read_text(encoding="utf-8")
    return {a["notion"]: a for a in tomllib.loads(texte)["article"]}


def figure(matiere: str, notion: str) -> str:
    """Le code TikZ de la figure de l'article."""
    return str(articles(matiere)[notion]["figure"])


def nombres(texte: str) -> list[float]:
    """Les nombres d'un texte, `-0.5` et `1e5` compris."""
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", texte)]


def expression(texte: str, variable: str = "x") -> sp.Expr:
    """Une expression pgfplots (`^`, `deg`, `abs`) lue comme sympy."""
    nette = texte.replace("^", "**").replace("deg(", "(").replace("abs(", "Abs(")
    return sp.sympify(nette, locals={variable: sp.Symbol(variable, real=True)})


def courbes(figure_tikz: str) -> list[str]:
    """Le corps `{...}` de chaque `\\addplot[...] ... {expression};`."""
    return re.findall(
        r"\\addplot\[[^\]]*\](?:\s*coordinates)?\s*\{([^}]*)\}", figure_tikz
    )


def couples(corps: str) -> list[tuple[float, float]]:
    """Les `(x,y)` d'un `coordinates {...}`."""
    trouves = re.findall(r"\(([^,()]+),([^,()]+)\)", corps)
    return [(float(a), float(b)) for a, b in trouves]
