r"""Mise en forme pour le document : ce qu'un programme affiche a l'ecran.

Un corrige de programmation montre des lignes de terminal. Elles se
composent en `\texttt`, et les caracteres que LaTeX reserve y sont
echappes -- a commencer par le tiret bas de `T_P`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

_RESERVES: Final = {
    "\\": r"\textbackslash{}",
    "_": r"\_",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "{": r"\{",
    "}": r"\}",
}


def machine(texte: str) -> str:
    r"""Une ligne de terminal sous `\texttt`, caracteres reserves echappes."""
    echappe = "".join(_RESERVES.get(caractere, caractere) for caractere in texte)
    return rf"\texttt{{{echappe}}}"


def session(lignes: Sequence[str]) -> str:
    r"""Plusieurs lignes de terminal, separees par `\\` : une par rangee de tableau."""
    if not lignes:
        msg = "session sans aucune ligne"
        raise ValueError(msg)
    return r" \\ ".join(machine(ligne) for ligne in lignes)
