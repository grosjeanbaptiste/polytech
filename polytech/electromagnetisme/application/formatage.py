"""Mise en forme des nombres et des expressions pour le document.

Les nombres sont publies **bruts**, a point decimal : c'est la syntaxe
d'entree de siunitx, qui se charge seul de la virgule decimale et de
l'espace des milliers. Refaire ce travail ici donnerait deux sources de
verite pour une meme decision de presentation.
"""

from __future__ import annotations

import sympy as sp


def nombre(valeur: float, decimales: int) -> str:
    """Formate avec ce nombre de decimales, zeros de fin retires.

    Les retirer evite d'afficher une precision que la donnee n'a pas :
    `0.20` laisserait croire a deux chiffres significatifs la ou l'enonce
    n'en donne qu'un.
    """
    if decimales < 0:
        msg = f"nombre de decimales negatif : {decimales}"
        raise ValueError(msg)
    rendu = f"{valeur:.{decimales}f}"
    return rendu.rstrip("0").rstrip(".") if "." in rendu else rendu


def scientifique(valeur: float, decimales: int) -> str:
    """Formate en notation `mantisse e exposant`, que siunitx sait lire."""
    if valeur == 0.0:
        return "0"
    rendu = f"{valeur:.{decimales}e}"
    mantisse, exposant = rendu.split("e")
    return f"{mantisse}e{int(exposant)}"


def expression_latex(expression: sp.Expr, decimales: int | None = None) -> str:
    """Rend une expression sympy en LaTeX mathematique.

    Aucune resimplification : le domaine a deja etabli la forme, et la
    refaire ici echangerait les membres d'une egalite ou defairait une
    mise en facteur voulue. Seuls les flottants sont arrondis, et c'est
    bien une decision de presentation.
    """
    rendu = expression
    if decimales is not None:
        flottants = rendu.atoms(sp.Float)
        rendu = rendu.xreplace({f: round(float(f), decimales) for f in flottants})
    return str(sp.latex(rendu))
