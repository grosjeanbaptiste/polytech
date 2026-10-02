r"""Mise en forme des nombres et des expressions pour le document.

Les nombres sont publies **bruts**, a point decimal : c'est la syntaxe
d'entree de siunitx, qui se charge seul de la virgule decimale et de
l'espace des milliers. Refaire ce travail ici donnerait deux sources de
verite pour une meme decision de presentation.

Le logarithme naturel est rendu `\ln` et non `\log`. Ce n'est pas une
preference : en notation francaise `\log` designe le logarithme
**decimal**, et composer `\log` ecrirait une autre fonction que celle
calculee.
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
    # `ln_notation` : sympy rend `\log` pour le logarithme naturel, ce qui
    # est l'usage anglo-saxon. En notation francaise `\log` designe le
    # logarithme **decimal** -- composer `\log` ecrirait donc une autre
    # fonction que celle calculee.
    return str(sp.latex(rendu, ln_notation=True))


def forme_signee(expression: sp.Expr) -> str:
    """Rend l'expression avec le signe **devant** la fraction.

    `sp.together` reunit bien les termes mais laisse le moins dans le
    numerateur : `frac{I(-R^2 - y(a-y))}{2 pi R^2 y}`. Les corriges
    manuscrits ecrivent le moins devant la fraction et le numerateur en
    positif, ce qui se lit mieux et evite de chercher ou le signe se
    trouve.

    Le procede : reunir l'**oppose**, puis nier. Sur une expression deja
    positive, l'oppose est negatif et la reunion le garde negatif, de
    sorte que la negation finale rend bien du positif -- une mise en
    forme ne doit jamais introduire un signe absent de l'original.
    """
    return expression_latex(-sp.together(-expression))
