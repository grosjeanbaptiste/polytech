"""Mise en forme des nombres destines au document.

La bibliotheque publie des nombres *bruts*, a point decimal : c'est la
syntaxe d'entree de siunitx, qui se charge seul de la typographie francaise
(virgule decimale, espace fine entre milliers). Dupliquer ce travail ici
ferait deux sources de verite pour une meme decision de presentation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from polytech.fluides.application.impression import latex_du_cours

if TYPE_CHECKING:
    import sympy as sp


def nombre(valeur: float, decimales: int) -> str:
    """Formate ``valeur`` avec ``decimales`` chiffres et un point decimal.

    Les zeros de fin sont retires : siunitx affiche ce qu'on lui donne, et un
    ``981.0`` afficherait une precision que la donnee n'a pas.
    """
    if decimales < 0:
        msg = f"nombre de decimales negatif : {decimales}"
        raise ValueError(msg)
    rendu = f"{valeur:.{decimales}f}"
    return rendu.rstrip("0").rstrip(".") if "." in rendu else rendu


def expression_latex(expression: sp.Expr) -> str:
    """Rend une expression sympy en LaTeX mathematique.

    Les resultats symboliques du corrige (divergence, vorticite, champ
    d'acceleration, fonction de courant, potentiel) sont ainsi *derives* par
    la bibliotheque et non recopies a la main dans le document.
    """
    # Aucune transformation ici : la couche de presentation rend la forme
    # que le domaine a etablie. La resimplifier echangeait les membres d'une
    # egalite et defaisait les mises en facteur voulues.
    rendu = latex_du_cours(expression)
    # Un indice fait de lettres nomme un fluide ou un point, pas une variable :
    # il se compose en romain, comme partout ailleurs dans le corrige.
    return re.sub(r"_\{([A-Za-z]{2,})\}", r"_{\\text{\1}}", rendu)
