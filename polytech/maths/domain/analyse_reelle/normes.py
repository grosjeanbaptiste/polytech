"""Les normes du syllabus : le sup des coordonnees, et la norme `L_p`.

Le theoreme 6.1 (systeme differentiel normal) note `||y - y0||` « le
supremum de la valeur absolue des differences de coordonnees » ; pour un
vecteur de fonctions, le supremum porte aussi sur la variable. La
definition 10.1 appelle `L_p` l'ensemble des fonctions telles que
`int |f|^p` est fini, et `||f||_p = (int |f|^p)^(1/p)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.series.uniforme import distance_uniforme

if TYPE_CHECKING:
    from collections.abc import Sequence


def norme_sup_du_vecteur(coordonnees: Sequence[object]) -> sp.Expr:
    """`max |v_i|`, la norme du supremum des coordonnees."""
    if not coordonnees:
        msg = "vecteur vide : pas de norme du supremum"
        raise ValueError(msg)
    return sp.Max(*(sp.Abs(sp.sympify(v)) for v in coordonnees))


def norme_sup_de_fonctions(
    fonctions: Sequence[sp.Expr], variable: sp.Symbol, intervalle: sp.Set
) -> sp.Expr:
    """`sup_t max_i |f_i(t)|` sur l'intervalle."""
    plus_grands = [distance_uniforme(f, 0, variable, intervalle) for f in fonctions]
    return norme_sup_du_vecteur(plus_grands)


def _integrale_de_la_puissance(
    fonction: sp.Expr, exposant: object, variable: sp.Symbol, domaine: sp.Set
) -> sp.Expr:
    """`int |f|^p` sur le domaine."""
    if sp.sympify(exposant) < 1:
        msg = f"exposant p >= 1 requis, recu {exposant}"
        raise ValueError(msg)
    bornes = (variable, domaine.inf, domaine.sup)
    return sp.simplify(sp.integrate(sp.Abs(fonction) ** exposant, bornes))


def est_dans_lp(
    fonction: sp.Expr, exposant: object, variable: sp.Symbol, domaine: sp.Set
) -> bool:
    """Vrai si `int |f|^p` est fini sur le domaine."""
    total = _integrale_de_la_puissance(fonction, exposant, variable, domaine)
    return bool(total.is_finite)


def norme_lp(
    fonction: sp.Expr, exposant: object, variable: sp.Symbol, domaine: sp.Set
) -> sp.Expr:
    """`(int |f|^p)^(1/p)` ; leve si `f` n'est pas de puissance `p`-sommable."""
    total = _integrale_de_la_puissance(fonction, exposant, variable, domaine)
    if not total.is_finite:
        msg = f"{fonction} n'est pas de puissance {exposant}-sommable sur {domaine}"
        raise ValueError(msg)
    return sp.simplify(total ** (1 / sp.sympify(exposant)))
