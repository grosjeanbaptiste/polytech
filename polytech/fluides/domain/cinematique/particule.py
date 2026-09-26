"""Cours III.4 -- ce que subit une particule fluide, arete par arete.

Deux resultats que le corrige utilisait sans les etablir :
la vitesse relative des deux extremites d'une arete materielle vaut
``(du/dx) delta_x`` au premier ordre, et le taux de dilatation de l'aire
qu'elles delimitent vaut la divergence. Le second decoule du premier.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.decomposition import (
    gradient_de_vitesse,
    partie_symetrique,
)
from polytech.fluides.domain.cinematique.symboles import T, X

ECART_MATERIEL = sp.Symbol("delta_x", positive=True)
"""Longueur de l'arete materielle, destinee a tendre vers zero."""

_ORDRE_LINEAIRE = 2


def allongement_d_une_arete(u: sp.Expr) -> sp.Expr:
    """Ecart de vitesse entre les deux bouts d'une arete portee par ``Ox``.

    C'est cet ecart qui allonge l'arete : il est proportionnel a sa longueur,
    d'ou une croissance exponentielle a taux constant.
    """
    ecart = u.subs(X, X + ECART_MATERIEL) - u
    developpement = sp.series(ecart, ECART_MATERIEL, 0, _ORDRE_LINEAIRE)
    # `doit` resout le Subs que laisse le developpement d'une fonction
    # abstraite : la derivee y est evaluee au point courant, pas en un muet.
    return developpement.removeO().doit()


def dilatation_relative(u: sp.Expr, v: sp.Expr) -> sp.Expr:
    """``(1/A) DA/Dt`` pour l'aire portee par deux aretes principales.

    Chaque arete croit a son propre taux ; deriver leur produit fait
    apparaitre la somme des deux taux, c'est-a-dire la divergence.
    """
    cote_x, cote_y = sp.Function("a")(T), sp.Function("b")(T)
    taux = partie_symetrique(gradient_de_vitesse(u, v))
    aire = cote_x * cote_y
    derivee = sp.diff(aire, T).subs(
        {
            sp.Derivative(cote_x, T): taux[0, 0] * cote_x,
            sp.Derivative(cote_y, T): taux[1, 1] * cote_y,
        },
    )
    return sp.simplify(derivee / aire)
