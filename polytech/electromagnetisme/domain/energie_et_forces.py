"""L'energie electromagnetique, et les forces qu'on en tire.

Le chapitre IV.2 enseigne une methode plutot qu'une formule : la force se
calcule en **derivant l'energie** par rapport au deplacement, a flux
constant ou a courant constant selon le probleme -- et les deux ne donnent
pas le meme signe de terme, ce qui est le piege du chapitre.

Pour un circuit magnetique, l'energie vaut `L i^2 / 2` et la force
`dW/dx`. Pour un condensateur, `C u^2 / 2`. Les deux sont ici, et la
derivation est symbolique : le corrige doit pouvoir montrer la derivee,
pas seulement son resultat.
"""

from __future__ import annotations

import sympy as sp

DEPLACEMENT = sp.Symbol("x", real=True)
"""La coordonnee selon laquelle la force est cherchee."""


def densite_d_energie_electrique(permittivite: float, champ: float) -> float:
    """`epsilon E^2 / 2`, en J/m^3."""
    if permittivite <= 0:
        msg = f"permittivite nulle ou negative : {permittivite} F/m"
        raise ValueError(msg)
    return permittivite * champ**2 / 2


def densite_d_energie_magnetique(permeabilite: float, champ: float) -> float:
    """`mu H^2 / 2`, en J/m^3."""
    if permeabilite <= 0:
        msg = f"permeabilite nulle ou negative : {permeabilite} H/m"
        raise ValueError(msg)
    return permeabilite * champ**2 / 2


def energie_d_une_inductance(inductance: sp.Expr, courant: sp.Expr) -> sp.Expr:
    """`L i^2 / 2`."""
    return inductance * courant**2 / 2


def energie_d_un_condensateur(capacite: sp.Expr, tension: sp.Expr) -> sp.Expr:
    """`C u^2 / 2`."""
    return capacite * tension**2 / 2


def force_a_courant_constant(energie: sp.Expr) -> sp.Expr:
    """`F = +dW/dx` a courant constant.

    Le signe est **positif**, et c'est contre-intuitif : a courant
    constant la source fournit du travail en plus de l'energie stockee, et
    les deux contributions se composent pour donner `+dW/dx`.
    """
    return sp.simplify(sp.diff(energie, DEPLACEMENT))


def force_a_flux_constant(energie: sp.Expr) -> sp.Expr:
    """`F = -dW/dx` a flux constant.

    Ici la source ne fournit rien -- le flux ne varie pas, donc aucune
    force electromotrice -- et toute la force vient de la decroissance de
    l'energie stockee. D'ou le signe oppose au cas precedent.
    """
    return sp.simplify(-sp.diff(energie, DEPLACEMENT))


def les_deux_signes_s_opposent(energie: sp.Expr) -> bool:
    """Vrai si les deux methodes donnent des forces opposees.

    C'est le piege du chapitre, et le verifier evite de le reinventer :
    les deux expressions ne diffèrent **que** par le signe.
    """
    somme = force_a_courant_constant(energie) + force_a_flux_constant(energie)
    return bool(sp.simplify(somme) == 0)
