r"""Macros du corrige du chapitre VIII, exercices 14 a 32, prefixe `AlgHuit`."""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import recueil_algebre_ch08b_bases as bases
from polytech.maths.application import recueil_algebre_ch08b_matrices as matrices
from polytech.maths.application import recueil_algebre_ch08b_suite as suite
from polytech.maths.application import recueil_algebre_espaces as espaces
from polytech.maths.application.recueil_algebre import ecriture, nom_de_la_cle

PREFIXE = "AlgHuit"


def _matrice_ou_vecteur(matrice: sp.Matrix) -> str:
    """Une colonne s'ecrit en uplet, comme le syllabus ecrit une image."""
    simplifiee = sp.simplify(matrice)
    return (
        espaces.vecteur(list(simplifiee)) if matrice.cols == 1 else ecriture(simplifiee)
    )


def macros_de_la_partie_b() -> dict[str, str]:
    """Toutes les formules que le corrige des exercices 14 a 32 compose."""
    toutes = matrices.toutes() | bases.toutes()
    macros = {
        nom_de_la_cle(PREFIXE, k): _matrice_ou_vecteur(m) for k, m in toutes.items()
    }
    dans_v, dans_v_prime = bases.coordonnees_du_20()
    noyau, _ = suite.noyau_et_rang_du_27()
    q, p = suite.reduction_du_30()
    macros |= {
        nom_de_la_cle(PREFIXE, "15b"): ecriture(matrices.homothetie(3)),
        nom_de_la_cle(PREFIXE, "20b"): espaces.vecteur(dans_v),
        nom_de_la_cle(PREFIXE, "20c"): espaces.vecteur(dans_v_prime),
        nom_de_la_cle(PREFIXE, "22Primitive"): ecriture(matrices.primitive_du_22()),
        nom_de_la_cle(PREFIXE, "27Noyau"): espaces.famille(noyau),
        nom_de_la_cle(PREFIXE, "30Q"): ecriture(q),
        nom_de_la_cle(PREFIXE, "30P"): ecriture(p),
        nom_de_la_cle(PREFIXE, "31"): espaces.vecteur(suite.exercice_31()),
    }
    return macros
