r"""Macros du corrige du chapitre IX du recueil d'algebre, prefixe `AlgNeuf`.

Une colonne s'ecrit en uplet, une famille entre accolades ; les cles des
modules d'application deviennent des noms sans chiffre.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from polytech.maths.application import recueil_algebre_ch09_fonctions as fonctions
from polytech.maths.application import recueil_algebre_ch09_vecteurs as vecteurs
from polytech.maths.application import recueil_algebre_espaces as espaces
from polytech.maths.application.formatage import nombre
from polytech.maths.application.recueil_algebre import ecriture, nom_de_la_cle

PREFIXE = "AlgNeuf"


def _rendu(valeur: Any) -> str:  # noqa: ANN401 -- colonne, matrice, famille ou expression
    if isinstance(valeur, tuple):
        return espaces.famille(valeur, _rendu)
    if isinstance(valeur, sp.MatrixBase) and valeur.cols == 1:
        return espaces.vecteur(list(sp.simplify(valeur)))
    return ecriture(sp.simplify(valeur))


def _points(couples: Any) -> str:  # noqa: ANN401 -- des couples de nombres exacts
    return " ".join(
        f"({nombre(float(x), 2)},{nombre(float(y), 2)})" for x, y in couples
    )


def _des_vecteurs() -> dict[str, Any]:
    projete, directions = vecteurs.exercice_9()
    valeurs: dict[str, Any] = dict(vecteurs.nombres())
    valeurs |= vecteurs.orthonormales()
    valeurs |= {f"{k}Orthogonale": f for k, f in vecteurs.orthogonales().items()}
    valeurs |= {
        "9Projete": projete,
        "9Directions": directions,
        "10b": vecteurs.exercice_10(),
        "12": vecteurs.exercice_12(),
        "14a": vecteurs.exercice_14(),
    }
    return valeurs


def macros_du_chapitre_neuf() -> dict[str, str]:
    """Toutes les formules que le corrige du chapitre IX compose."""
    valeurs = _des_vecteurs() | fonctions.reponses()
    pente, ordonnee = valeurs.pop("31")
    valeurs |= {
        "30": sp.Matrix(valeurs["30"]),
        "31Pente": pente,
        "31Ordonnee": ordonnee,
    }
    macros = {nom_de_la_cle(PREFIXE, cle): _rendu(v) for cle, v in valeurs.items()}
    approchee = nombre(float(vecteurs.nombres()["15"]), 3)
    macros[nom_de_la_cle(PREFIXE, "15Approchee")] = approchee
    macros[nom_de_la_cle(PREFIXE, "31Points")] = _points(fonctions.POINTS_31)
    macros[nom_de_la_cle(PREFIXE, "31Droite")] = _points(fonctions.extremites_du_31())
    return macros
