r"""Macros du corrige du chapitre VIII, exercices 33 a 45, prefixe `AlgHuit`.

Une base s'ecrit dans l'espace ou elle vit : des polynomes, des matrices
2 x 2 ou 2 x 3, des uplets. `ESPACES` le dit pour le depart et l'arrivee de
chaque application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

from polytech.maths.application import recueil_algebre_ch08c_enonces as enonces
from polytech.maths.application import recueil_algebre_ch08c_etude as etude
from polytech.maths.application import recueil_algebre_ch08c_metrique as metrique
from polytech.maths.application import recueil_algebre_espaces as espaces
from polytech.maths.application.recueil_algebre import ecriture, nom_de_la_cle

if TYPE_CHECKING:
    from collections.abc import Sequence

PREFIXE = "AlgHuit"


def _expressions(famille: Sequence[sp.Expr]) -> str:
    r"""Des polynomes deja formes, entre accolades."""
    interieur = r",\; ".join(ecriture(e) for e in famille)
    return rf"\left\{{{interieur}\right\}}"


def _deux_trois(termes: Sequence[object]) -> str:
    return ecriture(sp.Matrix(2, 3, [sp.sympify(t) for t in termes]))


P, Q, V, R = espaces.polynome, espaces.matrice, espaces.vecteur, _deux_trois
ESPACES: dict[int, tuple[Any, Any]] = {
    33: (P, P), 34: (Q, Q), 35: (Q, Q), 36: (Q, Q), 37: (Q, R), 38: (P, Q),
    39: (P, V), 40: (V, V), 41: (Q, P), 42: (V, P), 43: (Q, P), 44: (P, P),
    45: (Q, P),
}  # fmt: skip
"""Numero : comment s'ecrit un element du depart, puis de l'arrivee."""


def _nom(cle: str) -> str:
    return nom_de_la_cle(PREFIXE, cle)


def _etudes() -> dict[str, str]:
    macros: dict[str, str] = {}
    for numero, (depart, arrivee) in ESPACES.items():
        macros[_nom(f"{numero}Matrice")] = ecriture(etude.matrice(numero))
        for cle in (etude.GENERAL, *enonces.CAS.get(numero, {})):
            noyau, image = etude.noyau(numero, cle), etude.image(numero, cle)
            if noyau:
                macros[_nom(f"{numero}{cle}Noyau")] = espaces.famille(noyau, depart)
            if len(image) < etude.matrice(numero).rows:
                macros[_nom(f"{numero}{cle}Image")] = espaces.famille(image, arrivee)
    return macros


def _orthogonaux() -> dict[str, str]:
    cas = (
        (34, ""),
        (36, "Opposes"),
        (36, "Egaux"),
        (37, "Egaux"),
        (37, "Opposes"),
        (41, ""),
    )
    return {
        _nom(f"{n}{cle}Orthogonal"): espaces.famille(
            metrique.orthogonal_du_noyau(n, cle), Q
        )
        for n, cle in cas
    }


def _particuliers() -> dict[str, str]:
    orthogonal_33, sur_le_noyau, sur_l_orthogonal = metrique.exercice_33()
    _, image_42, orthogonal_42 = metrique.exercice_42()
    _, symetriques, associee = metrique.exercice_45()
    (antecedent_39,) = etude.antecedents(39, (0, 1, -2, 1))
    (antecedent_40,) = etude.antecedents(40, (1, 3))
    (antecedent_41,) = etude.antecedents(41, (3, 5, 5, 3))
    return {
        _nom("33Orthogonal"): _expressions(orthogonal_33),
        _nom("33SurLeNoyau"): ecriture(sur_le_noyau),
        _nom("33SurLOrthogonal"): ecriture(sur_l_orthogonal),
        _nom("35Det"): ecriture(etude.determinant(35)),
        _nom("36Det"): ecriture(etude.determinant(36)),
        _nom("36Inverse"): ecriture(etude.inverse_du_36()),
        _nom("39Antecedents"): P(antecedent_39),
        _nom("40Antecedents"): V(antecedent_40),
        _nom("41Antecedents"): Q(antecedent_41),
        _nom("41OrthogonauxAX"): _expressions(metrique.exercice_41()),
        _nom("42Image"): V(image_42),
        _nom("42Orthogonal"): espaces.famille(orthogonal_42, Q),
        _nom("44Somme"): espaces.famille(metrique.exercice_44(), P),
        _nom("45Symetriques"): espaces.famille(symetriques, Q),
        _nom("45Associee"): ecriture(associee),
    }


def macros_de_la_partie_c() -> dict[str, str]:
    """Toutes les formules que le corrige des exercices 33 a 45 compose."""
    return {**_etudes(), **_orthogonaux(), **_particuliers()}
