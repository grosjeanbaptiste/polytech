r"""Macros du chapitre X hors reductions : enonces, parametres, reflexions.

Elles completent `macros_recueil_algebre_ch10` : la matrice generale et le
polynome caracteristique des sujets a parametres, les candidats que les
conditions laissent et celui que la garde retient, puis les exercices 2 a
4 et 12 a 18.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import recueil_algebre_ch10_enonces as enonces
from polytech.maths.application import recueil_algebre_ch10_etude as etude
from polytech.maths.application import recueil_algebre_ch10_parametres as parametres
from polytech.maths.application import recueil_algebre_ch10_reflexions as reflexions
from polytech.maths.application.macros_recueil_algebre_ch10 import (
    PREFIXE,
    macros_des_etudes,
)
from polytech.maths.application.recueil_algebre import ecriture, liste, nom_de_la_cle
from polytech.maths.application.recueil_algebre_espaces import famille, vecteur

GENERAUX = (20, 25, 29, 31, 33, 35, 37, 39, 40)
"""Les sujets dont l'enonce porte un parametre, hors ceux deja etudies tels quels."""


def _nom(cle: str) -> str:
    return nom_de_la_cle(PREFIXE, cle)


def _candidat(candidat: parametres.Candidat) -> str:
    paires = sorted(candidat.items(), key=lambda paire: str(paire[0]))
    return r",\; ".join(f"{ecriture(k)} = {ecriture(v)}" for k, v in paires)


def _enonces() -> dict[str, str]:
    macros = {}
    for numero in GENERAUX:
        polynome = sp.collect(etude.caracteristique(numero), enonces.T)
        macros[_nom(f"{numero}Enonce")] = ecriture(etude.matrice(numero))
        macros[_nom(f"{numero}EnoncePoly")] = ecriture(polynome)
    return macros


def _parametres() -> dict[str, str]:
    macros = {}
    for numero in parametres.SUJETS:
        tous = [_candidat(c) for c in parametres.candidats(numero)]
        macros[_nom(f"{numero}Candidats")] = r" \quad\text{ou}\quad ".join(tous)
        macros[_nom(f"{numero}Retenus")] = _candidat(parametres.retenus(numero))
    en_a, lie = parametres.exercice_40()
    return macros | {_nom("40EnA"): ecriture(en_a), _nom("40Lie"): ecriture(lie)}


def _proprietes() -> dict[str, str]:
    valeurs, _ = reflexions.exercice_2()
    du_vecteur, autres, diagonale = reflexions.exercice_14()
    _, _, plus, moins = reflexions.exercice_16()
    return {
        _nom("2Enonce"): ecriture(enonces.MATRICES[2]),
        _nom("2Valeurs"): liste(valeurs),
        _nom("3Enonce"): ecriture(enonces.MATRICES[3]),
        _nom("3Image"): vecteur(tuple(enonces.MATRICES[3] * enonces.VECTEUR_3)),
        _nom("3Valeur"): ecriture(reflexions.exercice_3()),
        _nom("4Geometrique"): str(reflexions.exercice_4()[0]),
        _nom("14Enonce"): ecriture(enonces.MATRICES[14]),
        _nom("14Image"): vecteur(tuple(enonces.MATRICES[14] * enonces.VECTEUR_14)),
        _nom("14Valeur"): ecriture(du_vecteur),
        _nom("14Trace"): ecriture(enonces.MATRICES[14].trace()),
        _nom("14Autres"): liste(autres),
        _nom("14Diagonale"): ecriture(diagonale),
        _nom("16Enonce"): ecriture(enonces.MATRICES[16]),
        _nom("16Plus"): famille(plus),
        _nom("16Moins"): famille(moins),
    }


def _calculs() -> dict[str, str]:
    orthogonale, diagonale = reflexions.exercice_12()
    passage, du_carre, racine = reflexions.exercice_15()
    derivation, couples = reflexions.exercice_17()
    projection, droite, plan = reflexions.exercice_18()
    return {
        _nom("12Orthogonale"): ecriture(orthogonale),
        _nom("12Diagonale"): ecriture(diagonale),
        _nom("13Puissance"): ecriture(reflexions.exercice_13()),
        _nom("15Carre"): ecriture(enonces.CARRE_15),
        _nom("15Passage"): ecriture(passage),
        _nom("15Diagonale"): ecriture(du_carre),
        _nom("15Racine"): ecriture(racine),
        _nom("17Matrice"): ecriture(derivation),
        _nom("17Valeurs"): liste(valeur for valeur, _ in couples),
        _nom("18Matrice"): ecriture(projection),
        _nom("18Droite"): famille(droite),
        _nom("18Plan"): famille(plan),
        _nom("23Puissance"): ecriture(reflexions.exercice_23()),
    }


def macros_du_chapitre_dix() -> dict[str, str]:
    """Toutes les formules que le corrige du chapitre X compose."""
    return {
        **macros_des_etudes(),
        **_enonces(),
        **_parametres(),
        **_proprietes(),
        **_calculs(),
    }
