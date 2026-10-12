r"""Macros des reductions du chapitre X du recueil d'algebre, prefixe `AlgDix`.

Chaque etude -- un exercice, dans un cas -- compose la meme serie : la
matrice, son polynome caracteristique factorise, les multiplicites, les
sous-espaces propres, la matrice de passage et la matrice reduite. Quand il
faut deux etapes, la premiere est montree aussi.
"""

from __future__ import annotations

from polytech.maths.application import recueil_algebre_ch10_etude as etude
from polytech.maths.application.recueil_algebre import ecriture, nom_de_la_cle
from polytech.maths.application.recueil_algebre_espaces import vecteur

PREFIXE = "AlgDix"
TR = etude.TROUVE
Cas = tuple[int, str]
GENERAUX = (
    1,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    19,
    21,
    22,
    23,
    24,
    26,
    27,
    28,
    30,
    32,
    34,
    36,
)
A_TROUVER = (20, 25, 29, 31, 33, 35, 37, 40)
ETUDES: tuple[Cas, ...] = (
    *[(numero, etude.GENERAL) for numero in GENERAUX],
    *[(numero, TR) for numero in A_TROUVER],
    *[(26, "BUn"), (26, "BDeux"), (28, "AlphaUn"), (34, "AUn"), (34, "AMoinsUn")],
    *[(36, "AMoinsDeux"), (36, "ADeux"), (38, "AZero"), (38, "AMoinsUn")],
    *[(38, "APositif"), (39, "AUn")],
)
"""Les cas dont le corrige montre la reduction entiere."""
SPECTRES_SEULS: tuple[Cas, ...] = (
    *[(23, "AUn"), (23, "ADeux"), (24, "AUn"), (24, "ADeux")],
    *[(26, "BUnAZero"), (39, "AZero")],
)
"""Les cas qu'une reduction generale couvre deja : seules leurs multiplicites."""
DEUX_ETAPES = 2


def nom(cas: Cas, suffixe: str) -> str:
    """`AlgDixVingtTrouvePoly` pour l'exercice 20, parametres trouves."""
    return nom_de_la_cle(PREFIXE, f"{cas[0]}{cas[1]}", suffixe)


def _spectre(cas: Cas) -> str:
    lignes = [
        rf"\lambda = {ecriture(valeur)} : \; m_a = {algebrique},\; m_g = {len(base)}"
        for valeur, algebrique, base in etude.spectre(*cas)
    ]
    return r" \qquad ".join(lignes)


def _espaces(cas: Cas) -> str:
    lignes = [
        rf"S({ecriture(valeur)}) = \left\langle "
        + r",\; ".join(vecteur(v) for v in base)
        + r" \right\rangle"
        for valeur, _, base in etude.spectre(*cas)
    ]
    return r" \\ ".join(lignes)


def _etude(cas: Cas) -> dict[str, str]:
    reduction = etude.reduction(*cas)
    macros = {
        nom(cas, "Matrice"): ecriture(etude.matrice(*cas)),
        nom(cas, "Poly"): ecriture(etude.caracteristique(*cas)),
        nom(cas, "Spectre"): _spectre(cas),
        nom(cas, "Espaces"): _espaces(cas),
        nom(cas, "Passage"): ecriture(reduction.passage),
        nom(cas, "Reduite"): ecriture(reduction.reduite),
    }
    if reduction.etapes >= DEUX_ETAPES:
        passage, semblable = etude.premiere_etape(*cas)
        macros[nom(cas, "PremierPassage")] = ecriture(passage)
        macros[nom(cas, "Premiere")] = ecriture(semblable)
    return macros


def macros_des_etudes() -> dict[str, str]:
    """La serie de macros de chaque reduction montree."""
    macros: dict[str, str] = {}
    for cas in ETUDES:
        macros |= _etude(cas)
    for cas in SPECTRES_SEULS:
        macros[nom(cas, "Spectre")] = _spectre(cas)
    return macros
