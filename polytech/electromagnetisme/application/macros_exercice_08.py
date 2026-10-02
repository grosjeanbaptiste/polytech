"""Macros de l'exercice 8 : Thevenin avec une source de courant dans la maille.

Les deux questions sont publiees avec les termes qui y menent, pour que le
texte puisse ecrire le chemin et non seulement son resultat.

Deux publications meritent leur raison.

Les **fractions exactes** sont publiees a cote des decimales : `R_TH` vaut
`5/6` et non `0,83`, et c'est la forme fractionnaire qui explique pourquoi
le courant final est exactement deux amperes -- les sixiemes se
compensent. Publier seulement la decimale perdrait cette information.

Le **lapsus du manuscrit** est publie lui aussi. La transcription ecrit
`E_TH = 10 + 5 x (5/6 + 1) = 20 + 5 x 11/6 = 19,16 V`, et le membre du
milieu vaut `29,17`. Le document doit pouvoir citer les deux valeurs pour
nommer l'ecart, et aucune des deux n'est tapee a la main.
"""

from __future__ import annotations

from polytech.electromagnetisme.application import enonce_exercice_08 as donnees
from polytech.electromagnetisme.application.formatage import nombre
from polytech.electromagnetisme.domain.circuits.associations import (
    courant_dans_la_charge,
    courant_de_maille,
    thevenin_de_la_maille,
)

_D = donnees.DECIMALES_AFFICHEES
_F = donnees.DECIMALES_FINES


def macros_donnees_huit() -> dict[str, str]:
    """Les sept nombres de l'enonce, pour que le texte ne les tape pas."""
    return {
        "HuitPremiereTension": nombre(donnees.PREMIERE_TENSION, _D),
        "HuitSecondeTension": nombre(donnees.SECONDE_TENSION, _D),
        "HuitPremiereResistance": nombre(donnees.PREMIERE_RESISTANCE, _D),
        "HuitSecondeResistance": nombre(donnees.SECONDE_RESISTANCE, _D),
        "HuitCourantInjecte": nombre(donnees.COURANT_INJECTE, _D),
        "HuitCharge": nombre(donnees.CHARGE, _D),
        "HuitCourantEnParallele": nombre(donnees.COURANT_EN_PARALLELE, _D),
    }


def macros_exercice_huit() -> dict[str, str]:
    """Les deux reponses, leurs termes, et le lapsus du manuscrit."""
    maille = courant_de_maille(
        donnees.PREMIERE_TENSION,
        donnees.SECONDE_TENSION,
        donnees.PREMIERE_RESISTANCE,
        donnees.SECONDE_RESISTANCE,
        donnees.COURANT_INJECTE,
    )
    thevenin = thevenin_de_la_maille(
        donnees.PREMIERE_TENSION,
        donnees.SECONDE_TENSION,
        donnees.PREMIERE_RESISTANCE,
        donnees.SECONDE_RESISTANCE,
        donnees.COURANT_INJECTE,
    )
    courant = courant_dans_la_charge(
        thevenin,
        donnees.CHARGE,
        donnees.COURANT_EN_PARALLELE,
    )
    produit = donnees.PREMIERE_RESISTANCE * donnees.SECONDE_RESISTANCE
    somme = donnees.PREMIERE_RESISTANCE + donnees.SECONDE_RESISTANCE
    traverse = maille + donnees.COURANT_INJECTE
    return {
        "HuitProduitDesResistances": nombre(produit, _D),
        "HuitSommeDesResistances": nombre(somme, _D),
        "HuitResistanceThevenin": nombre(thevenin.resistance, _F),
        "HuitCourantDeMaille": nombre(maille, _F),
        "HuitChuteDeLaSource": nombre(
            donnees.SECONDE_RESISTANCE * donnees.COURANT_INJECTE,
            _D,
        ),
        "HuitCourantTraversant": nombre(traverse, _F),
        "HuitTensionThevenin": nombre(thevenin.tension_a_vide, _D),
        "HuitApportDeLaSourceEnParallele": nombre(
            thevenin.resistance * donnees.COURANT_EN_PARALLELE,
            _F,
        ),
        "HuitResistanceTotale": nombre(donnees.CHARGE + thevenin.resistance, _F),
        "HuitCourantDansLaCharge": nombre(courant, _D),
        "HuitLapsusDuManuscrit": nombre(
            donnees.PREMIERE_TENSION + donnees.SECONDE_RESISTANCE * traverse,
            _D,
        ),
    }
