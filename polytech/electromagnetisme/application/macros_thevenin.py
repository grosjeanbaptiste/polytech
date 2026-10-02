"""Macros de l'exercice 1 : equivalences Thevenin / Norton.

Les cinq questions de l'exercice sont publiees, et chacune l'est **avec
le chemin qui y mene** : un corrige pas a pas doit montrer d'ou vient le
nombre, non le poser. La tension a vide, par exemple, est publiee en
meme temps que ses deux termes `110` et `R I`, pour que le texte puisse
ecrire la somme et non seulement son resultat.

Le signe de la puissance est publie tel que le calcul le rend. Le
document ne le corrige pas : c'est le resultat de la question (e), et sa
negativite est ce qu'elle enseigne.
"""

from __future__ import annotations

from polytech.electromagnetisme.application import enonce
from polytech.electromagnetisme.application.formatage import nombre
from polytech.electromagnetisme.domain.circuits.stationnaires import (
    puissance_echangee,
    source_de_tension_et_source_de_courant,
)

_DECIMALES = enonce.DECIMALES_AFFICHEES


def macros_donnees() -> dict[str, str]:
    """Les quatre nombres de l'enonce, pour que le texte ne les tape pas."""
    return {
        "TensionDeLaSource": nombre(enonce.TENSION_DE_LA_SOURCE, _DECIMALES),
        "ResistanceEnSerie": nombre(enonce.RESISTANCE_EN_SERIE, _DECIMALES),
        "CourantDeLaSource": nombre(enonce.COURANT_DE_LA_SOURCE, _DECIMALES),
        "ChargeQuestionD": nombre(enonce.CHARGE_DE_LA_QUESTION_D, _DECIMALES),
    }


def macros_thevenin() -> dict[str, str]:
    """Les cinq reponses, et les termes intermediaires qui les justifient."""
    dipole = source_de_tension_et_source_de_courant(
        enonce.TENSION_DE_LA_SOURCE,
        enonce.RESISTANCE_EN_SERIE,
        enonce.COURANT_DE_LA_SOURCE,
    )
    norton = dipole.en_norton()
    charge = enonce.CHARGE_DE_LA_QUESTION_D
    debit = dipole.tension_a_vide / (dipole.resistance + charge)
    sur_la_charge = dipole.tension_sur(charge)
    puissance = puissance_echangee(sur_la_charge, -enonce.COURANT_DE_LA_SOURCE)
    return {
        "ChuteDansLaResistance": nombre(
            enonce.RESISTANCE_EN_SERIE * enonce.COURANT_DE_LA_SOURCE,
            _DECIMALES,
        ),
        "TensionAVide": nombre(dipole.tension_a_vide, _DECIMALES),
        "CourantDeCourtCircuitSeul": nombre(
            enonce.TENSION_DE_LA_SOURCE / enonce.RESISTANCE_EN_SERIE,
            _DECIMALES,
        ),
        "CourantDeCourtCircuit": nombre(norton.courant_de_court_circuit, _DECIMALES),
        "ResistanceEquivalente": nombre(dipole.resistance, _DECIMALES),
        "ConductanceEquivalente": nombre(norton.conductance(), _DECIMALES),
        "ResistanceTotaleQuestionD": nombre(dipole.resistance + charge, _DECIMALES),
        "CourantDebite": nombre(debit, _DECIMALES),
        "TensionSurLaCharge": nombre(sur_la_charge, _DECIMALES),
        "CourantDansLaSource": nombre(-enonce.COURANT_DE_LA_SOURCE, _DECIMALES),
        "PuissanceEchangee": nombre(puissance, _DECIMALES),
        "PuissanceEchangeeKilowatts": nombre(puissance / 1000, _DECIMALES),
    }
