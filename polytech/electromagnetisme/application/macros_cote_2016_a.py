"""Macros de l'exercice cote 2016, questions 1 a 3.

Prefixe `CoteSeize`. Les questions 1 et 2 publient des nombres, dans la
syntaxe d'entree de siunitx ; la question 3 est litterale et publie des
expressions.
"""

from __future__ import annotations

from polytech.electromagnetisme.application import cote_2016_couples as co
from polytech.electromagnetisme.application import cote_2016_enonce as en
from polytech.electromagnetisme.application import cote_2016_magnetique as ma
from polytech.electromagnetisme.application import cote_2016_regime as re
from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
)


def _thevenin() -> dict[str, str]:
    tension, resistance = re.thevenin()
    courant, aux_bornes, energie = re.charge_du_dipole()
    return {
        "CoteSeizeTensionAVide": expression_latex(tension),
        "CoteSeizeResistance": expression_latex(resistance),
        "CoteSeizeCourtCircuit": expression_latex(re.courant_de_court_circuit()),
        "CoteSeizeMillman": expression_latex(re.tension_par_millman()),
        "CoteSeizeCourantDeCharge": expression_latex(courant),
        "CoteSeizeTensionDeCharge": expression_latex(aux_bornes),
        "CoteSeizeEnergieEnMicrojoules": nombre(float(energie) * 1e6, 3),
        "CoteSeizeConstanteEnMicrosecondes": nombre(
            float(re.constante_de_charge()) * 1e6, 3
        ),
    }


def _circuit_magnetique() -> dict[str, str]:
    etat = ma.etat_des_colonnes()
    centrale, laterale = ma.longueurs_des_contours()
    courant = ma.courant_des_bobines()
    return {
        "CoteSeizeInductionLue": nombre(en.INDUCTION_LUE_A_250, 1),
        "CoteSeizeFluxCentral": nombre(etat.flux_central * 1e4, 2),
        "CoteSeizeFluxLateral": nombre(etat.flux_lateral * 1e4, 2),
        "CoteSeizeInductionLaterale": nombre(etat.induction_laterale, 2),
        "CoteSeizeLongueurCentrale": nombre(centrale * 1e2, 2),
        "CoteSeizeLongueurLaterale": nombre(laterale * 1e2, 2),
        "CoteSeizeAmperesTours": nombre(courant * en.SPIRES_Q2, 2),
        "CoteSeizeCourantDesBobines": nombre(courant, 3),
        "CoteSeizeInductanceApparente": nombre(ma.inductance_apparente() * 1e3, 1),
    }


def _couples() -> dict[str, str]:
    regime_1, regime_2 = co.regime()
    return {
        "CoteSeizeRegimePrimaire": expression_latex(regime_1),
        "CoteSeizeRegimeSecondaire": expression_latex(regime_2),
        "CoteSeizePenteDuSecondaire": expression_latex(
            co.pente_initiale_du_secondaire()
        ),
    }


def macros_cote_seize_debut() -> dict[str, str]:
    """Les macros des questions 1 a 3."""
    return {**_thevenin(), **_circuit_magnetique(), **_couples()}
