r"""Macros des exercices 2 de mathematique, seance I : equations differentielles.

Le prefixe est `MathDeux`. Les macros sont des formules, composees par
`sympy` ; le document les inclut en mode mathematique.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import courbes_exs_02 as co
from polytech.maths.application import modeles_exs_02 as mo
from polytech.maths.application.equations_exs_02 import (
    EQUATIONS,
    second_chemin_de_la_2,
)
from polytech.maths.application.formatage import coordonnees, formule
from polytech.maths.application.macros_exs_01 import NOMBRES
from polytech.maths.application.resolue_exs_02 import X
from polytech.maths.domain.symboles import TEMPS as T

DEPART = sp.Symbol("y_0", real=True)
PRESENTATIONS = {6: sp.factor}
"""Les solutions que la feuille ecrit factorisees."""


def _equations() -> dict[str, str]:
    macros = {}
    for numero, resoudre in EQUATIONS.items():
        resolue = resoudre()
        nom = NOMBRES[numero - 1]
        for lettre, etape in zip("ABC", resolue.etapes, strict=False):
            macros[f"MathDeuxEtape{nom}{lettre}"] = formule(etape)
        presenter = PRESENTATIONS.get(numero, sp.sympify)
        macros[f"MathDeuxSolution{nom}"] = formule(presenter(resolue.solution))
    facteur, potentiel = second_chemin_de_la_2()
    macros["MathDeuxFacteurDeux"] = formule(facteur)
    macros["MathDeuxPotentielDeux"] = formule(sp.expand(potentiel))
    return macros


def _champ_et_logistique() -> dict[str, str]:
    pente = sp.Symbol("p")
    principale, seconde = mo.isoclines(pente)
    generale, constante, de_cauchy = mo.logistique_resolue()
    en_z, _ = mo.logistique_par_bernoulli()
    return {
        "MathDeuxIsocline": formule(principale),
        "MathDeuxIsoclineBis": formule(seconde),
        "MathDeuxChampSolution": formule(mo.solution_du_champ(DEPART)),
        "MathDeuxLogistique": formule(generale),
        "MathDeuxLogistiqueConstante": formule(constante),
        "MathDeuxLogistiqueCauchy": formule(de_cauchy),
        "MathDeuxBernoulliP": formule(en_z.p),
        "MathDeuxBernoulliQ": formule(en_z.q),
        "MathDeuxInflexion": formule(mo.inflexion_de_la_logistique()),
    }


def _exactes_et_ressort() -> dict[str, str]:
    premier, second = mo.potentiels()
    hors, en_resonance = mo.reponses_forcees()
    du_repos, a_la_limite = mo.reponses_depuis_le_repos()
    en_temps = {X: T}
    return {
        "MathDeuxAlpha": formule(mo.parametre_d_exactitude()),
        "MathDeuxPotentielUn": formule(premier),
        "MathDeuxPotentielBis": formule(second),
        "MathDeuxHomogeneRessort": formule(
            mo.ressort().solution_homogene().subs(en_temps)
        ),
        "MathDeuxForcee": formule(hors.subs(en_temps)),
        "MathDeuxResonante": formule(en_resonance.subs(en_temps)),
        "MathDeuxDuRepos": formule(du_repos.subs(en_temps)),
        "MathDeuxDuReposResonante": formule(a_la_limite.subs(en_temps)),
    }


def _courbes() -> dict[str, str]:
    macros = {"MathDeuxChampSegments": co.en_traits_separes(*co.segments_du_champ())}
    for lettre, (temps, ordonnees) in zip(
        "ABCD", co.courbes_du_champ().values(), strict=True
    ):
        macros[f"MathDeuxChampCourbe{lettre}"] = coordonnees(temps, ordonnees, 3)
    logistiques = co.courbes_logistiques().values()
    for lettre, (temps, ordonnees) in zip("ABC", logistiques, strict=True):
        macros[f"MathDeuxLogistiqueCourbe{lettre}"] = coordonnees(temps, ordonnees, 3)
    temps, battements, resonance = co.courbes_du_ressort()
    macros["MathDeuxRessortBattements"] = coordonnees(temps, battements, 3)
    macros["MathDeuxRessortResonance"] = coordonnees(temps, resonance, 3)
    return macros


def macros_de_la_seance_un() -> dict[str, str]:
    """Les formules et les courbes de la seance I."""
    return {
        **_equations(),
        **_champ_et_logistique(),
        **_exactes_et_ressort(),
        **_courbes(),
    }
