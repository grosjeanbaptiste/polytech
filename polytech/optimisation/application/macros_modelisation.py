"""Macros de « la modelisation », problemes I, IV et V. Prefixe `Modele`."""

from __future__ import annotations

from polytech.optimisation.application import modele_personnel as pe
from polytech.optimisation.application import modele_pilotes as pi
from polytech.optimisation.application import modele_radios as ra
from polytech.optimisation.application.formatage import arrondi, exact, point
from polytech.optimisation.application.macros_modelisation_ateliers import (
    macros_des_ateliers,
)
from polytech.optimisation.application.macros_modelisation_variantes import (
    macros_des_variantes,
)

_LETTRES = "ABCDEFGHIJKL"
_NOMBRES = {2: "Deux", 5: "Cinq", 6: "Six", 7: "Sept"}
_COUT_A_QUATRE = 59890


def _personnel() -> dict[str, str]:
    plan = pe.plan()
    macros = {
        "ModelePersonnelCout": str(plan.cout),
        "ModelePersonnelLicencies": str(sum(plan.licenciements)),
        "ModelePersonnelDernier": str(plan.licenciements[-1]),
    }
    lignes = zip(
        pe.BESOINS,
        plan.effectifs,
        plan.embauches[:-1],
        plan.licenciements[:-1],
        strict=True,
    )
    for lettre, ligne in zip(_LETTRES, lignes, strict=True):
        macros[f"ModelePersonnelLigne{lettre}"] = " & ".join(str(v) for v in ligne)
    return macros


def _pilotes() -> dict[str, str]:
    plan = pi.PLAN_A_QUATRE
    macros = {
        "ModelePilotesRelaxTrois": arrondi(pi.relaxation(3), 2),
        "ModelePilotesRelaxQuatre": arrondi(pi.relaxation(4), 1),
        "ModelePilotesEntier": exact(pi.cout_du_plan(plan)),
        "ModelePilotesVols": point(plan.vols),
        "ModelePilotesRepos": point(plan.repos),
        "ModelePilotesMoinsChers": str(len(pi.choix_moins_chers(4, _COUT_A_QUATRE))),
        "ModelePilotesSansQuatre": exact(pi.vols_possibles(4, (0,) * 4)),
        "ModelePilotesSansCinq": exact(pi.vols_possibles(5, (0,) * 5)),
    }
    for semaines, nom in _NOMBRES.items():
        macros[f"ModelePilotesRelax{nom}"] = exact(pi.relaxation(semaines))
    return macros


def _radios() -> dict[str, str]:
    plan = ra.plan()
    marges = [ra.marge_d_un_travailleur(i) for i in range(1, ra.SEMAINES + 1)]
    return {
        "ModeleRadiosMarges": point(marges),
        "ModeleRadiosTravailleurs": point(plan.travailleurs),
        "ModeleRadiosInstructeurs": point(plan.instructeurs),
        "ModeleRadiosEleves": point(plan.eleves),
        "ModeleRadiosProfit": str(plan.profit),
        "ModeleRadiosRecettes": str(ra.recettes_nettes(plan)),
        "ModeleRadiosSalaires": str(ra.salaires(plan)),
    }


def macros_de_la_modelisation() -> dict[str, str]:
    """Toutes les valeurs du document de modelisation."""
    return {
        **_personnel(),
        **_pilotes(),
        **_radios(),
        **macros_des_ateliers(),
        **macros_des_variantes(),
    }
