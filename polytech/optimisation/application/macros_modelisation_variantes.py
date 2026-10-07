"""Macros de « la modelisation », parties II et III."""

from __future__ import annotations

from polytech.optimisation.application import modele_classiques as cl
from polytech.optimisation.application import modele_ordonnancement as od
from polytech.optimisation.application import modele_production as pr
from polytech.optimisation.application import modele_production_binaires as bi
from polytech.optimisation.application.formatage import exact, point
from polytech.optimisation.application.modele_production_croissante import (
    plus_petit_total_croissant,
)

_VARIANTES = ("I", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")
_CIRCUIT = (4, 5, 6)
_MACHINE = 3


def _ligne(cout: object, plan: tuple[object, ...]) -> str:
    return f"{cout} & ${point(plan)}$"


def _production() -> dict[str, str]:
    macros = {
        "ModeleProductionPlans": str(len(pr.plans())),
        "ModeleProductionLigneLibre": _ligne(*pr.meilleur("stock")),
        "ModeleProductionCroissant": str(plus_petit_total_croissant()),
        "ModeleProductionConstante": str(pr.avances_cumulees(pr.DEMANDES)),
        "ModeleProductionConstanteBis": str(pr.avances_cumulees(pr.ALTERNATIVE)),
        "ModeleProductionOrigine": str(pr.meilleur_pour(pr.DEMANDES)[0]),
        "ModeleProductionAlternative": str(pr.meilleur_pour(pr.ALTERNATIVE)[0]),
    }
    for variante in _VARIANTES:
        macros[f"ModeleProductionLigne{variante}"] = _ligne(*pr.meilleur(variante))
    for nom, corrige in (("Imprimee", False), ("Corrigee", True)):
        cout, delta, plan = bi.choix_de_la_demande(corrige=corrige)
        macros[f"ModeleProductionLigne{nom}"] = f"{cout} & {delta} & ${point(plan)}$"
    return macros


def _classiques() -> dict[str, str]:
    matrice = cl.matrice_d_affectation(3)
    cout, _ = cl.affectation_par_le_simplexe(cl.COUTS_D_ESSAI)
    return {
        "ModeleClassiquesMineurs": str(cl.mineurs(matrice)[0]),
        "ModeleClassiquesRang": str(matrice.rank()),
        "ModeleClassiquesAffectation": exact(cout),
        "ModeleClassiquesContradiction": str(cl.contradiction_du_sous_tour(_CIRCUIT)),
    }


def _ordonnancement() -> dict[str, str]:
    ordres = od.ordres_possibles()
    fin = ordres[0][0]
    return {
        "ModeleOrdoConflits": str(len(od.CONFLITS)),
        "ModeleOrdoPossibles": str(len(ordres)),
        "ModeleOrdoFin": str(fin),
        "ModeleOrdoOptimaux": str(sum(1 for f, _ in ordres if f == fin)),
        "ModeleOrdoCharge": str(od.charge_de_la_machine(_MACHINE)),
        "ModeleOrdoArrivee": str(od.arrivee_au_plus_tot(_MACHINE)),
    }


def macros_des_variantes() -> dict[str, str]:
    """Les valeurs des modelisations particulieres et des problemes classiques."""
    return {**_production(), **_classiques(), **_ordonnancement()}
