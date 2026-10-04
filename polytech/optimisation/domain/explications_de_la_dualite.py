"""Ecrire un dual, regle par regle, comme la page 91 les enonce.

`dual.dual_de` rend le probleme dual ; ce module rend le **pourquoi** de
chacune de ses lignes : d'ou vient le sens d'optimisation, pourquoi telle
variable duale est non negative et telle autre de signe quelconque,
pourquoi telle contrainte duale est une egalite. Un corrige peut citer
chaque etape telle quelle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.optimisation.domain.contraintes import Optimisation, Sens
from polytech.optimisation.domain.dual import dual_de, sous_la_convention
from polytech.optimisation.domain.ecriture import ecrire_contrainte, ecrire_lineaire
from polytech.raisonnement.domain.etapes import Etape, Raisonnement

if TYPE_CHECKING:
    from polytech.optimisation.domain.formulation import ProgrammeGeneral

SOURCE: Final = (
    "Optimisation/SLIDES/"
    "Slides_04_Definitions_proprietes_et_interpretation_de_la_dualite.pdf"
)
REFERENCE: Final = f"{SOURCE}, IV.1, p. 91-92"
NOMS: Final = {Optimisation.MIN: "minimisation", Optimisation.MAX: "maximisation"}


def _convention(primal: ProgrammeGeneral, ecrit: ProgrammeGeneral) -> list[Etape]:
    voulu = "<=" if primal.optimisation is Optimisation.MAX else ">="
    return [
        Etape(
            f"la contrainte {i} est retournee : {ecrire_contrainte(apres, 'x')}",
            f"par convention, les inegalites d'une {NOMS[primal.optimisation]} "
            f"s'ecrivent {voulu}",
            REFERENCE,
        )
        for i, (avant, apres) in enumerate(
            zip(primal.contraintes, ecrit.contraintes, strict=True), start=1
        )
        if avant != apres
    ]


def _variables_duales(ecrit: ProgrammeGeneral) -> list[Etape]:
    etapes = []
    for i, contrainte in enumerate(ecrit.contraintes, start=1):
        egalite = contrainte.sens is Sens.EGAL
        etapes.append(
            Etape(
                f"u_{i} est de signe quelconque" if egalite else f"u_{i} >= 0",
                f"la contrainte primale {i} est une "
                f"{'egalite' if egalite else 'inegalite'}",
                REFERENCE,
            )
        )
    return etapes


def _contraintes_duales(ecrit: ProgrammeGeneral, dual: ProgrammeGeneral) -> list[Etape]:
    etapes = []
    for j, contrainte in enumerate(dual.contraintes, start=1):
        libre = j - 1 in ecrit.libres
        nature, raison = (
            ("egalite", "de signe quelconque")
            if libre
            else ("inegalite", "soumise a une condition de non-negativite")
        )
        etapes.append(
            Etape(
                f"la contrainte duale {j} est une {nature} : "
                f"{ecrire_contrainte(contrainte, 'u')}",
                f"la variable primale x_{j} est {raison} ; ses coefficients "
                "t_ij deviennent ceux des u_i, son cout c_j le second membre",
                REFERENCE,
            )
        )
    return etapes


def expliquer_le_dual(primal: ProgrammeGeneral) -> Raisonnement:
    """Le dual du programme, chaque ligne rapportee a la regle qui la donne."""
    ecrit = sous_la_convention(primal)
    dual = dual_de(primal)
    objectif = ecrire_lineaire(dual.objectif, "u")
    sens = Etape(
        f"le dual est une {NOMS[dual.optimisation]}",
        f"le primal est une {NOMS[primal.optimisation]}",
        REFERENCE,
    )
    economique = Etape(
        f"la fonction economique du dual est {objectif}",
        "les seconds membres des contraintes primales deviennent les "
        "coefficients de la fonction economique du dual",
        REFERENCE,
    )
    etapes = (
        *_convention(primal, ecrit),
        sens,
        *_variables_duales(ecrit),
        *_contraintes_duales(ecrit, dual),
        economique,
    )
    conclusion = (
        f"{dual.optimisation.value} {objectif} sous {len(dual.contraintes)} "
        f"contraintes, {len(dual.objectif)} variables duales"
    )
    return Raisonnement("Quel est le probleme dual ?", etapes, conclusion)
