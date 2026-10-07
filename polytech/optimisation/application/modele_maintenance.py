"""Modelisation, probleme VIIa : la maintenance preventive de cinq ateliers.

Chaque atelier demande des ouvriers trois semaines de suite, et peut
commencer dans une fenetre de dates. `delta_ij` vaut 1 si l'atelier `i`
commence la semaine `j`. On choisit les dates, puis les embauches et
les licenciements, pour payer le moins possible.
"""

from __future__ import annotations

import itertools
from functools import cache
from typing import TYPE_CHECKING, Final

from polytech.optimisation.application.modele_variables import Variables, numerotees
from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    import sympy as sp

SEMAINES: Final = 8
CHARGES: Final = {1: (2, 2, 1), 2: (3, 2, 1), 3: (4, 2, 2), 4: (1, 1, 3), 5: (2, 2, 2)}
FENETRES: Final = {1: (1, 4), 2: (1, 3), 3: (2, 5), 4: (2, 6), 5: (3, 5)}
"""Planche 28 : les ouvriers des trois semaines ; le debut au plus tot, au plus tard."""
AU_DEPART, A_LA_FIN = 7, 9
SALAIRE, EMBAUCHE, LICENCIEMENT = 600, 300, 1200
COEFFICIENT_IMPRIME_PLANCHE_29: Final = {(1, 2): 3}
"""La ligne `T2` de la planche 29 ecrit `3 d12` ; la planche 31 ecrit `2 d12`."""
OUBLIE_PAR_LES_PLANCHES: Final = (3, 2)
"""Les planches 30 a 37 ne font commencer A3 qu'en semaine 3 : `d32` n'y pese rien."""
Debuts = tuple[int, ...]


@cache
def plannings() -> tuple[Debuts, ...]:
    """Tous les choix d'une semaine de debut par atelier, dans sa fenetre."""
    fenetres = (range(tot, tard + 1) for tot, tard in FENETRES.values())
    return tuple(itertools.product(*fenetres))


def charge(atelier: int, debut: int, semaine: int) -> int:
    """Les ouvriers que l'atelier occupe cette semaine-la s'il commence a `debut`."""
    rang = semaine - debut
    return CHARGES[atelier][rang] if 0 <= rang < len(CHARGES[atelier]) else 0


def besoins(debuts: Debuts, *, imprimes: bool = False) -> tuple[int, ...]:
    """Les ouvriers necessaires chacune des huit semaines.

    Avec `imprimes`, ce que les lignes des planches comptent : rien pour
    l'atelier A3 s'il commence en semaine 2.
    """
    comptes = [
        (a, d)
        for a, d in zip(CHARGES, debuts, strict=True)
        if not (imprimes and (a, d) == OUBLIE_PAR_LES_PLANCHES)
    ]
    return tuple(
        sum(charge(a, d, semaine) for a, d in comptes)
        for semaine in range(1, SEMAINES + 1)
    )


def ligne_des_besoins(semaine: int) -> dict[tuple[int, int], int]:
    """Le second membre de `T_semaine >= ...` : un coefficient par `delta_ij` actif."""
    return {
        (atelier, debut): charge(atelier, debut, semaine)
        for atelier, (tot, tard) in FENETRES.items()
        for debut in range(tot, tard + 1)
        if charge(atelier, debut, semaine)
    }


def ligne_imprimee(semaine: int) -> dict[tuple[int, int], int]:
    """La meme ligne telle que les planches 30 a 37 l'ecrivent."""
    ligne = ligne_des_besoins(semaine)
    return {k: v for k, v in ligne.items() if k != OUBLIE_PAR_LES_PLANCHES}


def _mouvement(avant: int, apres: int) -> int:
    """Ce que coute de passer d'un effectif au suivant."""
    ecart = apres - avant
    return EMBAUCHE * ecart if ecart > 0 else -LICENCIEMENT * ecart


def cout_du_personnel(les_besoins: tuple[int, ...]) -> int:
    """Le cout minimal des neuf semaines, par recurrence sur l'effectif."""
    plafond = max(*les_besoins, AU_DEPART, A_LA_FIN)

    @cache
    def suite(semaine: int, effectif: int) -> int:
        if semaine == SEMAINES:
            return SALAIRE * A_LA_FIN + _mouvement(effectif, A_LA_FIN)
        return min(
            SALAIRE * e + _mouvement(effectif, e) + suite(semaine + 1, e)
            for e in range(les_besoins[semaine], plafond + 1)
        )

    return suite(0, AU_DEPART)


def cout_par_le_simplexe(les_besoins: tuple[int, ...]) -> sp.Expr:
    """Le meme cout, par le modele de la planche 29 une fois les dates fixees."""
    semaines = SEMAINES + 1
    variables = Variables(tuple(n for p in "TEL" for n in numerotees(p, semaines)))
    ligne = variables.ligne
    couts = {"T": SALAIRE, "E": EMBAUCHE, "L": LICENCIEMENT}
    objectif = tuple(couts[nom[0]] for nom in variables.noms)
    contraintes = [Contrainte(ligne({f"T{semaines}": 1}), Sens.EGAL, A_LA_FIN)]
    for i in range(1, semaines + 1):
        bilan = {f"T{i}": 1, f"E{i}": -1, f"L{i}": 1}
        if i > 1:
            bilan[f"T{i - 1}"] = -1
        depart = AU_DEPART if i == 1 else 0
        contraintes.append(Contrainte(ligne(bilan), Sens.EGAL, depart))
        if i <= SEMAINES:
            besoin = Contrainte(ligne({f"T{i}": 1}), Sens.SUPERIEUR, les_besoins[i - 1])
            contraintes.append(besoin)
    modele = ProgrammeGeneral(Optimisation.MIN, objectif, tuple(contraintes))
    return optimum_fini(modele)[0]


@cache
def classement(*, imprimes: bool = False) -> tuple[tuple[int, int, Debuts], ...]:
    """Chaque planning avec son cout et sa pointe, du moins cher au plus cher."""
    lignes = []
    for debuts in plannings():
        demande = besoins(debuts, imprimes=imprimes)
        lignes.append((cout_du_personnel(demande), max(demande), debuts))
    return tuple(sorted(lignes))


def meilleur_planning() -> tuple[int, Debuts]:
    """Le cout minimal et les dates de debut qui l'atteignent."""
    cout, _, debuts = classement()[0]
    return cout, debuts


def plus_petite_pointe() -> tuple[int, tuple[Debuts, ...]]:
    """Probleme VIIb : le plus petit effectif constant, et les plannings qu'il sert."""
    pointe = min(p for _, p, _ in classement())
    return pointe, tuple(d for _, p, d in classement() if p == pointe)
