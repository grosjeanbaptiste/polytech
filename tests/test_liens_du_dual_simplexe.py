"""Les deux ecritures de l'algorithme dual doivent rendre le meme optimum.

`DualSimplexe` (tableau `B^-1 (A, I | b)`, sans ligne de cout) est
anterieur a la lecture page par page du chapitre V ; `phase_deux_duale`
suit le cours tableau par tableau. Confrontes sur l'illustration V.2.4 a)
(`Optimisation/SLIDES/Slides_05_Lalgorithme_dual_simplexe.pdf`, pages
117-118), ils doivent s'accorder sur le sommet optimal `(9/2, 3/2, 0)`.

Ce lien a trouve un defaut : `DualSimplexe` choisissait la colonne
entrante au plus grand `|a_lj|`, sans regarder les couts reduits. Il
retablissait l'admissibilite, mais en `(21/4, 0, 3/4)`, ou `z = 33` au
lieu de `63/2` -- un point admissible, non l'optimum.
"""

from __future__ import annotations

import sympy as sp

from polytech.optimisation.domain.algorithme_dual import phase_deux_duale
from polytech.optimisation.domain.contrainte_artificielle import (
    methode_de_la_contrainte_artificielle,
)
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.dual_simplexe import DualSimplexe
from polytech.optimisation.domain.forme_standard import ProgrammeLineaire
from polytech.optimisation.domain.formes import forme_standard
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.resolution import resoudre
from polytech.optimisation.domain.solution_duale import bases_du_couple
from polytech.optimisation.domain.tableau import TableauSimplexe

R = sp.Rational


def _programme() -> ProgrammeLineaire:
    """`max -6 x1 - 3 x2 - 2 x3`, les deux `>=` retournees en `<=`."""
    return ProgrammeLineaire(
        objectif=sp.Matrix([-6, -3, -2]),
        contraintes=sp.Matrix([[-1, -1, -1], [-2, 0, 2]]),
        seconds_membres=sp.Matrix([-6, -9]),
    )


def _par_l_ancienne_ecriture() -> DualSimplexe:
    programme = _programme()
    tableau = programme.matrice_etendue().row_join(programme.seconds_membres)
    return DualSimplexe(programme, (3, 4), tableau).retablir_la_faisabilite()


def test_les_deux_ecritures_rendent_le_sommet_optimal_du_cours() -> None:
    coefficients = sp.Matrix([[0, -6, -3, -2], [-6, -1, -1, -1], [-9, -2, 0, 2]])
    depart = TableauSimplexe((4, 5), (1, 2, 3), coefficients)
    du_cours = phase_deux_duale(depart, Optimisation.MIN).tableaux[-1]
    assert list(du_cours.solution(3)) == [R(9, 2), R(3, 2), 0]
    assert list(_par_l_ancienne_ecriture().point()) == [R(9, 2), R(3, 2), 0]


def test_la_colonne_entrante_suit_le_rapport_des_couts_reduits() -> None:
    """Page 117, second tableau : `x2` entre (`3/1 = min(6, 3, 4)`), non `x3`."""
    apres_un_pivot = _par_l_ancienne_ecriture_un_pas()
    assert apres_un_pivot.colonne_entrante(0) == 1


def _par_l_ancienne_ecriture_un_pas() -> DualSimplexe:
    programme = _programme()
    tableau = programme.matrice_etendue().row_join(programme.seconds_membres)
    return DualSimplexe(programme, (3, 4), tableau).pivoter()


def test_l_algorithme_dual_chemine_sur_les_sommets_9_3_1_du_polyedre_dual() -> None:
    """Page 118 : « sur le polyedre D2, au cheminement des points 9-3-1 ».

    Les bases visitees sont, dans le tableau de la page 110 du chapitre IV,
    les numeros 9, 3 et 1 -- toutes duales admissibles, la derniere seule
    etant aussi primale admissible.
    """
    primal = ProgrammeGeneral(
        Optimisation.MIN,
        (6, 3, 2),
        (
            Contrainte((1, 1, 1), Sens.SUPERIEUR, 6),
            Contrainte((2, 0, -2), Sens.SUPERIEUR, 9),
        ),
    )
    bases = bases_du_couple(forme_standard(primal))
    numeros = {frozenset(b.indices): n for n, b in enumerate(bases, start=1)}
    coefficients = sp.Matrix([[0, -6, -3, -2], [-6, -1, -1, -1], [-9, -2, 0, 2]])
    depart = TableauSimplexe((4, 5), (1, 2, 3), coefficients)
    tableaux = phase_deux_duale(depart, Optimisation.MIN).tableaux
    visitees = [numeros[frozenset(t.base)] for t in tableaux]
    assert visitees == [9, 3, 1]
    assert all(bases[n - 1].duale_admissible for n in visitees)
    assert [bases[n - 1].primale_admissible for n in visitees] == [False, False, True]


def test_l_algorithme_dual_et_la_resolution_generale_s_accordent() -> None:
    """Deux chemins -- dual, et variables artificielles puis primal -- un optimum."""
    primal = ProgrammeGeneral(
        Optimisation.MIN,
        (6, 3, -2),
        (
            Contrainte((1, 1, 1), Sens.SUPERIEUR, 6),
            Contrainte((2, 0, -2), Sens.SUPERIEUR, 9),
        ),
    )
    coefficients = sp.Matrix([[0, -6, -3, 2], [-6, -1, -1, -1], [-9, -2, 0, 2]])
    depart = TableauSimplexe((4, 5), (1, 2, 3), coefficients)
    dual = methode_de_la_contrainte_artificielle(depart, Optimisation.MIN)
    general = resoudre(primal)
    assert dual.tableaux[-1].valeur == general.valeur == 30
    assert tuple(dual.tableaux[-1].solution(3)) == general.point
