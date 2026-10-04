"""Chapitre XIV d'optimisation : relaxations, inegalites valides, priorites.

Oracles (`Optimisation/SLIDES/Slides_14_Introduction_a_la_programmation_`
`lineaire_en_variables_entieres.pdf`) : pages 299-300 pour les relaxations
generale et lagrangienne (`v_i(lambda) <= F_i` quel que soit `lambda >= 0`) ;
pages 293 a 295 pour la coupe -- `x1 <= 7/2` puis la contrainte `x1 <= 3`
de la fermeture convexe, sur l'illustration `min -3 x1 - 2 x2` dont
l'optimum entier vaut `-13` ; page 311 pour les priorites et la solution
connue d'avance.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.inegalites_valides import (
    est_une_coupe,
    est_une_inegalite_valide,
)
from polytech.optimisation.domain.priorites import variables_prioritaires
from polytech.optimisation.domain.relaxations import (
    relaxation_generale,
    relaxation_lagrangienne,
)
from polytech.optimisation.domain.resolution import resoudre
from polytech.optimisation.domain.separation_evaluation import (
    Cheminement,
    Procedure,
    explorer,
)
from polytech.optimisation.domain.sondage import Evaluation, Sondage

R = sp.Rational
INF, SUP = Sens.INFERIEUR, Sens.SUPERIEUR
BOITE = ((0, 6), (0, 6))
OPTIMUM_CONTINU = (R(25, 6), R(4, 3))
A, B, C = (
    Contrainte((-3, 4), INF, 6),
    Contrainte((2, 2), INF, 11),
    Contrainte((4, -5), INF, 10),
)


def _programme() -> ProgrammeGeneral:
    return ProgrammeGeneral(Optimisation.MIN, (-3, -2), (A, B, C))


def test_la_relaxation_generale_supprime_un_ensemble_de_contraintes() -> None:
    """Page 299 : `D' = {x | T1 x <= d1}`, les contraintes `T2 x <= d2` otees."""
    relaxe = relaxation_generale(_programme(), relachees=(1,))
    assert relaxe.contraintes == (A, C)
    assert relaxe.objectif == (-3, -2)


def test_la_relaxation_lagrangienne_fait_passer_la_contrainte_dans_l_objectif() -> None:
    """Page 300 : `min F(x) + lambda (T2 x - d2)` sur `D'`.

    Avec (b) `2 x1 + 2 x2 <= 11` et `lambda = 1` : `-3 x1 - 2 x2 + (2 x1 +
    2 x2 - 11) = -x1 - 11`.
    """
    relaxe, constante = relaxation_lagrangienne(_programme(), {1: 1})
    assert relaxe.contraintes == (A, C)
    assert relaxe.objectif == (-1, 0)
    assert constante == -11


def test_un_multiplicateur_nul_rend_la_relaxation_generale() -> None:
    relaxe, constante = relaxation_lagrangienne(_programme(), {1: 0})
    assert relaxe == relaxation_generale(_programme(), relachees=(1,))
    assert constante == 0


@pytest.mark.parametrize("multiplicateur", [R(1, 2), 1, R(5, 4), 2])
def test_la_relaxation_lagrangienne_minore_l_optimum_entier(
    multiplicateur: sp.Expr,
) -> None:
    """Page 300 : `v(lambda) <= F` quel que soit `lambda >= 0` ; ici `F = -13`."""
    relaxe, constante = relaxation_lagrangienne(_programme(), {0: multiplicateur})
    solution = resoudre(relaxe)
    assert solution.valeur is not None
    assert solution.valeur + constante <= -13


def test_une_contrainte_superieure_est_penalisee_dans_l_autre_sens() -> None:
    """`T x >= d` : c'est `d - T x`, positif quand elle est violee, que l'on pese."""
    programme = ProgrammeGeneral(
        Optimisation.MIN, (1, 1), (Contrainte((1, 2), SUP, 4),)
    )
    relaxe, constante = relaxation_lagrangienne(programme, {0: 3})
    assert relaxe.objectif == (1 - 3, 1 - 6)
    assert constante == 12
    assert relaxe.contraintes == ()


def test_un_multiplicateur_negatif_ou_une_maximisation_sont_refuses() -> None:
    with pytest.raises(ValueError, match="multiplicateur negatif"):
        relaxation_lagrangienne(_programme(), {1: -1})
    maximisation = ProgrammeGeneral(Optimisation.MAX, (1,), (Contrainte((1,), INF, 1),))
    with pytest.raises(ValueError, match="s'ecrit pour une minimisation"):
        relaxation_lagrangienne(maximisation, {0: 1})


def test_une_egalite_ne_se_relache_pas_avec_un_multiplicateur_signe() -> None:
    programme = ProgrammeGeneral(
        Optimisation.MIN, (1,), (Contrainte((1,), Sens.EGAL, 1),)
    )
    with pytest.raises(ValueError, match="egalite"):
        relaxation_lagrangienne(programme, {0: 1})


def test_une_inegalite_valide_ne_perd_aucun_point_entier() -> None:
    """Page 312 : « toutes les solutions entieres admissibles la verifient »."""
    assert est_une_inegalite_valide(Contrainte((1, 0), INF, 3), _programme(), BOITE)
    assert est_une_inegalite_valide(
        Contrainte((1, 0), INF, R(7, 2)), _programme(), BOITE
    )
    assert not est_une_inegalite_valide(Contrainte((1, 0), INF, 2), _programme(), BOITE)


def test_une_coupe_est_valide_et_ecarte_l_optimum_continu() -> None:
    """Page 295 : choisie pour ne pas etre verifiee par la solution du (LP)."""
    assert est_une_coupe(
        Contrainte((1, 0), INF, 3), _programme(), BOITE, OPTIMUM_CONTINU
    )
    assert not est_une_coupe(
        Contrainte((1, 0), INF, 5), _programme(), BOITE, OPTIMUM_CONTINU
    )
    assert not est_une_coupe(
        Contrainte((1, 0), INF, 2), _programme(), BOITE, OPTIMUM_CONTINU
    )


def test_chaque_coupe_releve_la_borne_de_la_relaxation() -> None:
    """Pages 293-294 : `D`, puis `x1 <= 7/2`, puis `x1 <= 3` -- sans depasser -13.

    `x1 <= 3` seule laisse `(3, 5/2)` et `-14` : il faut toutes les faces de
    la fermeture convexe pour atteindre l'optimum entier.
    """
    valeurs = [resoudre(_programme()).valeur]
    for borne in (R(7, 2), 3):
        coupe = Contrainte((1, 0), INF, borne)
        avec = ProgrammeGeneral(Optimisation.MIN, (-3, -2), (A, B, C, coupe))
        valeurs.append(resoudre(avec).valeur)
    assert valeurs == [R(-91, 6), R(-29, 2), -14]
    assert all(valeur <= -13 for valeur in valeurs)


def test_les_variables_de_plus_haute_priorite_sont_separees_les_premieres() -> None:
    """Page 311 : parmi les fractionnaires, celles de plus haute priorite."""
    priorites = {1: 1, 2: 5, 3: 5, 4: 9}
    assert variables_prioritaires(frozenset({1, 2, 3}), priorites) == frozenset({2, 3})
    assert variables_prioritaires(frozenset({1}), priorites) == frozenset({1})


def test_une_variable_sans_priorite_declaree_passe_apres_les_autres() -> None:
    assert variables_prioritaires(frozenset({1, 7}), {1: 1}) == frozenset({1})
    assert variables_prioritaires(frozenset({7, 8}), {1: 1}) == frozenset({7, 8})
    assert variables_prioritaires(frozenset(), {1: 1}) == frozenset()


def test_une_solution_connue_d_avance_elague_des_le_debut() -> None:
    """Page 311 a) : une bonne borne superieure `z^` permet d'elaguer rapidement."""
    arbre = {0: (1, 2)}

    def evaluer(noeud: int) -> Evaluation[int]:
        terminal = noeud != 0
        valeur = sp.Integer({0: 0, 1: 9, 2: 8}[noeud])
        return Evaluation(valeur, noeud if terminal else None, admissible=terminal)

    procedure = Procedure(evaluer, lambda n, _e: arbre[n], connue=(42, sp.Integer(5)))
    exploration = explorer(0, procedure, Cheminement.PROFONDEUR_D_ABORD)
    assert (exploration.solution, exploration.valeur) == (42, 5)
    assert [n.sondage for n in exploration.noeuds[1:]] == [Sondage.OPTIMALITE] * 2
    sans = explorer(
        0, Procedure(evaluer, lambda n, _e: arbre[n]), Cheminement.PROFONDEUR_D_ABORD
    )
    assert (sans.solution, sans.valeur) == (2, 8)


def test_une_solution_connue_sonde_la_racine_qu_elle_domine() -> None:
    def evaluer(_noeud: int) -> Evaluation[int]:
        return Evaluation(sp.Integer(6), None, admissible=False)

    procedure = Procedure(evaluer, lambda _n, _e: (1, 2), connue=(42, sp.Integer(5)))
    exploration = explorer(0, procedure, Cheminement.PROFONDEUR_D_ABORD)
    assert [n.sondage for n in exploration.noeuds] == [Sondage.DOMINANCE]
    assert exploration.solution == 42
