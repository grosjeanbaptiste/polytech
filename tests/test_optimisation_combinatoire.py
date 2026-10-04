"""Chapitre XIV.4.1 et XIV.4.3 d'optimisation : sac a dos, couverture, partition.

Source : `Optimisation/SLIDES/Slides_14_Introduction_a_la_programmation_`
`lineaire_en_variables_entieres.pdf`, pages 313 a 316. Le cours y donne
les formulations (KP), (SCP), (SPP) et le probleme d'emballage, sans
exemple chiffre : chaque modele est donc confronte a l'**enumeration** de
tous les choix binaires, chemin independant de la separation et
evaluation.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import pytest

from polytech.optimisation.domain.branch_and_bound import (
    resoudre_en_variables_entieres,
)
from polytech.optimisation.domain.combinatoire import (
    couverture,
    emballage,
    partition,
    sac_a_dos,
    sac_a_dos_est_non_trivial,
)
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.entiers import Nature, ProgrammeEntier
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    import sympy as sp

INF, SUP, EGA = Sens.INFERIEUR, Sens.SUPERIEUR, Sens.EGAL
VALEURS, POIDS, CAPACITE = (10, 7, 12, 4), (4, 3, 6, 2), 9
SOUS_ENSEMBLES = (
    frozenset({1, 2}),
    frozenset({2, 3}),
    frozenset({3, 4}),
    frozenset({1, 4}),
    frozenset({1, 2, 3}),
)
COUTS = (3, 2, 4, 2, 6)


def _par_enumeration(probleme: ProgrammeEntier) -> tuple[sp.Expr, set[tuple[int, ...]]]:
    """L'optimum et tous les points binaires qui le realisent."""
    n = len(probleme.programme.objectif)
    admissibles = [p for p in product((0, 1), repeat=n) if probleme.est_admissible(p)]
    sens = max if probleme.programme.optimisation is Optimisation.MAX else min
    optimum = sens(probleme.programme.valeur(p) for p in admissibles)
    return optimum, {p for p in admissibles if probleme.programme.valeur(p) == optimum}


def test_le_sac_a_dos_s_ecrit_avec_une_seule_contrainte_reelle() -> None:
    """Page 314 : `max sum p_j x_j`, `sum w_j x_j <= W`, `x_j = 0, 1`."""
    probleme = sac_a_dos(VALEURS, POIDS, CAPACITE)
    assert probleme.nature is Nature.ILP
    assert probleme.programme.optimisation is Optimisation.MAX
    assert probleme.programme.objectif == VALEURS
    assert probleme.programme.contraintes[0] == Contrainte(POIDS, INF, CAPACITE)
    assert probleme.programme.contraintes[1:] == tuple(
        Contrainte(tuple(int(i == j) for i in range(4)), INF, 1) for j in range(4)
    )


def test_le_sac_a_dos_resolu_par_separation_et_par_enumeration() -> None:
    """Objets 1, 2 et 4 : poids 9, valeur 21 -- mieux que 1 et 3 (10, hors capacite)."""
    probleme = sac_a_dos(VALEURS, POIDS, CAPACITE)
    optimum, points = _par_enumeration(probleme)
    exploration = resoudre_en_variables_entieres(probleme)
    assert optimum == 21
    assert points == {(1, 1, 0, 1)}
    assert exploration.valeur == 21
    assert exploration.solution == (1, 1, 0, 1)


def test_les_hypotheses_non_restrictives_du_sac_a_dos() -> None:
    """Page 314 : `w_j <= W` pour tout `j`, et `sum w_j > W`."""
    assert sac_a_dos_est_non_trivial(POIDS, CAPACITE)
    assert not sac_a_dos_est_non_trivial((4, 3, 10), 9)
    assert not sac_a_dos_est_non_trivial((4, 3), 9)


def test_des_valeurs_et_des_poids_en_nombre_different_sont_refuses() -> None:
    with pytest.raises(ValueError, match="poids pour"):
        sac_a_dos((1, 2), (1,), 3)


def test_la_couverture_demande_chaque_objet_au_moins_une_fois() -> None:
    """Page 315 : `sum t_ij x_j >= 1`, `t_ij = 1` si l'objet `i` est dans `P_j`."""
    probleme = couverture(4, SOUS_ENSEMBLES, COUTS)
    assert probleme.programme.optimisation is Optimisation.MIN
    assert probleme.programme.contraintes[:4] == (
        Contrainte((1, 0, 0, 1, 1), SUP, 1),
        Contrainte((1, 1, 0, 0, 1), SUP, 1),
        Contrainte((0, 1, 1, 0, 1), SUP, 1),
        Contrainte((0, 0, 1, 1, 0), SUP, 1),
    )


def test_la_couverture_de_cout_minimal() -> None:
    """`P_2` et `P_4` couvrent `{1, 2, 3, 4}` pour un cout de 4."""
    probleme = couverture(4, SOUS_ENSEMBLES, COUTS)
    optimum, points = _par_enumeration(probleme)
    assert optimum == 4
    assert points == {(0, 1, 0, 1, 0)}
    assert resoudre_en_variables_entieres(probleme).valeur == 4


def test_la_partition_demande_chaque_objet_exactement_une_fois() -> None:
    """Page 315 : une couverture dont les sous-ensembles sont disjoints."""
    probleme = partition(4, SOUS_ENSEMBLES, COUTS)
    assert all(c.sens is EGA for c in probleme.programme.contraintes[:4])
    optimum, points = _par_enumeration(probleme)
    assert optimum == 4
    assert points == {(0, 1, 0, 1, 0)}


def test_une_partition_coute_au_moins_autant_qu_une_couverture() -> None:
    """Toute partition est une couverture ; ici le recouvrement est moins cher."""
    sous_ensembles = (
        frozenset({1, 2}),
        frozenset({2, 3}),
        frozenset({3}),
        frozenset({1}),
    )
    couts = (1, 1, 5, 5)
    de_couverture, _ = _par_enumeration(couverture(3, sous_ensembles, couts))
    de_partition, _ = _par_enumeration(partition(3, sous_ensembles, couts))
    assert (de_couverture, de_partition) == (2, 6)


def test_l_emballage_choisit_des_sous_ensembles_disjoints_de_valeur_maximale() -> None:
    """Page 316 : `max sum v_j x_j`, `sum t_ij x_j <= 1`."""
    probleme = emballage(4, SOUS_ENSEMBLES, COUTS)
    assert probleme.programme.optimisation is Optimisation.MAX
    assert all(c.sens is INF for c in probleme.programme.contraintes)
    optimum, points = _par_enumeration(probleme)
    assert optimum == 7
    assert points == {(1, 0, 1, 0, 0)}
    assert resoudre_en_variables_entieres(probleme).valeur == 7


def test_un_objet_hors_de_l_ensemble_est_refuse() -> None:
    with pytest.raises(ValueError, match="objet hors de l'ensemble"):
        couverture(2, (frozenset({1, 3}),), (1,))


def test_un_sac_ou_tout_entre_exactement_est_trivial() -> None:
    """`sum w_j = W` : tous les objets entrent, il n'y a rien a choisir."""
    assert not sac_a_dos_est_non_trivial((4, 5), 9)


def test_une_maximisation_sans_solution_entiere_ne_rend_aucune_valeur() -> None:
    """Un emballage force a prendre deux sous-ensembles qui se recouvrent."""
    probleme = emballage(1, (frozenset({1}), frozenset({1})), (1, 1))
    force = Contrainte((1, 1), SUP, 2)
    contraintes = (*probleme.programme.contraintes, force)
    programme = ProgrammeGeneral(Optimisation.MAX, (1, 1), contraintes)
    impossible = ProgrammeEntier(programme, probleme.entieres)
    exploration = resoudre_en_variables_entieres(impossible)
    assert exploration.solution is None
    assert exploration.valeur is None
