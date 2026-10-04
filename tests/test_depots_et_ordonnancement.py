"""Chapitre XIV.4.4 et XIV.4.5 d'optimisation : implantation de depots, production.

Source : `Optimisation/SLIDES/Slides_14_Introduction_a_la_programmation_`
`lineaire_en_variables_entieres.pdf`, pages 316 a 319. Le cours donne les
formulations (UFLP), (CFLP), le probleme des machines paralleles et le
« flow shop » de permutation, sans exemple chiffre : chaque modele est
confronte a une enumeration ou a une simulation, chemins independants.
"""

from __future__ import annotations

from itertools import permutations, product

import pytest
import sympy as sp

from polytech.optimisation.domain.branch_and_bound import (
    resoudre_en_variables_entieres,
)
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.depots import (
    implantation_avec_capacites,
    implantation_sans_capacite,
)
from polytech.optimisation.domain.entiers import Nature
from polytech.optimisation.domain.ordonnancement import (
    duree_totale,
    flow_shop_de_permutation,
    machines_paralleles,
)

R = sp.Rational
INF, SUP, EGA = Sens.INFERIEUR, Sens.SUPERIEUR, Sens.EGAL
GAINS = ((6, 2), (5, 4), (1, 7))
COUTS_FIXES = (4, 9)
DUREES = ((3, 2), (1, 4), (2, 2))


def _ordre(solution: tuple[sp.Expr, ...], taches: int) -> tuple[int, ...]:
    """L'ordre lu sur les `x_ij` : la tache placee en chaque position."""
    return tuple(
        next(i + 1 for i in range(taches) if solution[i * taches + j] == 1)
        for j in range(taches)
    )


def _gain_net(ouverts: tuple[int, ...]) -> int:
    """Chaque client va au depot ouvert qui rapporte le plus."""
    if not any(ouverts):
        return -(10**6)
    recette = sum(
        max(g for g, o in zip(ligne, ouverts, strict=True) if o) for ligne in GAINS
    )
    return recette - sum(f for f, o in zip(COUTS_FIXES, ouverts, strict=True) if o)


def test_l_implantation_sans_capacite_s_ecrit_comme_page_316() -> None:
    """`max sum g_ij y_ij - sum f_j x_j`, (1) `sum_j y_ij = 1`, (2) `y_ij <= x_j`."""
    probleme = implantation_sans_capacite(GAINS, COUTS_FIXES)
    programme = probleme.programme
    assert programme.optimisation is Optimisation.MAX
    assert programme.objectif == (-4, -9, 6, 2, 5, 4, 1, 7)
    assert programme.contraintes[0] == Contrainte((0, 0, 1, 1, 0, 0, 0, 0), EGA, 1)
    assert programme.contraintes[3] == Contrainte((-1, 0, 1, 0, 0, 0, 0, 0), INF, 0)
    assert programme.contraintes[4] == Contrainte((0, -1, 0, 1, 0, 0, 0, 0), INF, 0)
    assert probleme.nature is Nature.ILP


def test_l_implantation_sans_capacite_par_separation_et_par_enumeration() -> None:
    """Le depot 1 seul : `6 + 5 + 1 - 4 = 8` ; les deux : `6 + 5 + 7 - 13 = 5`."""
    par_enumeration = max(_gain_net(o) for o in product((0, 1), repeat=2))
    exploration = resoudre_en_variables_entieres(
        implantation_sans_capacite(GAINS, COUTS_FIXES)
    )
    assert par_enumeration == 8
    assert exploration.valeur == 8
    assert exploration.solution is not None
    assert exploration.solution[:2] == (1, 0)


def test_des_gains_et_des_couts_fixes_incoherents_sont_refuses() -> None:
    with pytest.raises(ValueError, match="gains pour"):
        implantation_sans_capacite(((1, 2, 3),), (4, 9))


def test_l_implantation_avec_capacites_est_un_probleme_mixte() -> None:
    """Page 316 : `sum_j y_ij = d_i`, `sum_i y_ij <= q_j x_j`, `y_ij >= 0` continue."""
    probleme = implantation_avec_capacites(GAINS, COUTS_FIXES, (2, 1, 3), (4, 5))
    programme = probleme.programme
    assert probleme.nature is Nature.MILP
    assert probleme.entieres == frozenset({0, 1})
    assert programme.contraintes[0] == Contrainte((0, 0, 1, 1, 0, 0, 0, 0), EGA, 2)
    assert programme.contraintes[3] == Contrainte((-4, 0, 1, 0, 1, 0, 1, 0), INF, 0)
    assert programme.contraintes[4] == Contrainte((0, -5, 0, 1, 0, 1, 0, 1), INF, 0)


def test_l_implantation_avec_capacites_oblige_a_ouvrir_les_deux_depots() -> None:
    """Demande totale 6 pour des capacites 4 et 5 : aucun depot ne suffit seul.

    A la main : le client 3 (gain 7 contre 1) prend ses 3 unites au depot 2,
    le client 1 (6 contre 2) ses 2 unites au depot 1, le client 2 (5 contre
    4) son unite au depot 1 : `12 + 5 + 21 - 13 = 25`.
    """
    probleme = implantation_avec_capacites(GAINS, COUTS_FIXES, (2, 1, 3), (4, 5))
    exploration = resoudre_en_variables_entieres(probleme)
    assert exploration.valeur == 25
    assert exploration.solution == (1, 1, 2, 0, 1, 0, 0, 3)


def test_les_machines_paralleles_minimisent_la_machine_la_plus_chargee() -> None:
    """Page 317 : `min y`, `sum_i p_i x_ik <= y` ; ici 3 + 3 et 2 + 2 + 2."""
    durees = (3, 3, 2, 2, 2)
    probleme = machines_paralleles(durees, machines=2)
    assert probleme.nature is Nature.MILP
    assert probleme.programme.objectif == (0,) * 10 + (1,)
    par_enumeration = min(
        max(
            sum(p for p, k in zip(durees, choix, strict=True) if k == m) for m in (0, 1)
        )
        for choix in product((0, 1), repeat=5)
    )
    assert par_enumeration == 6
    assert resoudre_en_variables_entieres(probleme).valeur == 6


def test_chaque_tache_est_affectee_a_une_machine_et_une_seule() -> None:
    """Le cours laisse cette contrainte implicite ; sans elle, `y = 0`."""
    probleme = machines_paralleles((3, 2), machines=2)
    egalites = [c for c in probleme.programme.contraintes if c.sens is EGA]
    assert egalites == [
        Contrainte((1, 1, 0, 0, 0), EGA, 1),
        Contrainte((0, 0, 1, 1, 0), EGA, 1),
    ]
    charges = [c for c in probleme.programme.contraintes if c.second_membre == 0]
    assert charges == [
        Contrainte((3, 0, 2, 0, -1), INF, 0),
        Contrainte((0, 3, 0, 2, -1), INF, 0),
    ]


def test_la_duree_totale_d_un_ordre_se_simule_machine_par_machine() -> None:
    """Deux machines en serie : la tache attend que la machine suivante se libere."""
    assert duree_totale(DUREES, (1, 2, 3)) == 11
    assert duree_totale(DUREES, (2, 1, 3)) == 9
    assert duree_totale(((3, 2),), (1,)) == 5


def test_le_flow_shop_retrouve_le_meilleur_ordre_par_enumeration() -> None:
    """Trois taches, deux machines : la regle de Johnson donne l'ordre 2, 3, 1."""
    par_enumeration = min(duree_totale(DUREES, o) for o in permutations((1, 2, 3)))
    probleme, constante = flow_shop_de_permutation(DUREES)
    exploration = resoudre_en_variables_entieres(probleme)
    assert par_enumeration == 9
    assert constante == 2 + 4 + 2
    assert exploration.valeur == 1
    assert exploration.solution is not None
    assert duree_totale(DUREES, _ordre(exploration.solution, 3)) == 9


def test_sans_ses_conditions_initiales_le_modele_du_cours_sous_estime_la_duree() -> (
    None
):
    """Page 319 : rien n'y fixe `y_1k` ni `z_1k`.

    L'inactivite de la machine `m` avant la premiere tache est alors
    gratuite : le modele annonce 8, pour un ordre qui dure en realite 11.
    """
    incomplet, constante = flow_shop_de_permutation(DUREES, conditions_initiales=False)
    exploration = resoudre_en_variables_entieres(incomplet)
    assert exploration.solution is not None
    assert exploration.valeur == 0
    assert constante == 8
    assert duree_totale(DUREES, _ordre(exploration.solution, 3)) == 11


def test_le_flow_shop_a_trois_machines_s_accorde_avec_la_simulation() -> None:
    """La relation de la figure XIV.12 vaut pour chaque couple de machines voisines."""
    durees = ((2, 3, 1), (4, 1, 3), (1, 2, 2))
    par_enumeration = min(duree_totale(durees, o) for o in permutations((1, 2, 3)))
    probleme, constante = flow_shop_de_permutation(durees)
    exploration = resoudre_en_variables_entieres(probleme)
    assert exploration.solution is not None
    assert exploration.valeur == par_enumeration - constante == 11 - 6
    assert duree_totale(durees, _ordre(exploration.solution, 3)) == 11
