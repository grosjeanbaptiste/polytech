"""Chapitre XIV.2 d'optimisation : la procedure par separation et evaluation.

Oracles (`Optimisation/SLIDES/Slides_14_Introduction_a_la_programmation_`
`lineaire_en_variables_entieres.pdf`) : pages 296 a 303 pour les regles
(conservation, minoration, coincidence, les trois tests de sondage, les
deux cheminements) ; page 290 pour les optimums que l'exploration doit
retrouver -- (ILP) `(3, 2)` pour `-13`, (MILP) `(4, 3/2)` pour `-15` ;
figure XIV.8 pour l'ordre d'un cheminement en profondeur d'abord.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.branch_and_bound import (
    evaluer_par_relaxation_lineaire,
    resoudre_en_variables_entieres,
    separer_le_noeud,
    separer_sur_une_variable,
)
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.entiers import ProgrammeEntier, points_entiers
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.separation_evaluation import (
    Cheminement,
    Procedure,
    explorer,
)
from polytech.optimisation.domain.sondage import Evaluation, Sondage, sonder

R = sp.Rational
INF, SUP = Sens.INFERIEUR, Sens.SUPERIEUR
BOITE = ((0, 6), (0, 6))


def _probleme(entieres: frozenset[int]) -> ProgrammeEntier:
    contraintes = (
        Contrainte((-3, 4), INF, 6),
        Contrainte((2, 2), INF, 11),
        Contrainte((4, -5), INF, 10),
    )
    programme = ProgrammeGeneral(Optimisation.MIN, (-3, -2), contraintes)
    return ProgrammeEntier(programme, entieres)


def _arbre_de_la_figure_xiv_8() -> dict[int, tuple[int, ...]]:
    """Les fils de chaque noeud : 0 -> (1, 6), 1 -> (2, 3), 3 -> (4, 5), 6 -> (7, 8)."""
    return {0: (1, 6), 1: (2, 3), 3: (4, 5), 6: (7, 8)}


def test_l_evaluation_de_la_racine_est_la_relaxation_lineaire() -> None:
    evaluation = evaluer_par_relaxation_lineaire(_probleme(frozenset({0, 1})))
    assert evaluation.valeur == R(-91, 6)
    assert evaluation.solution == (R(25, 6), R(4, 3))
    assert not evaluation.admissible


def test_un_probleme_relaxe_impossible_n_a_pas_de_valeur() -> None:
    """Test alpha : `R = vide`, donc `S = vide`."""
    programme = ProgrammeGeneral(Optimisation.MIN, (1,), (Contrainte((1,), INF, -1),))
    evaluation = evaluer_par_relaxation_lineaire(
        ProgrammeEntier(programme, frozenset({0}))
    )
    assert evaluation.valeur is None
    assert evaluation.solution is None


def test_un_probleme_relaxe_non_borne_est_une_erreur() -> None:
    programme = ProgrammeGeneral(Optimisation.MIN, (-1,), (Contrainte((1,), SUP, 0),))
    with pytest.raises(ValueError, match="relaxation lineaire non bornee"):
        evaluer_par_relaxation_lineaire(ProgrammeEntier(programme, frozenset({0})))


def test_la_separation_respecte_la_regle_de_conservation() -> None:
    """XIV.2.1 b) : aucune solution eliminee, et les deux fils sont disjoints."""
    parent = _probleme(frozenset({0, 1}))
    gauche, droite = separer_sur_une_variable(parent, variable=0, valeur=R(25, 6))
    assert gauche.programme.contraintes[-1] == Contrainte((1, 0), INF, 4)
    assert droite.programme.contraintes[-1] == Contrainte((1, 0), SUP, 5)
    a, b = (points_entiers(p.programme, BOITE) for p in (gauche, droite))
    assert a | b == points_entiers(parent.programme, BOITE)
    assert a & b == frozenset()


def test_la_solution_du_parent_ne_verifie_aucune_des_deux_contraintes() -> None:
    """XIV.3.2 : « la solution ne verifie aucune de ces deux contraintes »."""
    gauche, droite = separer_sur_une_variable(
        _probleme(frozenset({0, 1})), variable=0, valeur=R(25, 6)
    )
    point = (R(25, 6), R(4, 3))
    assert not gauche.programme.contraintes[-1].est_satisfaite(point)
    assert not droite.programme.contraintes[-1].est_satisfaite(point)


@pytest.mark.parametrize("cheminement", list(Cheminement))
def test_l_exploration_retrouve_l_optimum_entier(cheminement: Cheminement) -> None:
    exploration = resoudre_en_variables_entieres(
        _probleme(frozenset({0, 1})), cheminement
    )
    assert exploration.solution == (3, 2)
    assert exploration.valeur == -13


@pytest.mark.parametrize("cheminement", list(Cheminement))
def test_l_exploration_retrouve_l_optimum_mixte(cheminement: Cheminement) -> None:
    """Page 290 : `x1` seule entiere, `(4, 3/2)` pour `-15`."""
    exploration = resoudre_en_variables_entieres(_probleme(frozenset({0})), cheminement)
    assert exploration.solution == (4, R(3, 2))
    assert exploration.valeur == -15


def test_l_evaluation_ne_decroit_jamais_d_un_noeud_a_ses_fils() -> None:
    """XIV.2.2 A : `v_i <= v_ik`."""
    exploration = resoudre_en_variables_entieres(_probleme(frozenset({0, 1})))
    valeurs = {n.chemin: n.evaluation for n in exploration.noeuds}
    for chemin, valeur in valeurs.items():
        if chemin and valeur is not None:
            assert valeurs[chemin[:-1]] <= valeur


def test_chaque_noeud_sonde_l_est_par_l_un_des_trois_tests() -> None:
    exploration = resoudre_en_variables_entieres(_probleme(frozenset({0, 1})))
    sondages = {n.sondage for n in exploration.noeuds if n.sondage is not None}
    assert sondages <= set(Sondage)
    assert Sondage.OPTIMALITE in sondages
    feuilles = [n for n in exploration.noeuds if n.sondage is not None]
    assert all(
        n.evaluation is None for n in feuilles if n.sondage is Sondage.IMPOSSIBILITE
    )


def test_un_probleme_sans_solution_entiere_ne_rend_rien() -> None:
    """`1/3 <= x <= 2/3` : tous les noeuds sont sondes par impossibilite."""
    programme = ProgrammeGeneral(
        Optimisation.MIN,
        (1,),
        (Contrainte((3,), SUP, 1), Contrainte((3,), INF, 2)),
    )
    exploration = resoudre_en_variables_entieres(
        ProgrammeEntier(programme, frozenset({0}))
    )
    assert exploration.solution is None
    assert exploration.valeur is None
    assert [n.sondage for n in exploration.noeuds[1:]] == [Sondage.IMPOSSIBILITE] * 2


def _evaluer_la_figure(noeud: int) -> Evaluation[int]:
    """Les feuilles 2, 4, 5, 7, 8 sont terminales ; 5 porte la meilleure valeur."""
    feuilles = {2: 9, 4: 8, 5: 7, 7: 9, 8: 9}
    if noeud in feuilles:
        return Evaluation(sp.Integer(feuilles[noeud]), noeud, admissible=True)
    return Evaluation(sp.Integer(0), None, admissible=False)


def test_le_cheminement_en_profondeur_suit_la_figure_xiv_8() -> None:
    """Toujours le sous-ensemble le plus a gauche, puis on remonte."""
    arbre = _arbre_de_la_figure_xiv_8()
    procedure = Procedure(_evaluer_la_figure, lambda n, _e: arbre[n])
    exploration = explorer(0, procedure, Cheminement.PROFONDEUR_D_ABORD)
    assert [n.probleme for n in exploration.noeuds] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert exploration.solution == 5
    assert exploration.valeur == 7


def test_la_dominance_elague_une_branche_qui_ne_peut_pas_faire_mieux() -> None:
    """Test gamma : `F^ <= v_i`. Le noeud 6, evalue a 7, n'est pas separe."""

    def evaluer(noeud: int) -> Evaluation[int]:
        if noeud == 6:
            return Evaluation(sp.Integer(7), None, admissible=False)
        return _evaluer_la_figure(noeud)

    arbre = _arbre_de_la_figure_xiv_8()
    procedure = Procedure(evaluer, lambda n, _e: arbre[n])
    exploration = explorer(0, procedure, Cheminement.PROFONDEUR_D_ABORD)
    assert [n.probleme for n in exploration.noeuds] == [0, 1, 2, 3, 4, 5, 6]
    assert exploration.noeuds[-1].sondage is Sondage.DOMINANCE


def test_la_meilleure_evaluation_d_abord_separe_le_noeud_actif_le_plus_prometteur() -> (
    None
):
    """XIV.2.3 beta : `v* = min v_i` sur les noeuds actifs."""
    arbre = {0: (1, 2), 1: (3, 4), 2: (5, 6)}
    valeurs = {0: 0, 1: 5, 2: 3, 3: 9, 4: 9, 5: 4, 6: 6}

    def evaluer(noeud: int) -> Evaluation[int]:
        terminal = noeud not in arbre
        solution = noeud if terminal else None
        return Evaluation(sp.Integer(valeurs[noeud]), solution, admissible=terminal)

    procedure = Procedure(evaluer, lambda n, _e: arbre[n])
    exploration = explorer(0, procedure, Cheminement.MEILLEURE_EVALUATION_D_ABORD)
    assert [n.probleme for n in exploration.noeuds] == [0, 1, 2, 5, 6]
    assert exploration.solution == 5
    assert exploration.valeur == 4
    assert exploration.noeuds[1].sondage is Sondage.DOMINANCE


def test_les_trois_tests_de_sondage_dans_leur_ordre() -> None:
    """Page 301 : impossibilite, optimalite, dominance ; sinon on separe."""
    impossible = Evaluation[int](None, None, admissible=False)
    entiere = Evaluation(sp.Integer(5), 1, admissible=True)
    fractionnaire = Evaluation[int](sp.Integer(5), None, admissible=False)
    assert sonder(impossible, None) is Sondage.IMPOSSIBILITE
    assert sonder(entiere, sp.Integer(3)) is Sondage.OPTIMALITE
    assert sonder(fractionnaire, sp.Integer(4)) is Sondage.DOMINANCE
    assert sonder(fractionnaire, sp.Integer(5)) is Sondage.DOMINANCE
    assert sonder(fractionnaire, sp.Integer(6)) is None
    assert sonder(fractionnaire, None) is None


def test_les_noeuds_sont_reperes_par_leurs_indices_depuis_la_racine() -> None:
    """Page 296 : niveau zero sans indice, niveau 1 un indice, niveau q : q indices."""
    arbre = _arbre_de_la_figure_xiv_8()
    procedure = Procedure(_evaluer_la_figure, lambda n, _e: arbre[n])
    exploration = explorer(0, procedure, Cheminement.PROFONDEUR_D_ABORD)
    chemins = {n.probleme: n.chemin for n in exploration.noeuds}
    assert chemins[0] == ()
    assert (chemins[1], chemins[6]) == ((1,), (2,))
    assert (chemins[4], chemins[5]) == ((1, 2, 1), (1, 2, 2))


def test_la_separation_porte_sur_la_premiere_variable_non_entiere() -> None:
    """A la racine `(25/6, 4/3)` : les deux sont fractionnaires, `x1` est separee."""
    exploration = resoudre_en_variables_entieres(_probleme(frozenset({0, 1})))
    premier_fils = exploration.noeuds[1].probleme
    assert premier_fils.programme.contraintes[-1] == Contrainte((1, 0), INF, 4)


def test_un_noeud_sans_solution_relaxee_ne_se_separe_pas() -> None:
    with pytest.raises(ValueError, match="noeud sans solution relaxee"):
        separer_le_noeud(
            _probleme(frozenset({0})), Evaluation(None, None, admissible=False)
        )


@pytest.mark.parametrize("minorant", [5, 6])
def test_un_minorant_bon_marche_sonde_sans_evaluer(minorant: int) -> None:
    """Page 307 : le test gamma des que `z_LP + P >= z^`, egalite comprise."""
    arbre = {0: (1, 2)}
    evalues: list[int] = []

    def evaluer(noeud: int) -> Evaluation[int]:
        evalues.append(noeud)
        terminal = noeud != 0
        valeur = sp.Integer(5 if terminal else 0)
        return Evaluation(valeur, noeud if terminal else None, admissible=terminal)

    procedure = Procedure(
        evaluer,
        lambda n, _e: arbre[n],
        minorer=lambda n: sp.Integer(minorant) if n == 2 else None,
    )
    exploration = explorer(0, procedure, Cheminement.PROFONDEUR_D_ABORD)
    assert evalues == [0, 1]
    assert exploration.noeuds[2].sondage is Sondage.DOMINANCE
    assert exploration.noeuds[2].evaluation == minorant


@pytest.mark.parametrize("cheminement", list(Cheminement))
def test_une_arborescence_sans_fin_est_une_erreur_et_non_une_attente(
    cheminement: Cheminement,
) -> None:
    """Regle de finitude (XIV.2.1 a) : une separation qui l'enfreint est arretee."""

    def evaluer(_noeud: int) -> Evaluation[int]:
        return Evaluation(sp.Integer(0), None, admissible=False)

    procedure = Procedure(evaluer, lambda n, _e: (n + 1, n + 2))
    with pytest.raises(RuntimeError, match="arborescence non elaguee"):
        explorer(0, procedure, cheminement, noeuds_maximaux=25)
