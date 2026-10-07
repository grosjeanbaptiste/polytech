"""Optimisation : la modelisation, les problemes classiques (planches 70 a 89).

Source : `Optimisation/EXS/Exs_Modelisation.pdf`. Trois affirmations des
planches se verifient par le calcul : la matrice d'affectation est
totalement unimodulaire, les variables `y` interdisent un sous-tour, et
le diagramme de Gantt de la planche 89 est optimal.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.application import modele_classiques as cl
from polytech.optimisation.application import modele_ordonnancement as od

# --- Le probleme d'affectation (planches 71 et 72) ---------------------------


def test_la_matrice_d_affectation_de_la_planche_72() -> None:
    """Pour n = 3 : trois lignes de ressources, trois d'activites, neuf colonnes."""
    attendue = sp.Matrix(
        [
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
        ]
    )
    assert cl.matrice_d_affectation(3) == attendue


def test_elle_est_totalement_unimodulaire() -> None:
    """Tous ses mineurs, de tous les ordres, valent -1, 0 ou 1."""
    mineurs, valeurs = cl.mineurs(cl.matrice_d_affectation(3))
    assert mineurs == 5004
    assert valeurs == {-1, 0, 1}


def test_une_matrice_qui_ne_l_est_pas() -> None:
    """Un cycle impair suffit a faire apparaitre un mineur egal a 2."""
    cycle = sp.Matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    assert 2 in cl.mineurs(cycle)[1]


def test_une_contrainte_d_affectation_est_de_trop() -> None:
    """Les six lignes ne sont pas independantes : leur rang est 5."""
    assert cl.matrice_d_affectation(3).rank() == 5


def test_la_relaxation_de_l_affectation_est_entiere() -> None:
    """Second chemin : le simplexe rend une permutation, au cout de la meilleure."""
    cout, affectation = cl.affectation_par_le_simplexe(cl.COUTS_D_ESSAI)
    assert cout == cl.meilleure_permutation(cl.COUTS_D_ESSAI)[0]
    assert set(affectation) == {0, 1}


# --- Le voyageur de commerce (planches 75 a 78) ------------------------------


def test_le_sous_tour_de_la_planche_78_est_interdit() -> None:
    """`y5 >= y4 + 1`, `y6 >= y5 + 1`, `y4 >= y6 + 1` : la somme donne `0 >= 3`."""
    assert cl.contradiction_du_sous_tour((4, 5, 6)) == 3
    assert not cl.ordre_possible(cl.SOLUTION_AP)


def test_le_tour_complet_admet_un_ordre() -> None:
    """La ville 1 est exemptee : les `y` peuvent croitre le long du tour."""
    assert cl.ordre_possible(cl.SOLUTION_TSP)
    assert cl.sous_tours(cl.SOLUTION_AP) == ((1, 2, 3), (4, 5, 6))
    assert cl.sous_tours(cl.SOLUTION_TSP) == ((1, 2, 3, 4, 5, 6),)


# --- L'ordonnancement (planches 87 a 89) -------------------------------------


def test_il_y_a_cinq_conflits_de_machine() -> None:
    """Planche 88 : cinq binaires, une par paire d'operations sur la meme machine."""
    assert len(od.CONFLITS) == 5


def test_huit_des_trente_deux_ordres_se_contredisent() -> None:
    assert len(od.ordres_possibles()) == 24


def test_le_diagramme_de_gantt_de_la_planche_89() -> None:
    """Fin a 99 ; les jalons 10, 30, 75 sur la machine 1 et 27, 55, 89 sur la 3."""
    fin, debuts = od.meilleur_ordonnancement()
    assert fin == 99
    assert debuts == od.GANTT
    assert od.est_realisable(od.GANTT)


def test_aucun_ordre_ne_finit_avant_99() -> None:
    fins = sorted(fin for fin, _ in od.ordres_possibles())
    assert fins[0] == 99
    assert fins.count(99) == 3


def test_une_borne_sans_enumeration() -> None:
    """La machine 3 a 72 minutes de travail et ne peut rien commencer avant 17."""
    assert od.charge_de_la_machine(3) == 72
    assert od.arrivee_au_plus_tot(3) == 17
    assert od.charge_de_la_machine(3) + od.arrivee_au_plus_tot(3) == 89


def test_un_ordonnancement_qui_chevauche_est_refuse() -> None:
    chevauche = dict(od.GANTT) | {(1, 1): 0}
    assert not od.est_realisable(chevauche)
    presse = dict(od.GANTT) | {(2, 3): 20, (3, 3): 60, (1, 3): 90}
    assert not od.est_realisable(presse)
    with pytest.raises(KeyError, match="machine inconnue : "):
        od.charge_de_la_machine(4)
