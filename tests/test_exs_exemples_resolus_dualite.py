"""Optimisation : les exemples resolus, parties II et III (dualite, algorithme dual).

Source : `Optimisation/EXS/Exs_Exemples_resolus.pdf`, planches 45 a 62. Le
probleme dual et son interpretation economique, l'algorithme dual, puis
la methode de la contrainte artificielle. Chaque resultat imprime est
recopie comme oracle et recalcule.
"""

from __future__ import annotations

import sympy as sp

from polytech.optimisation.application import exemples_dualite as du
from polytech.optimisation.domain.algorithme import Issue

Q = sp.Rational

# --- Planche 48 : ecrire un dual --------------------------------------------


def test_le_dual_de_la_planche_48() -> None:
    """Planche 48 : `min 7u1 - 4u2 + 5u3`, une inegalite, une egalite, `u3` libre."""
    objectif, lignes, libres = du.dual_de_l_illustration()
    assert objectif == (7, -4, 5)
    assert lignes == (((1, -3, 1), ">=", 1), ((0, 2, 1), "=", -4))
    assert libres == {3}


# --- Planches 50 a 52 : les bureaux -----------------------------------------


def test_les_bureaux_le_primal_et_le_dual() -> None:
    """Planche 52 : `x = (25, 75)`, `u = (25, 0, 25)`, et `z = w = 11250`."""
    assert du.optimum_des_bureaux() == (11250, (25, 75))
    assert du.prix_des_ressources() == (11250, (25, 0, 25))


def test_les_bureaux_la_ressource_en_exces_ne_vaut_rien() -> None:
    """Il reste 25 tiroirs : leur prix est nul ; les deux autres sont epuisees."""
    assert du.restes_des_ressources() == (0, 25, 0)


def test_les_bureaux_une_tablette_de_plus_rapporte_son_prix() -> None:
    """Planche 52, « Investissement ? » : le prix dual est le gain marginal."""
    assert [du.gain_d_une_unite_de_plus(i) for i in (1, 2, 3)] == [25, 0, 25]


def test_les_bureaux_un_nouveau_produit() -> None:
    """Planche 52, « Nouveau produit ? » : il vaut d'etre fabrique s'il bat son cout."""
    cout = du.cout_en_ressources((1, 1, 4))
    assert cout == 125
    assert du.entre_en_fabrication(gain=130, consommation=(1, 1, 4))
    assert not du.entre_en_fabrication(gain=120, consommation=(1, 1, 4))


# --- Planches 54 et 55 : l'algorithme dual ----------------------------------


def test_l_algorithme_dual_ses_trois_tableaux() -> None:
    """Planche 55 : z = 0, puis 27, puis 63/2 ; le dernier est admissible."""
    etapes = du.parcours_dual()
    assert [valeur for valeur, _ in etapes] == [0, 27, Q(63, 2)]
    assert [admissible for _, admissible in etapes] == [False, False, True]


def test_l_algorithme_dual_l_optimum() -> None:
    """Planche 55 : `x2 = 3/2`, `x1 = 9/2`."""
    assert du.optimum("dual") == (Q(63, 2), (Q(9, 2), Q(3, 2), 0))


def test_la_valeur_monte_la_ou_le_simplexe_la_fait_descendre() -> None:
    """Le dual part d'une valeur trop bonne et la degrade jusqu'a l'admissible."""
    valeurs = [valeur for valeur, _ in du.parcours_dual()]
    assert valeurs == sorted(valeurs)


# --- Planches 56 a 60 : la contrainte artificielle --------------------------


def test_la_contrainte_artificielle_exemple_i() -> None:
    """Planche 58 : `x1 = 21/4`, `x3 = 3/4`, et `z = 30`."""
    assert du.optimum("artificielle I") == (30, (Q(21, 4), 0, Q(3, 4)))


def test_la_contrainte_artificielle_exemple_ii() -> None:
    """Planche 60 : primale admissible, mais `z = -infini`."""
    assert du.issue("artificielle II") is Issue.NON_BORNEE


# --- Planches 61 et 62 : lire les variables duales --------------------------


def test_les_duales_se_lisent_dans_la_premiere_ligne() -> None:
    """Planche 62 : `a_0j = -u_j` pour un ecart hors base."""
    assert du.duales_lues_dans_le_tableau() == du.duales_par_le_dual()
    assert du.duales_par_le_dual() == (3, Q(3, 2))
