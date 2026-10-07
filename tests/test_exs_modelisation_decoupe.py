"""Optimisation : la modelisation, problemes II et VIII (decoupe, depistage).

Source : `Optimisation/EXS/Exs_Modelisation.pdf`, planches 6 a 11, 44 et
45. Les tableaux des planches sont recopies comme oracles ; les deux
optimums que la planche 11 affirme sont prouves par une borne.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.application import modele_decoupe as de
from polytech.optimisation.application import modele_decoupe_bornes as bo
from polytech.optimisation.application import modele_depistage as dp

Q = sp.Rational
PERTES = "0.1 1.3 0.3 0.42 0.8 0.3 0.5 0.0 1.5 0.62 0.12 1.0 0.5 0.0 0.74 0.24 1.62"
PERTES += " 1.12 0.62 0.12 2.0 1.5 1.0 0.5 0.0"
DECOUPES = (2, 1, 2, 2, 2, 3, 2, 2, 1, 2, 3, 2, 3, 3, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 4)

# --- Probleme II : la decoupe des rouleaux (planches 6 a 11) -----------------


def test_ii_les_vingt_cinq_decoupes_sont_toutes_la() -> None:
    """Planches 7 et 8 : ni oubli ni intrus parmi les facons de couper un rouleau."""
    assert len(de.MOTIFS) == 25
    assert set(de.MOTIFS) == set(de.toutes_les_decoupes())


def test_ii_les_colonnes_perte_et_decoupes() -> None:
    """Planches 7 et 8, recopiees colonne par colonne."""
    assert [de.perte(m) for m in de.MOTIFS] == [sp.nsimplify(p) for p in PERTES.split()]
    assert tuple(de.decoupes(m) for m in de.MOTIFS) == DECOUPES


def test_ii_une_chute_nulle_economise_un_trait_de_scie() -> None:
    """Autant de traits que de pieces, sauf si la derniere tombe juste."""
    assert de.decoupes((0, 2, 0, 1)) == 2
    assert de.decoupes((1, 1, 0, 0)) == 2


def test_ii_les_deux_solutions_de_la_planche_11() -> None:
    """28 rouleaux, la commande exacte, 12,6 m de perte, 56 decoupes."""
    for solution in de.SOLUTIONS:
        assert bo.bilan(solution) == (28, de.COMMANDES, Q(63, 5), 56)


def test_ii_les_relaxations() -> None:
    assert de.relaxation("rouleaux") == Q(55, 2)
    assert de.relaxation("perte") == Q(227, 20)
    assert de.relaxation("decoupes") == Q(277, 5)


def test_ii_les_bornes_prouvent_les_deux_optimums() -> None:
    """Un nombre entier de rouleaux, puis de decoupes : 28, 12,6 et 56."""
    assert bo.rouleaux_au_moins() == 28
    assert bo.perte_au_moins() == Q(63, 5)
    assert bo.decoupes_au_moins() == 56


def test_ii_la_perte_ne_depend_que_du_nombre_de_rouleaux() -> None:
    """`2,5 N - 57,4` : minimiser la perte, c'est minimiser les rouleaux."""
    assert bo.largeur_commandee() == Q(287, 5)
    assert bo.perte_de(28) == Q(63, 5)
    assert bo.perte_de(29) - bo.perte_de(28) == Q(5, 2)


def test_ii_la_borne_des_rouleaux_sans_simplexe() -> None:
    """Second chemin : 55 pieces de plus de 83 cm, deux au plus par rouleau."""
    assert bo.grandes_pieces() == 55
    assert bo.grandes_pieces_par_rouleau() == 2


def test_ii_la_borne_des_decoupes_sans_simplexe() -> None:
    """Second chemin : 63 pieces, et sept rouleaux sans chute au mieux."""
    assert sum(de.COMMANDES) == 63
    assert bo.motifs_sans_chute() == ((0, 2, 0, 1), (0, 1, 0, 3), (0, 0, 0, 5))
    assert bo.rouleaux_sans_chute_au_plus() == 7


def test_ii_un_objectif_inconnu_est_refuse() -> None:
    with pytest.raises(KeyError, match="objectif inconnu : "):
        de.relaxation("chutes")


# --- Probleme VIII : le depistage (planches 44 et 45) ------------------------


def test_viii_les_dix_contraintes_de_la_planche_45() -> None:
    """`D_ij` : les examens qui ne repondent pas pareil aux maladies i et j."""
    assert dp.examens_qui_distinguent(1, 2) == (1, 2)
    assert dp.examens_qui_distinguent(3, 4) == (5,)
    assert dp.examens_qui_distinguent(1, 3) == (2, 3, 4, 5)
    assert dp.examens_qui_distinguent(4, 5) == (1, 2, 3, 5)
    assert len(dp.programme().contraintes) == 10 + 5


def test_viii_la_solution_de_la_planche_45() -> None:
    """Planche 45 : `x = (1, 0, 1, 0, 1)`, `z = 14`."""
    assert dp.meilleur_choix() == (14, (1, 0, 1, 0, 1))


def test_viii_par_separation_et_evaluation() -> None:
    """Second chemin : le « Branch and Bound » sur le modele de la planche."""
    assert dp.par_separation() == (14, (1, 0, 1, 0, 1))


def test_viii_dix_choix_d_examens_suffisent_au_diagnostic() -> None:
    choix = dp.choix_suffisants()
    assert len(choix) == 10
    assert choix[1] == (16, (0, 1, 1, 0, 1))
    assert all(x[4] == 1 for _, x in choix)
