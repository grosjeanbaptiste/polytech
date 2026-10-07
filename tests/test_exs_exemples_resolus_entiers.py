"""Optimisation : les exemples resolus, partie « Branch and Bound ».

Source : `Optimisation/EXS/Exs_Exemples_resolus.pdf`, planches 63 a 85.
Une illustration deroulee selon trois regles de separation, un second
exemple, le pretraitement, puis quatre modelisations du « lot-sizing ».
Chaque evaluation, chaque penalite et chaque nombre de sommets des
planches est recopie comme oracle et recalcule.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.application import exemples_entiers as en
from polytech.optimisation.application import exemples_lots as lo
from polytech.optimisation.application import exemples_lots_plans as pl
from polytech.optimisation.domain.contraintes import Sens

Q = sp.Rational
INF, SUP = Sens.INFERIEUR, Sens.SUPERIEUR

# --- Planches 64 a 74 : l'illustration --------------------------------------


def test_la_relaxation_de_la_planche_66() -> None:
    """Planche 66 : `x = (0, 7/2, 3/2)`, et `v0 = 45/2`."""
    racine = en.noeud()
    assert (racine.point, racine.evaluation) == ((0, Q(7, 2), Q(3, 2)), Q(45, 2))


def test_les_penalites_de_la_planche_66() -> None:
    """Planche 66 : `P3 = (1/2, 3/2)`, `P2 = (3/8, 1/6)`."""
    assert en.penalites_a_la_racine() == {3: (Q(1, 2), Q(3, 2)), 2: (Q(3, 8), Q(1, 6))}


def test_la_regle_de_la_plus_grande_penalite() -> None:
    """Planches 67 a 69 : `x3 <= 1` donne `(0, 5, 1)` et 23 ; `x3 >= 2` est sonde."""
    assert en.plus_grande_penalite() == (3, "b", Q(3, 2))
    gauche = en.noeud((3, INF, 1))
    assert (gauche.point, gauche.evaluation) == ((0, 5, 1), 23)
    assert en.borne(Q(45, 2), Q(3, 2)) == 24
    assert en.noeud((3, SUP, 2)).evaluation == 24


def test_la_regle_de_la_plus_petite_penalite() -> None:
    """Planche 70 : `x2 >= 4` donne `v1 = 68/3`, non entier."""
    assert en.plus_petite_penalite() == (2, "b", Q(1, 6))
    premier = en.noeud((2, SUP, 4))
    assert (premier.point, premier.evaluation) == ((0, 4, Q(4, 3)), Q(68, 3))


def test_la_suite_de_la_plus_petite_penalite() -> None:
    """Planches 71 et 72 : 23 en `(0, 5, 1)` ; puis 84/3 et 183/8."""
    deuxieme = en.noeud((2, SUP, 4), (3, INF, 1))
    assert (deuxieme.point, deuxieme.evaluation) == ((0, 5, 1), 23)
    assert en.borne(Q(68, 3), Q(16, 3)) == 28
    assert en.noeud((2, SUP, 4), (3, SUP, 2)).evaluation == 28
    quatrieme = en.noeud((2, INF, 3))
    assert (quatrieme.point, quatrieme.evaluation) == (
        (Q(1, 8), 3, Q(13, 8)),
        Q(183, 8),
    )


def test_le_sommet_s4_ne_contient_rien_de_mieux() -> None:
    """Planche 73 : `v5 > 23` et `v6 > 23`. Son meilleur point entier vaut 25."""
    assert en.borne(Q(183, 8), Q(1, 8)) == 23
    valeur, points = en.meilleur_point_entier((2, INF, 3))
    assert valeur == 25
    assert all(point[1] <= 3 for point in points)


def test_l_optimum_entier_de_l_illustration() -> None:
    """Planches 68 et 73 : `x = (0, 5, 1)`, de valeur 23, quelle que soit la regle."""
    assert en.optimum_entier() == (23, (0, 5, 1))
    assert en.meilleur_point_entier() == (23, ((0, 5, 1),))


# --- Planche 75 : arrondir ne suffit pas ------------------------------------


def test_un_autre_exemple_en_variables_entieres() -> None:
    """Planche 75 : `(1,66 ; 3,33)` en continu, `(4 ; 0)` en entiers."""
    continu, entier = en.autre_exemple()
    assert continu == (Q(205, 3), (Q(5, 3), Q(10, 3)))
    assert entier == (68, (4, 0))


def test_l_arrondi_du_continu_n_est_pas_admissible() -> None:
    """`(2, 3)` viole la premiere contrainte ; `(1, 3)` ne vaut que 53."""
    assert not en.est_admissible_dans_l_autre((2, 3))
    assert en.valeur_dans_l_autre((1, 3)) == 53


# --- Planches 77 et 78 : le pretraitement -----------------------------------


def test_les_trois_gestes_du_pretraitement() -> None:
    """Planches 77 et 78 : redondante ; contradictoire ; puis `x1 <= 2` et `x3 = 1`."""
    assert en.est_redondante((1, 1, 1), 9)
    assert en.est_contradictoire((5, -1, 6), 7)
    bornes, simplifiee = en.resserree((2, -1, 5), 7)
    assert bornes == ((1, 2), (1, 3), (1, 1))
    assert simplifiee == ((2, -1, 0), 2)


# --- Planches 79 a 85 : le « lot-sizing » -----------------------------------


@pytest.mark.parametrize(("nom", "sommets"), [("I", 19), ("II", 11), ("III", 3)])
def test_le_nombre_de_sommets_des_planches(nom: str, sommets: int) -> None:
    """Planches 81 a 83 : 19, 11 puis 3 sommets pour les trois premieres ecritures."""
    assert lo.sommets_explores(nom) == sommets


def test_les_relaxations_se_resserrent() -> None:
    """Chaque ecriture a une relaxation plus haute que la precedente."""
    valeurs = [lo.relaxation(nom) for nom in ("I", "II", "III", "IV")]
    assert valeurs == sorted(valeurs)
    assert [float(v) for v in valeurs[:3]] == pytest.approx(
        [167.34, 183.03, 186.06], abs=5e-3
    )


def test_la_quatrieme_ecriture_est_la_fermeture_convexe() -> None:
    """Planche 84 : sa relaxation est entiere d'emblee, et vaut l'optimum."""
    assert lo.relaxation("IV") == 188
    assert lo.ouvertures_de_la_relaxation("IV") == (1, 1, 0, 1, 0)


def test_l_optimum_du_lot_sizing() -> None:
    """Second chemin : les 32 choix de mois ouverts, un a un."""
    valeur, ouverts, production = pl.optimum_par_enumeration()
    assert (valeur, ouverts, production) == (188, (1, 1, 0, 1, 0), (3, 11, 0, 11, 0))
    assert pl.cout_du_plan(production) == 188


def test_un_plan_qui_manque_une_demande_est_refuse() -> None:
    """Le stock ne peut pas etre negatif : on ne sert pas ce qu'on n'a pas."""
    with pytest.raises(
        ValueError, match="le plan ne sert pas la demande du mois "
    ) as refus:
        pl.cout_du_plan((3, 4, 7, 3, 8))
    assert "mois 2" in str(refus.value)


def test_un_plan_qui_produit_trop_est_refuse() -> None:
    """Rien ne doit rester apres le dernier mois."""
    with pytest.raises(ValueError, match="le plan laisse ") as refus:
        pl.cout_du_plan((3, 5, 6, 3, 9))
    assert "laisse 1 unites" in str(refus.value)


def test_le_cout_d_un_plan_mois_par_mois() -> None:
    """Tout produire chaque mois : aucun stock, cinq forfaits."""
    assert pl.cout_du_plan((3, 5, 6, 3, 8)) == 30 + 6 + 20 + 36 + 24 + 80
    assert pl.plan_des_mois_ouverts((1, 0, 0, 0, 0)) == (25, 0, 0, 0, 0)
    assert (
        pl.cout_du_plan((25, 0, 0, 0, 0)) == 10 + 50 + 3 * 22 + 2 * 17 + 3 * 11 + 2 * 8
    )


def test_le_nombre_de_contraintes_de_chaque_ecriture() -> None:
    """Cinq bilans, cinq liens, cinq `y <= 1` ; puis `y1 = 1`, puis trois renforts."""
    comptes = {
        nom: len(lo.ecriture(nom).contraintes) for nom in ("I", "II", "III", "IV")
    }
    assert comptes == {"I": 15, "II": 16, "III": 19, "IV": 25}
    for nom in ("I", "II", "III", "IV"):
        assert all(y <= 1 for y in lo.ouvertures_de_la_relaxation(nom))
