"""Chapitre I.3.4 d'optimisation : polyedres convexes.

Le polygone des essais est celui de la figure I.6 du cours
(`Optimisation/SLIDES/Slides_01_Les_bases_de_la_programmation_lineaire.pdf`,
page 38) : `x1 + x2 <= 8`, `-2 x1 + 3 x2 <= 6`, `x1 - x2 <= 2`, `x >= 0`.
Ses cinq sommets sont ceux que le tableau de la page 44 dit admissibles.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.combinaisons import est_combinaison_convexe
from polytech.optimisation.domain.polyedres import DemiEspace, Polyedre
from polytech.optimisation.domain.polyedres_non_bornes import (
    est_borne,
    est_un_polytope,
    rayons_extremes,
)

R = sp.Rational
SOMMETS = {(0, 0), (2, 0), (5, 3), (R(18, 5), R(22, 5)), (0, 2)}


def _figure_i6(sens: int = 1) -> Polyedre:
    """`sens = -1` retourne les trois contraintes reelles : la figure I.7."""
    reelles = [((1, 1), 8), ((-2, 3), 6), ((1, -1), 2)]
    demi_espaces = [
        DemiEspace(tuple(sens * a for a in normale), sens * borne)
        for normale, borne in reelles
    ]
    non_negativite = [DemiEspace((-1, 0), 0), DemiEspace((0, -1), 0)]
    return Polyedre(tuple(demi_espaces + non_negativite))


def test_un_demi_espace_contient_ce_qui_est_du_bon_cote() -> None:
    demi_espace = DemiEspace((-2, 3), 6)
    assert demi_espace.contient((2, 1))
    assert not demi_espace.contient((0, 3))


def test_la_normale_pointe_vers_l_exterieur() -> None:
    """`alpha (y - x) <= 0` pour `x` sur l'hyperplan et `y` dans le demi-espace."""
    demi_espace = DemiEspace((-2, 3), 6)
    sur_l_hyperplan, dedans = (0, 2), (2, 1)
    assert demi_espace.est_sur_l_hyperplan(sur_l_hyperplan)
    ecart = [y - x for x, y in zip(sur_l_hyperplan, dedans, strict=True)]
    assert sum(a * e for a, e in zip((-2, 3), ecart, strict=True)) <= 0


def test_un_point_interieur_verifie_toutes_les_inegalites_strictement() -> None:
    assert _figure_i6().est_interieur((1, 1))
    assert not _figure_i6().est_interieur((1, 0))


def test_un_point_frontiere_touche_un_hyperplan() -> None:
    assert _figure_i6().est_frontiere((1, 0))
    assert not _figure_i6().est_frontiere((1, 1))


def test_un_point_exterieur_n_est_ni_interieur_ni_frontiere() -> None:
    polyedre = _figure_i6()
    assert not polyedre.contient((9, 9))
    assert not polyedre.est_frontiere((9, 9))


def test_un_sommet_est_un_point_frontiere_particulier() -> None:
    polyedre = _figure_i6()
    assert polyedre.est_sommet((2, 0))
    assert not polyedre.est_sommet((1, 0))


def test_les_sommets_de_la_figure_i6_sont_ceux_du_tableau() -> None:
    assert {tuple(s) for s in _figure_i6().sommets()} == SOMMETS


def test_deux_sommets_relies_par_une_arete_sont_adjacents() -> None:
    polyedre = _figure_i6()
    assert polyedre.sont_adjacents((2, 0), (5, 3))
    assert not polyedre.sont_adjacents((2, 0), (R(18, 5), R(22, 5)))


def test_la_figure_i6_est_un_polytope() -> None:
    assert est_borne(_figure_i6())
    assert est_un_polytope(_figure_i6())


def test_la_figure_i7_n_est_pas_bornee() -> None:
    """Contraintes retournees : `D` a un sommet, (12, 10), et deux rayons."""
    polyedre = _figure_i6(sens=-1)
    assert {tuple(s) for s in polyedre.sommets()} == {(12, 10)}
    assert not est_borne(polyedre)
    assert not est_un_polytope(polyedre)


def test_un_rayon_reste_dans_le_polyedre() -> None:
    """`x + lambda d` appartient a `P` pour tout `lambda >= 0`."""
    polyedre = _figure_i6(sens=-1)
    rayons = rayons_extremes(polyedre)
    assert len(rayons) == 2
    for d in rayons:
        for pas in (1, 10, 1000):
            assert polyedre.contient((12 + pas * d[0], 10 + pas * d[1]))


def test_un_polyedre_dont_les_hyperplans_passent_par_l_origine_est_un_cone() -> None:
    cone = Polyedre((DemiEspace((1, -1), 0), DemiEspace((-1, -1), 0)))
    assert cone.est_un_cone()
    assert not _figure_i6().est_un_cone()


def test_un_point_du_polytope_est_combinaison_convexe_de_ses_sommets() -> None:
    assert est_combinaison_convexe((1, 1), sorted(SOMMETS, key=str))


def test_le_milieu_de_deux_sommets_est_leur_combinaison_convexe() -> None:
    assert est_combinaison_convexe((R(7, 2), R(3, 2)), [(2, 0), (5, 3)])


def test_un_point_exterieur_n_est_pas_combinaison_convexe() -> None:
    assert not est_combinaison_convexe((10, 10), sorted(SOMMETS, key=str))


def test_un_point_hors_de_la_droite_de_deux_points_n_en_est_pas_combinaison() -> None:
    assert not est_combinaison_convexe((0, 1), [(2, 0), (5, 3)])


def test_un_demi_espace_de_mauvaise_dimension_est_refuse() -> None:
    with pytest.raises(ValueError, match="dimensions differentes"):
        Polyedre((DemiEspace((1, 1), 0), DemiEspace((1, 1, 1), 0)))


def test_un_polyedre_sans_sommet_ne_dit_pas_s_il_est_borne() -> None:
    """Un seul demi-plan contient une droite : il n'a aucun sommet."""
    with pytest.raises(ValueError, match="contient une droite"):
        est_borne(Polyedre((DemiEspace((1, 1), 0),)))


def test_un_point_exterieur_n_est_pas_interieur() -> None:
    """(9, 9) ne touche aucun hyperplan, et n'est pas dans le polyedre."""
    assert not _figure_i6().est_interieur((9, 9))


def test_un_point_exterieur_sur_un_hyperplan_n_est_pas_frontiere() -> None:
    """(10, 0) est sur `x2 = 0`, mais viole `x1 - x2 <= 2`."""
    assert not _figure_i6().est_frontiere((10, 0))


def test_les_deux_rayons_de_la_figure_i7_longent_ses_aretes() -> None:
    """Le long de `-2 x1 + 3 x2 = 6` et de `x1 - x2 = 2`, en s'eloignant."""
    rayons = {tuple(d) for d in rayons_extremes(_figure_i6(sens=-1))}
    assert rayons == {(R(3, 5), R(2, 5)), (R(1, 2), R(1, 2))}


def test_les_rayons_du_polyedre_symetrique_partent_a_l_oppose() -> None:
    """Le symetrique par rapport a l'origine : les memes aretes, l'autre sens."""
    symetrique = Polyedre(
        tuple(
            DemiEspace(tuple(-a for a in d.normale), d.borne)
            for d in _figure_i6(sens=-1).demi_espaces
        )
    )
    rayons = {tuple(d) for d in rayons_extremes(symetrique)}
    assert rayons == {(-R(3, 5), -R(2, 5)), (-R(1, 2), -R(1, 2))}


def test_un_polyedre_borne_mais_vide_n_est_pas_un_polytope() -> None:
    """`x1 <= 0` et `x1 >= 1` : aucun point, donc aucun sommet."""
    vide = Polyedre(
        (
            DemiEspace((1, 0), 0),
            DemiEspace((-1, 0), -1),
            DemiEspace((0, 1), 1),
            DemiEspace((0, -1), 0),
        )
    )
    assert vide.sommets() == ()
    assert not est_un_polytope(vide)


def test_la_dimension_est_celle_de_l_espace_des_decisions() -> None:
    """Deux variables `x1`, `x2` : `D` se dessine dans le plan."""
    assert _figure_i6().dimension == 2


def test_les_demi_espaces_actifs_sont_ceux_dont_l_hyperplan_passe_par_le_point() -> (
    None
):
    """En (2, 0) : `x1 - x2 = 2` et `x2 = 0`, et eux seuls."""
    actifs = _figure_i6().actifs((2, 0))
    assert set(actifs) == {DemiEspace((1, -1), 2), DemiEspace((0, -1), 0)}
