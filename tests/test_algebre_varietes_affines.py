"""Notes de cours de 2011, annexe 2, section D : les varietes lineaires affines.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, pages
dactylographiees 41 et 42 (pages 40 et 41 du fichier, un scan sans couche
de texte, lu sur image). Translation, variete lineaire affine, sous-espace
directeur, dimension, droite, hyperplan, parallelisme, intersection,
points affinement independants.

Ces deux pages enoncent des proprietes et ne portent aucun exemple
chiffre. Chaque propriete est donc verifiee sur des varietes que le
syllabus d'algebre fournit : les ensembles de solutions de ses systemes
lineaires, qu'il ecrit sous la « forme generale » -- une solution
particuliere plus un sous-espace.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.sous_espaces import (
    base_du_noyau,
    engendrent_le_meme_sous_espace,
)
from polytech.maths.domain.algebre.systemes import SystemeLineaire
from polytech.maths.domain.algebre.varietes_affines import VarieteAffine

M = sp.Matrix
R = sp.Rational
EXEMPLE_3 = M([[1, -1, 2, -1], [2, 1, -2, -2], [-1, 2, -4, 1], [4, -1, 2, -4]])
SECOND_3 = M([-1, -2, 1, -4])
DIX = M([[1, 1, 2], [3, 2, 3], [5, 4, 7]])
PLAN = VarieteAffine((1, 0, 0), ((1, 1, 0), (0, 1, 1)))
DROITE = VarieteAffine((0, 0, 1), ((1, 2, 3),))


def _solutions(matrice: sp.Matrix, second: sp.Matrix) -> VarieteAffine:
    return VarieteAffine.des_solutions(SystemeLineaire(matrice, second))


def test_une_variete_est_le_translate_d_un_sous_espace() -> None:
    """Page 41 : `w_Y = {X + Y | X dans W}`, et `Y` appartient a `w_Y`."""
    assert PLAN.contient((1, 0, 0))
    assert PLAN.contient((2, 1, 0))
    assert PLAN.contient((2, 3, 2))
    assert not PLAN.contient((0, 0, 0))
    assert not PLAN.contient((1, 0, 1))


def test_une_variete_n_est_un_sous_espace_que_si_elle_contient_zero() -> None:
    """Page 41 : `w_Y` est un sous-espace si et seulement si `Y` est dans `W`."""
    assert not PLAN.est_un_sous_espace()
    par_l_origine = VarieteAffine((1, 2, 1), ((1, 1, 0), (0, 1, 1)))
    assert par_l_origine.est_un_sous_espace()
    assert par_l_origine.est_egale_a(VarieteAffine((0, 0, 0), ((1, 1, 0), (0, 1, 1))))


def test_tout_point_de_la_variete_la_definit() -> None:
    """Propriete 2 : si `Y2` est dans `w_Y1`, alors `w_Y2 = w_Y1`."""
    autre = VarieteAffine((2, 3, 2), PLAN.directeur)
    assert autre.est_egale_a(PLAN)
    ailleurs = VarieteAffine((0, 0, 0), PLAN.directeur)
    assert not ailleurs.est_egale_a(PLAN)
    assert VarieteAffine((1, 0, 0), ((1, 2, 1), (1, 0, -1))).est_egale_a(PLAN)


def test_la_dimension_est_celle_du_sous_espace_directeur() -> None:
    """Page 42 : dimension 1, une droite ; dimension `n - 1`, un hyperplan."""
    assert PLAN.dimension() == 2
    assert PLAN.est_un_hyperplan()
    assert not PLAN.est_une_droite()
    assert DROITE.dimension() == 1
    assert DROITE.est_une_droite()
    assert not DROITE.est_un_hyperplan()
    redondante = VarieteAffine((1, 0, 0), ((1, 1, 0), (0, 1, 1), (1, 2, 1)))
    assert redondante.dimension() == 2
    point = VarieteAffine((1, 2, 3), ())
    assert point.dimension() == 0
    assert point.contient((1, 2, 3))
    assert not point.contient((1, 2, 4))


def test_les_solutions_d_un_systeme_forment_une_variete_de_directeur_le_noyau() -> None:
    """Syllabus d'algebre, exemple III.4, et sa « forme generale »."""
    variete = _solutions(EXEMPLE_3, SECOND_3)
    assert variete.point == (-1, 0, 0, 0)
    assert variete.directeur == ((0, 2, 1, 0), (1, 0, 0, 1))
    assert variete.dimension() == 2
    assert engendrent_le_meme_sous_espace(variete.directeur, base_du_noyau(EXEMPLE_3))
    homogene = _solutions(EXEMPLE_3, sp.zeros(4, 1))
    assert homogene.est_un_sous_espace()
    assert homogene.est_parallele_a(variete)


def test_deux_varietes_paralleles_sont_egales_ou_disjointes() -> None:
    """Page 42, sur l'exercice III.10 : meme matrice, deux seconds membres."""
    premiere = _solutions(DIX, M([1, 4, 6]))
    seconde = _solutions(DIX, M([3, -3, 3]))
    assert premiere.contient((2, -1, 0))
    assert seconde.contient((-9, 12, 0))
    assert premiere.est_parallele_a(seconde)
    assert premiere.dimension() == seconde.dimension() == 1
    assert not premiere.est_egale_a(seconde)
    assert premiere.intersection(seconde) is None
    assert premiere.intersection(VarieteAffine((3, -4, 1), ((1, -3, 1),))) is not None


def test_des_varietes_de_directeurs_differents_ne_sont_pas_paralleles() -> None:
    """Le plan et la droite ; deux plans de directions distinctes."""
    assert not PLAN.est_parallele_a(DROITE)
    assert not PLAN.est_parallele_a(VarieteAffine((1, 0, 0), ((1, 0, 0), (0, 1, 0))))
    assert PLAN.est_parallele_a(VarieteAffine((5, 5, 5), ((1, 2, 1), (1, 0, -1))))


def test_une_droite_contenue_dans_un_plan_ne_lui_est_ni_parallele_ni_egale() -> None:
    """Meme sous-espace directeur, et non un directeur inclus dans l'autre."""
    droite = VarieteAffine((1, 0, 0), ((1, 1, 0),))
    assert PLAN.contient(droite.point)
    assert not droite.est_parallele_a(PLAN)
    assert not PLAN.est_parallele_a(droite)
    assert not PLAN.est_egale_a(droite)
    assert not droite.est_egale_a(PLAN)


def test_l_intersection_a_pour_directeur_l_intersection_des_directeurs() -> None:
    """Page 42 : deux plans de `R^3` se coupent suivant une droite."""
    autre = VarieteAffine((0, 0, 1), ((1, 0, 0), (0, 1, 0)))
    commune = PLAN.intersection(autre)
    assert commune is not None
    assert commune.est_une_droite()
    assert PLAN.contient(commune.point)
    assert autre.contient(commune.point)
    for direction in commune.directeur:
        assert PLAN.contient(tuple(M(PLAN.point) + M(direction)))
        assert direction[2] == 0


def test_une_droite_et_un_plan_se_coupent_en_un_point() -> None:
    """Dimension 0 : `1 + s = r`, `s + u = 2r`, `u = 2 + 3r` donnent `r = -1/2`."""
    droite = VarieteAffine((0, 0, 2), ((1, 2, 3),))
    commune = PLAN.intersection(droite)
    assert commune is not None
    assert commune.dimension() == 0
    assert PLAN.contient(commune.point)
    assert droite.contient(commune.point)
    assert commune.point == (-R(1, 2), -1, R(1, 2))
    assert DROITE.intersection(PLAN) is not None


def test_les_trois_hyperplans_d_un_systeme_de_cramer_se_coupent_en_sa_solution() -> (
    None
):
    """Syllabus d'algebre, exemple III.2 : la solution `(1, 2, 3)`."""
    lignes = (([1, 1, 2], 9), ([2, 4, -3], 1), ([3, 6, -5], 0))
    plans = [_solutions(M([ligne]), M([second])) for ligne, second in lignes]
    assert all(plan.est_un_hyperplan() for plan in plans)
    droite = plans[0].intersection(plans[1])
    assert droite is not None
    assert droite.est_une_droite()
    point = droite.intersection(plans[2])
    assert point is not None
    assert point.dimension() == 0
    assert point.point == (1, 2, 3)


def test_la_variete_engendree_par_des_points() -> None:
    """Page 42 : `k + 1` points engendrent une variete de dimension au plus `k`."""
    triangle = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    plan = VarieteAffine.engendree_par(triangle)
    assert plan.dimension() == 2
    assert all(plan.contient(sommet) for sommet in triangle)
    assert plan.contient((R(1, 3), R(1, 3), R(1, 3)))
    assert not plan.contient((0, 0, 0))
    assert VarieteAffine.sont_affinement_independants(triangle)
    alignes = ((0, 0, 0), (1, 1, 1), (2, 2, 2))
    assert VarieteAffine.engendree_par(alignes).dimension() == 1
    assert not VarieteAffine.sont_affinement_independants(alignes)


def test_un_point_et_un_directeur_vivent_dans_le_meme_espace() -> None:
    with pytest.raises(ValueError, match="n'est pas de la taille du point"):
        VarieteAffine((1, 0, 0), ((1, 1),))
    with pytest.raises(ValueError, match="aucun point"):
        VarieteAffine.engendree_par(())
