"""Mecanique rationnelle I : ce que le cours ecrit, refait par la bibliotheque.

Sources, toutes sous `MecaniqueRationnelle1/` :
`SLIDES/Slides_Statique_Equilibre_des_forces.pdf`, pages 35, 37 et 44 ;
`SYLLABUS/Syllabus_Partie_1_Statique.pdf`, pages 26 et 145 ;
`SYLLABUS/Syllabus_Partie_2_Cinematique_et_methodes_energetiques.pdf`,
pages 211 a 215 et 229 ; `SLIDES/Slides_Quantite_de_mouvement_Tenseurs.pdf`,
pages 19 a 22.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.mecanique.domain.inertie.moments import rayon_de_giration
from polytech.mecanique.domain.inertie.solides import brique, couronne, cylindre
from polytech.mecanique.domain.inertie.tenseur import (
    rotation_autour_de_z,
    tenseur_d_inertie,
    tourne,
)
from polytech.mecanique.domain.statique.forces import Force, moment
from polytech.mecanique.domain.statique.torseurs import Torseur

M, R, H, A, B, C, F = sp.symbols("m R h a b c F", positive=True)
THETA = sp.Symbol("theta", real=True)
ORIGINE = (0, 0, 0)


def test_l_illustration_du_moment_d_une_force() -> None:
    """Planches de statique, page 35 : `h F u_z`, `-h F u_z`, et zero."""
    assert moment(Force((0, -H, 0), (F, 0, 0)), ORIGINE) == (0, 0, H * F)
    assert moment(Force((0, H, 0), (F, 0, 0)), ORIGINE) == (0, 0, -H * F)
    assert moment(Force((A, 0, 0), (F, 0, 0)), ORIGINE) == (0, 0, 0)


def test_un_solide_est_en_equilibre_si_forces_et_moments_s_annulent() -> None:
    """Planches, page 37 : `R_e = 0` et `M_eO = 0` ; une poutre sur deux appuis."""
    poids, gauche, droite = (
        Force((1, 0, 0), (0, -60, 0)),
        Force((0, 0, 0), (0, 30, 0)),
        Force((2, 0, 0), (0, 30, 0)),
    )
    assert Torseur.des_forces((poids, gauche, droite), ORIGINE).est_nul()
    mal_place = Force((3, 0, 0), (0, 30, 0))
    assert not Torseur.des_forces((poids, gauche, mal_place), ORIGINE).est_nul()


def test_un_couple_a_un_moment_uniforme_dans_l_espace() -> None:
    """Syllabus de statique, page 26, (1.41) : resultante nulle, meme moment partout."""
    couple = Torseur.des_forces(
        (Force((0, 0, 0), (0, -F, 0)), Force((A, 0, 0), (0, F, 0))), ORIGINE
    )
    assert couple.resultante == (0, 0, 0)
    assert couple.en((B, C, H)).moment == (0, 0, A * F)


def test_le_tenseur_du_disque_mince() -> None:
    """Syllabus de cinematique, page 214, (5.18) ; planches, page 22."""
    attendu = sp.diag(M * R**2 / 4, M * R**2 / 4, M * R**2 / 2)
    assert sp.simplify(tenseur_d_inertie(couronne(M, R, 0)) - attendu) == sp.zeros(3)
    assert sp.simplify(rayon_de_giration(M * R**2 / 4, M) - R / 2) == 0


def test_les_tenseurs_du_parallelepipede_et_du_cylindre() -> None:
    """Syllabus de cinematique, pages 214 et 215, au centre de masse."""
    pave = sp.diag(B**2 + C**2, A**2 + C**2, A**2 + B**2) * M / 12
    assert sp.simplify(tenseur_d_inertie(brique(M, (A, B, C))) - pave) == sp.zeros(3)
    transverse = M * R**2 / 4 + M * H**2 / 12
    rond = sp.diag(transverse, transverse, M * R**2 / 2)
    assert sp.simplify(tenseur_d_inertie(cylindre(M, R, H)) - rond) == sp.zeros(3)


def test_l_estimation_du_moment_d_inertie_d_une_automobile() -> None:
    """Syllabus de cinematique, page 212, (5.16) : 722,5 et 360 kg m2 pour une tonne."""
    assert float(rayon_de_giration(sp.Float(722.5), sp.Float(1000))) == pytest.approx(
        0.85
    )
    assert float(rayon_de_giration(sp.Float(360), sp.Float(1000))) == pytest.approx(0.6)


def test_la_variance_tensorielle_avec_la_matrice_de_passage_a_47() -> None:
    """Syllabus de statique, page 145, (A.47) ; de cinematique, page 229."""
    passage = rotation_autour_de_z(THETA)
    s, c = sp.sin(THETA), sp.cos(THETA)
    assert passage == sp.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    quelconque = sp.Matrix([[2, -1, 0], [-1, 5, 3], [0, 3, 7]])
    tourne_ = tourne(quelconque, passage)
    assert sp.simplify(tourne_.trace() - quelconque.trace()) == 0
    assert sp.simplify(tourne_[0, 2] - (c * 0 + s * 3)) == 0
