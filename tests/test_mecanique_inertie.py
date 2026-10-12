"""Les moments et produits d'inertie, et les solides des exercices du cours.

Source : `MecaniqueRationnelle2/SYLLABUS/Syllabus_Theorie_2025_2026.pdf`,
chapitre 6, pages 105 a 116 (92 a 103) -- definitions (6.12) a (6.21),
relations (6.22) a (6.25), rayon de giration, exercices 6.7.2 a 6.7.5.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.mecanique.domain.inertie.moments import (
    moment_polaire,
    moments_axiaux,
    moments_planaires,
    produits_d_inertie,
    rayon_de_giration,
)
from polytech.mecanique.domain.inertie.solides import (
    brique,
    couronne,
    cylindre,
    tige,
)

M, R, R1, R2, H, L, A, B, C = sp.symbols("m R R_1 R_2 h L a b c", positive=True)


def test_le_cercle_homogene_autour_de_son_axe() -> None:
    """Exercice 6.7.2, page 114 : `I_O = M R^2 / 2`."""
    disque = couronne(M, R, 0)
    assert sp.simplify(moments_axiaux(disque)[2] - M * R**2 / 2) == 0


def test_la_couronne_autour_de_son_axe() -> None:
    """Exercice 6.7.3, page 115 : `I_O = m (R1^2 + R2^2) / 2`."""
    ecart = moments_axiaux(couronne(M, R1, R2))[2] - M * (R1**2 + R2**2) / 2
    assert sp.simplify(ecart) == 0


def test_le_prisme_par_rapport_a_sa_base_et_a_son_plan_median() -> None:
    """Exercice 6.7.4, page 115 : `m h^2 / 3` et `m h^2 / 12`."""
    pose = brique(M, (A, B, H), origine=(-A / 2, -B / 2, 0))
    centre = brique(M, (A, B, H))
    assert sp.simplify(moments_planaires(pose)[2] - M * H**2 / 3) == 0
    assert sp.simplify(moments_planaires(centre)[2] - M * H**2 / 12) == 0


def test_le_cylindre_autour_de_son_axe_et_par_rapport_a_un_plan_meridien() -> None:
    """Exercice 6.7.5, page 115 : `M R^2 / 2` et `M R^2 / 4`."""
    plein = cylindre(M, R, H)
    assert sp.simplify(moments_axiaux(plein)[2] - M * R**2 / 2) == 0
    assert sp.simplify(moments_planaires(plein)[0] - M * R**2 / 4) == 0
    assert sp.simplify(moments_planaires(plein)[1] - M * R**2 / 4) == 0


def test_la_tige_du_pendule_de_la_figure_6_1() -> None:
    """Page 103, (6.2) : l'inertie `integrale de rho r^2 dr = m L^2 / 3`."""
    assert sp.simplify(moments_axiaux(tige(M, L))[2] - M * L**2 / 3) == 0


def test_les_relations_entre_les_moments_en_un_point() -> None:
    """Page 106, (6.22) a (6.25) : axes, plans et point, sur une brique excentree."""
    solide = brique(M, (A, B, C), origine=(0, -B / 3, C))
    iyz, izx, ixy = moments_planaires(solide)
    ix, iy, iz = moments_axiaux(solide)
    assert sp.simplify(ix - (izx + ixy)) == 0
    assert sp.simplify(iy - (iyz + ixy)) == 0
    assert sp.simplify(iz - (izx + iyz)) == 0
    polaire = moment_polaire(solide)
    assert sp.simplify(polaire - (ixy + iyz + izx)) == 0
    assert sp.simplify(polaire - (ix + iy + iz) / 2) == 0


def test_les_produits_d_inertie_s_annulent_par_symetrie() -> None:
    """Page 106, (6.19) a (6.21) ; page 112 : un axe de symetrie est principal."""
    assert produits_d_inertie(brique(M, (A, B, C))) == (0, 0, 0)
    coin = produits_d_inertie(brique(M, (A, B, C), origine=(0, 0, 0)))
    assert sp.simplify(coin[0] - M * A * B / 4) == 0
    assert sp.simplify(coin[1] - M * B * C / 4) == 0
    assert sp.simplify(coin[2] - M * A * C / 4) == 0


def test_le_rayon_de_giration_est_defini_par_m_k_carre() -> None:
    """Page 106 : `m k^2 = I_a` ; pour le cercle, `k = R / racine(2)`."""
    assert sp.simplify(rayon_de_giration(M * R**2 / 2, M) - R / sp.sqrt(2)) == 0
    with pytest.raises(ValueError, match="masse nulle"):
        rayon_de_giration(sp.Integer(3), sp.Integer(0))


def test_une_couronne_dont_le_trou_depasse_le_bord_est_refusee() -> None:
    """Le rayon interieur d'une couronne est plus petit que l'exterieur."""
    with pytest.raises(ValueError, match="rayon interieur 3 hors de"):
        couronne(M, sp.Integer(2), sp.Integer(3))
    with pytest.raises(ValueError, match="rayon interieur 2 hors de"):
        couronne(M, sp.Integer(2), sp.Integer(2))
    with pytest.raises(ValueError, match="rayon interieur -1 hors de"):
        couronne(M, sp.Integer(2), sp.Integer(-1))


def test_le_cylindre_est_centre_sur_son_plan_median() -> None:
    """Exercice 6.7.4, page 115 : `m h^2 / 12` par rapport au plan median."""
    plein = cylindre(M, R, H)
    assert plein.centre_de_masse() == (0, 0, 0)
    assert sp.simplify(moments_planaires(plein)[2] - M * H**2 / 12) == 0
