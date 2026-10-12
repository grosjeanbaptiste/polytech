"""La statique : moment d'une force, torseur, changement de pole.

Source : `MecaniqueRationnelle1/SYLLABUS/Syllabus_Partie_1_Statique.pdf`,
sections 1.6.1 a 1.6.3, pages 13 a 18 -- definitions (1.8), (1.13), (1.14),
formule du changement de pole (1.16), equivalence (1.20) et (1.21).
"""

from __future__ import annotations

from typing import Any

import pytest
import sympy as sp

from polytech.mecanique.domain.statique.forces import Force, difference, moment
from polytech.mecanique.domain.statique.torseurs import Torseur

ORIGINE = (0, 0, 0)


def test_le_vecteur_de_o_vers_a_est_la_difference_des_positions() -> None:
    """Atelier de calcul vectoriel, pages 11 et 20 : `a - b`, par composantes."""
    assert difference((5, 2, -1), (1, 3, 4)) == (4, -1, -5)
    assert difference((1, 3, 4), (5, 2, -1)) == (-4, 1, 5)


def test_le_moment_est_oa_vectoriel_f() -> None:
    """Page 13, (1.8) : `M_O(F/A) = OA ^ F`."""
    force = Force(point=(2, 0, 0), vecteur=(0, 5, 0))
    assert moment(force, ORIGINE) == (0, 0, 10)
    assert moment(force, (2, 0, 0)) == (0, 0, 0)


def test_glisser_la_force_sur_sa_ligne_d_action_ne_change_pas_le_moment() -> None:
    """Page 14, (1.11) : `M_O(F/B) = M_O(F/A)` si `AB` est parallele a `F`."""
    en_a = Force(point=(2, 1, 0), vecteur=(1, 3, 0))
    en_b = Force(point=(4, 7, 0), vecteur=(1, 3, 0))
    assert moment(en_b, ORIGINE) == moment(en_a, ORIGINE)


def test_les_coordonnees_d_un_torseur_sont_la_resultante_et_le_moment() -> None:
    """Page 16, (1.13) et (1.14) : somme des forces, somme de leurs moments."""
    forces = (Force((1, 0, 0), (0, 2, 0)), Force((0, 1, 0), (3, 0, 0)))
    torseur = Torseur.des_forces(forces, ORIGINE)
    assert torseur.resultante == (3, 2, 0)
    assert torseur.moment == (0, 0, 2 - 3)


def test_le_changement_de_pole_est_la_formule_1_16() -> None:
    """Page 17, (1.16) : `M_B = M_A + R ^ AB`, sans refaire la somme."""
    forces = (Force((1, 0, 0), (0, 2, 0)), Force((0, 1, 2), (3, 0, -1)))
    nouveau = (4, -1, 2)
    transporte = Torseur.des_forces(forces, ORIGINE).en(nouveau)
    assert transporte == Torseur.des_forces(forces, nouveau)


def test_deux_forces_opposees_de_meme_ligne_d_action_s_equilibrent() -> None:
    """Page 18, (1.21) : resultante nulle et moment nul en un pole."""
    paire = (Force((0, 0, 0), (1, 1, 0)), Force((2, 2, 0), (-1, -1, 0)))
    assert Torseur.des_forces(paire, ORIGINE).est_nul()
    assert Torseur.des_forces(paire, (5, 0, 3)).est_nul()


def test_un_couple_a_une_resultante_nulle_et_un_moment_qui_ne_depend_pas_du_pole() -> (
    None
):
    """Page 17 : avec `R = 0`, le changement de pole laisse le moment."""
    couple = (Force((0, 0, 0), (0, -1, 0)), Force((2, 0, 0), (0, 1, 0)))
    torseur = Torseur.des_forces(couple, ORIGINE)
    assert torseur.resultante == (0, 0, 0)
    assert torseur.moment == (0, 0, 2)
    assert torseur.en((7, 3, 1)).moment == (0, 0, 2)
    assert not torseur.est_nul()


def test_des_forces_concourantes_equivalent_a_leur_resultante() -> None:
    """Page 18, theoreme de Varignon : memes coordonnees au point de concours."""
    concours = (1, 1, 0)
    forces = (Force(concours, (2, 0, 0)), Force(concours, (0, 3, 0)))
    seule = Torseur.des_forces((Force(concours, (2, 3, 0)),), ORIGINE)
    assert Torseur.des_forces(forces, ORIGINE).est_equivalent_a(seule)
    ailleurs = Torseur.des_forces((Force(ORIGINE, (2, 3, 0)),), ORIGINE)
    assert not Torseur.des_forces(forces, ORIGINE).est_equivalent_a(ailleurs)


def test_le_moment_reste_litteral() -> None:
    """Le bras de levier `h` de la figure 1.10 : `M = F h`."""
    bras, intensite = sp.symbols("h F", positive=True)
    assert (
        moment(Force((bras, 0, 0), (0, intensite, 0)), ORIGINE)[2] == bras * intensite
    )


def test_une_force_hors_de_l_espace_est_refusee() -> None:
    """Une force est un vecteur localise de l'espace a trois dimensions."""
    plan: Any = (0, 0)
    with pytest.raises(ValueError, match="trois composantes attendues, et non"):
        Force(point=plan, vecteur=(1, 0, 0))


def test_l_equivalence_se_juge_en_un_meme_pole() -> None:
    """Page 18, (1.20) : deux ecritures du meme torseur en deux poles differents."""
    forces = (Force((1, 0, 0), (0, 2, 0)), Force((0, 1, 2), (3, 0, -1)))
    ici = Torseur.des_forces(forces, ORIGINE)
    la_bas = Torseur.des_forces(forces, (4, -1, 2))
    assert ici.moment != la_bas.moment
    assert ici.est_equivalent_a(la_bas)


def test_un_torseur_sans_force_est_refuse() -> None:
    """Un torseur est un ensemble de forces : vide, il n'a pas de coordonnees."""
    with pytest.raises(ValueError, match="torseur sans force"):
        Torseur.des_forces((), ORIGINE)
