"""Devoir cote de fluides : l'etude parametrique, point par point.

Le devoir demande de faire varier au moins deux parametres parmi `D`,
`U`, `mu` et `rho`, et d'en dire l'effet sur le Reynolds, les vitesses
caracteristiques, la distribution de pression et la validite du modele.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from polytech.fluides.application import devoir_cylindre_etude as etude
from polytech.fluides.application import devoir_cylindre_fluides as dcf


def test_la_vitesse_maximale_est_le_double_de_l_amont() -> None:
    """Au sommet du cylindre, `|v_theta| = 2 U` (le modele potentiel)."""
    point = etude.Point(dcf.AIR, 10.0, 0.5)
    assert point.vitesse_maximale == pytest.approx(20.0)


def test_l_ecart_de_pression_maximal_vaut_deux_rho_u_carre() -> None:
    """Entre le point d'arret (`C_p = 1`) et le sommet (`C_p = -3`)."""
    point = etude.Point(dcf.EAU, 1.5, 2.0)
    attendu = 2 * dcf.EAU.masse_volumique * 1.5**2
    assert point.ecart_de_pression_maximal == pytest.approx(attendu)


def test_le_coefficient_de_pression_ne_depend_d_aucun_parametre() -> None:
    """Le resultat central de l'etude : `C_p(theta)` est le meme partout.

    La pression est calculee en pascals par Bernoulli, puis rapportee a la
    pression dynamique : l'adimensionnement efface `U`, `D`, `rho`, `mu`.
    """
    angle = math.radians(50.0)
    valeurs = {
        etude.Point(f, u, d).coefficient_de_pression(angle)
        for f in (dcf.AIR, dcf.EAU)
        for u in (0.1, 5.0, 40.0)
        for d in (0.01, 3.0)
    }
    assert max(valeurs) - min(valeurs) < 1e-12
    assert min(valeurs) == pytest.approx(1 - 4 * math.sin(angle) ** 2)


def test_le_reynolds_croit_comme_u_et_comme_d() -> None:
    vitesses = etude.balayage_des_vitesses(dcf.AIR, 0.5, (1.0, 2.0, 4.0))
    rapports = [b.reynolds / a.reynolds for a, b in pairwise(vitesses)]
    assert rapports == pytest.approx([2.0, 2.0])
    diametres = etude.balayage_des_diametres(dcf.AIR, 10.0, (0.1, 1.0))
    assert diametres[1].reynolds / diametres[0].reynolds == pytest.approx(10.0)


def test_changer_de_fluide_change_le_reynolds_du_rapport_des_nu() -> None:
    air, eau = etude.comparaison_des_fluides(1.0, 2.0)
    rapport = dcf.AIR.viscosite_cinematique / dcf.EAU.viscosite_cinematique
    assert eau.reynolds / air.reynolds == pytest.approx(rapport)


def test_le_mach_n_a_de_sens_que_dans_l_air() -> None:
    assert etude.Point(dcf.AIR, 34.32, 1.0).mach == pytest.approx(0.1, rel=1e-3)
    assert etude.Point(dcf.EAU, 1.0, 1.0).mach is None


def test_une_vitesse_ou_un_diametre_nuls_levent() -> None:
    with pytest.raises(ValueError, match="vitesse ou diametre non strictement"):
        etude.Point(dcf.AIR, 0.0, 1.0)
    with pytest.raises(ValueError, match="vitesse ou diametre non strictement"):
        etude.Point(dcf.AIR, 1.0, -2.0)


def test_la_pression_parietale_s_annule_a_trente_et_cent_cinquante_degres() -> None:
    """`1 - 4 sin^2 = 0` : `theta = pi/6` et `5 pi/6`, ou `p = p_inf`."""
    assert etude.angles_de_pression_nulle() == pytest.approx(
        (math.pi / 6, 5 * math.pi / 6)
    )
