"""Syllabus d'electricite generale, chapitre 4 : conducteurs et milieux isolants.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 4-1 a 4-18, lues sur les pages rendues.

Un ecart, constate en agrandissant la page 4-18 : le paragraphe 4.12 donne
a la terre une charge surfacique `sigma = eps_0 E` « de l'ordre de
0,9 C/m^2 » pour un champ de 100 V/m. Le produit vaut 0,885 . 10^-9 C/m^2 :
c'est 0,9 **nano**coulomb par metre carre, le prefixe est tombe.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.constantes import PERMITTIVITE_DU_VIDE
from polytech.electromagnetisme.domain.electrostatique.avancee import (
    capacite_plane,
    permittivite,
)
from polytech.electromagnetisme.domain.electrostatique.conducteurs import (
    charge_surfacique_depuis_le_champ,
    charge_surfacique_maximale,
)
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.dielectriques import (
    CHAMP_DISRUPTIF_DE_L_AIR,
    PERMITTIVITE_RELATIVE_DE_L_AIR,
    charge_apparente,
    charge_fictive,
    energie_volumique,
    polarisation_depuis_le_champ,
    polarisation_depuis_le_deplacement,
    susceptibilite,
    travail_volumique_du_cycle,
)
from polytech.electromagnetisme.domain.electrostatique.polarisation import deplacement
from polytech.electromagnetisme.domain.energie_et_forces import (
    densite_d_energie_electrique,
)
from polytech.electromagnetisme.domain.symboles import PULSATION, TEMPS

EPS_R = sp.Symbol("varepsilon_r", positive=True)


# --- 4.2 Permittivite ----------------------------------------------------------


def test_la_susceptibilite_electrique() -> None:
    """(4.2.5) : chi = eps_r - 1 ; nulle dans le vide."""
    assert susceptibilite(EPS_R) == EPS_R - 1
    assert susceptibilite(1) == 0
    assert susceptibilite(sp.Rational(5, 2)) == sp.Rational(3, 2)


def test_une_permittivite_relative_inferieure_a_un_est_refusee() -> None:
    """Page 4-5 : « eps >= eps_0 et eps_r >= 1 », P etant dirige suivant E."""
    with pytest.raises(ValueError, match="permittivite relative inferieure a 1"):
        susceptibilite(sp.Rational(1, 2))


def test_la_polarisation_d_un_dielectrique_lineaire() -> None:
    """(4.2.4) : P = chi eps_0 E, et alors D = eps_0 E + P = eps E (4.2.2)."""
    champ = (2, 0, -1)
    polarise = polarisation_depuis_le_champ(EPS_R, champ)
    assert polarise == tuple((EPS_R - 1) * PERMITTIVITE * c for c in champ)
    attendu = tuple(EPS_R * PERMITTIVITE * c for c in champ)
    obtenu = deplacement(champ, polarise)
    assert [sp.simplify(a - b) for a, b in zip(obtenu, attendu, strict=True)] == [0] * 3


def test_la_polarisation_en_fonction_du_deplacement() -> None:
    """(4.4.1) : P = (1 - eps_0 / eps) D."""
    d = sp.Symbol("D", real=True)
    assert (
        sp.simplify(polarisation_depuis_le_deplacement(EPS_R, d) - (1 - 1 / EPS_R) * d)
        == 0
    )
    champ = d / (EPS_R * PERMITTIVITE)
    par_le_champ, _, _ = polarisation_depuis_le_champ(EPS_R, (champ, 0, 0))
    assert sp.simplify(par_le_champ - polarisation_depuis_le_deplacement(EPS_R, d)) == 0


def test_la_permittivite_absolue() -> None:
    """(4.2.3) : eps = eps_r eps_0 -- le quadruple pour eps_r = 4."""
    assert math.isclose(permittivite(4.0), 4 * PERMITTIVITE_DU_VIDE)


# --- 4.3 Energie potentielle ---------------------------------------------------


def test_l_energie_volumique_d_un_milieu_lineaire() -> None:
    """(4.3.11) : w_e = D^2 / (2 eps) = eps E^2 / 2 = E . D / 2."""
    d, eps = sp.symbols("D varepsilon", positive=True)
    energie = energie_volumique(d / eps, d)
    assert energie == d**2 / (2 * eps)
    champ = d / eps
    assert sp.simplify(energie - eps * champ**2 / 2) == 0
    assert sp.simplify(energie - champ * d / 2) == 0


def test_l_energie_volumique_recoupe_celle_du_vide() -> None:
    """(2.21.14), que `energie_et_forces` portait deja : eps_0 E^2 / 2."""
    d = sp.Symbol("D", positive=True)
    champ = 3.0e5
    par_integration = energie_volumique(d / PERMITTIVITE_DU_VIDE, d).subs(
        d, PERMITTIVITE_DU_VIDE * champ
    )
    assert math.isclose(
        float(par_integration),
        densite_d_energie_electrique(PERMITTIVITE_DU_VIDE, champ),
    )


def test_hors_du_lineaire_l_energie_n_est_plus_le_demi_produit() -> None:
    """(4.3.8) : w = integrale de E dD ; E . D / 2 suppose E lineaire en D."""
    d = sp.Symbol("D", positive=True)
    champ = d + d**3
    energie = energie_volumique(champ, d)
    assert energie == d**2 / 2 + d**4 / 4
    assert sp.simplify(energie - champ * d / 2) != 0


# --- 4.4 et 4.6 Champ en presence de dielectriques -----------------------------


def test_la_charge_fictive_est_proportionnelle_a_la_charge_reelle() -> None:
    """(4.4.2) et (4.4.4) : eta' = -(1 - eps_0 / eps) eta, de meme pour sigma'."""
    charge = sp.Symbol("eta", real=True)
    assert sp.simplify(charge_fictive(EPS_R, charge) + (1 - 1 / EPS_R) * charge) == 0
    assert charge_fictive(1, charge) == 0
    assert charge_fictive(4, 8) == -6


def test_la_charge_apparente_est_la_charge_reelle_divisee_par_eps_r() -> None:
    """(4.4.3) et (4.4.5) : eta + eta' = (eps_0 / eps) eta."""
    charge = sp.Symbol("eta", real=True)
    apparente = charge_apparente(EPS_R, charge)
    assert sp.simplify(apparente - charge / EPS_R) == 0
    assert sp.simplify(apparente - charge - charge_fictive(EPS_R, charge)) == 0
    assert charge_apparente(4, 8) == 2


def test_un_dielectrique_multiplie_la_capacite() -> None:
    """Page 4-12 : la capacite vaut « eps_r fois » celle du condensateur a vide."""
    assert math.isclose(
        capacite_plane(2e-2, 1e-3, relative=4.0), 4 * capacite_plane(2e-2, 1e-3)
    )


def test_a_charges_constantes_le_dielectrique_divise_le_champ() -> None:
    """Page 4-12 : D inchange, V et E divises par eps_r."""
    sigma = sp.Symbol("sigma", positive=True)
    dans_le_vide = sigma / PERMITTIVITE
    dans_le_dielectrique = charge_apparente(EPS_R, sigma) / PERMITTIVITE
    assert sp.simplify(dans_le_dielectrique - dans_le_vide / EPS_R) == 0


# --- 4.8 Dielectriques reels, hysteresis ---------------------------------------


def test_le_travail_volumique_d_un_cycle_elliptique() -> None:
    """Page 4-15 : l'integrale de E dD sur un cycle est l'aire qu'il entoure."""
    e_0, d_0, retard = sp.symbols("E_0 D_0 delta", positive=True)
    champ = e_0 * sp.cos(PULSATION * TEMPS)
    induction = d_0 * sp.cos(PULSATION * TEMPS - retard)
    travail = travail_volumique_du_cycle(champ, induction, 2 * sp.pi / PULSATION)
    assert sp.simplify(travail - sp.pi * e_0 * d_0 * sp.sin(retard)) == 0


def test_un_dielectrique_sans_retard_ne_dissipe_rien() -> None:
    """Un milieu lineaire et parfait parcourt une droite : l'aire est nulle."""
    e_0, eps = sp.symbols("E_0 varepsilon", positive=True)
    champ = e_0 * sp.cos(PULSATION * TEMPS)
    assert travail_volumique_du_cycle(champ, eps * champ, 2 * sp.pi / PULSATION) == 0


# --- 4.10 a 4.12 Disruption, condensateurs, champ terrestre --------------------


def test_le_champ_disruptif_de_l_air() -> None:
    """Pages 4-17 et 4-18 : « quelque 3 MV/m », et eps_r « environ 1,000 6 »."""
    assert CHAMP_DISRUPTIF_DE_L_AIR == 3e6
    assert PERMITTIVITE_RELATIVE_DE_L_AIR == 1.0006
    assert math.isclose(
        permittivite(PERMITTIVITE_RELATIVE_DE_L_AIR), PERMITTIVITE_DU_VIDE, rel_tol=1e-3
    )


def test_la_charge_maximale_recoupe_le_champ_disruptif() -> None:
    """Remarque de la page 3-5 : 3 MV/m, donc environ 26 microcoulombs par m^2."""
    maximale = charge_surfacique_maximale(CHAMP_DISRUPTIF_DE_L_AIR)
    assert maximale == charge_surfacique_depuis_le_champ(CHAMP_DISRUPTIF_DE_L_AIR)


def test_la_charge_surfacique_de_la_terre() -> None:
    """4.12 : 100 V/m par beau temps -- 0,9 nanocoulomb par m^2, et non 0,9 C/m^2."""
    sigma = float(
        charge_surfacique_depuis_le_champ(100).subs(PERMITTIVITE, PERMITTIVITE_DU_VIDE)
    )
    assert math.isclose(sigma, 0.9e-9, rel_tol=0.02)
    assert not math.isclose(sigma, 0.9, rel_tol=0.5)
