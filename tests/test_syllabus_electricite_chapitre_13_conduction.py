"""Syllabus d'electricite generale, chapitre 13 : la conduction dans la matiere.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 13-2 a 13-9, lues sur les pages rendues.

Les nombres sont ceux du cuivre que la page 13-2 donne : 15 A/mm^2,
13,6 C/mm^3, un nombre de masse de 63. La loi d'Ohm (13.3.5) et la loi de
Joule (13.4.2) sont recoupees en repartant des especes de charges et de
leur frottement `h_a`, sans passer par la conductivite.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.conduction import (
    CHARGE_LIBRE_DU_CUIVRE,
    champ_au_bord_du_dielectrique,
    champ_d_ohm,
    charge_volumique_apparente,
    conductivite,
    puissance_volumique_de_joule,
    rapport_des_masses_volumiques,
    vitesse_relative,
)
from polytech.electromagnetisme.domain.courants import Espece, courant_de_conduction
from polytech.electromagnetisme.domain.effet_hall import (
    constante_de_hall,
    tension_de_hall,
    vitesse_des_porteurs,
)
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.symboles import ABSCISSE, CARTESIENNES


def test_les_electrons_du_cuivre_avancent_d_un_millimetre_par_seconde() -> None:
    """Page 13-2 : `i / eta = 15 / 13,6 = 1,1 mm/s`, en unites SI."""
    libre = CHARGE_LIBRE_DU_CUIVRE
    assert libre == pytest.approx(13.6e9)
    vitesse = vitesse_relative(15e6, CHARGE_LIBRE_DU_CUIVRE)
    assert vitesse == pytest.approx(1.1e-3, rel=5e-3)


def test_un_milieu_sans_charge_mobile_n_a_pas_de_vitesse() -> None:
    with pytest.raises(ValueError, match="charge volumique nulle"):
        vitesse_relative(15e6, 0.0)


def test_les_charges_mobiles_pesent_cent_seize_mille_fois_moins() -> None:
    """Page 13-2 : `63 x 1 836`. Le syllabus imprime « 16 000 » : un chiffre manque."""
    assert rapport_des_masses_volumiques(63) == 115_668
    assert rapport_des_masses_volumiques(63) != pytest.approx(16_000, rel=0.5)


def test_la_conductivite_somme_les_carres_des_charges_sur_les_frottements() -> None:
    """(13.3.3) : `gamma = somme des eta_a^2 / h_a`."""
    assert conductivite([(2.0, 4.0), (-3.0, 18.0)]) == pytest.approx(1.5)


def test_un_frottement_nul_n_a_pas_de_conductivite() -> None:
    with pytest.raises(ValueError, match="frottement nul ou negatif"):
        conductivite([(2.0, 0.0)])


def test_la_loi_d_ohm_sort_de_l_equilibre_des_especes() -> None:
    """(13.3.1) sans le terme de Hall : `eta_a E = h_a (v_a - v)`, puis (13.3.4).

    Deux especes de signes contraires derivent en sens opposes et leurs
    courants s'ajoutent.
    """
    champ, especes = sp.Rational(3), [(2, 4), (-3, 18)]
    en_derive = [
        Espece(eta, (sp.Rational(eta) * champ / frottement, 0, 0))
        for eta, frottement in especes
    ]
    courant = courant_de_conduction(en_derive, (0, 0, 0))
    resistivite = 1 / sp.nsimplify(conductivite(especes))
    assert champ_d_ohm(resistivite, courant) == (champ, 0, 0)


def test_la_loi_de_joule_est_la_puissance_des_frottements() -> None:
    """(13.4.1) `p_j = somme (v_a - v) . f_ra`, et (13.4.2) `rho i^2`."""
    champ, especes = sp.Rational(3), [(2, 4), (-3, 18)]
    frottements = sum(
        frottement * (sp.Rational(eta) * champ / frottement) ** 2
        for eta, frottement in especes
    )
    gamma = sp.nsimplify(conductivite(especes))
    joule = puissance_volumique_de_joule(1 / gamma, (gamma * champ, 0, 0))
    assert joule == frottements


def test_la_puissance_de_joule_ne_depend_pas_du_sens_du_courant() -> None:
    """Page 13-5 : « une quantite positive, quel que soit le sens »."""
    aller = puissance_volumique_de_joule(17e-9, (0.0, 3e6, 0.0))
    retour = puissance_volumique_de_joule(17e-9, (0.0, -3e6, 0.0))
    assert aller == retour == pytest.approx(17e-9 * 9e12)


def test_un_conducteur_homogene_au_repos_ne_porte_aucune_charge_volumique() -> None:
    """(13.5.1) : « la charge de ce conducteur est tout entiere sur sa surface »."""
    charge = charge_volumique_apparente(
        sp.Symbol("rho", positive=True),
        (sp.Symbol("i"), 0, 0),
        (0, 0, 0),
        (0, 0, 0),
    )
    assert charge == 0


def test_un_gradient_de_resistivite_retient_de_la_charge() -> None:
    """(13.5.1) recoupee par `eta = eps_0 div E`, avec `E = rho(x) i`."""
    base, pente, courant = sp.symbols("rho_0 a i", positive=True)
    resistivite = base * (1 + pente * ABSCISSE)
    charge = charge_volumique_apparente(
        resistivite, (courant, 0, 0), (0, 0, 0), (0, 0, 0)
    )
    champ = champ_d_ohm(resistivite, (courant, 0, 0))
    divergence = sum(sp.diff(c, x) for c, x in zip(champ, CARTESIENNES, strict=True))
    assert sp.simplify(charge - PERMITTIVITE * divergence) == 0
    assert sp.simplify(charge - PERMITTIVITE * courant * base * pente) == 0


def test_la_convection_et_le_champ_electromoteur_comptent_en_moins() -> None:
    """Les deux derniers termes de (13.5.1)."""
    rho = sp.Symbol("rho", positive=True)
    convection = charge_volumique_apparente(
        rho, (0, 0, 0), (2 * ABSCISSE, 0, 0), (0, 0, 0)
    )
    assert convection == -2 * PERMITTIVITE * rho
    electromoteur = charge_volumique_apparente(
        rho, (0, 0, 0), (0, 0, 0), (5 * ABSCISSE, 0, 0)
    )
    assert electromoteur == -5 * PERMITTIVITE


def test_le_champ_dans_le_dielectrique_s_incline_sur_la_surface() -> None:
    """(13.6.4) : tangentiel `rho i` (13.6.2), normal `sigma / epsilon` (13.6.3)."""
    tangentiel, normal = champ_au_bord_du_dielectrique(
        17e-9, 4e6, charge_surfacique=8.854e-12, permittivite=8.854e-12
    )
    assert tangentiel == pytest.approx(0.068)
    assert normal == pytest.approx(1.0)
    sans_courant = champ_au_bord_du_dielectrique(
        17e-9, 0.0, charge_surfacique=8.854e-12, permittivite=8.854e-12
    )
    # La remarque de la page 13-7 : l'electrostatique, ou le champ est normal.
    assert sans_courant[0] == 0


def test_une_permittivite_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="permittivite nulle ou negative"):
        champ_au_bord_du_dielectrique(
            17e-9, 4e6, charge_surfacique=1e-12, permittivite=0.0
        )


def test_la_constante_de_hall_des_electrons_est_positive_ici() -> None:
    """(13.7.4) : `R_H = -1 / (n q)`, `q` etant negatif."""
    assert constante_de_hall(-CHARGE_LIBRE_DU_CUIVRE) == pytest.approx(
        7.35e-11, rel=1e-3
    )
    # Page 13-9 : des « trous » positifs renversent le champ transversal.
    assert constante_de_hall(CHARGE_LIBRE_DU_CUIVRE) < 0


def test_la_tension_de_hall_du_cuivre_est_de_l_ordre_du_microvolt() -> None:
    """Page 13-8 : un ampere, un tesla, un dixieme de millimetre."""
    constante = constante_de_hall(-CHARGE_LIBRE_DU_CUIVRE)
    tension = tension_de_hall(constante, courant=1.0, induction=1.0, epaisseur=1e-4)
    assert tension == pytest.approx(0.735e-6, rel=1e-3)


def test_la_tension_de_hall_equilibre_la_force_magnetique() -> None:
    """(13.7.1) `v B = U_H / h` et (13.7.2) : la largeur `h` s'elimine."""
    charge = -CHARGE_LIBRE_DU_CUIVRE
    attendue = tension_de_hall(constante_de_hall(charge), 2.0, 0.5, 2e-4)
    for largeur in (1e-3, 5e-3):
        vitesse = vitesse_des_porteurs(2.0, largeur, 2e-4, charge)
        assert vitesse > 0
        assert vitesse * 0.5 * largeur == pytest.approx(attendue)


def test_un_semi_conducteur_donne_une_tension_un_million_de_fois_plus_grande() -> None:
    """Page 13-8 : un million de fois moins de porteurs."""
    metal = constante_de_hall(-CHARGE_LIBRE_DU_CUIVRE)
    semi_conducteur = constante_de_hall(-CHARGE_LIBRE_DU_CUIVRE / 1e6)
    assert semi_conducteur / metal == pytest.approx(1e6)


def test_les_gardes_de_l_effet_hall() -> None:
    with pytest.raises(ValueError, match="aucun porteur"):
        constante_de_hall(0.0)
    with pytest.raises(ValueError, match="epaisseur nulle ou negative"):
        tension_de_hall(1e-10, courant=1.0, induction=1.0, epaisseur=0.0)
    with pytest.raises(ValueError, match="section nulle ou negative"):
        vitesse_des_porteurs(1.0, 0.0, 1e-4, -1e9)
    with pytest.raises(ValueError, match="section nulle ou negative"):
        vitesse_des_porteurs(1.0, 1e-3, 0.0, -1e9)
    with pytest.raises(ValueError, match="aucun porteur"):
        vitesse_des_porteurs(1.0, 1e-3, 1e-4, 0.0)
