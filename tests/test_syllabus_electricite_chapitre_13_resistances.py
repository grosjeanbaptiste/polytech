"""Syllabus d'electricite generale, chapitre 13 : resistance, resistivite, puissance.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 13-9 a 13-26, lues sur les pages rendues.

Deux oracles chiffres viennent du cours : le tableau 13.10.1 et la figure
13.15.2 (`E_s = 4 V`, `E_r = 1,5 V`, `R_s = 1 ohm`, les trois autres
resistances de `0,5 ohm`). Le circuit simple est en outre resolu une
seconde fois par l'analyse nodale, qui ne choisit aucune maille.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.circuit_simple import CircuitSimple
from polytech.electromagnetisme.domain.circuits.reseau import (
    Reseau,
    Resistance,
    SourceDeTension,
)
from polytech.electromagnetisme.domain.circuits.resistance import (
    conductance_d_un_arc,
    conductance_en_parallele,
    conductance_en_serie,
    resistance_d_un_tube,
    resistance_uniforme,
)
from polytech.electromagnetisme.domain.circuits.resistivite import (
    METAUX,
    RESISTIVITE_DU_VERRE,
    resistance_d_un_ruban,
    resistivite_a_chaud,
    resistivite_superficielle,
)
from polytech.electromagnetisme.domain.circuits.sinusoidaux import (
    en_parallele,
    en_serie,
)
from polytech.electromagnetisme.domain.circuits.troncon import Troncon
from polytech.electromagnetisme.domain.magnetostatique.reluctances import (
    reluctance_d_un_tube,
)

S = sp.Symbol("s", nonnegative=True)
CUIVRE, ALPHA_DU_CUIVRE = METAUX["cuivre"]


def test_le_tableau_13_10_1_est_recopie() -> None:
    """Page 13-16 : resistivite a 20 degres en ohm-metre, coefficient par degre."""
    assert METAUX == {
        "cuivre": (17e-9, 0.004),
        "aluminium": (27e-9, 0.004),
        "fer": (100e-9, 0.006),
        "manganine": (470e-9, 0.00001),
        "graphite": (10e-6, -0.001),
    }
    assert RESISTIVITE_DU_VERRE / CUIVRE > 1e19


def test_la_resistance_d_un_fil_homogene_est_rho_l_sur_s() -> None:
    """(13.9.5) : cent metres de cuivre de 1,5 mm^2."""
    assert resistance_uniforme(CUIVRE, 100.0, 1.5e-6) == pytest.approx(17 / 15)


def test_une_geometrie_ou_une_resistivite_non_positive_est_refusee() -> None:
    with pytest.raises(ValueError, match="resistivite nulle ou negative"):
        resistance_uniforme(0.0, 100.0, 1.5e-6)
    with pytest.raises(ValueError, match="geometrie invalide"):
        resistance_uniforme(CUIVRE, 0.0, 1.5e-6)
    with pytest.raises(ValueError, match="geometrie invalide"):
        resistance_uniforme(CUIVRE, 100.0, 0.0)


def test_la_resistance_est_la_reluctance_ou_gamma_remplace_mu() -> None:
    """Remarque de la page 13-13, sur le tube conique de la figure 13.8.1."""
    rho, longueur, r1, r2 = sp.symbols("rho l r_1 r_2", positive=True)
    section = sp.pi * (r1 + (r2 - r1) * S / longueur) ** 2
    resistance = resistance_d_un_tube(section, S, longueur, rho)
    assert sp.simplify(resistance - rho * longueur / (sp.pi * r1 * r2)) == 0
    reluctance = reluctance_d_un_tube(section, S, longueur, 1 / rho)
    assert sp.simplify(resistance - reluctance) == 0


def test_un_arc_de_couronne_se_calcule_par_tubes_en_parallele() -> None:
    """(13.9.7), figure 13.9.2b : `dG = h dr / (rho theta r)`.

    Chaque tube d'epaisseur `dr` a la longueur `theta r` ; les conductances
    s'ajoutent, d'ou `G = h ln(r_2 / r_1) / (rho theta)`.
    """
    conductance = conductance_d_un_arc(
        2e-8, hauteur=1e-3, interieur=0.01, exterieur=0.02, angle=math.pi
    )
    assert conductance == pytest.approx(1e-3 * math.log(2) / (2e-8 * math.pi))


def test_un_arc_mince_retrouve_rho_l_sur_s() -> None:
    """Quand `r_2 - r_1` est petit, la longueur est celle du rayon moyen."""
    mince = conductance_d_un_arc(
        2e-8, hauteur=1e-3, interieur=0.1, exterieur=0.1001, angle=1.0
    )
    droite = resistance_uniforme(2e-8, 0.10005, 1e-3 * 1e-4)
    assert 1 / mince == pytest.approx(droite, rel=1e-6)


def test_un_arc_dont_les_rayons_sont_inverses_est_refuse() -> None:
    with pytest.raises(ValueError, match="rayons invalides"):
        conductance_d_un_arc(
            2e-8, hauteur=1e-3, interieur=0.02, exterieur=0.01, angle=1.0
        )
    with pytest.raises(ValueError, match="angle nul ou negatif"):
        conductance_d_un_arc(
            2e-8, hauteur=1e-3, interieur=0.01, exterieur=0.02, angle=0.0
        )


def test_les_conductances_s_associent_a_l_inverse_des_resistances() -> None:
    """(13.9.3) et (13.9.4), contre (13.9.1) et (13.9.2)."""
    resistances = (2.0, 3.0, 6.0)
    conductances = tuple(1 / r for r in resistances)
    assert 1 / conductance_en_serie(*conductances) == pytest.approx(
        float(en_serie(*(sp.Float(r) for r in resistances)))
    )
    assert 1 / conductance_en_parallele(*conductances) == pytest.approx(
        float(en_parallele(*(sp.Float(r) for r in resistances)))
    )
    assert 1 / conductance_en_parallele(*conductances) == pytest.approx(1.0)


def test_une_conductance_nulle_ne_s_associe_pas() -> None:
    with pytest.raises(ValueError, match="conductance nulle ou negative"):
        conductance_en_serie(0.5, 0.0)
    with pytest.raises(ValueError, match="conductance nulle ou negative"):
        conductance_en_parallele(0.5, -1.0)


def test_le_cuivre_resiste_un_cinquieme_de_plus_cinquante_degres_plus_haut() -> None:
    """(13.10.3) : `rho(t_2) = rho(t_1) [1 + alpha (t_2 - t_1)]`."""
    assert resistivite_a_chaud(CUIVRE, 50.0, ALPHA_DU_CUIVRE) == pytest.approx(
        1.2 * CUIVRE
    )


def test_la_manganine_ne_bouge_presque_pas_et_le_graphite_baisse() -> None:
    """Page 13-16 : des resistances tres stables ; `alpha` negatif du carbone."""
    manganine, alpha = METAUX["manganine"]
    assert resistivite_a_chaud(manganine, 50.0, alpha) / manganine == pytest.approx(
        1.0005
    )
    graphite, alpha = METAUX["graphite"]
    assert resistivite_a_chaud(graphite, 50.0, alpha) < graphite


def test_le_terme_quadratique_s_ajoute_au_lineaire() -> None:
    """(13.10.2) : `1 + alpha dt + beta dt^2`."""
    assert resistivite_a_chaud(1.0, 10.0, 0.004, beta=1e-5) == pytest.approx(1.041)


def test_la_resistivite_superficielle_est_la_resistance_d_un_carre() -> None:
    """(13.10.4) et (13.10.5), que (13.9.5) recoupe : `R = rho l / (e b)`.

    Le syllabus imprime `dR = rho_S dS / db` ; un carre de cote quelconque
    doit rendre `rho_S`, ce que seul `dl / db` permet.
    """
    superficielle = resistivite_superficielle(CUIVRE, 35e-6)
    ruban = resistance_d_un_ruban(superficielle, longueur=0.1, largeur=2e-3)
    assert ruban == pytest.approx(resistance_uniforme(CUIVRE, 0.1, 35e-6 * 2e-3))
    for cote in (1.0, 3e-3):
        assert resistance_d_un_ruban(superficielle, cote, cote) == pytest.approx(
            superficielle
        )


def test_une_couche_sans_epaisseur_ou_un_ruban_sans_largeur_sont_refuses() -> None:
    with pytest.raises(ValueError, match="epaisseur nulle ou negative"):
        resistivite_superficielle(CUIVRE, 0.0)
    with pytest.raises(ValueError, match="largeur nulle ou negative"):
        resistance_d_un_ruban(5e-4, longueur=0.1, largeur=0.0)


def test_la_loi_d_ohm_integree_d_un_troncon() -> None:
    """(13.8.10) `U_12 = E_21 + R_12 I_12`, et (13.8.15) a vide."""
    troncon = Troncon(fem=1.5, resistance=0.5)
    assert troncon.tension(1.0) == pytest.approx(2.0)
    assert troncon.tension(0.0) == pytest.approx(1.5)
    assert troncon.courant(2.0) == pytest.approx(1.0)


def test_un_troncon_recoit_ce_qu_il_convertit_et_ce_qu_il_dissipe() -> None:
    """(13.13.1) a (13.13.4) : `P_12 = E_21 I_12 + R_12 I_12^2 = U_12 I_12`."""
    troncon = Troncon(fem=1.5, resistance=0.5)
    assert troncon.puissance_convertie(1.0) == pytest.approx(1.5)
    assert troncon.puissance_de_joule(1.0) == pytest.approx(0.5)
    assert troncon.puissance_recue(1.0) == pytest.approx(troncon.tension(1.0) * 1.0)
    assert troncon.puissance_recue(1.0) == pytest.approx(2.0)


def test_le_signe_de_la_puissance_dit_recepteur_ou_source() -> None:
    """Figure 13.13.1 : `U_12 > E_21` recoit, `0 < U_12 < E_21` fournit."""
    troncon = Troncon(fem=4.0, resistance=1.0)
    en_charge = troncon.courant(5.0)
    assert en_charge > 0
    assert troncon.est_recepteur(en_charge)
    en_source = troncon.courant(3.0)
    assert en_source < 0
    assert not troncon.est_recepteur(en_source)
    # L'effet Joule, lui, ne change pas de signe.
    assert troncon.puissance_de_joule(en_source) > 0


def test_un_troncon_sans_resistance_ne_donne_pas_son_courant() -> None:
    with pytest.raises(ValueError, match="resistance negative"):
        Troncon(fem=1.0, resistance=-1.0)
    with pytest.raises(ValueError, match="troncon sans resistance"):
        Troncon(fem=1.0, resistance=0.0).courant(2.0)


FIGURE = CircuitSimple(
    source=Troncon(fem=4.0, resistance=1.0),
    recepteur=Troncon(fem=1.5, resistance=0.5),
    ligne=(0.5, 0.5),
)
"""La figure 13.15.2."""


def test_le_circuit_de_la_figure_13_15_2_debite_un_ampere() -> None:
    """(13.15.1), (13.15.2), (13.15.5) et (13.15.7)."""
    assert FIGURE.courant() == pytest.approx(1.0)
    assert FIGURE.tension_de_la_source() == pytest.approx(3.0)
    assert FIGURE.chute_en_ligne() == pytest.approx(1.0)
    assert FIGURE.tension_du_recepteur() == pytest.approx(2.0)
    assert FIGURE.recepteur.tension(FIGURE.courant()) == pytest.approx(2.0)


def test_l_analyse_nodale_retrouve_le_circuit_simple() -> None:
    """Un second chemin : la loi des noeuds, sans aucune maille."""
    reseau = Reseau(
        (
            SourceDeTension("a", "1", sp.Integer(4)),
            Resistance("a", "2", sp.Integer(1)),
            Resistance("2", "3", sp.Rational(1, 2)),
            Resistance("3", "b", sp.Rational(1, 2)),
            SourceDeTension("b", "4", sp.Rational(3, 2)),
            Resistance("4", "1", sp.Rational(1, 2)),
        ),
        reference="1",
    )
    potentiels = reseau.potentiels()
    assert potentiels["2"] == pytest.approx(FIGURE.tension_de_la_source())
    assert potentiels["3"] - potentiels["4"] == pytest.approx(
        FIGURE.tension_du_recepteur()
    )


def test_le_bilan_des_puissances_se_referme() -> None:
    """(13.15.4), (13.15.6), (13.15.8), (13.15.9) et le bilan (13.15.10)."""
    assert FIGURE.puissance_fournie() == pytest.approx(3.0)
    assert FIGURE.puissance_de_la_ligne() == pytest.approx(1.0)
    assert FIGURE.puissance_du_recepteur() == pytest.approx(2.0)
    assert FIGURE.puissance_utile() == pytest.approx(1.5)
    joule = 1.0 + FIGURE.puissance_de_la_ligne() + 0.5
    assert FIGURE.puissance_utile() == pytest.approx(4.0 * 1.0 - joule)


def test_la_puissance_recue_par_la_source_est_l_opposee_de_la_fournie() -> None:
    """(13.15.3) imprime `-E_s I - R_s I^2`, soit -5 W ; (13.15.4) dit +3 W fournis.

    `P_12 = U_21 I_21` avec `I_21 = -I` : le second signe est un plus.
    """
    recue = FIGURE.tension_de_la_source() * -FIGURE.courant()
    assert recue == pytest.approx(-FIGURE.puissance_fournie())
    assert recue == pytest.approx(-4.0 * 1.0 + 1.0 * 1.0**2)
    assert recue != pytest.approx(-4.0 * 1.0 - 1.0 * 1.0**2)


def test_un_recepteur_plus_fort_que_la_source_inverse_les_roles() -> None:
    """Page 13-24 : « sinon les roles source et recepteur seraient inverses »."""
    with pytest.raises(ValueError, match="roles inverses"):
        CircuitSimple(
            source=Troncon(fem=1.5, resistance=1.0),
            recepteur=Troncon(fem=4.0, resistance=0.5),
            ligne=(0.5, 0.5),
        )


def test_a_forces_electromotrices_egales_la_source_est_a_vide() -> None:
    """Page 13-24 : « circuit ouvert, ou bien `E_r = E_s` »."""
    equilibre = CircuitSimple(
        source=Troncon(fem=4.0, resistance=1.0),
        recepteur=Troncon(fem=4.0, resistance=0.5),
        ligne=(0.5, 0.5),
    )
    assert equilibre.courant() == 0
    assert equilibre.tension_de_la_source() == pytest.approx(4.0)


def test_un_circuit_sans_aucune_resistance_est_refuse() -> None:
    with pytest.raises(ValueError, match="circuit sans resistance"):
        CircuitSimple(
            source=Troncon(fem=4.0, resistance=0.0),
            recepteur=Troncon(fem=1.5, resistance=0.0),
            ligne=(0.0, 0.0),
        )


def test_un_arc_sans_epaisseur_sans_trou_ou_sans_resistivite_est_refuse() -> None:
    with pytest.raises(ValueError, match="rayons invalides"):
        conductance_d_un_arc(
            2e-8, hauteur=1e-3, interieur=0.01, exterieur=0.01, angle=1.0
        )
    with pytest.raises(ValueError, match="rayons invalides"):
        conductance_d_un_arc(
            2e-8, hauteur=1e-3, interieur=0.0, exterieur=0.01, angle=1.0
        )
    with pytest.raises(ValueError, match="resistivite nulle ou negative"):
        conductance_d_un_arc(
            0.0, hauteur=1e-3, interieur=0.01, exterieur=0.02, angle=1.0
        )


def test_a_vide_un_troncon_n_est_ni_source_ni_recepteur() -> None:
    """(13.8.15) : aucun courant, donc aucune puissance."""
    assert not Troncon(fem=4.0, resistance=1.0).est_recepteur(0.0)
