"""Syllabus d'electricite generale, chapitre 7 : generalites sur les courants.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 7-1 a 7-13, lues sur les pages rendues.

Le chapitre est litteral. Ses relations sont eprouvees sur le cas qu'il
decrit lui-meme page 7-2 : un conducteur metallique, ou « il n'y a que
deux especes de corpuscules », ions positifs lies au metal et electrons
libres.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.courants import (
    Espece,
    charge_dynamique,
    charge_statique,
    courant_de_conduction,
    courant_de_convection,
    courant_de_polarisation,
    courant_volumique,
    intensite_a_travers,
)
from polytech.electromagnetisme.domain.electrostatique.polarisation import (
    charge_volumique_fictive,
)
from polytech.electromagnetisme.domain.magnetostatique.lois import (
    intensite_depuis_la_densite,
)
from polytech.electromagnetisme.domain.maxwell import (
    conserve_la_charge,
    equation_de_continuite,
)
from polytech.electromagnetisme.domain.symboles import ABSCISSE, PULSATION, TEMPS

DENSITE, CHARGE = sp.symbols("n e", positive=True)
ENTRAINEMENT = (sp.Symbol("v", real=True), 0, 0)
DERIVE = sp.Symbol("u", real=True)
IONS = Espece(DENSITE * CHARGE, ENTRAINEMENT)
ELECTRONS = Espece(-DENSITE * CHARGE, (ENTRAINEMENT[0] + DERIVE, 0, 0))
"""Un metal qui se deplace a la vitesse v, et dont les electrons derivent de u."""


# --- 7.3 Charge statique, charge dynamique -------------------------------------


def test_la_charge_volumique_dynamique_d_une_espece() -> None:
    """(7.3.1) : eta_a = N_a Q_a / Delta V."""
    assert charge_dynamique(nombre=6, charge=-2, volume=3) == -4
    with pytest.raises(ValueError, match="volume strictement positif attendu"):
        charge_dynamique(nombre=6, charge=-2, volume=0)


def test_la_charge_statique_est_la_somme_des_charges_dynamiques() -> None:
    """(7.3.5) : eta = somme des eta_a -- nulle dans un metal neutre."""
    assert charge_statique([IONS, ELECTRONS]) == 0
    assert charge_statique([IONS]) == DENSITE * CHARGE


# --- 7.4 Courants volumiques ---------------------------------------------------


def test_le_courant_de_conduction_ne_voit_que_le_mouvement_relatif() -> None:
    """(7.4.3) : somme des eta_a (v_a - v), les corpuscules lies n'y contribuant pas."""
    conduction = courant_de_conduction([IONS, ELECTRONS], ENTRAINEMENT)
    assert conduction == (-DENSITE * CHARGE * DERIVE, 0, 0)


def test_le_courant_de_convection_d_un_metal_neutre_est_nul() -> None:
    """(7.4.4) : i_cv = eta v, et eta = 0."""
    assert courant_de_convection(charge_statique([IONS, ELECTRONS]), ENTRAINEMENT) == (
        0,
        0,
        0,
    )


def test_le_courant_de_convection_d_un_corps_charge_en_mouvement() -> None:
    """(7.4.4) : un corps de charge volumique eta entraine a la vitesse v."""
    eta = sp.Symbol("eta", real=True)
    assert courant_de_convection(eta, (3, 0, -1)) == (3 * eta, 0, -eta)


def test_la_conduction_et_la_convection_somment_les_especes() -> None:
    """(7.4.15) : i_cd + i_cv = somme sur toutes les especes des eta_a v_a."""
    especes = [IONS, ELECTRONS, Espece(sp.Symbol("eta_2", real=True), (0, 4, 0))]
    conduction = courant_de_conduction(especes, ENTRAINEMENT)
    convection = courant_de_convection(charge_statique(especes), ENTRAINEMENT)
    somme = [
        sum(espece.charge_volumique * espece.vitesse[axe] for espece in especes)
        for axe in range(3)
    ]
    ecarts = [
        sp.simplify(c + v - s)
        for c, v, s in zip(conduction, convection, somme, strict=True)
    ]
    assert ecarts == [0, 0, 0]


def test_le_courant_volumique_est_la_moyenne_des_charges_par_leurs_vitesses() -> None:
    """(7.4.1) : somme des q_i v_i / Delta V, sur les particules elles-memes."""
    particules = [(2, (1, 0, 0)), (2, (1, 0, 0)), (-2, (4, 0, 0)), (-2, (4, 0, 0))]
    microscopique = courant_volumique(particules, volume=2)
    lies, libres = Espece(2, (1, 0, 0)), Espece(-2, (4, 0, 0))
    macroscopique = courant_de_conduction([lies, libres], (1, 0, 0))
    assert microscopique == macroscopique == (-6, 0, 0)


def test_le_courant_de_polarisation() -> None:
    """(7.4.5) : i_p = dP/dt."""
    amplitude = sp.Symbol("P_0", positive=True)
    polarise = (0, 0, amplitude * sp.cos(PULSATION * TEMPS))
    attendu = (0, 0, -amplitude * PULSATION * sp.sin(PULSATION * TEMPS))
    assert courant_de_polarisation(polarise) == attendu
    assert courant_de_polarisation((1, 2, 3)) == (0, 0, 0)


# --- 7.5 Intensite de courant --------------------------------------------------


def test_l_intensite_est_le_flux_du_courant_volumique() -> None:
    """(7.5.1) : I = u_n . i S, pour un courant uniforme sur une surface plane."""
    assert intensite_a_travers((5, 0, 0), normale=(1, 0, 0), aire=2) == 10
    assert intensite_a_travers((5, 0, 0), normale=(0, 1, 0), aire=2) == 0
    oblique = (sp.Rational(3, 5), sp.Rational(4, 5), 0)
    assert intensite_a_travers((5, 0, 0), normale=oblique, aire=2) == 6


def test_l_intensite_recoupe_la_densite_de_courant_du_fascicule() -> None:
    """(7.5.6) : i = dI/dS quand le courant est normal a la section -- I = J S."""
    assert intensite_a_travers((0, 0, 2.5e6), normale=(0, 0, 1), aire=1e-6) == (
        intensite_depuis_la_densite(2.5e6, 1e-6)
    )


def test_l_intensite_refuse_une_normale_non_unitaire() -> None:
    """Le flux se prend sur un vecteur unitaire."""
    with pytest.raises(ValueError, match="normale unitaire attendue"):
        intensite_a_travers((5, 0, 0), normale=(2, 0, 0), aire=2)


def test_l_intensite_est_la_charge_reelle_qui_traverse_par_unite_de_temps() -> None:
    """Theoreme de la page 7-10 : pendant dt passe la charge du volume u_n dS . v dt."""
    duree, aire = sp.symbols("dt S", positive=True)
    courant = courant_de_conduction([IONS, ELECTRONS], ENTRAINEMENT)
    balayee = ELECTRONS.charge_volumique * DERIVE * duree * aire
    assert (
        sp.simplify(intensite_a_travers(courant, (1, 0, 0), aire) * duree - balayee)
        == 0
    )


# --- 7.6 Equation de continuite ------------------------------------------------


def test_un_paquet_de_charge_en_translation_conserve_la_charge() -> None:
    """(7.6.4) : div(i_cd + i_cv) + d eta/dt = 0, pour eta(x - v t) entrainee a v."""
    vitesse = sp.Symbol("v", positive=True)
    eta = sp.exp(-((ABSCISSE - vitesse * TEMPS) ** 2))
    assert conserve_la_charge(courant_de_convection(eta, (vitesse, 0, 0)), eta)


def test_un_courant_qui_ne_suit_pas_la_charge_viole_la_continuite() -> None:
    """Le meme paquet, « transporte » deux fois trop vite."""
    vitesse = sp.Symbol("v", positive=True)
    eta = sp.exp(-((ABSCISSE - vitesse * TEMPS) ** 2))
    assert not conserve_la_charge(courant_de_convection(eta, (2 * vitesse, 0, 0)), eta)


def test_le_courant_de_polarisation_conserve_la_charge_fictive() -> None:
    """Remarque 1 de la page 7-12 : div i_p + d eta'/dt = 0, avec eta' = -div P."""
    polarise = (sp.sin(ABSCISSE) * sp.cos(PULSATION * TEMPS), 0, ABSCISSE * TEMPS**2)
    fictive = charge_volumique_fictive(polarise)
    assert equation_de_continuite(courant_de_polarisation(polarise), fictive) == 0


def test_la_continuite_d_un_courant_lineique() -> None:
    """Remarque 3 de la page 7-13 : dI/dl + d lambda/dt = 0, avec I = lambda v."""
    vitesse = sp.Symbol("v", positive=True)
    lineique = sp.cos(ABSCISSE - vitesse * TEMPS)
    intensite = lineique * vitesse
    assert sp.simplify(sp.diff(intensite, ABSCISSE) + sp.diff(lineique, TEMPS)) == 0
