"""Syllabus de mathematiques appliquees, chapitre 2 : quelques applications.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 39 et 44 a 49. L'horloge a eau (2.2) est de la mecanique des
fluides, la loi de refroidissement de Newton (2.3) de la thermique, le
systeme masse-amortisseur-ressort (2.4) une equation lineaire du second
ordre -- la meme que celle du circuit RLC.

Deux coquilles du syllabus sont etablies : le rayon de l'orifice (page 46)
et le signe de l'equation des mulots (page 39).
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.rlc import (
    COURANT,
    equation_du_circuit_rlc,
)
from polytech.electromagnetisme.domain.circuits.rlc import TEMPS as TEMPS_DU_CIRCUIT
from polytech.fluides.domain.fondamentales.bernoulli import GRAVITE_EXACTE
from polytech.fluides.domain.fondamentales.clepsydre import (
    baisse_du_niveau,
    coefficient_du_profil,
    rayon_de_l_orifice,
)
from polytech.fluides.domain.fondamentales.vidange import coefficient_de_vidange
from polytech.maths.domain.differentielles.autonomes import EquationAutonome
from polytech.maths.domain.differentielles.operateurs import D
from polytech.maths.domain.differentielles.oscillateurs import (
    gain,
    masse_amortisseur_ressort,
)
from polytech.maths.domain.differentielles.particulieres import pour_une_sinusoide
from polytech.maths.domain.differentielles.premier_ordre import INCONNUE as Y
from polytech.maths.domain.symboles import TEMPS as T
from polytech.thermique.domain.refroidissement import (
    delai_de_refroidissement,
    taux_de_refroidissement,
    temperature_du_corps,
)
from polytech.thermique.domain.transitoire import CorpsTransitoire

RAYON = sp.Symbol("r", positive=True)
OMEGA = sp.Symbol("omega", positive=True)


def test_le_profil_en_r_puissance_quatre_donne_une_baisse_constante() -> None:
    """(2.13) : `z = b^2 r^4` rend `dz/dt = -k b / pi`, quel que soit `r`."""
    b, k = sp.symbols("b k", positive=True)
    baisse = baisse_du_niveau(b**2 * RAYON**4, RAYON, k)
    assert sp.simplify(baisse + k * b / sp.pi) == 0


def test_un_cone_ne_se_vide_pas_a_vitesse_constante() -> None:
    """`z = r` : `dz/dt = -k / (pi r^(3/2))`, qui s'emballe pres du fond."""
    k = sp.Symbol("k", positive=True)
    baisse = baisse_du_niveau(RAYON, RAYON, k)
    assert baisse.has(RAYON)
    assert sp.simplify(baisse + k / (sp.pi * RAYON ** sp.Rational(3, 2))) == 0


def test_le_coefficient_du_profil_de_l_horloge() -> None:
    """Pleine : `z_0 = 2 m`, `r_0 = 1/2 m`, donc `b^2 = 32` et `b = 4 racine(2)`."""
    b = coefficient_du_profil(2, sp.Rational(1, 2))
    assert b**2 == 32
    assert b == 4 * sp.sqrt(2)


def test_le_rayon_de_l_orifice_de_l_horloge() -> None:
    """`r_e^2 = (0,1 / 3600) / (8 racine(9,81))`, soit `r_e` de 1,05 mm."""
    b = coefficient_du_profil(2, sp.Rational(1, 2))
    rayon = rayon_de_l_orifice(b, sp.Rational(1, 10) / 3600)
    assert float(rayon**2) == pytest.approx(0.1 / 3600 / (8 * math.sqrt(9.81)))
    assert float(rayon) == pytest.approx(1.0529e-3, rel=1e-4)


def test_le_rayon_imprime_page_46_ne_donne_pas_dix_centimetres_par_heure() -> None:
    """Le syllabus conclut `r_e = 1/5400 m` ; son propre `r_e^2` donne 1/950 m.

    Avec 1/5400 m, le niveau baisserait de 3 mm par heure et non de 10 cm.
    """
    b = coefficient_du_profil(2, sp.Rational(1, 2))
    k = sp.pi * sp.Rational(1, 5400) ** 2 * sp.sqrt(2 * GRAVITE_EXACTE)
    par_heure = -baisse_du_niveau(b**2 * RAYON**4, RAYON, k) * 3600
    assert float(par_heure) == pytest.approx(0.0031, abs=1e-4)
    juste = rayon_de_l_orifice(b, sp.Rational(1, 10) / 3600)
    assert 1 / float(juste) == pytest.approx(950, abs=1)


def test_l_horloge_et_le_seau_troue_obeissent_a_la_meme_loi_de_torricelli() -> None:
    """Lien : pour un cylindre, `dz/dt = -k racine(z) / S` est le `gamma` du seau."""
    orifice, surface, hauteur = sp.symbols("s S z", positive=True)
    k = orifice * sp.sqrt(2 * GRAVITE_EXACTE)
    cylindre = -k * sp.sqrt(hauteur) / surface
    gamma = coefficient_de_vidange(orifice, surface)
    assert sp.simplify(cylindre - gamma * sp.sqrt(hauteur)) == 0


def test_le_taux_de_refroidissement_du_corps() -> None:
    """2.3 : 26,5 puis 24 degres a une heure d'ecart, dans 21 degres : 0,6061 /h."""
    assert taux_de_refroidissement(26.5, 24, 21, 1) == pytest.approx(0.6061, abs=5e-5)


def test_le_delai_depuis_le_deces() -> None:
    """2.3 : parti de 37 degres, le corps est a 26,5 degres apres 1,76 h."""
    taux = taux_de_refroidissement(26.5, 24, 21, 1)
    assert delai_de_refroidissement(37, 26.5, 21, taux) == pytest.approx(1.76, abs=5e-3)


def test_la_temperature_suit_la_loi_de_newton() -> None:
    """`T(t) = T_inf + (T_0 - T_inf) e^(-beta t)` redonne les deux mesures."""
    taux = taux_de_refroidissement(26.5, 24, 21, 1)
    delai = delai_de_refroidissement(37, 26.5, 21, taux)
    assert temperature_du_corps(37, 21, taux, 0) == pytest.approx(37)
    assert temperature_du_corps(37, 21, taux, delai) == pytest.approx(26.5)
    assert temperature_du_corps(37, 21, taux, delai + 1) == pytest.approx(24)
    assert temperature_du_corps(37, 21, taux, 1e3) == pytest.approx(21)


def test_un_intervalle_plus_long_donne_le_meme_taux() -> None:
    """Deux mesures a deux heures d'ecart : le taux se divise par l'intervalle."""
    taux = taux_de_refroidissement(26.5, 24, 21, 1)
    plus_tard = temperature_du_corps(26.5, 21, taux, 2)
    assert taux_de_refroidissement(26.5, plus_tard, 21, 2) == pytest.approx(taux)


def test_un_corps_qui_ne_se_rapproche_pas_de_l_ambiante_est_refuse() -> None:
    """Une seconde mesure plus loin de l'ambiante que la premiere contredit la loi."""
    with pytest.raises(ValueError, match="ne se rapproche pas de l'ambiante"):
        taux_de_refroidissement(24, 26.5, 21, 1)


def test_un_corps_a_la_temperature_ambiante_ne_date_rien() -> None:
    with pytest.raises(ValueError, match="ne se rapproche pas de l'ambiante"):
        delai_de_refroidissement(37, 21, 21, 0.6)


def test_la_loi_de_newton_est_celle_du_corps_thermiquement_mince() -> None:
    """Lien avec la thermique : `beta = 1 / tau`, `tau = rho c L / K`."""
    corps = CorpsTransitoire(
        echange=10,
        longueur_caracteristique=0.01,
        conductivite=50,
        capacite_volumique=4e6,
    )
    taux = 1 / corps.constante_de_temps()
    ecart = (temperature_du_corps(37, 21, taux, 1800) - 21) / (37 - 21)
    assert ecart == pytest.approx(corps.ecart_relatif(1800))


def test_l_equation_des_mulots_imprimee_page_39_a_le_mauvais_signe() -> None:
    """Le syllabus ecrit `y' = -y/2 - 450`, mais code `dy = 0.5*y - 450`.

    C'est le code qui est juste : l'equilibre annonce est 900, et 1000
    mulots croissent. Avec le signe imprime, l'equilibre serait a -900.
    """
    assert EquationAutonome(Y / 2 - 450).points_critiques() == (900,)
    assert EquationAutonome(-Y / 2 - 450).points_critiques() == (-900,)
    assert (Y / 2 - 450).subs(Y, 1000) > 0
    assert (Y / 2 - 450).subs(Y, 800) < 0


def test_l_operateur_du_systeme_masse_amortisseur_ressort() -> None:
    """`m y'' = -C y' - K y + x(t)`, soit `(m D^2 + C D + K) y = x(t)`."""
    m, c, k = sp.symbols("m C K", positive=True)
    assert masse_amortisseur_ressort(m, c, k).polynome == m * D**2 + c * D + k


def test_la_reponse_forcee_hors_de_la_resonance() -> None:
    """Page 48 : `y'' + y = sin(wt)` a pour reponse `sin(wt) / (1 - w^2)`."""
    oscillateur = masse_amortisseur_ressort(1, 0, 1)
    reponse = pour_une_sinusoide(oscillateur, sp.sin, 2)
    assert sp.simplify(reponse - sp.sin(2 * T) / (1 - 2**2)) == 0


def test_la_resonance_fait_croitre_l_amplitude_avec_le_temps() -> None:
    """Page 48 : pour `w = 1`, `y = -(t/2) cos t`."""
    oscillateur = masse_amortisseur_ressort(1, 0, 1)
    reponse = pour_une_sinusoide(oscillateur, sp.sin, 1)
    assert sp.simplify(reponse + T / 2 * sp.cos(T)) == 0
    assert sp.simplify(oscillateur.appliquer(reponse) - sp.sin(T)) == 0


def test_le_gain_depend_de_la_pulsation() -> None:
    """Le rapport des amplitudes : `1 / |1 - w^2|`, infini a la pulsation propre."""
    oscillateur = masse_amortisseur_ressort(1, 0, 1)
    assert gain(oscillateur, 2) == sp.Rational(1, 3)
    assert gain(oscillateur, sp.Rational(1, 2)) == sp.Rational(4, 3)
    assert gain(oscillateur, 1) == sp.oo


def test_l_amortissement_borne_le_gain_a_la_resonance() -> None:
    """Avec un amortisseur, `K(j) = j C` ne s'annule plus : le gain vaut `1/C`."""
    assert gain(masse_amortisseur_ressort(1, sp.Rational(1, 4), 1), 1) == 4


def test_le_circuit_rlc_est_un_systeme_masse_amortisseur_ressort() -> None:
    """Lien, page 48 : `L i'' + R i' + i/C = u'` -- `L` masse, `R` amortisseur."""
    inductance, resistance, capacite = sp.symbols("L R C", positive=True)
    mecanique = masse_amortisseur_ressort(inductance, resistance, 1 / capacite)
    circuit = equation_du_circuit_rlc(resistance, inductance, capacite)
    electrique = circuit.rhs.subs(TEMPS_DU_CIRCUIT, T)
    assert sp.simplify(mecanique.appliquer(COURANT(T)) - electrique) == 0


def test_deux_mesures_egales_ne_donnent_aucun_taux() -> None:
    """Un ecart qui ne bouge pas : le logarithme serait nul, et le delai infini."""
    with pytest.raises(ValueError, match="ne se rapproche pas de l'ambiante"):
        taux_de_refroidissement(24, 24, 21, 1)
