"""Le syllabus de thermique, tome 1 : l'inertie thermique.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_1.pdf`, pages 63 a 70
(premiere partie, chapitre 3) et exercices I.4.1 et I.4.2 (pages 81 a
83), lus sur les pages rendues.

Les oracles : (3.1) a (3.5), les 63,2 % de la figure 3.2, l'erreur de
trainage `B R C`, et les deux exercices -- `Bi = 0,0054`, 12,73 et
236,54, 3011 s, 7837 s ; puis 0,2612 s, 32,8, 0,03 et 88 degres.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.signaux.domain.symboles import IMAGE, TEMPS
from polytech.thermique.domain.composant_electronique import ComposantSurRefroidisseur
from polytech.thermique.domain.inertie import SolideIsotherme
from polytech.thermique.domain.transitoire import CorpsTransitoire

P, TAU = IMAGE, TEMPS
R, C = sp.symbols("R C", positive=True)


def test_la_constante_de_temps_est_un_produit_r_c() -> None:
    """(3.2) a (3.4) : `tau_c = rho c V / (K S) = R C`."""
    solide = SolideIsotherme.depuis(echange=10.0, surface=2.0, capacite=400.0)
    assert solide.resistance == pytest.approx(0.05)
    assert solide.constante_de_temps() == pytest.approx(20.0)
    with pytest.raises(ValueError, match="surface d'echange nulle ou negative"):
        SolideIsotherme.depuis(echange=10.0, surface=0.0, capacite=400.0)


def test_l_ecart_decroit_exponentiellement() -> None:
    """(3.1) et (3.3) : `rho c V dt/dtau = K S (t_f - t)`, donc `exp(-tau / tau_c)`."""
    solide = SolideIsotherme(resistance=R, capacite=C)
    ecart = solide.ecart_reduit(TAU)
    assert sp.simplify(C * sp.diff(ecart, TAU) + ecart / R) == 0
    assert ecart.subs(TAU, 0) == 1
    chiffre = SolideIsotherme(resistance=0.05, capacite=400.0)
    assert 1 - float(chiffre.ecart_reduit(20.0)) == pytest.approx(0.632, abs=5e-4)


def test_le_solide_isotherme_est_un_systeme_du_premier_ordre() -> None:
    """(3.5) : `[t] / [t_f] = 1 / (1 + p R C)`, celui du cours de signaux."""
    solide = SolideIsotherme(resistance=R, capacite=C)
    systeme = solide.premier_ordre()
    assert sp.simplify(systeme.de_bode().expression - 1 / (1 + P * R * C)) == 0
    assert sp.simplify(systeme.indicielle() - (1 - solide.ecart_reduit(TAU))) == 0


def test_la_rampe_et_l_erreur_de_trainage() -> None:
    """Fig. 3.3 : `t = t_0 + B tau - R C B (1 - exp(-tau / RC))`, erreur `B R C`."""
    pente = sp.Symbol("B", positive=True)
    solide = SolideIsotherme(resistance=R, capacite=C)
    reponse = pente * solide.premier_ordre().reponse_a_la_rampe()
    attendue = pente * TAU - R * C * pente * (1 - sp.exp(-TAU / (R * C)))
    assert sp.simplify(reponse - attendue) == 0
    assert solide.erreur_de_trainage(pente) == pente * R * C
    assert sp.limit(pente * TAU - reponse, TAU, sp.oo) == pente * R * C


def test_l_attenuation_et_le_dephasage_d_une_sollicitation_periodique() -> None:
    """Page 67 : `1 / racine(1 + w^2 R^2 C^2)` et `-arctan(w R C)`."""
    solide = SolideIsotherme(resistance=0.05, capacite=400.0)
    attenuation, dephasage = solide.reponse_periodique(0.1)
    assert attenuation == pytest.approx(1 / math.sqrt(1 + 4.0))
    assert dephasage == pytest.approx(-math.atan(2.0))


def test_l_exercice_i_4_1_la_bille_d_acier() -> None:
    """Diametre 0,05 m, de 600 a 100 degres dans une ambiance a 60, `K = 10`."""
    rayon = 0.025
    bille = CorpsTransitoire(10.0, rayon, 46.5, 7850 * 460.4)
    assert bille.biot() == pytest.approx(0.0054, abs=5e-5)
    assert bille.est_thermiquement_mince()
    solide = SolideIsotherme.depuis(
        echange=10.0,
        surface=4 * math.pi * rayon**2,
        capacite=4 / 3 * math.pi * rayon**3 * 7850 * 460.4,
    )
    assert solide.resistance == pytest.approx(12.73, abs=0.005)
    assert solide.capacite == pytest.approx(236.54, abs=0.01)
    assert solide.constante_de_temps() == pytest.approx(3011, abs=1)
    assert solide.constante_de_temps() / 3600 == pytest.approx(0.84, abs=0.005)
    duree = solide.duree_pour((100 - 60) / (600 - 60))
    assert duree == pytest.approx(7837, abs=3)
    assert duree / 3600 == pytest.approx(2.18, abs=0.005)
    with pytest.raises(ValueError, match="rapport hors de"):
        solide.duree_pour(1.5)


def test_l_exercice_i_4_2_le_fil_ne_suit_pas_des_gaz_a_20_hz() -> None:
    """Fil de 0,1 mm, `K = 250` : `tau_c = rho c d / (4 K) = 0,2612 s`.

    Le syllabus conclut « phi = arctan(-32,8) = 88 degres » : c'est -88,
    un retard. Il annonce aussi un fil de tungstene, puis les proprietes
    « du platine » ; 19 350 kg/m^3 est bien le tungstene.
    """
    diametre, longueur = 1e-4, 1.0
    fil = SolideIsotherme.depuis(
        echange=250.0,
        surface=math.pi * diametre * longueur,
        capacite=19350 * 135 * math.pi * diametre**2 / 4 * longueur,
    )
    assert fil.constante_de_temps() == pytest.approx(0.2612, abs=5e-5)
    assert fil.constante_de_temps() == pytest.approx(19350 * 135 * diametre / (4 * 250))
    pulsation = 2 * math.pi * 20
    assert pulsation * fil.constante_de_temps() == pytest.approx(32.8, abs=0.05)
    attenuation, dephasage = fil.reponse_periodique(pulsation)
    assert attenuation == pytest.approx(0.03, abs=0.001)
    assert math.degrees(dephasage) == pytest.approx(-88.0, abs=0.3)


MONTAGE = ComposantSurRefroidisseur(
    jonction_boitier=3.125, boitier_refroidisseur=0.25, vers_l_ambiance=2.0
)
"""Page 70 : 3,125 et 0,25 degres par watt ; la troisieme est un exemple."""


def test_la_temperature_moyenne_de_la_jonction() -> None:
    """`t_J = t_A + Phi (R_JB + R_BR) + (Phi + Phi_2) R_RA`, en valeurs moyennes.

    Deux lignes plus haut, le syllabus ecrit `t_B` pour `t_R`, et un bilan
    « 0 = Phi + (t_J - t_B) / R_JB » qui contredit la ligne qui le
    precede ; son resultat final, lui, est juste.
    """
    moyenne = MONTAGE.jonction_moyenne(ambiance=25.0, puissance=4.0, des_autres=6.0)
    refroidisseur = 25.0 + (4.0 + 6.0) * 2.0
    assert moyenne == pytest.approx(refroidisseur + 4.0 * (3.125 + 0.25))
    boitier = refroidisseur + 4.0 * 0.25
    assert moyenne - boitier == pytest.approx(4.0 * 3.125)
    assert 4.0 + (refroidisseur - boitier) / 0.25 == pytest.approx(0.0)
    assert 4.0 + (moyenne - boitier) / 3.125 != pytest.approx(0.0)


def test_la_transmittance_de_la_jonction() -> None:
    """Page 70 : `F(p) = R_JB + R_BR / (1 + p R_BR C_B)`."""
    capacite = sp.Symbol("C_B", positive=True)
    transmittance = MONTAGE.transmittance(capacite)
    rjb, rbr = sp.Rational(25, 8), sp.Rational(1, 4)
    assert sp.simplify(transmittance - (rjb + rbr / (1 + P * rbr * capacite))) == 0
    assert transmittance.subs(P, 0) == rjb + rbr
    assert sp.limit(transmittance, P, sp.oo) == rjb


def test_une_puissance_tres_rapide_ne_voit_que_r_jb() -> None:
    """« Si la periode est tres courte, la transmittance se reduit a R_JB »."""
    lente = MONTAGE.amplitude_de_la_jonction(2.0, pulsation=1e-4, capacite=40.0)
    rapide = MONTAGE.amplitude_de_la_jonction(2.0, pulsation=1e4, capacite=40.0)
    assert lente == pytest.approx(2.0 * (3.125 + 0.25), rel=1e-4)
    assert rapide == pytest.approx(2.0 * 3.125, rel=1e-4)


def test_a_la_coupure_la_jonction_voit_un_module_et_non_une_partie_reelle() -> None:
    """En `w R_BR C_B = 1` : `|R_JB + R_BR / (1 + j)|`."""
    amplitude = MONTAGE.amplitude_de_la_jonction(1.0, pulsation=0.1, capacite=40.0)
    assert amplitude == pytest.approx(math.hypot(3.125 + 0.125, 0.125), rel=1e-9)
    assert amplitude > 3.25
