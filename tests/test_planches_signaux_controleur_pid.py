"""Planches de signaux et systemes : le controleur PID et sa synthese.

Source : `SignauxSystemes/SLIDES/Slides_Plan_du_cours_et_corrections.pdf`
(36 planches), lues sur les pages rendues.

Les oracles : l'exemple illustratif complet des planches 26 a 31, pour
`10 exp(-0,5 p) / (5 p^3 + 15,5 p^2 + 11,5 p + 1)` -- quatre essais et
leurs `G_c`, `T_i`, `T_d`, temps de montee et depassement --, puis la
methode de Ziegler-Nichols des planches 33 a 36 (`R = 0,74`, `K_p = 1,1`,
`tau_i = 3`, `tau_d = 0,75`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from polytech.signaux.domain.boucle_fermee import Boucle
from polytech.signaux.domain.controleur import ProportionnelIntegralDerive
from polytech.signaux.domain.erreurs_de_regime import erreur_de_consigne
from polytech.signaux.domain.fonction_m import (
    caracteristiques_de_m,
    depassement_estime,
    temps_de_montee_estime,
)
from polytech.signaux.domain.marges import arguments_deroules, marges_de_stabilite
from polytech.signaux.domain.pid_serie import (
    PidSerie,
    bande_passante_minimale,
    ecart_maximal_de_m,
)
from polytech.signaux.domain.simulation_a_delai import indicielle_en_boucle_fermee
from polytech.signaux.domain.symboles import IMAGE
from polytech.signaux.domain.synthese_pid import Cible, synthetiser_le_pid
from polytech.signaux.domain.temps_caracteristiques import temps_de_montee
from polytech.signaux.domain.ziegler_nichols import (
    ModeleDeZieglerNichols,
    reglage_de_ziegler_nichols,
)

P = IMAGE
RATIONNEL = 10 / (5 * P**3 + sp.Rational(31, 2) * P**2 + sp.Rational(23, 2) * P + 1)
PROCEDE = RATIONNEL * sp.exp(-P / 2)
"""Planche 26 : `G_s G_1 G_2`, avec un delai d'une demi-seconde."""


def test_le_pid_serie_et_ses_cas_particuliers() -> None:
    """Planches 12 et 13 : `G_c (1 + p T_i) / (p T_i) (1 + p T_d) / (1 + p l T_d)`."""
    gain, integrale, derivee, filtre = sp.symbols("G_c T_i T_d lambda", positive=True)
    pid = PidSerie(gain, integrale, derivee, filtre)
    attendue = gain * (1 + P * integrale) / (P * integrale)
    attendue *= (1 + P * derivee) / (1 + P * filtre * derivee)
    assert sp.simplify(pid.transmittance() - attendue) == 0
    assert pid.alpha() == integrale / derivee
    sans_derivee = PidSerie(gain, integrale, 0, filtre).transmittance()
    assert sp.simplify(sans_derivee - gain * (1 + P * integrale) / (P * integrale)) == 0
    assert PidSerie.depuis_la_bande_proportionnelle(50, 10, 1).gain == 2
    with pytest.raises(ValueError, match="constante d'integration nulle"):
        PidSerie(1, 0, 1)


def test_l_argument_du_pid_reste_entre_moins_90_degres_et_un_maximum() -> None:
    """Planches 17 et 23 : le maximum ne depend que de `alpha` et de `lambda`.

    La planche 23 annonce 54 degres pour `alpha = 10` ; c'est 53,1. Les
    54 degres sont la limite des grands `alpha`.
    """
    assert PidSerie(1, 10, 1).avance_maximale() == pytest.approx(53.1, abs=0.05)
    assert PidSerie(3, 70, 7).avance_maximale() == pytest.approx(53.1, abs=0.05)
    assert PidSerie(1, 100, 1).avance_maximale() == pytest.approx(54.7, abs=0.05)
    assert PidSerie(1, 1, 1).avance_maximale() == pytest.approx(40.8, abs=0.05)
    basse = complex(PidSerie(1, 10, 1).transmittance().subs(P, 1e-6j))
    assert math.degrees(np.angle(basse)) == pytest.approx(-90.0, abs=0.01)


def test_les_specifications_temporelles_deviennent_des_contraintes() -> None:
    """Planche 9 : `|M|max - |M(0)| < (D - 9) / 6` et `BP > 0,38 / t_m`."""
    assert ecart_maximal_de_m(15.0) == pytest.approx(1.0)
    assert depassement_estime(ecart_maximal_de_m(15.0)) == pytest.approx(15.0)
    assert ecart_maximal_de_m(9.0) == 0
    assert bande_passante_minimale(1.9) == pytest.approx(0.2)
    bas, haut = temps_de_montee_estime(bande_passante_minimale(1.9))
    assert bas < 1.9 < haut
    with pytest.raises(ValueError, match="temps de montee nul ou negatif"):
        bande_passante_minimale(0.0)


ESSAIS = {
    1: (Cible(-120.0, 0.0), 0.8, 10, (0.58, 21.0, 2.1)),
    2: (Cible(-120.0, 0.0), 0.6, 10, (0.59, 12.7, 1.27)),
    3: (Cible(-120.0, 0.0), 0.6, 7, (0.56, 9.74, 1.39)),
    4: (Cible(-118.0, -0.5), 0.56, 7, (0.50, 9.66, 1.38)),
}
"""Planches 28 a 31 : le point cible, `w_c`, `alpha`, puis `G_c`, `T_i`, `T_d`."""


@pytest.mark.parametrize("essai", sorted(ESSAIS))
def test_les_quatre_essais_de_l_exemple_illustratif(essai: int) -> None:
    cible, pulsation, alpha, (gain, integrale, derivee) = ESSAIS[essai]
    pid = synthetiser_le_pid(PROCEDE, cible, pulsation=pulsation, alpha=alpha)
    assert float(pid.gain) == pytest.approx(gain, abs=0.006)
    assert float(pid.integrale) == pytest.approx(integrale, rel=0.003)
    assert float(pid.derivee) == pytest.approx(derivee, abs=0.006)


@pytest.mark.parametrize("essai", sorted(ESSAIS))
def test_la_courbe_corrigee_passe_par_le_point_cible(essai: int) -> None:
    """Planche 21 : deux equations, le module et l'argument en `w_c`."""
    cible, pulsation, alpha, _ = ESSAIS[essai]
    pid = synthetiser_le_pid(PROCEDE, cible, pulsation=pulsation, alpha=alpha)
    corrigee = complex((pid.transmittance() * PROCEDE).subs(P, 1j * pulsation))
    assert 20 * math.log10(abs(corrigee)) == pytest.approx(cible.gain, abs=1e-6)
    phase = math.degrees(np.angle(corrigee)) % 360 - 360
    assert phase == pytest.approx(cible.argument, abs=1e-6)


def test_l_argument_du_procede_se_deroule_au_dela_de_moins_180_degres() -> None:
    """`(1 + 10 p)(1 + p)(1 + 0,5 p)` et le delai : -166,25 degres en 0,8 rad/s."""
    pulsations = np.logspace(-4, math.log10(3.0), 30_001)
    valeurs = sp.lambdify(P, PROCEDE, "numpy")(1j * pulsations)
    phases = np.degrees(arguments_deroules(PROCEDE, valeurs))
    en_w_c = float(np.interp(0.8, pulsations, phases))
    somme = math.atan(8.0) + math.atan(0.8) + math.atan(0.4) + 0.4
    assert en_w_c == pytest.approx(-math.degrees(somme), abs=0.01)
    assert en_w_c == pytest.approx(-166.25, abs=0.01)
    assert phases[-1] < -270


def test_un_point_cible_hors_de_portee_est_refuse() -> None:
    """Planche 23 : l'argument du PID ne depasse pas son maximum."""
    with pytest.raises(ValueError, match="hors de portee du PID"):
        synthetiser_le_pid(PROCEDE, Cible(-120.0, 0.0), pulsation=1.5, alpha=10)


def _indicielle(essai: int) -> tuple[np.ndarray, np.ndarray]:
    cible, pulsation, alpha, _ = ESSAIS[essai]
    pid = synthetiser_le_pid(PROCEDE, cible, pulsation=pulsation, alpha=alpha)
    instants = np.arange(0, 40, 0.005)
    rationnelle = pid.transmittance() * RATIONNEL
    return instants, indicielle_en_boucle_fermee(rationnelle, 0.5, instants)


LUS = {
    1: (1.23, 17.0, 1.09),
    2: (1.58, 17.0, 1.07),
    3: (1.54, 17.0, 1.08),
    4: (1.84, 8.0, 1.03),
}
"""Le temps de montee et le depassement imprimes, puis le sommet lu sur la courbe."""


@pytest.mark.parametrize("essai", sorted(LUS))
def test_les_reponses_indicielles_des_quatre_essais(essai: int) -> None:
    """Planches 28 a 31 : les courbes sont retrouvees, non les « D » imprimes.

    Les sommets traces -- 1,09, 1,07, 1,08 et 1,03 -- sont ceux de la
    simulation, que recoupe une inversion frequentielle independante. Les
    depassements imprimes, 17 %, 17 %, 17 % et 8 %, sont environ le double
    de ce que montrent les courbes de la meme planche. Les temps de montee
    imprimes sont retrouves a 6 % pres.
    """
    imprime, depassement, sommet = LUS[essai]
    instants, reponse = _indicielle(essai)
    assert reponse.max() == pytest.approx(sommet, abs=0.006)
    assert 100 * (reponse.max() - 1) < 0.6 * depassement
    montee = temps_de_montee(instants, reponse, regime=1.0)
    assert montee == pytest.approx(imprime, rel=0.065)
    assert montee > imprime


def test_l_inversion_frequentielle_recoupe_la_simulation() -> None:
    """`y(t) = (2 / pi) integrale de Re M(j w) sin(w t) / w`, sur l'essai 1."""
    cible, pulsation, alpha, _ = ESSAIS[1]
    pid = synthetiser_le_pid(PROCEDE, cible, pulsation=pulsation, alpha=alpha)
    pulsations = np.linspace(1e-6, 400, 2_000_001)
    boucle = sp.lambdify(P, pid.transmittance() * PROCEDE, "numpy")(1j * pulsations)
    reelle = (boucle / (1 + boucle)).real
    reponse = _indicielle(1)[1]
    for instant in (2.0, 3.0, 6.0):
        noyau = reelle * np.sin(pulsations * instant) / pulsations
        inversee = 2 / math.pi * np.trapezoid(noyau, pulsations)
        simulee = reponse[round(instant / 0.005)]
        assert simulee == pytest.approx(inversee, abs=2e-3)


def test_le_trainage_se_lit_sur_le_creux_de_m() -> None:
    """« Trainage !! », « Leger trainage residuel », puis « Controleur ok ! »."""
    creux = {}
    for essai in (1, 2, 4):
        cible, pulsation, alpha, _ = ESSAIS[essai]
        pid = synthetiser_le_pid(PROCEDE, cible, pulsation=pulsation, alpha=alpha)
        creux[essai] = caracteristiques_de_m(pid.transmittance() * PROCEDE).minimum
    assert creux[4] is None
    assert creux[1] is not None
    assert creux[2] is not None
    assert 20 * math.log10(creux[1]) < -1 < -0.5 < 20 * math.log10(creux[2]) < 0
    instants, reponse = _indicielle(1)
    assert reponse[instants > 10].max() < 0.99


def test_le_pid_de_l_essai_4_stabilise_et_longe_le_lieu_zero_decibel() -> None:
    """Planches 26 et 31 : integrateur, donc erreur nulle ; `|M|` sans pic ni creux."""
    cible, pulsation, alpha, _ = ESSAIS[4]
    pid = synthetiser_le_pid(PROCEDE, cible, pulsation=pulsation, alpha=alpha)
    corrigee = pid.transmittance() * PROCEDE
    assert marges_de_stabilite(corrigee).de_phase > 45
    m = caracteristiques_de_m(corrigee)
    assert m.en_continu == pytest.approx(1.0, abs=1e-3)
    assert m.maximum_en_decibels() < 0.5
    sans_delai = Boucle(pid.transmittance(), 1, RATIONNEL, 1)
    assert erreur_de_consigne(sans_delai, 1) == 0


def test_la_simulation_a_delai_retrouve_le_cas_sans_delai() -> None:
    """Un recoupement : `1 / (p (1 + p))` boucle, soit `1 / (p^2 + p + 1)`."""
    instants = np.arange(0, 12, 0.002)
    simulee = indicielle_en_boucle_fermee(1 / (P * (1 + P)), 0.0, instants)
    racine = math.sqrt(3) / 2
    amortie = np.exp(-instants / 2)
    exacte = 1 - amortie * (
        np.cos(racine * instants) + np.sin(racine * instants) / (2 * racine)
    )
    assert np.max(np.abs(simulee - exacte)) < 2e-3
    with pytest.raises(ValueError, match="delai negatif"):
        indicielle_en_boucle_fermee(1 / (P * (1 + P)), -1.0, instants)


def test_la_methode_de_ziegler_nichols_de_la_planche_35() -> None:
    """`K = 10`, `Delta T = 13,5 s`, `tau = 1,5 s`."""
    modele = ModeleDeZieglerNichols(gain=10.0, montee=13.5, retard=1.5)
    assert modele.pente() == pytest.approx(0.74, abs=0.001)
    assert (
        sp.simplify(modele.transmittance() - modele.pente() * sp.exp(-1.5 * P) / P) == 0
    )
    gain, integrale, derivee = reglage_de_ziegler_nichols(modele, "PID")
    assert gain == pytest.approx(1.1, abs=0.03)
    assert (integrale, derivee) == (3.0, 0.75)
    assert reglage_de_ziegler_nichols(modele, "P") == (pytest.approx(0.9), None, None)
    proportionnel_integral = reglage_de_ziegler_nichols(modele, "PI")
    assert proportionnel_integral == (pytest.approx(0.81), 4.5, None)
    with pytest.raises(ValueError, match="controleur inconnu"):
        reglage_de_ziegler_nichols(modele, "PD")


def test_ziegler_nichols_depasse_bien_plus_que_la_methode_frequentielle() -> None:
    """Planche 36 : le sommet passe 1,6 ; la methode frequentielle reste a 1,08."""
    modele = ModeleDeZieglerNichols(gain=10.0, montee=13.5, retard=1.5)
    gain, integrale, derivee = reglage_de_ziegler_nichols(modele, "PID")
    parallele = ProportionnelIntegralDerive(
        sp.Float(gain), sp.Float(integrale), sp.Float(derivee)
    )
    instants = np.arange(0, 40, 0.005)
    rationnelle = parallele.transmittance().expression * RATIONNEL
    reponse = indicielle_en_boucle_fermee(rationnelle, 0.5, instants)
    assert reponse.max() == pytest.approx(1.65, abs=0.05)
    assert reponse.max() > _indicielle(4)[1].max() + 0.5
