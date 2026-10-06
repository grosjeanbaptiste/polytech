"""Signaux et systemes : le TP « Reservoir », trois cuves melangees en serie.

Source : `SignauxSystemes/TP/TP_Reservoir_2023_2024_Enonce.pdf`, six pages.
L'enonce donne le modele non lineaire, ses parametres, et le modele
linearise auquel on doit aboutir -- `0,009`, `794`, `0,98`, et les
constantes de temps 298, 443 et 145 s. Ces nombres sont les oracles de la
linearisation. Le reste -- marges, reponses -- n'a pas de corrige ; chaque
resultat y est recoupe par un second chemin : la simulation du modele non
lineaire, la formule fermee du troisieme ordre, `scipy.signal`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from polytech.signaux.application import cuves_boucle as bo
from polytech.signaux.application import cuves_courbes as co
from polytech.signaux.application import cuves_modele as mo
from polytech.signaux.application import cuves_pid as pid

# --- 1. Le point de repos et le modele linearise ----------------------------


def test_le_debit_auxiliaire_au_repos() -> None:
    """`F_A = F_Amax R^(nu - 1)` pour `nu = 0,75` : 2,30 litres par seconde."""
    assert mo.debit_auxiliaire(0.75) == pytest.approx(0.00612 * 50**-0.25, rel=1e-12)
    assert mo.debit_auxiliaire(0.75) * 1e3 == pytest.approx(2.30, abs=5e-3)


def test_la_concentration_au_repos() -> None:
    """Le melange des deux debits : 6,86 %, la meme dans les trois cuves."""
    assert mo.concentration_de_repos() == pytest.approx(6.864, abs=5e-4)


def test_la_vanne_linearisee_de_l_enonce() -> None:
    """Page 4, travail analytique : `F_A = 0,009 nu`."""
    assert mo.gain_de_la_vanne() == pytest.approx(0.009, abs=5e-5)
    attendu = mo.debit_auxiliaire(0.75) * math.log(50)
    assert mo.gain_de_la_vanne() == pytest.approx(attendu, rel=1e-9)


def test_le_melange_linearise_de_l_enonce() -> None:
    """Page 4 : `X_0 = 794 F_A + 0,98 X_B`."""
    par_le_debit, par_la_concentration = mo.gains_du_melange()
    assert par_le_debit == pytest.approx(794.0, abs=0.5)
    assert par_la_concentration == pytest.approx(0.98, abs=5e-3)


def test_les_constantes_de_temps_de_l_enonce() -> None:
    """Page 4 : `(1 + 298 p)(1 + 443 p)(1 + 145 p)`."""
    assert mo.constantes_de_temps() == pytest.approx((298.0, 443.0, 145.0), abs=0.5)


def test_la_linearisation_par_differences_finies() -> None:
    """Second chemin : les memes pentes, mesurees sur le modele non lineaire."""
    pas = 1e-6
    vanne = (mo.debit_auxiliaire(0.75 + pas) - mo.debit_auxiliaire(0.75 - pas)) / (
        2 * pas
    )
    assert mo.gain_de_la_vanne() == pytest.approx(vanne, rel=1e-6)
    debit = mo.debit_auxiliaire(0.75)
    melange = (
        mo.entree_des_cuves(debit + pas, 5.0) - mo.entree_des_cuves(debit - pas, 5.0)
    ) / (2 * pas)
    assert mo.gains_du_melange()[0] == pytest.approx(melange, rel=1e-6)


# --- 1 bis. La boucle ouverte : echelons de commande et de perturbation -----


def test_les_valeurs_finales_en_boucle_ouverte() -> None:
    """`0,009 x 794 x 0,1 = 0,715` point pour la vanne ; `0,98 x (-0,5)` pour `x_B`."""
    commande, perturbation = mo.variations_finales_lineaires()
    assert commande == pytest.approx(0.715, abs=1e-3)
    assert perturbation == pytest.approx(-0.490, abs=1e-3)


def test_le_modele_non_lineaire_confirme_la_perturbation() -> None:
    """Second chemin : la simulation des trois bilans ; `x_B` entre lineairement."""
    _, concentrations = mo.simuler(ouverture=0.75, alimentation=4.5, duree=6000.0)
    _, perturbation = mo.variations_finales_lineaires()
    obtenue = concentrations[-1, 2] - mo.concentration_de_repos()
    assert obtenue == pytest.approx(perturbation, abs=2e-3)


def test_le_modele_non_lineaire_s_ecarte_un_peu_pour_la_commande() -> None:
    """Dix points d'ouverture : la vanne exponentielle rend 0,87 au lieu de 0,715."""
    _, concentrations = mo.simuler(ouverture=0.85, alimentation=5.0, duree=6000.0)
    obtenue = concentrations[-1, 2] - mo.concentration_de_repos()
    lineaire, _ = mo.variations_finales_lineaires()
    assert obtenue > lineaire
    assert obtenue == pytest.approx(0.867, abs=2e-3)


def test_les_cuves_repondent_dans_l_ordre() -> None:
    """Pendant le transitoire, `x_1` precede `x_2`, qui precede `x_3`."""
    _, concentrations = mo.simuler(ouverture=0.85, alimentation=5.0, duree=600.0)
    milieu = concentrations[len(concentrations) // 2]
    assert milieu[0] > milieu[1] > milieu[2] > mo.concentration_de_repos()


# --- 2. La boucle fermee, controleur proportionnel --------------------------


def test_le_gain_statique_de_la_boucle() -> None:
    """`0,009 x 794 = 7,15` pour `K = 1`."""
    assert bo.gain_statique() == pytest.approx(7.146, abs=1e-3)


def test_la_marge_de_gain_analytique() -> None:
    """Travail analytique, page 4 : `w^2 = somme(T)/produit(T)`, puis le module."""
    pulsation, marge = bo.marge_de_gain_analytique()
    t1, t2, t3 = 298.0, 443.0, 145.0
    assert pulsation == pytest.approx(math.sqrt((t1 + t2 + t3) / (t1 * t2 * t3)))
    assert marge == pytest.approx(1.41, abs=5e-3)
    assert 20 * math.log10(marge) == pytest.approx(3.0, abs=0.05)


def test_la_marge_de_gain_par_routh() -> None:
    """Second chemin : `K_max` annule la condition du troisieme degre."""
    _, marge = bo.marge_de_gain_analytique()
    assert bo.gain_limite_par_routh() == pytest.approx(marge, rel=1e-9)


def test_les_marges_numeriques() -> None:
    """Ce que rend la commande `margin` : 11,5 degres et 3,0 dB."""
    marges = bo.marges(1.0)
    _, analytique = bo.marge_de_gain_analytique()
    assert marges.de_gain == pytest.approx(analytique, rel=1e-4)
    assert marges.de_phase == pytest.approx(11.5, abs=0.05)


def test_l_erreur_de_regime_decroit_avec_le_gain() -> None:
    """`1/(1 + 7,15 K)` : 22 % pour `K = 0,5`, 12 % pour `K = 1` -- jamais nulle."""
    assert bo.erreur_de_regime(0.5) == pytest.approx(0.219, abs=1e-3)
    assert bo.erreur_de_regime(1.0) == pytest.approx(0.123, abs=1e-3)


def test_le_depassement_croit_avec_le_gain() -> None:
    """Simulation : plus `K` approche de la limite, plus la reponse oscille."""
    faible, fort = bo.reponse(0.3), bo.reponse(1.0)
    assert faible.depassement < fort.depassement
    assert faible.depassement == pytest.approx(26.0, abs=1.0)
    assert fort.depassement == pytest.approx(71.0, abs=1.0)
    assert faible.montee > fort.montee
    assert fort.finale == pytest.approx(1 - bo.erreur_de_regime(1.0), abs=2e-3)


def test_a_la_limite_la_reponse_oscille_sans_s_amortir() -> None:
    """Pour `K = K_max` : deux poles imaginaires purs, a la pulsation de la marge."""
    pulsation, marge = bo.marge_de_gain_analytique()
    poles = bo.poles_en_boucle_fermee(marge)
    sur_l_axe = [pole for pole in poles if abs(pole.real) < 1e-9]
    assert sorted(abs(pole.imag) for pole in sur_l_axe) == pytest.approx(
        [pulsation, pulsation]
    )


# --- 2 bis. La boucle fermee, controleur PID --------------------------------


def test_le_regime_sous_un_echelon_de_consigne() -> None:
    """Travail analytique, page 6 : `y = 1`, et la commande vaut `1/7,15 = 0,14`."""
    sortie, commande = pid.regime_sous_consigne()
    assert sortie == 1
    assert float(commande) == pytest.approx(0.140, abs=5e-4)


def test_les_marges_apres_correction() -> None:
    """La marge de phase passe de 11,5 a 60,6 degres, celle de gain a 18,3 dB."""
    marges = pid.marges()
    assert marges.de_phase == pytest.approx(60.6, abs=0.1)
    assert marges.de_gain_en_decibels() == pytest.approx(18.3, abs=0.1)


def test_les_caracteristiques_de_m() -> None:
    """`|M(0)| = 0` dB, aucun pic, bande passante de 6,2 mrad/s."""
    m = pid.caracteristiques()
    assert m.en_continu_en_decibels() == pytest.approx(0.0, abs=1e-3)
    assert m.maximum_en_decibels() == pytest.approx(0.0, abs=0.01)
    assert m.bande_passante * 1e3 == pytest.approx(6.15, abs=0.05)


def test_la_predetermination_et_la_simulation() -> None:
    """La regle annonce 9 % et 306 a 459 s ; la reponse simulee fait mieux."""
    estimation = pid.estimation()
    simulee = pid.reponse_a_la_consigne()
    assert estimation.depassement == pytest.approx(9.0, abs=0.1)
    assert simulee.finale == pytest.approx(1.0, abs=1e-3)
    assert simulee.depassement < estimation.depassement
    assert estimation.montee[0] * 0.5 < simulee.montee < estimation.montee[1]


def test_la_perturbation_est_rejetee() -> None:
    """Question 5 : un echelon de perturbation s'efface, l'integrale le compense."""
    instants, ecart = pid.reponse_a_la_perturbation()
    assert abs(ecart[-1]) < 1e-3
    assert np.max(np.abs(ecart)) == pytest.approx(0.25, abs=0.01)
    assert instants[int(np.argmax(np.abs(ecart)))] > 0


def test_par_scipy() -> None:
    """Second chemin : les marges du PID, relues sur `scipy.signal.bode`."""
    assert pid.marge_de_phase_par_scipy() == pytest.approx(
        pid.marges().de_phase, abs=0.05
    )


def test_la_courbe_de_black_se_deroule_sous_moins_cent_quatre_vingts() -> None:
    """Trois poles : l'argument descend jusqu'a -270 degres, sans saut de 360."""
    arguments, gains = co.points_de_black(bo.PROCEDE, 1e-4, 1.0)
    assert np.all(np.diff(arguments) < 0)
    assert arguments[0] == pytest.approx(0.0, abs=6.0)
    assert arguments[-1] == pytest.approx(-270.0, abs=1.0)
    assert np.all(np.diff(gains) < 0)
