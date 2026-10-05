"""Le syllabus de signaux et systemes, chapitre 5 : le second ordre.

Source : `SignauxSystemes/SYLLABUS/Syllabus_2019_Chapitre_05_Analyse_temporelle.pdf`,
sections 5.2 et 5.3 (pages 9 a 25), lues sur les pages rendues.

Les oracles : les poles de la figure 5.22 (`-0,5 +/- 0,87 j`,
`-0,1 +/- 0,99 j`, `+/- j`), les couples `(T, xi)` de la figure 5.23
(0,71 et 1,06 ; 0,45 et 1,34 ; 0,22 et 2,35), l'exemple 5.3 du circuit
RLC, et les relations (5.47) a (5.49).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from polytech.signaux.domain.heaviside import original, residu
from polytech.signaux.domain.identification import identifier_le_second_ordre
from polytech.signaux.domain.modeles import resistance_critique
from polytech.signaux.domain.poles_numeriques import reponse_indicielle
from polytech.signaux.domain.reponse_indicielle import (
    bande_passante_du_second_ordre,
    depassement_du_second_ordre,
    temps_de_montee_depuis_la_bande,
)
from polytech.signaux.domain.reponses_du_second_ordre import (
    impulsionnelle,
    indicielle,
    instant_du_maximum,
    periode_propre,
)
from polytech.signaux.domain.second_ordre import SecondOrdre
from polytech.signaux.domain.symboles import IMAGE, PULSATION, TEMPS
from polytech.signaux.domain.temps_caracteristiques import (
    temps_d_etablissement,
    temps_de_montee,
)

P, T, W = IMAGE, TEMPS, PULSATION
DEMI = sp.Rational(1, 2)
INSTANTS = np.array([0.0, 0.4, 1.3, 5.0])


def _proches(calcule: sp.Expr, attendu: sp.Expr) -> bool:
    ecart = sp.lambdify(T, calcule - attendu, "numpy")
    return bool(np.allclose(ecart(INSTANTS), 0, atol=1e-10))


def test_les_formes_de_bode_et_d_evans_du_second_ordre() -> None:
    """(5.13) a (5.15) : `K_e = K / T^2`, `sigma = xi / T`, `rho = 1 / T`."""
    gain, constante, amortissement = sp.symbols("K T xi", positive=True)
    systeme = SecondOrdre(gain, constante, amortissement)
    d_evans, sigma, rho = systeme.d_evans()
    assert (d_evans, sigma, rho) == (
        gain / constante**2,
        amortissement / constante,
        1 / constante,
    )
    bode = systeme.de_bode().expression
    assert sp.simplify(bode - d_evans / (P**2 + 2 * sigma * P + rho**2)) == 0
    retour = SecondOrdre.depuis_evans(d_evans, sigma, rho)
    assert (retour.gain, retour.constante, retour.amortissement) == (
        gain,
        constante,
        amortissement,
    )


@pytest.mark.parametrize(
    ("amortissement", "reelle", "imaginaire"),
    [(0.5, -0.5, 0.87), (0.1, -0.1, 0.99), (0.0, 0.0, 1.0)],
)
def test_les_poles_de_la_figure_5_22(
    amortissement: float, reelle: float, imaginaire: float
) -> None:
    """`K = 5`, `T = 1` : `p = -xi / T +/- j racine(1 - xi^2) / T`."""
    systeme = SecondOrdre(5, 1, amortissement)
    pole = complex(systeme.poles()[1])
    assert pole.real == pytest.approx(reelle, abs=0.005)
    assert pole.imag == pytest.approx(imaginaire, abs=0.005)
    assert float(systeme.pulsation_propre()) == pytest.approx(imaginaire, abs=0.005)


def test_sigma_et_rho_sont_les_moyennes_des_poles() -> None:
    """(5.16) et (5.17) : arithmetique pour `-sigma`, geometrique pour `rho`."""
    systeme = SecondOrdre(5, 2, DEMI)
    premier, second = systeme.poles()
    _, sigma, rho = systeme.d_evans()
    assert sp.simplify((premier + second) / 2 + sigma) == 0
    assert sp.simplify(sp.sqrt(premier * second) - rho) == 0
    rapport = (premier + second) / (2 * sp.sqrt(premier * second))
    assert sp.simplify(rapport + systeme.amortissement) == 0


def test_l_argument_des_poles_et_le_facteur_de_qualite() -> None:
    """(5.18) et (5.19) : `sin(theta) = xi`, `Q = rho / (2 sigma) = 1 / (2 xi)`."""
    systeme = SecondOrdre(5, 2, DEMI)
    _, sigma, rho = systeme.d_evans()
    assert sp.sin(systeme.argument_des_poles()) == DEMI
    tangente = sp.tan(systeme.argument_des_poles())
    assert sp.simplify(tangente - sigma / systeme.pulsation_propre()) == 0
    assert systeme.facteur_de_qualite() == 1
    assert systeme.facteur_de_qualite() == rho / (2 * sigma)
    with pytest.raises(ValueError, match="facteur de qualite infini"):
        SecondOrdre(5, 2, 0).facteur_de_qualite()


@pytest.mark.parametrize(
    ("courte", "constante", "amortissement"),
    [(0.5, 0.71, 1.06), (0.2, 0.45, 1.34), (0.05, 0.22, 2.35)],
)
def test_les_poles_reels_de_la_figure_5_23(
    courte: float, constante: float, amortissement: float
) -> None:
    """(5.20) : `T = racine(T_1 T_2)`, `xi = (T_1 + T_2) / (2 racine(T_1 T_2))`."""
    systeme = SecondOrdre.depuis_les_constantes_de_temps(5, 1.0, courte)
    assert float(systeme.constante) == pytest.approx(constante, abs=0.005)
    assert float(systeme.amortissement) == pytest.approx(amortissement, abs=0.005)
    poles = sorted(float(pole) for pole in systeme.poles())
    assert poles == pytest.approx([-1 / courte, -1.0])


def test_les_trois_regimes_du_second_ordre() -> None:
    assert SecondOrdre(5, 1, DEMI).regime() == "poles complexes conjugues"
    assert SecondOrdre(5, 1, 1).regime() == "poles reels confondus"
    assert SecondOrdre(5, 1, 2).regime() == "poles reels distincts"
    with pytest.raises(ValueError, match="amortissement negatif"):
        SecondOrdre(5, 1, -DEMI)
    with pytest.raises(ValueError, match="poles reels : pas de pulsation propre"):
        SecondOrdre(5, 1, 2).pulsation_propre()
    with pytest.raises(ValueError, match="poles reels : pas de pulsation propre"):
        SecondOrdre(5, 1, 1).pulsation_propre()


@pytest.mark.parametrize("amortissement", [DEMI, 1, 2])
def test_la_reponse_impulsionnelle_dans_les_trois_regimes(
    amortissement: sp.Expr,
) -> None:
    """(5.29), (5.34) et (5.37), contre l'inverse de Laplace de `H(p)`."""
    systeme = SecondOrdre(5, 2, amortissement)
    assert _proches(impulsionnelle(systeme), original(systeme.de_bode().expression))


@pytest.mark.parametrize("amortissement", [DEMI, 1, 2])
def test_la_reponse_indicielle_dans_les_trois_regimes(amortissement: sp.Expr) -> None:
    """(5.39), (5.42) et (5.45), contre l'inverse de Laplace de `H(p) / p`."""
    systeme = SecondOrdre(5, 2, amortissement)
    assert _proches(indicielle(systeme), original(systeme.de_bode().expression / P))
    assert sp.limit(indicielle(systeme), T, sp.oo) == 5


def test_la_constante_de_temps_d_amortissement_vaut_t_sur_xi() -> None:
    """Page 13 : `tau = T / xi = 1 / sigma`, et non `T`, sauf si `xi = 1`."""
    systeme = SecondOrdre(5, 2, DEMI)
    enveloppe = impulsionnelle(systeme) / sp.sin(systeme.pulsation_propre() * T)
    assert sp.simplify(sp.diff(sp.log(enveloppe), T) + DEMI / 2) == 0


def test_l_exemple_5_3_le_circuit_rlc_serie() -> None:
    """(5.21) a (5.24) : `2 sigma = R / L`, `rho^2 = 1 / (L C)`."""
    resistance, inductance, capacite = sp.symbols("R L C", positive=True)
    courant = (1 / inductance) / (
        P**2 + P * resistance / inductance + 1 / (inductance * capacite)
    )
    systeme = SecondOrdre.depuis_evans(
        1 / inductance,
        resistance / (2 * inductance),
        1 / sp.sqrt(inductance * capacite),
    )
    assert sp.simplify(systeme.de_bode().expression - courant) == 0
    critique = resistance_critique(inductance, capacite)
    assert sp.simplify(systeme.amortissement.subs(resistance, critique)) == 1


def test_l_instant_du_maximum_et_le_depassement() -> None:
    """(5.47) et (5.48) : `t_max = pi T / racine(1 - xi^2)`, `D = exp(-xi pi / ...)`."""
    systeme = SecondOrdre(5, 2, DEMI)
    sommet = instant_du_maximum(systeme)
    assert sommet == 2 * sp.pi / sp.sqrt(1 - DEMI**2)
    assert sp.simplify(sp.diff(indicielle(systeme), T).subs(T, sommet)) == 0
    depassement = (indicielle(systeme).subs(T, sommet) - 5) / 5
    assert sp.simplify(depassement - depassement_du_second_ordre(DEMI)) == 0
    assert periode_propre(systeme) == 2 * sommet


def test_la_periode_propre_non_amortie_de_la_figure_5_22() -> None:
    """« Periode propre non amortie : 2 pi T »."""
    assert periode_propre(SecondOrdre(5, 1, 0)) == 2 * sp.pi
    with pytest.raises(ValueError, match="pas de pulsation propre"):
        instant_du_maximum(SecondOrdre(5, 1, 2))


def test_l_identification_du_second_ordre_par_son_maximum() -> None:
    """5.2.3 : `t_max` et `D` rendent `xi` et `T`."""
    systeme = SecondOrdre(5, 2, DEMI)
    sommet = float(instant_du_maximum(systeme))
    depassement = float(depassement_du_second_ordre(DEMI))
    amortissement, constante = identifier_le_second_ordre(
        instant_du_maximum=sommet, depassement=depassement
    )
    assert amortissement == pytest.approx(0.5)
    assert constante == pytest.approx(2.0)
    with pytest.raises(ValueError, match="depassement hors de"):
        identifier_le_second_ordre(instant_du_maximum=sommet, depassement=1.5)


def _indicielle_reduite(amortissement: float) -> tuple[np.ndarray, np.ndarray]:
    instants = np.linspace(0, 40, 40_001)
    return instants, reponse_indicielle([1.0], [1.0, 2 * amortissement, 1.0], instants)


def test_le_temps_de_montee_vaut_0_32_sur_la_bande_passante() -> None:
    """(5.49), a quelques pour cent pres pour un amortissement de 0,7."""
    instants, reponse = _indicielle_reduite(0.7)
    montee = temps_de_montee(instants, reponse, regime=1.0)
    bande_en_hertz = bande_passante_du_second_ordre(1.0, 0.7) / (2 * math.pi)
    assert montee == pytest.approx(0.32 / bande_en_hertz, rel=0.08)
    basse, haute = temps_de_montee_depuis_la_bande(
        bande_passante_du_second_ordre(1.0, 0.7)
    )
    assert basse <= montee <= haute


def test_le_temps_d_etablissement_est_minimal_pres_de_racine_de_deux_sur_deux() -> None:
    """Page 20 : `t_e` minimal pour `xi = racine(2) / 2`.

    Le minimum exact tombe un peu avant, la ou le depassement vaut juste
    cinq pour cent, soit `xi = 0,6901` ; le syllabus arrondit. En deca,
    la reponse ressort de la bande et le temps d'etablissement bondit.
    """
    racine = math.sqrt(2) / 2
    durees = {
        xi: temps_d_etablissement(*_indicielle_reduite(xi), regime=1.0)
        for xi in (0.5, 0.6, 0.69, 0.7, racine, 0.8, 1.0)
    }
    autres = (durees[0.5], durees[0.6], durees[0.69], durees[0.8], durees[1.0])
    assert durees[0.7] < durees[racine] < min(autres)
    assert float(depassement_du_second_ordre(sp.Float(0.6901))) == pytest.approx(
        0.05, abs=1e-4
    )
    with pytest.raises(ValueError, match="regime nul"):
        temps_d_etablissement(*_indicielle_reduite(0.7), regime=0.0)
    with pytest.raises(ValueError, match="n'atteint pas"):
        temps_de_montee(np.array([0.0, 1.0]), np.array([0.0, 0.5]), regime=1.0)


def test_la_reponse_d_ordre_quelconque_porte_un_terme_de_regime() -> None:
    """(5.52) et (5.53) : residus `H(0)` en `p = 0` et `H(j omega)` en `p = j omega`."""
    transmittance = (P + 3) / ((P + 1) * (P**2 + 2 * P + 5))
    assert residu(transmittance / P, 0) == transmittance.subs(P, 0)
    excitation = 2 * sp.I
    harmonique = residu(transmittance / (P - excitation), excitation)
    assert sp.simplify(harmonique - transmittance.subs(P, excitation)) == 0
