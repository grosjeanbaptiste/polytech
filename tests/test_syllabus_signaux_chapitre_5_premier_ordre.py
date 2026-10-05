"""Le syllabus de signaux et systemes, chapitre 5 : le premier ordre.

Source : `SignauxSystemes/SYLLABUS/Syllabus_2019_Chapitre_05_Analyse_temporelle.pdf`,
section 5.1 (pages 1 a 9), lue sur les pages rendues.

Les oracles : le tableau 5.1, les 63,2 % de la figure 5.6, l'exemple 5.2
(`K = 5,05`, `T = 2,96`), l'erreur de trainage `K T` et les reponses
(5.3) a (5.12).
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from polytech.electronique.domain.laplace import (
    impedance_de_condensateur,
    impedances_en_parallele,
)
from polytech.signaux.domain.harmonique import AnalyseHarmonique
from polytech.signaux.domain.heaviside import original
from polytech.signaux.domain.identification import (
    identifier_le_premier_ordre,
    parametres_depuis_la_droite,
)
from polytech.signaux.domain.premier_ordre import PremierOrdre
from polytech.signaux.domain.symboles import IMAGE, PULSATION, TEMPS
from polytech.signaux.domain.temporel import reponse_a_une_rampe

P, T, W = IMAGE, TEMPS, PULSATION
K, TAU = sp.symbols("K T", positive=True)
SYSTEME = PremierOrdre(gain=K, constante=TAU)
INSTANTS = np.array([0.0, 0.4, 1.3, 5.0])


def _proches(
    calcule: sp.Expr, attendu: sp.Expr, valeurs: dict[sp.Symbol, float]
) -> bool:
    ecart = sp.lambdify(T, (calcule - attendu).subs(valeurs), "numpy")
    return bool(np.allclose(ecart(INSTANTS), 0, atol=1e-10))


def test_les_formes_de_bode_et_d_evans() -> None:
    """(5.1) et (5.2) : `K / (p T + 1) = K_e / (p + a)`, `K_e = K / T`, `a = 1 / T`."""
    gain_d_evans, pole = SYSTEME.d_evans()
    assert gain_d_evans == K / TAU
    assert pole == 1 / TAU
    bode = SYSTEME.de_bode().expression
    assert sp.simplify(bode - gain_d_evans / (P + pole)) == 0
    with pytest.raises(ValueError, match="constante de temps nulle ou negative"):
        PremierOrdre(gain=1, constante=0)


def test_l_exemple_5_1_le_circuit_rc_sous_une_impulsion_de_courant() -> None:
    """`U(p) = R / (1 + p R C)`, soit `K = R` et `T = R C`."""
    resistance, capacite = sp.symbols("R C", positive=True)
    tension = impedances_en_parallele(
        resistance, impedance_de_condensateur(capacite, P)
    )
    circuit = PremierOrdre(gain=resistance, constante=resistance * capacite)
    assert sp.simplify(tension - circuit.de_bode().expression) == 0
    assert circuit.d_evans() == (1 / capacite, 1 / (resistance * capacite))


def test_la_reponse_impulsionnelle_et_sa_tangente_a_l_origine() -> None:
    """(5.3) : `h = (K / T) exp(-t / T)`, de pente initiale `-1 / T` fois `h(0)`."""
    reponse = SYSTEME.impulsionnelle()
    assert sp.simplify(reponse - original(SYSTEME.de_bode().expression)) == 0
    depart = reponse.subs(T, 0)
    assert sp.simplify(sp.diff(reponse, T).subs(T, 0) + depart / TAU) == 0


@pytest.mark.parametrize(("multiple", "reste"), [(1, 0.368), (3, 0.050), (5, 0.007)])
def test_le_tableau_5_1_de_l_exponentielle_decroissante(
    multiple: int, reste: float
) -> None:
    unitaire = PremierOrdre(gain=1, constante=1)
    assert float(unitaire.impulsionnelle().subs(T, multiple)) == pytest.approx(
        reste, abs=5e-4
    )


def test_la_reponse_indicielle_atteint_63_pour_cent_en_une_constante_de_temps() -> None:
    """(5.4), fig. 5.6 : `a = K (1 - exp(-t / T))`, tangente coupant `K` en `T`."""
    reponse = SYSTEME.indicielle()
    assert sp.simplify(reponse - original(SYSTEME.de_bode().expression / P)) == 0
    assert float((reponse / K).subs(T, TAU)) == pytest.approx(0.632, abs=5e-4)
    assert sp.simplify(sp.diff(reponse, T).subs(T, 0) * TAU - K) == 0


def test_l_exemple_5_2_la_droite_du_graphique_logarithmique() -> None:
    """(5.5) : `ln(K - a) = ln K - t / T` ; `y = -0,338 x + 1,62` rend 5,05 et 2,96."""
    gain, constante = parametres_depuis_la_droite(pente=-0.338, ordonnee=1.62)
    assert gain == pytest.approx(5.05, abs=0.005)
    assert constante == pytest.approx(2.96, abs=0.005)
    with pytest.raises(ValueError, match="pente nulle ou positive"):
        parametres_depuis_la_droite(pente=0.1, ordonnee=1.62)


def test_l_identification_retrouve_le_systeme_de_l_exemple_5_2() -> None:
    """Sans bruit, la mesure de `5 / (1 + 3 p)` rend exactement 5 et 3."""
    instants = np.linspace(0, 10, 101)
    mesure = 5 * (1 - np.exp(-instants / 3))
    gain, constante = identifier_le_premier_ordre(instants, mesure, regime=5.0)
    assert gain == pytest.approx(5.0)
    assert constante == pytest.approx(3.0)
    with pytest.raises(ValueError, match="regime deja atteint"):
        identifier_le_premier_ordre(instants, mesure, regime=4.0)


def test_la_reponse_a_la_rampe_et_l_erreur_de_trainage() -> None:
    """(5.8) : `y = K t - K T (1 - exp(-t / T))`, en retard de `T` sur la rampe."""
    reponse = SYSTEME.reponse_a_la_rampe()
    valeurs = {K: 5.0, TAU: 3.0}
    par_laplace = reponse_a_une_rampe(SYSTEME.de_bode())
    assert _proches(reponse, par_laplace, valeurs)
    assert SYSTEME.erreur_de_trainage() == K * TAU
    assert sp.limit(K * T - reponse, T, sp.oo) == SYSTEME.erreur_de_trainage()


def test_la_reponse_a_une_cosinusoide() -> None:
    """(5.10) : un terme de regime, et un transitoire qui annule la sortie en zero."""
    valeurs = {K: 5.0, TAU: 3.0}
    reponse = SYSTEME.reponse_au_cosinus(2)
    par_laplace = original(SYSTEME.de_bode().expression * P / (P**2 + 4))
    assert _proches(reponse, par_laplace, valeurs)
    assert sp.simplify(reponse.subs(T, 0)) == 0


def test_la_reponse_a_une_sinusoide_est_la_meme_dephasee_d_un_quart_de_tour() -> None:
    """(5.11), par une phase initiale de `-pi / 2`."""
    valeurs = {K: 5.0, TAU: 3.0}
    reponse = SYSTEME.reponse_au_cosinus(2, phase=-sp.pi / 2)
    par_laplace = original(SYSTEME.de_bode().expression * 2 / (P**2 + 4))
    assert _proches(reponse, par_laplace, valeurs)


def test_le_regime_harmonique_ne_differe_de_l_entree_que_par_un_gain_et_une_phase() -> (
    None
):
    """(5.12) : `K / racine(omega^2 T^2 + 1)` et `-arctan(omega T)`."""
    gain, phase = SYSTEME.regime_harmonique(W)
    assert gain == K / sp.sqrt(W**2 * TAU**2 + 1)
    assert phase == -sp.atan(W * TAU)
    analyse = AnalyseHarmonique(SYSTEME.de_bode())
    nombres = {K: 5.0, TAU: 3.0, W: 2.0}
    assert float(analyse.module(W).subs(nombres)) == pytest.approx(
        float(gain.subs(nombres))
    )
    assert float(analyse.argument(W).subs(nombres)) == pytest.approx(
        float(phase.subs(nombres))
    )
