"""Les examens de signaux : des reponses des sujets refaites par la bibliotheque.

Le releve `cursus/examens/SignauxSystemes.toml` nomme les notions que chaque
sujet mobilise. Les solutions officielles sont des scans manuscrits que la
conversion n'a pas lus ; trois sujets ecrivent pourtant leurs reponses en
clair, et ce fichier les refait par `polytech.signaux`, `polytech.maths` et
sympy. Il tient aussi les resultats que la fiche de revision derive.

Les sujets : `Exam_2017_Juin_Solution_partie_1_v1` (session de septembre
2016), `Exam_2019_Septembre_Enonce`, `Exam_2018_Juin_Enonce`,
`Exam_2016_Juin_Enonce` et `Exam_2026_Juin_Charleroi_Enonce`, dans
`SignauxSystemes/EXAMENS/`.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.domain import abcd
from polytech.maths.domain.laplace.theoremes import valeur_finale, valeur_initiale
from polytech.maths.domain.symboles import IMAGE
from polytech.signaux.domain.classification import (
    puissance_sur_une_periode,
    valeur_efficace_sur_une_periode,
)
from polytech.signaux.domain.elementaires import rapport_signal_sur_bruit_en_decibels
from polytech.signaux.domain.frequence import INSTANT as T
from polytech.signaux.domain.second_ordre import SecondOrdre
from polytech.signaux.domain.stabilite_des_poles import nature_de_la_stabilite
from polytech.signaux.domain.symboles import IMAGE as P

PERIODE = sp.Symbol("T_0", positive=True)
PULSATION = 2 * sp.pi / PERIODE


def _en_image(transmittance: sp.Expr) -> sp.Expr:
    """La meme fraction, dans la variable que `polytech.maths` emploie."""
    return transmittance.subs(P, IMAGE)


# --- Septembre 2016 ---------------------------------------------------------


def test_un_carre_et_une_sinusoide_de_meme_amplitude_sont_a_trois_decibels() -> None:
    """Question 1, « Reponse : 3dB » : le carre porte deux fois la puissance."""
    sinusoide = float(puissance_sur_une_periode(sp.sin(2 * PULSATION * T), PERIODE))
    carre = 1.0
    assert sinusoide == pytest.approx(0.5)
    assert rapport_signal_sur_bruit_en_decibels(carre, sinusoide) == pytest.approx(
        3.0, abs=0.02
    )


def _deux_cellules_cr() -> sp.Expr:
    """Deux cellules « condensateur en serie, resistance en derivation »."""
    resistance, capacite = sp.symbols("R C", positive=True)
    cellule = abcd.cascade(abcd.serie(1 / (P * capacite)), abcd.derivation(resistance))
    return sp.simplify(1 / abcd.cascade(cellule, cellule)[0, 0])


def test_la_transmittance_de_deux_cellules_cr_en_cascade() -> None:
    """Question 3, reponse a : `p^2 / (p^2 + 3 p / RC + 1 / R^2 C^2)`."""
    resistance, capacite = sp.symbols("R C", positive=True)
    constante = resistance * capacite
    attendu = P**2 / (P**2 + 3 * P / constante + 1 / constante**2)
    assert sp.simplify(_deux_cellules_cr() - attendu) == 0


def test_ses_poles_sont_reels_et_la_reponse_s_eteint_en_cinq_constantes() -> None:
    """Reponses c et d, pour R = 10 ohms et C = 1 mF."""
    resistance, capacite = sp.symbols("R C", positive=True)
    chiffree = _deux_cellules_cr().subs(
        {resistance: 10, capacite: sp.Rational(1, 1000)}
    )
    poles = sorted(float(pole) for pole in sp.roots(sp.denom(sp.together(chiffree)), P))
    assert poles == pytest.approx([-261.8, -38.2], abs=0.05)
    assert nature_de_la_stabilite(chiffree) == "strictement stable"
    assert 5 / abs(poles[-1]) == pytest.approx(0.131, abs=1e-3)


def test_la_reponse_impulsionnelle_de_ce_filtre_s_annule_en_regime() -> None:
    """Reponse b : une impulsion au depart, zero a l'infini."""
    assert valeur_finale(_en_image(_deux_cellules_cr())) == 0
    assert sp.limit(_deux_cellules_cr(), P, sp.oo) == 1


# --- Septembre 2019 ---------------------------------------------------------


def test_la_periode_qui_met_le_signal_vingt_decibels_sous_le_bruit() -> None:
    """Reponse 1 : `P(x) = 1 / T0`, `P(sinus) = 1 / 2`, donc `T0 = 200 s`."""
    creneau = sp.Piecewise((1, sp.Abs(T) < sp.Rational(1, 2)), (0, True))
    for periode in (2, 50, 200):
        puissance = puissance_sur_une_periode(creneau, sp.Integer(periode))
        assert puissance == sp.Rational(1, periode)
    bruit = puissance_sur_une_periode(sp.sin(2 * sp.pi * T), sp.Integer(1))
    assert bruit == sp.Rational(1, 2)
    solution = sp.solve(sp.Eq(10 * sp.log((1 / PERIODE) / bruit, 10), -20), PERIODE)
    assert solution == [200]


# --- Juin 2018, question 3 --------------------------------------------------


def test_trois_decibels_de_mieux_demandent_de_faire_sortir_la_moitie() -> None:
    """Le bruit est proportionnel au nombre d'etudiants : 128, puis 64."""
    restants = 128 / 10 ** (3 / 10)
    assert round(restants) == 64
    assert rapport_signal_sur_bruit_en_decibels(128, restants) == pytest.approx(3.0)


def test_un_seul_etudiant_bruyant_laisse_cinquante_et_un_decibels() -> None:
    assert 30 + rapport_signal_sur_bruit_en_decibels(128, 1) == pytest.approx(
        51.07, abs=0.01
    )


def test_trois_decibels_par_la_voix_demandent_racine_de_deux_en_amplitude() -> None:
    """La puissance double ; l'amplitude est multipliee par 1,41."""
    amplitude = 10 ** (3 / 20)
    assert amplitude == pytest.approx(math.sqrt(2), rel=2e-3)


# --- Juin 2016, question 3 --------------------------------------------------


def test_la_boucle_ouverte_de_juin_2016_est_stable_et_son_gain_tend_vers_deux() -> None:
    ouverte = 20 * P * (P + sp.Rational(1, 2)) / ((10 * P + 1) * (P + 1))
    assert nature_de_la_stabilite(ouverte) == "strictement stable"
    assert sp.limit(ouverte, P, sp.oo) == 2
    assert float(20 * sp.log(2, 10)) == pytest.approx(6.02, abs=0.01)


# --- Juin 2026 ---------------------------------------------------------------


def test_la_puissance_de_la_somme_de_deux_harmoniques_orthogonales() -> None:
    """Question 1 : `3 cos(w0 t) - sin(2 w0 t)` porte 9/2 + 1/2 = 5."""
    signal = 3 * sp.cos(PULSATION * T) - sp.sin(2 * PULSATION * T)
    assert sp.simplify(puissance_sur_une_periode(signal, PERIODE)) == 5
    efficace = valeur_efficace_sur_une_periode(signal, PERIODE)
    assert sp.simplify(efficace - sp.sqrt(5)) == 0


def test_l_amplitude_du_bruit_carre_pour_dix_decibels() -> None:
    """Un carre d'amplitude B porte `B^2` : `B^2 = 5 / 10`."""
    amplitude = sp.Symbol("B", positive=True)
    solution = sp.solve(sp.Eq(10 * sp.log(5 / amplitude**2, 10), 10), amplitude)
    assert solution == [sp.sqrt(2) / 2]


def test_le_denominateur_impose_a_la_question_2_est_au_regime_critique() -> None:
    """`p^2 + 2000 p + 1e6 = (p + 1000)^2` : poles reels confondus."""
    racines = sp.roots(P**2 + 2000 * P + 10**6, P)
    assert racines == {-1000: 2}
    forme = SecondOrdre(gain=1, constante=sp.Rational(1, 1000), amortissement=1)
    assert forme.regime() == "poles reels confondus"
    assert set(forme.poles()) == {-1000}


# --- Ce que la fiche derive elle-meme ---------------------------------------


def test_un_integrateur_dans_la_boucle_annule_l_erreur_de_consigne() -> None:
    """Article « Les erreurs en regime » : classe 0, `1 / (1 + K)` ; classe 1, zero."""
    gain = 4
    sans = valeur_finale((1 / (1 + gain / (IMAGE + 1))) / IMAGE)
    avec = valeur_finale((1 / (1 + gain / (IMAGE * (IMAGE + 1)))) / IMAGE)
    assert sans == sp.Rational(1, 1 + gain)
    assert avec == 0


def test_la_valeur_initiale_d_une_reponse_indicielle_est_le_gain_a_l_infini() -> None:
    """Article « Les valeurs initiale et finale » : `a(0+) = H(oo)`, `a(oo) = H(0)`."""
    transmittance = (2 * IMAGE + 3) / (IMAGE + 5)
    assert valeur_initiale(transmittance / IMAGE) == 2
    assert valeur_finale(transmittance / IMAGE) == sp.Rational(3, 5)
