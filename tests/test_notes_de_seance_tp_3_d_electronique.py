"""Les notes de seance du TP 3 d'electronique, recalculees.

Source : `ElectroniqueInstrumentation/TP/TP_3_Notes_de_seance_2026_2027.pdf`,
scan manuscrit de dix pages (la derniere blanche), transcrit a la main dans
`transcriptions/ElectroniqueInstrumentation/TP/`.

Le manuscrit inverse la transformee de Laplace par la table et les fractions
simples ; les tests reprennent chaque original par `sp.inverse_laplace_transform`
ou par la transformee directe, et chaque quadripole par une cascade de
briques, la ou le manuscrit prend les formules de la cellule en T. C'est ce
second chemin qui dement le signe de la forme rationalisee de `c_2^+`, que la
transcription signale.
"""

from __future__ import annotations

import cmath
import math
from pathlib import Path

import sympy as sp

from polytech.electronique.domain.abcd import (
    cascade,
    cellule_en_t,
    derivation,
    gain_en_tension,
    serie,
)
from polytech.electronique.domain.laplace import (
    decomposition_en_fractions_simples,
    diviseur_de_courant,
    diviseur_de_tension,
    impedance_de_bobine,
    impedance_de_condensateur,
    impedance_de_resistance,
    impedances_en_parallele,
    impedances_en_serie,
    residu_en_un_pole,
)

RACINE = Path(__file__).resolve().parent.parent
NOTES = (
    RACINE
    / "transcriptions/ElectroniqueInstrumentation/TP"
    / "TP_3_Notes_de_seance_2026_2027.md"
)
S = sp.Symbol("s")
T = sp.Symbol("t", positive=True)
W = sp.Symbol("omega", positive=True)
R, L, C, R_L = sp.symbols("R L C R_L", positive=True)
Z, Z_1, Z_2, Z_3 = sp.symbols("Z Z_1 Z_2 Z_3")


def _nul(expression: sp.Expr) -> bool:
    return bool(sp.simplify(expression) == 0)


def _passe_haut() -> sp.Expr:
    return diviseur_de_tension(
        impedance_de_condensateur(C, S), impedance_de_resistance(R, S)
    )


def test_l_exemple_du_rappel_a_un_pole_double() -> None:
    """`(s + 1)/(s + 2)^2 = 1/(s + 2) - 1/(s + 2)^2` : `a_{1,2} = -1`."""
    termes = decomposition_en_fractions_simples((S + 1) / (S + 2) ** 2, S)
    assert set(termes) == {1 / (S + 2), -1 / (S + 2) ** 2}


def test_le_transfert_du_passe_haut() -> None:
    """Diviseur de tension : `sCR / (sCR + 1) = s / (s + 1/RC)`."""
    assert _nul(_passe_haut() - S / (S + 1 / (R * C)))


def test_la_reponse_indicielle_s_eteint() -> None:
    """`V_o = 1/(s + 1/RC)`, d'ou `exp(-t/RC)`, qui tend vers zero."""
    sortie = sp.simplify(_passe_haut() / S)
    assert _nul(sortie - 1 / (S + 1 / (R * C)))
    assert residu_en_un_pole(sortie, S, -1 / (R * C)) == 1
    original = sp.inverse_laplace_transform(sortie, S, T)
    assert _nul(original - sp.exp(-T / (R * C)))
    assert sp.limit(original, T, sp.oo) == 0


def _sortie_au_cosinus() -> sp.Expr:
    return sp.simplify(_passe_haut() * S / (S**2 + W**2))


def test_les_coefficients_de_la_reponse_au_cosinus() -> None:
    """`c_1 = 1/(1 + R^2 C^2 w^2)` et `c_2^+ = jw / (2 (jw + 1/RC))`."""
    sortie = _sortie_au_cosinus()
    assert _nul(sortie - S**2 / ((S + 1 / (R * C)) * (S**2 + W**2)))
    c_1 = residu_en_un_pole(sortie, S, -1 / (R * C))
    assert _nul(c_1 - 1 / (1 + R**2 * C**2 * W**2))
    c_2 = residu_en_un_pole(sortie, S, sp.I * W)
    assert _nul(c_2 - sp.I * W / (2 * (sp.I * W + 1 / (R * C))))
    assert _nul(sp.Abs(c_2) - W / (2 * sp.sqrt(1 / (R**2 * C**2) + W**2)))
    for valeur in (0.1, 1.0, 10.0):
        point = {W: valeur, R: 1, C: 1}
        phase = cmath.phase(complex(c_2.subs(point)))
        assert math.isclose(phase, math.pi / 2 - math.atan(valeur))


def test_la_forme_rationalisee_de_c_2_porte_un_plus() -> None:
    """`(w^2 + jw/RC) / (2 (1/R^2C^2 + w^2))`, et non `w^2 - jw/RC`."""
    c_2 = sp.I * W / (2 * (sp.I * W + 1 / (R * C)))
    denominateur = 2 * (1 / (R**2 * C**2) + W**2)
    assert _nul(c_2 - (W**2 + sp.I * W / (R * C)) / denominateur)
    assert not _nul(c_2 - (W**2 - sp.I * W / (R * C)) / denominateur)
    assert "et non" in NOTES.read_text(encoding="utf-8")


def _original_du_manuscrit() -> sp.Expr:
    transitoire = sp.exp(-T / (R * C)) / (1 + W**2 * R**2 * C**2)
    amplitude = W / sp.sqrt(1 / (R**2 * C**2) + W**2)
    return transitoire + amplitude * sp.cos(W * T + sp.pi / 2 - sp.atan(W * R * C))


def test_l_original_redonne_la_transformee() -> None:
    """La transformee directe de `v_o(t)`, en trois points numeriques."""
    for point in ({R: 1, C: 1, W: 2}, {R: 2, C: sp.Rational(1, 3), W: 5}):
        original = _original_du_manuscrit().subs(point)
        transformee = sp.laplace_transform(original, T, S, noconds=True)
        attendue = _sortie_au_cosinus().subs(point)
        for valeur in (1, 3, 7):
            assert math.isclose(
                float(transformee.subs(S, valeur)), float(attendue.subs(S, valeur))
            )


def test_a_pulsation_nulle_on_retrouve_l_echelon() -> None:
    """`w = 0` : le cosinus devient l'echelon, et `v_o = exp(-t/RC)`."""
    assert _nul(_original_du_manuscrit().subs(W, 0) - sp.exp(-T / (R * C)))


def test_la_partie_asymptotique_est_celle_du_phaseur() -> None:
    """Le point 4, que le manuscrit barre : `|H(jw)| cos(wt + arg H(jw))`."""
    reponse = _passe_haut().subs(S, sp.I * W)
    point = {R: 2, C: sp.Rational(1, 4), W: 3}
    phaseur = complex(reponse.subs(point))
    amplitude = W / sp.sqrt(1 / (R**2 * C**2) + W**2)
    assert math.isclose(abs(phaseur), float(amplitude.subs(point)))
    assert math.isclose(
        cmath.phase(phaseur), math.pi / 2 - math.atan(float((W * R * C).subs(point)))
    )


def test_la_cellule_en_t_par_la_cascade_de_ses_briques() -> None:
    """`Z_1` en serie, `Z_3` en derivation, `Z_2` en serie : A, B, C, D."""
    briques = cascade(serie(Z_1), derivation(Z_3), serie(Z_2))
    attendue = sp.Matrix(
        [[1 + Z_1 / Z_3, Z_1 + Z_2 + Z_1 * Z_2 / Z_3], [1 / Z_3, 1 + Z_2 / Z_3]]
    )
    assert sp.simplify(briques - attendue) == sp.zeros(2, 2)
    assert sp.simplify(cellule_en_t(Z_1, Z_2, Z_3) - attendue) == sp.zeros(2, 2)


def test_d_par_le_diviseur_de_courant() -> None:
    """`-I_2 = I_1 Z_3 / (Z_2 + Z_3)`, donc `D = 1 + Z_2/Z_3`."""
    part = diviseur_de_courant(Z_2, Z_3)
    assert _nul(part - Z_3 / (Z_2 + Z_3))
    assert _nul(1 / part - (1 + Z_2 / Z_3))


def test_les_deux_topologies_simples() -> None:
    """`Z_3 -> oo` donne la serie, `Z_1 = Z_2 = 0` la derivation."""
    serie_limite = cellule_en_t(Z, 0, Z_3).applyfunc(lambda x: sp.limit(x, Z_3, sp.oo))
    assert serie_limite == serie(Z)
    assert cellule_en_t(0, 0, Z) == derivation(Z)


def _cascade_de_l_exercice_3_2() -> sp.Matrix:
    return cascade(
        serie(impedance_de_resistance(R, S)),
        derivation(impedance_de_condensateur(C, S)),
        serie(impedance_de_bobine(L, S)),
    )


def test_la_matrice_globale_de_l_exercice_3_2() -> None:
    """`[[1 + sCR, sL + R + s^2 LCR], [sC, 1 + s^2 LC]]`."""
    attendue = sp.Matrix(
        [[1 + S * C * R, S * L + R + S**2 * L * C * R], [S * C, 1 + S**2 * L * C]]
    )
    assert sp.simplify(_cascade_de_l_exercice_3_2() - attendue) == sp.zeros(2, 2)


def test_le_transfert_charge_par_deux_diviseurs() -> None:
    """Le gain ABCD, recoupe par deux diviseurs de tension successifs."""
    abcd = gain_en_tension(_cascade_de_l_exercice_3_2(), R_L)
    attendu = R_L / (S**2 * L * C * R + S * (C * R * R_L + L) + R + R_L)
    assert _nul(abcd - attendu)
    branche = impedances_en_serie(impedance_de_bobine(L, S), R_L)
    noeud = impedances_en_parallele(impedance_de_condensateur(C, S), branche)
    diviseurs = diviseur_de_tension(R, noeud) * diviseur_de_tension(
        impedance_de_bobine(L, S), R_L
    )
    assert _nul(diviseurs - attendu)


def _gain_numerique() -> sp.Expr:
    valeurs = {
        R: 32,
        R_L: 32,
        L: 32 / sp.sqrt(2) * sp.Rational(1, 10**5),
        C: 1 / (32 * sp.sqrt(2)) * sp.Rational(1, 10**5),
    }
    return gain_en_tension(_cascade_de_l_exercice_3_2(), R_L).subs(valeurs)


def test_le_gain_numerique_est_un_butterworth() -> None:
    """`|H(jw)|^2 = (1/4) / (1 + (10^-5 w / 2)^4)`, et `|H(j0)|^2 = 1/4`."""
    module = sp.Abs(_gain_numerique().subs(S, sp.I * W)) ** 2
    attendu = sp.Rational(1, 4) / (1 + (W / (2 * 10**5)) ** 4)
    assert _nul(module - attendu)
    assert sp.limit(module, W, 0) == sp.Rational(1, 4)


def test_la_coupure_a_deux_cent_mille_radians_par_seconde() -> None:
    """`w_c = 2 10^5 rad/s`, ou la perte d'insertion vaut 3,0103 dB."""
    module = sp.Abs(_gain_numerique().subs(S, sp.I * W)) ** 2
    rapport = module / sp.Rational(1, 4)
    assert sp.solve(sp.Eq(rapport, sp.Rational(1, 2)), W) == [2 * 10**5]
    perte = -10 * math.log10(float(rapport.subs(W, 2 * 10**5)))
    assert math.isclose(perte, 3.0103, abs_tol=5e-5)
