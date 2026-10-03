"""Deux transmittances que le TP 1 fait etablir : un circuit, une equation.

Le circuit `L`-`C` serie ferme sur `R` se lit comme un diviseur
d'impedances ; l'equation differentielle se transforme terme a terme, la
derivation devenant une multiplication par `p` a etat initial nul. Les
oracles sont la forme du cours, `R C p / (L C p^2 + R C p + 1)`, et la
condition de poles reels, `R > 2 racine(L / C)`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.signaux.domain.modeles import (
    coefficients,
    resistance_critique,
    transmittance_d_une_equation,
    transmittance_lc_serie_sur_r,
)
from polytech.signaux.domain.symboles import IMAGE

R, L, C = sp.symbols("R L C", positive=True)


def test_le_diviseur_donne_la_forme_du_cours() -> None:
    attendu = R * C * IMAGE / (L * C * IMAGE**2 + R * C * IMAGE + 1)
    assert sp.simplify(transmittance_lc_serie_sur_r(R, L, C).expression - attendu) == 0


def test_a_la_resistance_critique_le_discriminant_s_annule() -> None:
    denominateur = transmittance_lc_serie_sur_r(R, L, C).denominateur()
    a, b, c = sp.Poly(denominateur, IMAGE).all_coeffs()
    discriminant = (b**2 - 4 * a * c).subs(R, resistance_critique(L, C))
    assert sp.simplify(discriminant) == 0


def test_au_dela_de_la_resistance_critique_les_poles_sont_reels() -> None:
    valeurs = {L: 1, C: sp.Rational(1, 10**7)}
    au_dela = 2 * resistance_critique(L, C).subs(valeurs)
    poles = transmittance_lc_serie_sur_r(au_dela, 1, sp.Rational(1, 10**7)).poles()
    assert all(sp.im(pole) == 0 for pole in poles)


def test_une_equation_differentielle_se_transforme_terme_a_terme() -> None:
    """`y''' + y'' + 10 y' + 15 y = 5 r` donne `5 / (p^3 + p^2 + 10 p + 15)`."""
    transmittance = transmittance_d_une_equation((1, 1, 10, 15), (5,))
    attendu = 5 / (IMAGE**3 + IMAGE**2 + 10 * IMAGE + 15)
    assert sp.simplify(transmittance.expression - attendu) == 0


def test_une_equation_sans_terme_de_sortie_est_refusee() -> None:
    with pytest.raises(ValueError, match="sortie"):
        transmittance_d_une_equation((), (5,))


def test_les_coefficients_sont_rendus_par_degre_decroissant() -> None:
    transmittance = transmittance_d_une_equation((1, 1, 10, 15), (5,))
    assert coefficients(transmittance) == ([5.0], [1.0, 1.0, 10.0, 15.0])
