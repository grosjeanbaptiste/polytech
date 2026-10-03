"""L'enclenchement d'un circuit du premier ordre sur un regime quelconque.

`x(t) = x_r(t) + (x(0) - x_r(0)) e^{-t/tau}` : le regime force, plus
l'exponentielle qui raccorde la condition initiale. C'est la methode du
cours -- solution particuliere plus solution homogene -- et elle vaut
pour un regime sinusoidal comme pour un palier.

L'oracle n'est pas la formule elle-meme : c'est l'equation differentielle
du circuit, que le resultat doit annuler, et la condition initiale, qu'il
doit respecter. Deux verifications qui ne passent pas par la construction.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.enclenchement import enclenchement
from polytech.electromagnetisme.domain.symboles import PULSATION, TEMPS

E, R, L, THETA = sp.symbols("E R L theta", positive=True)
TAU = L / R
AMPLITUDE = E * sp.sqrt(2) / sp.sqrt(R**2 + (PULSATION * L) ** 2)
REGIME = AMPLITUDE * sp.cos(PULSATION * TEMPS + THETA - sp.atan(PULSATION * L / R))
"""Le regime force du circuit `R`-`L` sous `E racine 2 cos(omega t + theta)`."""


def test_le_courant_annule_l_equation_du_circuit_rl() -> None:
    """`L di/dt + R i = e(t)` : le residu doit etre nul."""
    courant = enclenchement(REGIME, sp.Integer(0), TAU)
    source = E * sp.sqrt(2) * sp.cos(PULSATION * TEMPS + THETA)
    residu = L * sp.diff(courant, TEMPS) + R * courant - source
    assert sp.simplify(sp.expand_trig(residu)) == 0


def test_le_courant_part_de_sa_valeur_initiale() -> None:
    """Une bobine interdit le saut : `i(0) = 0` avant comme apres."""
    courant = enclenchement(REGIME, sp.Integer(0), TAU)
    assert sp.simplify(courant.subs(TEMPS, 0)) == 0


def test_une_valeur_initiale_non_nulle_est_respectee() -> None:
    """Le raccord porte la difference, et non la seule valeur du regime."""
    initial = sp.Symbol("i_0")
    courant = enclenchement(REGIME, initial, TAU)
    assert sp.simplify(courant.subs(TEMPS, 0) - initial) == 0


def test_un_palier_redonne_la_reponse_du_premier_ordre() -> None:
    """Regime constant : `y_inf + (y_0 - y_inf) e^{-t/tau}`, la forme connue."""
    palier, depart = sp.symbols("y_inf y_0")
    attendu = palier + (depart - palier) * sp.exp(-TEMPS / TAU)
    assert sp.simplify(enclenchement(palier, depart, TAU) - attendu) == 0


def test_une_constante_de_temps_nulle_est_refusee() -> None:
    """`tau = 0` : pas de transitoire, et une division par zero."""
    with pytest.raises(ValueError, match="constante de temps"):
        enclenchement(REGIME, sp.Integer(0), sp.Integer(0))
