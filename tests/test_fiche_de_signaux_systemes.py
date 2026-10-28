"""Fiche de signaux : transmittances, poles, convolution, second ordre.

Les articles du bloc « Transformees et systemes » de
`cursus/fiches/SignauxSystemes.toml`, refaits par `polytech.signaux`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electronique.domain import abcd, laplace
from polytech.maths.domain.comparaison import egales, proches
from polytech.signaux.domain.bode import classe_du_systeme
from polytech.signaux.domain.fluence import (
    coefficients_normalises,
    regle_de_mason,
    transmittance_du_graphe,
)
from polytech.signaux.domain.frequence import INSTANT
from polytech.signaux.domain.produits import convolution_en
from polytech.signaux.domain.stabilite_des_poles import nature_de_la_stabilite
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T
from polytech.signaux.domain.transmittance import Transmittance

R, C = sp.symbols("R C", positive=True)
A, B = sp.symbols("a b", positive=True)
H_ = sp.Symbol("h", positive=True)


def test_le_diviseur_donne_la_cellule_rc_passe_bas() -> None:
    """Article « La transmittance » : `Z_aval / (Z_amont + Z_aval)`."""
    zc = laplace.impedance_de_condensateur(C, P)
    h = laplace.diviseur_de_tension(laplace.impedance_de_resistance(R, P), zc)
    assert egales(h, 1 / (1 + R * C * P))


def test_deux_cellules_en_cascade_portent_un_trois_et_non_un_deux() -> None:
    """Septembre 2016 (3) : `p^2 / (p^2 + 3p/RC + 1/R^2C^2)`."""
    cellule = abcd.cascade(abcd.serie(1 / (P * C)), abcd.derivation(R))
    deux = abcd.cascade(cellule, cellule)
    h = 1 / deux[0, 0]
    assert egales(h, P**2 / (P**2 + 3 * P / (R * C) + 1 / (R * C) ** 2))
    assert not egales(h, (1 / cellule[0, 0]) ** 2)


def test_ordre_et_classe_d_une_transmittance() -> None:
    """Article « Poles, zeros, ordre et classe »."""
    h = Transmittance(expression=3 * (P + 1) / (P**2 * (P + 2) * (P + 3)))
    assert (h.degre(), classe_du_systeme(h.expression)) == (4, 2)
    assert set(h.zeros()) == {-1}
    assert set(h.poles()) == {0, -2, -3}


def test_la_stabilite_se_lit_sur_les_poles() -> None:
    """Juin 2016 (3) : poles `-0,1` et `-1`, `H(0) = 0`, `H(oo) = 2`."""
    h = 20 * P * (P + sp.Rational(1, 2)) / ((10 * P + 1) * (P + 1))
    assert set(Transmittance(expression=h).poles()) == {sp.Rational(-1, 10), -1}
    assert nature_de_la_stabilite(h) == "strictement stable"
    assert (h.subs(P, 0), sp.limit(h, P, sp.oo)) == (0, 2)
    assert nature_de_la_stabilite(1 / (P * (P + 1))) == "stable au sens large"
    assert nature_de_la_stabilite(1 / (P - 1)) == "instable"


def test_la_cascade_cr_a_des_poles_reels_distincts_et_cinq_tau() -> None:
    """Septembre 2016 (3) : la reponse s'eteint en `5 tau` du pole le plus lent."""
    denominateur = Transmittance(expression=1 / (P**2 + 3 * P + 1))
    lent, rapide = sorted(denominateur.poles(), key=lambda pole: abs(float(pole)))
    assert float(rapide) < float(lent) < 0
    tau = -1 / lent
    assert proches(float(sp.exp(-5)), 0.0067, 1e-4)
    assert egales(tau, 2 / (3 - sp.sqrt(5)))


def test_la_sortie_est_nulle_hors_de_la_somme_des_supports() -> None:
    """Article « La convolution » : `[t_x + t_h, T_x + T_h]`, ici `[4, 7]`."""
    entree = sp.Piecewise((1, (INSTANT >= 1) & (INSTANT <= 2)), (0, True))
    reponse = sp.Piecewise((1, (INSTANT >= 3) & (INSTANT <= 5)), (0, True))
    mesures = [convolution_en(entree, reponse, sp.Rational(x)) for x in (3.9, 7.1)]
    assert mesures == [0, 0]
    assert egales(convolution_en(entree, reponse, sp.Rational(11, 2)), 1)


def test_la_reponse_indicielle_est_l_integrale_de_la_reponse_impulsionnelle() -> None:
    """Article « La convolution » : `a(t) = integrale de 0 a t de h`."""
    u = sp.Symbol("u", positive=True)
    h = laplace.reponse_impulsionnelle(1 / (P + 2), P, u)
    a = laplace.reponse_a_un_echelon(1 / (P + 2), 1, P, T)
    assert egales(sp.integrate(h, (u, 0, T)), a)


def test_le_graphe_canonique_a_deux_integrateurs_donne_un_second_ordre() -> None:
    """Article « Mason » : `1/(p^2 + a p + b)` par les parcours et les boucles."""
    mason = regle_de_mason([1 / P**2], [-A / P, -B / P**2])
    assert egales(mason, 1 / (P**2 + A * P + B))
    assert egales(transmittance_du_graphe(retours=[-A, -B], directs=[0, 1]), mason)
    assert coefficients_normalises(mason)[0] == (A, B)
