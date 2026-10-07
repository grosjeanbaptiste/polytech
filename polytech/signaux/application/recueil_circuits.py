"""Recueil d'exercices, planches 7, 8, 10, 15 et 19 : des circuits a leur reponse.

Les planches 7, 8 et 10 sont celles de la seance « Chap. 4-5 », dont les
calculs sont repris tels quels. La planche 15 shunte une bobine par une
resistance ; la planche 19 fait suivre une bobine de `R` et `C` en
parallele.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.signaux.application import circuits_exs_45, systemes_exs_45
from polytech.signaux.domain.fluence import coefficients_normalises
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T

R, L, C = sp.symbols("R L C", positive=True)
DE_LA_PLANCHE_19: Final = {L: sp.Rational(1, 10**3), C: sp.Rational(1, 10**7)}
"""`L = 1` mH et `C = 0,1` microfarad."""


def transmittances_de_la_planche_7() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Le passe-bas `RC`, le passe-haut `RL`, puis le graphe."""
    return systemes_exs_45.transmittances()


def impulsionnelle_du_graphe(instant: object) -> sp.Expr:
    """Planches 7 (c) et 10 : `h(t)` du graphe a deux integrateurs."""
    return sp.simplify(systemes_exs_45.impulsionnelle_du_graphe().subs(T, instant))


def resistance_limite_du_parallele() -> sp.Expr:
    """Planche 8 : la resistance au-dessus de laquelle `u(t)` oscille."""
    return circuits_exs_45.resistance_limite()


def constante_et_pulsation_du_parallele() -> tuple[sp.Expr, sp.Expr]:
    """Planche 8 : `tau` et la pulsation des oscillations amorties."""
    return circuits_exs_45.constante_et_pulsation()


def transmittance_de_la_planche_15() -> sp.Expr:
    """Un diviseur : `R` en sortie, `R` et `L` en parallele en serie."""
    shunt = sp.cancel(1 / (1 / R + 1 / (P * L)))
    return sp.cancel(R / (R + shunt))


def limites_de_l_indicielle_15() -> tuple[sp.Expr, sp.Expr]:
    """`a(0+)` et `a(infini)` : les limites de `H(p)` a l'infini et en zero."""
    transmittance = transmittance_de_la_planche_15()
    return sp.limit(transmittance, P, sp.oo), transmittance.subs(P, 0)


def limites_physiques_15() -> tuple[sp.Expr, sp.Expr]:
    """Bobine ouverte : deux `R` en diviseur. Bobine en fil : la sortie suit."""
    return sp.cancel(R / (R + R)), sp.cancel(R / (R + 0))


def constante_de_temps_15() -> sp.Expr:
    """`tau`, l'oppose de l'inverse du pole."""
    (pole,) = sp.roots(sp.fraction(transmittance_de_la_planche_15())[1], P)
    return sp.simplify(-1 / pole)


def indicielle_15(instant: object) -> sp.Expr:
    """`a(t)`, l'original de `H(p)/p`."""
    brut = sp.inverse_laplace_transform(transmittance_de_la_planche_15() / P, P, T)
    return sp.simplify(brut.subs(sp.Heaviside(T), 1).subs(T, instant))


def gains_du_graphe_15() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Le gain direct sans integrateur, puis le gain et le retour de l'integrateur.

    Le graphe canonique prend sa branche directe **apres** le sommateur
    d'entree : `H = (d + g p^-1) / (1 - r p^-1)`. Diviser haut et bas par
    le terme de tete du denominateur donne les trois gains.
    """
    haut, bas = (sp.Poly(e, P) for e in sp.fraction(transmittance_de_la_planche_15()))
    (h1, h0), (b1, b0) = haut.all_coeffs(), bas.all_coeffs()
    return sp.simplify(h1 / b1), sp.simplify(h0 / b1), sp.simplify(-b0 / b1)


def transmittance_de_la_planche_19() -> sp.Expr:
    """Un diviseur : `L` en serie, `R` et `C` en parallele en sortie."""
    parallele = sp.cancel(1 / (1 / R + P * C))
    return sp.cancel(parallele / (P * L + parallele))


def limites_19() -> dict[str, tuple[sp.Expr, sp.Expr]]:
    """Les valeurs initiale et finale de `h(t)`, puis de `a(t)`."""
    h = transmittance_de_la_planche_19()
    return {
        "h": (sp.limit(P * h, P, sp.oo), sp.limit(P * h, P, 0)),
        "a": (sp.limit(h, P, sp.oo), h.subs(P, 0)),
    }


def resistance_limite_19() -> sp.Expr:
    """La resistance au-dessus de laquelle `h(t)` oscille, en ohms."""
    denominateur = sp.fraction(transmittance_de_la_planche_19())[1]
    (limite,) = sp.solve(sp.discriminant(denominateur, P), R)
    return limite.subs(DE_LA_PLANCHE_19)


def gains_du_graphe_19() -> tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]]:
    """Les retours puis les gains directs, a la resistance limite."""
    chiffree = transmittance_de_la_planche_19().subs(DE_LA_PLANCHE_19)
    alphas, betas = coefficients_normalises(chiffree.subs(R, resistance_limite_19()))
    return tuple(-alpha for alpha in alphas), betas


def poles_19(resistance: object) -> dict[sp.Expr, int]:
    """Les poles et leur multiplicite, pour cette resistance en ohms."""
    chiffree = transmittance_de_la_planche_19().subs(DE_LA_PLANCHE_19)
    return dict(sp.roots(sp.fraction(sp.cancel(chiffree.subs(R, resistance)))[1], P))
