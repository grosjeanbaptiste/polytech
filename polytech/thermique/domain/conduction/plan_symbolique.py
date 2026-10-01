"""La meme couche plane, resolue en symboles et non en chiffres.

Pourquoi les deux. Un corrige pas a pas doit montrer **d'ou vient** la
formule avant de l'appliquer : le lecteur qui ne verrait que
`C_1 = 189,72` n'apprend rien de transposable. La version symbolique donne
l'expression generale, la version chiffree l'applique, et un test verifie
que la substitution des donnees dans l'une redonne l'autre -- ce qui ferme
la boucle au lieu de la laisser ouverte sur une confiance.

Une singularite apparente, et elle vaut d'etre dite. La solution generale
divise par `K_b` : elle n'est donc pas definie pour une paroi adiabatique.
Ce n'est pas la physique qui est singuliere, c'est la formule -- quand
`K_b = 0` la condition basse s'ecrit directement `lambda C_1 = 0`, donc
`C_1 = 0`, sans aucune division. La limite de la formule generale vaut
bien zero, et un test la calcule.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.thermique.domain.conduction.symboles import (
    ABSCISSE,
    CONSTANTES,
    TEMPERATURE,
)

CONDUCTIVITE: Final = sp.Symbol("lambda", positive=True)
GENERATION: Final = sp.Symbol("q", positive=True)
EPAISSEUR: Final = sp.Symbol("e", positive=True)
ECHANGE_BAS: Final = sp.Symbol("K_b", nonnegative=True)
ECHANGE_HAUT: Final = sp.Symbol("K_h", positive=True)
TEMPERATURE_BAS: Final = sp.Symbol("t_s", real=True)
TEMPERATURE_HAUT: Final = sp.Symbol("t_a", real=True)
FLUX_ABSORBE: Final = sp.Symbol("varphi", nonnegative=True)


def loi_locale() -> sp.Eq:
    """Fourier-Kirchhoff stationnaire, unidimensionnel, avec source."""
    derivee = sp.Derivative(TEMPERATURE(ABSCISSE), ABSCISSE, 2)
    return sp.Eq(CONDUCTIVITE * derivee + GENERATION, 0)


def profil_general() -> sp.Expr:
    """t(x) = -q x^2 / (2 lambda) + C_1 x + C_2."""
    premiere, seconde = CONSTANTES
    return (
        -GENERATION * ABSCISSE**2 / (2 * CONDUCTIVITE) + premiere * ABSCISSE + seconde
    )


def gradient_general() -> sp.Expr:
    """dt/dx = -q x / lambda + C_1."""
    return sp.diff(profil_general(), ABSCISSE)


def condition_au_bas() -> sp.Eq:
    """Robin en x = 0 : lambda dt/dx(0) = K_b (t(0) - t_s).

    Le flux descendant est emporte par le sol.
    """
    descendant = CONDUCTIVITE * gradient_general().subs(ABSCISSE, 0)
    ecart = profil_general().subs(ABSCISSE, 0) - TEMPERATURE_BAS
    return sp.Eq(descendant, ECHANGE_BAS * ecart)


def condition_au_sommet() -> sp.Eq:
    """-lambda dt/dx(e) + varphi = K_h (t(e) - t_a) : bilan de la surface."""
    conduit = -CONDUCTIVITE * gradient_general().subs(ABSCISSE, EPAISSEUR)
    ecart = profil_general().subs(ABSCISSE, EPAISSEUR) - TEMPERATURE_HAUT
    return sp.Eq(conduit + FLUX_ABSORBE, ECHANGE_HAUT * ecart)


def constantes() -> dict[sp.Symbol, sp.Expr]:
    """Les deux constantes en fonction des seules donnees, K_b non nul."""
    systeme = [condition_au_bas(), condition_au_sommet()]
    resolu = sp.solve(systeme, CONSTANTES, dict=True)
    if len(resolu) != 1:
        msg = f"le systeme symbolique n'a pas de solution unique : {resolu}"
        raise ValueError(msg)
    premiere, seconde = CONSTANTES
    return {
        premiere: sp.simplify(resolu[0][premiere]),
        seconde: sp.simplify(resolu[0][seconde]),
    }


def abscisse_du_sommet() -> sp.Expr:
    """x* = lambda C_1 / q, ou le gradient s'annule."""
    premiere, _ = CONSTANTES
    return CONDUCTIVITE * premiere / GENERATION
