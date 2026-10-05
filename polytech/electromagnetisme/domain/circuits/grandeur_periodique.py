"""Une grandeur periodique : ses harmoniques, ses valeurs, ses facteurs.

Le syllabus d'electricite generale (17.1, 17.2 et 17.6). Une grandeur de
periode `T` est alternative si sa valeur moyenne (17.1.2) est nulle,
pulsatoire sinon. Son developpement de Fourier s'ecrit en electricite
`x = X_0 + somme X_n racine(2) cos(n omega t + theta_n)` (17.2.1) : `X_0`
est la composante continue, le terme de rang un le fondamental, les
autres les harmoniques. Le calcul des coefficients est celui du contexte
des mathematiques ; seule la traduction en valeur efficace et en phase
est faite ici.

Si deux alternances consecutives ne different que par le signe,
`x(t + T / 2) = -x(t)`, la grandeur est dite symetrique et n'a pas
d'harmonique pair.

La valeur redressee est la moyenne de la valeur absolue (17.6.1) ; le
facteur de crete rapporte la valeur de crete a la valeur efficace
(17.6.2), le facteur de forme la valeur efficace a la valeur redressee
(17.6.3), le taux d'harmoniques le residu a la grandeur (17.6.5).
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.maths.domain.fourier.series import SerieDeFourier
from polytech.maths.domain.symboles import VARIABLE

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def _serie(grandeur: sp.Expr, periode: Grandeur) -> SerieDeFourier:
    motif = sp.sympify(grandeur).subs(TEMPS, VARIABLE)
    return SerieDeFourier(motif, sp.Integer(0), sp.sympify(periode))


def valeur_moyenne(grandeur: sp.Expr, periode: Grandeur) -> sp.Expr:
    """`X_0 = (1 / T) integrale de x dt` (17.1.2) : la composante continue."""
    return _serie(grandeur, periode).coefficient(0)


def est_alternative(grandeur: sp.Expr, periode: Grandeur) -> bool:
    """Vrai si la valeur moyenne est nulle."""
    return bool(valeur_moyenne(grandeur, periode) == 0)


def est_symetrique(grandeur: sp.Expr, periode: Grandeur) -> bool:
    """Vrai si `x(t + T / 2) = -x(t)`."""
    decalee = sp.sympify(grandeur).subs(TEMPS, TEMPS + sp.sympify(periode) / 2)
    return bool(sp.simplify(sp.expand_trig(decalee + grandeur)) == 0)


def harmonique(
    grandeur: sp.Expr, periode: Grandeur, rang: int
) -> tuple[sp.Expr, sp.Expr]:
    """`(X_n, theta_n)` de (17.2.4) et (17.2.5) ; phase nulle si `X_n` l'est."""
    serie = _serie(grandeur, periode)
    amplitude = sp.simplify(serie.amplitude(rang))
    if amplitude == 0:
        return sp.Integer(0), sp.Integer(0)
    return amplitude / sp.sqrt(2), sp.simplify(serie.phase(rang))


def valeur_redressee(grandeur: sp.Expr, periode: Grandeur) -> sp.Expr:
    """`X_r = (1 / T) integrale de |x| dt` (17.6.1)."""
    return valeur_moyenne(sp.Abs(grandeur), periode)


def _non_nulle(valeur: Grandeur, nom: str) -> Grandeur:
    if valeur == 0:
        msg = f"valeur {nom} nulle : le rapport n'est pas defini"
        raise ValueError(msg)
    return valeur


def facteur_de_crete(crete: Grandeur, efficace: Grandeur) -> sp.Expr:
    """`FC = crete / X` (17.6.2)."""
    return sp.simplify(crete / _non_nulle(efficace, "efficace"))


def facteur_de_forme(efficace: Grandeur, redressee: Grandeur) -> sp.Expr:
    """`FF = X / X_r` (17.6.3)."""
    return sp.simplify(efficace / _non_nulle(redressee, "redressee"))


def taux_d_harmoniques(efficace: Grandeur, fondamental: Grandeur) -> sp.Expr:
    """`h = racine(X^2 - X_1^2) / X` (17.6.5)."""
    residu = _non_nulle(efficace, "efficace") ** 2 - fondamental**2
    if (sp.sympify(residu) < 0) == sp.true:
        msg = f"fondamental superieur a la grandeur : {fondamental} contre {efficace}"
        raise ValueError(msg)
    return sp.simplify(sp.sqrt(residu) / efficace)
