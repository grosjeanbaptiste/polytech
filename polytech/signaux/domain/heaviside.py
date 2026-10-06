"""Le developpement d'Heaviside : inverser une fraction rationnelle.

Syllabus de signaux et systemes, section 2.2.4. L'integrale d'inversion
ne sert guere : les images usuelles sont des fractions `N(p) / D(p)`,
que l'on decompose en fractions simples.

* Pole simple `p_i` : residu `A_i = F(p)(p - p_i)` en `p_i`, composante
  `A_i exp(p_i t)` (2.35, 2.36).
* Pole double : `A_2 = F(p)(p - p_i)^2` en `p_i`, `A_1` sa derivee, et
  la composante `(A_1 + A_2 t) exp(p_i t)` (2.38, 2.39).
* Numerateur de degre non inferieur : on divise d'abord, et le quotient
  `q_0 + q_1 p + ...` donne `q_0 delta(t) + q_1 delta'(t) + ...` (2.41).

Le cours s'arrete aux poles doubles, « peu utile a l'ingenieur » au-dela.

Le residu lui-meme vient de l'analyse complexe, ou il est enseigne : c'est
celui d'un quotient `N / D`, que `polytech.maths` calcule.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.analyse_complexe.quotient import residu_d_un_quotient
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T

DOUBLE = 2


def _fraction(image: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    numerateur, denominateur = sp.fraction(sp.cancel(image))
    return (numerateur, denominateur)


def _multiplicite(image: sp.Expr, pole: sp.Expr) -> int:
    racines = sp.roots(sp.Poly(_fraction(image)[1], P))
    egales = (m for racine, m in racines.items() if sp.simplify(racine - pole) == 0)
    return int(next(egales, 0))


def _en(expression: sp.Expr, pole: sp.Expr, ordre: int = 0) -> sp.Expr:
    """La derivee d'ordre donne, evaluee au pole."""
    return sp.diff(expression, P, ordre).subs(P, pole)


def residu(image: sp.Expr, pole: sp.Expr) -> sp.Expr:
    """`A_i = F(p)(p - p_i)` en `p = p_i`, soit `N(p_i) / D'(p_i)`."""
    if _multiplicite(image, pole) != 1:
        msg = f"{pole} n'est pas un pole simple de {image}"
        raise ValueError(msg)
    numerateur, denominateur = _fraction(image)
    return residu_d_un_quotient(numerateur, denominateur, P, pole)


def residus_d_un_pole_double(image: sp.Expr, pole: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`(A_1, A_2)` : les coefficients de `1 / (p - p_i)` et de son carre.

    Avec `D = (p - p_i)^2 Q`, `Q(p_i)` est la moitie de la derivee seconde
    de `D` et `Q'(p_i)` le sixieme de sa derivee troisieme : `A_2 = N / Q`,
    et `A_1` en est la derivee.
    """
    if _multiplicite(image, pole) != DOUBLE:
        msg = f"{pole} n'est pas un pole double de {image}"
        raise ValueError(msg)
    numerateur, denominateur = _fraction(image)
    reste = _en(denominateur, pole, 2) / 2
    premier = residu_d_un_quotient(numerateur, denominateur, P, pole)
    return (premier, sp.simplify(_en(numerateur, pole) / reste))


def division(image: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`(Q, R / D)` : le quotient, et une fraction strictement propre."""
    numerateur, denominateur = sp.fraction(sp.cancel(image))
    quotient, reste = sp.div(numerateur, denominateur, P)
    return (sp.expand(quotient), sp.cancel(reste / denominateur))


def _impulsions(quotient: sp.Expr) -> sp.Expr:
    """`q_0 delta(t) + q_1 delta'(t) + ...`."""
    coefficients = sp.Poly(quotient, P).all_coeffs()[::-1]
    termes = (q * sp.DiracDelta(T, rang) for rang, q in enumerate(coefficients))
    return sum(termes, sp.S.Zero)


def original(image: sp.Expr) -> sp.Expr:
    """`f(t)` pour `t >= 0`, impulsions comprises."""
    quotient, propre = division(image)
    impulsions = _impulsions(quotient) if quotient != 0 else sp.S.Zero
    signal = sp.S.Zero
    denominateur = sp.fraction(propre)[1]
    for pole, multiplicite in sp.roots(sp.Poly(denominateur, P)).items():
        if multiplicite == 1:
            signal += residu(propre, pole) * sp.exp(pole * T)
        elif multiplicite == DOUBLE:
            premier, second = residus_d_un_pole_double(propre, pole)
            signal += (premier + second * T) * sp.exp(pole * T)
        else:
            msg = f"pole de multiplicite superieure a deux, hors du cours : {pole}"
            raise ValueError(msg)
    return impulsions + sp.simplify(sp.expand_complex(signal))
