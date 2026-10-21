"""`int_R R(x) e^(i a x) dx` pour une fraction rationnelle `R = p / q`.

On ferme la droite par un demi-cercle (ou un rectangle) du cote ou
`e^(i a z)` est bornee : en haut si `a > 0`, en bas si `a < 0`. L'integrale
sur le bord ajoute s'eteint quand `deg q >= deg p + 1` -- cas (a), et
cas (b) par le rectangle. Les zeros reels simples de `q` donnent une
valeur principale, et chacun compte pour un demi-residu (cas (c)) :

  `a > 0` : `2 pi i somme_(Im > 0) Res + i pi somme_(Im = 0) Res`
  `a < 0` : `-2 pi i somme_(Im < 0) Res - i pi somme_(Im = 0) Res`

Source : Mathematiques pour l'ingenieur III, planches du calcul
d'integrales, planches 19 a 22.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.analyse_complexe.residus import residu


def _poles(denominateur: sp.Expr, variable: sp.Symbol) -> dict[sp.Expr, int]:
    """Les zeros de `q` et leur multiplicite ; leve s'ils ne s'expriment pas tous."""
    racines = sp.roots(sp.Poly(denominateur, variable))
    if sum(racines.values()) != sp.degree(denominateur, variable):
        msg = f"zeros de {denominateur} hors de portee de sympy"
        raise ValueError(msg)
    return dict(racines)


def integrale_oscillante(
    rationnelle: sp.Expr, variable: sp.Symbol, frequence: object
) -> sp.Expr:
    """La valeur (principale, s'il y a des poles reels) de `int R(x) e^(i a x) dx`."""
    a = sp.sympify(frequence)
    if not (a.is_real and a.is_nonzero):
        msg = (
            f"frequence {a} : non nulle et reelle, sinon voir integrale_par_les_residus"
        )
        raise ValueError(msg)
    numerateur, denominateur = sp.fraction(sp.cancel(rationnelle))
    if sp.degree(denominateur, variable) < sp.degree(numerateur, variable) + 1:
        msg = (
            "degre de q inferieur a deg p + 1 : l'integrale sur le bord ne s'eteint pas"
        )
        raise ValueError(msg)
    f = rationnelle * sp.exp(sp.I * a * variable)
    sens = 1 if a > 0 else -1
    total = sp.Integer(0)
    for pole, multiplicite in _poles(denominateur, variable).items():
        partie_imaginaire = sp.im(pole)
        if partie_imaginaire == 0:
            if multiplicite > 1:
                msg = f"pole reel multiple en {pole} : pas de valeur principale"
                raise ValueError(msg)
            total += sp.I * sp.pi * residu(f, variable, pole)
        elif sp.sign(partie_imaginaire) == sens:
            total += 2 * sp.pi * sp.I * residu(f, variable, pole)
    return sp.simplify(sens * total)
