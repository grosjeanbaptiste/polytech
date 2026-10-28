"""Les familles de courbes, et l'equation differentielle qui les caracterise.

`Phi(x, y, C) = 0` definit une courbe par valeur de `C`. En derivant par
rapport a `x` puis en **eliminant** `C`, on obtient une equation
`F(x, y, y') = 0` que toutes les courbes de la famille verifient. De meme
une famille a `n` parametres est la solution generale d'une equation
d'ordre `n`.

En un point, la trajectoire **orthogonale** a pour pente `-1/y'` ; la
trajectoire **isogonale**, qui coupe sous l'angle `alpha`,
`(y' + tan alpha) / (1 - y' tan alpha)`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.differentielles.premier_ordre import INCONNUE, VARIABLE


def pente_de_la_famille(famille: sp.Expr, parametre: sp.Symbol) -> sp.Expr:
    """`y' = f(x, y)`, le parametre elimine entre `Phi = 0` et sa derivee."""
    pente = sp.idiff(famille, INCONNUE, VARIABLE)
    (valeur,) = sp.solve(sp.Eq(famille, 0), parametre)
    return sp.simplify(pente.subs(parametre, valeur))


def pentes_de_la_famille(famille: sp.Expr, parametre: sp.Symbol) -> tuple[sp.Expr, ...]:
    """Toutes les pentes `y'` de la famille, une par racine du parametre.

    Quand `Phi = 0` donne plusieurs valeurs du parametre (`K = +- (racine y -
    racine x)` pour `y = (racine x + K)^2`), chacune a sa pente : aucune n'est
    ecartee en silence, contrairement a `pente_de_la_famille` qui exige une
    racine unique.
    """
    pente = sp.idiff(famille, INCONNUE, VARIABLE)
    valeurs = sp.solve(sp.Eq(famille, 0), parametre)
    if not valeurs:
        msg = f"le parametre {parametre} ne se tire pas de {famille} = 0"
        raise ValueError(msg)
    return tuple(sp.simplify(pente.subs(parametre, valeur)) for valeur in valeurs)


def valeurs_de_la_constante(
    relation: sp.Expr, fonction: sp.Expr, constante: sp.Symbol
) -> tuple[sp.Expr, ...]:
    """Les valeurs de `constante` annulant `relation(x, y, c)` sur `y = fonction`.

    Une racine qui depend de `x` n'est pas une constante : elle est ecartee.
    C'est ce qui relie la constante d'une integrale complete a celle de la
    famille de solutions (`c = K^2` pour `y = (racine x + K)^2`).
    """
    residu = sp.expand(relation.subs(INCONNUE, fonction))
    valeurs = sp.solve(sp.Eq(residu, 0), constante)
    constantes = (sp.simplify(v) for v in valeurs)
    return tuple(v for v in constantes if not v.has(VARIABLE))


def pente_orthogonale(pente: object) -> sp.Expr:
    """`-1 / y'`."""
    return -1 / sp.sympify(pente)


def pente_isogonale(pente: object, angle: object) -> sp.Expr:
    """`(y' + tan alpha) / (1 - y' tan alpha)`."""
    tangente = sp.tan(sp.sympify(angle))
    return (sp.sympify(pente) + tangente) / (1 - sp.sympify(pente) * tangente)


def equation_d_ordre_n(famille: sp.Expr, parametres: tuple[sp.Symbol, ...]) -> sp.Eq:
    """L'equation d'ordre `n` de solution generale `y = f(x, C_1, ..., C_n)`."""
    ordre = len(parametres)
    y = sp.Function("y")(VARIABLE)
    derivees = [sp.diff(famille, VARIABLE, k) for k in range(ordre + 1)]
    relations = [sp.Eq(y.diff(VARIABLE, k), derivees[k]) for k in range(ordre)]
    (valeurs,) = sp.solve(relations, parametres, dict=True)
    return sp.Eq(y.diff(VARIABLE, ordre), sp.simplify(derivees[ordre].subs(valeurs)))
