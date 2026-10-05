"""Les suites et les series de nombres, et leurs criteres de convergence.

Une suite `u_n` converge vers `u` si `|u_n - u| < eps` des que `n > N(eps)`.
Une serie converge si la suite de ses **sommes partielles** converge -- et
non celle de ses termes : il faut que `u_n` tende vers 0, mais cela ne
suffit pas.

Criteres : la convergence absolue entraine la convergence ; une serie
positive se compare a une serie de reference (Riemann `somme 1/n^alpha`,
qui converge ssi `alpha > 1` ; geometrique `somme x^n`, ssi `|x| < 1`) ou
lui est equivalente ; le quotient `|u_(n+1) / u_n|` et la racine
`|u_n|^(1/n)` concluent selon que leur limite est inferieure ou superieure
a 1, et ne disent rien si elle vaut 1 ; une serie alternee converge des
que `|u_n|` tend vers 0 en decroissant.
"""

from __future__ import annotations

from enum import Enum

import sympy as sp

PATIENCE = 50
"""Le nombre de rangs consecutifs sous `eps` exiges avant de conclure."""


class Verdict(Enum):
    """Ce que disent les criteres du quotient et de la racine."""

    CONVERGE = "limite inferieure a 1 : la serie converge"
    DIVERGE = "limite superieure a 1 : la serie diverge"
    INDECIS = "limite egale a 1 : pas de conclusion"


def somme_partielle(terme: sp.Expr, indice: sp.Symbol, rang: int) -> sp.Expr:
    """`S_n = u_1 + ... + u_n`."""
    return sum((terme.subs(indice, k) for k in range(1, rang + 1)), sp.Integer(0))


def converge(terme: sp.Expr, indice: sp.Symbol) -> bool:
    """Vrai si la serie `somme u_n`, depuis `n = 1`, converge."""
    return bool(sp.Sum(terme, (indice, 1, sp.oo)).is_convergent())


def converge_absolument(terme: sp.Expr, indice: sp.Symbol) -> bool:
    """Vrai si la serie des `|u_n|` converge."""
    return converge(sp.Abs(terme), indice)


def _verdict(limite: sp.Expr) -> Verdict:
    if limite == 1:
        return Verdict.INDECIS
    return Verdict.CONVERGE if limite < 1 else Verdict.DIVERGE


def critere_du_quotient(terme: sp.Expr, indice: sp.Symbol) -> tuple[sp.Expr, Verdict]:
    """`A = lim |u_(n+1) / u_n|`, et ce qu'il permet de conclure."""
    suivant = terme.subs(indice, indice + 1)
    limite = sp.limit(sp.Abs(sp.simplify(suivant / terme)), indice, sp.oo)
    return limite, _verdict(limite)


def critere_de_la_racine(terme: sp.Expr, indice: sp.Symbol) -> tuple[sp.Expr, Verdict]:
    """`A = lim |u_n|^(1/n)`, et ce qu'il permet de conclure."""
    limite = sp.limit(sp.Abs(terme) ** (1 / indice), indice, sp.oo)
    return limite, _verdict(limite)


def sont_equivalentes(premiere: sp.Expr, seconde: sp.Expr, indice: sp.Symbol) -> bool:
    """Vrai si `u_n / v_n` a une limite finie et non nulle : meme nature."""
    limite = sp.limit(premiere / seconde, indice, sp.oo)
    return bool(limite.is_finite and limite != 0)


def est_alternee_convergente(module: sp.Expr, indice: sp.Symbol) -> bool:
    """Vrai si `a_n` tend vers 0 en decroissant : `somme (-1)^n a_n` converge alors."""
    continue_ = sp.Symbol("x", positive=True)
    prolongee = sp.sympify(module).subs(indice, continue_)
    decroit = sp.is_decreasing(prolongee, sp.Interval(1, sp.oo), continue_)
    return bool(decroit and sp.limit(module, indice, sp.oo) == 0)


def somme_de_la_serie_geometrique(raison: object) -> sp.Expr:
    """`somme_(n >= 0) x^n = 1 / (1 - x)` ; leve si `|x| >= 1`."""
    x = sp.sympify(raison)
    if not sp.Abs(x) < 1:
        msg = f"raison de module superieur ou egal a 1 : la serie de {x}^n diverge"
        raise ValueError(msg)
    return 1 / (1 - x)


def rang_suffisant(
    terme: sp.Expr, indice: sp.Symbol, *, limite: object, epsilon: object
) -> int:
    """Le plus petit `N` tel que `|u_n - u| < eps` pour `n > N`.

    Cherche, et ne prouve pas : la limite est d'abord controlee, puis le rang
    est le dernier ou l'ecart atteint `eps`, suivi de `PATIENCE` rangs
    consecutifs sous `eps`. Sur pour un ecart qui finit par decroitre.
    """
    cible = sp.sympify(limite)
    if sp.limit(terme, indice, sp.oo) != cible:
        msg = f"aucun rang ne convient : {terme} ne tend pas vers {cible}"
        raise ValueError(msg)
    dernier, n = 0, 0
    while n - dernier < PATIENCE:
        n += 1
        if sp.Abs(terme.subs(indice, n) - cible) >= epsilon:
            dernier = n
    return dernier
