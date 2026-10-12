"""Les parametres que les sujets d'examen du chapitre X demandent de fixer.

Chaque sujet donne des conditions sur les valeurs propres. Elles deviennent
des **equations** sur les parametres, dont les solutions sont les candidats ;
une **garde** -- un signe, un parametre non nul, une valeur double -- ecarte
ensuite ceux que l'enonce refuse. Les candidats ecartes sont rendus aussi :
le corrige les nomme.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

from polytech.maths.application import recueil_algebre_ch10_enonces as enonces

T = enonces.T
Candidat = dict[sp.Symbol, sp.Expr]


def caracteristique(numero: int) -> sp.Expr:
    """`det(A - tI)`, developpe."""
    matrice = enonces.MATRICES[numero]
    return sp.expand((matrice - T * sp.eye(matrice.rows)).det())


def _racines(numero: int) -> list[sp.Expr]:
    return list(sp.roots(sp.Poly(caracteristique(numero), T), multiple=True))


def _moins_un_double(numero: int) -> list[sp.Expr]:
    p = caracteristique(numero)
    return [p.subs(T, -1), sp.diff(p, T).subs(T, -1)]


def _triple(numero: int) -> list[sp.Expr]:
    """`det(A - tI) = -(t - tr A / 3)^3`, coefficient par coefficient."""
    centre = enonces.MATRICES[numero].trace() / 3
    ecart = sp.expand(caracteristique(numero) + (T - centre) ** 3)
    return list(sp.Poly(ecart, T).all_coeffs())


def _somme_double_du_produit(numero: int) -> list[sp.Expr]:
    racines = _racines(numero)
    return [sp.expand(sum(racines) - 2 * sp.prod(racines))]


def _racines_de_la_cible(numero: int) -> list[sp.Expr]:
    ecart = sp.expand(caracteristique(numero) + enonces.CIBLE_33)
    return list(sp.Poly(ecart, T).all_coeffs())


def _somme_des_carres(numero: int) -> list[sp.Expr]:
    carres = sum(racine**2 for racine in _racines(numero))
    return [sp.expand(carres - enonces.SOMME_DES_CARRES_35)]


def _racine_multiple(numero: int) -> list[sp.Expr]:
    return [sp.discriminant(sp.Poly(caracteristique(numero), T))]


def _valeurs(numero: int, candidat: Candidat) -> list[sp.Expr]:
    return list(enonces.MATRICES[numero].subs(candidat).eigenvals())


def _non_nuls(candidat: Candidat, _: list[sp.Expr]) -> bool:
    """`a, b` dans `R_0` : l'enonce ecarte zero."""
    return all(valeur != 0 for valeur in candidat.values())


Equations = Callable[[int], list[sp.Expr]]
Garde = Callable[[Candidat, list[sp.Expr]], bool]
ORDRE = 3
SUJETS: dict[int, tuple[Equations, Garde]] = {
    20: (_moins_un_double, lambda _, valeurs: max(valeurs) > 0),
    25: (_moins_un_double, lambda _, valeurs: max(valeurs) < 0),
    29: (_triple, _non_nuls),
    31: (_somme_double_du_produit, lambda _, valeurs: len(valeurs) < ORDRE),
    33: (_racines_de_la_cible, lambda _, valeurs: bool(valeurs)),
    35: (_somme_des_carres, lambda _, valeurs: min(valeurs) >= 0),
    37: (_racine_multiple, lambda _, valeurs: min(valeurs) > 0),
    40: (_triple, _non_nuls),
}
"""Numero : les equations que l'enonce pose, et la garde qui trie les candidats."""


def _inconnues(numero: int) -> list[sp.Symbol]:
    if numero not in SUJETS:
        msg = f"exercice {numero} : aucun parametre a fixer"
        raise KeyError(msg)
    equations = sp.Matrix(SUJETS[numero][0](numero))
    return sorted(equations.free_symbols, key=str)


def candidats(numero: int) -> tuple[Candidat, ...]:
    """Les solutions des equations, avant la garde ; un cas particulier est omis."""
    inconnues = _inconnues(numero)
    solutions = sp.solve(SUJETS[numero][0](numero), inconnues, dict=True)
    return tuple(
        c for c in solutions if not any(d.items() < c.items() for d in solutions)
    )


def retenus(numero: int) -> Candidat:
    """Le seul candidat que la garde de l'enonce accepte."""
    garde = SUJETS[numero][1]
    (retenu,) = [c for c in candidats(numero) if garde(c, _valeurs(numero, c))]
    return retenu


def exercice_40() -> tuple[sp.Expr, sp.Expr]:
    """`p(a)`, puis `p` quand `b = a + 1` : ce que `lambda_2 = a` impose."""
    p = caracteristique(40)
    en_a = sp.factor(p.subs(T, enonces.A))
    return en_a, sp.factor(p.subs(enonces.B, enonces.A + 1))
