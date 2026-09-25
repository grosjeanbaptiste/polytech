"""Cours III.3 -- fonction de courant, potentiel, trajectoires.

L'existence de ``psi`` equivaut a ``div V = 0`` ; celle de ``phi`` a
``omega_z = 0``. Quand la condition n'est pas remplie, ces fonctions
n'existent pas : le module *leve* plutot que de renvoyer un ``None`` qu'un
appelant pourrait oublier de tester.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import T, X, Y


class FonctionDeCourantInexistante(Exception):
    """L'ecoulement n'est pas a divergence nulle : ``psi`` n'existe pas."""


class PotentielInexistant(Exception):
    """L'ecoulement est rotationnel : ``phi`` n'existe pas."""


def _primitive_par_recollement(
    partielle_premiere: sp.Expr,
    variable_premiere: sp.Symbol,
    partielle_seconde: sp.Expr,
    variable_seconde: sp.Symbol,
) -> sp.Expr:
    """Reconstruit une fonction a partir de ses deux derivees partielles."""
    partielle = sp.integrate(partielle_premiere, variable_premiere)
    reste = sp.simplify(partielle_seconde - sp.diff(partielle, variable_seconde))
    return sp.simplify(partielle + sp.integrate(reste, variable_seconde))


def fonction_de_courant(u: sp.Expr, v: sp.Expr) -> sp.Expr:
    """``psi`` telle que ``u = dpsi/dy`` et ``v = -dpsi/dx``."""
    if sp.simplify(sp.diff(u, X) + sp.diff(v, Y)) != 0:
        msg = "divergence non nulle : la fonction de courant n'existe pas"
        raise FonctionDeCourantInexistante(msg)
    return _primitive_par_recollement(u, Y, -v, X)


def potentiel_des_vitesses(u: sp.Expr, v: sp.Expr) -> sp.Expr:
    """``phi`` telle que ``V = grad phi``."""
    if sp.simplify(sp.diff(v, X) - sp.diff(u, Y)) != 0:
        msg = "ecoulement rotationnel : le potentiel des vitesses n'existe pas"
        raise PotentielInexistant(msg)
    return _primitive_par_recollement(u, X, v, Y)


def definition_ligne_de_courant() -> sp.Eq:
    """Enonce du cours III.3 : ``dx/u = dy/v`` a instant fige."""
    u_gen = sp.Function("u")(X, Y, T)
    v_gen = sp.Function("v")(X, Y, T)
    return sp.Eq(sp.Symbol("dx") / u_gen, sp.Symbol("dy") / v_gen, evaluate=False)


def equation_lignes_de_courant(u: sp.Expr, v: sp.Expr) -> sp.Eq:
    """Equation implicite des lignes de courant, ``invariant = C``.

    En ecoulement a divergence nulle, l'invariant est la fonction de courant
    elle-meme : ``psi = cte`` (cours III.3).
    """
    # Un invariant est defini a un facteur pres : on retire le coefficient
    # numerique pour obtenir la forme reduite qu'attend la lecture (xy = C).
    _, reduit = fonction_de_courant(u, v).as_coeff_Mul()
    return sp.Eq(reduit, sp.Symbol("C"))


def trajectoire(
    u: sp.Expr,
    v: sp.Expr,
    x0: sp.Expr | float,
    y0: sp.Expr | float,
) -> tuple[sp.Expr, sp.Expr]:
    """Position d'une particule partie de ``(x0, y0)`` a ``t = 0``.

    Integre ``dx/dt = u`` et ``dy/dt = v`` (cours III.3). En stationnaire, la
    courbe obtenue coincide avec la ligne de courant.
    """
    fx, fy = sp.Function("fx"), sp.Function("fy")
    substitution = {X: fx(T), Y: fy(T)}
    solution = sp.dsolve(
        [
            sp.Eq(fx(T).diff(T), u.subs(substitution)),
            sp.Eq(fy(T).diff(T), v.subs(substitution)),
        ],
        [fx(T), fy(T)],
        ics={fx(0): x0, fy(0): y0},
    )
    return (sp.simplify(solution[0].rhs), sp.simplify(solution[1].rhs))
