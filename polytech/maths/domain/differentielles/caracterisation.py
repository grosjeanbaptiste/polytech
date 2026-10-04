"""Ce qu'est une equation differentielle ordinaire, et ce qui la caracterise.

Une EDO lie une variable independante `x`, une fonction inconnue `y` et
ses derivees : `F(x, y, y', ..., y^(n)) = 0`. Trois caracteres (planche 9) :

* son **ordre** `n`, celui de la plus haute derivee ;
* sa **linearite** : `y` et ses derivees n'y figurent qu'au premier
  degre, sans produit entre elles -- les coefficients, eux, peuvent
  dependre de `x` ;
* pour une equation lineaire, des coefficients **constants** ou
  **variables**.

Sa forme **normale** isole la plus haute derivee :
`y^(n) = f[x, y, ..., y^(n-1)]`. Une **solution** est une fonction qui,
reportee dans l'equation, la verifie pour tout `x` d'un intervalle.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.differentielles.premier_ordre import FONCTION, VARIABLE


@dataclass(frozen=True, slots=True)
class Caracterisation:
    """L'ordre, la linearite, et la constance des coefficients."""

    ordre: int
    lineaire: bool
    coefficients_constants: bool


def _reste(equation: sp.Eq | sp.Expr) -> sp.Expr:
    """`F`, l'equation ecrite `F = 0`."""
    return equation.lhs - equation.rhs if isinstance(equation, sp.Eq) else equation


def _ordre(reste: sp.Expr) -> int:
    ordres = [d.derivative_count for d in reste.atoms(sp.Derivative)]
    if not ordres:
        msg = f"aucune derivee dans l'equation : {reste} = 0 n'est pas differentielle"
        raise ValueError(msg)
    return int(max(ordres))


def caracteriser(equation: sp.Eq | sp.Expr) -> Caracterisation:
    """L'ordre, la linearite et la nature des coefficients d'une EDO en `y(x)`."""
    reste = _reste(equation)
    ordre = _ordre(reste)
    inconnue = FONCTION(VARIABLE)
    muettes = sp.symbols(f"Y0:{ordre + 1}")
    remplacement = {inconnue.diff(VARIABLE, k): muettes[k] for k in range(ordre, 0, -1)}
    algebrique = reste.subs(remplacement).subs(inconnue, muettes[0])
    try:
        polynome = sp.Poly(algebrique, *muettes)
    except sp.PolynomialError:
        return Caracterisation(ordre, lineaire=False, coefficients_constants=False)
    lineaire = polynome.total_degree() == 1
    coefficients = [c for monome, c in polynome.terms() if sum(monome) == 1]
    constants = lineaire and not any(c.has(VARIABLE) for c in coefficients)
    return Caracterisation(ordre, lineaire, coefficients_constants=constants)


def est_solution(equation: sp.Eq | sp.Expr, candidate: sp.Expr) -> bool:
    """Vrai si la fonction, reportee dans l'equation, la verifie identiquement."""
    reporte = _reste(equation).subs(FONCTION(VARIABLE), candidate).doit()
    return bool(sp.simplify(reporte) == 0)


def forme_normale(equation: sp.Eq | sp.Expr) -> sp.Expr:
    """`f` tel que `y^(n) = f[x, y, ..., y^(n-1)]`."""
    reste = _reste(equation)
    plus_haute = FONCTION(VARIABLE).diff(VARIABLE, _ordre(reste))
    solutions = sp.solve(reste, plus_haute)
    if len(solutions) != 1:
        msg = f"la plus haute derivee ne s'isole pas de facon unique : {solutions}"
        raise ValueError(msg)
    return solutions[0]
