"""Reconnaitre une EDO du premier ordre, puis lui appliquer son geste.

Dans cet ordre, parce que c'est l'ordre du cours : ses planches enumerent
quatre types -- variables separees, homogene de degre 0, exacte, lineaire
-- et chaque geste suit du type reconnu. Appeler `dsolve` sauterait
precisement l'etape qu'un corrige pas a pas doit montrer.

La garde qui refuse de deviner est la plus importante du module. Un type
mal devine ferait appliquer le mauvais geste, et le corrige afficherait
une separation de variables sur une equation qui ne s'y prete pas -- sans
que rien ne tombe.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

VARIABLE = sp.Symbol("x", real=True)
INCONNUE = sp.Symbol("y", real=True)
FONCTION = sp.Function("y")

SONDES = (1, 2, 3, -1, sp.Rational(1, 2))
"""Les valeurs de `y` ou lire `f(x)` : il en faut une qui n'annule pas `g`."""


class Type(Enum):
    """Les types que le cours enumere.

    `EquationDuPremierOrdre.type` reconnait les trois premiers ;
    `reconnaissance.type_du_cours` les trois suivants, qui demandent la
    fraction ecrite (syllabus de mathematiques appliquees, 1.3.5 a 1.3.8).
    """

    VARIABLES_SEPAREES = "a variables separees"
    HOMOGENE = "homogene de degre 0"
    EXACTE = "exacte"
    LINEAIRE = "lineaire"
    FACTEUR_INTEGRANT = "exacte apres un facteur integrant"
    BERNOULLI = "de Bernoulli"


@dataclass(frozen=True, slots=True)
class EquationDuPremierOrdre:
    """`y' = f(x, y)`, donnee par son second membre."""

    second_membre: sp.Expr

    def _separation(self) -> tuple[sp.Expr, sp.Expr] | None:
        """`(1/g(y), f(x))` si le second membre se factorise, sinon rien.

        `f(x)` se lit en figeant `y` ; encore faut-il le figer ailleurs
        qu'en un zero de `g`, ou le second membre s'annule pour tout `x`.
        Plusieurs valeurs sont donc essayees : sonder en `y = 1` seul
        ecartait `y' = x (y^2 - 1) / y`, qui est pourtant separable.
        """
        for sonde in SONDES:
            en_x = sp.simplify(self.second_membre.subs(INCONNUE, sonde))
            if en_x == 0 or en_x.has(sp.zoo, sp.nan):
                continue
            en_y = sp.simplify(self.second_membre / en_x)
            if en_y.has(VARIABLE) or en_x.has(INCONNUE):
                return None
            return sp.simplify(1 / en_y), en_x
        return None

    def _est_lineaire(self) -> bool:
        """Vrai si le second membre est affine en `y`."""
        return bool(sp.degree(sp.Poly(self.second_membre, INCONNUE), INCONNUE) == 1)

    def _est_homogene(self) -> bool:
        """Vrai si `f(tx, ty) = f(x, y)` : degre zero d'homogeneite."""
        facteur = sp.Symbol("tau", positive=True)
        dilate = self.second_membre.subs(
            {VARIABLE: facteur * VARIABLE, INCONNUE: facteur * INCONNUE},
        )
        return bool(sp.simplify(dilate - self.second_membre) == 0)

    def type(self) -> Type:
        """Le type du cours auquel cette equation appartient ; leve sinon."""
        if self._separation() is not None:
            return Type.VARIABLES_SEPAREES
        if self._est_homogene():
            return Type.HOMOGENE
        try:
            lineaire = self._est_lineaire()
        except sp.PolynomialError:
            lineaire = False
        if lineaire:
            return Type.LINEAIRE
        msg = f"type non reconnu parmi ceux du cours : {self.second_membre}"
        raise ValueError(msg)

    def membres_separes(self) -> tuple[sp.Expr, sp.Expr]:
        """`dy/g(y) = f(x) dx` : les deux integrandes, a montrer."""
        separation = self._separation()
        if separation is None:
            msg = f"equation non separable : {self.second_membre}"
            raise ValueError(msg)
        return separation

    def solution_verifiee(self) -> bool:
        """Vrai si la solution de sympy satisfait vraiment l'equation."""
        equation = sp.Eq(
            sp.Derivative(FONCTION(VARIABLE), VARIABLE),
            self.second_membre.subs(INCONNUE, FONCTION(VARIABLE)),
        )
        solution = sp.dsolve(equation, FONCTION(VARIABLE))
        reste = equation.lhs - equation.rhs
        return bool(
            sp.simplify(reste.subs(FONCTION(VARIABLE), solution.rhs).doit()) == 0
        )
