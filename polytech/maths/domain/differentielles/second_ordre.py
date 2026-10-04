"""L'equation lineaire du second ordre a coefficients constants.

`a y'' + b y' + c y = f(x)`. L'equation homogene se resout par
l'**equation caracteristique** `a r^2 + b r + c = 0` :

* deux racines reelles distinctes : `c1 e^(r1 x) + c2 e^(r2 x)` ;
* une racine double : `(c1 + c2 x) e^(r x)` ;
* deux racines conjuguees `alpha +- j beta` :
  `e^(alpha x) (c1 cos(beta x) + c2 sin(beta x))`.

La solution generale ajoute une solution particuliere de l'equation
complete. Pour un systeme non amorti (`b = 0`), de pulsation propre
`omega_0 = racine(c / a)`, une sollicitation sinusoidale a cette meme
pulsation donne une reponse dont l'amplitude croit avec le temps : c'est
la **resonance** (exercice I.5, le systeme masse-ressort).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.maths.domain.differentielles.premier_ordre import FONCTION, VARIABLE

C1, C2 = sp.symbols("c_1 c_2")
"""Les deux constantes de la solution homogene."""

_RACINE: Final = sp.Symbol("r")


@dataclass(frozen=True, slots=True)
class EquationDuSecondOrdre:
    """Les coefficients `a`, `b`, `c` de `a y'' + b y' + c y`."""

    a: sp.Expr
    b: sp.Expr
    c: sp.Expr

    def __post_init__(self) -> None:
        """Rend les coefficients exacts et refuse `a = 0`."""
        for nom in ("a", "b", "c"):
            object.__setattr__(self, nom, sp.sympify(getattr(self, nom)))
        if self.a == 0:
            msg = "coefficient de y'' nul : l'equation n'est pas du second ordre"
            raise ValueError(msg)

    def racines_caracteristiques(self) -> tuple[sp.Expr, ...]:
        """Les racines de `a r^2 + b r + c`, une racine double comptee deux fois."""
        polynome = sp.Poly(self.a * _RACINE**2 + self.b * _RACINE + self.c, _RACINE)
        return tuple(r for r, n in sp.roots(polynome).items() for _ in range(n))

    def pulsation_propre(self) -> sp.Expr | None:
        """`omega_0`, si les racines sont imaginaires pures ; sinon `None`."""
        rapport = self.c / self.a
        if self.b != 0 or not rapport.is_positive:
            return None
        return sp.sqrt(rapport)

    def est_en_resonance(self, pulsation: object) -> bool:
        """Vrai si la sollicitation a la pulsation propre du systeme."""
        return bool(self.pulsation_propre() == sp.sympify(pulsation))

    def solution_homogene(self) -> sp.Expr:
        """La solution generale de `a y'' + b y' + c y = 0`."""
        premiere, seconde = self.racines_caracteristiques()
        if premiere == seconde:
            return (C1 + C2 * VARIABLE) * sp.exp(premiere * VARIABLE)
        if premiere.is_real:
            return C1 * sp.exp(premiere * VARIABLE) + C2 * sp.exp(seconde * VARIABLE)
        reelle, imaginaire = sp.re(premiere), sp.Abs(sp.im(premiere))
        oscillation = C1 * sp.cos(imaginaire * VARIABLE) + C2 * sp.sin(
            imaginaire * VARIABLE
        )
        return sp.exp(reelle * VARIABLE) * oscillation

    def solution_particuliere(self, second_membre: sp.Expr) -> sp.Expr:
        """Une solution de l'equation complete, sans composante homogene libre."""
        inconnue = FONCTION(VARIABLE)
        membre = (
            self.a * inconnue.diff(VARIABLE, 2)
            + self.b * inconnue.diff(VARIABLE)
            + self.c * inconnue
        )
        generale = sp.dsolve(sp.Eq(membre, second_membre), inconnue).rhs
        libres = {s: 0 for s in generale.free_symbols if s.name.startswith("C")}
        return sp.simplify(generale.subs(libres))
