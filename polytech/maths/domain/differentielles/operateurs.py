"""L'algebre des operateurs differentiels lineaires a coefficients constants.

`a_n y^(n) + ... + a_1 y' + a_0 y = x(t)` s'ecrit `K(D) y = x(t)`, ou `D`
est la derivation et `K` un polynome. Ces operateurs s'additionnent et se
composent comme les polynomes s'additionnent et se multiplient : on peut
donc les **factoriser**, et tout se lit sur les racines de l'equation
caracteristique `K(D) = 0`.

Une racine reelle `b` de multiplicite `k` apporte `e^(bt), t e^(bt), ...,
t^(k-1) e^(bt)` ; une paire `alpha +- j omega`, les memes puissances de
`t` devant `e^(alpha t) cos(omega t)` et `e^(alpha t) sin(omega t)`.
L'espace des solutions de `K(D) y = 0` est de dimension `n`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.maths.domain.symboles import TEMPS

D: Final = sp.Symbol("D")
"""La derivation, traitee comme l'indeterminee d'un polynome."""


@dataclass(frozen=True, slots=True)
class OperateurDifferentiel:
    """`K(D)`, donne par son polynome en `D`."""

    polynome: sp.Expr

    def __post_init__(self) -> None:
        """Developpe le polynome, pour que deux ecritures du meme soient egales."""
        developpe = sp.expand(self.polynome)
        if developpe == 0:
            msg = "operateur nul : K(D) y = x(t) ne contraint plus y"
            raise ValueError(msg)
        object.__setattr__(self, "polynome", developpe)

    @property
    def ordre(self) -> int:
        """Le degre de `K`, donc l'ordre de l'equation."""
        return int(sp.degree(self.polynome, D))

    def appliquer(self, fonction: sp.Expr) -> sp.Expr:
        """`K(D) y`, pour une fonction `y(t)`."""
        coefficients = sp.Poly(self.polynome, D).all_coeffs()[::-1]
        return sum(
            (a * sp.diff(fonction, TEMPS, k) for k, a in enumerate(coefficients)),
            sp.Integer(0),
        )

    def composer(self, autre: OperateurDifferentiel) -> OperateurDifferentiel:
        """`A(D)(B(D) y) = (A.B)(D) y` : composer, c'est multiplier."""
        return OperateurDifferentiel(self.polynome * autre.polynome)

    def decaler(self, alpha: object) -> OperateurDifferentiel:
        """`K(D + alpha)`, l'operateur que voit `z` quand `y = e^(alpha t) z`."""
        return OperateurDifferentiel(self.polynome.subs(D, D + sp.sympify(alpha)))

    def en(self, valeur: object) -> sp.Expr:
        """`K(valeur)`, `D` etant pris pour un nombre."""
        return sp.simplify(self.polynome.subs(D, sp.sympify(valeur)))

    def discriminant(self) -> sp.Expr:
        """Le discriminant de `K(D)` : son signe dit si les racines sont reelles."""
        return sp.discriminant(self.polynome, D)

    def racines(self) -> dict[sp.Expr, int]:
        """Les racines de l'equation caracteristique, avec leur multiplicite."""
        return dict(sp.roots(self.polynome, D))

    def base_des_solutions(self) -> tuple[sp.Expr, ...]:
        """Une base reelle de l'espace des solutions de `K(D) y = 0`."""
        base: list[sp.Expr] = []
        for racine, multiplicite in sorted(
            self.racines().items(), key=lambda couple: sp.default_sort_key(couple[0])
        ):
            alpha, omega = racine.as_real_imag()
            if omega < 0:
                continue
            enveloppe = sp.exp(alpha * TEMPS)
            formes = (
                (enveloppe,)
                if omega == 0
                else (
                    enveloppe * sp.cos(omega * TEMPS),
                    enveloppe * sp.sin(omega * TEMPS),
                )
            )
            base.extend(TEMPS**i * f for f in formes for i in range(multiplicite))
        return tuple(base)

    def inverse_tronque(self, degre: int) -> sp.Expr:
        """`1 / K(D)` par puissances croissantes, jusqu'a `D^degre` compris."""
        if self.en(0) == 0:
            msg = "division par puissances croissantes est impossible : K(0) = 0"
            raise ValueError(msg)
        return sp.series(1 / self.polynome, D, 0, degre + 1).removeO()
