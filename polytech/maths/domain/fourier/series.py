"""La serie de Fourier d'une fonction periodique.

`f` de periode `T`, de pulsation `w = 2 pi / T`, se projette sur la
famille orthonormee `e^(j n w x)` :

    c_n = (1/T) int_a^(a+T) f(x) e^(-j n w x) dx.

Pour `f` reelle, `c_(-n) = conj(c_n)` et, avec `c_n = (a_n - j b_n) / 2`,

    f ~ c_0 + somme (a_n cos(n w x) + b_n sin(n w x)),
    a_n = (2/T) int f cos(n w x),     b_n = (2/T) int f sin(n w x).

La famille etant complete, `||f||^2 = c_0^2 + somme A_n^2 / 2` (Parseval),
ou `A_n = racine(a_n^2 + b_n^2)`. En un point de discontinuite, la serie
vaut la moyenne des limites a gauche et a droite.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE


@dataclass(frozen=True, slots=True)
class SerieDeFourier:
    """Le motif de `f` sur `[debut, debut + periode[`, ecrit en `x`."""

    motif: sp.Expr
    debut: sp.Expr
    periode: sp.Expr

    def __post_init__(self) -> None:
        """Refuse une periode qui n'est pas strictement positive."""
        if not sp.sympify(self.periode).is_positive:
            msg = f"periode strictement positive attendue, {self.periode} recue"
            raise ValueError(msg)

    @property
    def pulsation(self) -> sp.Expr:
        """`w = 2 pi / T`."""
        return 2 * sp.pi / sp.sympify(self.periode)

    def _moyenne(self, fonction: sp.Expr) -> sp.Expr:
        """`(1/T) int` sur une periode, les valeurs absolues ecrites par morceaux."""
        morceaux = sp.piecewise_fold(sp.sympify(fonction).rewrite(sp.Piecewise))
        bornes = (VARIABLE, self.debut, sp.sympify(self.debut) + self.periode)
        return sp.simplify(sp.integrate(morceaux, bornes) / self.periode)

    def coefficient(self, rang: object) -> sp.Expr:
        """`c_n`."""
        noyau = sp.exp(-sp.I * sp.sympify(rang) * self.pulsation * VARIABLE)
        return sp.simplify(sp.expand_complex(self._moyenne(self.motif * noyau)))

    def a(self, rang: object) -> sp.Expr:
        """`a_n = (2/T) int f cos(n w x)`."""
        return 2 * self._moyenne(self.motif * sp.cos(rang * self.pulsation * VARIABLE))

    def b(self, rang: object) -> sp.Expr:
        """`b_n = (2/T) int f sin(n w x)`."""
        return 2 * self._moyenne(self.motif * sp.sin(rang * self.pulsation * VARIABLE))

    def amplitude(self, rang: object) -> sp.Expr:
        """`A_n = racine(a_n^2 + b_n^2)`."""
        return sp.sqrt(self.a(rang) ** 2 + self.b(rang) ** 2)

    def phase(self, rang: object) -> sp.Expr:
        """`phi_n`, tel que `c_n = A_n e^(j phi_n) / 2`."""
        return sp.atan2(-self.b(rang), self.a(rang))

    def somme_partielle(self, ordre: int) -> sp.Expr:
        """`c_0 + somme (a_n cos + b_n sin)` jusqu'a `n = ordre`."""
        angle = self.pulsation * VARIABLE
        harmoniques = (
            self.a(n) * sp.cos(n * angle) + self.b(n) * sp.sin(n * angle)
            for n in range(1, ordre + 1)
        )
        return self.coefficient(0) + sum(harmoniques, sp.Integer(0))

    def norme_au_carre(self) -> sp.Expr:
        """`||f||^2 = (1/T) int f^2`."""
        return self._moyenne(self.motif**2)

    def energie(self, ordre: int) -> sp.Expr:
        """`c_0^2 + somme A_n^2 / 2`, la somme partielle de Parseval."""
        harmoniques = (self.amplitude(n) ** 2 / 2 for n in range(1, ordre + 1))
        return sp.simplify(self.coefficient(0) ** 2 + sum(harmoniques, sp.Integer(0)))

    def valeur_en(self, point: object) -> sp.Expr:
        """La somme de la serie : la moyenne des limites a gauche et a droite."""
        ecart = sp.Mod(sp.sympify(point) - self.debut, self.periode)
        ramene = self.debut + ecart
        gauche = ramene if ecart != 0 else sp.sympify(self.debut) + self.periode
        limites = (
            sp.limit(self.motif, VARIABLE, ramene, "+"),
            sp.limit(self.motif, VARIABLE, gauche, "-"),
        )
        return sp.simplify(sum(limites) / 2)
