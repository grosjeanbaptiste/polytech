"""La jonction pn abrupte : l'equation de Poisson a une dimension.

Le syllabus d'electricite generale (exemple 2.15.1) accole un
semi-conducteur de type p (`N_A` accepteurs par unite de volume) a un
semi-conducteur de type n (`N_D` donneurs). La diffusion laisse une zone
de depletion de `-x_p` a `x_n`, de charge volumique

    eta = -e N_A  pour -x_p < x < 0,     eta = +e N_D  pour 0 < x < x_n,

globalement neutre : `N_A x_p = N_D x_n`. Deux integrations de
`d2V/dx2 = -eta / eps` (2.15.3) donnent un champ lineaire par morceaux et
un potentiel en deux arcs de parabole, d'ou la hauteur de la barriere

    U_b = V(x_n) - V(-x_p) = (e / 2 eps) (N_A x_p^2 + N_D x_n^2).

L'exemple releve de la physique des semi-conducteurs ; le cours le donne
comme application de l'equation de Poisson, et c'est a ce titre qu'il vit
ici.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import sympy as sp

from polytech.electromagnetisme.domain.symboles import ABSCISSE


@dataclass(frozen=True, slots=True)
class JonctionAbrupte:
    """Une jonction pn, par ses deux dopages et la largeur depeuplee du cote p."""

    accepteurs: sp.Expr
    donneurs: sp.Expr
    largeur_p: sp.Expr

    CHARGE: ClassVar[sp.Symbol] = sp.Symbol("e", positive=True)
    PERMITTIVITE: ClassVar[sp.Symbol] = sp.Symbol("varepsilon", positive=True)

    @property
    def largeur_n(self) -> sp.Expr:
        """`x_n = N_A x_p / N_D`, par la neutralite globale."""
        return self.accepteurs * self.largeur_p / self.donneurs

    def charge_volumique(self) -> sp.Expr:
        """`eta(x)` : nulle hors de la zone de depletion."""
        x = ABSCISSE
        return sp.Piecewise(
            (0, x < -self.largeur_p),
            (-self.CHARGE * self.accepteurs, x < 0),
            (self.CHARGE * self.donneurs, x < self.largeur_n),
            (0, True),
        )

    def champ_p(self) -> sp.Expr:
        """`E = -e N_A (x + x_p) / eps`, pour `-x_p < x < 0`."""
        return (
            -self.CHARGE
            * self.accepteurs
            * (ABSCISSE + self.largeur_p)
            / self.PERMITTIVITE
        )

    def champ_n(self) -> sp.Expr:
        """`E = e N_D (x - x_n) / eps`, pour `0 < x < x_n`."""
        return (
            self.CHARGE
            * self.donneurs
            * (ABSCISSE - self.largeur_n)
            / self.PERMITTIVITE
        )

    def potentiel_p(self) -> sp.Expr:
        """`V = -integrale de E` depuis `-x_p`, ou le potentiel est pris nul."""
        muette = sp.Symbol("x_muette", real=True)
        champ = self.champ_p().subs(ABSCISSE, muette)
        return sp.factor(-sp.integrate(champ, (muette, -self.largeur_p, ABSCISSE)))

    def potentiel_n(self) -> sp.Expr:
        """`V = V(0) - integrale de E` depuis l'origine."""
        muette = sp.Symbol("x_muette", real=True)
        champ = self.champ_n().subs(ABSCISSE, muette)
        a_l_origine = self.potentiel_p().subs(ABSCISSE, 0)
        return a_l_origine - sp.integrate(champ, (muette, 0, ABSCISSE))

    def barriere(self) -> sp.Expr:
        """`U_b = V(x_n) - V(-x_p)`."""
        haut = self.potentiel_n().subs(ABSCISSE, self.largeur_n)
        return sp.simplify(haut - self.potentiel_p().subs(ABSCISSE, -self.largeur_p))
