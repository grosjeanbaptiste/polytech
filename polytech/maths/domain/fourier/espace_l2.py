"""L'espace `L^2(mu)` des fonctions de carre integrable.

Une densite `p(x) >= 0` pondere les points de l'ensemble `X` et definit
une mesure : `int f dmu = int f(x) p(x) dx`. Les fonctions telles que
`int |f|^2 dmu` est fini forment un espace vectoriel complexe, muni du
produit scalaire hermitien

    <f, g> = int f . conj(g) dmu,

de la norme `||f|| = racine(<f, f>)` et de la distance `||f - g||`. Tout
s'y passe comme dans `R^n` : inegalite de Cauchy-Schwarz, inegalite
triangulaire, orthogonalite quand `<f, g> = 0`.

Pour les series de Fourier, la mesure est `dx / longueur` : l'intervalle
entier y pese 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.calculus.util import function_range


@dataclass(frozen=True, slots=True)
class EspaceL2:
    """La variable, l'ensemble `X` et la densite de la mesure."""

    variable: sp.Symbol
    domaine: sp.Set
    densite: sp.Expr = sp.S.One

    def __post_init__(self) -> None:
        """Refuse une densite qui prend des valeurs negatives."""
        valeurs = function_range(sp.sympify(self.densite), self.variable, self.domaine)
        if valeurs.inf.is_negative:
            msg = f"densite negative sur {self.domaine} : ce n'est pas une mesure"
            raise ValueError(msg)

    @classmethod
    def normalise(cls, variable: sp.Symbol, debut: object, fin: object) -> EspaceL2:
        """`dmu = dx / (b - a)` sur `[a, b]` : la mesure des series de Fourier."""
        a, b = sp.sympify(debut), sp.sympify(fin)
        return cls(variable, sp.Interval(a, b), 1 / (b - a))

    def integrale(self, fonction: sp.Expr) -> sp.Expr:
        """`int f dmu`."""
        bornes = (self.variable, self.domaine.inf, self.domaine.sup)
        return sp.simplify(sp.integrate(fonction * self.densite, bornes))

    def produit_scalaire(self, premiere: sp.Expr, seconde: sp.Expr) -> sp.Expr:
        """`<f, g> = int f . conj(g) dmu`."""
        return self.integrale(sp.sympify(premiere) * sp.conjugate(seconde))

    def norme(self, fonction: sp.Expr) -> sp.Expr:
        """`||f||` ; leve si `|f|^2` n'est pas integrable."""
        carre = self.produit_scalaire(fonction, fonction)
        if not carre.is_finite:
            msg = f"{fonction} n'est pas de carre integrable sur {self.domaine}"
            raise ValueError(msg)
        return sp.sqrt(carre)

    def distance(self, premiere: sp.Expr, seconde: sp.Expr) -> sp.Expr:
        """`d(f, g) = ||f - g||`."""
        return self.norme(sp.sympify(premiere) - seconde)

    def sont_orthogonales(self, premiere: sp.Expr, seconde: sp.Expr) -> bool:
        """Vrai si `<f, g> = 0`."""
        return bool(self.produit_scalaire(premiere, seconde) == 0)
