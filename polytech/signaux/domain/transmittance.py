"""Une transmittance, et les deux assemblages que le cours lui fait subir.

La mise en boucle fermee est le geste central du cours : `T = G / (1 + G)`
pour un retour unitaire. L'ecrire ici, et non a chaque exercice, evite la
faute de signe qui transforme une contre-reaction en reaction positive.

Les poles sont rendus **tries**, et une transmittance sans pole leve : un
gain pur n'est pas un systeme, il n'a ni dynamique ni stabilite, et le
traiter comme tel donnerait un tableau de Routh vide.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE


@dataclass(frozen=True, slots=True)
class Transmittance:
    """Une fraction rationnelle en `p`."""

    expression: sp.Expr

    def numerateur(self) -> sp.Expr:
        """Le numerateur, sous forme developpee."""
        return sp.expand(sp.fraction(sp.cancel(self.expression))[0])

    def denominateur(self) -> sp.Expr:
        """Le denominateur, sous forme developpee."""
        return sp.expand(sp.fraction(sp.cancel(self.expression))[1])

    def degre(self) -> int:
        """Le degre du denominateur : le nombre de poles, multiplicite comprise."""
        return int(sp.degree(sp.Poly(self.denominateur(), IMAGE), IMAGE))

    def poles(self) -> tuple[sp.Expr, ...]:
        """Les racines du denominateur, triees ; leve s'il n'y en a aucune."""
        racines = sp.solve(sp.Eq(self.denominateur(), 0), IMAGE)
        if not racines:
            msg = f"aucun pole : {self.expression} n'est pas un systeme dynamique"
            raise ValueError(msg)
        return tuple(sorted(racines, key=sp.default_sort_key))

    def zeros(self) -> tuple[sp.Expr, ...]:
        """Les racines du numerateur, triees ; vide si le numerateur est constant."""
        racines = sp.solve(sp.Eq(self.numerateur(), 0), IMAGE)
        return tuple(sorted(racines, key=sp.default_sort_key))

    def en_boucle_fermee(self) -> Transmittance:
        """`G / (1 + G)` : le retour unitaire du cours."""
        ferme = sp.cancel(self.expression / (1 + self.expression))
        return Transmittance(expression=ferme)

    def en_serie_avec(self, autre: Transmittance) -> Transmittance:
        """Deux blocs en cascade se multiplient."""
        return Transmittance(expression=sp.cancel(self.expression * autre.expression))

    def reponse_frequentielle(self) -> sp.Expr:
        """`G(j omega)`, point de depart de Bode, Nyquist et Black."""
        return self.expression.subs(IMAGE, sp.I * sp.Symbol("omega", positive=True))
