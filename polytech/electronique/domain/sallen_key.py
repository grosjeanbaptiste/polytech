"""La cellule Sallen-Key passe-bas, et sa forme canonique du second ordre.

Les trois nombres que l'enonce demande -- `G`, `omega_0`, `Q` -- ne sont pas
lus dans une table : ils sortent des deux equations de noeuds du montage.
Les calculer ainsi, plutot que de recopier la formule, est ce qui permet au
test de contredire le corrige si celui-ci se trompe.

L'amplificateur est monte en suiveur : sa sortie vaut la tension de son
entree non inverseuse, et son entree ne tire aucun courant.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

LAPLACE = sp.Symbol("s")
"""Variable de Laplace utilisee pour extraire les coefficients."""

DEGRE_ATTENDU = 3
"""Un denominateur du second ordre porte trois coefficients."""


@dataclass(frozen=True, slots=True)
class CelluleSallenKey:
    """Les deux resistances series et les deux capacites du montage."""

    r1: sp.Expr
    r2: sp.Expr
    c1: sp.Expr
    c2: sp.Expr

    def fonction_de_transfert(self, variable: sp.Expr) -> sp.Expr:
        """Resout les deux equations de noeuds et rend `V_o / V_e`."""
        entree, milieu, sortie = sp.symbols("V_e V_a V_o")
        equations = [
            sp.Eq(
                (entree - milieu) / self.r1,
                (milieu - sortie) / self.r2 + (milieu - sortie) * variable * self.c1,
            ),
            sp.Eq((milieu - sortie) / self.r2, sortie * variable * self.c2),
        ]
        solution = sp.solve(equations, [milieu, sortie], dict=True)
        return sp.cancel(solution[0][sortie] / entree)

    def _coefficients(self) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
        """Numerateur constant et les trois coefficients du denominateur."""
        numerateur, denominateur = sp.fraction(
            sp.together(self.fonction_de_transfert(LAPLACE)),
        )
        coefficients = sp.Poly(sp.expand(denominateur), LAPLACE).all_coeffs()
        if len(coefficients) != DEGRE_ATTENDU:
            msg = f"denominateur qui n'est pas du second ordre : {coefficients}"
            raise ValueError(msg)
        a2, a1, a0 = coefficients
        return numerateur, a2, a1, a0

    def pulsation_propre(self) -> float:
        """`omega_0 = sqrt(a_0 / a_2)`, lue sur le denominateur resolu."""
        _, a2, _, a0 = self._coefficients()
        return float(sp.sqrt(a0 / a2))

    def frequence_propre(self) -> float:
        """La meme pulsation, en hertz."""
        return self.pulsation_propre() / float(2 * sp.pi)

    def facteur_de_qualite(self) -> float:
        """`Q = sqrt(a_0 a_2) / a_1`, par identification a la forme canonique."""
        _, a2, a1, a0 = self._coefficients()
        return float(sp.sqrt(a0 * a2) / a1)

    def gain(self) -> float:
        """`G = N / a_0` : le gain continu de la cellule."""
        numerateur, _, _, a0 = self._coefficients()
        return float(numerateur / a0)
