"""La serie de Laurent d'une fonction dans une couronne `R1 < |z - z0| < R2`.

Theoreme 14 : `f`, holomorphe sur la couronne, y vaut
`somme_(k in Z) a_k (z - z0)^k`, avec
`a_k = (1 / 2 i pi) cercle_r f(zeta) / (zeta - z0)^(k+1) d zeta` pour tout
cercle `C_r` de la couronne (relation (99)). Le theoreme des residus le
calcule : `a_k` est la somme des residus de `f(zeta) (zeta - z0)^(-k-1)` aux
singularites qu'entoure `C_r` -- celles de `f` dans le disque interieur, et
`z0` lui-meme quand `k >= 0`.

La couronne doit etre libre de singularites : sinon `f` n'y est pas
holomorphe et la serie n'existe pas ; le calcul leve. Les singularites de
`f` doivent former un ensemble fini, que sympy enumere.

Source : Mathematiques pour l'ingenieur III, syllabus, section 8, pages 23
a 25 ; exemple (106).
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.calculus.singularities import singularities

from polytech.maths.domain.analyse_complexe.residus import residu


@dataclass(frozen=True)
class Couronne:
    """`R1 < |z - z0| < R2` ; `R1 = 0` pour un disque epointe, `R2 = oo` au-dela."""

    centre: object
    interieur: object
    exterieur: object

    def _entourees(self, fonction: sp.Expr, variable: sp.Symbol) -> list[sp.Expr]:
        """Les singularites du disque interieur ; leve s'il en est dans la couronne."""
        trouvees = singularities(fonction, variable, sp.S.Complexes)
        if not trouvees.is_finite_set:
            msg = f"singularites de {fonction} en nombre infini ou inconnu : {trouvees}"
            raise ValueError(msg)
        dedans = []
        for point in trouvees:
            distance = sp.Abs(point - sp.sympify(self.centre))
            if sp.Le(distance, self.interieur) == sp.true:
                dedans.append(point)
            elif sp.Lt(distance, self.exterieur) == sp.true:
                msg = f"singularite dans la couronne : {point}"
                raise ValueError(msg)
        return dedans

    def coefficient(
        self, fonction: sp.Expr, variable: sp.Symbol, indice: int
    ) -> sp.Expr:
        """`a_k`, par (99) et les residus."""
        a = sp.sympify(self.centre)
        f = sp.sympify(fonction)
        points = set(self._entourees(f, variable)) | ({a} if indice >= 0 else set())
        integrande = f / (variable - a) ** (indice + 1)
        return sp.simplify(
            sum((residu(integrande, variable, p) for p in points), sp.Integer(0))
        )

    def somme_partielle(
        self, fonction: sp.Expr, variable: sp.Symbol, de: int, a: int
    ) -> sp.Expr:
        """`somme_(k = de)^a a_k (z - z0)^k`."""
        z0 = sp.sympify(self.centre)
        termes = (
            self.coefficient(fonction, variable, k) * (variable - z0) ** k
            for k in range(de, a + 1)
        )
        return sum(termes, sp.Integer(0))
