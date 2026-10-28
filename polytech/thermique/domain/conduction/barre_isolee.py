"""Une barre a generation interne, a base imposee et bout isole.

`t'' + S = 0` avec `S = q / lambda`, `t(0) = T_0` et `t'(L) = 0`. Le profil
est resolu par `sympy.dsolve`, et non recopie : c'est la fiche qui est
comparee a ce resultat.
"""

from __future__ import annotations

import sympy as sp

from polytech.thermique.domain.conduction.symboles import TEMPERATURE

type Grandeur = sp.Expr | float


def profil_de_barre_isolee(
    abscisse: sp.Symbol, longueur: Grandeur, base: Grandeur, source: Grandeur
) -> sp.Expr:
    """`t(x) = T_0 + S (L x - x^2 / 2)`, `S = q / lambda` pouvant etre negatif."""
    t = TEMPERATURE(abscisse)
    equation = sp.Eq(sp.diff(t, abscisse, 2) + source, 0)
    pente_en_bout = sp.diff(t, abscisse).subs(abscisse, longueur)
    solution = sp.dsolve(equation, t, ics={t.subs(abscisse, 0): base, pente_en_bout: 0})
    return sp.expand(solution.rhs)
