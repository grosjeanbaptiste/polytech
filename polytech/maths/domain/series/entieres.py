"""Les series entieres `somme a_n x^n`, et leur rayon de convergence.

Par le critere de la racine, la serie converge absolument si
`lim |a_n|^(1/n) . |x| < 1` et diverge si cette limite depasse 1. D'ou le
**rayon de convergence** `R = 1 / lim |a_n|^(1/n)`, que le critere du
quotient donne aussi : `R = 1 / lim |a_(n+1) / a_n|`. Pour `|x| = R` on ne
peut rien conclure : chaque bord se regarde a part.

A l'interieur de `]-R, R[` la serie s'integre et se derive terme a terme,
sans que le rayon change.

Ces deux limites n'existent pas toujours : une serie dont des coefficients
sont nuls de loin en loin (`cos x`) n'a ni l'une ni l'autre. Les deux
methodes levent alors, et renvoient a la limite superieure de
`lacunaires.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

INDICE: Final = sp.Symbol("n", integer=True, nonnegative=True)
"""L'indice `n` du terme general."""


RANGS_SONDES: Final = 12
"""Les premiers rangs ou l'on cherche des coefficients nuls."""


def _rayon(limite: sp.Expr) -> sp.Expr:
    """`1 / limite`, avec les deux cas extremes."""
    if limite == 0:
        return sp.oo
    return sp.Integer(0) if limite == sp.oo else 1 / limite


@dataclass(frozen=True, slots=True)
class SerieEntiere:
    """Le coefficient `a_n`, et le rang a partir duquel la serie commence."""

    terme: sp.Expr
    depuis: int = 0

    def coefficient(self, rang: int) -> sp.Expr:
        """`a_rang`, nul avant le premier rang."""
        if rang < self.depuis:
            return sp.Integer(0)
        return sp.simplify(sp.sympify(self.terme).subs(INDICE, rang))

    def somme_partielle(self, variable: sp.Expr, ordre: int) -> sp.Expr:
        """`somme a_n x^n` jusqu'a `n = ordre` compris."""
        return sum(
            (self.coefficient(n) * variable**n for n in range(ordre + 1)),
            sp.Integer(0),
        )

    def _exiger_une_limite(self) -> None:
        """Leve si plusieurs des premiers coefficients sont nuls."""
        rangs = range(self.depuis, self.depuis + RANGS_SONDES)
        nuls = [n for n in rangs if self.coefficient(n) == 0]
        if len(nuls) > 1:
            msg = (
                f"coefficients nuls aux rangs {nuls} : la limite n'existe pas, "
                "le rayon se calcule par la limite superieure"
            )
            raise ValueError(msg)

    def rayon_par_la_racine(self) -> sp.Expr:
        """`R = 1 / lim |a_n|^(1/n)` ; leve si cette limite n'existe pas."""
        self._exiger_une_limite()
        return _rayon(sp.limit(sp.Abs(self.terme) ** (1 / INDICE), INDICE, sp.oo))

    def rayon_par_le_quotient(self) -> sp.Expr:
        """`R = 1 / lim |a_(n+1) / a_n|` ; leve si cette limite n'existe pas."""
        self._exiger_une_limite()
        suivant = sp.sympify(self.terme).subs(INDICE, INDICE + 1)
        return _rayon(sp.limit(sp.Abs(suivant / self.terme), INDICE, sp.oo))

    def _converge_en(self, bord: sp.Expr) -> bool:
        general = sp.simplify(self.terme * bord**INDICE)
        return bool(sp.Sum(general, (INDICE, self.depuis, sp.oo)).is_convergent())

    def domaine_de_convergence(self) -> sp.Set:
        """L'ensemble des `x` ou la serie converge, bords examines un a un."""
        rayon = self.rayon_par_le_quotient()
        if rayon == 0:
            return sp.FiniteSet(0)
        if rayon == sp.oo:
            return sp.S.Reals
        return sp.Interval(
            -rayon,
            rayon,
            left_open=not self._converge_en(-rayon),
            right_open=not self._converge_en(rayon),
        )

    def derivee(self) -> SerieEntiere:
        """`S' = somme (n + 1) a_(n+1) x^n`."""
        suivant = sp.sympify(self.terme).subs(INDICE, INDICE + 1)
        return SerieEntiere((INDICE + 1) * suivant, max(self.depuis - 1, 0))

    def primitive(self) -> SerieEntiere:
        """`int_0^x S = somme a_(n-1) x^n / n`, qui commence un rang plus loin."""
        precedent = sp.sympify(self.terme).subs(INDICE, INDICE - 1)
        return SerieEntiere(sp.simplify(precedent / INDICE), self.depuis + 1)
