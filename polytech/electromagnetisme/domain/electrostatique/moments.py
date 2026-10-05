"""Le centre electrique et le moment electrique d'un systeme de charges.

Le syllabus d'electricite generale (2.2 et 2.3) :

* les centres electriques des charges positives et negatives,
  `r+ = somme(q_i r_i) / Q+` et `r- = somme(q_i r_i) / Q-` (2.2.1), et celui
  du systeme, `r_ce = somme(q_i r_i) / Q` (2.2.3) -- « equivalent au centre
  de masse », sauf que la charge totale peut etre nulle ;
* le moment electrique par rapport a un point `A`,
  `p(r_A) = somme q_i (r_i - r_A)` (2.3.1), qui ne depend plus de `A` quand
  le systeme est neutre : c'est alors le moment dipolaire, `p = q d` pour
  un doublet (2.3.4) ;
* le moment quadrupolaire `p_uv = somme q_i u_i v_i` (2.7.18d), « analogue
  aux moments d'inertie de la Mecanique ».
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import sympy as sp

from polytech.electromagnetisme.domain.symboles import ABSCISSE

Point = tuple[Any, Any, Any]
ORIGINE: Final[Point] = (0, 0, 0)
_AZIMUT: Final = sp.Symbol("varphi", real=True)


@dataclass(frozen=True, slots=True)
class SystemeDeCharges:
    """Des charges ponctuelles, chacune avec sa position."""

    charges: tuple[tuple[Any, Point], ...]

    def _somme(self, retenues: list[tuple[Any, Point]], reference: Point) -> Point:
        composantes = (
            sum(
                (
                    sp.sympify(q) * (point[axe] - reference[axe])
                    for q, point in retenues
                ),
                sp.Integer(0),
            )
            for axe in range(3)
        )
        x, y, z = composantes
        return (x, y, z)

    def _centre(self, retenues: list[tuple[Any, Point]]) -> Point:
        if not retenues:
            msg = "aucune charge de ce signe : le centre electrique n'existe pas"
            raise ValueError(msg)
        total = sum((sp.sympify(q) for q, _ in retenues), sp.Integer(0))
        if total == 0:
            msg = "charge totale nulle : le centre electrique n'est pas defini"
            raise ValueError(msg)
        x, y, z = self._somme(retenues, ORIGINE)
        return (x / total, y / total, z / total)

    def charge_positive(self) -> Any:  # noqa: ANN401 -- exacte ou reelle, selon les charges
        """`Q+`, la somme des seules charges positives."""
        return sum(q for q, _ in self.charges if q > 0)

    def charge_negative(self) -> Any:  # noqa: ANN401 -- exacte ou reelle, selon les charges
        """`Q-`, la somme des seules charges negatives."""
        return sum(q for q, _ in self.charges if q < 0)

    def charge_totale(self) -> Any:  # noqa: ANN401 -- exacte ou reelle, selon les charges
        """`Q = Q+ + Q-`."""
        return self.charge_positive() + self.charge_negative()

    def centre_des_positives(self) -> Point:
        """`r+`, le centre electrique des charges positives."""
        return self._centre([c for c in self.charges if c[0] > 0])

    def centre_des_negatives(self) -> Point:
        """`r-`, le centre electrique des charges negatives."""
        return self._centre([c for c in self.charges if c[0] < 0])

    def centre_electrique(self) -> Point:
        """`r_ce`, defini seulement si la charge totale n'est pas nulle."""
        return self._centre(list(self.charges))

    def moment_electrique(self, reference: Point = ORIGINE) -> Point:
        """`p(r_A) = somme q_i (r_i - r_A)`."""
        return self._somme(list(self.charges), reference)

    def moment_quadrupolaire(self) -> sp.Matrix:
        """Le tenseur `p_uv = somme q_i u_i v_i`, par rapport a l'origine."""
        return sp.Matrix(
            3,
            3,
            lambda u, v: sum(
                sp.sympify(q) * point[u] * point[v] for q, point in self.charges
            ),
        )


def moment_d_un_doublet(charge: Any, *, negative: Point, positive: Point) -> Point:  # noqa: ANN401 -- exacte ou reelle
    """`p = q d`, `d` allant de la charge negative a la charge positive."""
    x, y, z = (
        charge * (plus - moins) for plus, moins in zip(positive, negative, strict=True)
    )
    return (x, y, z)


def moment_d_un_filet(densite: sp.Expr, debut: object, fin: object) -> sp.Expr:
    """`integrale de lambda(x) x dx` : le moment d'un filet porte par l'axe x."""
    return sp.simplify(sp.integrate(densite * ABSCISSE, (ABSCISSE, debut, fin)))


def moment_d_une_coque(
    densite: sp.Expr, rayon: sp.Expr, colatitude: sp.Symbol
) -> Point:
    """`integrale de sigma r dS` sur une sphere, par rapport a son centre."""
    position = (
        sp.sin(colatitude) * sp.cos(_AZIMUT),
        sp.sin(colatitude) * sp.sin(_AZIMUT),
        sp.cos(colatitude),
    )
    element = densite * rayon**3 * sp.sin(colatitude)
    x, y, z = (
        sp.integrate(element * axe, (colatitude, 0, sp.pi), (_AZIMUT, 0, 2 * sp.pi))
        for axe in position
    )
    return (x, y, z)
