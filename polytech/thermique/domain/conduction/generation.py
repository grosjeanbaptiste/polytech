"""La conduction stationnaire avec generation interne de chaleur.

Syllabus de thermique, tome 1, premiere partie, section 2.4. Effet Joule,
reaction nucleaire ou chimique : `laplacien(t) + q / lambda = 0`.

Dans un barreau de rayon `R` dont la surface est tenue a `t_0`, le
profil est parabolique, `t - t_0 = q (R^2 - r^2) / (4 lambda)`, maximal
sur l'axe. Tout ce qui est genere sort par la surface, ou la densite de
flux est imposee : `phi_R = R q / 2`. Au contact d'un fluide, la surface
se place donc a `t_f + R q / (2 K)`.

Le crayon de combustible d'un reacteur en est l'application : l'ecart
entre l'axe et le fluide est la somme de trois chutes -- dans la
pastille, dans la gaine, puis dans le fluide --, que le meme flux
lineique `pi d_1^2 q / 4` traverse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import sympy as sp

type Grandeur = sp.Expr | float


@dataclass(frozen=True, slots=True)
class BarreauGenerateur:
    """La generation volumique `q`, le rayon `R` et la conductivite."""

    generation: Grandeur
    rayon: Grandeur
    conductivite: Grandeur

    def echauffement_au_centre(self) -> Grandeur:
        """`t_c - t_0 = q R^2 / (4 lambda)`."""
        return self.generation * self.rayon**2 / (4 * self.conductivite)

    def profil(self, rayon: Grandeur, *, surface: Grandeur) -> Grandeur:
        """`t = t_0 + q (R^2 - r^2) / (4 lambda)`."""
        ecart = self.rayon**2 - rayon**2
        return surface + self.generation * ecart / (4 * self.conductivite)

    def densite_a_la_surface(self) -> Grandeur:
        """`phi_R = R q / 2`."""
        return self.rayon * self.generation / 2

    def profil_dans_un_fluide(
        self, rayon: Grandeur, echange: Grandeur, fluide: Grandeur
    ) -> Grandeur:
        """`t = t_f + R q / (2 K) + q (R^2 - r^2) / (4 lambda)`."""
        surface = fluide + self.densite_a_la_surface() / echange
        return self.profil(rayon, surface=surface)


def _flux_lineique(generation: float, pastille: float) -> float:
    """`pi d_1^2 q / 4`, par metre de crayon."""
    return math.pi * pastille**2 * generation / 4


def flux_lineique(generation: float, pastille: float) -> float:
    """`pi d_1^2 q / 4`, par metre de crayon, en W/m."""
    return _flux_lineique(generation, pastille)


def chute_dans_une_gaine(
    generation: float, pastille: float, gaine: float, *, conductivite: float
) -> float:
    """`t_1 - t_2 = d_1^2 q ln(d_2 / d_1) / (8 lambda_g)`."""
    if gaine <= pastille:
        msg = f"gaine plus etroite que la pastille : {gaine} <= {pastille}"
        raise ValueError(msg)
    resistance = math.log(gaine / pastille) / (2 * math.pi * conductivite)
    return _flux_lineique(generation, pastille) * resistance


def chute_de_convection(
    generation: float, pastille: float, gaine: float, *, echange: float
) -> float:
    """`t_2 - t_f = d_1^2 q / (4 K d_2)`."""
    return _flux_lineique(generation, pastille) / (echange * math.pi * gaine)
