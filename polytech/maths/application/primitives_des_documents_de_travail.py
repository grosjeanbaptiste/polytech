"""Les deux primitives que les « documents de travail » posent hors du recueil.

`Syllabus_Notes_de_cours_2011.pdf` reprend trois documents de travail sur
les primitives (pages 48 a 59 et 73 a 75 du PDF). Ils detaillent des
exemples et renvoient aux numeros du recueil -- sauf deux fois : page 54,
« la primitive de l'examen de juin 2002 », et page 59, une primitive
donnee avec sa solution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.application.recueil_primitives_sommes import par_morceaux
from polytech.maths.domain.symboles import VARIABLE as X

if TYPE_CHECKING:
    from polytech.maths.application.resolution_exs_01 import Resolution

JUIN_2002 = sp.cos(X) ** 2 / (1 - sp.tan(X) ** 2)
TRINOME = 4 * X**2 - 2 * X + 1
PAGE_59 = (2 * X + 1) / sp.sqrt(TRINOME)


def examen_de_juin_2002() -> Resolution:
    """`cos^2 x / (1 - tg^2 x) = cos^4 x / cos 2x`, linearise en trois termes."""
    double = sp.cos(2 * X)
    morceaux = (
        (1 / (4 * double), sp.atanh(sp.sin(2 * X)) / 8),
        (sp.Rational(1, 2), X / 2),
        (double / 4, sp.sin(2 * X) / 8),
    )
    return par_morceaux(JUIN_2002, morceaux, "linearisation")


def sur_la_racine_du_trinome() -> Resolution:
    """Le numerateur, ecrit avec la derivee `8x - 2` du trinome et une constante."""
    racine = sp.sqrt(TRINOME)
    centre = (4 * X - 1) / sp.sqrt(3)
    morceaux = (
        ((8 * X - 2) / (4 * racine), racine / 2),
        (3 / (2 * racine), sp.Rational(3, 4) * sp.asinh(centre)),
    )
    return par_morceaux(PAGE_59, morceaux, "derivee du trinome, puis argsh")


def primitives() -> tuple[Resolution, Resolution]:
    """Celle de la page 54, puis celle de la page 59."""
    return examen_de_juin_2002(), sur_la_racine_du_trinome()
