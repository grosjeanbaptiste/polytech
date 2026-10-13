"""Recueil de primitives, exercice 8 : rationnelles en `sin x` et `cos x`.

Le changement universel est `u = tg(x/2)`, qui rend tout rationnel. Il est
rarement le plus court : quand l'integrande change de signe avec `sin x`,
`u = cos x` suffit ; avec `cos x`, `u = sin x` ; quand il est inchange si
l'on change les deux, `u = tg x`. Et une puissance paire se linearise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.application.recueil_primitives_enonces import (
    EXERCICES_SUIVANTS as F,
)
from polytech.maths.application.recueil_primitives_enonces import A, B, H, W, X
from polytech.maths.application.recueil_primitives_gestes import (
    U,
    changement,
    terme_a_terme,
)
from polytech.maths.application.recueil_primitives_sommes import par_morceaux
from polytech.maths.domain.integration.trigonometrie import lineariser

if TYPE_CHECKING:
    from polytech.maths.application.resolution_exs_01 import Resolution

sin, cos, tan, sqrt = sp.sin, sp.cos, sp.tan, sp.sqrt
LINEARISATION = "linearisation"
SOMME = "produit en somme"
MOITIE = tan(X / 2)


def par_un_changement() -> dict[str, Resolution]:
    """Les primitives de l'exercice 8 qu'un changement de variable rend rationnelles."""
    return {
        "8a": changement(F["8a"], 2 / (7 - 3 * U**2), MOITIE),
        "8b": changement(F["8b"], 1 / (1 - U**2), sin(X)),
        "8f": changement(F["8f"], 2 / (1 + 2 * U - U**2), MOITIE),
        "8g": changement(F["8g"], 1 / (U**2 * (1 - U**2)), sin(X)),
        "8h": changement(F["8h"], -1 / U, cos(X) - sin(X)),
        "8l": changement(F["8l"], (1 - U**2) / (1 + U**2), sin(X)),
        "8m": changement(F["8m"], 1 / ((3 - U**2) * (1 + U**2)), tan(X)),
        "8n": changement(F["8n"], 1 / (1 + 5 * U), MOITIE),
        "8o": changement(F["8o"], -(1 - U**2), cos(X)),
    }


def par_une_reecriture() -> dict[str, Resolution]:
    """Celles qu'une reecriture en somme suffit a primitiver."""
    denominateur = A * cos(X) + B * sin(X)
    derive = sp.diff(denominateur, X)
    carre = A**2 + B**2
    en_deux = (
        (A / carre, A * X / carre),
        (B * derive / (carre * denominateur), B * sp.log(sp.Abs(denominateur)) / carre),
    )
    conjugue = (
        (sp.Integer(-1), -X),
        (1 / cos(X) ** 2, tan(X)),
        (sin(X) / cos(X) ** 2, 1 / cos(X)),
    )
    produit_e = tuple(
        (cos(omega * X) / 2, sin(omega * X) / (2 * omega)) for omega in (A - B, A + B)
    )
    produit_j = (sin(2 * W * X + H) - sin(H)) / 2
    return {
        "8c": par_morceaux(F["8c"], en_deux, "denominateur et sa derivee"),
        "8d": terme_a_terme(F["8d"], lineariser(F["8d"]), LINEARISATION),
        "8e": par_morceaux(F["8e"], produit_e, SOMME),
        "8i": terme_a_terme(F["8i"], lineariser(F["8i"]), LINEARISATION),
        "8j": terme_a_terme(F["8j"], produit_j, SOMME),
        "8k": par_morceaux(F["8k"], conjugue, "quantite conjuguee"),
        "8p": terme_a_terme(F["8p"], lineariser(F["8p"]), SOMME),
    }
