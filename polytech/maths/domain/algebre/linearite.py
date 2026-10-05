"""Decider si une application est lineaire, et si des images en determinent une.

Le syllabus d'algebre (VIII.1) : `f : V -> W` est lineaire si et seulement
si `f(X + Y) = f(X) + f(Y)` et `f(alpha X) = alpha f(X)`. Les deux
conditions n'en font qu'une, `f(alpha X + beta Y) = alpha f(X) + beta f(Y)`,
que l'on verifie sur deux elements generiques et deux scalaires
symboliques. Les scalaires comptent : la conjugaison `z -> z_barre` est
lineaire quand `C` est vu comme espace sur `R`, et ne l'est pas sur `C`.

Une application lineaire est determinee par les images d'une base. Des
images imposees a des vecteurs lies ne sont compatibles que si elles
satisfont les memes relations de dependance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

from polytech.maths.domain.algebre.familles_de_vecteurs import rang_de_la_famille

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from polytech.maths.domain.algebre.familles_de_vecteurs import Famille


def respecte_les_combinaisons(
    regle: Callable[[Any], Any],
    premier: Any,  # noqa: ANN401 -- un vecteur, un polynome, une matrice, une fonction
    second: Any,  # noqa: ANN401 -- un element generique du meme espace
    scalaires: Sequence[sp.Symbol] | None = None,
) -> bool:
    """Vrai si `f(alpha X + beta Y) = alpha f(X) + beta f(Y)` identiquement."""
    alpha, beta = scalaires or sp.symbols("alpha beta")
    gauche = regle(alpha * premier + beta * second)
    droite = alpha * regle(premier) + beta * regle(second)
    ecart = sp.simplify(sp.expand(gauche - droite))
    if isinstance(ecart, sp.MatrixBase):
        return bool(ecart.is_zero_matrix)
    return bool(ecart == 0)


def existe_une_application_lineaire(antecedents: Famille, images: Famille) -> bool:
    """Vrai si les images satisfont chaque relation de dependance des antecedents."""
    jointes = tuple((*x, *y) for x, y in zip(antecedents, images, strict=True))
    return rang_de_la_famille(jointes) == rang_de_la_famille(antecedents)
