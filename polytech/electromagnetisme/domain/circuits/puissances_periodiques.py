"""Les puissances d'une branche en regime periodique non sinusoidal.

Le syllabus d'electricite generale (17.4 et 17.5). Seuls les harmoniques
presents a la fois dans la tension et dans le courant portent de la
puissance moyenne : `P = U_0 I_0 + somme U_n I_n cos(phi_n)` (17.4.1). Le
facteur de puissance `lambda = P / (U I)` (17.4.4) ne depasse pas un en
valeur absolue, mais n'est plus le cosinus d'un dephasage.

La puissance apparente reste `S = U I`. Si l'on definit la puissance
reactive par `Q = somme U_n I_n sin(phi_n)` (17.5.3), il faut une
troisieme grandeur pour fermer le bilan : `S^2 = P^2 + Q^2 + D^2`
(17.5.4), `D` etant la puissance deformante.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def _produits(
    tensions: Sequence[Grandeur],
    courants: Sequence[Grandeur],
    ecarts: Sequence[Grandeur],
) -> list[tuple[Grandeur, Grandeur]]:
    if not len(tensions) == len(courants) == len(ecarts):
        msg = (
            f"il faut autant de tensions, de courants et de dephasages : "
            f"{len(tensions)}, {len(courants)} et {len(ecarts)}"
        )
        raise ValueError(msg)
    apparentes = (u * i for u, i in zip(tensions, courants, strict=True))
    return list(zip(apparentes, ecarts, strict=True))


def puissance_active_des_harmoniques(
    tensions: Sequence[Grandeur],
    courants: Sequence[Grandeur],
    ecarts: Sequence[Grandeur],
    *,
    continues: tuple[Grandeur, Grandeur] = (0, 0),
) -> sp.Expr:
    """`P = U_0 I_0 + somme U_n I_n cos(phi_n)` (17.4.1), en watts."""
    produits = _produits(tensions, courants, ecarts)
    alternative = sp.Add(*(s * sp.cos(ecart) for s, ecart in produits))
    return continues[0] * continues[1] + alternative


def puissance_reactive_des_harmoniques(
    tensions: Sequence[Grandeur],
    courants: Sequence[Grandeur],
    ecarts: Sequence[Grandeur],
) -> sp.Expr:
    """`Q = somme U_n I_n sin(phi_n)` (17.5.3), en vars."""
    produits = _produits(tensions, courants, ecarts)
    return sp.Add(*(s * sp.sin(ecart) for s, ecart in produits))


def puissance_deformante_generale(
    apparente: Grandeur, active: Grandeur, reactive: Grandeur
) -> sp.Expr:
    """`D = racine(S^2 - P^2 - Q^2)` (17.5.4)."""
    carre = sp.simplify(sp.sympify(apparente) ** 2 - active**2 - reactive**2)
    if (carre < 0) == sp.true:
        msg = f"P^2 + Q^2 depasse la puissance apparente au carre de {-carre}"
        raise ValueError(msg)
    return sp.sqrt(carre)
