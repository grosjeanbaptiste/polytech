"""Evolution materielle d'une particule (cours III.4).

Le volume et les ecarts materiels s'obtiennent en *integrant dans le temps*
les taux instantanes -- divergence pour le volume, taux de deformation
lineique pour chaque direction.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import T, X, Y

_AXES = (0, 1)


def loi_de_volume(divergence: sp.Expr) -> sp.Expr:
    """Rapport ``V(t)/V0``, integre depuis la divergence."""
    return sp.simplify(sp.exp(sp.integrate(divergence, (T, 0, T))))


def ecart_materiel(tenseur: sp.Matrix, axe: int) -> sp.Expr:
    """Rapport ``delta(t)/delta_0`` le long d'un axe principal."""
    if axe not in _AXES:
        msg = f"axe hors des deux directions du plan : {axe}"
        raise ValueError(msg)
    taux = tenseur[axe, axe]
    return sp.simplify(sp.exp(sp.integrate(taux, (T, 0, T))))


def image_du_cercle(tenseur: sp.Matrix) -> sp.Eq:
    """Equation de ce que devient un disque materiel de rayon ``r_0``.

    Un point du cercle, repere par son angle polaire, est transporte par les
    deux taux principaux. Eliminer l'angle entre les deux coordonnees --
    c'est-a-dire invoquer ``cos^2 + sin^2 = 1`` -- donne l'equation de
    l'image, sans supposer d'avance qu'elle est une ellipse.
    """
    angle = sp.Symbol("theta", real=True)
    rayon = sp.Symbol("r_0", positive=True)
    demi_axes = [rayon * ecart_materiel(tenseur, axe) for axe in _AXES]
    invariant = (X / demi_axes[0]) ** 2 + (Y / demi_axes[1]) ** 2
    sur_l_image = {X: demi_axes[0] * sp.cos(angle), Y: demi_axes[1] * sp.sin(angle)}
    if sp.simplify(invariant.subs(sur_l_image)) != 1:
        msg = "l'angle polaire ne s'elimine pas : les axes ne sont pas principaux"
        raise ValueError(msg)
    return sp.Eq(invariant, 1, evaluate=False)
