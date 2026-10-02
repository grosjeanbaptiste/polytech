"""Un fil rectiligne et une spire rectangulaire -- seance 2.1, exercice 2.

Le fil est porte par `Oy` et parcouru par `I_1` vers `+y`. La spire, de
cotes `a` selon `x` et `b` selon `y`, a son cote gauche a la distance `d`.
Elle porte `I_2` dans le sens direct autour de `+u_z` : vers le bas sur
le cote gauche, vers la droite en bas, vers le haut a droite, vers la
gauche en haut.

**Deux routes vers la force**, qui ne partagent rien. `force_cote_par_cote`
integre la force de Laplace sur chacun des quatre cotes ;
`force_par_les_travaux_virtuels` derive l'inductance mutuelle.

**Les cotes haut et bas ne sont pas inertes.** Chacun porte une force non
nulle selon `Oy` ; c'est leur somme qui s'annule. Les quatre forces
pointent vers l'**interieur** : la spire est comprimee, parce que son
propre flux s'oppose a celui du fil et qu'elle tend a en embrasser moins.

**Le signe de l'inductance mutuelle est une convention.** Avec la normale
`+u_z`, le champ du fil -- porte par `-u_z` -- traverse la spire a
rebours : le flux est negatif. La normale opposee donnerait l'oppose.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

ABSCISSE = sp.Symbol("x_integration", positive=True)
"""La variable muette des integrales selon `x`."""


def induction_du_fil(
    courant: sp.Expr,
    abscisse: sp.Expr,
    permeabilite: sp.Expr,
) -> sp.Expr:
    """La composante `B_z` du champ du fil, a l'abscisse `x > 0`.

    `B = mu_0 I / (2 pi x)` en module, et la regle de la main droite
    autour d'un courant vers `+y` le porte par `-u_z` du cote `x > 0`.
    """
    return -permeabilite * courant / (2 * sp.pi * abscisse)


@dataclass(frozen=True, slots=True)
class SpireDevantUnFil:
    """La spire rectangulaire, reperee par rapport au fil."""

    largeur: sp.Expr
    hauteur: sp.Expr
    distance: sp.Expr

    def flux(self, courant_du_fil: sp.Expr, permeabilite: sp.Expr) -> sp.Expr:
        """Le flux du champ du fil a travers la spire, normale `+u_z`."""
        induction = induction_du_fil(courant_du_fil, ABSCISSE, permeabilite)
        bornes = (ABSCISSE, self.distance, self.distance + self.largeur)
        return self.hauteur * sp.integrate(induction, bornes)

    def inductance_mutuelle(self, permeabilite: sp.Expr) -> sp.Expr:
        """`L_12 = phi_21 / I_1` : le flux par unite de courant du fil."""
        courant = sp.Symbol("I_unite", positive=True)
        return sp.simplify(self.flux(courant, permeabilite) / courant)


def force_cote_par_cote(
    spire: SpireDevantUnFil,
    courant_du_fil: sp.Expr,
    courant_de_la_spire: sp.Expr,
    permeabilite: sp.Expr,
) -> dict[str, tuple[sp.Expr, sp.Expr]]:
    """La force de Laplace sur chacun des quatre cotes, en `(F_x, F_y)`.

    `dF = I_2 ds ^ B` avec `B = B_z u_z`. Un element `dy u_y` donne
    `B_z dy u_x` ; un element `dx u_x` donne `-B_z dx u_y`. Le signe de
    chaque cote vient du sens dans lequel `I_2` le parcourt.
    """
    proche = spire.distance
    lointain = spire.distance + spire.largeur
    le_long_de_y = courant_de_la_spire * spire.hauteur
    le_long_de_x = courant_de_la_spire * sp.integrate(
        induction_du_fil(courant_du_fil, ABSCISSE, permeabilite),
        (ABSCISSE, proche, lointain),
    )
    zero = sp.Integer(0)
    return {
        "gauche": (
            -le_long_de_y * induction_du_fil(courant_du_fil, proche, permeabilite),
            zero,
        ),
        "droit": (
            le_long_de_y * induction_du_fil(courant_du_fil, lointain, permeabilite),
            zero,
        ),
        "bas": (zero, -le_long_de_x),
        "haut": (zero, le_long_de_x),
    }


def resultante(
    forces: dict[str, tuple[sp.Expr, sp.Expr]],
) -> tuple[sp.Expr, sp.Expr]:
    """La somme des quatre forces, composante par composante."""
    return (
        sp.simplify(sum(force[0] for force in forces.values())),
        sp.simplify(sum(force[1] for force in forces.values())),
    )


def force_par_les_travaux_virtuels(
    spire: SpireDevantUnFil,
    courant_du_fil: sp.Expr,
    courant_de_la_spire: sp.Expr,
    permeabilite: sp.Expr,
) -> tuple[sp.Expr, sp.Expr]:
    """`F = I_1 I_2 grad(L_12)`, a courants constants.

    Seul le terme mutuel de l'energie depend de la position. Deplacer la
    spire selon `x` fait varier `d` ; la deplacer selon `y` ne change
    rien, l'inductance mutuelle ne dependant pas de `y`.
    """
    mutuelle = spire.inductance_mutuelle(permeabilite)
    produit = courant_du_fil * courant_de_la_spire
    selon_x = sp.simplify(produit * sp.diff(mutuelle, spire.distance))
    return selon_x, sp.Integer(0)
