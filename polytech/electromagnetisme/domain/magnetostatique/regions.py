"""Le champ de deux conducteurs, region par region -- exercice 7.

Deux conducteurs cylindriques de rayon `R`, d'axes en `y = 0` et `y = a`,
parcourus par des courants **opposes**. Le long de `Oy`, sept regions se
succedent -- et les trois exterieures donnent la meme expression.

**Les trois fonctions sont construites par superposition**, non recopiees
du corrige. Chacune somme deux contributions, et la seule chose qui change
d'une region a l'autre est **laquelle** des deux emploie le courant
enlace : celle du conducteur dans lequel on se trouve.

**La regle du courant enlace** porte sur le rapport des **aires** et non
des rayons : `I_C = I rho^2 / R^2`. Confondre les deux donnerait un champ
interieur constant au lieu de lineaire -- et ferait echouer le
recollement a la frontiere, ou la regle doit rendre le courant entier.

**Pourquoi une seule expression couvre les trois regions exterieures.**
C'est un fait algebrique : en laissant `y` **signe**, le changement de
signe porte l'inversion des directions. Au-dessous du conducteur
inferieur, les deux contributions s'opposent la ou elles s'ajoutaient dans
l'entrefer, et la formule le rend par le seul signe de `y`.
`champ_au_dessous_par_superposition` refait le calcul physique, avec des
distances positives et des sens explicites, pour que l'egalite se
verifie au lieu de se constater.

**Le champ de l'entrefer est minimal au milieu**, en `y = a/2`, parce que
le denominateur `y (a - y)` y est maximal. On s'attendrait au contraire.
"""

from __future__ import annotations

import sympy as sp


def courant_enlace(
    courant: sp.Expr,
    distance: sp.Expr,
    rayon: sp.Expr,
) -> sp.Expr:
    """`I_C = I rho^2 / R^2` : le rapport des **aires**, pas des rayons.

    Le courant etant reparti uniformement sur la section, la fraction
    enlacee par un cercle de rayon `rho` est le rapport de son aire a
    l'aire totale. En `rho = R` la regle rend le courant entier, ce qui
    est la condition pour que les expressions se recollent.
    """
    return courant * distance**2 / rayon**2


def _contribution_exterieure(courant: sp.Expr, distance: sp.Expr) -> sp.Expr:
    """`I / (2 pi rho)` : la contribution d'un conducteur vu de dehors."""
    return courant / (2 * sp.pi * distance)


def _contribution_interieure(
    courant: sp.Expr,
    distance: sp.Expr,
    rayon: sp.Expr,
) -> sp.Expr:
    """`I_C / (2 pi rho)`, qui se simplifie en `I rho / (2 pi R^2)`.

    Lineaire en `rho` : le `rho^2` du courant enlace l'emporte sur le
    `1/rho` d'Ampere. Le champ croit donc de zero au centre jusqu'a sa
    valeur de surface, au lieu de diverger.
    """
    return courant_enlace(courant, distance, rayon) / (2 * sp.pi * distance)


def champ_a_l_exterieur(
    courant: sp.Expr,
    ecart: sp.Expr,
    ordonnee: sp.Expr,
) -> sp.Expr:
    """La composante selon `u_x` du champ, hors des deux conducteurs.

    Les deux contributions s'ajoutent dans l'entrefer -- les courants
    etant opposes, leurs champs y pointent du meme cote. Le signe global
    est negatif : le champ est porte par `-u_x`.

    Valable pour les trois regions exterieures si `ordonnee` est **signe**.
    """
    du_bas = _contribution_exterieure(courant, ordonnee)
    du_haut = _contribution_exterieure(courant, ecart - ordonnee)
    return -(du_bas + du_haut)


def champ_dans_le_conducteur_superieur(
    courant: sp.Expr,
    ecart: sp.Expr,
    rayon: sp.Expr,
    ordonnee: sp.Expr,
) -> sp.Expr:
    """Dans le conducteur du haut : le sien enlace, l'autre entier."""
    du_bas = _contribution_exterieure(courant, ordonnee)
    du_haut = _contribution_interieure(courant, ecart - ordonnee, rayon)
    return -(du_bas + du_haut)


def champ_dans_le_conducteur_inferieur(
    courant: sp.Expr,
    ecart: sp.Expr,
    rayon: sp.Expr,
    ordonnee: sp.Expr,
) -> sp.Expr:
    """Dans le conducteur du bas : le sien enlace, l'autre entier."""
    du_bas = _contribution_interieure(courant, ordonnee, rayon)
    du_haut = _contribution_exterieure(courant, ecart - ordonnee)
    return -(du_bas + du_haut)


def champ_au_dessous_par_superposition(
    courant: sp.Expr,
    ecart: sp.Expr,
    profondeur: sp.Expr,
) -> sp.Expr:
    """Le champ a la profondeur `h` sous le conducteur inferieur.

    Refait avec des distances **positives** et des sens explicites : le
    conducteur du bas est a la distance `h` et son champ pointe vers
    `+u_x` de ce cote-la, celui du haut est a `a + h` et pointe vers
    `-u_x`. Les deux s'**opposent** ici, la ou ils s'ajoutaient dans
    l'entrefer.

    Sert a demontrer que la formule generale, prise avec un `y` negatif,
    rend bien ce champ -- ce que le manuscrit constate sans le montrer.
    """
    du_bas = _contribution_exterieure(courant, profondeur)
    du_haut = _contribution_exterieure(courant, ecart + profondeur)
    return du_bas - du_haut
