"""Les formules de trigonometrie du formulaire, et deux gestes qui en sortent.

Section 2 de `Formulaire_Aide_memoire.pdf` (page 3) : l'addition, Carnot,
Simpson, Simpson inverses et les fonctions hyperboliques. Elles sont ici
une **donnee**, recopiee de la page ; un test eprouve chaque ligne en des
points, pour qu'une faute de recopie ne passe pas.

Simpson transforme une somme en produit, ce qui factorise une equation ;
Simpson inverses fait le chemin contraire, pour integrer
(`trigonometrie.lineariser`). Fichier de donnees : chaque table a la meme
forme, membre de gauche puis membre de droite.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE as X

Y: Final = sp.Symbol("y", real=True)
"""Le second angle des formules a deux angles."""

_DEMI_SOMME, _DEMI_ECART = (X + Y) / 2, (X - Y) / 2

ADDITION: Final[dict[sp.Expr, sp.Expr]] = {
    sp.sin(X + Y): sp.sin(X) * sp.cos(Y) + sp.cos(X) * sp.sin(Y),
    sp.cos(X + Y): sp.cos(X) * sp.cos(Y) - sp.sin(X) * sp.sin(Y),
}
CARNOT: Final[dict[sp.Expr, sp.Expr]] = {
    sp.cos(X) ** 2: (1 + sp.cos(2 * X)) / 2,
    sp.sin(X) ** 2: (1 - sp.cos(2 * X)) / 2,
}
SIMPSON: Final[dict[sp.Expr, sp.Expr]] = {
    sp.sin(X) + sp.sin(Y): 2 * sp.sin(_DEMI_SOMME) * sp.cos(_DEMI_ECART),
    sp.cos(X) + sp.cos(Y): 2 * sp.cos(_DEMI_SOMME) * sp.cos(_DEMI_ECART),
    sp.sin(X) - sp.sin(Y): 2 * sp.sin(_DEMI_ECART) * sp.cos(_DEMI_SOMME),
    sp.cos(X) - sp.cos(Y): -2 * sp.sin(_DEMI_SOMME) * sp.sin(_DEMI_ECART),
}
SIMPSON_INVERSES: Final[dict[sp.Expr, sp.Expr]] = {
    sp.sin(X) * sp.sin(Y): (sp.cos(X - Y) - sp.cos(X + Y)) / 2,
    sp.sin(X) * sp.cos(Y): (sp.sin(X - Y) + sp.sin(X + Y)) / 2,
    sp.cos(X) * sp.cos(Y): (sp.cos(X - Y) + sp.cos(X + Y)) / 2,
}
HYPERBOLIQUES: Final[dict[sp.Expr, sp.Expr]] = {
    sp.cosh(X): (sp.exp(X) + sp.exp(-X)) / 2,
    sp.sinh(X): (sp.exp(X) - sp.exp(-X)) / 2,
    sp.cosh(X) ** 2 - sp.sinh(X) ** 2: sp.Integer(1),
}


def en_produit(somme: sp.Expr) -> sp.Expr:
    """Le produit que Simpson donne a `sin a +- sin b` ou `cos a +- cos b`."""
    premier, second = sp.Wild("premier"), sp.Wild("second")
    for gauche, droite in SIMPSON.items():
        modele = gauche.subs({X: premier, Y: second}, simultaneous=True)
        angles = sp.sympify(somme).match(modele)
        if angles is not None:
            return droite.subs(
                {X: angles[premier], Y: angles[second]}, simultaneous=True
            )
    msg = f"ni une somme ni une difference de deux sinus ou de deux cosinus : {somme}"
    raise ValueError(msg)


def en_exponentielles(expression: sp.Expr) -> sp.Expr:
    """L'expression ou `ch` et `sh` sont remplaces par leur definition."""
    return (
        sp.sympify(expression)
        .replace(sp.cosh, lambda u: (sp.exp(u) + sp.exp(-u)) / 2)
        .replace(sp.sinh, lambda u: (sp.exp(u) - sp.exp(-u)) / 2)
    )
