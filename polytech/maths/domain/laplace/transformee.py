"""La transformee de Laplace, notee comme le cours la note.

Les planches ecrivent la variable image `p`, et non `s`. Ce n'est pas un
detail : un corrige qui ecrirait `s` obligerait le lecteur a traduire a
chaque ligne, et c'est exactement le genre d'ecart qui fait perdre une
heure sur un examen.

`de_la_derivee` est ce qui rend la methode utile : `L{y'} = p Y(p) - y(0)`
fait entrer la condition initiale dans l'algebre, au lieu de la reserver
pour la fin.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.symboles import IMAGE, TEMPS


@dataclass(frozen=True, slots=True)
class Transformee:
    """Un original et son image, dans la variable `p` du cours."""

    original: sp.Expr
    image: sp.Expr

    @classmethod
    def de(cls, original: sp.Expr) -> Transformee:
        """Transforme ; leve si sympy ne conclut pas."""
        calculee = sp.laplace_transform(original, TEMPS, IMAGE, noconds=True)
        if calculee.has(sp.LaplaceTransform):
            msg = f"transformee de Laplace non calculable : {original}"
            raise ValueError(msg)
        return cls(original=original, image=sp.simplify(calculee))

    @staticmethod
    def de_la_derivee(image: sp.Expr, valeur_initiale: sp.Expr) -> sp.Expr:
        """`L{y'} = p Y(p) - y(0)`, la propriete qui fait toute la methode."""
        return IMAGE * image - valeur_initiale

    def inverse(self) -> sp.Expr:
        """Le retour a l'original, pour refermer la boucle."""
        return sp.simplify(
            sp.inverse_laplace_transform(self.image, IMAGE, TEMPS, noconds=True),
        )
