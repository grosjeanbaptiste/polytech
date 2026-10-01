"""Les transformees du cours, et les proprietes qui servent vraiment.

Le cours en retient trois : la linearite, le retard -- qui devient un
facteur `exp(-p tau)` -- et le theoreme de la derivation, qui fait entrer
la condition initiale dans l'algebre. Ce sont elles que les exercices
emploient, et elles seules sont exposees ici ; une table exhaustive serait
une table, pas un geste.

La transformee de Fourier n'est pas celle de Laplace avec `p = j omega` :
elle integre de moins l'infini a plus l'infini, Laplace de zero a plus
l'infini. Sur un signal causal les deux coincident, et c'est seulement
dans ce cas que le cours se permet le raccourci. La garde le rappelle.
"""

from __future__ import annotations

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE, PULSATION, TEMPS


def transformee_de_laplace(signal: sp.Expr) -> sp.Expr:
    """`L{x}(p)` ; leve si sympy ne conclut pas."""
    image = sp.laplace_transform(signal, TEMPS, IMAGE, noconds=True)
    if image.has(sp.LaplaceTransform):
        msg = f"transformee de Laplace non calculable : {signal}"
        raise ValueError(msg)
    return sp.simplify(image)


def retard(image: sp.Expr, duree: sp.Expr) -> sp.Expr:
    """Retarder de `tau` multiplie l'image par `exp(-p tau)`."""
    return sp.simplify(image * sp.exp(-IMAGE * duree))


def derivation(image: sp.Expr, valeur_initiale: sp.Expr) -> sp.Expr:
    """`L{x'} = p X(p) - x(0)` : la propriete qui rend la methode utile."""
    return sp.simplify(IMAGE * image - valeur_initiale)


def fourier_depuis_laplace(image: sp.Expr) -> sp.Expr:
    """`X(j omega)` pour un signal **causal**, par substitution `p = j omega`.

    Le raccourci n'est licite que sur un signal causal : Fourier integre
    sur toute la droite reelle, Laplace sur la demi-droite positive, et
    les deux ne coincident que si le signal est nul avant zero.
    """
    return sp.simplify(image.subs(IMAGE, sp.I * PULSATION))
