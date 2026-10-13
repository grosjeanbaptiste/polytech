"""Les exercices que les deux syllabus de theorie posent au fil des pages.

`Syllabus_Notes_de_cours_2011.pdf` et
`Syllabus_Mathematiques_appliquees_2016_2017.pdf` n'ont pas de recueil pour
leurs chapitres de theorie ; ils laissent de loin en loin un « Exercice »
ou un « a titre d'exercice ». Ce module porte ceux qui se calculent ; les
demonstrations sont dans le corrige, avec ici leurs temoins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.application import recueil_algebre_ch03_resolution as resolution
from polytech.maths.domain.edp.classification import discriminant
from polytech.maths.domain.laplace.convolution import produit_de_convolution
from polytech.maths.domain.laplace.definition import transformee_par_l_integrale
from polytech.maths.domain.series.entieres import INDICE, SerieEntiere
from polytech.maths.domain.symboles import IMAGE, TEMPS

if TYPE_CHECKING:
    from polytech.maths.domain.algebre.solutions import Nature

OMEGA: Final = sp.Symbol("omega", positive=True)
X: Final = sp.Symbol("x", real=True)
Y: Final = sp.Symbol("y", real=True)
N: Final = sp.Symbol("n", integer=True, positive=True)
EXEMPLE_13: Final = sp.Matrix([[3, -1, 1, 2], [5, 1, -2, 6], [-2, -2, 3, 7]])
"""L'exemple 2 des notes manuscrites, page 13 : un systeme impossible."""
MODIFIE_13: Final = sp.Matrix([[3, -1, 1, 2], [5, 1, -2, 6], [-2, -2, 3, -4]])
"""Le meme, « si le second membre de la troisieme equation est -4 au lieu de 7 »."""
SERIES: Final = (
    SerieEntiere((-1) ** INDICE / (INDICE + 1)),
    SerieEntiere((-1) ** INDICE / ((INDICE + 1) * 2**INDICE)),
    SerieEntiere(3**INDICE / ((INDICE + 1) * (INDICE + 2) * (INDICE + 3))),
)
"""Une serie par domaine demande : `]-1, 1]`, `]-2, 2]`, `[-1/3, 1/3]`."""
FONDAMENTALES: Final = (TEMPS**2, TEMPS**3, TEMPS**-2)
"""Les trois solutions de `t^3 y''' - 6 t y' + 12 y = 0` que l'exemple donne."""
TEMOINS: Final = (TEMPS, sp.exp(TEMPS), sp.sin(TEMPS))


def nature_d_origine() -> Nature:
    """Le systeme de l'exemple, avant la modification."""
    return resolution.nature(EXEMPLE_13)


def systeme_modifie() -> tuple[Nature, tuple[sp.Expr, ...]]:
    """La nature du systeme modifie, et sa solution generale."""
    return resolution.nature(MODIFIE_13), resolution.solution(MODIFIE_13)


def laplace_du_sinus() -> tuple[sp.Expr, sp.Expr]:
    """`L(sin wt)` par l'integrale, puis par la linearite sur `e^(+-jwt)`."""
    complexe = 1 / (IMAGE - sp.I * OMEGA) - 1 / (IMAGE + sp.I * OMEGA)
    par_linearite = sp.simplify(complexe / (2 * sp.I))
    return transformee_par_l_integrale(sp.sin(OMEGA * TEMPS)), par_linearite


def convolutions() -> dict[str, tuple[sp.Expr, sp.Expr]]:
    """Les deux membres de chaque propriete, sur `t`, `e^t` et `sin t`."""
    f, g, h = TEMOINS
    etoile = produit_de_convolution
    return {
        "Commutativite": (etoile(f, g), etoile(g, f)),
        "Distributivite": (etoile(f, g + h), etoile(f, g) + etoile(f, h)),
        "Associativite": (etoile(f, etoile(g, h)), etoile(etoile(f, g), h)),
    }


def mineurs_du_wronskien() -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """`det W`, et les cofacteurs `W_31, W_32, W_33` de sa derniere ligne."""
    lignes = [[sp.diff(g, TEMPS, k) for g in FONDAMENTALES] for k in range(3)]
    wronskien = sp.Matrix(lignes)
    mineurs = tuple(sp.simplify(wronskien.cofactor(2, j)) for j in range(3))
    return sp.simplify(wronskien.det()), mineurs


def equation_d_ordre_trois() -> tuple[sp.Expr, sp.Expr]:
    """La solution particuliere pour `x(t) = t^4` par la formule (10.11), et la SGNH."""
    determinant, mineurs = mineurs_du_wronskien()
    force = TEMPS**4 / TEMPS**3
    integrales = [sp.integrate(m * force / determinant, TEMPS) for m in mineurs]
    termes = [g * k for g, k in zip(FONDAMENTALES, integrales, strict=True)]
    particuliere = sp.simplify(sp.Add(*termes))
    constantes = sp.symbols("c1:4")
    homogene = sp.Add(*[c * g for c, g in zip(constantes, FONDAMENTALES, strict=True)])
    return particuliere, homogene + particuliere


def terme_integre(rang: int) -> sp.Expr:
    """`int_0^(pi/2) cos(nx) / n^3 dx`, pour un rang donne."""
    return sp.integrate(sp.cos(rang * X) / rang**3, (X, 0, sp.pi / 2))


def serie_telescopique() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """La somme partielle de `x^(k-1) - x^k`, sa limite sur `[0, 1]`, l'ecart."""
    k = sp.Symbol("k", integer=True, positive=True)
    partielle = sp.simplify(sp.summation(X ** (k - 1) - X**k, (k, 1, N)))
    limite = sp.Piecewise((1, X < 1), (0, True))
    ecart = sp.limit(X**N, X, 1, "-")
    return partielle, limite, ecart


def discriminant_de_l_edp() -> sp.Expr:
    """`ab - h^2` pour `y u_xx + 2x u_xy + y u_yy = 0`, factorise."""
    return sp.factor(discriminant(Y, X, Y))
