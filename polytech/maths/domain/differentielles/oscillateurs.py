"""Le systeme masse-amortisseur-ressort, et sa reponse a une sinusoide.

`m y'' = -C y' - K y + x(t)` : une equation lineaire du second ordre a
coefficients constants, la meme que celle du circuit RLC. Son
comportement **libre** est la solution de l'equation homogene ; son
comportement **force**, sous `x(t) = sin(wt)`, est une sinusoide de meme
pulsation. Le **gain** -- rapport des amplitudes de sortie et d'entree --
vaut `1 / |K(jw)|` et depend de `w` ; il est infini quand `jw` est racine
de l'equation caracteristique : c'est la resonance, ou l'amplitude croit
avec le temps.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.differentielles.operateurs import D, OperateurDifferentiel


def masse_amortisseur_ressort(
    masse: object, amortissement: object, raideur: object
) -> OperateurDifferentiel:
    """`m D^2 + C D + K`."""
    return OperateurDifferentiel(
        sp.sympify(masse) * D**2 + sp.sympify(amortissement) * D + sp.sympify(raideur)
    )


def gain(operateur: OperateurDifferentiel, pulsation: object) -> sp.Expr:
    """`1 / |K(jw)|`, infini a la resonance."""
    module = sp.Abs(operateur.en(sp.I * sp.sympify(pulsation)))
    return sp.oo if module == 0 else 1 / module
