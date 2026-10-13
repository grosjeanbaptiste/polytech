"""La table des derivees du formulaire : vingt lignes.

Section 3.2 de `Formulaire_Aide_memoire.pdf` (page 4). La planche 33 du
cours en donne quatorze (`derivees.py`) ; le formulaire y ajoute `x^alpha`,
la cotangente, `a^x`, `log_a x`, la cotangente hyperbolique et son
argument, et il ecrit deux fois la derivee de la tangente et celle de la
cotangente : `1 / cos^2 x = 1 + tg^2 x`.

Fichier de donnees : une ligne par fonction, ses ecritures dans l'ordre
du formulaire. Un test redérive chaque ligne.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE as X

ALPHA: Final = sp.Symbol("alpha", real=True)
"""L'exposant reel de `x^alpha`."""
BASE: Final = sp.Symbol("a", positive=True)
"""La base de `a^x` et de `log_a x`."""

DERIVEES_DU_FORMULAIRE: Final[dict[sp.Expr, tuple[sp.Expr, ...]]] = {
    X**ALPHA: (ALPHA * X ** (ALPHA - 1),),
    sp.sin(X): (sp.cos(X),),
    sp.cos(X): (-sp.sin(X),),
    sp.tan(X): (1 / sp.cos(X) ** 2, 1 + sp.tan(X) ** 2),
    sp.cot(X): (-1 / sp.sin(X) ** 2, -(1 + sp.cot(X) ** 2)),
    sp.exp(X): (sp.exp(X),),
    sp.log(X): (1 / X,),
    sp.sinh(X): (sp.cosh(X),),
    sp.cosh(X): (sp.sinh(X),),
    sp.tanh(X): (1 / sp.cosh(X) ** 2,),
    sp.coth(X): (-1 / sp.sinh(X) ** 2,),
    sp.asin(X): (1 / sp.sqrt(1 - X**2),),
    sp.acos(X): (-1 / sp.sqrt(1 - X**2),),
    sp.atan(X): (1 / (1 + X**2),),
    BASE**X: (BASE**X * sp.log(BASE),),
    sp.log(X, BASE): (1 / (X * sp.log(BASE)),),
    sp.asinh(X): (1 / sp.sqrt(X**2 + 1),),
    sp.acosh(X): (1 / sp.sqrt(X**2 - 1),),
    sp.atanh(X): (1 / (1 - X**2),),
    sp.acoth(X): (1 / (1 - X**2),),
}
"""`f(x)` et les ecritures de sa derivee."""


def derivee_du_formulaire(fonction: sp.Expr) -> sp.Expr:
    """La premiere ecriture que la table donne ; leve hors de la table."""
    if fonction not in DERIVEES_DU_FORMULAIRE:
        msg = f"{fonction} n'est pas dans la table du formulaire (section 3.2)"
        raise ValueError(msg)
    return DERIVEES_DU_FORMULAIRE[fonction][0]
