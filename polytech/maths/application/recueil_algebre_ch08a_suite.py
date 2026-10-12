"""Recueil d'algebre, chapitre VIII, exercices 10 a 13 : noyau, image, rang.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, section VIII.4
(pages 103 et 104). Les espaces d'ordre ou de degre `n` se calculent a un
`n` donne ; le corrige raisonne pour tout `n`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.domain.algebre.applications_lineaires import ApplicationLineaire
from polytech.maths.domain.algebre.familles_de_vecteurs import prolonger_en_base
from polytech.maths.domain.algebre.rang_exact import rangs_selon_le_parametre

if TYPE_CHECKING:
    from polytech.maths.application.recueil_algebre_ch07_familles import Base

M = sp.Matrix
PARAMETRE: Final = sp.Symbol("t")
_T = PARAMETRE
MATRICE_13: Final = M([[1, _T, 1], [_T, 1, _T], [1, _T, -_T]])
SINGULIERES_13: Final = (-1, 1)


def _exiger_un_ordre(ordre: int) -> None:
    if ordre < 1:
        msg = f"ordre non strictement positif : {ordre}"
        raise ValueError(msg)


def _noyau_et_rang(application: ApplicationLineaire) -> tuple[int, int]:
    return len(application.noyau()), application.rang()


def exercice_10(degre: int, ordre: int) -> tuple[int, int]:
    """Les dimensions de `Ker D^k` et de `Im D^k` sur les polynomes de degre <= n."""
    _exiger_un_ordre(ordre)
    derivation = M(degre + 1, degre + 1, lambda i, j: j if j == i + 1 else 0)
    return _noyau_et_rang(ApplicationLineaire(derivation**ordre))


def _sur_les_matrices(ordre: int, signe: int) -> ApplicationLineaire:
    def regle(*termes: sp.Symbol) -> list[sp.Expr]:
        matrice = M(ordre, ordre, termes)
        return list((matrice + signe * matrice.T) / 2)

    return ApplicationLineaire.depuis_la_regle(regle, ordre * ordre)


def exercice_11(ordre: int) -> dict[str, tuple[int, int]]:
    """Pour `f1` et `f2` : la dimension du noyau, puis celle de l'image."""
    _exiger_un_ordre(ordre)
    return {
        "f1": _noyau_et_rang(_sur_les_matrices(ordre, 1)),
        "f2": _noyau_et_rang(_sur_les_matrices(ordre, -1)),
    }


def exercice_12(ordre: int) -> int:
    """La dimension du noyau de la trace, sur les matrices d'ordre `n`."""
    _exiger_un_ordre(ordre)
    trace = ApplicationLineaire.depuis_la_regle(
        lambda *termes: [M(ordre, ordre, termes).trace()], ordre * ordre
    )
    return len(trace.noyau())


def exercice_13() -> tuple[int, dict[sp.Expr, int]]:
    """Le rang de `f` hors des valeurs singulieres, et en chacune d'elles."""
    return rangs_selon_le_parametre(MATRICE_13, PARAMETRE)


def images_du_13() -> dict[int, Base]:
    """Une base de `Im f` en `t = -1` et en `t = 1`."""
    return {
        valeur: ApplicationLineaire(MATRICE_13.subs(PARAMETRE, valeur)).image()
        for valeur in SINGULIERES_13
    }


def completees_du_13() -> dict[int, Base]:
    """Chacune de ces bases, completee en une base de `R^3`."""
    return {valeur: prolonger_en_base(b) for valeur, b in images_du_13().items()}
