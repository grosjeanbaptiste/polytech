"""Macros du jeu « algorithme simplexe », exercices I a IV. Prefixe `Simplexe`."""

from __future__ import annotations

from polytech.optimisation.application import simplexe_bornes as bo
from polytech.optimisation.application import simplexe_decompte as dc
from polytech.optimisation.application import simplexe_dual as du
from polytech.optimisation.application import simplexe_optimums as op
from polytech.optimisation.application import simplexe_sensibilite as se
from polytech.optimisation.application.formatage import exact, intervalle, point
from polytech.optimisation.application.macros_simplexe_entiers import (
    macros_des_entiers,
)

_LETTRES = "ABCDEF"
_NOMS = ("Un", "Deux", "Trois")


def _lignes(prefixe: str, etapes: tuple[bo.Etape, ...]) -> dict[str, str]:
    """Une ligne de table par tableau : le point, la valeur, l'admissibilite."""
    return {
        f"{prefixe}{lettre}": (
            f"${point(etape.point)}$ & ${exact(etape.valeur)}$ & "
            f"{'oui' if etape.admissible else 'non'}"
        )
        for lettre, etape in zip(_LETTRES, etapes, strict=False)
    }


def _bornes() -> dict[str, str]:
    valeur, optimum = bo.optimum()
    seconde, second = dc.optimum(translate=False)
    return {
        **_lignes("SimplexeUnLigne", bo.parcours()),
        **_lignes("SimplexeUnTranslate", bo.parcours_translate()),
        "SimplexeUnOptimum": point(optimum),
        "SimplexeUnValeur": exact(valeur),
        "SimplexeUnDecompose": " & ".join(map(str, dc.decompte(translate=False))),
        "SimplexeUnDecale": " & ".join(map(str, dc.decompte(translate=True))),
        "SimplexeUnSecondOptimum": point(second),
        "SimplexeUnSecondeValeur": exact(seconde),
    }


def _optimums_et_dual() -> dict[str, str]:
    sommets = sorted(op.sommets_optimaux(), reverse=True)
    return {
        "SimplexeDeuxSommets": r",\ ".join(point(s) for s in sommets),
        "SimplexeDeuxValeur": exact(op.valeur_optimale()),
        "SimplexeTroisDuales": point(du.variables_duales()),
        "SimplexeTroisValeur": exact(du.valeur_du_dual()),
    }


def _sensibilite() -> dict[str, str]:
    valeur, optimum = se.optimum()
    macros = {
        "SimplexeQuatreOptimum": point(optimum),
        "SimplexeQuatreValeur": exact(valeur),
        "SimplexeQuatreDuales": point(se.variables_duales()),
    }
    for nom, bornes in zip("XYZ", se.intervalles_des_couts(), strict=True):
        macros[f"SimplexeQuatreCout{nom}"] = intervalle(*bornes)
    membres = se.intervalles_des_seconds_membres()
    for nom, bornes in zip(_NOMS, membres, strict=True):
        macros[f"SimplexeQuatreMembre{nom}"] = intervalle(*bornes)
    return macros


def macros_du_simplexe() -> dict[str, str]:
    """Les macros des six exercices du jeu."""
    return {
        **_bornes(),
        **_optimums_et_dual(),
        **_sensibilite(),
        **macros_des_entiers(),
    }
