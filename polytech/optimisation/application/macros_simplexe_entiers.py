"""Macros du jeu « algorithme simplexe », exercices V et VI. Prefixe `Simplexe`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.application import simplexe_entiers as en
from polytech.optimisation.application.formatage import exact, point
from polytech.optimisation.domain.contraintes import Sens

if TYPE_CHECKING:
    from polytech.optimisation.domain.formulation import ProgrammeGeneral

_INF, _SUP = Sens.INFERIEUR, Sens.SUPERIEUR
_NOMS = {1: "Un", 2: "Deux", 3: "Trois"}


def _penalites(prefixe: str, probleme: ProgrammeGeneral) -> dict[str, str]:
    return {
        f"{prefixe}{_NOMS[variable]}": point(couple)
        for variable, couple in en.penalites_a_la_racine(probleme).items()
    }


def _cinq() -> dict[str, str]:
    racine = en.noeud(en.CINQ)
    gauche = en.noeud(en.CINQ, ((1, _INF, 3),))
    valeur, optimum = en.optimum_entier(en.CINQ)
    return {
        "SimplexeCinqRelaxe": point(racine.point or ()),
        "SimplexeCinqEvaluation": exact(racine.evaluation),
        **_penalites("SimplexeCinqPenalites", en.CINQ),
        "SimplexeCinqGauche": exact(gauche.evaluation),
        "SimplexeCinqPointGauche": point(gauche.point or ()),
        "SimplexeCinqOptimum": point(optimum),
        "SimplexeCinqValeur": exact(valeur),
    }


def _six() -> dict[str, str]:
    racine = en.noeud(en.SIX)
    premier = en.noeud(en.SIX, ((1, _INF, 3),))
    troisieme = en.noeud(en.SIX, ((1, _INF, 3), (3, _SUP, 6)))
    quatrieme = en.noeud(en.SIX, ((1, _SUP, 4),))
    valeur, optimum = en.optimum_entier(en.SIX)
    variable, _, petite = en.plus_petite_penalite(en.SIX)
    return {
        "SimplexeSixRelaxe": point(racine.point or ()),
        "SimplexeSixEvaluation": exact(racine.evaluation),
        **_penalites("SimplexeSixPenalites", en.SIX),
        "SimplexeSixPetite": exact(petite),
        "SimplexeSixVariable": str(variable),
        "SimplexeSixPointUn": point(premier.point or ()),
        "SimplexeSixEvaluationUn": exact(premier.evaluation),
        "SimplexeSixEvaluationTrois": exact(troisieme.evaluation),
        "SimplexeSixPointTrois": point(troisieme.point or ()),
        "SimplexeSixEvaluationQuatre": exact(quatrieme.evaluation),
        "SimplexeSixPointQuatre": point(quatrieme.point or ()),
        "SimplexeSixOptimum": point(optimum),
        "SimplexeSixValeur": exact(valeur),
    }


def macros_des_entiers() -> dict[str, str]:
    """Les macros des deux exercices de « Branch and Bound »."""
    return {**_cinq(), **_six()}
