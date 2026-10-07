"""Macros des « exemples resolus », « Branch and Bound » et « lot-sizing »."""

from __future__ import annotations

from polytech.optimisation.application import exemples_entiers as en
from polytech.optimisation.application import exemples_lots as lo
from polytech.optimisation.application import exemples_lots_plans as pl
from polytech.optimisation.application.formatage import arrondi, exact, point
from polytech.optimisation.domain.contraintes import Sens

_INF, _SUP = Sens.INFERIEUR, Sens.SUPERIEUR
_NOMS = {2: "Deux", 3: "Trois"}
_ECRITURES = {"I": "Un", "II": "Deux", "III": "Trois", "IV": "Quatre"}


def _noeuds() -> dict[str, str]:
    noeuds = {
        "Racine": en.noeud(),
        "Bas": en.noeud((3, _INF, 1)),
        "Haut": en.noeud((3, _SUP, 2)),
        "Un": en.noeud((2, _SUP, 4)),
        "Trois": en.noeud((2, _SUP, 4), (3, _SUP, 2)),
        "Quatre": en.noeud((2, _INF, 3)),
    }
    macros = {}
    for nom, noeud in noeuds.items():
        macros[f"ExemplesArbrePoint{nom}"] = point(noeud.point or ())
        macros[f"ExemplesArbreEvaluation{nom}"] = exact(noeud.evaluation)
    return macros


def _illustration() -> dict[str, str]:
    valeur, optimum = en.optimum_entier()
    variable, _, grande = en.plus_grande_penalite()
    autre, _, petite = en.plus_petite_penalite()
    macros = {
        f"ExemplesArbrePenalites{_NOMS[j]}": point(couple)
        for j, couple in en.penalites_a_la_racine().items()
    }
    return {
        **macros,
        **_noeuds(),
        "ExemplesArbreGrande": exact(grande),
        "ExemplesArbreVariableGrande": str(variable),
        "ExemplesArbrePetite": exact(petite),
        "ExemplesArbreVariablePetite": str(autre),
        "ExemplesArbreValeur": exact(valeur),
        "ExemplesArbreOptimum": point(optimum),
        "ExemplesArbreMeilleurQuatre": exact(en.meilleur_point_entier((2, _INF, 3))[0]),
    }


def _autre() -> dict[str, str]:
    (continu, point_continu), (entier, point_entier) = en.autre_exemple()
    bornes, (_, second_membre) = en.resserree((2, -1, 5), 7)
    return {
        "ExemplesAutreContinu": exact(continu),
        "ExemplesAutrePointContinu": point(point_continu),
        "ExemplesAutreEntier": exact(entier),
        "ExemplesAutrePointEntier": point(point_entier),
        "ExemplesAutreArrondi": exact(en.valeur_dans_l_autre((1, 3))),
        "ExemplesPretraitementPlafond": str(bornes[0][1]),
        "ExemplesPretraitementFixee": str(bornes[2][1]),
        "ExemplesPretraitementReste": exact(second_membre),
    }


def _lots() -> dict[str, str]:
    valeur, ouverts, production = pl.optimum_par_enumeration()
    macros = {
        "ExemplesLotsValeur": exact(valeur),
        "ExemplesLotsOuverts": point(ouverts),
        "ExemplesLotsProduction": point(production),
        "ExemplesLotsRestes": point(lo.RESTES),
    }
    for nom, suffixe in _ECRITURES.items():
        relaxation = lo.relaxation(nom)
        macros[f"ExemplesLotsRelaxation{suffixe}"] = exact(relaxation)
        if not relaxation.is_integer:
            macros[f"ExemplesLotsDecimale{suffixe}"] = arrondi(relaxation, 2)
        macros[f"ExemplesLotsOuvertures{suffixe}"] = point(
            lo.ouvertures_de_la_relaxation(nom)
        )
    for nom in ("I", "II", "III"):
        macros[f"ExemplesLotsSommets{_ECRITURES[nom]}"] = str(lo.sommets_explores(nom))
    return macros


def macros_des_entiers() -> dict[str, str]:
    """Les valeurs des planches 63 a 85."""
    return {**_illustration(), **_autre(), **_lots()}
