"""Macros des « exemples resolus », parties I a III. Prefixe `Exemples`."""

from __future__ import annotations

from polytech.optimisation.application import exemples_dualite as du
from polytech.optimisation.application import exemples_formes as fo
from polytech.optimisation.application import exemples_issues as iss
from polytech.optimisation.application import exemples_sommets as so
from polytech.optimisation.application.formatage import exact, point
from polytech.optimisation.application.macros_exemples_entiers import (
    macros_des_entiers,
)

_RANGS = ("Zero", "Un", "Deux")


def _formes() -> dict[str, str]:
    return {
        "ExemplesFormesDedoublees": str(fo.variables(decomposees=True)),
        "ExemplesFormesTranslatees": str(fo.variables(decomposees=False)),
        "ExemplesFormesRayonCent": exact(fo.valeur(fo.point_du_rayon(100))),
        "ExemplesFormesPointCent": point(fo.point_du_rayon(100)),
    }


def _sommets() -> dict[str, str]:
    valeur, optimaux = so.sommets_optimaux(so.ARETE)
    macros = {
        "ExemplesBases": str(so.bases_regulieres(so.PREMIER)),
        "ExemplesBasesSecond": str(so.bases_regulieres(so.SECOND)),
        "ExemplesBasesDuSommet": str(len(so.bases_du_sommet(so.DEGENERE, (2, 0)))),
        "ExemplesAreteValeur": exact(valeur),
        "ExemplesAreteDebut": point(optimaux[0]),
        "ExemplesAreteFin": point(optimaux[1]),
    }
    for rang, (sommet, cout) in zip(_RANGS, so.parcours(), strict=True):
        macros[f"ExemplesUnPoint{rang}"] = point(sommet)
        macros[f"ExemplesUnValeur{rang}"] = exact(cout)
    reduits = so.couts_reduits_finaux()
    macros["ExemplesUnReduitCinq"] = exact(reduits[5])
    macros["ExemplesUnReduitTrois"] = exact(reduits[3])
    return macros


def _issues() -> dict[str, str]:
    valeur, optimum = iss.optimum("I")
    retourne, point_retourne = iss.optimum("I retourne")
    penalise, artificielle = iss.optimum_du_probleme_penalise()
    rayon, sommet = iss.optimum("rayon")
    return {
        "ExemplesUnValeur": exact(valeur),
        "ExemplesUnOptimum": point(optimum),
        "ExemplesRetourneValeur": exact(retourne),
        "ExemplesRetourneOptimum": point(point_retourne),
        "ExemplesPenalise": exact(penalise),
        "ExemplesArtificielle": exact(artificielle),
        "ExemplesRayonValeur": exact(rayon),
        "ExemplesRayonSommet": point(sommet),
    }


def _dualite() -> dict[str, str]:
    gain, quantites = du.optimum_des_bureaux()
    _, prix = du.prix_des_ressources()
    etapes = du.parcours_dual()
    valeur, optimum = du.optimum("dual")
    artificielle, point_artificiel = du.optimum("artificielle I")
    macros = {
        "ExemplesBureauxGain": exact(gain),
        "ExemplesBureauxQuantites": point(quantites),
        "ExemplesBureauxPrix": point(prix),
        "ExemplesBureauxRestes": point(du.restes_des_ressources()),
        "ExemplesBureauxCout": exact(du.cout_en_ressources((1, 1, 4))),
        "ExemplesDualValeur": exact(valeur),
        "ExemplesDualOptimum": point(optimum),
        "ExemplesDualDuales": point(du.duales_par_le_dual()),
        "ExemplesArtificielleValeur": exact(artificielle),
        "ExemplesArtificielleOptimum": point(point_artificiel),
    }
    for rang, (cout, _) in zip(_RANGS, etapes, strict=True):
        macros[f"ExemplesDualEtape{rang}"] = exact(cout)
    return macros


def macros_des_exemples() -> dict[str, str]:
    """Toutes les valeurs du document des exemples resolus."""
    return {
        **_formes(),
        **_sommets(),
        **_issues(),
        **_dualite(),
        **macros_des_entiers(),
    }
