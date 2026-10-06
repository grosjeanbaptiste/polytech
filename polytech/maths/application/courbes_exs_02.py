"""Mathematique, exercices 2 : les courbes des figures du corrige.

Chaque courbe est l'echantillonnage d'une solution que les modules des
seances etablissent ; rien n'est integre numeriquement ici.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import sympy as sp

from polytech.maths.application import modeles_exs_02 as mo
from polytech.maths.application import reponses_exs_02 as re
from polytech.maths.application import systemes_exs_02 as sy
from polytech.maths.application.formatage import nombre
from polytech.maths.application.resolue_exs_02 import X, Y
from polytech.maths.domain.symboles import TEMPS as T

Courbe = tuple[np.ndarray, np.ndarray]
POINTS: Final = 81
PULSATION_VOISINE: Final = 0.9
"""Une pulsation proche de la resonance, pour montrer les battements."""
RAYON: Final = 2.8
"""Le rayon du cercle d'ou partent les trajectoires du portrait de phase."""
TRAJECTOIRES: Final = 12


def _echantillon(expression: sp.Expr, temps: np.ndarray) -> np.ndarray:
    valeurs = sp.lambdify(T, expression, "numpy")(temps)
    return np.broadcast_to(np.asarray(valeurs, dtype=float), temps.shape).copy()


def courbes_du_champ() -> dict[int, Courbe]:
    """I.2 : quatre courbes integrales dans le cadre de l'enonce."""
    temps = np.linspace(-4.0, 4.0, POINTS)
    return {
        depart: (temps, _echantillon(mo.solution_du_champ(depart), temps))
        for depart in (-3, -1, 1, 3)
    }


def courbes_logistiques() -> dict[float, Courbe]:
    """I.3 : `K = lambda = 1`, trois populations initiales."""
    temps = np.linspace(0.0, 6.0, POINTS)
    unites = {mo.CAPACITE: 1, mo.TAUX: 1}
    solution = mo.logistique_resolue()[2].subs(unites)
    return {
        depart: (temps, _echantillon(solution.subs(mo.DEPART, depart), temps))
        for depart in (0.1, 0.5, 2.5)
    }


def courbes_du_ressort() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """I.5 : parti du repos, pres de la resonance puis a la resonance."""
    temps = np.linspace(0.0, 70.0, 4 * POINTS)
    du_repos, a_la_resonance = (r.subs(X, T) for r in mo.reponses_depuis_le_repos())
    voisine = du_repos.subs(mo.PULSATION, sp.Float(PULSATION_VOISINE))
    return temps, _echantillon(voisine, temps), _echantillon(a_la_resonance, temps)


def courbes_des_creneaux() -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    """II.6 : la reponse au creneau pour trois hauteurs, et sa limite."""
    temps = np.linspace(0.0, 4.0, POINTS)
    pendant, apres = re.au_creneau()
    creneaux = {}
    for hauteur in (1, 2, 8):
        avant = _echantillon(pendant.subs(re.HAUTEUR, hauteur), temps)
        ensuite = _echantillon(apres.subs(re.HAUTEUR, hauteur), temps)
        creneaux[hauteur] = np.where(temps <= 1 / hauteur, avant, ensuite)
    return temps, creneaux, _echantillon(re.impulsionnelle(), temps)


def courbes_du_courant() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """II.7 : le courant pour `u = sin t`, et son regime etabli."""
    temps = np.linspace(0.0, 15.0, 2 * POINTS)
    courant = re.courants()[2]
    return (
        temps,
        _echantillon(courant, temps),
        _echantillon(re.regime_etabli()[1], temps),
    )


def trajectoires_de_phase() -> tuple[Courbe, ...]:
    """III.1 : des trajectoires du plan `(y1, y2)`, parties d'un cercle."""
    temps = np.linspace(0.0, 6.0, POINTS)
    courbes = []
    for rang in range(TRAJECTOIRES):
        angle = 2 * sp.pi * rang / TRAJECTOIRES
        depart = sp.Matrix([RAYON * sp.cos(angle), RAYON * sp.sin(angle)])
        trajectoire = sy.trajectoire_depuis(depart)
        courbes.append(
            (_echantillon(trajectoire[0], temps), _echantillon(trajectoire[1], temps))
        )
    return tuple(courbes)


Segment = tuple[tuple[float, float], tuple[float, float]]
LONGUEUR: Final = 0.5
"""La longueur d'un segment du champ de pentes."""


def segments_du_champ() -> tuple[Segment, ...]:
    """I.2 : un segment de pente `f(y)` en chaque noeud entier du cadre."""
    pente = sp.lambdify(Y, mo.champ().second_membre, "math")
    segments = []
    for abscisse in range(-4, 5):
        for ordonnee in range(-4, 5):
            angle = math.atan(pente(ordonnee))
            dx, dy = LONGUEUR / 2 * math.cos(angle), LONGUEUR / 2 * math.sin(angle)
            segments.append(
                ((abscisse - dx, ordonnee - dy), (abscisse + dx, ordonnee + dy))
            )
    return tuple(segments)


def en_traits_separes(*segments: Segment) -> str:
    """Des segments en une liste de coordonnees, une rupture entre deux traits."""
    traits = (
        f"({nombre(x1, 3)},{nombre(y1, 3)}) ({nombre(x2, 3)},{nombre(y2, 3)})"
        for (x1, y1), (x2, y2) in segments
    )
    return " (nan,nan) ".join(traits)
