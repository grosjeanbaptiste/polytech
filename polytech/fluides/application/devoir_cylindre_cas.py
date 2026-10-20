"""Les cas de l'etude parametrique : trois cylindres que le cours chiffre.

L'enonce cite des applications -- pylones, cheminees, tubes d'echangeurs,
piles de pont -- sans les chiffrer, et le cours de fluides n'en chiffre
aucune. Le cours de thermique, lui, pose trois cylindres dans le vent, et
ce sont eux qui sont etudies, avec les donnees de leur enonce :

- une cheminee de four, `de = 0,5 m`, « vent horizontal de 5 m/s »,
  air a -10 degres (examen d'aout 2016) ;
- une cheminee d'usine, `De = Di + 2 e = 1,040 m`, « vitesse du vent de
  5 m/s », air a -10 degres (examens 2022 a 2025) ;
- une serre en demi-cylindre, « cercles de rayon R = 2,5 m », vent de
  0,75 m/s, air a 25 degres ; l'enonce assimile l'ecoulement exterieur a
  celui autour d'un cylindre (examen de janvier 2026).

Les proprietes de l'air a ces temperatures sont celles de `air_a`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from polytech.fluides.application.devoir_cylindre_etude import Point
from polytech.fluides.application.devoir_cylindre_fluides import air_a

_EXAMENS: Final = "ThermiqueFondements/EXAMENS/"


@dataclass(frozen=True, slots=True)
class CasDEtude:
    """Un cas nomme, et le document qui le chiffre."""

    cle: str
    nom: str
    point: Point
    source: str


CAS: Final = (
    CasDEtude(
        "ChemineeFour",
        "cheminée de four",
        Point(air_a(-10.0), 5.0, 0.5),
        _EXAMENS + "Exam_2016_Aout_Exercice_cote_Enonce.pdf",
    ),
    CasDEtude(
        "ChemineeUsine",
        "cheminée d'usine",
        Point(air_a(-10.0), 5.0, 1.04),
        _EXAMENS + "Exam_2022_2025_Exercices_cotes_resolus.pdf",
    ),
    CasDEtude(
        "Serre",
        "serre en demi-cylindre",
        Point(air_a(25.0), 0.75, 5.0),
        _EXAMENS + "Exam_2026_Janvier.pdf",
    ),
)


def cas(cle: str) -> CasDEtude:
    """Le cas de cette cle ; leve s'il n'existe pas."""
    for candidat in CAS:
        if candidat.cle == cle:
            return candidat
    msg = f"aucun cas d'etude {cle!r}"
    raise KeyError(msg)
