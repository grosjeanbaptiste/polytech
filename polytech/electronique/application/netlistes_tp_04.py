"""Les six circuits du TP 4 d'electronique, composant par composant.

Un fichier de donnees : une liste d'elements par circuit, une ligne par
composant, dans l'ordre ou le schema de l'enonce (`TP_4.pdf`) les montre.
Les noeuds et les sens des courants sont ceux de ces schemas.

Un condensateur est une admittance `s C` ; une source commandee
`g_m (V_a - V_b)` est une `SourceCommandee` dont la fleche va de
`depart` vers `arrivee`.
"""

from __future__ import annotations

from polytech.electronique.application.symboles_tp_04 import (
    COL,
    GM,
    GM1,
    GM2,
    I1,
    IS,
    IX,
    IY,
    R1,
    R2,
    R3,
    R4,
    R5,
    REQ,
    RL,
    RO,
    RO1,
    RO2,
    RPI1,
    RPI2,
    RS,
    VS,
    VX,
    VY,
    C,
    R,
    S,
)
from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceCommandee,
    SourceDeCourant,
    SourceDeTension,
)

CIRCUIT_1: list[Element] = [
    SourceDeTension(1, 0, VS, I1),
    Admittance(1, 2, 1 / R1),
    Admittance(2, 0, 1 / R2),
    SourceDeCourant(0, 2, IS),
    Admittance(2, 0, 1 / R3),
]
CIRCUIT_2: list[Element] = [
    SourceDeTension(1, 0, VS, I1),
    Admittance(1, 2, 1 / R1),
    Admittance(2, 0, 1 / R4),
    Admittance(2, 3, 1 / R2),
    Admittance(3, 0, 1 / R5),
    Admittance(3, 4, 1 / R3),
    SourceDeCourant(0, 4, IS),
]
CIRCUIT_3: list[Element] = [
    SourceDeTension(1, 0, VY, IY),
    SourceDeTension(3, 2, VX, IX),
    SourceDeCourant(1, 2, IS),
    Admittance(1, 2, 1 / R1),
    Admittance(2, 0, 1 / R2),
    Admittance(3, 0, 1 / R3),
]
DEPHASEUR: list[Element] = [
    SourceDeTension(1, 0, VS, I1),
    Admittance(1, 2, S * C),
    Admittance(2, 0, 1 / R),
    Admittance(2, 3, S * C),
    Admittance(3, 0, 1 / R),
    Admittance(3, 4, S * C),
    Admittance(4, 0, 1 / R),
    Admittance(4, 0, 1 / RL),
]
SOURCE_COMMUNE: list[Element] = [
    SourceDeTension(1, 0, VS, IS),
    Admittance(1, 2, 1 / RS),
    Admittance(2, 4, S * COL),
    SourceCommandee(4, 3, GM, 2, 3),
    Admittance(4, 3, 1 / RO),
    Admittance(3, 0, 1 / R),
    Admittance(4, 0, 1 / REQ),
]
CASCODE: list[Element] = [
    SourceDeTension(1, 0, VS, IS),
    Admittance(1, 2, 1 / RS),
    Admittance(2, 0, 1 / RPI2),
    SourceCommandee(3, 0, GM2, 2, 0),
    Admittance(3, 0, 1 / RO2),
    Admittance(3, 0, 1 / RPI1),
    SourceCommandee(4, 3, GM1, 0, 3),
    Admittance(4, 3, 1 / RO1),
    Admittance(4, 0, 1 / RL),
]
