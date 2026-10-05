"""L'effet Hall dans un conducteur parallelipipedique.

Le syllabus d'electricite generale (13.7). Un conducteur d'epaisseur `d`
et de largeur `h`, parcouru par un courant `I` et plonge dans une
induction `B` normale a sa grande face : la force magnetique accumule les
porteurs sur une face laterale, jusqu'a ce qu'un champ transversal
l'equilibre, `q v B = q U_H / h` (13.7.1). Comme `I = -h d n q v`
(13.7.2), la largeur s'elimine : `U_H = R_H I B / d` (13.7.3), avec la
constante de Hall `R_H = -1 / (n q)` (13.7.4).

Le signe est celui du syllabus, dont la charge `q` des electrons est
negative : `R_H` y est positive pour un metal, et negative quand la
conduction se fait par des « trous ». Les porteurs sont donnes par leur
charge volumique `n q`.
"""

from __future__ import annotations


def _exiger_des_porteurs(charge_volumique: float) -> None:
    if charge_volumique == 0:
        msg = "aucun porteur : charge volumique n q nulle"
        raise ValueError(msg)


def constante_de_hall(charge_volumique: float) -> float:
    """`R_H = -1 / (n q)` (13.7.4), en metres cubes par coulomb."""
    _exiger_des_porteurs(charge_volumique)
    return -1 / charge_volumique


def tension_de_hall(
    constante: float, courant: float, induction: float, epaisseur: float
) -> float:
    """`U_H = R_H I B / d` (13.7.3), en volts."""
    if epaisseur <= 0:
        msg = f"epaisseur nulle ou negative : {epaisseur} m"
        raise ValueError(msg)
    return constante * courant * induction / epaisseur


def vitesse_des_porteurs(
    courant: float, largeur: float, epaisseur: float, charge_volumique: float
) -> float:
    """`v = -I / (h d n q)` (13.7.2), en metres par seconde."""
    if largeur <= 0 or epaisseur <= 0:
        msg = f"section nulle ou negative : {largeur} m par {epaisseur} m"
        raise ValueError(msg)
    _exiger_des_porteurs(charge_volumique)
    return -courant / (largeur * epaisseur * charge_volumique)
