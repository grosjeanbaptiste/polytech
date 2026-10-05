"""La realisation d'un aimant permanent : un tore et son entrefer.

Le syllabus d'electricite generale (12.1). Un tore aimante a saturation,
dont on retire le bouchon : le theoreme d'Ampere donne
`H l + H_e l_e = 0` (12.1.1) et la loi d'Hopkinson
`H_e l_e = R_e B S` (12.1.2). Le champ dans le materiau est donc
demagnetisant, `H = -R_e S B / l` (12.1.3), et l'etat magnetique se place
a l'intersection de la courbe de desaimantation et de la droite `d`,
`B = -l H / (R_e S)` (12.1.4).

Des pieces polaires reduisent la reluctance de l'entrefer : la droite se
redresse, et l'induction remonte.

La courbe de desaimantation est ici la droite qui joint `B_r` au champ
coercitif, modele de `aimants`.
"""

from __future__ import annotations


def pente_de_la_droite_de_charge(
    longueur: float, section: float, reluctance_entrefer: float
) -> float:
    """`-l / (R_e S)` (12.1.4), en henrys par metre : negative."""
    if longueur <= 0 or section <= 0 or reluctance_entrefer <= 0:
        msg = (
            f"geometrie invalide : longueur {longueur} m, section {section} m^2, "
            f"reluctance d'entrefer {reluctance_entrefer} A/Wb"
        )
        raise ValueError(msg)
    return -longueur / (reluctance_entrefer * section)


def champ_dans_l_aimant(
    induction: float, longueur: float, section: float, reluctance_entrefer: float
) -> float:
    """`H = -R_e S B / l` (12.1.3), en amperes par metre."""
    pente = pente_de_la_droite_de_charge(longueur, section, reluctance_entrefer)
    return induction / pente


def point_de_fonctionnement(
    remanente: float, coercitif: float, pente: float
) -> tuple[float, float]:
    """`(H, B)` au point 3 : la droite `d` coupe `B = B_r (1 + H / H_c)`."""
    if pente >= 0:
        msg = f"pente positive ou nulle : {pente} H/m -- pas une droite de charge"
        raise ValueError(msg)
    champ = remanente / (pente - remanente / coercitif)
    return champ, pente * champ
