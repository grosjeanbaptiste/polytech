"""Les transformations d'un gaz parfait : premier principe, isochore, isobare.

Source : `Thermodynamique/EXAMENS/Exam_2023_Novembre_Exercice_cote_*` (2 kg
d'air, `c_V = 718 J/(kg K)`, travail de detente de 800 kJ). Les
temperatures se donnent en degres Celsius ; la conversion en kelvins, la
seule qui compte pour un rapport de pressions, passe par la constante du
depot (`thermique.domain.grandeurs.ZERO_ABSOLU`).
"""

from __future__ import annotations

from polytech.thermique.domain.grandeurs import ZERO_ABSOLU


def en_kelvin(celsius: float) -> float:
    """`T = t + 273,15`."""
    return celsius + ZERO_ABSOLU


def capacite_isochore(masse: float, chaleur_massique: float) -> float:
    """`m c_V`, en joules par kelvin : la chaleur d'un degre a volume constant."""
    return masse * chaleur_massique


def chaleur_isochore(
    masse: float,
    chaleur_massique_a_volume_constant: float,
    depart: float,
    arrivee: float,
) -> float:
    """`Q = m c_V (t_arrivee - t_depart)`, en joules ; `W = 0` a volume constant."""
    capacite = capacite_isochore(masse, chaleur_massique_a_volume_constant)
    return capacite * (arrivee - depart)


def temperature_apres_detente_adiabatique(
    depart: float,
    masse: float,
    chaleur_massique_a_volume_constant: float,
    travail_du_gaz: float,
) -> float:
    """Premier principe sans chaleur : `W = m c_V (t_depart - t_fin)`, en Celsius."""
    capacite = capacite_isochore(masse, chaleur_massique_a_volume_constant)
    return depart - travail_du_gaz / capacite


def rapport_des_pressions_a_volume_constant(depart: float, arrivee: float) -> float:
    """`p_arrivee / p_depart = T_arrivee / T_depart`, temperatures en Celsius."""
    return en_kelvin(arrivee) / en_kelvin(depart)
