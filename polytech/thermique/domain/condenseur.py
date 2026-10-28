"""Le bilan d'un condenseur a tubes horizontaux (exercice III.5.2).

Le debit de vapeur se partage entre les colonnes de tubes ; le debit par
metre de colonne pilote le coefficient d'echange, et le bilan global
`G l_v = K S (t_sat - t_eau)` donne la saturation.
"""

from __future__ import annotations

import math


def surface_des_tubes(tubes: int, longueur: float, diametre: float) -> float:
    """`n L pi d`, la surface exterieure des tubes, en m^2."""
    return tubes * longueur * math.pi * diametre


def debit_par_metre_de_colonne(
    debit: float, tubes: int, tubes_par_colonne: int, longueur: float
) -> float:
    """`G / (L n / n_colonne)`, en kg/(s m), sur les colonnes de tubes."""
    return debit / (tubes / tubes_par_colonne * longueur)


def temperature_de_saturation(
    entree: float,
    debit: float,
    chaleur_latente: float,
    surface: float,
    echange: float,
) -> float:
    """`t_e + G l_v / (K S)` : la vapeur cede `G l_v` a travers `K S`."""
    return entree + debit * chaleur_latente / (surface * echange)
