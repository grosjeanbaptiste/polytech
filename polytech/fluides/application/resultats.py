"""Modele de lecture (DTO) des resultats de l'activite.

Cote requete du CQRS : une structure plate, faconnee pour l'affichage, et
non les agregats du domaine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Resultats:
    """Tous les nombres et expressions que le corrige doit citer."""

    pression_air_kpa: float
    pression_relative_kpa: float
    contributions_pa: tuple[float, float, float]
    part_mercure_pourcent: float
    sensibilite_mercure_kpa_par_m: float
    hauteur_eau_equivalente_m: float
    hauteur_mercure_equivalente_m: float
    pression_air_niveau_mer_kpa: float
    rapport_poussee_prismatique: float
    rapport_moment_prismatique: float
    rapport_hauteur_vallee_v: float
    rapport_poussee_vallee_v: float
    divergence_b: str
    taux_rotation_b: str
    champ_acceleration_b: tuple[str, str]
    acceleration_b: tuple[float, float]
    norme_acceleration_b: float
    angle_acceleration_b_degres: float
    fonction_de_courant_c: str
    potentiel_c: str
