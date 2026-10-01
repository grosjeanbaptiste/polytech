"""Macros de l'exercice 2 : le reservoir d'hydrogene liquide.

C'est l'exercice ou le corrige officiel ne donne aucun chiffre -- sa
planche de valeurs numeriques est une image dont rien ne s'extrait. Les
deux chemins de calcul sont donc publies tous les deux, pour que le
lecteur voie qu'ils tombent sur le meme watt.
"""

from __future__ import annotations

from polytech.thermique.application import enonce
from polytech.thermique.application.formatage import (
    expression_latex,
    nombre,
    scientifique,
)

_SECONDES_PAR_JOUR = 86_400


def macros_reservoir() -> dict[str, str]:
    """Les resistances, le flux par deux voies, et le debit evapore."""
    reservoir = enonce.RESERVOIR
    coque = reservoir.coque.resistance().kelvins_par_watt
    film = reservoir.resistance_exterieure().kelvins_par_watt
    totale = reservoir.resistance_totale().kelvins_par_watt
    debit = reservoir.debit_evapore()
    return {
        "SurfaceExterieure": nombre(reservoir.coque.exterieur.surface_de_sphere, 4),
        "ResistanceCoque": nombre(coque, 4),
        "ResistanceFilm": nombre(film, 5),
        "ResistanceTotale": nombre(totale, 4),
        "PartResistanceCoque": nombre(100.0 * coque / totale, 2),
        "EcartMoteur": nombre(reservoir.ecart_moteur(), 0),
        "FluxReservoir": nombre(reservoir.flux(), 3),
        "FluxReservoirDirect": nombre(reservoir.flux_par_resolution_directe(), 3),
        "TemperatureParoiExterieure": nombre(
            reservoir.temperature_de_paroi_exterieure().kelvins,
            2,
        ),
        "DebitEvapore": scientifique(debit, 3),
        "DebitEvaporeParJour": nombre(debit * _SECONDES_PAR_JOUR, 3),
        "ProfilSpherique": expression_latex(reservoir.coque.profil_general()),
    }
