"""Le gaz parfait : `p V = n R T`, et ce que la masse molaire en fait.

`R` est la constante **molaire** des gaz, la meme pour tous. Rapportee a
la masse, la loi s'ecrit `p = rho (R / M) T` : `R / M` est la constante
**specifique** du gaz, celle que les cours de mecanique des fluides et de
thermique notent simplement `R` dans `P = rho R T`. La masse molaire `M`
est donc ce qui relie l'ecriture du chimiste a celle du mecanicien.

Sources : `MecaniqueFluides/SLIDES/Slides_01_Introduction_2025_2026.pdf`
(I.2, « gaz parfait P = rho R T ») et
`ThermiqueFondements/SYLLABUS/Syllabus_Tome_2_2013.pdf` (gaz parfaits).
"""

from __future__ import annotations

from typing import Final

CONSTANTE_DES_GAZ: Final = 8.31446261815324
"""`R = N_A k_B`, en J/(mol K) ; exacte depuis la redefinition du SI de 2019."""

AIR_SEC: Final = 0.02896
"""Masse molaire moyenne de l'air sec, en kg/mol (valeur normalisee)."""


def constante_specifique(masse_molaire: float) -> float:
    """`R / M`, en J/(kg K) : le `R` de `P = rho R T`."""
    if masse_molaire <= 0:
        msg = f"masse molaire non strictement positive : {masse_molaire}"
        raise ValueError(msg)
    return CONSTANTE_DES_GAZ / masse_molaire


def masse_volumique_du_gaz_parfait(
    pression: float,
    temperature: float,
    masse_molaire: float,
) -> float:
    """`rho = p M / (R T)`, en kg/m3 ; pression en Pa, temperature en K."""
    if temperature <= 0:
        msg = f"temperature non strictement positive : {temperature}"
        raise ValueError(msg)
    return pression / (constante_specifique(masse_molaire) * temperature)


def quantite_de_gaz(pression: float, volume: float, temperature: float) -> float:
    """`n = P V / (R T)`, en moles ; pression en Pa, volume en m3, temperature en K."""
    if temperature <= 0:
        msg = f"temperature non strictement positive : {temperature}"
        raise ValueError(msg)
    if volume < 0:
        msg = f"volume negatif : {volume} m3"
        raise ValueError(msg)
    return pression * volume / (CONSTANTE_DES_GAZ * temperature)


def pression_partielle(fraction_molaire: float, pression_totale: float) -> float:
    """Loi de Dalton : `p_i = x_i p`, dans l'unite de la pression totale."""
    if not 0 <= fraction_molaire <= 1:
        msg = f"fraction molaire hors de [0, 1] : {fraction_molaire}"
        raise ValueError(msg)
    if pression_totale < 0:
        msg = f"pression totale negative : {pression_totale}"
        raise ValueError(msg)
    return fraction_molaire * pression_totale
