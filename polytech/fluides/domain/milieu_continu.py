"""L'hypothese du milieu continu, et le nombre de Knudsen.

Chapitre I.1 des planches. Un litre d'air compte de l'ordre de `10^23`
molecules : suivre chacune est hors de portee, et l'on construit un
modele ou la matiere est continue. La masse volumique y est la limite de
`delta m / delta V` quand le volume tend vers un volume `delta V*` -- un
micron cube environ -- assez grand pour lisser les fluctuations
moleculaires, assez petit pour rester local.

Le modele vaut tant que le libre parcours moyen `l` des molecules est
petit devant l'echelle `L` des phenomenes : leur rapport est le nombre de
Knudsen `Kn = l / L`, et le gaz est dit rarefie quand `Kn >= 1`.
"""

from __future__ import annotations

from typing import Final

SEUIL_DE_RAREFACTION: Final = 1.0
"""`Kn >= 1` : gaz rarefie, le modele continu ne vaut plus."""


def nombre_de_knudsen(libre_parcours: float, echelle: float) -> float:
    """`Kn = l / L`, sans dimension."""
    if echelle <= 0:
        msg = f"echelle nulle ou negative : {echelle} m"
        raise ValueError(msg)
    if libre_parcours < 0:
        msg = f"libre parcours negatif : {libre_parcours} m"
        raise ValueError(msg)
    return libre_parcours / echelle


def est_rarefie(knudsen: float) -> bool:
    """Vrai si `Kn >= 1`."""
    return knudsen >= SEUIL_DE_RAREFACTION
