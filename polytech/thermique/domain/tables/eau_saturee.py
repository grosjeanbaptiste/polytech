"""Les proprietes de l'eau liquide sur la ligne de saturation.

Syllabus de thermique, tome 2, table 3.1 (« UK Steam Tables in SI
Units », Arnold, 1970), colonnes de l'eau, de 0,01 a 100 degres. Les
unites de la table -- bar, kJ/(kg K), 1e-3 N/m, 1e-6 N s/m^2, 1e-6 m^2/s
-- sont ramenees au systeme international.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from polytech.thermique.domain.tables.interpolation import interpoler


@dataclass(frozen=True, slots=True)
class EauSaturee:
    """Une ligne : `t` en degres, `p_s`, `c_p`, `sigma`, `mu`, `nu`, `lambda`, `Pr`."""

    temperature: float
    pression: float
    chaleur_massique: float
    tension: float
    viscosite: float
    viscosite_cinematique: float
    conductivite: float
    prandtl: float


TABLE_DE_L_EAU_SATUREE: Final = (
    EauSaturee(0.01, 611.0, 4218.0, 75.60e-3, 1786e-6, 1.786e-6, 0.569, 13.2),
    EauSaturee(10, 1227.0, 4194.0, 74.24e-3, 1304e-6, 1.305e-6, 0.587, 9.32),
    EauSaturee(20, 2337.0, 4182.0, 72.78e-3, 1002e-6, 1.004e-6, 0.603, 6.95),
    EauSaturee(30, 4241.0, 4179.0, 71.23e-3, 798.3e-6, 0.802e-6, 0.618, 5.40),
    EauSaturee(40, 7375.0, 4179.0, 69.61e-3, 653.9e-6, 0.659e-6, 0.631, 4.33),
    EauSaturee(50, 12335.0, 4181.0, 67.93e-3, 547.8e-6, 0.554e-6, 0.643, 3.56),
    EauSaturee(60, 19920.0, 4185.0, 66.19e-3, 467.3e-6, 0.473e-6, 0.653, 2.99),
    EauSaturee(70, 31162.0, 4191.0, 64.40e-3, 404.8e-6, 0.414e-6, 0.662, 2.56),
    EauSaturee(80, 47360.0, 4198.0, 62.57e-3, 355.4e-6, 0.366e-6, 0.670, 2.23),
    EauSaturee(90, 70109.0, 4207.0, 60.69e-3, 315.6e-6, 0.327e-6, 0.676, 1.96),
    EauSaturee(100, 101330.0, 4218.0, 58.78e-3, 283.1e-6, 0.295e-6, 0.681, 1.75),
)


def eau_saturee(temperature: float) -> EauSaturee:
    """L'eau liquide saturee a cette temperature, en degres Celsius."""
    return interpoler(TABLE_DE_L_EAU_SATUREE, temperature)
