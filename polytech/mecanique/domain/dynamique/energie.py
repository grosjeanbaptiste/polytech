"""Le travail et l'energie mecanique d'une particule.

Source : syllabus « Notions fondamentales » de « Physique generale »,
chapitre 6 : (6.1) page 62, (6.24) page 68, (6.32) page 71, (6.37) page 72.
La vitesse par l'energie se calcule par les energies seules ; qu'elle soit
celle de la cinematique a acceleration -g (2.15), c'est
`tests/test_mecanique_energie.py` qui le verifie.
"""

from __future__ import annotations

import math
from typing import Any


def travail(force: Any, longueur: Any, angle: Any) -> Any:  # noqa: ANN401 -- nombre ou expression sympy
    """(6.1) W = F l cos(angle) : une force constante sur un deplacement rectiligne."""
    if isinstance(angle, (int, float)):
        return force * longueur * math.cos(angle)
    import sympy as sp  # noqa: PLC0415 -- le chemin symbolique

    return force * longueur * sp.cos(angle)


def energie_cinetique(masse: Any, vitesse: Any) -> Any:  # noqa: ANN401 -- nombre ou expression sympy
    """(6.24) Ec = m v^2 / 2."""
    return masse * vitesse**2 / 2


def energie_potentielle_pesanteur(masse: Any, g: Any, hauteur: Any) -> Any:  # noqa: ANN401 -- nombre ou expression sympy
    """(6.32) Ep = m g y, la hauteur comptee depuis l'origine choisie."""
    return masse * g * hauteur


def vitesse_par_l_energie(vi: Any, yi: Any, yf: Any, g: Any) -> Any:  # noqa: ANN401 -- nombre ou expression sympy
    """(6.37) m g yi + m vi^2 / 2 = m g yf + m vf^2 / 2, resolue pour |vf|.

    Calculee par les energies elles-memes (Ec_f = Ec_i + Ep_i - Ep_f, la
    masse se simplifie : on la prend unitaire), et non par la cinematique :
    l'accord avec (2.15) est un test de lien, pas une recopie. Leve si yf est
    trop haute pour l'energie disponible.
    """
    finale = energie_cinetique(1, vi) + energie_potentielle_pesanteur(1, g, yi)
    finale -= energie_potentielle_pesanteur(1, g, yf)
    if isinstance(finale, (int, float)):
        if finale < 0:
            msg = (
                f"hauteur {yf} jamais atteinte : energie cinetique finale {finale} < 0"
            )
            raise ValueError(msg)
        return math.sqrt(2 * finale)
    import sympy as sp  # noqa: PLC0415 -- le chemin symbolique

    return sp.sqrt(2 * finale)
