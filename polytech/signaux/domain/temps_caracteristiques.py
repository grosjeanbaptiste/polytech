"""Les deux temps qui caracterisent une reponse indicielle.

Syllabus de signaux et systemes, section 5.2.3. Le temps de montee
separe le passage a 10 % de la valeur de regime du premier passage a
90 %. Le temps d'etablissement a 5 % est celui au bout duquel la sortie
reste entre 95 % et 105 % de sa valeur finale.

Tous deux se lisent ici sur une reponse echantillonnee, comme on les
lirait sur un releve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

BAS, HAUT = 0.1, 0.9
"""Les deux seuils du temps de montee, en fraction du regime."""


def _premier_passage(
    instants: NDArray[np.float64], reduite: NDArray[np.float64], seuil: float
) -> float:
    atteints = np.nonzero(reduite >= seuil)[0]
    if len(atteints) == 0:
        msg = f"la reponse n'atteint pas {seuil:.0%} de son regime"
        raise ValueError(msg)
    return float(instants[atteints[0]])


def temps_de_montee(
    instants: NDArray[np.float64], reponse: NDArray[np.float64], *, regime: float
) -> float:
    """De 10 % au premier passage a 90 % du regime."""
    reduite = reponse / regime
    bas = _premier_passage(instants, reduite, BAS)
    return _premier_passage(instants, reduite, HAUT) - bas


def temps_d_etablissement(
    instants: NDArray[np.float64],
    reponse: NDArray[np.float64],
    *,
    regime: float,
    tolerance: float = 0.05,
) -> float:
    """L'instant apres lequel la reponse reste dans la bande autour du regime."""
    if regime == 0:
        msg = "regime nul : la bande de tolerance n'a pas de largeur"
        raise ValueError(msg)
    dehors = np.nonzero(np.abs(reponse / regime - 1) > tolerance)[0]
    if len(dehors) == 0:
        return float(instants[0])
    return float(instants[min(dehors[-1] + 1, len(instants) - 1)])
