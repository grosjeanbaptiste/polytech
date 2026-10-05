"""Les signaux echantillonnes du TP 1 : portes, peignes, convolution.

Le TP echantillonne a 1 Hz : un echantillon par seconde, donc des suites
d'entiers ou les durees se comptent en echantillons. C'est ce qui permet
a `scipy.signal.convolve` de verifier une reponse trouvee au crayon.

La convolution a un inverse, la **deconvolution**, qui repond a la
question du TP sans rien deviner : elle lit `b` dans `c = a * b`. Elle
leve si le reste n'est pas nul, car `c` n'est alors le produit de `a`
par aucune suite finie, et rendre le quotient serait mentir.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

TOLERANCE_DU_RESTE = 1e-9
"""Au-dela, le reste n'est pas un arrondi : la division n'est pas exacte."""


def porte(largeur: int, total: int) -> np.ndarray:
    """`largeur` uns suivis de zeros, sur `total` echantillons."""
    if not 0 < largeur <= total:
        msg = f"largeur {largeur} hors de [1, {total}]"
        raise ValueError(msg)
    return np.concatenate((np.ones(largeur), np.zeros(total - largeur)))


def peigne(periode: int, impulsions: int) -> np.ndarray:
    """`impulsions` impulsions unite, espacees de `periode` echantillons."""
    if impulsions < 1 or periode < 1:
        msg = f"peigne sans impulsion ou sans periode : {impulsions}, {periode}"
        raise ValueError(msg)
    suite = np.zeros((impulsions - 1) * periode + 1)
    suite[::periode] = 1.0
    return suite


def convoluer(premier: np.ndarray, second: np.ndarray) -> np.ndarray:
    """`premier * second`, comme `scipy.signal.convolve` le calcule."""
    return np.asarray(signal.convolve(premier, second), dtype=float)


def deconvoluer(produit: np.ndarray, facteur: np.ndarray) -> np.ndarray:
    """La suite `b` telle que `facteur * b = produit` ; leve si elle n'existe pas."""
    quotient, reste = signal.deconvolve(produit, facteur)
    if np.max(np.abs(reste)) > TOLERANCE_DU_RESTE:
        msg = f"reste non nul ({np.max(np.abs(reste)):.3g}) : pas un produit exact"
        raise ValueError(msg)
    return np.asarray(quotient, dtype=float)


def reponse_par_superposition(
    entree: np.ndarray, impulsionnelle: np.ndarray, pas: float
) -> np.ndarray:
    """`y(t) ~ somme x(t_k) h(t - t_k) Delta` : la convolution, discretisee."""
    if pas <= 0:
        msg = f"pas nul ou negatif : {pas} s"
        raise ValueError(msg)
    return convoluer(entree, impulsionnelle) * pas
