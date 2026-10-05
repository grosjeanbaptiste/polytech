"""La transformee de Fourier d'une porte, predite puis estimee.

Prediction : `rect(t) -> sinc(f)`, et le changement d'echelle donne
`rect(t/T) -> T sinc(T f)`. Estimation : `freqz` sur la porte
echantillonnee, qui rend la transformee **discrete** -- une somme et non
une integrale. Multipliee par la periode d'echantillonnage `1/f_e`, elle
approche la transformee continue : c'est la methode des rectangles.

Les bords portent la moitie de la porte. `rect` vaut un demi en
`|t| = T/2`, et c'est aussi ce qui rend exacte la methode des trapezes ;
compter un plein echantillon aux deux bords ajouterait `1/f_e` a l'aire,
donc a la valeur en zero du spectre.

Les zeros ajoutes ne changent pas la transformee ; ils resserrent les
frequences ou `freqz` l'evalue, et rendent visibles les lobes.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def porte_echantillonnee(
    duree: float,
    frequence_d_echantillonnage: float,
    zeros_ajoutes: int,
) -> np.ndarray:
    """`rect(t/T)` echantillonnee, bords a un demi, puis des zeros."""
    if frequence_d_echantillonnage <= 0:
        msg = (
            f"frequence d'echantillonnage non positive : {frequence_d_echantillonnage}"
        )
        raise ValueError(msg)
    demi = round(duree * frequence_d_echantillonnage / 2)
    porte = np.ones(2 * demi + 1)
    porte[[0, -1]] = 0.5
    return np.concatenate((porte, np.zeros(zeros_ajoutes)))


def transformee_de_la_porte(duree: float, frequences: np.ndarray) -> np.ndarray:
    """`|T sinc(T f)|`, avec `sinc(x) = sin(pi x) / (pi x)`."""
    return np.asarray(duree * np.abs(np.sinc(duree * frequences)), dtype=float)


def spectre_estime(
    echantillons: np.ndarray,
    frequence_d_echantillonnage: float,
    frequences: np.ndarray,
) -> np.ndarray:
    """`|freqz| / f_e` : la transformee discrete ramenee a la continue."""
    _, reponse = signal.freqz(
        echantillons, worN=frequences, fs=frequence_d_echantillonnage
    )
    return np.asarray(np.abs(reponse) / frequence_d_echantillonnage, dtype=float)


def raies_estimees(
    echantillons: np.ndarray,
    frequence_d_echantillonnage: float,
    frequences: np.ndarray,
) -> np.ndarray:
    """`|freqz| / N` : l'amplitude des raies d'un signal periodique."""
    if len(echantillons) == 0:
        msg = "aucun echantillon : la transformee discrete n'est pas definie"
        raise ValueError(msg)
    _, reponse = signal.freqz(
        echantillons, worN=frequences, fs=frequence_d_echantillonnage
    )
    return np.asarray(np.abs(reponse) / len(echantillons), dtype=float)
