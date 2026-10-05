"""Le multiplexage : un convertisseur pour plusieurs capteurs.

Chapitre 4, partie III, planches 35 a 52. Un multiplexeur relie au plus
une de ses entrees a sa sortie ; un microcontroleur le fait tourner, et un
seul convertisseur echantillonne toutes les voies a tour de role. Pour
`m` voies a `f_s` hertz chacune, on economise `m - 1` convertisseurs, au
prix d'un convertisseur a `m f_s` hertz.

Les voies sont echantillonnees a des instants decales, mais chacune
regulierement : le theoreme d'echantillonnage n'en demande pas plus.

Un filtre ne se place **jamais** apres le multiplexeur : le saut d'une
voie a la suivante y laisse un residu `exp(-2 pi f_c dt)` qui n'a pas
fini de s'eteindre quand l'echantillon est pris.
"""

from __future__ import annotations

import math


def convertisseurs_economises(voies: int) -> int:
    """`m - 1` : ce qu'un multiplexeur a `m` voies remplace."""
    if voies < 1:
        msg = f"au moins une voie est attendue : {voies}"
        raise ValueError(msg)
    return voies - 1


def paires_multiplexeur_convertisseur(
    capteurs: int, voies_par_multiplexeur: int
) -> int:
    """Le nombre de paires qu'il faut pour servir tous les capteurs."""
    if capteurs < 1 or voies_par_multiplexeur < 1:
        msg = f"comptes invalides : {capteurs} capteurs, {voies_par_multiplexeur} voies"
        raise ValueError(msg)
    return math.ceil(capteurs / voies_par_multiplexeur)


def instants_de_la_voie(
    voie: int, *, voies: int, cadence: float, nombre: int
) -> tuple[float, ...]:
    """Les `nombre` premiers instants ou cette voie est echantillonnee."""
    if not 0 <= voie < voies:
        msg = f"voie hors de [0, {voies - 1}] : {voie}"
        raise ValueError(msg)
    if cadence <= 0:
        msg = f"cadence par voie nulle ou negative : {cadence} Hz"
        raise ValueError(msg)
    decalage = voie / (voies * cadence)
    return tuple(decalage + rang / cadence for rang in range(nombre))


def residu_de_commutation(coupure: float, duree: float) -> float:
    """`exp(-2 pi f_c dt)` : la part du saut qu'un passe-bas d'ordre un garde."""
    if coupure <= 0 or duree < 0:
        msg = f"coupure {coupure} Hz ou duree {duree} s invalide"
        raise ValueError(msg)
    return math.exp(-2 * math.pi * coupure * duree)
