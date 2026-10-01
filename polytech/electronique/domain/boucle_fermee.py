"""Ce que la retroaction fait aux grandeurs du montage.

Le chapitre 2 partie I tient en une quantite : le **rapport de boucle**
`1 + A beta`. Il divise le gain, multiplie l'impedance d'entree, divise
l'impedance de sortie, divise la distorsion non lineaire, et multiplie la
bande passante. Tout le reste du chapitre -- l'inverseur, le
non-inverseur, l'effet d'une resistance de source ou de charge -- n'est
qu'une facon de calculer `beta`.

La consequence qui fait l'interet du montage : quand `A beta` est grand
devant un, le gain tend vers `1 / beta`, c'est-a-dire qu'il ne depend plus
que de deux resistances. L'AOP peut alors varier de dix pour cent sans
que le montage s'en apercoive.

Un rapport de boucle **sous l'unite** est refuse, et ce n'est pas une
precaution formelle : cela veut dire que la boucle amplifie l'ecart au
lieu de le corriger. Tous les resultats du chapitre s'inversent alors, et
une impedance d'entree qui **diminue** passerait pour un resultat.
"""

from __future__ import annotations

SEUIL_D_EFFICACITE = 100.0
"""`A beta` au-dela duquel l'ecart au gain idealise tombe sous le pour cent.

Le seuil n'est pas choisi pour arranger un test : il est verifie. A
`A beta = 100`, `A / (1 + A beta)` s'ecarte de `1 / beta` de moins de un
pour cent, et le test le calcule.
"""


def rapport_de_boucle(gain_ouvert: float, retour: float) -> float:
    """`1 + A beta` ; leve si la boucle reagit positivement."""
    rapport = 1 + gain_ouvert * retour
    if rapport < 1:
        msg = (
            f"reaction positive : 1 + A beta = {rapport:.4g} est sous l'unite, "
            f"la boucle amplifie l'ecart au lieu de le corriger"
        )
        raise ValueError(msg)
    return rapport


def gain_en_boucle_fermee(gain_ouvert: float, retour: float) -> float:
    """`A / (1 + A beta)`, qui tend vers `1 / beta`."""
    return gain_ouvert / rapport_de_boucle(gain_ouvert, retour)


def impedance_d_entree_en_boucle_fermee(
    impedance_ouverte: float,
    gain_ouvert: float,
    retour: float,
) -> float:
    """`Z_i (1 + A beta)` : la retroaction serie **multiplie** l'entree."""
    if impedance_ouverte <= 0:
        msg = f"impedance d'entree nulle ou negative : {impedance_ouverte} ohm"
        raise ValueError(msg)
    return impedance_ouverte * rapport_de_boucle(gain_ouvert, retour)


def impedance_de_sortie_en_boucle_fermee(
    impedance_ouverte: float,
    gain_ouvert: float,
    retour: float,
) -> float:
    """`Z_o / (1 + A beta)` : la retroaction parallele **divise** la sortie."""
    if impedance_ouverte <= 0:
        msg = f"impedance de sortie nulle ou negative : {impedance_ouverte} ohm"
        raise ValueError(msg)
    return impedance_ouverte / rapport_de_boucle(gain_ouvert, retour)


def distorsion_apres_retroaction(
    distorsion: float,
    gain_ouvert: float,
    retour: float,
) -> float:
    """La distorsion non lineaire, divisee par le meme rapport.

    C'est le benefice que la planche range sous « other benefits », et il
    est souvent la vraie raison d'une contre-reaction : on ne cherche pas
    a baisser le gain, on cherche a linéariser.
    """
    if distorsion < 0:
        msg = f"distorsion negative : {distorsion}"
        raise ValueError(msg)
    return distorsion / rapport_de_boucle(gain_ouvert, retour)


def coefficient_de_retour(vers_la_masse: float, contre_reaction: float) -> float:
    """`beta = R_1 / (R_1 + R_f)` : le diviseur que le reseau de retour forme."""
    total = vers_la_masse + contre_reaction
    if total <= 0:
        msg = (
            f"reseau de retour sans resistance : {vers_la_masse} et "
            f"{contre_reaction} ohm"
        )
        raise ValueError(msg)
    return vers_la_masse / total


def produit_gain_bande(gain: float, bande: float) -> float:
    """`A f` : la grandeur que la retroaction conserve."""
    if bande <= 0:
        msg = f"bande nulle ou negative : {bande} Hz"
        raise ValueError(msg)
    return gain * bande


def bande_apres_retroaction(
    gain_ouvert: float,
    bande_ouverte: float,
    retour: float,
) -> float:
    """`f (1 + A beta)` : la bande s'elargit de ce que le gain perd.

    Leur produit ne bouge donc pas, et c'est ce qui permet de lire la
    bande d'un montage directement sur la fiche technique de l'AOP.
    """
    if bande_ouverte <= 0:
        msg = f"bande nulle ou negative : {bande_ouverte} Hz"
        raise ValueError(msg)
    return bande_ouverte * rapport_de_boucle(gain_ouvert, retour)


def retroaction_efficace(gain_ouvert: float, retour: float) -> bool:
    """Vrai si `A beta` depasse le seuil au-dela duquel le gain s'idealise."""
    return gain_ouvert * retour >= SEUIL_D_EFFICACITE
