"""Associations de resistances et mailles -- exercice 8 du chapitre III.

Trois difficultes que l'exercice 1 n'avait pas.

**Le condensateur.** En regime stationnaire il ne passe aucun courant,
donc se comporte en circuit ouvert. C'est ce qui simplifie le circuit
avant tout calcul, et un corrige qui commencerait par les mailles sans
cette remarque calculerait dans un circuit qui n'existe pas.

**Eteindre les sources n'est pas la meme chose selon leur nature.** Une
source de tension eteinte impose zero volt, donc se court-circuite ; une
source de courant eteinte impose zero ampere, donc s'**ouvre**. Confondre
les deux donnerait `R_1 + R_2` au lieu de leur parallele -- l'erreur la
plus courante de l'exercice, et la plus facile a attraper : un parallele
est toujours plus petit que la plus petite des deux resistances.

**Le courant de maille n'est pas celui de la source.** La source impose
`I` dans sa branche ; le courant `I_1` de la maille s'en deduit par la
loi des mailles, ou la source apparait par le terme `-R_2 I` -- elle
traverse `R_2` en plus de `I_1`, et c'est ainsi qu'elle entre dans la
maille sans y etre une force electromotrice.

Tout est calcule en flottants jusqu'au bout. Arrondir `E_TH` a deux
decimales avant la derniere etape transformerait un resultat **exact** --
deux amperes, les sixiemes se compensant -- en resultat approche, et
l'exercice perdrait l'information qu'il donne gratuitement.
"""

from __future__ import annotations

from polytech.electromagnetisme.domain.circuits.stationnaires import Thevenin


def _exiger_les_resistances(*valeurs: float) -> None:
    """Une resistance nulle ou negative n'est pas un cas du cours."""
    for valeur in valeurs:
        if valeur <= 0:
            msg = f"resistance nulle ou negative : {valeur} ohm"
            raise ValueError(msg)


def resistance_en_parallele(premiere: float, seconde: float) -> float:
    """`R_1 R_2 / (R_1 + R_2)`, toujours plus petite que la plus petite.

    La propriete garde le calcul : deux chemins conduisent mieux qu'un, et
    une somme la violerait aussitot.
    """
    _exiger_les_resistances(premiere, seconde)
    return premiere * seconde / (premiere + seconde)


def courant_de_maille(
    premiere_tension: float,
    seconde_tension: float,
    premiere_resistance: float,
    seconde_resistance: float,
    courant_injecte: float,
) -> float:
    """`I_1 = (E_1 - E_2 - R_2 I) / (R_1 + R_2)`.

    A courant injecte nul, il reste le diviseur ordinaire
    `(E_1 - E_2) / (R_1 + R_2)` : le terme supplementaire est donc bien
    la contribution de la source, et une faute de signe s'y verrait.
    """
    somme = premiere_resistance + seconde_resistance
    if somme <= 0:
        msg = f"resistance nulle ou negative : somme de maille {somme} ohm"
        raise ValueError(msg)
    numerateur = (
        premiere_tension - seconde_tension - seconde_resistance * courant_injecte
    )
    return numerateur / somme


def thevenin_de_la_maille(
    premiere_tension: float,
    seconde_tension: float,
    premiere_resistance: float,
    seconde_resistance: float,
    courant_injecte: float,
) -> Thevenin:
    """Le dipole equivalent vu des bornes `A`-`B`.

    `E_TH = E_2 + R_2 (I_1 + I)` : la seconde resistance est traversee par
    la somme du courant de maille et du courant injecte, et c'est cette
    somme qui porte la tension lue aux bornes.

    `R_TH` est le parallele, les deux sources etant eteintes selon leur
    nature.
    """
    maille = courant_de_maille(
        premiere_tension,
        seconde_tension,
        premiere_resistance,
        seconde_resistance,
        courant_injecte,
    )
    return Thevenin(
        tension_a_vide=seconde_tension
        + seconde_resistance * (maille + courant_injecte),
        resistance=resistance_en_parallele(premiere_resistance, seconde_resistance),
    )


def courant_dans_la_charge(
    dipole: Thevenin,
    charge: float,
    courant_en_parallele: float,
) -> float:
    """`I = (E_TH + R_TH I') / (R' + R_TH)`.

    La source de courant `I'` en parallele de la charge y ajoute sa
    contribution : vue du dipole, elle equivaut a une force electromotrice
    de plus, `R_TH I'`.
    """
    _exiger_les_resistances(charge)
    numerateur = dipole.tension_a_vide + dipole.resistance * courant_en_parallele
    return numerateur / (charge + dipole.resistance)
