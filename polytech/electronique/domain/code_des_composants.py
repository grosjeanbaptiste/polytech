"""Lire une resistance, un condensateur et un boitier DIP.

La planche « Presentation des labos » a l'air logistique -- horaires,
sachets, protoboard -- et pour moitie elle l'est. Mais elle enseigne
quatre lectures, dont deux sont du calcul, et elle donne **ses propres
oracles** : le tableau complet des couleurs, et huit marquages de
condensateur avec leur valeur.

Le tableau des couleurs porte deux pieges.

Le premier est le **noir en multiplicateur** : « Noir : 0, x 1 ». Un
exposant nul laisse la valeur intacte, si bien que marron-noir-noir vaut
dix ohms et non cent.

Le second est que les memes couleurs servent dans les deux colonnes,
mais pas toutes. Rouge et marron sont a la fois des chiffres et des
tolerances ; vert n'est qu'un chiffre. Et violet, gris et blanc n'ont
**pas** de multiplicateur -- la planche porte un tiret en face des trois.
Les accepter en troisieme bague serait inventer trois decades que le
cours ne donne pas, et la planche dit quoi faire a la place : « dans le
doute, utiliser un ohmmetre ».

Le marquage des condensateurs prend quatre formes dans les huit exemples
de la planche -- trois chiffres (`104`), prefixe collant (`µ1`, `n10`),
virgule initiale (`.22J`), lettre de tolerance (`22M`, `103F50`). Un
analyseur qui ne traiterait que la premiere echouerait sur cinq des huit.
"""

from __future__ import annotations

import re
from typing import Final

CHIFFRES: Final = {
    "noir": 0,
    "marron": 1,
    "rouge": 2,
    "orange": 3,
    "jaune": 4,
    "vert": 5,
    "bleu": 6,
    "violet": 7,
    "gris": 8,
    "blanc": 9,
}
"""La colonne « Chiffre » du tableau, les dix couleurs."""

MULTIPLICATEURS: Final = ("noir", "marron", "rouge", "orange", "jaune", "vert", "bleu")
"""La colonne « Exposant », qui s'arrete au megohm : sept couleurs sur dix."""

TOLERANCES: Final = {"argent": 0.10, "or": 0.05, "rouge": 0.02, "marron": 0.01}
"""La colonne « Tolerance » : quatre couleurs, dont deux sont aussi des chiffres."""

TOLERANCES_LITTERALES: Final = {"M": 0.20, "J": 0.05, "F": 0.01}
"""Les lettres de tolerance que les huit exemples de la planche emploient."""

PREFIXES: Final = {"µ": 1e-6, "u": 1e-6, "n": 1e-9, "p": 1e-12}
"""Les prefixes collants : `µ1` vaut 0,1 µF, `n10` vaut 0,10 nF."""

_TROIS_CHIFFRES: Final = re.compile(r"^(\d{2})(\d)")
_PREFIXE_COLLANT: Final = re.compile(r"^(\d*)([µunp])(\d*)")


def resistance_des_bagues(dizaine: str, unite: str, exposant: str) -> float:
    """Les trois premieres bagues, en ohms."""
    for couleur in (dizaine, unite):
        if couleur not in CHIFFRES:
            msg = f"couleur inconnue au tableau : {couleur!r}"
            raise ValueError(msg)
    if exposant not in MULTIPLICATEURS:
        msg = (
            f"{exposant!r} n'a pas de multiplicateur au tableau, qui s'arrete "
            f"au megohm -- la planche dit alors d'utiliser un ohmmetre"
        )
        raise ValueError(msg)
    valeur = 10 * CHIFFRES[dizaine] + CHIFFRES[unite]
    return float(valeur * 10 ** CHIFFRES[exposant])


def tolerance_de_la_bague(couleur: str) -> float:
    """La bague de tolerance, en fraction de l'unite."""
    if couleur not in TOLERANCES:
        msg = f"{couleur!r} n'est pas une bague de tolerance du tableau"
        raise ValueError(msg)
    return TOLERANCES[couleur]


def capacite_du_marquage(marquage: str) -> float:
    """La capacite que ce marquage designe, en farads.

    Les huit exemples de la planche la verifient, un par un.
    """
    if not marquage:
        msg = "marquage vide : un condensateur sans marquage se mesure"
        raise ValueError(msg)
    corps = marquage.rstrip("0123456789")
    for lettre, tolerance in TOLERANCES_LITTERALES.items():
        if corps.endswith(lettre) and tolerance:
            marquage = corps[: -len(lettre)]
            break
    if (collant := _PREFIXE_COLLANT.match(marquage)) is not None:
        entier, prefixe, decimales = collant.groups()
        nombre = float(f"{entier or 0}.{decimales or 0}")
        return nombre * PREFIXES[prefixe]
    if marquage.startswith("."):
        return float(marquage) * PREFIXES["µ"]
    if (trois := _TROIS_CHIFFRES.match(marquage)) is not None:
        significatifs, exposant = trois.groups()
        # `10 ** int(...)` rend `Any` a mypy des que la base est litterale ;
        # le `float()` englobant rend le type explicite.
        return float(float(significatifs) * 10 ** int(exposant) * PREFIXES["p"])
    return float(marquage) * PREFIXES["p"]


def numero_oppose_sur_un_dip(patte: int, pattes: int) -> int:
    """La patte qui fait face, le parcours etant trigonometrique.

    Sur le boitier a huit pattes de la planche, 1 fait face a 8 et la
    somme vaut neuf. C'est la regle qui evite de compter a l'envers, et
    l'erreur de cablage qui en resulte est celle qui grille un composant.
    """
    if pattes % 2:
        msg = f"nombre de pattes impair : {pattes} -- un DIP a deux rangees egales"
        raise ValueError(msg)
    if not 1 <= patte <= pattes:
        msg = f"patte hors du boitier : {patte} sur {pattes}"
        raise ValueError(msg)
    return pattes + 1 - patte


PATTE_LONGUE: Final = "+"
"""« Longue patte = +, cote avec un "-" = - », pour la LED et l'electrolytique."""

ROTATION_TRIGONOMETRIQUE: Final = ("premiere", "seconde")
"""Les deux rangees du boitier, parcourues depuis le repere."""


def intervalle_de_tolerance(nominale: float, tolerance: float) -> tuple[float, float]:
    """Les deux bornes entre lesquelles la resistance peut valoir.

    C'est a quoi sert la quatrieme bague, et c'est l'intervalle qui
    compte dans un dimensionnement : un diviseur calcule au nominal peut
    sortir de ses specifications sur les bornes.
    """
    if tolerance < 0:
        msg = f"tolerance negative : {tolerance}"
        raise ValueError(msg)
    return nominale * (1 - tolerance), nominale * (1 + tolerance)


def polarite_de_la_patte_longue() -> str:
    """Le signe que porte la patte longue d'un composant polarise.

    La planche le dit pour le condensateur et pour la LED. C'est l'erreur
    qui grille le composant au premier branchement : un electrolytique a
    l'envers ne pardonne pas.
    """
    return PATTE_LONGUE


def rangee_de_la_patte(patte: int, pattes: int) -> str:
    """Laquelle des deux rangees porte cette patte.

    Le parcours part du repere -- « point ou encoche : cote superieur » --
    et suit le sens trigonometrique : la premiere moitie des numeros d'un
    cote, la seconde de l'autre.
    """
    if pattes % 2:
        msg = f"nombre de pattes impair : {pattes} -- un DIP a deux rangees egales"
        raise ValueError(msg)
    if not 1 <= patte <= pattes:
        msg = f"patte hors du boitier : {patte} sur {pattes}"
        raise ValueError(msg)
    return ROTATION_TRIGONOMETRIQUE[0 if patte <= pattes // 2 else 1]
