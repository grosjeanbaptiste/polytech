"""Macros de l'exercice 11 : force par unite de longueur, symbolique.

L'enonce ne donne **aucun nombre**, et c'est ce qui rend ce jeu de macros
different des autres : il publie des **expressions LaTeX**, non des
valeurs. La forme est celle que le domaine etablit, et elle n'est pas
resimplifiee -- une mise en facteur voulue porte un sens pedagogique que
`simplify` ne connait pas.

Trois formes du module sont publiees, et chacune sert une etape du
raisonnement : la compacte, la developpee, et celle du cas ou les deux
fils opposes se compensent. La troisieme est ce qui prouve que le terme
croise est bien negatif -- avec un `+`, le module ne pourrait jamais
descendre a la seule contribution du deuxieme fil.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.formatage import expression_latex
from polytech.electromagnetisme.domain.magnetostatique.superposition import (
    Fil,
    champ_de_trois_fils,
    force_par_unite_de_longueur,
    module_de_la_force,
)

PERMEABILITE = sp.Symbol("mu_0", positive=True)
COURANT = sp.Symbol("I", positive=True)
COURANTS = sp.symbols("I_1 I_2 I_3", positive=True)
DISTANCES = sp.symbols("d_1 d_2 d_3", positive=True)


def _trois_fils() -> tuple[Fil, Fil, Fil]:
    """Les trois conducteurs de l'enonce, avec leurs symboles.

    Construits un par un plutot que par comprehension : un `tuple(...)`
    engendre n'a pas de longueur connue du verificateur de types, et le
    lui faire admettre aurait demande une suppression. Or une suppression
    sans necessite est une dette, et ici l'ecriture explicite est aussi
    lisible -- elle nomme meme les trois distances.
    """
    premier, deuxieme, troisieme = COURANTS
    proche, moyenne, lointaine = DISTANCES
    return (
        Fil(courant=premier, distance=proche),
        Fil(courant=deuxieme, distance=moyenne),
        Fil(courant=troisieme, distance=lointaine),
    )


def macros_forces_onze() -> dict[str, str]:
    """Le champ, la force, et les trois formes du module."""
    premier, deuxieme, troisieme = _trois_fils()
    champ = champ_de_trois_fils(premier, deuxieme, troisieme)
    force = force_par_unite_de_longueur(COURANT, champ, PERMEABILITE)
    i1, i2, i3 = COURANTS
    d1, d2, d3 = DISTANCES
    facteur = PERMEABILITE * COURANT / (2 * sp.pi)
    compact = facteur * sp.sqrt(i2**2 / d2**2 + (i3 / d3 - i1 / d1) ** 2)
    developpe = facteur * sp.sqrt(
        i2**2 / d2**2 + i3**2 / d3**2 + i1**2 / d1**2 - 2 * i1 * i3 / (d1 * d3),
    )
    compense = champ_de_trois_fils(premier, deuxieme, premier)
    return {
        "OnzeChampX": expression_latex(champ[0]),
        "OnzeChampY": expression_latex(champ[1]),
        "OnzeModuleDuPremier": expression_latex(premier.module_du_champ()),
        "OnzeForceX": expression_latex(sp.simplify(force[0])),
        "OnzeForceY": expression_latex(sp.simplify(force[1])),
        "OnzeModuleCompact": expression_latex(compact),
        "OnzeModuleDeveloppe": expression_latex(developpe),
        "OnzeTermeCroise": expression_latex(-2 * i1 * i3 / (d1 * d3)),
        "OnzeChampYCompense": expression_latex(sp.simplify(compense[1])),
        "OnzeModuleCompense": expression_latex(
            sp.simplify(module_de_la_force(COURANT, compense, PERMEABILITE)),
        ),
        "OnzeProduitScalaire": expression_latex(
            sp.simplify(force[0] * champ[0] + force[1] * champ[1]),
        ),
    }
