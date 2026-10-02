"""Le systeme de l'exercice 5 : cinq equations de Laplace, cinq inconnues.

Le circuit porte trois mailles, deux condensateurs et deux inductances.
Le manuscrit le met en equations, passe en Laplace, reduit -- et
s'arrete en constatant que le systeme est resoluble. Ce module le
resout.

**Ce que le passage en Laplace fait des conditions initiales.** Il les
transforme en **termes de l'equation** plutot qu'en conditions annexes :
`L di/dt` devient `L (p I(p) - i(0))`, et `C du/dt` devient
`C (p U(p) - u(0))`. C'est tout l'interet de la methode -- l'etat initial
entre dans l'algebre au lieu de rester a cote.

Les deux etats initiaux non nuls sont `u_{C_1}(0) = E_2 - E_1` (le
condensateur de gauche s'est charge sous la difference des deux sources
pendant que l'interrupteur etait ouvert) et rien d'autre : `i_2(0)`,
`i_3(0)` et `u_{C_4}(0)` sont nuls, le dernier parce qu'aucun courant ne
traversait `R_3`-`L_3`.

**Les symboles sont au niveau du module** et non passes en arguments :
l'exercice est entierement litteral, il n'y a pas de second jeu de
valeurs a traiter, et neuf parametres dans une signature auraient cache
la structure au lieu de la montrer.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.transitoire.laplace import VARIABLE

FORCES: Final = sp.symbols("E_1 E_2", positive=True)
RESISTANCES: Final = sp.symbols("R_1 R_2 R_3", positive=True)
INDUCTANCES: Final = sp.symbols("L_2 L_3", positive=True)
CAPACITES: Final = sp.symbols("C_1 C_4", positive=True)

TENSION_C1: Final = sp.Symbol("U_{C_1}")
TENSION_C4: Final = sp.Symbol("U_{C_4}")
COURANT_1: Final = sp.Symbol("I_1")
COURANT_2: Final = sp.Symbol("I_2")
COURANT_3: Final = sp.Symbol("I_3")

INCONNUES: Final = (TENSION_C1, TENSION_C4, COURANT_1, COURANT_2, COURANT_3)
"""Les cinq transformees que le systeme determine."""

ETAT_INITIAL_C1: Final = FORCES[1] - FORCES[0]
"""`u_{C_1}(0) = E_2 - E_1`, le seul etat initial non nul du circuit.

Pendant que l'interrupteur etait ouvert, le circuit se reduisait aux deux
sources et aux deux resistances en serie, sans courant : la tension du
condensateur s'etait donc etablie a la difference des deux forces
electromotrices.
"""


def terme_source_de_la_premiere_maille() -> sp.Expr:
    """`E_1 + u_{C_1}(0)`, qui se reduit a `E_2` -- et c'est le resultat.

    La charge accumulee par `C_1` annule **exactement** la source qui
    l'a chargee. Le circuit se comporte donc comme si seule `E_2`
    existait, et `E_1` disparait des trois courants.

    Rendue comme un calcul plutot qu'ecrite `E_2` : le document compose
    cette reduction, et l'ecrire a la main serait reaffirmer ce qu'on
    veut montrer.
    """
    return sp.simplify(FORCES[0] + ETAT_INITIAL_C1)


def equations() -> tuple[sp.Eq, ...]:
    """Le systeme reduit du manuscrit, equation par equation.

    Les impedances operationnelles `R + pL` apparaissent telles quelles :
    c'est la forme que le manuscrit obtient apres avoir reporte les
    conditions initiales nulles des deux inductances.
    """
    force_1, force_2 = FORCES
    res_1, res_2, res_3 = RESISTANCES
    ind_2, ind_3 = INDUCTANCES
    cap_1, cap_4 = CAPACITES
    impedance_2 = res_2 + VARIABLE * ind_2
    impedance_3 = res_3 + VARIABLE * ind_3
    return (
        # (I) la maille de gauche, ou la charge initiale de C_1 intervient.
        sp.Eq(
            force_1 / VARIABLE,
            -TENSION_C1 + res_1 * COURANT_1 + impedance_3 * COURANT_3,
        ),
        # (II) la maille de droite.
        sp.Eq(force_2 / VARIABLE, TENSION_C4 + impedance_2 * COURANT_2),
        # (III) la branche centrale definit la tension de C_4.
        sp.Eq(0, -TENSION_C4 + impedance_3 * COURANT_3),
        # La relation charge-courant de C_1, portant son etat initial.
        sp.Eq(cap_1 * ETAT_INITIAL_C1, VARIABLE * cap_1 * TENSION_C1 + COURANT_1),
        # La loi des noeuds, avec C_4 initialement decharge.
        sp.Eq(0, COURANT_1 + COURANT_2 - COURANT_3 - VARIABLE * cap_4 * TENSION_C4),
    )


@lru_cache(maxsize=1)
def solution() -> dict[sp.Symbol, sp.Expr]:
    """Les cinq transformees, resolues une fois pour toutes.

    Memoisee parce que la resolution symbolique d'un 5x5 coute quelques
    secondes et que la fonction est pure : le resultat ne depend de rien
    d'autre que du systeme lui-meme.

    Le refus est explicite plutot que de rendre un dictionnaire vide :
    un systeme singulier donnerait zero ou plusieurs solutions, et rendre
    le vide ferait echouer les appelants bien plus loin.
    """
    resolutions = sp.solve(equations(), list(INCONNUES), dict=True)
    if len(resolutions) != 1:
        msg = f"systeme sans solution unique : {len(resolutions)} trouvees"
        raise ValueError(msg)
    return {inconnue: sp.simplify(resolutions[0][inconnue]) for inconnue in INCONNUES}


@lru_cache(maxsize=1)
def degre_caracteristique() -> int:
    """L'ordre du circuit : le degre de son polynome caracteristique.

    Il egale le nombre d'elements emmagasinant de l'energie -- ici quatre,
    `C_1`, `C_4`, `L_2` et `L_3` -- parce que chacun apporte une variable
    d'etat. C'est donc le seul controle qui atteste que les quatre sont
    bien dans les equations : les limites de valeur, elles, sont
    insensibles a `L_3`.

    Le `p` de l'echelon est retire avant de mesurer. Une source constante
    appliquee a l'instant zero a pour transformee `E/p`, et ce pole se
    retrouve au denominateur de chaque courant sans appartenir au
    circuit : le confondre avec un mode propre donnerait cinq.
    """
    _, denominateur = sp.fraction(sp.cancel(sp.together(solution()[COURANT_2])))
    entier = sp.Poly(denominateur, VARIABLE)
    propre = sp.quo(entier, sp.Poly(VARIABLE, VARIABLE))
    return int(sp.degree(propre))
