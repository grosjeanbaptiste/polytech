"""Les f.e.m. induites dans la spire -- seance 2.1, exercice 2, suite.

Trois causes font varier le flux du fil a travers la spire : elle
s'eloigne, elle glisse le long du fil, ou le courant du fil varie.

**Seule la vitesse normale au fil compte.** Le flux ne depend de la
position que par `d` : une translation selon `y` ne le change pas. Une
translation oblique ne vaut donc que par sa composante selon `x`, et
c'est pourquoi `fem_de_translation` ne recoit que celle-la.

**Deux routes pour la translation.** `fem_de_translation` derive le
flux ; `fem_par_le_champ_electromoteur` integre `v ^ B` le long de la
spire, sans aucun flux. Elles s'accordent ici parce que le circuit est
un fil bien defini qui se deplace d'un bloc -- ce qui n'est pas le cas
du disque de l'exercice 12.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.fil_et_spire import (
    SpireDevantUnFil,
    induction_du_fil,
)
from polytech.electromagnetisme.domain.magnetostatique.induction import (
    force_electromotrice,
)
from polytech.electromagnetisme.domain.symboles import PULSATION, TEMPS


def _spire_en_mouvement(
    spire: SpireDevantUnFil,
    vitesse_normale: sp.Expr,
) -> SpireDevantUnFil:
    """La meme spire, a l'instant `t` : `d` devient `d + v t`."""
    return SpireDevantUnFil(
        largeur=spire.largeur,
        hauteur=spire.hauteur,
        distance=spire.distance + vitesse_normale * TEMPS,
    )


def fem_de_translation(
    spire: SpireDevantUnFil,
    courant_du_fil: sp.Expr,
    vitesse_normale: sp.Expr,
    permeabilite: sp.Expr,
) -> sp.Expr:
    """`e = -d(phi)/dt` quand la spire s'eloigne a la vitesse `v_x`."""
    mobile = _spire_en_mouvement(spire, vitesse_normale)
    return force_electromotrice(mobile.flux(courant_du_fil, permeabilite))


def fem_par_le_champ_electromoteur(
    spire: SpireDevantUnFil,
    courant_du_fil: sp.Expr,
    vitesse_normale: sp.Expr,
    permeabilite: sp.Expr,
) -> sp.Expr:
    """La circulation de `v ^ B` le long de la spire, dans le sens direct.

    `v u_x ^ B_z u_z = -v B_z u_y` : le champ electromoteur est porte
    par `Oy`. Les cotes haut et bas lui sont perpendiculaires et ne
    contribuent pas. Le cote droit est parcouru vers `+y`, le gauche
    vers `-y` : leurs contributions se retranchent, et il ne reste que
    l'ecart du champ entre les deux.
    """
    mobile = _spire_en_mouvement(spire, vitesse_normale)
    proche = mobile.distance
    lointain = mobile.distance + mobile.largeur

    def electromoteur(abscisse: sp.Expr) -> sp.Expr:
        induction = induction_du_fil(courant_du_fil, abscisse, permeabilite)
        return -vitesse_normale * induction

    circulation = spire.hauteur * (electromoteur(lointain) - electromoteur(proche))
    return sp.simplify(circulation)


def fem_a_courant_sinusoidal(
    spire: SpireDevantUnFil,
    valeur_efficace: sp.Expr,
    permeabilite: sp.Expr,
) -> sp.Expr:
    """`e = -d(phi)/dt` pour `i_1 = I sqrt(2) cos(omega t)`, spire immobile."""
    courant = valeur_efficace * sp.sqrt(2) * sp.cos(PULSATION * TEMPS)
    return force_electromotrice(spire.flux(courant, permeabilite))
