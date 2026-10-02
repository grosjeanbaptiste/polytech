"""Macros de l'exercice 8 : dix spires emboitees, de largeurs doublees.

Premier exercice **numerique** de la serie des champs magnetiques : le
jeu publie des valeurs et non des expressions.

Il publie le champ juste et celui qu'on obtient en prenant la largeur
d'une spire pour sa distance a l'axe -- la moitie. C'est le piege de
l'enonce, et les deux valeurs cote a cote le montrent mieux qu'un
avertissement.

Ce jeu a porte une macro qui chiffrait une « erreur du manuscrit ». Le
manuscrit n'en commet pas ; la macro est retiree avec la these.

Les donnees de l'enonce et les sommes geometriques sont dans
`macros_champ_08_sommes`.

Le prefixe est `ChampHuit`.
"""

from __future__ import annotations

from polytech.electromagnetisme.application.formatage import nombre
from polytech.electromagnetisme.application.macros_champ_08_sommes import (
    COURANT,
    DEPUIS_UN_CENTIMETRE,
    distances_a_l_axe,
    largeurs_de_l_enonce,
    macros_de_l_enonce,
    macros_des_sommes,
)
from polytech.electromagnetisme.domain.magnetostatique.nappe import (
    champ_d_un_seul_cote,
    champ_de_la_nappe,
    champ_de_la_nappe_infinie,
)

DECIMALES_DU_CHAMP = 1
"""Un chiffre apres la virgule : c'est la precision que le manuscrit donne."""

DECIMALES_DE_LA_LIMITE = 2
"""Deux pour la nappe infinie : a un chiffre, 190,99 s'arrondirait en 191.

Et « 191 » face a « 190,8 » se lirait comme une valeur moins precise,
alors que c'est l'inverse -- c'est la limite exacte de la serie. Le
second chiffre est donc necessaire pour que l'ecart se voie.
"""


def _macros_du_champ() -> dict[str, str]:
    """Le champ juste, celui d'un seul cote, et celui du piege."""
    distances = distances_a_l_axe()
    return {
        "ChampHuitJuste": nombre(
            float(champ_de_la_nappe(COURANT, distances)),
            DECIMALES_DU_CHAMP,
        ),
        "ChampHuitUnSeulCote": nombre(
            float(champ_d_un_seul_cote(COURANT, distances)),
            DECIMALES_DU_CHAMP,
        ),
        "ChampHuitParLesLargeurs": nombre(
            float(champ_de_la_nappe(COURANT, largeurs_de_l_enonce())),
            DECIMALES_DU_CHAMP,
        ),
    }


def _macros_de_la_convergence() -> dict[str, str]:
    """La suite infinie de spires, et la part qu'en donnent les dix."""
    fini = champ_de_la_nappe(COURANT, distances_a_l_axe())
    infini = champ_de_la_nappe_infinie(COURANT, DEPUIS_UN_CENTIMETRE)
    return {
        "ChampHuitNappeInfinie": nombre(float(infini), DECIMALES_DE_LA_LIMITE),
        "ChampHuitPartAtteinte": nombre(float(fini / infini) * 100, 2),
        "ChampHuitManqueAbsolu": nombre(
            float(infini - fini),
            DECIMALES_DE_LA_LIMITE,
        ),
    }


def macros_champ_huit() -> dict[str, str]:
    """L'enonce, les sommes, le champ et la convergence."""
    return {
        **macros_de_l_enonce(),
        **macros_des_sommes(),
        **_macros_du_champ(),
        **_macros_de_la_convergence(),
    }
