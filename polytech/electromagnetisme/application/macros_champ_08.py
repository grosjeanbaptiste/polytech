"""Macros de l'exercice 8 : une nappe de conducteurs en progression.

Premier exercice **numerique** de la serie des champs magnetiques : le
jeu publie des valeurs et non des expressions.

Il publie **trois** champs pour le meme probleme, et c'est tout l'objet du
corrige : celui que la figure donne, celui que la somme du manuscrit
donne, et celui que son developpement donne. Les deux erreurs se
compensent, et les trois valeurs cote a cote le montrent.

Les donnees de l'enonce et les deux sommes geometriques sont dans
`macros_champ_08_sommes`, d'ou ce module les reprend : la coupure separe
ce qui releve de la **lecture** de l'enonce de ce qui releve du
**raisonnement**.

Le prefixe est `ChampHuit`.
"""

from __future__ import annotations

from polytech.electromagnetisme.application.formatage import nombre
from polytech.electromagnetisme.application.macros_champ_08_sommes import (
    COURANT,
    DEPUIS_UN_CENTIMETRE,
    distances_de_la_figure,
    distances_de_la_somme,
    macros_de_l_enonce,
    macros_des_sommes,
)
from polytech.electromagnetisme.domain.magnetostatique.nappe import (
    champ_d_un_seul_cote,
    champ_de_la_nappe,
    champ_de_la_nappe_infinie,
    champ_par_le_developpement_du_manuscrit,
)

DECIMALES_DU_CHAMP = 1
"""Un chiffre apres la virgule : c'est la precision que le manuscrit donne."""

DECIMALES_DE_LA_LIMITE = 2
"""Deux pour la nappe infinie : a un chiffre, 190,99 s'arrondirait en 191.

Et « 191 » face a « 190,8 » se lirait comme une valeur moins precise,
alors que c'est l'inverse -- c'est la limite exacte de la serie. Le
second chiffre est donc necessaire pour que l'ecart se voie.
"""


def _macros_des_trois_champs() -> dict[str, str]:
    """Le juste, celui de la somme du manuscrit, celui de son developpement."""
    figure = distances_de_la_figure()
    manuscrit = distances_de_la_somme()
    return {
        "ChampHuitJuste": nombre(
            float(champ_de_la_nappe(COURANT, figure)),
            DECIMALES_DU_CHAMP,
        ),
        "ChampHuitParLaFormuleSurLaSommeDuManuscrit": nombre(
            float(champ_de_la_nappe(COURANT, manuscrit)),
            DECIMALES_DU_CHAMP,
        ),
        "ChampHuitParLeDeveloppement": nombre(
            float(champ_par_le_developpement_du_manuscrit(COURANT, manuscrit)),
            DECIMALES_DU_CHAMP,
        ),
        "ChampHuitUnSeulCote": nombre(
            float(champ_d_un_seul_cote(COURANT, figure)),
            DECIMALES_DU_CHAMP,
        ),
    }


def _macros_de_la_convergence() -> dict[str, str]:
    """La nappe infinie, et la part qu'en donnent dix paires."""
    fini = champ_de_la_nappe(COURANT, distances_de_la_figure())
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
    """L'enonce, les sommes, les trois champs et la convergence."""
    return {
        **macros_de_l_enonce(),
        **macros_des_sommes(),
        **_macros_des_trois_champs(),
        **_macros_de_la_convergence(),
    }
