"""Macros de la seance 3.2, exercice 1, et la reunion des cinq exercices.

Exercice 1 : le courant de la charge vient du resolveur de reseau, pas
des mailles du manuscrit. Les deux puissances des alternateurs sont
publiees en plus : leur somme doit redonner celle de la charge, puisque
les inductances n'absorbent aucun actif -- un second chemin, gratuit.

Le prefixe est `TroisDeuxUn`. L'exercice 5 est dans
`macros_seance_32_cinq`, le 6 dans `macros_seance_32_six`, les litteraux
(2 et 7) dans `macros_seance_32_litteraux`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application import symboles_seance_32 as d
from polytech.electromagnetisme.application.enonces_seance_32 import circuit_du_un
from polytech.electromagnetisme.application.formatage import nombre
from polytech.electromagnetisme.application.macros_seance_32_cinq import (
    complexe,
    degres,
    macros_du_cinq,
)
from polytech.electromagnetisme.application.macros_seance_32_litteraux import (
    macros_des_litteraux,
)
from polytech.electromagnetisme.application.macros_seance_32_six import (
    macros_du_six,
)
from polytech.electromagnetisme.domain.circuits.phaseurs import (
    en_regime_sinusoidal,
    puissance_complexe,
)
from polytech.electromagnetisme.domain.circuits.reseau import SourceDeTension


def _phaseurs_du_un() -> tuple[complex, complex]:
    """La tension de la charge et son courant."""
    potentiels = circuit_du_un().potentiels()
    tension = complexe(en_regime_sinusoidal(potentiels["h"], d.OMEGA_DU_UN))
    return tension, tension / float(d.UN_R)


def puissances_des_sources(retard_en_degres: sp.Expr) -> tuple[float, float]:
    """L'actif que fournit chaque alternateur, `Re(E I*)` en convention generateur.

    Le courant d'une source est rendu sortant par sa borne plus : c'est
    la convention generateur, et `E I*` y compte ce que la source fournit.
    """
    reseau = circuit_du_un(retard_en_degres)
    premiere, seconde = (
        complexe(
            puissance_complexe(
                s.valeur,
                en_regime_sinusoidal(reseau.courant(s), d.OMEGA_DU_UN),
            ),
        ).real
        for s in reseau.elements
        if isinstance(s, SourceDeTension)
    )
    return premiere, seconde


def _macros_du_un() -> dict[str, str]:
    """Les donnees, les reactances, le courant, ses dephasages, les puissances."""
    tension, courant = _phaseurs_du_un()
    fournie_1, fournie_2 = puissances_des_sources(d.UN_RETARD)
    sans_1, sans_2 = puissances_des_sources(sp.Integer(0))
    deux = complexe(d.UN_E2 * sp.exp(-sp.I * sp.pi * d.UN_RETARD / 180))
    x_1 = float(d.OMEGA_DU_UN * d.UN_L1)
    x_2 = float(d.OMEGA_DU_UN * d.UN_L2)
    terme = deux / (1j * x_2)
    return {
        "TroisDeuxUnFrequence": nombre(float(d.UN_FREQUENCE), 0),
        "TroisDeuxUnEUn": nombre(float(d.UN_E1), 0),
        "TroisDeuxUnEDeux": nombre(float(d.UN_E2), 1),
        "TroisDeuxUnRetard": nombre(float(d.UN_RETARD), 1),
        "TroisDeuxUnR": nombre(float(d.UN_R), 0),
        "TroisDeuxUnLUn": nombre(float(d.UN_L1) * 1e3, 0),
        "TroisDeuxUnLDeux": nombre(float(d.UN_L2) * 1e3, 0),
        "TroisDeuxUnEDeuxReel": nombre(deux.real, 2),
        "TroisDeuxUnEDeuxImag": nombre(-deux.imag, 2),
        "TroisDeuxUnXUn": nombre(x_1, 4),
        "TroisDeuxUnXDeux": nombre(x_2, 4),
        "TroisDeuxUnRapport": nombre(float(d.UN_R) / x_2, 3),
        "TroisDeuxUnTermeReel": nombre(-terme.real, 2),
        "TroisDeuxUnTermeImag": nombre(-terme.imag, 2),
        "TroisDeuxUnCourant": nombre(abs(courant), 2),
        "TroisDeuxUnCourantReel": nombre(courant.real, 2),
        "TroisDeuxUnCourantImag": nombre(-courant.imag, 2),
        "TroisDeuxUnDephasageUn": nombre(degres(courant), 2),
        "TroisDeuxUnDephasageDeux": nombre(degres(courant) + float(d.UN_RETARD), 2),
        "TroisDeuxUnTension": nombre(abs(tension), 2),
        "TroisDeuxUnPuissance": nombre(float(d.UN_R) * abs(courant) ** 2, 0),
        "TroisDeuxUnPuissanceUn": nombre(fournie_1, 0),
        "TroisDeuxUnPuissanceDeux": nombre(fournie_2, 0),
        "TroisDeuxUnSansRetardUn": nombre(sans_1, 0),
        "TroisDeuxUnSansRetardDeux": nombre(sans_2, 0),
    }


def macros_de_la_seance_trois_deux() -> dict[str, str]:
    """Les cinq exercices de la seance."""
    return {
        **_macros_du_un(),
        **macros_du_cinq(),
        **macros_du_six(),
        **macros_des_litteraux(),
    }
