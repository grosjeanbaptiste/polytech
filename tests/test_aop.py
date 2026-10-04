"""L'AOP a gain fini, et le zero virtuel comme sa limite.

Oracles : les gains du suiveur, de l'inverseur et du non-inverseur que le
cours etablit par le zero virtuel (`TP_5_Solution.pdf`, exercices 10 et
14). La bibliotheque, elle, ne suppose aucun zero virtuel : elle resout
le circuit avec un gain `A` fini, puis fait tendre `A` vers l'infini.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electronique.domain.aop import (
    GAIN,
    aop,
    saturer,
    sortie_ideale,
    sortie_saturee,
)
from polytech.electronique.domain.limites import limite_a_l_infini
from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceDeTension,
    resoudre,
)

R1, R2, VIN, IS = sp.symbols("R_1 R_2 V_in I_S", positive=True)


def _inverseur() -> list[Element]:
    return [
        SourceDeTension(1, 0, VIN, IS),
        Admittance(1, 2, 1 / R1),
        Admittance(2, 3, 1 / R2),
        *aop(0, 2, 3),
    ]


def test_le_suiveur_recopie_son_entree() -> None:
    circuit = [SourceDeTension(1, 0, VIN, IS), *aop(1, 2, 2)]
    assert sp.simplify(sortie_ideale(circuit, 2, 2) - VIN) == 0


def test_l_inverseur_rend_moins_r2_sur_r1() -> None:
    assert sp.simplify(sortie_ideale(_inverseur(), 3, 3) + R2 / R1 * VIN) == 0


def test_le_non_inverseur_rend_un_plus_r2_sur_r1() -> None:
    circuit: list[Element] = [
        SourceDeTension(1, 0, VIN, IS),
        Admittance(2, 0, 1 / R1),
        Admittance(2, 3, 1 / R2),
        *aop(1, 2, 3),
    ]
    attendu = (1 + R2 / R1) * VIN
    assert sp.simplify(sortie_ideale(circuit, 3, 3) - attendu) == 0


def test_saturer_borne_aux_rails() -> None:
    assert saturer(40.0, 12.0) == 12.0
    assert saturer(-40.0, 12.0) == -12.0
    assert saturer(5.0, 12.0) == 5.0


def test_une_alimentation_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="alimentation"):
        saturer(1.0, 0.0)


def test_une_sortie_saturee_est_une_source_de_tension() -> None:
    """Sortie collee a `+12 V` : l'inverseur devient un simple diviseur."""
    v_moins = sortie_saturee(_inverseur(), 3, 3, sp.Integer(12))[1]
    attendu = VIN + (12 - VIN) * R1 / (R1 + R2)
    assert sp.simplify(v_moins - attendu) == 0


def test_une_charge_d_un_ohm_survit_a_la_saturation() -> None:
    """Seul l'AOP est retire : une charge sur sa sortie reste en place."""
    circuit = [*_inverseur(), Admittance(3, 0, sp.Integer(1))]
    courant = sortie_saturee(circuit, 3, 3, sp.Integer(12))[-1]
    attendu = 12 + (12 - VIN) / (R1 + R2)
    assert sp.simplify(courant - attendu) == 0


def test_seul_l_aop_sature_est_retire() -> None:
    """Un suiveur en tete garde son AOP quand le second sature."""
    circuit: list[Element] = [
        SourceDeTension(1, 0, VIN, IS),
        *aop(1, 2, 2),
        Admittance(2, 3, 1 / R1),
        Admittance(3, 4, 1 / R2),
        *aop(0, 3, 4),
    ]
    tensions = sortie_saturee(circuit, 4, 4, sp.Integer(12))
    assert sp.simplify(limite_a_l_infini(tensions[1], GAIN) - VIN) == 0


def test_a_gain_fini_l_inverseur_garde_le_signe_de_sa_boucle() -> None:
    """`V_s = -A R2 / (R1 + R2 + A R1) V_in` : la limite seule ne suffit pas.

    Inverser les entrees de l'AOP rend la contre-reaction positive, et la
    limite `A -> oo` n'en dit rien : elle est la meme. Seul le gain fini
    porte le sens de la boucle.
    """
    sortie = resoudre(_inverseur(), 3)[2]
    attendu = -GAIN * R2 / (R1 + R2 + GAIN * R1) * VIN
    assert sp.simplify(sortie - attendu) == 0
