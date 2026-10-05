"""Planches d'electromagnetisme, chapitres III.2 et III.3.

Sources : `Electromagnetisme/SLIDES/Slides_03_Etats_stationnaires_partie_2.pdf`
(champ electromoteur et champ induit dynamique, planches 2 a 10) et
`Slides_03_Etats_stationnaires_partie_3.pdf` (lois des circuits a courants
stationnaires, planches 2 a 21), lues sur les pages rendues.

Les deux fascicules resument les chapitres 6, 9 et 13 du syllabus : aucun
code neuf. Les relations encadrees sont eprouvees sur les cas dessines :
le conducteur rectiligne de la planche 9, la spire de la planche 8, le
bipole des planches 18 a 21.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.mailles import (
    Branche,
    residus_des_noeuds,
    resoudre_par_les_mailles,
)
from polytech.electromagnetisme.domain.circuits.resistance import (
    conductance_en_parallele,
    resistance_uniforme,
)
from polytech.electromagnetisme.domain.circuits.resistivite import (
    METAUX,
    RESISTIVITE_DU_VERRE,
    resistivite_a_chaud,
)
from polytech.electromagnetisme.domain.circuits.stationnaires import Thevenin
from polytech.electromagnetisme.domain.circuits.theoremes import par_superposition
from polytech.electromagnetisme.domain.circuits.troncon import Troncon
from polytech.electromagnetisme.domain.conduction import (
    champ_d_ohm,
    conductivite,
    puissance_volumique_de_joule,
)
from polytech.electromagnetisme.domain.force_electromotrice import (
    fem_le_long_d_un_trajet,
)
from polytech.electromagnetisme.domain.magnetostatique.induction import (
    champ_induit_dynamique,
    fem_de_deplacement,
    force_electromotrice,
)
from polytech.electromagnetisme.domain.symboles import TEMPS


def test_la_force_electromotrice_est_la_circulation_du_champ_electromoteur() -> None:
    """III.2, planche 4 : `E_12 = integrale de 1 a 2 de E_e . dl`, en volts."""
    parametre = sp.Symbol("s", real=True)
    fem = fem_le_long_d_un_trajet((3, 0, 0), (parametre, 0, 0), parametre, 0, 2)
    assert fem == 6


def test_le_champ_induit_dynamique_est_v_vectoriel_b() -> None:
    """III.2, planche 6 : `E_d = v ^ B`."""
    assert champ_induit_dynamique((2.0, 0.0, 0.0), (0.0, 0.0, 3.0)) == pytest.approx(
        (0.0, -6.0, 0.0)
    )


def test_le_conducteur_rectiligne_de_la_planche_9() -> None:
    """III.2, planche 9 : `E = B l v`, la circulation de `E_d` le long du conducteur."""
    dynamique = champ_induit_dynamique((2.0, 0.0, 0.0), (0.0, 0.0, 3.0))
    parametre = sp.Symbol("s", real=True)
    le_long = fem_le_long_d_un_trajet(dynamique, (0, -parametre, 0), parametre, 0, 0.5)
    assert float(le_long) == pytest.approx(fem_de_deplacement(3.0, 0.5, 2.0))


def test_la_spire_de_la_planche_8() -> None:
    """III.2, planche 8 : pour une ligne de courant fermee, `E_12 = -d(phi_12)/dt`."""
    flux = 4 * sp.cos(3 * TEMPS)
    assert force_electromotrice(flux) == 12 * sp.sin(3 * TEMPS)


def test_la_loi_d_ohm_et_la_loi_de_joule_locales() -> None:
    """III.3, planches 4 a 6 : `sigma`, puis `E = rho i` et `p_j = rho i^2`."""
    sigma = conductivite([(2.0, 4.0)])
    assert sigma == pytest.approx(1.0)
    assert champ_d_ohm(1 / sigma, (3.0, 0.0, 0.0)) == pytest.approx((3.0, 0.0, 0.0))
    assert puissance_volumique_de_joule(1 / sigma, (3.0, 0.0, 0.0)) == pytest.approx(
        9.0
    )


def test_le_circuit_ferme_et_le_circuit_ouvert_de_la_planche_9() -> None:
    """III.3 : `E_12 = R_12 I_12` si `U_12 = 0` ; `U_12 = E_21` si `I_12 = 0`.

    La force electromotrice `E_12` de la planche est l'opposee de `E_21`.
    """
    troncon = Troncon(fem=-6.0, resistance=2.0)
    assert troncon.courant(0.0) == pytest.approx(3.0)
    assert troncon.tension(0.0) == pytest.approx(-6.0)
    assert troncon.puissance_de_joule(3.0) == pytest.approx(2.0 * 3.0**2)


def test_l_association_des_resistances_de_la_planche_10() -> None:
    """III.3 : en serie elles s'ajoutent, en parallele leurs inverses s'ajoutent."""
    serie = (Branche("a", "b", 2), Branche("b", "c", 3), Branche("c", "a", 6, 22))
    assert resoudre_par_les_mailles(serie, (0, 1))[0] == 2
    assert 1 / conductance_en_parallele(1 / 2, 1 / 3, 1 / 6) == pytest.approx(1.0)


def test_la_resistance_d_un_conducteur_homogene_et_le_tableau() -> None:
    """III.3, planches 11 et 12 : `R = rho l / S`, et le tableau des resistivites."""
    cuivre, alpha = METAUX["cuivre"]
    assert resistance_uniforme(cuivre, 1.0, 1e-6) == pytest.approx(17e-3)
    assert RESISTIVITE_DU_VERRE == 1e12
    assert resistivite_a_chaud(cuivre, 25.0, alpha) == pytest.approx(1.1 * cuivre)


RESEAU = (
    Branche("a", "b", 2, 10),
    Branche("b", "c", 3),
    Branche("c", "a", 6, -4),
    Branche("b", "a", 4),
    Branche("c", "b", 5, 7),
)
"""Trois noeuds, cinq branches, trois forces electromotrices."""


def test_les_deux_lois_de_kirchhoff_de_la_planche_15() -> None:
    """III.3 : la somme des courants a un noeud est nulle, celle des tensions aussi."""
    courants = resoudre_par_les_mailles(RESEAU, (0, 1))
    assert set(residus_des_noeuds(RESEAU, courants).values()) == {0}
    tensions = [b.resistance * i - b.fem for b, i in zip(RESEAU, courants, strict=True)]
    assert sp.simplify(tensions[0] + tensions[1] + tensions[2]) == 0


def test_le_theoreme_de_superposition_de_la_planche_17() -> None:
    """III.3 : chaque force electromotrice seule, les autres en court-circuit."""
    assert par_superposition(RESEAU, (0, 1)) == resoudre_par_les_mailles(RESEAU, (0, 1))


def test_thevenin_et_norton_des_planches_18_a_21() -> None:
    """III.3 : `I_b = E_T / (R + R_T)` ; `R_T = R_N`, `E_T = R_N I_N`, `I_cc = I_N`."""
    thevenin = Thevenin(tension_a_vide=12.0, resistance=4.0)
    charge = 2.0
    assert thevenin.tension_sur(charge) / charge == pytest.approx(12.0 / (2.0 + 4.0))
    norton = thevenin.en_norton()
    assert norton.resistance == thevenin.resistance
    assert norton.courant_de_court_circuit == pytest.approx(3.0)
    assert norton.en_thevenin() == thevenin
    assert norton.conductance() == pytest.approx(1 / 4.0)
