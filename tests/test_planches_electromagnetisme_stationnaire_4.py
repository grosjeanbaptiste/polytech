"""Planches d'electromagnetisme, chapitre III.4 : inductances et couplage.

Sources : `Electromagnetisme/SLIDES/Slides_03_Etats_stationnaires_partie_4.pdf`,
planches 2 a 21, lues sur les pages rendues, et la note manuscrite du
transparent 15 (`Slides_03_Note_T15_Identite_de_Gauss.pdf`).

Le fascicule resume le chapitre 9 du syllabus et y ajoute une regle
pratique : le signe d'une inductance mutuelle se lit sur les bornes
pointees des deux circuits (planches 16 a 20).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.coefficients import (
    residu_de_l_identite_de_gauss,
)
from polytech.electromagnetisme.domain.electrostatique.lois import (
    rotationnel_cartesien,
)
from polytech.electromagnetisme.domain.magnetostatique.bornes_pointees import (
    flux_des_circuits_couples,
    inductance_en_serie,
    signe_de_la_mutuelle,
)
from polytech.electromagnetisme.domain.magnetostatique.champ_magnetique import (
    potentiel_scalaire_du_fil,
)
from polytech.electromagnetisme.domain.magnetostatique.couplage import (
    CouplageMagnetique,
)
from polytech.electromagnetisme.domain.magnetostatique.inductances import (
    SystemeDeCircuits,
    flux_d_un_circuit_de_section_finie,
    flux_total,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES

L1, L2, M = sp.symbols("L_1 L_2 L_12", positive=True)
COURANT = sp.Symbol("I", real=True)


def test_hors_des_courants_le_champ_derive_d_un_potentiel_scalaire() -> None:
    """Planche 3 : `rot H = 0`, donc `H = -grad U`, autour d'un fil."""
    potentiel = potentiel_scalaire_du_fil(COURANT)
    champ = tuple(-sp.diff(potentiel, axe) for axe in CARTESIENNES)
    assert rotationnel_cartesien(champ) == (0, 0, 0)


def test_le_flux_d_un_circuit_de_section_finie_est_une_moyenne_ponderee() -> None:
    """Planche 5 : `phi = (1 / I) integrale de phi_0 dI`."""
    assert flux_d_un_circuit_de_section_finie([(4, 1), (10, 2)]) == 8


def test_les_equations_electromagnetiques_sortent_de_l_identite_de_gauss() -> None:
    """Planches 10 a 13 et note du transparent 15 : `phi_i = somme L_ik I_k`.

    Le second etat de la note : un ampere dans le circuit `i`, aucun
    ailleurs, et les flux `L_i1`, `L_i2`, `L_i3` qui s'installent.
    """
    matrice = sp.Matrix(
        3, 3, lambda i, k: sp.Symbol(f"L_{min(i, k) + 1}{max(i, k) + 1}")
    )
    courants = sp.symbols("I_1:4", real=True)
    flux = sp.symbols("phi_1:4", real=True)
    rang = 1
    unitaire = [1 if k == rang else 0 for k in range(3)]
    residu = residu_de_l_identite_de_gauss(
        courants, flux, unitaire, list(matrice.row(rang))
    )
    (solution,) = sp.solve(residu, flux[rang])
    attendu = SystemeDeCircuits(matrice).flux(courants)[rang]
    assert sp.expand(solution - attendu) == 0


def test_la_matrice_des_inductances_est_symetrique() -> None:
    """Planche 13 : `L = L^T`, « en appliquant l'identite de Gauss »."""
    asymetrique = sp.Matrix([[2, 1], [sp.Rational(1, 2), 3]])
    with pytest.raises(ValueError, match="non symetriques"):
        SystemeDeCircuits(asymetrique)


def test_la_mutuelle_est_positive_si_les_courants_entrent_par_les_points() -> None:
    """Planches 17 a 19, cas 1 a 4 : `+` si les deux entrent ou les deux sortent."""
    assert (
        signe_de_la_mutuelle(premier_par_le_point=True, second_par_le_point=True) == 1
    )
    assert (
        signe_de_la_mutuelle(premier_par_le_point=True, second_par_le_point=False) == -1
    )
    assert (
        signe_de_la_mutuelle(premier_par_le_point=False, second_par_le_point=True) == -1
    )
    assert (
        signe_de_la_mutuelle(premier_par_le_point=False, second_par_le_point=False) == 1
    )


def test_les_flux_de_deux_inductances_couplees() -> None:
    """Planche 17 : `phi_1 = L_11 I_1 +- L_12 I_2`, le signe venant des points."""
    i1, i2 = sp.symbols("I_1 I_2", real=True)
    cas_1 = flux_des_circuits_couples(
        (L1, L2), M, (i1, i2), premier_par_le_point=True, second_par_le_point=True
    )
    assert cas_1 == (L1 * i1 + M * i2, M * i1 + L2 * i2)
    cas_2 = flux_des_circuits_couples(
        (L1, L2), M, (i1, i2), premier_par_le_point=True, second_par_le_point=False
    )
    assert cas_2 == (L1 * i1 - M * i2, -M * i1 + L2 * i2)


def test_l_association_en_serie_depend_du_sens_du_flux_mutuel() -> None:
    """Planche 20 : `L_1 + L_2 + 2 L_12`, ou `L_1 + L_2 - 2 L_12`.

    La planche imprime deux fois « L_1 + L_1 » pour `L_1 + L_2` ; ses
    propres developpements, juste a gauche, portent bien `L_2`.
    """
    aidante = inductance_en_serie(
        (L1, L2), M, premier_par_le_point=True, second_par_le_point=True
    )
    opposee = inductance_en_serie(
        (L1, L2), M, premier_par_le_point=True, second_par_le_point=False
    )
    assert aidante == L1 + L2 + 2 * M
    assert opposee == L1 + L2 - 2 * M
    assert aidante != 2 * L1 + 2 * M
    couplage = CouplageMagnetique(3.0, 5.0, 1.5)
    assert couplage.en_serie(aidant=True) == pytest.approx(
        float(aidante.subs({L1: 3, L2: 5, M: 1.5}))
    )
    assert couplage.en_serie(aidant=False) == pytest.approx(
        float(opposee.subs({L1: 3, L2: 5, M: 1.5}))
    )


def test_le_flux_total_d_une_bobine_est_n_fois_le_flux_par_spire() -> None:
    """Planche 21 : `phi = N Phi = L I`."""
    assert flux_total(spires=200, par_spire=2.5e-5) == pytest.approx(5e-3)
