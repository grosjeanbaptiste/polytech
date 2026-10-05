"""Planches d'electromagnetisme, chapitre III.1 : lois du champ magnetique.

Source : `Electromagnetisme/SLIDES/Slides_03_Etats_stationnaires_partie_1.pdf`,
planches 2 a 36, lues sur les pages rendues.

Le fascicule resume les chapitres 7 et 8 du syllabus : aucun code neuf.
Chaque relation surlignee est eprouvee sur le cas que la planche dessine :
le troncon dans un champ uniforme de la planche 16, le fil infiniment
long de la planche 29, la nappe de la planche 36.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.courants import (
    Espece,
    courant_de_conduction,
    courant_de_convection,
    courant_de_polarisation,
    intensite_a_travers,
)
from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
    rotationnel_cartesien,
)
from polytech.electromagnetisme.domain.electrostatique.passage import (
    residu_tangentiel,
    saut_normal,
)
from polytech.electromagnetisme.domain.etats import Etat, etat_du_systeme
from polytech.electromagnetisme.domain.magnetostatique.actions import (
    force_electromagnetique,
)
from polytech.electromagnetisme.domain.magnetostatique.aimantation import (
    aimantation,
    courant_surfacique_fictif,
    courant_volumique_fictif,
    moment_amperien,
)
from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    PERMEABILITE,
    induction_d_un_element,
)
from polytech.electromagnetisme.domain.magnetostatique.champ_magnetique import (
    champ_magnetique,
)
from polytech.electromagnetisme.domain.magnetostatique.induction import (
    fem_de_deplacement,
)
from polytech.electromagnetisme.domain.magnetostatique.lois import (
    flux_magnetique,
    force_sur_un_element_de_courant,
)
from polytech.electromagnetisme.domain.magnetostatique.potentiel_vecteur import (
    changement_de_jauge,
    est_en_jauge_de_coulomb,
    induction_du_potentiel,
)
from polytech.electromagnetisme.domain.maxwell import (
    equation_de_continuite,
    residu_de_gauss_magnetique,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES, TEMPS

X, Y, Z = CARTESIENNES
NUL = (0, 0, 0)


def test_le_courant_volumique_a_trois_termes() -> None:
    """Planches 4 et 5 : conduction (mouvement relatif), convection, polarisation."""
    electrons = Espece(-5, (3, 0, 0))
    ions = Espece(5, (1, 0, 0))
    entrainement = (1, 0, 0)
    conduction = courant_de_conduction([electrons, ions], entrainement)
    assert conduction == (-10, 0, 0)
    # Les ions sont fixes au conducteur : leur vitesse relative est nulle.
    assert courant_de_conduction([ions], entrainement) == NUL
    assert courant_de_convection(4, entrainement) == (4, 0, 0)
    assert courant_de_polarisation((3 * TEMPS, 0, 0)) == (3, 0, 0)


def test_l_intensite_est_le_flux_du_courant_volumique() -> None:
    """Planche 6 : `I = integrale de u_n . i dS`."""
    assert intensite_a_travers((0, 0, 2), (0, 0, 1), 3) == 6
    assert intensite_a_travers((2, 0, 0), (0, 0, 1), 3) == 0


def test_l_equation_de_continuite_et_la_stationnarite() -> None:
    """Planches 7, 9 et 10 : la continuite, puis `div i_cd = 0` en stationnaire."""
    charge = sp.exp(-TEMPS) * sp.cos(X)
    assert equation_de_continuite((sp.exp(-TEMPS) * sp.sin(X), 0, 0), charge) == 0
    # En regime stationnaire, les lignes de courant sont fermees.
    tourbillon = (-Y, X, 0)
    assert equation_de_continuite(tourbillon, sp.Integer(7)) == 0
    assert divergence_cartesienne(tourbillon) == 0
    etat = etat_du_systeme(charges_en_mouvement=True, courant_constant=True)
    assert etat is Etat.STATIONNAIRE


def test_la_force_de_lorentz_et_la_force_sur_un_element_de_courant() -> None:
    """Planches 14 et 15 : `F = q (E + v ^ B)` et `dF = dI dl ^ B`."""
    force = force_electromagnetique(
        2.0, (1.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)
    )
    assert force == pytest.approx((2.0 * (1.0 + 12.0), 0.0, 0.0))
    element = force_sur_un_element_de_courant(5.0, (0.0, 0.1, 0.0), (0.0, 0.0, 2.0))
    assert element == pytest.approx((1.0, 0.0, 0.0))


def test_le_troncon_de_la_planche_16_subit_b_l_i() -> None:
    """`F = B l I`, l'induction etant normale au troncon."""
    force = force_sur_un_element_de_courant(5.0, (0.0, 0.4, 0.0), (0.0, 0.0, 2.0))
    assert force[0] == pytest.approx(2.0 * 0.4 * 5.0)
    # La meme geometrie donne la force electromotrice `B l v` du chapitre III.2.
    assert fem_de_deplacement(2.0, 0.4, 5.0) == pytest.approx(force[0])


def test_le_flux_magnetique_et_l_induction_solenoidale() -> None:
    """Planches 17, 19 et 20 : `phi = B S` ; `div B = 0`, donc `B = rot A`."""
    assert flux_magnetique(0.5, 0.02) == pytest.approx(0.01)
    quelconque = tuple(sp.Function(n)(X, Y, Z) for n in ("A_x", "A_y", "A_z"))
    assert residu_de_gauss_magnetique(induction_du_potentiel(quelconque)) == 0


def test_le_potentiel_vecteur_est_defini_a_un_gradient_pres() -> None:
    """Planche 22 : `rot (A + grad(phi)) = rot A`, et la jauge de Coulomb."""
    vecteur = (-Y / 2, X / 2, 0)
    assert est_en_jauge_de_coulomb(vecteur)
    autre = changement_de_jauge(vecteur, X**2 * Y)
    assert induction_du_potentiel(autre) == induction_du_potentiel(vecteur)
    assert not est_en_jauge_de_coulomb(autre)


def test_le_moment_amperien_d_un_electron_est_oppose_a_la_normale() -> None:
    """Planche 23 : `m = (q / T) S u_n`, oppose a `u_n` pour un electron lie."""
    charge, periode, aire = sp.symbols("e T S", positive=True)
    moment = moment_amperien(
        charge=-charge, periode=periode, aire=aire, normale=(0, 0, 1)
    )
    assert moment == (0, 0, -charge * aire / periode)
    assert aimantation([moment, moment], 4) == (0, 0, -charge * aire / (2 * periode))
    # « Nulle si les moments ont une orientation totalement aleatoire. »
    oppose = tuple(-composante for composante in moment)
    assert aimantation([moment, oppose], 4) == NUL


def test_une_aimantation_uniforme_ne_laisse_de_courants_qu_en_surface() -> None:
    """Planches 25 et 26 : `i' = rot M` nul, `i'_S = u_n ^ M` sur le contour."""
    uniforme = (0, 0, sp.Symbol("M_0", positive=True))
    assert courant_volumique_fictif(uniforme) == NUL
    surfacique = courant_surfacique_fictif(uniforme, normale_interieure=(-1, 0, 0))
    assert surfacique == (0, uniforme[2], 0)
    assert courant_volumique_fictif((0, 0, 3 * X)) != NUL


def test_l_induction_d_un_element_de_courant_est_tangentielle() -> None:
    """Planche 29 : pour un fil selon `z`, `dB` est normal au plan `(dl, R)`."""
    induction = induction_d_un_element(2, (0, 0, 1), (3, 0, 0))
    assert induction[0] == 0
    assert induction[2] == 0
    assert sp.simplify(induction[1] - PERMEABILITE * 2 / (4 * sp.pi * 9)) == 0


def test_le_theoreme_d_ampere_local_dans_un_conducteur_cylindrique() -> None:
    """Planches 33 et 34 : `rot H = i`, avec `H = B / mu_0 - M`.

    A l'interieur d'un fil d'axe `z` parcouru par un courant volumique
    uniforme `i_0` : `H = (i_0 / 2) (-y, x, 0)`.
    """
    densite = sp.Symbol("i_0", positive=True)
    champ = (-densite * Y / 2, densite * X / 2, 0)
    assert rotationnel_cartesien(champ) == (0, 0, densite)
    induction = tuple(PERMEABILITE * composante for composante in champ)
    retour = champ_magnetique(induction, NUL)
    assert [sp.simplify(a - b) for a, b in zip(retour, champ, strict=True)] == [0] * 3


def test_les_continuites_de_l_induction_et_du_champ_magnetique() -> None:
    """Planche 36 : `B_n` continue, `H_t` saute du courant surfacique."""
    nappe = sp.Symbol("K", positive=True)
    dessus, dessous = (0, -nappe / 2, 0), (0, nappe / 2, 0)
    assert saut_normal(dessus, dessous, normale_1=(0, 0, 1)) == 0
    assert residu_tangentiel(dessus, dessous, normale_1=(0, 0, 1)) == (nappe, 0, 0)
