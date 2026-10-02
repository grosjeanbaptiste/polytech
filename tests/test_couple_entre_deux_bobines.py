"""La seance 2.1, exercice 17 : le couple entre deux bobines couplees.

Deux bobines d'inductances propres `L_11` et `L_22` sont couplees par une
mutuelle qui depend de l'angle `alpha`. On donne sa pente en `alpha_0`,
en microhenrys **par degre**, et les deux courants.

Le manuscrit applique `C = (dW/d alpha)` a courants constants et encadre
`-2,6e-5` N.m. Le calcul est juste. Deux choses meritent d'etre dites :

- **le piege est l'unite de la pente.** Un couple est une energie par
  **radian**. Laisser la pente en degres donne un resultat cinquante-sept
  fois trop petit, sans que rien ne le signale ;
- **`L_11` et `L_22` ne servent a rien.** L'enonce les donne, et elles
  disparaissent a la derivation : elles ne dependent pas de l'angle.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.couple_des_bobines import (
    couple_a_courants_constants,
    energie_de_deux_bobines,
    pente_par_radian,
)

# Donnees de la transcription.
PENTE_PAR_DEGRE = -90e-6
PREMIER_COURANT = 1.0
SECOND_COURANT = 5e-3

ANGLE = sp.Symbol("alpha", real=True)


def test_la_pente_par_radian_est_cinquante_sept_fois_celle_par_degre() -> None:
    """`180 / pi` degres dans un radian : la pente grandit d'autant."""
    assert pente_par_radian(1.0) == pytest.approx(180 / math.pi)


def test_le_couple_est_celui_qu_encadre_le_manuscrit() -> None:
    """`C = I_1 I_2 dL_12/d alpha = -2,6e-5` N.m, a l'arrondi du manuscrit."""
    couple = couple_a_courants_constants(
        PREMIER_COURANT,
        SECOND_COURANT,
        pente_par_radian(PENTE_PAR_DEGRE),
    )
    assert couple == pytest.approx(-2.6e-5, abs=0.05e-5)


def test_laisser_la_pente_en_degres_donne_un_couple_trop_petit() -> None:
    """Le piege : `-4,5e-7` au lieu de `-2,6e-5`, sans aucun signal."""
    faux = couple_a_courants_constants(PREMIER_COURANT, SECOND_COURANT, PENTE_PAR_DEGRE)
    juste = couple_a_courants_constants(
        PREMIER_COURANT,
        SECOND_COURANT,
        pente_par_radian(PENTE_PAR_DEGRE),
    )
    assert faux == pytest.approx(-4.5e-7)
    assert juste / faux == pytest.approx(180 / math.pi)


def test_les_inductances_propres_disparaissent_a_la_derivation() -> None:
    """Le couple derive de l'energie ne contient ni `L_11` ni `L_22`.

    Route independante de la formule : on ecrit l'energie complete, on la
    derive par rapport a l'angle, et on constate que seul le terme mutuel
    survit.
    """
    propre_1, propre_2 = sp.symbols("L_11 L_22", positive=True)
    courant_1, courant_2 = sp.symbols("I_1 I_2", positive=True)
    mutuelle = sp.Function("L_12")(ANGLE)
    energie = energie_de_deux_bobines(
        propre_1, propre_2, mutuelle, courant_1, courant_2
    )
    couple = sp.diff(energie, ANGLE)
    assert couple.free_symbols.isdisjoint({propre_1, propre_2})
    assert sp.simplify(couple - courant_1 * courant_2 * sp.diff(mutuelle, ANGLE)) == 0


def test_le_couple_negatif_tend_a_diminuer_l_angle() -> None:
    """Le signe suit celui de la pente : le systeme va vers `L_12` croissant."""
    couple = couple_a_courants_constants(
        PREMIER_COURANT,
        SECOND_COURANT,
        pente_par_radian(PENTE_PAR_DEGRE),
    )
    assert couple < 0


def test_inverser_un_courant_inverse_le_couple() -> None:
    """Le couple est en `I_1 I_2` : il change de signe avec l'un des deux."""
    direct = couple_a_courants_constants(1.0, 2.0, 3.0)
    inverse = couple_a_courants_constants(-1.0, 2.0, 3.0)
    assert inverse == pytest.approx(-direct)


def test_l_energie_est_celle_de_l_inductance_serie_quand_le_courant_est_commun() -> (
    None
):
    """Oracle independant : `W = (L_11 + L_22 + 2 L_12) I^2 / 2`.

    Deux bobines en serie parcourues par le meme courant equivalent a une
    seule, d'inductance `L_11 + L_22 + 2 L_12` -- la regle de mise en
    serie que `CouplageMagnetique.en_serie` porte deja. Sans ce test,
    rien ne fixait les termes propres de l'energie : seule sa derivee
    etait eprouvee, et elle les ignore.
    """
    propre_1, propre_2, mutuelle = sp.symbols("L_11 L_22 L_12", positive=True)
    courant = sp.Symbol("I", positive=True)
    energie = energie_de_deux_bobines(propre_1, propre_2, mutuelle, courant, courant)
    serie = propre_1 + propre_2 + 2 * mutuelle
    assert sp.simplify(energie - serie * courant**2 / 2) == 0


def test_sans_couplage_l_energie_est_la_somme_des_deux_energies_propres() -> None:
    """`L_12 = 0` : chaque bobine stocke `L I^2 / 2`, sans terme croise."""
    propre_1, propre_2 = sp.symbols("L_11 L_22", positive=True)
    courant_1, courant_2 = sp.symbols("I_1 I_2", positive=True)
    energie = energie_de_deux_bobines(propre_1, propre_2, 0, courant_1, courant_2)
    attendue = propre_1 * courant_1**2 / 2 + propre_2 * courant_2**2 / 2
    assert sp.simplify(energie - attendue) == 0
