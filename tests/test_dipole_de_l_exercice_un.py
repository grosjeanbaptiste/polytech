"""Ce que l'exercice 1 demande au-dela de la conversion Thevenin-Norton.

Le domaine portait deja les deux ecritures et leur equivalence. Deux
grandeurs manquaient, et ce sont celles des questions (c) et (e) de
l'exercice :

* la **conductance** de l'ecriture de Norton, que la transcription note
  `G_0 = 1 ohm^-1`. Le cours la donne parce qu'une source de courant en
  parallele se decrit naturellement par une conductance, de meme qu'une
  source de tension en serie se decrit par une resistance ;
* la **puissance echangee** par la source de courant, et surtout son
  **signe**. La transcription conclut `P_12 = -1 kW` et commente : « le
  signe negatif : la source de courant **fournit** de la puissance au
  reste du circuit ». C'est la convention recepteur, et c'est le seul
  endroit de l'exercice ou un resultat juste en valeur absolue serait
  faux.
"""

from __future__ import annotations

import pytest

from polytech.electromagnetisme.domain.circuits.stationnaires import (
    Norton,
    puissance_echangee,
    source_de_tension_et_source_de_courant,
)

# Donnees de la transcription : 110 V en serie avec 1 ohm, en parallele
# avec une source de 10 A.
TENSION = 110.0
RESISTANCE = 1.0
COURANT = 10.0
CHARGE = 5.0


def test_la_conductance_est_l_inverse_de_la_resistance() -> None:
    """`G_0 = 1 / R`, en siemens. La transcription l'ecrit `1 ohm^-1`."""
    norton = Norton(courant_de_court_circuit=120.0, resistance=RESISTANCE)
    assert norton.conductance() == pytest.approx(1.0)
    assert Norton(courant_de_court_circuit=1.0, resistance=4.0).conductance() == (
        pytest.approx(0.25)
    )


def test_la_puissance_echangee_est_negative_quand_la_source_fournit() -> None:
    """`P_12 = U_12 I_12` avec `I_12 = -10 A` : la source **fournit**.

    C'est le resultat de la question (e), et son signe est tout l'enjeu :
    en convention recepteur, une puissance negative signifie que
    l'element delivre au reste du circuit. Un corrige qui rendrait
    `+1 kW` serait juste en grandeur et faux en sens.
    """
    thevenin = source_de_tension_et_source_de_courant(TENSION, RESISTANCE, COURANT)
    tension_sur_la_charge = thevenin.tension_sur(CHARGE)
    assert puissance_echangee(tension_sur_la_charge, -COURANT) == pytest.approx(-1000.0)


def test_la_tension_sur_la_charge_vaut_cent_volts() -> None:
    """Chemin de la transcription : `U_12 = R I = 5 x 20 = 100 V`.

    Et le module y arrive autrement -- par le diviseur de tension du
    dipole de Thevenin, `U = E_0 R_c / (R_0 + R_c)`. Les deux doivent
    concorder, et c'est ce qui verifie l'un par l'autre.
    """
    thevenin = source_de_tension_et_source_de_courant(TENSION, RESISTANCE, COURANT)
    par_le_diviseur = thevenin.tension_sur(CHARGE)
    courant_debite = thevenin.tension_a_vide / (thevenin.resistance + CHARGE)
    par_la_loi_d_ohm = CHARGE * courant_debite
    assert par_le_diviseur == pytest.approx(100.0)
    assert par_la_loi_d_ohm == pytest.approx(par_le_diviseur)


def test_le_courant_debite_sur_cinq_ohms_vaut_vingt_amperes() -> None:
    """Question (d) : `120 = (5 + 1) I`, donc `I = 20 A`."""
    thevenin = source_de_tension_et_source_de_courant(TENSION, RESISTANCE, COURANT)
    assert thevenin.tension_a_vide / (thevenin.resistance + CHARGE) == (
        pytest.approx(20.0)
    )


def test_une_puissance_echangee_a_courant_nul_est_nulle() -> None:
    """Un element qui ne conduit pas n'echange rien, quelle que soit sa tension."""
    assert puissance_echangee(100.0, 0.0) == pytest.approx(0.0)
