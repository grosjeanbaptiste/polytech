"""L'instant ou une montee du premier ordre atteint un seuil donne.

C'est la question de l'exercice 1 de la seance 3.1 -- un relais qui
s'enclenche quand le courant de sa bobine atteint 0,05 A -- et de tout
exercice qui demande un temps de montee.

La reponse est fermee : `t = -tau ln(1 - s / y_inf)` pour un depart a
zero. Elle evite de resoudre une equation transcendante, ce que le
calcul formel fait tres lentement sur des nombres rationnels : la
premiere ecriture de l'exercice passait par `solve` et mettait plus de
trois minutes.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.transitoires import (
    instant_du_seuil,
    reponse_du_premier_ordre,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

FINALE = sp.Rational(3)
CONSTANTE = sp.Rational(1, 4)


def test_la_reponse_vaut_bien_le_seuil_a_l_instant_rendu() -> None:
    """Controle par substitution dans la reponse temporelle elle-meme."""
    seuil = sp.Rational(1)
    instant = instant_du_seuil(FINALE, CONSTANTE, seuil)
    reponse = reponse_du_premier_ordre(sp.Integer(0), FINALE, CONSTANTE)
    assert sp.simplify(reponse.subs(TEMPS, instant) - seuil) == 0


def test_la_moitie_de_la_valeur_finale_est_atteinte_en_tau_ln_deux() -> None:
    """Le repere que tout cours donne : la demi-montee dure `tau ln 2`."""
    instant = instant_du_seuil(FINALE, CONSTANTE, FINALE / 2)
    assert float(instant) == pytest.approx(float(CONSTANTE) * math.log(2))


def test_un_seuil_egal_a_la_valeur_finale_n_est_jamais_atteint() -> None:
    """L'exponentielle s'en approche sans y toucher : aucun instant fini."""
    with pytest.raises(ValueError, match="jamais atteint"):
        instant_du_seuil(FINALE, CONSTANTE, FINALE)


def test_un_seuil_au_dela_de_la_valeur_finale_n_est_jamais_atteint() -> None:
    """Et rendre un nombre complexe ou negatif serait pire que de refuser."""
    with pytest.raises(ValueError, match="jamais atteint"):
        instant_du_seuil(FINALE, CONSTANTE, FINALE + 1)


def test_un_seuil_negatif_ou_nul_est_refuse() -> None:
    """La montee part de zero : un seuil nul est franchi a l'instant zero."""
    with pytest.raises(ValueError, match="seuil"):
        instant_du_seuil(FINALE, CONSTANTE, sp.Integer(0))
