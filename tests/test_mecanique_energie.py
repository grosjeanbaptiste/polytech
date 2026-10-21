"""L'energie, d'apres le syllabus « Notions fondamentales », chapitre 6.

Source : `PhysiqueGenerale/SYLLABUS/Syllabus_Notions_fondamentales_2026_2027.pdf`.
Les tests de lien font repondre deux chapitres aux memes questions : le
theoreme de l'energie cinetique (6.25) se refait par la cinematique (2.15)
et la deuxieme loi, et la vitesse tiree de la conservation (6.37) est celle
que donne le MRUA a acceleration -g.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.mecanique.domain.dynamique.cinematique import vitesse_par_la_position
from polytech.mecanique.domain.dynamique.energie import (
    energie_cinetique,
    energie_potentielle_pesanteur,
    travail,
    vitesse_par_l_energie,
)

m, g, vi, longueur, acc = sp.symbols("m g v_i l a", positive=True)


def test_travail_eq_6_1() -> None:
    """(6.1) W = F l cos theta, p. 62 : une force perpendiculaire ne travaille pas."""
    assert travail(10.0, 2.0, 0.0) == pytest.approx(20.0)
    assert travail(10.0, 2.0, sp.pi / 2) == 0


def test_energie_cinetique_eq_6_24() -> None:
    """(6.24) Ec = m v^2 / 2, p. 68."""
    assert energie_cinetique(2.0, 3.0) == pytest.approx(9.0)


def test_theoreme_de_l_energie_cinetique_eq_6_25_par_la_cinematique() -> None:
    """(6.21) a (6.25), p. 68 : W = F l = m a l, et v_f par (2.15) ; W = Ec_f - Ec_i."""
    vf = vitesse_par_la_position(vi, acc, 0, longueur)
    w = travail(m * acc, longueur, 0)
    assert sp.simplify(w - (energie_cinetique(m, vf) - energie_cinetique(m, vi))) == 0


def test_energie_potentielle_eq_6_32_est_le_travail_du_poids_change_de_signe() -> None:
    """(6.31)-(6.32), p. 71 : Ep(yf) - Ep(yi) = -W du poids = m g (yf - yi)."""
    yi, yf = sp.symbols("y_i y_f", real=True)
    travail_du_poids = travail(
        m * g, yi - yf, 0
    )  # le poids vers le bas, la chute yi - yf
    variation = energie_potentielle_pesanteur(m, g, yf) - energie_potentielle_pesanteur(
        m, g, yi
    )
    assert sp.simplify(variation + travail_du_poids) == 0


def test_vitesse_par_la_conservation_eq_6_37_est_celle_du_mrua() -> None:
    """(6.37), p. 72, et (2.15), p. 18 : deux chapitres, une meme vitesse."""
    yi, yf = sp.symbols("y_i y_f", real=True)
    par_l_energie = vitesse_par_l_energie(vi, yi, yf, g)
    par_la_cinematique = vitesse_par_la_position(vi, -g, yi, yf)
    assert sp.simplify(par_l_energie - par_la_cinematique) == 0


def test_chute_de_cinq_metres() -> None:
    """Un cas chiffre : lache de 5 m, g = 9,81 m/s^2 : v = racine(2 g h)."""
    assert vitesse_par_l_energie(0.0, 5.0, 0.0, 9.81) == pytest.approx(
        (2 * 9.81 * 5) ** 0.5
    )


def test_une_hauteur_jamais_atteinte_leve() -> None:
    """Monter de 10 m avec 1 m/s : l'energie ne suffit pas."""
    with pytest.raises(ValueError, match="jamais atteinte"):
        vitesse_par_l_energie(1.0, 0.0, 10.0, 9.81)
    # Juste trop haut : Ec finale = 0,5 - 0,5886 < 0 par kilogramme.
    with pytest.raises(ValueError, match="jamais atteinte"):
        vitesse_par_l_energie(1.0, 0.0, 0.06, 9.81)
