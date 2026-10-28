"""Les charges reparties et la pesanteur.

Source : `MecaniqueRationnelle1/SYLLABUS/Syllabus_Partie_1_Statique.pdf`,
section 2.2.6, pages 96 et 97 -- relation (2.5) et le moment `-P x_G` --, et
l'exemple 2.2.8 de la poutre bi-appuyee, pages 99 et 100.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.mecanique.domain.pesanteur import PESANTEUR, poids
from polytech.mecanique.domain.statique.charges import ChargeRepartie

X, L, Q = sp.symbols("x L q", positive=True)


def test_la_charge_totale_est_l_integrale_de_la_densite() -> None:
    """Page 97, (2.5) : `P = integrale de p(x) dx`."""
    assert ChargeRepartie(Q, X, 0, L).charge_totale() == Q * L
    assert ChargeRepartie(Q * X / L, X, 0, L).charge_totale() == Q * L / 2


def test_le_moment_a_l_origine_est_la_charge_fois_l_abscisse_du_centre() -> None:
    """Page 97 : `integrale de p(x) x dx = P x_G`."""
    uniforme = ChargeRepartie(Q, X, 0, L)
    assert uniforme.abscisse_du_centre() == L / 2
    triangulaire = ChargeRepartie(Q * X / L, X, 0, L)
    assert triangulaire.abscisse_du_centre() == 2 * L / 3
    ecart = triangulaire.moment_a_l_origine() - triangulaire.charge_totale() * 2 * L / 3
    assert sp.simplify(ecart) == 0


def test_la_poutre_bi_appuyee_de_l_exemple_2_2_8() -> None:
    """Pages 99 et 100 : 20 kN/m sur 3 m, 60 kN a 1,5 m ; reactions 45 et 15."""
    charge = ChargeRepartie(sp.Integer(20), X, 0, 3)
    assert charge.charge_totale() == 60
    assert charge.abscisse_du_centre() == sp.Rational(3, 2)
    en_b = charge.moment_a_l_origine() / 2
    assert en_b == 45
    assert charge.charge_totale() - en_b == 15
    # 20 kN/m en N/m : les reactions valent 45 000 N et 15 000 N, pas 45 N et 15 N.
    en_newtons = ChargeRepartie(sp.Integer(20_000), X, 0, 3)
    assert en_newtons.moment_a_l_origine() / 2 == 45_000
    assert en_newtons.charge_totale() - en_newtons.moment_a_l_origine() / 2 == 15_000


def test_le_torseur_d_une_charge_vers_le_bas() -> None:
    """Page 97 : resultante `-P u_y`, moment `-P x_G u_z` au pole origine."""
    torseur = ChargeRepartie(sp.Integer(20), X, 0, 3).torseur_a_l_origine()
    assert torseur.resultante == (0, -60, 0)
    assert torseur.moment == (0, 0, -90)


def test_une_charge_sans_resultante_n_a_pas_de_centre() -> None:
    """Une charge de resultante nulle est un couple : le quotient n'existe pas."""
    with pytest.raises(ValueError, match="charge totale nulle"):
        ChargeRepartie(X - L / 2, X, 0, L).abscisse_du_centre()


def test_le_poids_est_la_masse_fois_la_pesanteur() -> None:
    """Section 1.7.4, page 22 : la force de pesanteur, `P = m g`."""
    assert pytest.approx(9.81) == PESANTEUR
    assert poids(2.0) == pytest.approx(19.62)
    with pytest.raises(ValueError, match="masse negative :"):
        poids(-1.0)
