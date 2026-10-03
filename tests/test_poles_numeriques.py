"""Poles, zeros et reponses d'une transmittance numerique.

Le TP fait calculer les poles par `tf2zpk` puis lire sur les reponses la
constante de temps et la frequence d'oscillation. Les oracles sont ceux
du cours : un premier ordre `1/(p + a)` a un pole en `-a` et une
constante de temps `1/a` ; un second ordre `(p + s)^2 + w^2` oscille a
`w / 2 pi` hertz.
"""

from __future__ import annotations

import numpy as np
import pytest

from polytech.signaux.domain.poles_numeriques import (
    approximation_dominante,
    constante_de_temps,
    est_stable,
    frequence_d_oscillation,
    poles_dominants,
    poles_et_zeros,
    reponse_impulsionnelle,
    reponse_indicielle,
    routh_du_troisieme_degre,
)


def test_un_premier_ordre_a_son_pole_en_moins_a() -> None:
    poles, zeros, gain = poles_et_zeros([2.0], [1.0, 4.0])
    assert poles.tolist() == [-4.0]
    assert zeros.size == 0
    assert gain == pytest.approx(2.0)


def test_la_constante_de_temps_est_l_inverse_de_l_amortissement() -> None:
    assert constante_de_temps(complex(-4, 3)) == pytest.approx(0.25)


def test_un_pole_instable_n_a_pas_de_constante_de_temps() -> None:
    with pytest.raises(ValueError, match="instable"):
        constante_de_temps(complex(0.5, 3))


def test_la_frequence_d_oscillation_est_la_partie_imaginaire_sur_deux_pi() -> None:
    assert frequence_d_oscillation(complex(-1, 2 * np.pi)) == pytest.approx(1.0)


def test_les_poles_dominants_sont_les_plus_proches_de_l_axe() -> None:
    poles = np.array([-5.0, complex(-1, 2), complex(-1, -2), -3.0])
    dominants = poles_dominants(poles)
    assert sorted(dominants.imag.tolist()) == [-2.0, 2.0]


def test_un_pole_a_partie_reelle_positive_rend_instable() -> None:
    assert not est_stable(np.array([complex(0.2, 3), complex(0.2, -3), -1.4]))


def test_des_poles_a_gauche_rendent_stable() -> None:
    assert est_stable(np.array([-1.0, complex(-2, 1)]))


def test_routh_du_troisieme_degre_compare_a2_a1_a_a3_a0() -> None:
    """`p^3 + p^2 + 10 p + 15` : `1 x 10 - 1 x 15 = -5`, donc instable."""
    assert routh_du_troisieme_degre([1.0, 1.0, 10.0, 15.0]) == pytest.approx(-5.0)


def test_routh_refuse_un_autre_degre() -> None:
    with pytest.raises(ValueError, match="troisieme degre"):
        routh_du_troisieme_degre([1.0, 2.0, 3.0])


def test_la_reponse_impulsionnelle_d_un_premier_ordre_est_une_exponentielle() -> None:
    instants = np.linspace(0.0, 1.0, 11)
    reponse = reponse_impulsionnelle([1.0], [1.0, 4.0], instants)
    assert np.allclose(reponse, np.exp(-4.0 * instants), atol=1e-6)


def test_la_reponse_indicielle_d_un_premier_ordre_monte_vers_son_gain() -> None:
    instants = np.linspace(0.0, 1.0, 11)
    reponse = reponse_indicielle([4.0], [1.0, 4.0], instants)
    assert np.allclose(reponse, 1.0 - np.exp(-4.0 * instants), atol=1e-6)


def test_l_approximation_dominante_garde_le_gain_statique() -> None:
    """`H(0)` est conserve : la valeur finale de la reponse indicielle aussi."""
    numerateur = [10.0]
    denominateur = np.polymul([1.0, 2.0, 5.0], [1.0, 10.0]).tolist()
    approche_num, approche_den = approximation_dominante(numerateur, denominateur)
    assert approche_num[-1] / approche_den[-1] == pytest.approx(10.0 / 50.0)
    assert np.allclose(approche_den, [1.0, 2.0, 5.0])
