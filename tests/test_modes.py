"""La reponse impulsionnelle decomposee en modes, un par pole.

`H(p) = somme r_i / (p - p_i)` donne `h(t) = somme r_i e^{p_i t}`. Garder
certains termes, c'est « negliger l'effet des autres poles » au sens le
plus litteral. L'oracle est la reconstruction : tous les modes ensemble
doivent redonner la reponse que `scipy.signal.impulse` calcule par un
tout autre chemin, l'integration d'une representation d'etat.
"""

from __future__ import annotations

import numpy as np
import pytest

from polytech.signaux.domain.modes import contribution_des_modes, residus
from polytech.signaux.domain.poles_numeriques import reponse_impulsionnelle

NUMERATEUR = [1.0, 3.0]
DENOMINATEUR = np.polymul([1.0, 1.0], [1.0, 2.0, 10.0]).tolist()


def test_tous_les_modes_redonnent_la_reponse_impulsionnelle() -> None:
    instants = np.linspace(0.0, 4.0, 81)
    poles = residus(NUMERATEUR, DENOMINATEUR)[1]
    reconstruite = contribution_des_modes(NUMERATEUR, DENOMINATEUR, poles, instants)
    attendue = reponse_impulsionnelle(NUMERATEUR, DENOMINATEUR, instants)
    assert np.allclose(reconstruite, attendue, atol=1e-6)


def test_un_premier_ordre_a_pour_residu_son_gain() -> None:
    valeurs, poles = residus([2.0], [1.0, 4.0])
    assert poles.tolist() == [-4.0]
    assert valeurs.tolist() == pytest.approx([2.0])


def test_un_pole_absent_de_la_transmittance_est_refuse() -> None:
    with pytest.raises(ValueError, match="pole absent"):
        contribution_des_modes(
            NUMERATEUR, DENOMINATEUR, np.array([-7.0]), np.array([0.0])
        )


def test_un_numerateur_aussi_haut_que_le_denominateur_est_refuse() -> None:
    """Un terme direct serait une impulsion de Dirac : pas un mode."""
    with pytest.raises(ValueError, match="strictement propre"):
        residus([1.0, 0.0], [1.0, 1.0])
