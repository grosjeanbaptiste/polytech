"""Les reponses numeriques valent aux instants donnes, meme s'ils ne partent pas de zero.

Regression : `scipy.signal.step(T=...)` simule depuis `T[0]` comme si c'etait
l'origine ; `reponse_indicielle([1],[1,1],[0.75, 1.5])` rendait `[0, 0.5276]`
au lieu de `[0.5276, 0.7769]`.
"""

from __future__ import annotations

import numpy as np
import pytest

from polytech.signaux.domain.poles_numeriques import (
    reponse_impulsionnelle,
    reponse_indicielle,
)

INSTANTS = np.array([0.75, 1.5, 4.0])


def test_la_reponse_indicielle_ne_depend_pas_du_premier_instant() -> None:
    obtenue = reponse_indicielle([1.0], [1.0, 1.0], INSTANTS)
    assert obtenue == pytest.approx(1 - np.exp(-INSTANTS), abs=1e-6)


def test_la_reponse_impulsionnelle_ne_depend_pas_du_premier_instant() -> None:
    obtenue = reponse_impulsionnelle([1.0], [1.0, 1.0], INSTANTS)
    assert obtenue == pytest.approx(np.exp(-INSTANTS), abs=1e-6)


def test_un_instant_isole_est_lu_comme_les_autres() -> None:
    seul = reponse_indicielle([1.0], [1.0, 1.0], np.array([0.75]))
    assert seul == pytest.approx(1 - np.exp(-0.75), abs=1e-6)


def test_depuis_zero_les_valeurs_sont_inchangees() -> None:
    zero = np.array([0.0, 1.0, 2.0])
    assert reponse_indicielle([1.0], [1.0, 1.0], zero) == pytest.approx(
        1 - np.exp(-zero), abs=1e-6
    )


def test_un_instant_negatif_leve() -> None:
    with pytest.raises(ValueError, match="instant negatif"):
        reponse_indicielle([1.0], [1.0, 1.0], np.array([-1.0, 1.0]))
