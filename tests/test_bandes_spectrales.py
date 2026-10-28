"""`absorptivite_par_bandes` et `rayon_critique` : bornes et refus."""

from __future__ import annotations

import pytest

from polytech.maths.domain.comparaison import proches
from polytech.thermique.domain.conduction.rayon_critique import rayon_critique
from polytech.thermique.domain.rayonnement.bandes import absorptivite_par_bandes


def test_un_facteur_uniforme_se_retrouve_tel_quel() -> None:
    """Une seule valeur sur toutes les bandes : la moyenne est cette valeur."""
    assert proches(absorptivite_par_bandes((0.4, 0.4), (3e-6,), 1000.0), 0.4, 1e-12)


def test_sans_coupure_la_moyenne_est_le_facteur() -> None:
    """Une bande unique, de zero a l'infini."""
    assert proches(absorptivite_par_bandes((0.7,), (), 500.0), 0.7, 1e-12)


@pytest.mark.parametrize(
    ("facteurs", "coupures", "motif"),
    [
        ((0.5,), (1e-6,), "un de plus attendu"),
        ((0.5, 1.5), (1e-6,), "hors de"),
        ((0.5, 0.5, 0.5), (2e-6, 1e-6), "non croissantes"),
    ],
)
def test_les_entrees_incoherentes_sont_refusees(
    facteurs: tuple[float, ...], coupures: tuple[float, ...], motif: str
) -> None:
    """Jamais de moyenne calculee sur un decoupage faux."""
    with pytest.raises(ValueError, match=motif):
        absorptivite_par_bandes(facteurs, coupures, 1000.0)


def test_le_rayon_critique_refuse_une_conductivite_nulle() -> None:
    """`lambda / K_e` n'a de sens que pour des grandeurs positives."""
    with pytest.raises(ValueError, match="conductivite"):
        rayon_critique(0.0, 8.0)
    with pytest.raises(ValueError, match="coefficient"):
        rayon_critique(0.04, 0.0)
