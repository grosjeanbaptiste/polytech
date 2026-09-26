"""Le budget doit recenser la temperature, et sa conclusion en depend.

Avec les seules incertitudes de lecture, le corrige concluait que la
lecture du mercure domine. En ajoutant la dependance en temperature des
masses volumiques -- temperature que l'enonce ne donne jamais -- c'est
elle qui devient la source principale.
"""

import pytest

from polytech.fluides.application import enonce
from polytech.fluides.domain.grandeurs import Pression
from polytech.fluides.domain.statique.incertitude import (
    BudgetDIncertitude,
    budget_du_manometre,
)

RESOLUTION_MM = 0.001
RESOLUTION_ATM = Pression(pascals=100.0)
PLAGE_THERMIQUE = 20.0
"""Etendue supposee des temperatures possibles, en kelvins (10 a 30 C)."""


def _budget(plage: float | None) -> BudgetDIncertitude:
    return budget_du_manometre(
        enonce.MANOMETRE,
        resolution_hauteur=RESOLUTION_MM,
        resolution_atmospherique=RESOLUTION_ATM,
        plage_thermique=plage,
    )


def test_sans_plage_thermique_le_budget_est_celui_d_avant() -> None:
    """Retrocompatibilite : l'argument est optionnel et neutre par defaut."""
    assert _budget(None).ecart_type_compose() == pytest.approx(48.3, abs=0.2)


def test_la_temperature_ajoute_une_contribution_par_fluide() -> None:
    budget = _budget(PLAGE_THERMIQUE)
    noms = {c.nom for c in budget.contributions}
    assert {"Hg (température)", "eau (température)", "huile (température)"} <= noms


def test_la_temperature_du_mercure_devient_la_source_dominante() -> None:
    """Elle supplante la lecture : c'est un renversement de conclusion."""
    assert _budget(PLAGE_THERMIQUE).dominante().nom == "Hg (température)"


def test_le_budget_complet_est_nettement_plus_large() -> None:
    """Omettre la temperature sous-estimait l'incertitude de pres de moitie."""
    etroit = _budget(None).ecart_type_compose()
    complet = _budget(PLAGE_THERMIQUE).ecart_type_compose()
    assert complet > 1.3 * etroit


def test_les_deux_decimales_affichees_restent_justifiees() -> None:
    """Le budget s'elargit, mais pas au point de changer l'affichage."""
    assert _budget(PLAGE_THERMIQUE).decimales_justifiees_en_kilopascals() == 2


def test_une_plage_thermique_negative_est_refusee() -> None:
    with pytest.raises(ValueError, match="plage"):
        _budget(-5.0)
