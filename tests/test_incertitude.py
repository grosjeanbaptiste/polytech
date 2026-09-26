"""Le nombre de chiffres affiches est une affirmation : il doit se justifier.

Le corrige annonce p1 = 129,65 kPa et conclut « seule la lecture du mercure
doit etre soignee ». Ni la precision du resultat ni cette hierarchie
n'etaient calculees : les sensibilites etaient donnees une a une, jamais
composees.
"""

import math

import pytest

from polytech.fluides.application import enonce
from polytech.fluides.domain.grandeurs import Pression
from polytech.fluides.domain.statique.incertitude import (
    BudgetDIncertitude,
    Contribution,
    budget_du_manometre,
    incertitude_de_lecture,
)

RESOLUTION_MM = 0.001
RESOLUTION_ATM = Pression(pascals=100.0)
"""p_atm est donnee au dixieme de kilopascal : 100 Pa de resolution."""


def test_une_resolution_rectangulaire_suit_la_regle_du_gum() -> None:
    """Demi-etendue sur racine de trois, pour une loi rectangulaire."""
    assert incertitude_de_lecture(RESOLUTION_MM) == pytest.approx(
        0.0005 / math.sqrt(3),
    )


def test_les_contributions_se_composent_en_quadrature() -> None:
    """Sources independantes : c'est la somme des carres qui s'ajoute."""
    budget = budget_du_manometre(
        enonce.MANOMETRE,
        resolution_hauteur=RESOLUTION_MM,
        resolution_atmospherique=RESOLUTION_ATM,
    )
    attendu = math.hypot(*[c.ecart_type for c in budget.contributions])
    assert budget.ecart_type_compose() == pytest.approx(attendu)


def test_le_mercure_domine_le_budget() -> None:
    """Ce que le corrige affirmait : verifions-le, et chiffrons-le."""
    budget = budget_du_manometre(
        enonce.MANOMETRE,
        resolution_hauteur=RESOLUTION_MM,
        resolution_atmospherique=RESOLUTION_ATM,
    )
    assert budget.dominante().nom == "Hg"


def test_la_pression_atmospherique_n_est_pas_negligeable() -> None:
    """Le corrige l'oubliait : elle pese plus du quart de la variance."""
    budget = budget_du_manometre(
        enonce.MANOMETRE,
        resolution_hauteur=RESOLUTION_MM,
        resolution_atmospherique=RESOLUTION_ATM,
    )
    atmospherique = next(c for c in budget.contributions if c.nom == "atmosphere")
    assert atmospherique.part_de_variance(budget) > 0.25


def test_l_eau_et_l_huile_sont_bien_negligeables() -> None:
    """Un millimetre sur l'eau ne coute presque rien : c'est mesurable."""
    budget = budget_du_manometre(
        enonce.MANOMETRE,
        resolution_hauteur=RESOLUTION_MM,
        resolution_atmospherique=RESOLUTION_ATM,
    )
    legers = [c for c in budget.contributions if c.nom in {"eau", "huile"}]
    assert sum(c.part_de_variance(budget) for c in legers) < 0.01


def test_le_budget_dit_combien_de_decimales_sont_justifiees() -> None:
    """129,65 kPa : deux decimales, et le budget doit le confirmer."""
    budget = budget_du_manometre(
        enonce.MANOMETRE,
        resolution_hauteur=RESOLUTION_MM,
        resolution_atmospherique=RESOLUTION_ATM,
    )
    assert budget.decimales_justifiees_en_kilopascals() == 2


def test_une_resolution_negative_est_refusee() -> None:
    with pytest.raises(ValueError, match="resolution"):
        incertitude_de_lecture(-1.0)


def test_un_budget_vide_est_refuse() -> None:
    """Sans source recensee, parler de precision n'aurait aucun sens."""
    with pytest.raises(ValueError, match="aucune source"):
        BudgetDIncertitude(contributions=())


def test_une_contribution_negative_est_refusee() -> None:
    with pytest.raises(ValueError, match="ecart-type"):
        Contribution(nom="x", ecart_type=-1.0)
