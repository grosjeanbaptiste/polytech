"""Optimisation : la modelisation, problemes VIa et VIb (les meubles).

Source : `Optimisation/EXS/Exs_Modelisation.pdf`, planches 23 a 27. Huit
semaines, deux meubles, cinq ateliers. Le modele de la planche 24 est
resolu ; il echange les deux stocks finals que l'enonce demande, et
l'extension de capacite de la planche 25 se revele sans effet.
"""

from __future__ import annotations

import pytest

from polytech.optimisation.application import modele_meubles as me
from polytech.optimisation.application import modele_meubles_plan as pl
from polytech.optimisation.application.modele_meubles_semaine import une_semaine

IMPRIME, ENONCE = me.STOCKS_IMPRIMES, me.STOCKS_DE_L_ENONCE

# --- Le modele ---------------------------------------------------------------


def test_via_la_taille_du_modele() -> None:
    """Par semaine : huit productions, deux ventes ; quatorze stocks en tout."""
    modele = me.programme(IMPRIME)
    assert len(modele.objectif) == 8 * 10 + 14
    assert len(modele.contraintes) == 8 * (5 + 2 + 2 + 2)


def test_via_les_stocks_finals_sont_echanges_sur_la_planche() -> None:
    """L'enonce : 200 « De Luxe » et 100 « Standard ». La planche : l'inverse."""
    assert ENONCE == {"L": 200, "S": 100}
    assert IMPRIME == {"L": 100, "S": 200}


def test_via_ce_qu_un_atelier_peut_sortir_en_une_semaine() -> None:
    """La finition limite le « De Luxe » a 450 ; l'assemblage le « Standard » a 560."""
    assert pl.plafond_hebdomadaire("L") == 450
    assert pl.plafond_hebdomadaire("S") == 560
    assert pl.plafond_hebdomadaire("S", etendu=True) == 860
    assert pl.plafond_hebdomadaire("L", etendu=True) == 450


# --- Les optimums ------------------------------------------------------------


def test_via_le_gain_du_modele_imprime() -> None:
    plan = pl.plan(tuple(IMPRIME.items()))
    assert plan.gain == 1_005_500
    assert plan.production["L"] == (450, 350, 300, 450, 450, 450, 450, 350)
    assert plan.production["S"] == (0, 200, 300, 0, 0, 0, 0, 200)


def test_via_le_gain_avec_les_stocks_de_l_enonce() -> None:
    """Cent « De Luxe » de plus a garder, cent « Standard » de moins : -15000."""
    plan = pl.plan(tuple(ENONCE.items()))
    assert plan.gain == 990_500
    assert plan.gain == pl.plan(tuple(IMPRIME.items())).gain - 15_000


@pytest.mark.parametrize("stocks", [IMPRIME, ENONCE])
def test_via_le_plan_relu_sur_l_enonce(stocks: dict[str, int]) -> None:
    """Second chemin : bilans, ventes plafonnees, gain, sans le programme."""
    plan = pl.plan(tuple(stocks.items()))
    assert pl.stock_final(plan) == stocks
    assert pl.ventes_dans_les_previsions(plan)
    assert pl.gain_relu(plan) == plan.gain
    for semaine in range(8):
        quantites = (plan.production["L"][semaine], plan.production["S"][semaine])
        assert une_semaine(*quantites)


def test_via_la_finition_est_saturee_chaque_semaine() -> None:
    """Vingt minutes par « De Luxe », dix par « Standard » : 9000 minutes, toujours."""
    plan = pl.plan(tuple(ENONCE.items()))
    assert pl.minutes_de_finition(plan) == (9000,) * 8


def test_via_une_semaine_trop_chargee_n_est_pas_fabricable() -> None:
    assert une_semaine(450, 0)
    assert not une_semaine(451, 0)
    assert not une_semaine(450, 1)
    assert une_semaine(0, 560)
    assert not une_semaine(0, 561)


# --- Probleme VIb : doubler F2 et AS2 ----------------------------------------


def test_vib_l_extension_ne_rapporte_rien() -> None:
    """Meme doublee des la semaine 4, la capacite ne change pas le gain brut."""
    etendu = pl.plan(tuple(IMPRIME.items()), extension=4)
    assert etendu.gain == pl.plan(tuple(IMPRIME.items())).gain


def test_vib_les_deux_binaires_restent_a_zero() -> None:
    """5000 ou 7500 de cout fixe pour un gain nul : ni `delta` ni `gamma`."""
    assert pl.gain_net_de_l_extension(6) == -5000
    assert pl.gain_net_de_l_extension(4) == -7500


def test_une_semaine_d_extension_inconnue_est_refusee() -> None:
    with pytest.raises(KeyError, match="aucune extension prevue la semaine "):
        pl.gain_net_de_l_extension(5)
