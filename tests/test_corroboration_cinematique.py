"""La partie II n'etait corroboree par rien.

La partie I confronte son resultat a la reponse publiee par le cours. La
partie II n'avait aucun recoupement : tout y venait du meme moteur
symbolique, donc d'une seule et meme facon de se tromper. Ces tests
integrent les equations *numeriquement* -- methode independante -- et
comparent.
"""

import pytest

from polytech.fluides.application import enonce
from polytech.fluides.domain.cinematique.corroboration import (
    ecart_a_la_trajectoire_symbolique,
    ecart_aux_lignes_de_courant,
    variation_de_l_invariant,
)

DEPART = (1.5, 2.0)
DUREE = 0.6
PAS = 1e-3
TOLERANCE = 1e-6


def test_la_trajectoire_numerique_de_c_confirme_la_solution_symbolique() -> None:
    """x0 e^{2t}, y0 e^{-2t} : obtenu par dsolve, verifie par Runge-Kutta."""
    ecart = ecart_a_la_trajectoire_symbolique(enonce.CHAMP_C, DEPART, DUREE, PAS)
    assert ecart < TOLERANCE


def test_le_produit_xy_reste_constant_le_long_de_la_trajectoire_de_c() -> None:
    """Confirme xy = C sans jamais resoudre l'equation des lignes de courant."""
    variation = variation_de_l_invariant(enonce.CHAMP_C, DEPART, DUREE, PAS)
    assert variation < TOLERANCE


def test_la_trajectoire_de_b_n_a_pas_de_forme_close_a_corroborer() -> None:
    """On ne peut pas comparer a une solution formelle qui n'existe pas.

    C'est une propriete du champ, pas une limite de l'outil : l'ordonnee
    conduit a une primitive non elementaire. Le corrige le dit, et c'est
    pourquoi il n'expose que les *lignes de courant* de cet ecoulement.
    """
    assert enonce.CHAMP_B.trajectoire_est_elementaire() is False


def test_les_lignes_de_courant_de_b_resistent_a_l_integration_numerique() -> None:
    """La solution par facteur integrant est le calcul le plus delicat du corrige.

    On integre dy/dx = v/u a instant fige, par une methode sans rapport avec
    celle du document, et on compare a sa forme close.
    """
    ecart = ecart_aux_lignes_de_courant(
        enonce.CHAMP_B,
        instant=2.0,
        depart=(1.0, 0.5),
        arrivee=3.0,
        pas=1e-4,
    )
    assert ecart < 1e-5


def test_un_pas_negatif_est_refuse() -> None:
    with pytest.raises(ValueError, match="pas"):
        ecart_a_la_trajectoire_symbolique(enonce.CHAMP_C, DEPART, DUREE, -1.0)
