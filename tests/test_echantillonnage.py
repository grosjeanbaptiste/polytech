"""Les fleches du trace doivent venir du champ, pas d'un calcul a la main.

La figure des lignes de courant portait une table de 25 vecteurs saisis
manuellement. Exacts -- mais invérifiables par le lecteur et couverts par
aucun test.
"""

import math

import pytest

from polytech.fluides.application import enonce
from polytech.fluides.domain.cinematique.echantillonnage import (
    GrilleDeTrace,
    fleches_du_champ,
)

GRILLE = GrilleDeTrace(
    abscisses=(0.5, 1.5, 2.5),
    ordonnees=(0.6, 1.8, 3.0),
    longueur=0.55,
)


def test_la_grille_produit_une_fleche_par_noeud() -> None:
    assert len(fleches_du_champ(enonce.CHAMP_C, GRILLE)) == 9


def test_chaque_fleche_a_la_longueur_demandee() -> None:
    """Les fleches sont normalisees : seule leur direction porte du sens."""
    for fleche in fleches_du_champ(enonce.CHAMP_C, GRILLE):
        assert math.hypot(fleche.u, fleche.v) == pytest.approx(GRILLE.longueur)


def test_chaque_fleche_pointe_dans_le_sens_du_champ() -> None:
    """U = 2x > 0 : vers la droite ; v = -2y < 0 : vers le bas."""
    for fleche in fleches_du_champ(enonce.CHAMP_C, GRILLE):
        assert fleche.u > 0
        assert fleche.v < 0
        # Colinearite au champ reel : le produit vectoriel doit s'annuler.
        assert fleche.u * (-2 * fleche.y) - fleche.v * (2 * fleche.x) == pytest.approx(
            0.0,
            abs=1e-9,
        )


def test_une_grille_sans_noeud_est_refusee() -> None:
    with pytest.raises(ValueError, match="grille"):
        GrilleDeTrace(abscisses=(), ordonnees=(1.0,), longueur=0.5)


def test_une_longueur_de_fleche_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="longueur"):
        GrilleDeTrace(abscisses=(1.0,), ordonnees=(1.0,), longueur=0.0)


def test_un_point_d_arret_est_refuse_car_sa_direction_n_existe_pas() -> None:
    """En O la vitesse est nulle : normaliser y serait une division par zero."""
    grille = GrilleDeTrace(abscisses=(0.0,), ordonnees=(0.0,), longueur=0.5)
    with pytest.raises(ValueError, match="vitesse nulle"):
        fleches_du_champ(enonce.CHAMP_C, grille)
