"""Le chapitre 5 du tome 1 du syllabus : les methodes numeriques.

Absent de la bibliotheque jusqu'ici pour la meme raison que le
rayonnement des gaz -- la mesure de couverture ne regardait que les
planches, et ce chapitre n'est que dans le syllabus.

Le syllabus donne la prose en clair et les formules en images. Deux
consequences sur ce qui est teste ici :

* ce que la prose etablit est implemente : differences centrees d'erreur
  `O(Delta x^2)` « les plus utilisees », laplacien a cinq points sur un
  reseau quadrille, Euler explicite « qui pose de gros problemes de
  stabilite », implicite « inconditionnellement stable » ;
* la **borne** de stabilite, dont le syllabus dit seulement « on peut
  montrer qu'il faut » suivi d'une formule en image, n'est pas affirmee :
  elle est verifiee en faisant effectivement diverger le schema au-dela,
  et converger en dessous. C'est la meme discipline que pour les corriges
  d'examen -- refaire le raisonnement par un autre chemin.
"""

from __future__ import annotations

import math

import pytest

from polytech.thermique.domain.differences_finies import (
    derivee_seconde_centree,
    laplacien_a_cinq_points,
    nombre_de_fourier_de_maille,
    pas_de_temps_critique,
    temperature_nodale_stationnaire,
)
from polytech.thermique.domain.stabilite_numerique import est_stable_en_explicite


def test_la_derivee_seconde_centree_retrouve_celle_d_une_parabole() -> None:
    """Sur `x^2` la derivee seconde vaut 2, et le schema est exact.

    Une parabole n'a pas de terme d'ordre trois : l'erreur de troncature
    en `Delta x^2` porte sur ce terme, donc elle s'annule ici. Le schema
    doit rendre 2 **exactement**, et pas approximativement.
    """
    pas = 0.25
    x = 1.0
    valeurs = ((x - pas) ** 2, x**2, (x + pas) ** 2)
    assert derivee_seconde_centree(*valeurs, pas=pas) == pytest.approx(2.0)


def test_l_erreur_du_schema_centre_est_d_ordre_deux() -> None:
    """Diviser le pas par deux divise l'erreur par **quatre**, non par deux.

    C'est ce que « erreur d'ordre `Delta x^2` » signifie, et le syllabus
    en fait l'argument qui designe les differences centrees comme « les
    plus utilisees ». La verification porte sur `sin`, dont la derivee
    seconde est connue et dont le terme d'ordre quatre ne s'annule pas.
    """
    x = 0.7
    exacte = -math.sin(x)

    def ecart(pas: float) -> float:
        approchee = derivee_seconde_centree(
            math.sin(x - pas),
            math.sin(x),
            math.sin(x + pas),
            pas=pas,
        )
        return abs(approchee - exacte)

    assert ecart(0.1) / ecart(0.05) == pytest.approx(4.0, rel=0.02)


def test_un_noeud_stationnaire_est_la_moyenne_de_ses_quatre_voisins() -> None:
    """Sur un reseau quadrille et sans source, c'est tout le chapitre 5.2.2.

    Le resultat n'est pas pose : il est **deduit** du laplacien a cinq
    points, en exigeant que la temperature rendue l'annule.
    """
    voisins = (120.0, 80.0, 60.0, 100.0)
    nodale = temperature_nodale_stationnaire(voisins)
    assert nodale == pytest.approx(90.0)
    assert laplacien_a_cinq_points(nodale, voisins, pas=0.01) == pytest.approx(0.0)


def test_un_champ_lineaire_a_un_laplacien_nul() -> None:
    """Une rampe de temperature est solution de Laplace, et le schema le voit.

    Garde utile : un schema mal ecrit -- un signe, un facteur quatre --
    rendrait un laplacien non nul sur une rampe, qui est le cas le plus
    simple du cours.
    """
    pas = 0.5
    centre = 50.0
    gauche, droite = centre - 10.0, centre + 10.0
    bas, haut = centre - 10.0, centre + 10.0
    laplacien = laplacien_a_cinq_points(centre, (gauche, droite, bas, haut), pas=pas)
    assert laplacien == pytest.approx(0.0)


def test_le_nombre_de_fourier_de_maille_est_sans_dimension() -> None:
    """`Fo = a Delta tau / Delta x^2` : c'est lui qui porte la stabilite.

    Son invariance est ce qui rend la borne universelle : doubler le pas
    d'espace permet de quadrupler le pas de temps a `Fo` constant.
    """
    fourier = nombre_de_fourier_de_maille(
        diffusivite=1e-5,
        pas_de_temps=4.0,
        pas_d_espace=0.02,
    )
    assert fourier == pytest.approx(0.1)
    double = nombre_de_fourier_de_maille(
        diffusivite=1e-5,
        pas_de_temps=16.0,
        pas_d_espace=0.04,
    )
    assert double == pytest.approx(fourier)


def test_le_pas_critique_decroit_comme_le_carre_du_maillage() -> None:
    """Raffiner le maillage coute **quadratiquement** en pas de temps.

    C'est la raison des « gros problemes de stabilite » que le syllabus
    reproche a l'explicite : diviser le maillage par dix multiplie par
    cent le nombre de pas de temps, a duree simulee egale.
    """
    large = pas_de_temps_critique(diffusivite=1e-5, pas_d_espace=0.02, dimensions=2)
    fin = pas_de_temps_critique(diffusivite=1e-5, pas_d_espace=0.002, dimensions=2)
    assert large / fin == pytest.approx(100.0)


def test_la_stabilite_se_juge_sur_le_pas_critique() -> None:
    """En dessous du pas critique le schema est stable, au-dela non."""
    assert est_stable_en_explicite(fourier=0.2, dimensions=2)
    assert not est_stable_en_explicite(fourier=0.3, dimensions=2)
    assert est_stable_en_explicite(fourier=0.49, dimensions=1)
    assert not est_stable_en_explicite(fourier=0.51, dimensions=1)


def test_une_diffusivite_nulle_est_refusee() -> None:
    """Sans diffusivite il n'y a pas de transitoire, et le pas serait infini."""
    with pytest.raises(ValueError, match="diffusivite nulle ou negative"):
        pas_de_temps_critique(diffusivite=0.0, pas_d_espace=0.01, dimensions=2)


def test_un_pas_d_espace_nul_est_refuse() -> None:
    """Le nombre de Fourier de maille y divise par zero."""
    with pytest.raises(ValueError, match="pas d'espace nul ou negatif"):
        nombre_de_fourier_de_maille(
            diffusivite=1e-5,
            pas_de_temps=1.0,
            pas_d_espace=0.0,
        )


def test_un_nombre_de_dimensions_hors_de_un_a_trois_est_refuse() -> None:
    """La borne depend de la dimension, et le cours en traite trois."""
    with pytest.raises(ValueError, match="dimensions hors de 1 a 3"):
        est_stable_en_explicite(fourier=0.1, dimensions=4)
