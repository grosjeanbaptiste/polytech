"""L'activite d'integration 2 de mecanique des fluides : ses trois exercices.

Enonce : `MecaniqueFluides/EXS/Exs_Activite_integration_2.pdf`, quatre
planches. Les trois exercices sont ceux du chapitre IV du cours, que la
bibliotheque porte deja ; ce fichier tient ce que le corrige en publie,
et recoupe chaque resultat par un second chemin.

Oracles : la planche 3 de l'enonce donne le profil du film,
`w = rho g x (x - 2h) / (2 mu)` et `w* = x* (x* - 2) / 2` ; le
bateau-pompe est l'exercice 7 de la planche 72 du cours, que
`tests/test_planches_fluides_bernoulli.py` tient deja a 50,9 m/s et 201 kW.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.fluides.application import activite_2 as a2
from polytech.fluides.application.macros_activite_2 import macros
from polytech.fluides.domain.fondamentales.film_tombant import vitesse_du_film
from polytech.fluides.domain.grandeurs import PESANTEUR

X, H, RHO, MU, G = sp.symbols("x h rho mu g", positive=True)

# --- Le film d'huile ---------------------------------------------------------


def test_le_profil_du_film_est_celui_de_la_planche_3() -> None:
    """`w = rho g x (x - 2h) / (2 mu)`."""
    planche = RHO * G * X * (X - 2 * H) / (2 * MU)
    assert sp.simplify(vitesse_du_film(X, H, RHO, MU, G) - planche) == 0


def test_le_profil_normalise_est_celui_de_la_planche_3() -> None:
    """Avec `x* = x/h` et `w* = w mu / (rho g h^2)` : `w* = x* (x* - 2) / 2`."""
    etoile = a2.ABSCISSE_REDUITE
    assert sp.simplify(a2.profil_normalise() - etoile * (etoile - 2) / 2) == 0


def test_le_profil_se_retrouve_en_integrant_navier_stokes() -> None:
    """Second chemin : `mu w'' = rho g`, `w(0) = 0`, `w'(h) = 0`."""
    integre = a2.profil_par_integration()
    assert sp.simplify(integre - vitesse_du_film(a2.X, a2.H, a2.RHO, a2.MU, a2.G)) == 0


def test_la_paroi_porte_le_poids_du_film() -> None:
    """Le cisaillement a la paroi, `mu w'(0)`, equilibre `rho g h` par unite d'aire."""
    assert sp.simplify(a2.cisaillement_a_la_paroi() + a2.RHO * a2.G * a2.H) == 0


def test_la_vitesse_moyenne_vaut_les_deux_tiers_de_celle_de_la_surface() -> None:
    assert a2.rapport_moyenne_sur_surface() == sp.Rational(2, 3)


# --- Le reservoir perce ------------------------------------------------------


def test_la_hauteur_maximale_annule_la_montee_du_niveau() -> None:
    """Second chemin : `dz/dt = 0` dans le bilan de masse."""
    debit, section, orifice, niveau = sp.symbols("m A_T A_o z", positive=True)
    montee = debit / (a2.RHO * section) - orifice / section * sp.sqrt(2 * a2.G * niveau)
    (stationnaire,) = sp.solve(sp.Eq(montee, 0), niveau)
    assert sp.simplify(stationnaire - a2.hauteur_maximale(debit, orifice)) == 0


def test_la_hauteur_maximale_en_diametre_va_comme_l_inverse_de_la_puissance_4() -> None:
    assert macros()["ReservoirGainOrificeMoitie"] == "16"


def test_le_niveau_met_4_3_temps_caracteristiques_a_atteindre_99_pourcent() -> None:
    """`t / T = -s - ln(1 - s)` avec `s = racine(0,99)` ; `T = 2 rho A_T z_max / m`."""
    s = math.sqrt(0.99)
    assert float(a2.duree_reduite(0.99)) == pytest.approx(
        -s - math.log(1 - s), rel=1e-9
    )
    assert macros()["ReservoirDureeQuatreVingtDixNeuf"] == "4.3"


# --- Le bateau-pompe ---------------------------------------------------------


def test_le_bateau_pompe_rend_les_valeurs_de_la_planche_72() -> None:
    bateau = a2.bateau_pompe()
    assert bateau.ejection == pytest.approx(50.9, abs=0.05)
    assert bateau.hauteur == pytest.approx(139.2, abs=0.1)
    assert bateau.arbre == pytest.approx(201e3, rel=3e-3)


def test_la_puissance_se_retrouve_par_l_energie_massique() -> None:
    """Second chemin : `m (g z + V^2 / 2 + g h_L)`, sans passer par la hauteur."""
    bateau = a2.bateau_pompe()
    massique = PESANTEUR * 4.0 + bateau.ejection**2 / 2 + PESANTEUR * 3.0
    assert bateau.hydraulique == pytest.approx(1030.0 * 0.1 * massique, rel=1e-12)


def test_l_energie_cinetique_du_jet_domine_la_hauteur() -> None:
    assert macros()["BateauPartCinetique"] == "95"


def test_chaque_macro_porte_un_nom_de_lettres() -> None:
    """LaTeX n'accepte que des lettres dans un nom de commande."""
    noms = macros()
    assert len(noms) >= 20
    assert all(nom.isalpha() and nom.isascii() for nom in noms)


def test_la_courbe_du_film_va_de_la_paroi_a_la_surface_libre() -> None:
    """Onze points de `w*(x*)`, de `(0, 0)` a `(1, -1/2)` : ce que la figure trace."""
    points = macros()["FilmCourbe"].split()
    assert (points[0], points[-1], len(points)) == ("(0,0)", "(1,-0.5)", 11)


def test_la_courbe_du_reservoir_monte_vers_son_asymptote() -> None:
    """`z / z_max` en fonction de `t / T` : croissante, et jamais a un."""
    points = [
        tuple(float(v) for v in point.strip("()").split(","))
        for point in macros()["ReservoirCourbe"].split()
    ]
    assert points[0] == (0.0, 0.0)
    assert points == sorted(points)
    assert 0.98 < points[-1][1] < 1.0
