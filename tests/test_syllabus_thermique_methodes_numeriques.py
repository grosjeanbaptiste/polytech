"""Le syllabus de thermique, tome 1 : les methodes numeriques.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_1.pdf`, pages 85 a
102 (premiere partie, chapitre 5), lues sur les pages rendues.

Les oracles : les differences finies (5.1) a (5.7), le schema en pi des
equations (5.9) a (5.11), l'application de la paroi de four -- 77 et
260 MJ/m^2 selon l'ordre des couches -- et les fonctions de forme (5.14).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from polytech.maths.domain.derivation_numerique import (
    derivee_a_droite,
    derivee_a_gauche,
    derivee_centree,
    derivee_seconde_d_ordre_quatre,
)
from polytech.thermique.domain.conduction.elements_finis import (
    fonctions_de_forme,
    rigidite_d_un_triangle,
)
from polytech.thermique.domain.conduction.noeuds_de_frontiere import (
    noeud_fictif_de_fourier,
    noeud_fictif_de_neumann,
    temperature_de_frontiere,
)
from polytech.thermique.domain.conduction.paroi_en_tranches import (
    ParoiEnTranches,
    Tranches,
)
from polytech.thermique.domain.differences_finies import (
    derivee_seconde_centree,
    laplacien_a_cinq_points,
)
from polytech.thermique.domain.stabilite_numerique import est_stable_en_explicite


def _erreur(schema: str, pas: float) -> float:
    exacte, point = math.exp(1.0), 1.0
    avant, centre, apres = (math.exp(point + k * pas) for k in (-1, 0, 1))
    approchee = {
        "droite": derivee_a_droite(centre, apres, pas),
        "gauche": derivee_a_gauche(avant, centre, pas),
        "centree": derivee_centree(avant, apres, pas),
    }[schema]
    return abs(approchee - exacte)


def test_les_differences_decentrees_sont_d_ordre_un() -> None:
    """(5.1) et (5.2) : diviser le pas par deux divise l'erreur par deux."""
    for schema in ("droite", "gauche"):
        rapport = _erreur(schema, 0.02) / _erreur(schema, 0.01)
        assert rapport == pytest.approx(2.0, abs=0.03)
    assert derivee_a_droite(1.0, 4.0, 0.5) == 6.0
    assert derivee_a_gauche(1.0, 4.0, 0.5) == 6.0
    with pytest.raises(ValueError, match="pas nul ou negatif"):
        derivee_a_droite(1.0, 4.0, 0.0)


def test_la_difference_centree_est_d_ordre_deux() -> None:
    """(5.3) : l'erreur est divisee par quatre."""
    rapport = _erreur("centree", 0.02) / _erreur("centree", 0.01)
    assert rapport == pytest.approx(4.0, abs=0.01)


def test_la_derivee_seconde_a_cinq_noeuds_est_d_ordre_quatre() -> None:
    """Page 87 : `(-t_-2 + 16 t_-1 - 30 t_0 + 16 t_1 - t_2) / (12 h^2)`."""
    pas = 0.1
    abscisses = [1.0 + k * pas for k in (-2, -1, 0, 1, 2)]
    quintique = [x**5 - 2 * x**4 for x in abscisses]
    exacte = 20 * 1.0**3 - 24 * 1.0**2
    assert derivee_seconde_d_ordre_quatre(quintique, pas) == pytest.approx(exacte)
    grossiere = [math.exp(1.0 + k * 0.2) for k in (-2, -1, 0, 1, 2)]
    fine = [math.exp(1.0 + k * 0.1) for k in (-2, -1, 0, 1, 2)]
    erreurs = (
        abs(derivee_seconde_d_ordre_quatre(grossiere, 0.2) - math.e),
        abs(derivee_seconde_d_ordre_quatre(fine, 0.1) - math.e),
    )
    assert erreurs[0] / erreurs[1] == pytest.approx(16.0, rel=0.03)
    centree = derivee_seconde_centree(fine[1], fine[2], fine[3], 0.1)
    assert erreurs[1] < abs(centree - math.e) / 100
    with pytest.raises(ValueError, match="cinq valeurs attendues"):
        derivee_seconde_d_ordre_quatre([1.0, 2.0, 3.0], 0.1)


def test_l_equation_nodale_avec_generation() -> None:
    """(5.6) : `somme des voisins - 4 t_ij = -q Delta x^2 / lambda`."""
    generation, conductivite, pas = 3.0e4, 1.5, 0.01

    def champ(x: float, y: float) -> float:
        return -generation * (x**2 + y**2) / (4 * conductivite)

    voisins = (champ(-pas, 0.0), champ(pas, 0.0), champ(0.0, -pas), champ(0.0, pas))
    laplacien = laplacien_a_cinq_points(champ(0.0, 0.0), voisins, pas)
    assert laplacien == pytest.approx(-generation / conductivite)


def test_le_noeud_fictif_d_une_condition_de_neumann() -> None:
    """Page 88 : `t_3* = t_1 + 2 Delta x phi_0 / lambda_0`."""
    assert noeud_fictif_de_neumann(50.0, 0.01, 2.0, flux=400.0) == pytest.approx(54.0)
    with pytest.raises(ValueError, match="conductivite nulle ou negative"):
        noeud_fictif_de_neumann(50.0, 0.01, 0.0, flux=400.0)


def test_l_equation_du_noeud_de_frontiere_a_le_signe_plus() -> None:
    """Le syllabus imprime `... = q Dx^2 / lambda - 2 Dx phi_0 / lambda`.

    Avec son propre noeud fictif, (5.7) donne le signe plus. Sur un champ
    lineaire, que le flux entrant `phi_0` traverse, la formule imprimee
    n'est pas verifiee.
    """
    flux, conductivite, pas = 400.0, 2.0, 0.01
    surface = 80.0
    interieur = surface - flux * pas / conductivite
    frontiere = temperature_de_frontiere(
        interieur, (surface, surface), pas, conductivite, flux=flux
    )
    assert frontiere == pytest.approx(surface)
    membre = 4 * surface - 2 * interieur - surface - surface
    assert membre == pytest.approx(2 * pas * flux / conductivite)
    assert membre != pytest.approx(-2 * pas * flux / conductivite)


def test_le_noeud_fictif_d_une_condition_de_fourier() -> None:
    """Page 89 : `t_3* = t_1 + (2 Delta x / lambda_0) K (t_f - t_0)`."""
    fictif = noeud_fictif_de_fourier(
        50.0, 0.01, 2.0, echange=100.0, fluide=90.0, surface=60.0
    )
    assert fictif == pytest.approx(50.0 + 0.01 * 100.0 * 30.0)
    par_neumann = noeud_fictif_de_neumann(50.0, 0.01, 2.0, flux=100.0 * 30.0)
    assert fictif == pytest.approx(par_neumann)


def _pas_explicite(valeurs: list[float], fourier: float) -> list[float]:
    interieur = [
        centre + fourier * (gauche - 2 * centre + droite)
        for gauche, centre, droite in zip(
            valeurs, valeurs[1:], valeurs[2:], strict=False
        )
    ]
    return [valeurs[0], *interieur, valeurs[-1]]


def test_la_condition_de_stabilite_est_imprimee_a_l_envers() -> None:
    """Page 90 : « il faut Fo = a Dtau / Dx^2 > 0,5 ». C'est l'inverse.

    Au-dela d'un demi le schema explicite diverge ; en deca il s'amortit.
    """
    assert est_stable_en_explicite(0.4, 1)
    assert not est_stable_en_explicite(0.6, 1)
    stable = instable = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    for _ in range(40):
        stable, instable = _pas_explicite(stable, 0.4), _pas_explicite(instable, 0.6)
    assert max(abs(v) for v in stable) < 1e-2
    assert max(abs(v) for v in instable) > 10


BRIQUES = Tranches(epaisseur=0.23, conductivite=1.15, capacite=3150 * 1050, nombre=6)
FIBRES = Tranches(epaisseur=0.05, conductivite=0.15, capacite=215 * 1050, nombre=5)
"""Page 93 : briques refractaires et matelas de fibres ceramiques."""


def test_le_schema_en_pi_d_une_tranche() -> None:
    """(5.9) et fig. 5.5 : `R = b / lambda`, `C = rho c b`, en deux moities."""
    assert FIBRES.resistance() == pytest.approx(0.01 / 0.15)
    assert FIBRES.capacite_de_tranche() == pytest.approx(215 * 1050 * 0.01)
    paroi = ParoiEnTranches((FIBRES, BRIQUES), interieur=20.0, exterieur=15.0)
    capacites = paroi.capacites()
    assert len(capacites) == 12
    c1, c2 = FIBRES.capacite_de_tranche(), BRIQUES.capacite_de_tranche()
    assert capacites[0] == pytest.approx(c1 / 2)
    assert capacites[5] == pytest.approx((c1 + c2) / 2)
    assert capacites[11] == pytest.approx(c2 / 2)
    assert sum(capacites) == pytest.approx(5 * c1 + 6 * c2)


def test_les_equations_des_noeuds_de_la_paroi() -> None:
    """Page 94 : la premiere, une interieure, et la derniere."""
    paroi = ParoiEnTranches((FIBRES, BRIQUES), interieur=20.0, exterieur=15.0)
    temperatures = np.linspace(400.0, 70.0, 12)
    derivees = paroi.derivees(temperatures, four=1000.0, ambiance=25.0)
    r1, c1 = FIBRES.resistance(), FIBRES.capacite_de_tranche()
    r2, c2 = BRIQUES.resistance(), BRIQUES.capacite_de_tranche()
    t = temperatures
    premiere = (t[1] - t[0]) / r1 + 20.0 * (1000.0 - t[0])
    assert derivees[0] == pytest.approx(premiere / (c1 / 2))
    assert derivees[1] == pytest.approx(((t[2] - t[1]) + (t[0] - t[1])) / (r1 * c1))
    jonction = (t[4] - t[5]) / r1 + (t[6] - t[5]) / r2
    assert derivees[5] == pytest.approx(jonction / ((c1 + c2) / 2))
    derniere = (t[10] - t[11]) / r2 + 15.0 * (25.0 - t[11])
    assert derivees[11] == pytest.approx(derniere / (c2 / 2))


def test_la_paroi_conserve_l_energie() -> None:
    """`phi_pi - phi_pe = phi_acc` : la somme des `C_i dt_i/dtau`."""
    paroi = ParoiEnTranches((FIBRES, BRIQUES), interieur=20.0, exterieur=15.0)
    temperatures = np.linspace(400.0, 70.0, 12) ** 1.1
    derivees = paroi.derivees(temperatures, four=1000.0, ambiance=25.0)
    accumule = float(np.dot(paroi.capacites(), derivees))
    entrant = 20.0 * (1000.0 - temperatures[0])
    sortant = 15.0 * (temperatures[-1] - 25.0)
    assert accumule == pytest.approx(entrant - sortant)


def _four(instant: float) -> float:
    """Fig. page 93 : de 25 a 1250 degres en deux heures, puis un palier."""
    return 25.0 + 1225.0 * min(instant / 7200.0, 1.0)


@pytest.mark.parametrize(
    ("couches", "imprimee", "calculee"),
    [((FIBRES, BRIQUES), 77.0, 74.2), ((BRIQUES, FIBRES), 260.0, 268.1)],
)
def test_l_energie_de_chauffe_de_la_paroi_de_four(
    couches: tuple[Tranches, Tranches], imprimee: float, calculee: float
) -> None:
    """Page 94 : 77 MJ/m^2 le leger a l'intrados, 260 le lourd.

    L'integration des douze equations rend 74,2 et 268,1 : a 4 % du
    tableau, dont le pas de temps n'est pas dit.
    """
    paroi = ParoiEnTranches(couches, interieur=20.0, exterieur=15.0)
    energie = paroi.energie_fournie(_four, ambiance=25.0, duree=8 * 3600.0)
    assert energie / 1e6 == pytest.approx(calculee, abs=0.3)
    assert energie / 1e6 == pytest.approx(imprimee, rel=0.04)


def test_les_deux_ordres_de_couches_ont_le_meme_regime() -> None:
    """« Rigoureusement equivalentes en regime stationnaire (memes resistances) »."""
    directe = ParoiEnTranches((FIBRES, BRIQUES), interieur=20.0, exterieur=15.0)
    inverse = ParoiEnTranches((BRIQUES, FIBRES), interieur=20.0, exterieur=15.0)
    assert directe.resistance_totale() == pytest.approx(inverse.resistance_totale())
    attendue = 1 / 20 + 0.05 / 0.15 + 0.23 / 1.15 + 1 / 15
    assert directe.resistance_totale() == pytest.approx(attendue)


TRIANGLE = ((0, 0), (4, 0), (1, 3))
X, Y = sp.symbols("x y")


def test_les_fonctions_de_forme_du_triangle_lineaire() -> None:
    """(5.14) : `N_i(x_j) = delta_ij`, et leur somme vaut un."""
    formes = fonctions_de_forme(TRIANGLE)
    for i, forme in enumerate(formes):
        for j, (x, y) in enumerate(TRIANGLE):
            assert forme.subs({X: x, Y: y}) == (1 if i == j else 0)
        assert sp.Poly(forme, X, Y).total_degree() == 1
    assert sp.simplify(sum(formes) - 1) == 0


def test_les_fonctions_de_forme_du_triangle_quadratique() -> None:
    """Page 100 : six noeuds, les sommets et les milieux des cotes."""
    milieux = (
        (2, 0),
        (sp.Rational(5, 2), sp.Rational(3, 2)),
        (sp.Rational(1, 2), sp.Rational(3, 2)),
    )
    noeuds = (*TRIANGLE, *milieux)
    formes = fonctions_de_forme(noeuds)
    for i, forme in enumerate(formes):
        for j, (x, y) in enumerate(noeuds):
            assert forme.subs({X: x, Y: y}) == (1 if i == j else 0)
        assert sp.Poly(forme, X, Y).total_degree() == 2
    assert sp.simplify(sum(formes) - 1) == 0
    with pytest.raises(ValueError, match="trois ou six noeuds"):
        fonctions_de_forme(TRIANGLE[:2])


def test_la_matrice_de_rigidite_d_un_triangle() -> None:
    """Page 99 : `K_ij = integrale de lambda grad N_i . grad N_j`."""
    rigidite = rigidite_d_un_triangle(TRIANGLE, conductivite=2)
    assert rigidite == rigidite.T
    assert all(sum(rigidite.row(i)) == 0 for i in range(3))
    formes = fonctions_de_forme(TRIANGLE)
    gradients = [(sp.diff(f, X), sp.diff(f, Y)) for f in formes]
    aire = sp.Rational(1, 2) * 4 * 3
    produit = gradients[0][0] * gradients[1][0] + gradients[0][1] * gradients[1][1]
    assert rigidite[0, 1] == 2 * aire * produit


def test_un_champ_lineaire_n_a_aucun_residu() -> None:
    """Un champ que les fonctions de forme portent exactement : `[K]{t} = 0`."""
    rigidite = rigidite_d_un_triangle(TRIANGLE, conductivite=2)
    nodales = sp.Matrix([3 * x - 2 * y + 7 for x, y in TRIANGLE])
    interpole = sum(
        n * f for n, f in zip(nodales, fonctions_de_forme(TRIANGLE), strict=True)
    )
    assert sp.simplify(interpole - (3 * X - 2 * Y + 7)) == 0
    flux_nodaux = rigidite * nodales
    assert sum(flux_nodaux) == 0
    assert flux_nodaux != sp.zeros(3, 1)
