"""Planches de signaux et systemes : du systeme au modele.

Source : `SignauxSystemes/SLIDES/Slides_03_Systemes.pdf` (48 planches),
lues sur les pages rendues.

Les oracles : le circuit RLC (planches 16 et 24 a 26), le reservoir
(planches 17 et 27 a 32, et les courbes des planches 39 a 42 : `h^0`
vaut 0,444 pour `q_e = 0,1` et 0,64 pour 0,12, 2,8 pour 0,25, soit
`alpha = 0,15`), la suspension active (planches 33 a 37) et le modele
identifie `1 / (p^2 + p + 2,72)` de la planche 50.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from polytech.signaux.domain.modeles import transmittance_d_une_equation
from polytech.signaux.domain.reponse_indicielle import depassement_du_second_ordre
from polytech.signaux.domain.reservoir import Reservoir
from polytech.signaux.domain.second_ordre import SecondOrdre
from polytech.signaux.domain.suspension import Suspension, sortie_par_superposition
from polytech.signaux.domain.symboles import IMAGE
from polytech.signaux.domain.systemes import (
    Portee,
    linearisation_de_taylor,
    point_de_fonctionnement,
    portee_du_modele,
)

P = IMAGE
RESERVOIR = Reservoir(section=1.0, coefficient=0.15)


def test_l_exemple_2_le_circuit_rlc_donne_un_second_ordre() -> None:
    """Planches 24 a 26 : `L C u_C'' + R C u_C' + u_C = u`, puis sa transmittance."""
    resistance, inductance, capacite = sp.symbols("R L C", positive=True)
    modele = transmittance_d_une_equation(
        sortie=(inductance * capacite, resistance * capacite, 1), entree=(1,)
    )
    attendue = 1 / (P**2 * inductance * capacite + P * resistance * capacite + 1)
    assert sp.simplify(modele.expression - attendue) == 0
    assert portee_du_modele(lineaire=True) is Portee.GLOBALE


def test_le_point_de_fonctionnement_du_reservoir() -> None:
    """Planche 28 : `q_s^0 = q_e^0` et `h^0 = (q_e^0 / alpha)^2`."""
    assert RESERVOIR.hauteur_de_repos(0.1) == pytest.approx(0.444, abs=5e-4)
    assert RESERVOIR.hauteur_de_repos(0.12) == pytest.approx(0.64)
    assert RESERVOIR.hauteur_de_repos(0.25) == pytest.approx(2.8, abs=0.03)
    assert RESERVOIR.derivee(RESERVOIR.hauteur_de_repos(0.1), 0.1) == pytest.approx(0.0)
    hauteur = sp.Symbol("h", positive=True)
    (repos,) = point_de_fonctionnement(0.1 - 0.15 * sp.sqrt(hauteur), hauteur)
    assert float(repos) == pytest.approx(RESERVOIR.hauteur_de_repos(0.1))


def test_la_linearisation_du_debit_de_sortie() -> None:
    """Planche 29 : `racine(h) = racine(h^0) + Delta h / (2 racine(h^0))`."""
    hauteur, repos = sp.symbols("h h_0", positive=True)
    locale = linearisation_de_taylor(sp.sqrt(hauteur), hauteur, repos)
    attendue = sp.sqrt(repos) + (hauteur - repos) / (2 * sp.sqrt(repos))
    assert sp.simplify(locale - attendue) == 0
    assert RESERVOIR.pente_de_sortie(0.1) == pytest.approx(0.15**2 / (2 * 0.1))
    assert portee_du_modele(lineaire=False) is Portee.LOCALE


def test_le_modele_local_du_reservoir() -> None:
    """Planche 32 : `H(p) / Q_e(p) = 1 / (p S + alpha / (2 racine(h^0)))`."""
    locale = RESERVOIR.modele_local(0.1)
    pente = 0.15 / (2 * 0.1 / 0.15)
    assert sp.simplify(locale.expression - 1 / (P + pente)) == 0
    gain, constante = RESERVOIR.premier_ordre_local(0.1)
    assert gain == pytest.approx(1 / pente)
    assert constante == pytest.approx(1.0 / pente)


def test_le_modele_local_suit_le_modele_global_pres_du_point() -> None:
    """Planche 42 : de 0,10 a 0,12, le niveau va de 0,444 a 0,64 ; le local a 0,622."""
    instants = np.linspace(0, 120, 2401)
    globale = RESERVOIR.simuler(lambda _: 0.12, depart=0.1, instants=instants)
    assert globale[0] == pytest.approx(0.444, abs=5e-4)
    assert globale[-1] == pytest.approx(0.64, abs=1e-3)
    gain, constante = RESERVOIR.premier_ordre_local(0.1)
    locale = 0.444444 + gain * 0.02 * (1 - np.exp(-instants / constante))
    assert locale[-1] == pytest.approx(0.622, abs=1e-3)
    assert np.max(np.abs(locale - globale)) < 0.02
    with pytest.raises(ValueError, match="debit nul ou negatif"):
        RESERVOIR.hauteur_de_repos(0.0)


def test_la_section_ne_change_que_la_vitesse_du_reservoir() -> None:
    """`S` divise la derivee et multiplie la constante de temps, pas le repos."""
    large = Reservoir(section=2.0, coefficient=0.15)
    assert large.hauteur_de_repos(0.1) == RESERVOIR.hauteur_de_repos(0.1)
    assert large.derivee(0.2, 0.1) == pytest.approx(RESERVOIR.derivee(0.2, 0.1) / 2)
    assert large.premier_ordre_local(0.1)[1] == pytest.approx(
        2 * RESERVOIR.premier_ordre_local(0.1)[1]
    )


def test_le_reservoir_n_est_pas_lineaire() -> None:
    """Planche 17 : doubler le debit quadruple la hauteur de repos."""
    simple, double = RESERVOIR.hauteur_de_repos(0.1), RESERVOIR.hauteur_de_repos(0.2)
    assert double == pytest.approx(4 * simple)


M, K, F, L0, G = sp.symbols("M k f L_0 g", positive=True)
SUSPENSION = Suspension(masse=M, raideur=K, frottement=F, longueur=L0, pesanteur=G)


def test_le_point_de_fonctionnement_de_la_suspension() -> None:
    """Planche 34 : `x_v^0 = x_r^0 + L_0 + (F^0 - M g) / k`."""
    route, force = sp.symbols("x_r F_0")
    repos = SUSPENSION.position_de_repos(route, force)
    assert sp.simplify(repos - (route + L0 + (force - M * G) / K)) == 0
    assert sp.simplify(SUSPENSION.acceleration(repos, route, force)) == 0


def test_les_deux_transmittances_de_la_suspension() -> None:
    """Planche 36 : `(p f + k) / (p^2 M + p f + k)` et `1 / (p^2 M + p f + k)`."""
    de_la_route, de_la_force = SUSPENSION.transmittances()
    denominateur = P**2 * M + P * F + K
    assert sp.simplify(de_la_route.expression - (P * F + K) / denominateur) == 0
    assert sp.simplify(de_la_force.expression - 1 / denominateur) == 0
    assert de_la_route.expression.subs(P, 0) == 1


def test_la_sortie_est_la_somme_des_effets_de_chaque_entree() -> None:
    """Planches 37 et 51 : `Y = G_u U + G_p P`."""
    de_la_route, de_la_force = SUSPENSION.transmittances()
    route, force = sp.symbols("X_r F_p")
    sortie = sortie_par_superposition({de_la_route: route, de_la_force: force})
    attendue = ((P * F + K) * route + force) / (P**2 * M + P * F + K)
    assert sp.simplify(sortie - attendue) == 0
    with pytest.raises(ValueError, match="aucune entree"):
        sortie_par_superposition({})


def test_le_modele_identifie_de_la_planche_50() -> None:
    """`1 / (p^2 + p + 2,72)` : regime 0,37 et sommet 0,50, comme le releve."""
    modele = SecondOrdre.depuis_evans(
        1, sp.Rational(1, 2), sp.sqrt(sp.Rational(272, 100))
    )
    assert float(modele.gain) == pytest.approx(0.368, abs=5e-4)
    depassement = float(depassement_du_second_ordre(modele.amortissement))
    assert float(modele.gain) * (1 + depassement) == pytest.approx(0.50, abs=0.005)
