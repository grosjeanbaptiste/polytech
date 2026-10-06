"""Signaux et systemes : la feuille « Chapitre 3 : Systemes », six exercices.

Source : `SignauxSystemes/EXS/Exs_03_Systemes.pdf`, trois pages. Chaque
exercice y porte sa reponse, sans le chemin ; ces reponses sont recopiees
comme oracles, et chacune est recoupee par un autre chemin : la matrice
jacobienne, la simulation du modele non lineaire, le reseau du contexte
d'electromagnetisme.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.reseau import (
    Reseau,
    Resistance,
    SourceDeCourant,
)
from polytech.signaux.application import circuits_exs_03 as ci
from polytech.signaux.application import linearises_exs_03 as li
from polytech.signaux.application import procede_exs_03 as pr
from polytech.signaux.application import reservoir_exs_03 as re
from polytech.signaux.domain.symboles import IMAGE as P

# --- Exercice 1 : une equation non lineaire du second ordre ------------------


def test_ex1_les_deux_points_de_repos() -> None:
    """`2 x^2 = u` pour `u = 0,5` : `x = +-1/2`, la feuille retient le positif."""
    assert li.repos_du_premier() == (-sp.Rational(1, 2), sp.Rational(1, 2))


def test_ex1_la_transmittance_de_la_feuille() -> None:
    """Page 1 : `(3p + 1)/(p^2 + 3p + 2)`."""
    attendue = (3 * P + 1) / (P**2 + 3 * P + 2)
    assert sp.cancel(li.transmittance_du_premier().expression - attendue) == 0


def test_ex1_l_autre_point_de_repos_est_instable() -> None:
    """Autour de `x = -1/2` : `p^2 + 3p - 2`, un pole a partie reelle positive."""
    autre = li.transmittance_du_premier(repos=-sp.Rational(1, 2))
    assert autre.denominateur() == P**2 + 3 * P - 2
    assert max(sp.re(pole) for pole in autre.poles()) > 0


def test_ex1_par_la_simulation_du_modele_non_lineaire() -> None:
    """Second chemin : un petit echelon sur le modele global, face au modele local."""
    instants = np.linspace(0.0, 6.0, 61)
    globale, locale = li.reponses_a_un_petit_echelon(instants, amplitude=1e-3)
    assert np.max(np.abs(globale - locale)) < 1e-6
    assert locale[-1] == pytest.approx(0.5e-3, rel=2e-2, abs=0)


def test_ex1_l_erreur_du_modele_local_est_du_second_ordre() -> None:
    """Un echelon deux fois plus grand : l'ecart au modele global quadruple."""
    instants = np.linspace(0.0, 6.0, 61)
    ecarts = []
    for amplitude in (5e-3, 1e-2):
        globale, locale = li.reponses_a_un_petit_echelon(instants, amplitude=amplitude)
        ecarts.append(float(np.max(np.abs(globale - locale))))
    assert ecarts[1] / ecarts[0] == pytest.approx(4.0, rel=2e-2, abs=0)


# --- Exercice 2 : trois equations en cascade ---------------------------------


def test_ex2_le_point_de_repos_de_la_feuille() -> None:
    """Page 1 : `x1 = x2 = 1,5` et `y = 3,75`."""
    assert li.repos_du_second() == (
        sp.Rational(3, 2),
        sp.Rational(3, 2),
        sp.Rational(15, 4),
    )


def test_ex2_la_transmittance_de_la_feuille() -> None:
    """Page 1 : `6 (4p + 5)/((p + 2)^2 (p + 1))`."""
    attendue = 6 * (4 * P + 5) / ((P + 2) ** 2 * (P + 1))
    assert sp.cancel(li.transmittance_du_second().expression - attendue) == 0


def test_ex2_le_zero_de_la_feuille() -> None:
    """Page 1 : `p = -5/4`."""
    assert li.transmittance_du_second().zeros() == (-sp.Rational(5, 4),)


def test_ex2_par_la_matrice_jacobienne() -> None:
    """Second chemin : `C (pI - A)^-1 B`, les derivees prises au point de repos."""
    ecart = li.transmittance_par_l_etat() - li.transmittance_du_second().expression
    assert sp.cancel(ecart) == 0


def test_ex2_le_gain_statique_est_la_pente_de_la_caracteristique() -> None:
    """`y^0(u) = 15 u^2/4` : sa derivee en `u = 1` vaut `T(0) = 15/2`."""
    assert li.transmittance_du_second().expression.subs(P, 0) == sp.Rational(15, 2)
    assert li.pente_de_la_caracteristique() == sp.Rational(15, 2)


# --- Exercice 3 : le reservoir -----------------------------------------------


def test_ex3_le_point_de_fonctionnement_de_la_feuille() -> None:
    """Page 2 : `h^0 = 1,15` m."""
    assert re.hauteur_de_repos() == pytest.approx(1.15, abs=5e-3)
    assert re.hauteur_de_repos() == pytest.approx((0.75 / 0.7) ** 2, rel=1e-12, abs=0)


def test_ex3_le_premier_ordre_de_la_feuille() -> None:
    """Page 2 : `3,06/(1,53 p + 1)`."""
    gain, constante = re.premier_ordre()
    assert gain == pytest.approx(3.06, abs=5e-3)
    assert constante == pytest.approx(1.53, abs=5e-3)


def test_ex3_l_erreur_de_la_feuille() -> None:
    """Page 2 : `h_exacte - h_linearisee = 0,011` m pour dix pour cent de debit."""
    exacte, linearisee = re.hauteurs_apres_la_variation()
    assert exacte - linearisee == pytest.approx(0.011, abs=5e-4)
    assert exacte > linearisee


def test_ex3_l_erreur_est_le_terme_du_second_ordre() -> None:
    """`h = (q/k)^2` est un polynome : l'erreur vaut exactement `(dq/k)^2`."""
    exacte, linearisee = re.hauteurs_apres_la_variation()
    attendue = (0.1 * 0.75 / 0.7) ** 2
    assert exacte - linearisee == pytest.approx(attendue, rel=1e-9, abs=0)


def test_ex3_par_la_simulation_du_reservoir() -> None:
    """Second chemin : le modele global integre jusqu'au nouveau regime."""
    exacte, _ = re.hauteurs_apres_la_variation()
    assert re.hauteur_simulee() == pytest.approx(exacte, rel=1e-6, abs=0)


# --- Exercice 4 : un procede a quatre variables ------------------------------


def test_ex4_le_regime_de_la_feuille() -> None:
    """Page 2 : `y = 60`, `x1 = x2 = 5`, `x3 = 50`, `x4 = 10`."""
    assert pr.regime() == {"y": 60, "x1": 5, "x2": 5, "x3": 50, "x4": 10}


def test_ex4_la_transmittance_du_schema_bloc() -> None:
    """Page 3 : `0,5/(p+0,5)` plus l'entree, puis `10/(p+2)`, plus `2/(p+1)`."""
    demi = sp.Rational(1, 2)
    attendue = (1 + demi / (P + demi)) * 10 / (P + 2) + 2 / (P + 1)
    assert sp.cancel(pr.transmittance().expression - attendue) == 0


def test_ex4_le_gain_statique_redonne_le_regime() -> None:
    """Second chemin : `T(0) = 12`, et `12 x 5 = 60`."""
    assert pr.transmittance().expression.subs(P, 0) == 12
    assert pr.transmittance().expression.subs(P, 0) * pr.COMMANDE == pr.regime()["y"]


def test_ex4_le_procede_est_stable() -> None:
    """Trois poles reels negatifs : `-1/2`, `-1` et `-2`."""
    assert pr.transmittance().poles() == (-2, -1, -sp.Rational(1, 2))


# --- Exercice 5 : masse, ressort et amortisseur sur un chariot --------------


def test_ex5_la_transmittance_de_la_feuille() -> None:
    """Page 3 : `(p c + k)/(p^2 m + p c + k)`."""
    attendue = (P * ci.C + ci.K) / (P**2 * ci.M + P * ci.C + ci.K)
    assert sp.cancel(ci.transmittance_du_chariot().expression - attendue) == 0


def test_ex5_a_basse_frequence_la_masse_suit_le_chariot() -> None:
    assert ci.transmittance_du_chariot().expression.subs(P, 0) == 1


def test_ex5_par_la_simulation() -> None:
    """Second chemin : Newton integre, compare a la reponse indicielle."""
    instants = np.linspace(0.0, 8.0, 41)
    simulee, analytique = ci.reponses_du_chariot(instants)
    assert np.max(np.abs(simulee - analytique)) < 1e-6


# --- Exercice 6 : un circuit alimente en courant -----------------------------


def test_ex6_la_transmittance_de_la_feuille() -> None:
    """Page 3 : `(p L + R)/(p^2 L C + p R C + 1)`."""
    attendue = (P * ci.L + ci.R) / (
        P**2 * ci.L * ci.CAPACITE + P * ci.R * ci.CAPACITE + 1
    )
    assert sp.cancel(ci.transmittance_du_circuit().expression - attendue) == 0


def test_ex6_par_le_reseau_d_electromagnetisme() -> None:
    """Lien entre les cours : le meme circuit, resolu par ses noeuds en impedances."""
    circuit = Reseau(
        (
            SourceDeCourant("masse", "haut", sp.Integer(1)),
            Resistance("haut", "masse", 1 / (P * ci.CAPACITE)),
            Resistance("haut", "milieu", P * ci.L),
            Resistance("milieu", "masse", ci.R),
        ),
        "masse",
    )
    tension = circuit.potentiels()["haut"]
    assert sp.cancel(tension - ci.transmittance_du_circuit().expression) == 0


def test_ex6_en_continu_le_condensateur_prend_la_tension_de_r() -> None:
    assert ci.transmittance_du_circuit().expression.subs(P, 0) == ci.R
