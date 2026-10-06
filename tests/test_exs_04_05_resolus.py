"""Signaux et systemes : la seance d'exercices « Chap. 4-5 », huit exercices.

Source : `SignauxSystemes/EXS/Exs_04_05_Resolus.pdf`, quatre pages. La
feuille porte une reponse breve sous la plupart des enonces -- une
condition, une association, deux croquis --, recopiees ici comme oracles.
Chacune est recoupee par un autre chemin : `scipy.signal`, le solveur de
reseaux du cours d'electromagnetisme, l'inversion formelle de Laplace.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp
from scipy import signal

from polytech.electromagnetisme.domain.circuits.reseau import (
    Reseau,
    Resistance,
    SourceDeCourant,
    SourceDeTension,
)
from polytech.signaux.application import amortie_exs_45 as am
from polytech.signaux.application import association_exs_45 as asso
from polytech.signaux.application import circuits_exs_45 as ci
from polytech.signaux.application import systemes_exs_45 as sy
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T

# --- Exercice 1 : dessiner une sinusoide amortie ----------------------------


def test_ex1_les_reperes_du_dessin() -> None:
    """Periode de 10 ms, constante de temps de 100 ms, depart en `cos(pi/4)`."""
    reperes = am.reperes_de_la_sinusoide()
    assert reperes.periode == pytest.approx(0.01, rel=1e-12)
    assert reperes.constante == pytest.approx(0.1, rel=1e-12)
    assert reperes.depart == pytest.approx(math.sqrt(2) / 2, rel=1e-12)
    assert reperes.periodes_par_constante == pytest.approx(10.0, rel=1e-12)


# --- Exercice 2 : l'allure sans inverser Laplace ----------------------------


def test_ex2_les_deux_theoremes() -> None:
    """`u(0+) = lim p U = 1` et `u(infini) = 0`."""
    assert am.valeurs_extremes() == (1, 0)


def test_ex2_le_sens_physique_des_poles() -> None:
    """`sigma = 1`, `rho = racine(2)`, donc une pulsation de `1` rad/s."""
    assert am.amortissement_et_pulsation() == (1, 1)


def test_ex2_la_reponse_de_la_feuille() -> None:
    """« Amorti apres 5 s => meme pas une periode » : `5 tau = 5 < 2 pi`."""
    duree, periode = am.duree_et_periode()
    assert duree == 5
    assert periode == 2 * sp.pi
    assert duree < periode


def test_ex2_par_l_inversion_formelle() -> None:
    """Second chemin, celui que l'enonce interdit : `e^-t (cos t - sin t)`."""
    attendu = sp.exp(-T) * (sp.cos(T) - sp.sin(T))
    assert sp.simplify(am.original() - attendu) == 0


# --- Exercice 3 : trois systemes --------------------------------------------


def test_ex3_les_trois_transmittances() -> None:
    """`1/(1 + RCp)`, `Lp/(R + Lp)`, et `p/(p^2 + 20p + 200)` pour le graphe."""
    rc, rl, graphe = sy.transmittances()
    assert sp.cancel(rc - 1 / (1 + sp.Rational(1, 10**7) * P)) == 0
    assert sp.cancel(rl - P / (10**5 + P)) == 0
    assert sp.cancel(graphe - P / (P**2 + 20 * P + 200)) == 0


def test_ex3_les_constantes_de_temps() -> None:
    """`RC = 0,1` microseconde ; `L/R = 10` microsecondes."""
    assert sy.constantes_de_temps() == pytest.approx((1e-7, 1e-5), rel=1e-12)


def test_ex3_les_valeurs_initiales_et_finales() -> None:
    """Question (c) : les deux theoremes sur les reponses indicielles."""
    assert sy.extremes_des_indicielles() == {
        "rc": (0, 1),
        "rl": (1, 0),
        "graphe": (0, 0),
    }


def test_ex3_la_reponse_impulsionnelle_du_graphe() -> None:
    """`h(t) = e^(-10 t) (cos 10 t - sin 10 t)` : part de 1, s'annule en `pi/40`."""
    attendu = sp.exp(-10 * T) * (sp.cos(10 * T) - sp.sin(10 * T))
    assert sp.simplify(sy.impulsionnelle_du_graphe() - attendu) == 0
    assert sy.premier_zero_du_graphe() == sp.pi / 40
    assert float(sy.premier_zero_du_graphe()) == pytest.approx(0.0785, abs=5e-5)


def test_ex3_par_scipy() -> None:
    """Second chemin : `scipy.signal.impulse` sur les coefficients du graphe."""
    instants = np.linspace(0.0, 0.5, 51)
    _, numerique = signal.impulse(([1, 0], [1, 20, 200]), T=instants)
    formelle = sp.lambdify(T, sy.impulsionnelle_du_graphe(), "numpy")(instants)
    assert np.max(np.abs(numerique - formelle)) < 1e-6


# --- Exercice 4 : associer transmittances, reponses et diagrammes -----------


def test_ex4_l_association_de_la_feuille() -> None:
    """Page 2 : « a2, b5, c1, d4, e3 » -- reponse indicielle, puis diagramme."""
    assert asso.reponses_et_diagrammes() == {"a": 2, "b": 5, "c": 1, "d": 4, "e": 3}


def test_ex4_les_transmittances_derriere_l_association() -> None:
    """Ce que la feuille ne dit pas : quelle transmittance porte chaque couple."""
    assert asso.association() == {
        "A": ("e", 3),
        "B": ("b", 5),
        "C": ("a", 2),
        "D": ("d", 4),
        "E": ("c", 1),
    }


def test_ex4_les_depassements_distinguent_c_de_d() -> None:
    """Second chemin : `C` depasse de 44 %, `D` de 44 % aussi -- c'est la vitesse."""
    assert asso.depassement("C") == pytest.approx(asso.depassement("D"), abs=1e-9)
    assert asso.periode("D") < asso.periode("C") / 1.9


def test_ex4_les_valeurs_du_depassement_et_des_periodes() -> None:
    """`xi = 1/4` pour `C` et `D` : `e^(-pi xi/racine(1 - xi^2))`, soit 44,4 %."""
    attendu = math.exp(-math.pi * 0.25 / math.sqrt(1 - 0.0625))
    assert asso.depassement("C") == pytest.approx(attendu, rel=1e-12)
    assert asso.periode("C") == pytest.approx(2 * math.pi / math.sqrt(3.75), rel=1e-12)
    assert asso.periode("D") == pytest.approx(2 * math.pi / math.sqrt(15), rel=1e-12)


def test_ex4_les_cinq_transmittances_sont_stables() -> None:
    """Aucun pole a droite : sans cela les reponses de la feuille divergeraient."""
    for nom, fonction in asso.FONCTIONS.items():
        poles = sp.roots(sp.fraction(fonction)[1], P)
        assert all(sp.re(pole) < 0 for pole in poles), nom


# --- Exercice 5 : un circuit RLC parallele ----------------------------------


def test_ex5_la_condition_de_la_feuille() -> None:
    """Page 3 : `R > 1/2 racine(L/C)`."""
    assert ci.resistance_limite() == sp.sqrt(ci.L / ci.C) / 2


def test_ex5_la_constante_de_temps_et_la_pulsation() -> None:
    """`tau = 2RC`, `omega = racine(1/(LC) - 1/(2RC)^2)`."""
    constante, pulsation = ci.constante_et_pulsation()
    assert sp.simplify(constante - 2 * ci.R * ci.C) == 0
    attendue = sp.sqrt(1 / (ci.L * ci.C) - 1 / (2 * ci.R * ci.C) ** 2)
    assert sp.simplify(pulsation**2 - attendue**2) == 0


def test_ex5_par_le_reseau_d_electromagnetisme() -> None:
    """Lien entre les cours : l'impedance vue par la source de courant."""
    circuit = Reseau(
        (
            SourceDeCourant("masse", "haut", sp.Integer(1)),
            Resistance("haut", "masse", 1 / (P * ci.C)),
            Resistance("haut", "masse", P * ci.L),
            Resistance("haut", "masse", ci.R),
        ),
        "masse",
    )
    assert sp.cancel(circuit.potentiels()["haut"] - ci.impedance_parallele()) == 0


# --- Exercice 6 : un circuit dont le condensateur ne compte pas -------------


def test_ex6_la_transmittance() -> None:
    """`R/(R + Lp)` : le condensateur, en parallele sur l'entree, n'y figure pas."""
    assert sp.cancel(ci.transmittance_du_six() - ci.R / (ci.R + ci.L * P)) == 0


def test_ex6_la_reponse_de_la_feuille() -> None:
    """Page 3, croquis : `h(0) = 10^4`, constante de temps `10^-4` s."""
    depart, constante = ci.reperes_du_six()
    assert depart == pytest.approx(1e4, rel=1e-12)
    assert constante == pytest.approx(1e-4, rel=1e-12)


def test_ex6_le_train_d_impulsions() -> None:
    """Periode de 10 ms, cent constantes de temps : chaque reponse s'eteint a temps."""
    assert ci.periode_en_constantes_de_temps() == pytest.approx(100.0, rel=1e-12)
    instants = np.array([0.0, 5e-5, 0.01, 0.01 + 5e-5])
    reponse = ci.reponse_au_train(instants)
    assert reponse[0] == pytest.approx(1e4, rel=1e-12)
    assert reponse[2] == pytest.approx(reponse[0], rel=1e-12)
    assert reponse[3] == pytest.approx(reponse[1], rel=1e-12)


def test_ex6_par_le_reseau_d_electromagnetisme() -> None:
    """Lien entre les cours : le circuit complet, condensateur compris."""
    circuit = Reseau(
        (
            SourceDeTension("entree", "masse", sp.Integer(1)),
            Resistance("entree", "masse", 1 / (P * ci.C)),
            Resistance("entree", "sortie", P * ci.L),
            Resistance("sortie", "masse", ci.R),
        ),
        "masse",
    )
    assert sp.cancel(circuit.potentiels()["sortie"] - ci.transmittance_du_six()) == 0


# --- Exercice 7 : regler un coefficient du graphe ---------------------------


def test_ex7_la_condition_de_la_feuille() -> None:
    """Page 4 : `alpha >= 20 racine(2)`."""
    assert sy.alpha_limite() == 20 * sp.sqrt(2)


def test_ex7_de_part_et_d_autre_de_la_limite() -> None:
    """Second chemin : les poles sont reels au-dessus, complexes en dessous."""
    assert all(pole.is_real for pole in sy.poles_pour_alpha(30))
    assert not any(pole.is_real for pole in sy.poles_pour_alpha(28))


# --- Exercice 8 : une reponse indicielle ------------------------------------


def test_ex8_la_transmittance_de_la_feuille() -> None:
    """Page 4 : `R/(p^2 RLC + p (R^2 C + L) + 2R)`."""
    denominateur = P**2 * ci.R * ci.L * ci.C + P * (ci.R**2 * ci.C + ci.L) + 2 * ci.R
    assert sp.cancel(ci.transmittance_du_huit() - ci.R / denominateur) == 0


def test_ex8_les_reperes_de_la_reponse() -> None:
    """Poles en `-1000 +- 1000 j` : valeur finale `1/2`, depassement de `e^-pi`."""
    reperes = ci.reperes_du_huit()
    assert reperes.finale == sp.Rational(1, 2)
    assert reperes.poles == (-1000 - 1000 * sp.I, -1000 + 1000 * sp.I)
    assert float(reperes.depassement) == pytest.approx(math.exp(-math.pi), rel=1e-9)


def test_ex8_par_scipy() -> None:
    """Second chemin : `scipy.signal.step` sur les coefficients chiffres."""
    instants = np.linspace(0.0, 0.008, 81)
    _, numerique = signal.step(([1.0], [1e-6, 2e-3, 2.0]), T=instants)
    formelle = sp.lambdify(T, ci.indicielle_du_huit(), "numpy")(instants)
    assert np.max(np.abs(numerique - formelle)) < 1e-6


def test_ex8_par_le_reseau_d_electromagnetisme() -> None:
    """Lien entre les cours : le circuit resolu par ses noeuds, en impedances."""
    circuit = Reseau(
        (
            SourceDeTension("entree", "masse", sp.Integer(1)),
            Resistance("entree", "milieu", ci.R),
            Resistance("milieu", "masse", 1 / (P * ci.C)),
            Resistance("milieu", "sortie", P * ci.L),
            Resistance("sortie", "masse", ci.R),
        ),
        "masse",
    )
    assert sp.cancel(circuit.potentiels()["sortie"] - ci.transmittance_du_huit()) == 0
