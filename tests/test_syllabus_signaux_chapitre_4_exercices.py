"""Le syllabus de signaux et systemes, chapitre 4 : les exercices resolus.

Source : `SignauxSystemes/SYLLABUS/Syllabus_2019_Chapitre_04_Transmittances.pdf`,
pages 23 a 31. Les oracles sont les solutions imprimees : coefficients
passes a `impulse` et `step`, limites par les theoremes des valeurs
initiale et finale, lectures des graphes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from polytech.electronique.domain.laplace import (
    diviseur_de_tension,
    impedance_de_bobine,
    impedance_de_condensateur,
)
from polytech.maths.domain.laplace import theoremes
from polytech.maths.domain.symboles import IMAGE as P_DES_MATHS
from polytech.signaux.domain.composantes import nature_de_la_composante
from polytech.signaux.domain.fluence import transmittance_du_graphe
from polytech.signaux.domain.heaviside import original
from polytech.signaux.domain.modeles import transmittance_d_une_equation
from polytech.signaux.domain.poles_numeriques import (
    constante_de_temps,
    poles_et_zeros,
    reponse_impulsionnelle,
    reponse_indicielle,
)
from polytech.signaux.domain.stabilite_des_poles import nature_de_la_stabilite
from polytech.signaux.domain.symboles import IMAGE, TEMPS

P, T = IMAGE, TEMPS
RESISTANCE, CAPACITE, INDUCTANCE = 100, sp.Rational(1, 10**9), sp.Rational(1, 1000)
"""Exercice 4.1 : 100 ohms, 1 nF, 1 mH."""


def _limites(transmittance: sp.Expr) -> tuple[sp.Expr, ...]:
    """`h(0+)`, `h(infini)`, `a(0+)`, `a(infini)` par les deux theoremes."""
    image = transmittance.subs(P, P_DES_MATHS)
    indicielle = image / P_DES_MATHS
    return (
        theoremes.valeur_initiale(image),
        theoremes.valeur_finale(image),
        theoremes.valeur_initiale(indicielle),
        theoremes.valeur_finale(indicielle),
    )


def test_l_exercice_4_1_les_quatre_quadripoles_du_premier_ordre() -> None:
    """`[1],[RC 1]`, `[R],[L R]`, `[RC 0],[RC 1]` et `[L 0],[L R]`."""
    capacite = impedance_de_condensateur(CAPACITE, P)
    bobine = impedance_de_bobine(INDUCTANCE, P)
    rc, tau = RESISTANCE * CAPACITE, INDUCTANCE / RESISTANCE
    attendues = (
        (diviseur_de_tension(RESISTANCE, capacite), 1 / (rc * P + 1)),
        (diviseur_de_tension(bobine, RESISTANCE), 1 / (tau * P + 1)),
        (diviseur_de_tension(capacite, RESISTANCE), rc * P / (rc * P + 1)),
        (diviseur_de_tension(RESISTANCE, bobine), tau * P / (tau * P + 1)),
    )
    for calculee, imprimee in attendues:
        assert sp.simplify(calculee - imprimee) == 0


def test_l_exercice_4_1_les_reponses_impulsionnelles_lues_sur_les_graphes() -> None:
    """Departs a `10 x 10^6` et `10 x 10^4`, puis leurs opposes sous un Dirac."""
    rc, tau = RESISTANCE * CAPACITE, INDUCTANCE / RESISTANCE
    passe_bas = original(1 / (rc * P + 1))
    assert passe_bas.subs(T, 0) == 10**7
    assert original(1 / (tau * P + 1)).subs(T, 0) == 10**5
    passe_haut = original(rc * P / (rc * P + 1))
    assert passe_haut.coeff(sp.DiracDelta(T)) == 1
    assert (passe_haut - sp.DiracDelta(T)).subs(T, 0) == -(10**7)
    assert sp.simplify(original(1 / (P * (rc * P + 1))) - (1 - sp.exp(-T / rc))) == 0
    assert sp.simplify(original(rc / (rc * P + 1)) - sp.exp(-T / rc)) == 0


FAMILLES = {
    "A": (1 / (P + 1), 1, "exponentielle decroissante"),
    "B": (P / (P + 1), 0, "exponentielle decroissante"),
    "C": (4 / (P**2 + P + 4), 1, "sinusoide amortie"),
    "D": (16 / (P**2 + 2 * P + 16), 1, "sinusoide amortie"),
    "E": (P / (P**2 + P + 4), 0, "sinusoide amortie"),
}
"""Exercice 4.2 : la transmittance, la valeur finale de l'indicielle, la forme."""


@pytest.mark.parametrize("lettre", sorted(FAMILLES))
def test_l_exercice_4_2_chaque_famille_se_reconnait(lettre: str) -> None:
    """La valeur finale separe {A, C, D} de {B, E} ; les poles font le reste."""
    transmittance, finale, forme = FAMILLES[lettre]
    indicielle = (transmittance / P).subs(P, P_DES_MATHS)
    assert theoremes.valeur_finale(indicielle) == finale
    denominateur = sp.fraction(sp.cancel(transmittance))[1]
    for pole in sp.Poly(denominateur, P).all_roots():
        assert nature_de_la_composante(complex(pole)) == forme


def test_l_exercice_4_2_c_et_d_se_distinguent_par_leur_pulsation() -> None:
    """`C` oscille a 1,94 rad/s et s'amortit en 2 s ; `D` a 3,87 rad/s, en 1 s."""
    poles_c = sp.Poly(P**2 + P + 4, P).all_roots()
    poles_d = sp.Poly(P**2 + 2 * P + 16, P).all_roots()
    assert abs(complex(poles_c[0]).imag) == pytest.approx(1.94, abs=0.01)
    assert abs(complex(poles_d[0]).imag) == pytest.approx(3.87, abs=0.01)
    assert constante_de_temps(complex(poles_c[0])) == pytest.approx(2.0)
    assert constante_de_temps(complex(poles_d[0])) == pytest.approx(1.0)


def test_l_exercice_4_3_les_limites_du_circuit_de_l_exemple_4_6() -> None:
    """1 ohm, 1 mF, 1 mH : `h` part de 0 et y revient, `a` va de 0 a 1/2."""
    capacite = inductance = sp.Rational(1, 1000)
    transmittance = 1 / (P**2 * inductance * capacite + P * (capacite + inductance) + 2)
    assert _limites(transmittance) == (0, 0, 0, sp.Rational(1, 2))


def test_l_exercice_4_3_le_sommet_des_deux_graphes() -> None:
    """`h` culmine vers 320 a 0,8 ms ; `a` depasse a peine 0,5."""
    numerateur, denominateur = [1.0], [1e-6, 2e-3, 2.0]
    instants = np.linspace(0, 7e-3, 7001)
    impulsionnelle = reponse_impulsionnelle(numerateur, denominateur, instants)
    assert impulsionnelle.max() == pytest.approx(322.4, abs=0.5)
    assert instants[impulsionnelle.argmax()] == pytest.approx(math.pi / 4e3, abs=2e-6)
    indicielle = reponse_indicielle(numerateur, denominateur, instants)
    assert indicielle.max() == pytest.approx(0.5 * (1 + math.exp(-math.pi)), abs=1e-4)
    assert indicielle[-1] == pytest.approx(0.5, abs=1e-3)


def test_l_exercice_4_4_le_graphe_a_deux_integrateurs() -> None:
    """Retours -20 et -200, gains directs 1 et 0 : `p / (p^2 + 20 p + 200)`."""
    transmittance = transmittance_du_graphe(retours=(-20, -200), directs=(1, 0))
    assert sp.simplify(transmittance - P / (P**2 + 20 * P + 200)) == 0
    assert _limites(transmittance) == (1, 0, 0, 0)
    poles, zeros, _ = poles_et_zeros([1.0, 0.0], [1.0, 20.0, 200.0])
    assert sorted(poles.imag) == pytest.approx([-10.0, 10.0])
    assert constante_de_temps(complex(poles[0])) == pytest.approx(0.1)
    assert zeros == pytest.approx([0.0])


def test_l_exercice_4_5_deux_poles_reels() -> None:
    """`(p + 1) / (p^2 + 20 p + 20)`.

    Le syllabus annonce des constantes de temps « egales a 1 s et 1/20 s » :
    ce sont des ordres de grandeur, les poles valant -1,056 et -18,94.
    """
    transmittance = transmittance_du_graphe(retours=(-20, -20), directs=(1, 1))
    assert sp.simplify(transmittance - (P + 1) / (P**2 + 20 * P + 20)) == 0
    assert _limites(transmittance) == (1, 0, 0, sp.Rational(1, 20))
    poles, _, _ = poles_et_zeros([1.0, 1.0], [1.0, 20.0, 20.0])
    lentes = sorted(constante_de_temps(complex(pole)) for pole in poles)
    assert lentes == pytest.approx([0.0528, 0.947], abs=1e-3)


def test_l_exercice_4_6_la_capsule_spatiale_est_instable() -> None:
    """`theta''' + theta'' + 10 theta' + 15 theta = 5 r`.

    Les poles sont -1,41 et `0,2 +/- 3,25 j` : l'oscillation croit, a
    3,25 rad/s (le syllabus arrondit a 3 rad/s et `T = 2 s`). La formule
    de la valeur finale rend 1/3, mais le theoreme ne s'applique pas.
    """
    transmittance = transmittance_d_une_equation(sortie=(1, 1, 10, 15), entree=(5,))
    assert sp.simplify(transmittance.expression - 5 / (P**3 + P**2 + 10 * P + 15)) == 0
    assert nature_de_la_stabilite(transmittance.expression) == "instable"
    poles, _, _ = poles_et_zeros([5.0], [1.0, 1.0, 10.0, 15.0])
    complexes = [pole for pole in poles if abs(pole.imag) > 1]
    assert complexes[0].real == pytest.approx(0.2, abs=0.01)
    assert abs(complexes[0].imag) == pytest.approx(3.25, abs=0.01)
    assert 2 * math.pi / abs(complexes[0].imag) == pytest.approx(1.93, abs=0.01)
    indicielle = (transmittance.expression / P).subs(P, P_DES_MATHS)
    assert theoremes.limite_en_zero(indicielle) == sp.Rational(1, 3)
    with pytest.raises(ValueError, match="theoreme inapplicable"):
        theoremes.valeur_finale(indicielle)
