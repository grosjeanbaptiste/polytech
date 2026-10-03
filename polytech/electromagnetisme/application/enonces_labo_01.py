"""Les circuits et les appareils du laboratoire I d'electromagnetisme.

Source : `Electromagnetisme/LABOS/Lab_01_Mesures_electriques_Enonce.pdf`.

**La sonde** (figures 2 a 4) : `R2` -- avec `C2` en parallele s'il y en a
une -- entre la tension a mesurer `V1` et l'entree de l'oscilloscope, et
cette entree modelisee par `R1 = 1 Mohm` en parallele avec `C1 = 11 pF`.

**Les filtres** (figures 5 et 6) : le `R-C` de la partie 5, et le `R-L-C`
serie dont la sortie est prise sur `R = 1 kohm`, avec `C = 0,1 uF` et
`L = 10 mH`.

**Les appareils** : les precisions de la notice que l'enonce recopie.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.circuits.elements_operationnels import (
    Bobine,
    Capacite,
)
from polytech.electromagnetisme.domain.circuits.operationnel import ReseauOperationnel
from polytech.electromagnetisme.domain.circuits.reseau import (
    Resistance,
    SourceDeTension,
)
from polytech.electromagnetisme.domain.instruments import (
    precision_d_un_quotient,
    precision_d_une_lecture,
)
from polytech.electromagnetisme.domain.transitoire.laplace import VARIABLE

R_ENTREE: Final = sp.Integer(10**6)
R_SONDE: Final = sp.Integer(9 * 10**6)
C_ENTREE: Final = sp.Rational(11, 10**12)
RESISTANCE_MESUREE: Final = 2200.0
"""La resistance que la partie 2 fait mesurer, en ohms."""
L_RLC, C_RLC, R_RLC = sp.Rational(1, 100), sp.Rational(1, 10**7), sp.Integer(1000)


def rapport_de_la_sonde(
    capacite_d_entree: sp.Expr, capacite_de_sonde: sp.Expr
) -> sp.Expr:
    """`V2 / V1` en domaine de Laplace ; une capacite nulle est absente."""
    elements: list[Resistance | SourceDeTension | Capacite] = [
        SourceDeTension("v1", "m", 1),
        Resistance("v1", "v2", R_SONDE),
        Resistance("v2", "m", R_ENTREE),
    ]
    if capacite_d_entree != 0:
        elements.append(Capacite("v2", "m", capacite_d_entree))
    if capacite_de_sonde != 0:
        elements.append(Capacite("v1", "v2", capacite_de_sonde))
    reseau = ReseauOperationnel(elements=tuple(elements), reference="m")
    return sp.simplify(reseau.potentiels()["v2"])


def _poles(transmittance: sp.Expr) -> list[sp.Expr]:
    """Les racines du denominateur, sur une variable libre de tout signe.

    La variable de Laplace du domaine est declaree positive, pour que les
    limites soient reelles ; un pole stable, negatif ou complexe, n'y
    serait donc jamais trouve.
    """
    libre = sp.Symbol("s")
    return list(
        sp.solve(sp.denom(sp.together(transmittance)).subs(VARIABLE, libre), libre)
    )


def capacite_de_compensation() -> sp.Expr:
    """La `C2` qui annule la derivee du rapport, cherchee sans formule."""
    c2 = sp.Symbol("C_2", positive=True)
    derivee = sp.together(sp.diff(rapport_de_la_sonde(C_ENTREE, c2), VARIABLE))
    numerateur = sp.Poly(sp.numer(derivee), VARIABLE)
    solutions = sp.solve(numerateur.coeffs(), c2, dict=True)
    if len(solutions) != 1:
        msg = f"compensation introuvable ou ambigue : {solutions}"
        raise ValueError(msg)
    return sp.nsimplify(solutions[0][c2])


def coupure_de_la_sonde() -> float:
    """La frequence du pole de la sonde non compensee, en hertz."""
    pole = _poles(rapport_de_la_sonde(C_ENTREE, 0))
    return float(abs(pole[0]) / (2 * sp.pi))


def _sortie_du_passe_bande() -> sp.Expr:
    """`V_R / V_s` du `R-L-C` serie de la figure 6."""
    reseau = ReseauOperationnel(
        elements=(
            SourceDeTension("s", "m", 1),
            Capacite("s", "a", C_RLC),
            Bobine("a", "r", L_RLC),
            Resistance("r", "m", R_RLC),
        ),
        reference="m",
    )
    return reseau.potentiels()["r"]


def resonance_du_passe_bande() -> float:
    """La frequence ou le module est maximal : `1 / (2 pi racine(LC))`.

    Elle se lit sur le **produit** des deux poles, `1/(LC)`, et non sur
    le module de l'un d'eux : avec ces valeurs le circuit est suramorti,
    ses poles sont reels, et aucun n'est a la pulsation de resonance.
    """
    premier, second = _poles(_sortie_du_passe_bande())
    return float(sp.sqrt(abs(premier * second)) / (2 * sp.pi))


def coupures_du_passe_bande() -> tuple[float, float]:
    """Les deux frequences ou le module vaut `1/racine 2`."""
    omega = sp.Symbol("omega", positive=True)
    reponse = sp.together(_sortie_du_passe_bande().subs(VARIABLE, sp.I * omega))
    numerateur, denominateur = sp.fraction(reponse)
    module_carre = sp.expand(numerateur * sp.conjugate(numerateur)) / sp.expand(
        denominateur * sp.conjugate(denominateur)
    )
    racines = sp.solve(sp.Eq(module_carre, sp.Rational(1, 2)), omega)
    basse, haute = sorted(float(r) / (2 * math.pi) for r in racines)
    return basse, haute


def precision_de_l_ohmmetre() -> float:
    """2,2 kohms sur le calibre de 10 kohms, en ohms."""
    return precision_d_une_lecture(RESISTANCE_MESUREE, 10_000.0, 0.0020, 0.0005)


def mesure_volt_amperemetrique(tension: float) -> tuple[float, float]:
    """`R = U / I` et son incertitude, calibres de 10 V et de 10 mA."""
    courant = tension / RESISTANCE_MESUREE * 1e3
    incertitude_u = precision_d_une_lecture(tension, 10.0, 0.0015, 0.0004)
    incertitude_i = precision_d_une_lecture(courant, 10.0, 0.005, 0.010)
    resistance, incertitude = precision_d_un_quotient(
        tension, incertitude_u, courant, incertitude_i
    )
    return resistance * 1e3, incertitude * 1e3
