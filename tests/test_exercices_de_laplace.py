"""Mathematique, exercices 2, seance II : les transformees de Laplace.

Source : `Mathematique/EXS/Exs_02_Equations_differentielles.pdf`, pages 9
a 13. Les solutions officielles sont les oracles. Deux d'entre elles ne
tiennent pas, et un test le montre pour chacune :

* II.3 -- le theoreme de la valeur finale est applique a
  `1/(p (p - a)^2)` avec `a > 0`, dont l'original diverge ;
* II.7.3 -- les deux exponentielles complexes de la reponse au sinus ont
  echange leurs coefficients.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electronique.domain.laplace import (
    impedance_de_bobine,
    impedance_de_condensateur,
    impedance_de_resistance,
    impedances_en_serie,
)
from polytech.maths.domain.comparaison import egales
from polytech.maths.domain.laplace.ordre_n import ProblemeLineaire
from polytech.maths.domain.laplace.proprietes import (
    d_une_fonction_periodique,
    de_la_convolution,
    du_retard,
)
from polytech.maths.domain.laplace.theoremes import (
    limite_en_zero,
    poles_genants,
    valeur_finale,
)
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T
from polytech.signaux.domain import symboles as signaux
from polytech.signaux.domain.systemes import convolution

R = sp.Rational
OMEGA = sp.Symbol("omega", positive=True)
PHI = sp.Symbol("phi", real=True)
sin, cos, exp, sqrt = sp.sin, sp.cos, sp.exp, sp.sqrt
ECHELON = sp.Heaviside


def _original(image: sp.Expr) -> sp.Expr:
    return Transformee(sp.Integer(0), image).inverse().subs(ECHELON(T), 1)


def _memes(obtenu: sp.Expr, attendu: sp.Expr) -> bool:
    return egales(obtenu, attendu)


TRANSFORMEES = {
    "1a": (
        9 * T**7 - 3 * T**5 + T**2 - 3,
        9 * sp.factorial(7) / P**8 - 3 * sp.factorial(5) / P**6 + 2 / P**3 - 3 / P,
    ),
    "1b": (
        exp(-2 * T) * (cos(T) + 2 * T**2 - 1),
        (P + 2) / ((P + 2) ** 2 + 1) + 4 / (P + 2) ** 3 - 1 / (P + 2),
    ),
    "1c": (sp.sinh(2 * T), 2 / (P**2 - 4)),
    "1d": (
        cos(OMEGA * T + PHI),
        (P * cos(PHI) - OMEGA * sin(PHI)) / (P**2 + OMEGA**2),
    ),
    "1e": (
        exp(-OMEGA * T) * sin(OMEGA * T + PHI),
        (OMEGA * (cos(PHI) + sin(PHI)) + P * sin(PHI))
        / (P**2 + 2 * OMEGA * P + 2 * OMEGA**2),
    ),
    "2f": ((T - 1) ** 3, 6 / P**4 - 6 / P**3 + 3 / P**2 - 1 / P),
}


@pytest.mark.parametrize("numero", list(TRANSFORMEES))
def test_les_transformees_de_l_exercice_ii_1(numero: str) -> None:
    original, officielle = TRANSFORMEES[numero]
    assert _memes(Transformee.de(original).image, officielle)


def test_la_fonction_retardee_n_a_pas_la_transformee_de_la_fonction_decalee() -> None:
    """II.1.2 : `(t - 1)^3 H(t)` contre `tau_1(H t^3)`, d'image `e^(-p) 3!/p^4`."""
    retardee = du_retard(Transformee.de(T**3).image, 1)
    assert retardee == exp(-P) * 6 / P**4
    assert not _memes(retardee, TRANSFORMEES["2f"][1])


def test_la_dent_de_scie_de_l_exercice_ii_1_3a() -> None:
    """`F = 1/p^2 - (e^(-p)/p) / (1 - e^(-p))`, par la formule des periodiques."""
    motif = sp.integrate(T * exp(-P * T), (T, 0, 1))
    officielle = 1 / P**2 - exp(-P) / P / (1 - exp(-P))
    assert _memes(d_une_fonction_periodique(motif, 1), officielle)


def test_le_creneau_de_l_exercice_ii_1_3b() -> None:
    """Un sur `[0, 1[`, zero sur `[1, 2[`, de periode 2 : `F = 1/(p (1 + e^(-p)))`."""
    motif = sp.integrate(exp(-P * T), (T, 0, 1))
    assert _memes(d_une_fonction_periodique(motif, 2), 1 / (P * (1 + exp(-P))))


ORIGINAUX = {
    1: (1 / (P**2 + 1), sin(T)),
    2: (1 / (P**2 + P + 1), 2 / sqrt(3) * exp(-T / 2) * sin(sqrt(3) / 2 * T)),
    3: (1 / (3 * P + 4), exp(-R(4, 3) * T) / 3),
    4: (2 * P / (3 * P**2 + 6 * P + 3), R(2, 3) * exp(-T) * (1 - T)),
    5: (P / (P**2 + 2 * P - 3), R(3, 4) * exp(-3 * T) + exp(T) / 4),
    6: (
        (P**3 + P**2 - 2 * P - 2) / (P**2 - 1) ** 2,
        exp(T) / 4 * (5 - 2 * T) - exp(-T) / 4,
    ),
    7: ((3 * P + 4) / (P**2 + 9), 3 * cos(3 * T) + R(4, 3) * sin(3 * T)),
}


@pytest.mark.parametrize("numero", list(ORIGINAUX))
def test_les_originaux_de_l_exercice_ii_2(numero: int) -> None:
    """Dans les deux sens : l'inverse rend l'original, l'original rend l'image."""
    image, officiel = ORIGINAUX[numero]
    assert _memes(_original(image), officiel)
    assert _memes(Transformee.de(officiel).image, image)


def test_la_formule_de_la_valeur_finale_rend_un_sur_a_carre() -> None:
    """II.3, solution : `lim p F(p) = 1/a^2` pour `F = 1/(p (p - a)^2)`."""
    a = sp.Symbol("a", positive=True)
    assert limite_en_zero(1 / (P * (P - a) ** 2)) == 1 / a**2


def test_mais_pour_a_positif_l_original_diverge() -> None:
    """II.3 pose `a > 0` : `f = (1 - e^(at) + a t e^(at)) / a^2` n'a pas de limite.

    Le theoreme suppose que `lim f` existe ; ici le pole double en
    `p = a > 0` fait croitre `f` sans borne, et `1/a^2` ne decrit rien.
    """
    image = 1 / (P * (P - 2) ** 2)
    original = _original(image)
    assert _memes(original, (1 - exp(2 * T) + 2 * T * exp(2 * T)) / 4)
    assert sp.limit(original, T, sp.oo) == sp.oo
    assert poles_genants(image) == (2,)
    with pytest.raises(ValueError, match="theoreme inapplicable"):
        valeur_finale(image)


def test_avec_a_negatif_le_theoreme_s_applique_et_la_limite_vaut_un_sur_a_carre() -> (
    None
):
    """Le meme calcul pour `a = -2` : `f` tend bien vers `1/4`."""
    image = 1 / (P * (P + 2) ** 2)
    assert valeur_finale(image) == R(1, 4)
    assert sp.limit(_original(image), T, sp.oo) == R(1, 4)


def test_un_oscillateur_non_amorti_n_a_pas_de_valeur_finale() -> None:
    """`F = 1/(p^2 + 1)` : `sin t` oscille, les poles `+-j` sont sur l'axe."""
    assert set(poles_genants(1 / (P**2 + 1))) == {sp.I, -sp.I}


def test_le_produit_de_convolution_de_l_exercice_ii_4() -> None:
    """`L[t * (sin t - e^t)] = (1/p^2)(1/(1 + p^2) - 1/(p - 1))`, puis l'inverse."""
    image = de_la_convolution(1 / P**2, 1 / (1 + P**2) - 1 / (P - 1))
    decomposee = 1 / P + 2 / P**2 - 1 / (1 + P**2) - 1 / (P - 1)
    assert _memes(image, decomposee)
    assert _memes(_original(image), 1 + 2 * T - sin(T) - exp(T))


def test_la_convolution_calculee_directement_donne_le_meme_resultat() -> None:
    """L'integrale `int_0^t (t - s)(sin s - e^s) ds`, sans passer par Laplace."""
    t = signaux.TEMPS
    directe = convolution(t, sin(t) - exp(t))
    assert _memes(directe.subs(t, T), 1 + 2 * T - sin(T) - exp(T))


def test_les_equations_de_l_exercice_ii_5() -> None:
    """`y'' - 5y' + 6y = 0` et `y'' + 2y' + y = 4 e^(-t)`, conditions comprises."""
    premiere = ProblemeLineaire((1, -5, 6), sp.Integer(0), (1, 1))
    assert _memes(premiere.solution(), 2 * exp(2 * T) - exp(3 * T))
    seconde = ProblemeLineaire((1, 2, 1), 4 * exp(-T), (2, -1))
    assert _memes(seconde.solution(), (2 + T + 2 * T**2) * exp(-T))
    assert premiere.solution_verifiee()
    assert seconde.solution_verifiee()


def test_le_second_membre_par_morceaux_de_l_exercice_ii_5_3() -> None:
    """`y'' + 4y = t` puis `1` des `t = 1` : `(2t - sin 2t)/8` moins sa retardee."""
    rampe = (2 * T - sin(2 * T)) / 8
    officielle = rampe - rampe.subs(T, T - 1) * ECHELON(T - 1)
    second_membre = T - (T - 1) * ECHELON(T - 1)
    for instant in (R(1, 2), R(3, 2), 3):
        derivee = sp.diff(officielle, T, 2).subs(T, instant)
        reste = (
            derivee + 4 * officielle.subs(T, instant) - second_membre.subs(T, instant)
        )
        assert egales(reste, 0)
    assert officielle.subs(T, 0) == 0
    assert sp.diff(officielle, T).subs(T, 0) == 0


def test_la_reponse_indicielle_de_l_exercice_ii_6() -> None:
    """`y'' + 4y' + 3y = H(t)` : `1/3 - e^(-t)/2 + e^(-3t)/6`."""
    probleme = ProblemeLineaire((1, 4, 3), sp.Integer(1), (0, 0))
    assert _memes(probleme.solution(), R(1, 3) - exp(-T) / 2 + exp(-3 * T) / 6)


def test_la_reponse_impulsionnelle_est_la_limite_des_reponses_aux_creneaux() -> None:
    """II.6.2-3 : `h_n` pour `t > 1/n`, puis `n -> oo` : `(e^(-t) - e^(-3t))/2`."""
    n = sp.Symbol("n", positive=True)
    indicielle = R(1, 3) - exp(-T) / 2 + exp(-3 * T) / 6
    h_n = n * (indicielle - indicielle.subs(T, T - 1 / n))
    officielle = n * (
        (-exp(-T) / 2 + exp(-3 * T) / 6)
        - (-exp(-(T - 1 / n)) / 2 + exp(-3 * (T - 1 / n)) / 6)
    )
    assert _memes(h_n, officielle)
    assert _memes(sp.limit(h_n, n, sp.oo), (exp(-T) - exp(-3 * T)) / 2)


def test_la_reponse_impulsionnelle_est_aussi_la_reponse_libre_a_une_vitesse_unite() -> (
    None
):
    """II.6.4 : `x = 0`, `y(0) = 0`, `y'(0) = 1` donne la meme fonction `h`."""
    libre = ProblemeLineaire((1, 4, 3), sp.Integer(0), (0, 1))
    assert _memes(libre.solution(), (exp(-T) - exp(-3 * T)) / 2)
    assert _memes(_original(1 / (P**2 + 4 * P + 3)), (exp(-T) - exp(-3 * T)) / 2)


def _courant(tension_en_image: sp.Expr) -> sp.Expr:
    """II.7 : `R = 3`, `L = 1`, `C = 1/2` en serie ; `I = U / Z`."""
    impedance = impedances_en_serie(
        impedance_de_resistance(sp.Integer(3), P),
        impedance_de_bobine(sp.Integer(1), P),
        impedance_de_condensateur(R(1, 2), P),
    )
    return _original(tension_en_image / impedance)


def test_l_impedance_du_circuit_de_l_exercice_ii_7() -> None:
    """`u = L i' + R i + (1/C) int i` devient `U = (p + 3 + 2/p) I`."""
    impedance = impedances_en_serie(
        impedance_de_resistance(sp.Integer(3), P),
        impedance_de_bobine(sp.Integer(1), P),
        impedance_de_condensateur(R(1, 2), P),
    )
    assert _memes(impedance, P + 3 + 2 / P)


def test_les_reponses_indicielle_et_impulsionnelle_du_circuit() -> None:
    """II.7.1 et II.7.2 : `e^(-t) - e^(-2t)`, puis sa derivee `-e^(-t) + 2 e^(-2t)`."""
    indicielle = _courant(1 / P)
    impulsionnelle = _courant(sp.Integer(1))
    assert _memes(indicielle, exp(-T) - exp(-2 * T))
    assert _memes(impulsionnelle, -exp(-T) + 2 * exp(-2 * T))
    assert _memes(sp.diff(indicielle, T), impulsionnelle)


def test_la_reponse_au_sinus_du_circuit() -> None:
    """II.7.3, `u = sin t` : `-e^(-t)/2 + 2 e^(-2t)/5 + (cos t + 3 sin t)/10`."""
    reponse = _courant(1 / (P**2 + 1))
    attendue = -exp(-T) / 2 + R(2, 5) * exp(-2 * T) + (cos(T) + 3 * sin(T)) / 10
    assert _memes(reponse, attendue)
    assert reponse.subs(T, 0) == 0
    assert sp.diff(reponse, T).subs(T, 0) == 0


def test_la_solution_officielle_de_ii_7_3_a_echange_ses_deux_exponentielles() -> None:
    """La page 13 ecrit `e^(-jt)/(2(1 + 3j)) + e^(jt)/(2(1 - 3j))`.

    Le residu en `p = j` est `1/(2(1 + 3j))` : c'est lui qui multiplie
    `e^(jt)`. Tel qu'imprime, le courant aurait une pente `-3/5` a
    l'instant initial, alors que `L i'(0) = u(0) - R i(0) - q(0)/C = 0`.
    """
    j = sp.I
    communes = -exp(-T) / 2 + R(2, 5) * exp(-2 * T)
    imprimee = (
        communes + exp(-j * T) / (2 * (1 + 3 * j)) + exp(j * T) / (2 * (1 - 3 * j))
    )
    corrigee = (
        communes + exp(j * T) / (2 * (1 + 3 * j)) + exp(-j * T) / (2 * (1 - 3 * j))
    )
    assert egales(sp.diff(imprimee, T).subs(T, 0), R(-3, 5))
    assert egales(sp.diff(corrigee, T).subs(T, 0), 0)
    assert _memes(sp.re(sp.expand_complex(corrigee)), _courant(1 / (P**2 + 1)))
