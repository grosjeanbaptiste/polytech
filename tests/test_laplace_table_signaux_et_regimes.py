"""Laplace : la table du formulaire, les signaux causaux, les regimes.

Sources :

- `Mathematique/FORMULAIRES/Formulaire_Aide_memoire.pdf`, page 10 : huit
  transformees, de `delta` a `cosh` ;
- `Syllabus_Mathematiques_appliquees_2016_2017.pdf` :
  - 3.1, page 52 : l'echelon, la fonction rendue causale, et « si
    Re(p) > 0 », « si Re(p) > alpha » sous les deux premieres images ;
  - 3.3.1 : la linearite, et l'exemple 1 du 3.4 (page 57) qui en tire
    `L(cos omega t)` ;
  - 3.4, page 57, exemples 4 a 6 : l'echelon retarde, le creneau
    `nH - n tau_(1/n) H`, le triangle `t - 2 tau_1 t + tau_2 t` ;
  - 3.6, pages 60 a 62 : reponse, regime et transitoire ;
  - 3.8.2 : la mesure de Dirac d'un ensemble ;
  - exemple 3.2, pages 70 et 71 : le circuit de transmittance
    `p / ((p + 1)(p + 2))`, ses reponses impulsionnelle et indicielle.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.laplace.regimes import (
    regime_permanent,
    regime_transitoire,
    reponse_impulsionnelle,
    reponse_indicielle,
)
from polytech.maths.domain.laplace.signaux import (
    creneau,
    echelon,
    mesure_de_dirac,
    rendre_causale,
    triangle,
)
from polytech.maths.domain.laplace.table import (
    ALPHA,
    OMEGA,
    ORDRE,
    TABLE_DE_LAPLACE,
    abscisse_de_convergence,
    image_de_la_table,
    par_linearite,
)
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T

R = sp.Rational
VALEURS = {ALPHA: 2, OMEGA: 3, ORDRE: 4}


def test_la_table_a_les_huit_lignes_du_formulaire() -> None:
    assert len(TABLE_DE_LAPLACE) == 8


@pytest.mark.parametrize("original", list(TABLE_DE_LAPLACE), ids=str)
def test_chaque_ligne_est_ce_que_l_integrale_donne(original: sp.Expr) -> None:
    calculee = Transformee.de(original.subs(VALEURS)).image
    de_la_table = image_de_la_table(original).subs(VALEURS)
    assert sp.simplify(calculee - de_la_table) == 0


def test_une_fonction_hors_de_la_table_leve() -> None:
    with pytest.raises(ValueError, match="n'est pas dans la table"):
        image_de_la_table(sp.tan(T))


def test_les_demi_plans_des_deux_premiers_exemples() -> None:
    assert abscisse_de_convergence(echelon()) == 0
    assert abscisse_de_convergence(sp.exp(ALPHA * T)) == ALPHA


@pytest.mark.parametrize(
    "original", [sp.exp(ALPHA * T), sp.cosh(ALPHA * T), sp.sinh(ALPHA * T)], ids=str
)
def test_l_integrale_converge_au_dela_de_l_abscisse_et_pas_en_deca(
    original: sp.Expr,
) -> None:
    borne = sp.Symbol("beta", positive=True)
    abscisse = abscisse_de_convergence(original).subs(VALEURS)
    for ecart, converge in ((R(1, 2), True), (-R(1, 2), False)):
        noyau = sp.exp(-(abscisse + ecart) * T) * original.subs(VALEURS)
        limite = sp.limit(sp.integrate(noyau, (T, 0, borne)), borne, sp.oo)
        assert bool(limite.is_finite) is converge


def test_la_linearite_donne_le_cosinus_par_les_exponentielles() -> None:
    # 3.4, exemple 1 : cos(omega t) = (e^(j omega t) + e^(-j omega t)) / 2.
    j = sp.I
    image = par_linearite(
        [
            (R(1, 2), sp.exp(ALPHA * T), {ALPHA: j * OMEGA}),
            (R(1, 2), sp.exp(ALPHA * T), {ALPHA: -j * OMEGA}),
        ]
    )
    assert sp.simplify(image - P / (P**2 + OMEGA**2)) == 0


def test_la_linearite_sur_une_combinaison() -> None:
    image = par_linearite([(3, echelon(), {}), (-2, T**ORDRE, {ORDRE: 2})])
    assert sp.simplify(image - (3 / P - 4 / P**3)) == 0


def test_l_echelon_et_la_fonction_rendue_causale() -> None:
    assert echelon().subs(T, 0) == 1
    assert Transformee.de(echelon()).image == 1 / P
    causale = rendre_causale(sp.cos(T))
    assert Transformee.de(causale).image == Transformee.de(sp.cos(T)).image
    assert echelon(2).subs(T, 2) == 1
    assert echelon(2).subs(T, 1) == 0
    assert sp.simplify(Transformee.de(echelon(2)).image - sp.exp(-2 * P) / P) == 0


@pytest.mark.parametrize("n", [1, 2, 5])
def test_le_creneau_de_l_exemple_5(n: int) -> None:
    attendue = n / P * (1 - sp.exp(-P / n))
    assert sp.simplify(Transformee.de(creneau(n)).image - attendue) == 0
    assert sp.integrate(creneau(n).rewrite(sp.Piecewise), (T, 0, 2)) == 1


def test_un_creneau_de_largeur_nulle_leve() -> None:
    with pytest.raises(ValueError, match="strictement positif"):
        creneau(0)


def test_le_triangle_de_l_exemple_6() -> None:
    attendue = (1 - sp.exp(-P)) ** 2 / P**2
    assert sp.simplify(Transformee.de(triangle()).image - attendue) == 0
    valeurs = [triangle().subs(T, x) for x in (0, R(1, 2), 1, R(3, 2), 2, 3)]
    assert valeurs == [0, R(1, 2), 1, R(1, 2), 0, 0]


def test_la_mesure_de_dirac_dit_si_l_ensemble_contient_zero() -> None:
    assert mesure_de_dirac(sp.Interval(-1, 1)) == 1
    assert mesure_de_dirac(sp.Interval(1, 2)) == 0
    assert mesure_de_dirac(sp.Interval.open(0, 1)) == 0


def test_la_reponse_indicielle_du_3_6_se_separe_en_deux_regimes() -> None:
    reponse = R(1, 2) + sp.exp(-2 * T) / 2 - sp.exp(-T)
    assert regime_permanent(reponse) == R(1, 2)
    assert regime_transitoire(reponse) == sp.exp(-2 * T) / 2 - sp.exp(-T)


def test_un_regime_sinusoidal_ne_s_eteint_pas() -> None:
    reponse = -sp.exp(-T) / 2 + R(2, 5) * sp.exp(-2 * T) + sp.cos(T) / 10
    assert regime_permanent(reponse) == sp.cos(T) / 10
    assert regime_permanent(sp.exp(-T) * sp.cos(T)) == 0


def test_une_reponse_qui_diverge_n_a_pas_de_regime() -> None:
    with pytest.raises(ValueError, match="diverge"):
        regime_permanent(sp.exp(T) + 1)


def test_l_exemple_3_2_les_deux_reponses_du_circuit() -> None:
    transmittance = P / ((P + 1) * (P + 2))
    impulsionnelle = reponse_impulsionnelle(transmittance)
    indicielle = reponse_indicielle(transmittance)
    assert sp.simplify(impulsionnelle - (-sp.exp(-T) + 2 * sp.exp(-2 * T))) == 0
    assert sp.simplify(indicielle - (sp.exp(-T) - sp.exp(-2 * T))) == 0
    assert sp.simplify(sp.diff(indicielle, T) - impulsionnelle) == 0
