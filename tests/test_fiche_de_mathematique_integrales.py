"""Les articles d'integration de la fiche de « Mathematique », refaits.

Parties, fractions simples, changement de variable : la formule, la
raison, le piege et chaque cas d'examen de `cursus/fiches/Mathematique.toml`
sont recalcules par `polytech.maths.domain.integration`, puis compares
par `polytech.maths.domain.comparaison`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.comparaison import egales
from polytech.maths.domain.integration.definie import IntegraleDefinie
from polytech.maths.domain.integration.fractions import FractionRationnelle
from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.integration.primitives import Primitive
from polytech.maths.domain.integration.regles_de_derivation import du_produit
from polytech.maths.domain.integration.substitution import NOUVELLE, Substitution
from polytech.maths.domain.integration.trigonometrie import (
    par_la_tangente_de_l_arc_moitie,
)
from polytech.maths.domain.symboles import VARIABLE as X


def test_les_parties_sont_la_derivee_du_produit_integree() -> None:
    u, dv = sp.log(X), X**2
    parties = IntegrationParParties(u, dv)
    assert egales(
        du_produit(u, parties.v()), parties.derivee_de_u() * parties.v() + u * dv
    )
    restante = sp.integrate(parties.v() * parties.derivee_de_u(), X)
    assert egales(parties.resultat(), u * parties.v() - restante)


def test_la_recurrence_de_janvier_2021_4() -> None:
    def i(n: int) -> sp.Expr:
        return IntegraleDefinie(
            1 / (1 + X**2) ** n, sp.Integer(0), sp.Integer(1)
        ).valeur()

    for n in (1, 2, 3, 4):
        assert egales(2 * n * i(n + 1), (2 * n - 1) * i(n) + sp.Rational(1, 2**n))


def test_janvier_2022_1_l_integrale_vaut_ln_de_8_racine_3_sur_9() -> None:
    valeur = IntegraleDefinie(
        sp.log(1 + X) / X**2, sp.Integer(1), sp.Integer(2)
    ).valeur()
    assert egales(valeur, sp.log(8 * sp.sqrt(3) / 9))
    parties = IntegrationParParties(sp.log(1 + X), 1 / X**2)
    assert egales(parties.v(), -1 / X)
    fraction = FractionRationnelle(sp.Integer(1), X * (1 + X))
    assert egales(sum(fraction.termes()), 1 / X - 1 / (1 + X))


def test_x_carre_moins_4_se_decompose_en_un_quart_et_moins_un_quart() -> None:
    fraction = FractionRationnelle(sp.Integer(1), X**2 - 4)
    assert egales(sum(fraction.termes()), 1 / (4 * (X - 2)) - 1 / (4 * (X + 2)))
    a, b = sp.Rational(1, 4), -sp.Rational(1, 4)
    assert egales(a * (X + 2) + b * (X - 2), 1)
    assert egales(sp.diff(Primitive.de(1 / (X - 2)).expression, X), 1 / (X - 2))


def test_janvier_2025_1_le_facteur_sans_racine_donne_un_arctan() -> None:
    fraction = FractionRationnelle(4 * X**2, X**4 - 1)
    assert egales(sp.factor(X**4 - 1), (X - 1) * (X + 1) * (X**2 + 1))
    assert egales(sum(fraction.termes()), 2 / (X**2 + 1) - 1 / (X + 1) + 1 / (X - 1))
    assert egales(sp.diff(2 * sp.atan(X), X), 2 / (X**2 + 1))
    attendue = sp.log(sp.Rational(3, 2)) + 2 * sp.atan(3) - 2 * sp.atan(2)
    assert egales(
        IntegraleDefinie(4 * X**2 / (X**4 - 1), sp.Integer(2), sp.Integer(3)).valeur(),
        attendue,
    )


def test_une_fraction_impropre_se_divise_avant_de_se_decomposer() -> None:
    impropre = FractionRationnelle(X**2, X**2 - 1)
    assert not impropre.est_propre()
    assert egales(impropre.partie_entiere(), 1)
    with pytest.raises(ValueError, match="impropre"):
        impropre.termes()


def test_janvier_2021_1_j_la_fraction_est_deja_propre() -> None:
    fraction = FractionRationnelle(1 + 2 * X - 3 * X**2, X - X**3)
    assert fraction.est_propre()
    assert egales(sp.factor(X - X**3), X * (1 - X) * (1 + X))


def test_changement_de_variable_ln_x_donne_xi_sur_1_moins_xi_carre() -> None:
    integrande = sp.log(X) / (X * (1 - sp.log(X) ** 2))
    substitution = Substitution(integrande, sp.log(X))
    assert egales(substitution.differentielle(), 1 / X)
    assert egales(substitution.integrale_transformee(), NOUVELLE / (1 - NOUVELLE**2))


def test_janvier_2021_1_i_un_plus_cosinus_vaut_deux_cosinus_carre_du_demi() -> None:
    assert egales(1 + sp.cos(X), 2 * sp.cos(X / 2) ** 2)
    parties = IntegrationParParties(X, 1 / (2 * sp.cos(X / 2) ** 2))
    assert egales(parties.v(), sp.tan(X / 2))


def test_la_tangente_de_l_arc_moitie_saute_de_pi_en_pi() -> None:
    en_t = par_la_tangente_de_l_arc_moitie(1 / (sp.cos(X) + 2 * sp.sin(X) + 3))
    t = next(iter(en_t.free_symbols))
    assert egales(en_t, 1 / ((t + 1) ** 2 + 1))
    primitive = sp.atan(sp.tan(X / 2) + 1)
    saut = sp.limit(primitive, X, sp.pi, "-") - sp.limit(primitive, X, sp.pi, "+")
    assert egales(saut, sp.pi)
