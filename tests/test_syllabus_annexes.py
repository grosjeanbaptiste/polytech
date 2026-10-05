"""Syllabus de mathematiques appliquees, annexes A et B.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 194 a 199. Les rappels sur les suites et les series numeriques --
convergence, suite de Cauchy, les huit criteres -- et la table des
transformees de Laplace, verifiee ligne a ligne.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.laplace.convolution import (
    IMPULSION_DE_DIRAC,
    produit_de_convolution,
)
from polytech.maths.domain.laplace.proprietes import (
    de_l_integrale,
    de_la_convolution,
    de_la_derivee_n,
    du_produit_par_une_exponentielle,
    du_retard,
)
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.series.numeriques import (
    Verdict,
    converge,
    converge_absolument,
    critere_de_la_racine,
    critere_du_quotient,
    est_alternee_convergente,
    rang_suffisant,
    somme_de_la_serie_geometrique,
    somme_partielle,
    sont_equivalentes,
)
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T

N = sp.Symbol("n", integer=True, positive=True)
R = sp.Rational


def test_une_suite_converge_des_que_le_rang_est_suffisant() -> None:
    """A.1 : `|u_n - u| < eps` des que `n > N(eps)`. Pour `1/n`, `N(1/100) = 100`."""
    assert rang_suffisant(1 / N, N, limite=0, epsilon=R(1, 100)) == 100
    assert rang_suffisant((N + 1) / N, N, limite=1, epsilon=R(1, 10)) == 10
    assert rang_suffisant(R(1, 2) ** N, N, limite=0, epsilon=R(1, 1000)) == 9


def test_une_suite_qui_ne_s_approche_pas_de_sa_limite_supposee() -> None:
    """`(-1)^n` ne s'approche de rien : aucun rang ne convient."""
    with pytest.raises(ValueError, match="aucun rang"):
        rang_suffisant((-1) ** N, N, limite=0, epsilon=R(1, 2))


def test_les_sommes_partielles_de_la_serie_oscillante() -> None:
    """A.2 : `somme (-1)^n` vaut -1 aux rangs impairs, 0 aux pairs ; elle diverge."""
    assert [somme_partielle((-1) ** N, N, rang) for rang in (1, 2, 3, 4)] == [
        -1,
        0,
        -1,
        0,
    ]
    assert not converge((-1) ** N, N)


def test_le_terme_general_d_une_serie_convergente_tend_vers_zero() -> None:
    """Critere 2 ; la reciproque est fausse : `1/n` tend vers 0 et sa serie diverge."""
    assert converge(1 / N**2, N)
    assert sp.limit(1 / N**2, N, sp.oo) == 0
    assert sp.limit(1 / N, N, sp.oo) == 0
    assert not converge(1 / N, N)


def test_la_convergence_absolue_entraine_la_convergence() -> None:
    """Critere 3 ; la serie harmonique alternee converge sans converger absolument."""
    assert converge_absolument((-1) ** N / N**2, N)
    assert converge((-1) ** N / N**2, N)
    assert converge((-1) ** N / N, N)
    assert not converge_absolument((-1) ** N / N, N)


@pytest.mark.parametrize(
    ("exposant", "attendu"), [(2, True), (R(3, 2), True), (1, False), (R(1, 2), False)]
)
def test_les_series_de_riemann(exposant: sp.Expr, *, attendu: bool) -> None:
    """Series tests : `somme 1/n^alpha` converge ssi `alpha > 1`."""
    assert converge(1 / N**exposant, N) is attendu


def test_la_serie_geometrique() -> None:
    """Serie test : `somme x^n = 1/(1 - x)` ssi `|x| < 1`."""
    assert somme_de_la_serie_geometrique(R(1, 2)) == 2
    assert somme_de_la_serie_geometrique(-R(1, 3)) == R(3, 4)
    partielle = somme_partielle(R(1, 2) ** N, N, 30) + 1
    assert abs(partielle - 2) < R(1, 10**8)


@pytest.mark.parametrize("raison", [1, -1, 2])
def test_une_serie_geometrique_de_raison_trop_grande_n_a_pas_de_somme(
    raison: int,
) -> None:
    with pytest.raises(ValueError, match="raison de module"):
        somme_de_la_serie_geometrique(raison)


def test_le_critere_de_comparaison() -> None:
    """Critere 5 : `0 <= 1/(n^2 + 1) <= 1/n^2`, et la majorante converge."""
    assert sp.simplify(1 / N**2 - 1 / (N**2 + 1)).is_nonnegative
    assert converge(1 / N**2, N)
    assert converge(1 / (N**2 + 1), N)


def test_le_critere_d_equivalence() -> None:
    """Critere 6 : `u_n / v_n` a une limite finie non nulle ; meme nature."""
    assert sont_equivalentes(1 / (N**2 + N), 1 / N**2, N)
    assert converge(1 / (N**2 + N), N)
    assert sont_equivalentes((N + 1) / N**2, 1 / N, N)
    assert not converge((N + 1) / N**2, N)
    assert not sont_equivalentes(1 / N**2, 1 / N, N)
    assert not sont_equivalentes(1 / N, 1 / N**2, N)


@pytest.mark.parametrize(
    ("terme", "limite", "verdict"),
    [
        (R(1, 2) ** N, R(1, 2), Verdict.CONVERGE),
        (N / 3**N, R(1, 3), Verdict.CONVERGE),
        (2**N / N, 2, Verdict.DIVERGE),
        (1 / N**2, 1, Verdict.INDECIS),
        (1 / N, 1, Verdict.INDECIS),
        ((-1) ** N / sp.factorial(N), 0, Verdict.CONVERGE),
    ],
)
def test_le_critere_du_quotient(
    terme: sp.Expr, limite: sp.Expr, verdict: Verdict
) -> None:
    """Critere 7 a : `A < 1` converge, `A > 1` diverge, `A = 1` ne conclut pas."""
    assert critere_du_quotient(terme, N) == (limite, verdict)


@pytest.mark.parametrize(
    ("terme", "limite", "verdict"),
    [
        (R(1, 2) ** N, R(1, 2), Verdict.CONVERGE),
        ((N / (2 * N + 1)) ** N, R(1, 2), Verdict.CONVERGE),
        (3**N / N**2, 3, Verdict.DIVERGE),
        (1 / N**2, 1, Verdict.INDECIS),
    ],
)
def test_le_critere_de_la_racine(
    terme: sp.Expr, limite: sp.Expr, verdict: Verdict
) -> None:
    """Critere 7 b : meme conclusion sur `lim |u_n|^(1/n)`."""
    assert critere_de_la_racine(terme, N) == (limite, verdict)


def test_les_deux_criteres_laissent_indecises_deux_series_de_nature_opposee() -> None:
    """« Tout est possible » : `1/n^2` converge, `1/n` diverge, `A = 1` aux deux."""
    assert critere_du_quotient(1 / N**2, N)[1] is Verdict.INDECIS
    assert critere_du_quotient(1 / N, N)[1] is Verdict.INDECIS
    assert converge(1 / N**2, N)
    assert not converge(1 / N, N)


def test_les_series_alternees() -> None:
    """Critere 8 : `somme (-1)^n a_n` converge si `a_n` tend vers 0 en decroissant."""
    assert est_alternee_convergente(1 / N, N)
    assert est_alternee_convergente(1 / sp.sqrt(N), N)
    assert not est_alternee_convergente(N / (N + 1), N)
    assert not est_alternee_convergente(sp.Integer(1), N)


def test_la_table_des_fonctions_usuelles() -> None:
    """B.2 : `H`, `e^(at)`, `t^n`, `cos`, `sin`, `cosh`, `sinh`."""
    alpha, omega = sp.symbols("alpha omega", positive=True)
    lignes = {
        sp.Integer(1): 1 / P,
        T**3: sp.factorial(3) / P**4,
        sp.cos(omega * T): P / (P**2 + omega**2),
        sp.sin(omega * T): omega / (P**2 + omega**2),
    }
    for original, image in lignes.items():
        assert sp.simplify(Transformee.de(original).image - image) == 0
    hyperboliques = {
        sp.exp(2 * T): 1 / (P - 2),
        sp.cosh(2 * T): P / (P**2 - 4),
        sp.sinh(2 * T): 2 / (P**2 - 4),
    }
    for original, image in hyperboliques.items():
        calculee = sp.laplace_transform(original, T, P, noconds=True)
        assert sp.simplify(calculee - image) == 0
    assert alpha.is_positive


def test_la_puissance_de_t_de_la_table() -> None:
    """`L(t^n) = n! / p^(n+1)` -- le formulaire, tel qu'il se lit, ecrit « p^n + 1 »."""
    for n in range(5):
        attendue = sp.factorial(n) / P ** (n + 1)
        assert sp.simplify(Transformee.de(T**n).image - attendue) == 0
        assert (
            sp.simplify(Transformee.de(T**n).image - sp.factorial(n) / (P**n + 1)) != 0
        )


def test_la_table_des_proprietes() -> None:
    """B.2 : `e^(at) f`, le retard, la derivee, l'integrale, la convolution."""
    original = sp.sin(T)
    image = Transformee.de(original).image
    amortie = Transformee.de(sp.exp(-2 * T) * original).image
    assert sp.simplify(du_produit_par_une_exponentielle(image, -2) - amortie) == 0
    assert du_retard(image, 3) == sp.exp(-3 * P) * image
    derivee = Transformee.de(sp.diff(original, T, 2)).image
    assert sp.simplify(de_la_derivee_n(image, (0, 1)) - derivee) == 0
    primitive = sp.integrate(original.subs(T, sp.Symbol("u")), (sp.Symbol("u"), 0, T))
    assert sp.simplify(de_l_integrale(image) - Transformee.de(primitive).image) == 0
    autre = sp.exp(-T)
    produit = produit_de_convolution(original, autre)
    attendue = de_la_convolution(image, Transformee.de(autre).image)
    assert sp.simplify(Transformee.de(produit).image - attendue) == 0


def test_l_image_de_l_impulsion_de_dirac_est_un() -> None:
    """B.2, premiere ligne : `L(delta) = 1`."""
    assert Transformee.de(IMPULSION_DE_DIRAC).image == 1


def test_les_criteres_portent_sur_les_valeurs_absolues() -> None:
    """« Ne pas oublier les valeurs absolues ! » : `(-2)^n` diverge."""
    assert critere_du_quotient((-2) ** N, N) == (2, Verdict.DIVERGE)
    assert critere_du_quotient((-R(1, 2)) ** N, N) == (R(1, 2), Verdict.CONVERGE)
    assert critere_de_la_racine((-2) ** N, N) == (2, Verdict.DIVERGE)
    assert critere_de_la_racine((-R(1, 2)) ** N, N) == (R(1, 2), Verdict.CONVERGE)


def test_un_module_qui_ne_decroit_pas_des_le_premier_rang_n_est_pas_retenu() -> None:
    """`n e^(-n/5)` tend vers 0 mais croit jusqu'a `n = 5`.

    Le critere, tel qu'il est enonce, ne s'applique pas depuis le rang 1.
    """
    module = N * sp.exp(-N / 5)
    assert sp.limit(module, N, sp.oo) == 0
    assert module.subs(N, 5) > module.subs(N, 1)
    assert not est_alternee_convergente(module, N)
