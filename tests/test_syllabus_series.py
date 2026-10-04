"""Syllabus de mathematiques appliquees, chapitre 4 : les series de fonctions.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 77 a 86. Le rayon de convergence par la racine et par le quotient
(exemples 4.1), les proprietes des series entieres (theoremes 4.3 a 4.5),
l'equation d'Airy resolue par une serie (4.2), la convergence uniforme et
le critere de la serie majorante (exemples 4.2).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.series.entieres import INDICE as N
from polytech.maths.domain.series.entieres import SerieEntiere
from polytech.maths.domain.series.equations import (
    INCONNUES,
    solution_en_serie,
)
from polytech.maths.domain.series.mac_laurin import (
    coefficients_de_mac_laurin,
    reste_de_lagrange,
)
from polytech.maths.domain.series.uniforme import (
    distance_uniforme,
    est_majoree_par,
    serie_majorante_convergente,
)
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational
fact = sp.factorial
GEOMETRIQUE = SerieEntiere(sp.Integer(1))
EXPONENTIELLE = SerieEntiere(1 / fact(N))


@pytest.mark.parametrize(
    ("terme", "rayon"),
    [
        (sp.Integer(1), 1),
        (1 / (N + 1), 1),
        (1 / fact(N), sp.oo),
        (fact(N), 0),
        (1 / ((N + 1) * (N + 2) * (N + 3)), 1),
        (3**N / (N + 1) ** 2, R(1, 3)),
    ],
)
def test_le_rayon_de_convergence_des_exemples_4_1(
    terme: sp.Expr, rayon: object
) -> None:
    """(4.4) et remarque 4.2 : la racine et le quotient donnent le meme rayon."""
    serie = SerieEntiere(terme)
    assert serie.rayon_par_le_quotient() == rayon
    assert serie.rayon_par_la_racine() == rayon


@pytest.mark.parametrize(
    ("terme", "domaine"),
    [
        (sp.Integer(1), sp.Interval.open(-1, 1)),
        (1 / (N + 1), sp.Interval.Ropen(-1, 1)),
        (1 / fact(N), sp.S.Reals),
        (fact(N), sp.FiniteSet(0)),
        (1 / ((N + 1) * (N + 2) * (N + 3)), sp.Interval(-1, 1)),
    ],
)
def test_le_domaine_de_convergence_des_exemples_4_1(
    terme: sp.Expr, domaine: sp.Set
) -> None:
    """« Converge ssi » : au bord `|x| = R` on ne peut rien conclure sans regarder."""
    assert SerieEntiere(terme).domaine_de_convergence() == domaine


@pytest.mark.parametrize(
    ("terme", "domaine"),
    [
        ((-1) ** N / (N + 1), sp.Interval.Lopen(-1, 1)),
        ((-1) ** N / ((N + 1) * 2**N), sp.Interval.Lopen(-2, 2)),
        (3**N / (N + 1) ** 2, sp.Interval(-R(1, 3), R(1, 3))),
    ],
)
def test_les_trois_series_laissees_a_titre_d_exercice(
    terme: sp.Expr, domaine: sp.Set
) -> None:
    """Page 79 : convergence ssi `x` est dans `]-1, 1]`, `]-2, 2]`, `[-1/3, 1/3]`."""
    assert SerieEntiere(terme).domaine_de_convergence() == domaine


@pytest.mark.parametrize(
    ("terme", "somme"),
    [
        (sp.Integer(1), 1 / (1 - X)),
        (1 / fact(N), sp.exp(X)),
        ((1 + (-1) ** N) / (2 * fact(N)), sp.cosh(X)),
        ((1 - (-1) ** N) / (2 * fact(N)), sp.sinh(X)),
    ],
)
def test_les_sommes_des_exemples_4_1(terme: sp.Expr, somme: sp.Expr) -> None:
    """La somme partielle est le debut du developpement de la somme annoncee."""
    partielle = SerieEntiere(terme).somme_partielle(X, 7)
    assert sp.expand(partielle - sp.series(somme, X, 0, 8).removeO()) == 0


def test_une_serie_entiere_s_integre_terme_a_terme() -> None:
    """Theoreme 4.3 : `somme a_n x^(n+1) / (n + 1)` ; ici `-ln(1 - x)`."""
    primitive = GEOMETRIQUE.primitive()
    assert primitive.coefficient(0) == 0
    assert [primitive.coefficient(n) for n in (1, 2, 5)] == [1, R(1, 2), R(1, 5)]
    attendu = sp.series(-sp.log(1 - X), X, 0, 7).removeO()
    assert sp.expand(primitive.somme_partielle(X, 6) - attendu) == 0
    assert primitive.rayon_par_le_quotient() == 1


def test_une_serie_entiere_se_derive_terme_a_terme() -> None:
    """Theoreme 4.4 : `S' = somme (n + 1) a_(n+1) x^n`, de meme rayon."""
    derivee = GEOMETRIQUE.derivee()
    assert [derivee.coefficient(n) for n in range(4)] == [1, 2, 3, 4]
    attendu = sp.series(1 / (1 - X) ** 2, X, 0, 6).removeO()
    assert sp.expand(derivee.somme_partielle(X, 5) - attendu) == 0
    assert derivee.rayon_par_le_quotient() == GEOMETRIQUE.rayon_par_le_quotient()


def test_l_exponentielle_est_sa_propre_derivee() -> None:
    assert sp.simplify(EXPONENTIELLE.derivee().terme - EXPONENTIELLE.terme) == 0


def test_deriver_puis_primitiver_rend_la_serie_a_son_terme_constant_pres() -> None:
    serie = SerieEntiere(1 / (N + 1))
    retour = serie.derivee().primitive()
    assert retour.coefficient(0) == 0
    assert [retour.coefficient(n) for n in (1, 2, 3)] == [R(1, 2), R(1, 3), R(1, 4)]


def test_les_coefficients_sont_ceux_de_mac_laurin() -> None:
    """Theoreme 4.5 : `a_n = S^(n)(0) / n!`."""
    assert coefficients_de_mac_laurin(1 / (1 - X), 5) == (1, 1, 1, 1, 1)
    attendus = tuple(EXPONENTIELLE.coefficient(n) for n in range(5))
    assert coefficients_de_mac_laurin(sp.exp(X), 5) == attendus
    assert coefficients_de_mac_laurin(sp.sin(X), 6) == (0, 1, 0, -R(1, 6), 0, R(1, 120))


def test_le_reste_de_lagrange() -> None:
    """Remarque 4.3 : `f^(n+1)(theta x) x^(n+1) / (n + 1)!`, ici de limite nulle."""
    theta = sp.Symbol("theta", positive=True)
    reste = reste_de_lagrange(sp.exp(X), 3, theta)
    assert reste == sp.exp(theta * X) * X**4 / 24
    lointain = reste_de_lagrange(sp.exp(X), 40, theta).subs({X: 2, theta: R(1, 2)})
    assert abs(float(lointain)) < 1e-30


def test_une_fonction_peut_differer_de_sa_serie_de_mac_laurin() -> None:
    """« Ce n'est pas toujours le cas » : `e^(-1/x^2)` a tous ses coefficients nuls.

    L'exemple n'est pas dans le syllabus, qui s'en tient a l'avertissement ;
    c'est le contre-exemple classique.
    """
    plate = sp.exp(-1 / X**2)
    derivees = [sp.limit(sp.diff(plate, X, k), X, 0) for k in range(4)]
    assert derivees == [0, 0, 0, 0]
    assert plate.subs(X, 1) != 0


def test_les_coefficients_de_l_equation_d_airy() -> None:
    """4.2 : `y'' - x y = 0` donne `a_2 = 0` et `a_(n+3) = a_n / ((n + 2)(n + 3))`."""
    a0, a1 = INCONNUES
    solution = solution_en_serie((-X, 0, 1), 10)
    coefficients = [solution.coeff(X, n) for n in range(11)]
    assert coefficients[2] == coefficients[5] == coefficients[8] == 0
    assert coefficients[3] == a0 / (2 * 3)
    assert coefficients[6] == a0 / (2 * 3 * 5 * 6)
    assert coefficients[9] == a0 / (2 * 3 * 5 * 6 * 8 * 9)
    assert coefficients[4] == a1 / (3 * 4)
    assert coefficients[7] == a1 / (3 * 4 * 6 * 7)
    assert coefficients[10] == a1 / (3 * 4 * 6 * 7 * 9 * 10)
    for n in range(8):
        assert coefficients[n + 3] == coefficients[n] / ((n + 2) * (n + 3))


def test_la_serie_d_airy_est_bien_la_fonction_d_airy() -> None:
    """Second chemin : `a_0 = Ai(0)` et `a_1 = Ai'(0)` redonnent la serie de `Ai`."""
    a0, a1 = INCONNUES
    valeurs = {a0: sp.airyai(0), a1: sp.airyaiprime(0)}
    serie = solution_en_serie((-X, 0, 1), 7).subs(valeurs)
    reference = sp.series(sp.airyai(X), X, 0, 8).removeO()
    for point in (R(1, 5), R(-1, 3)):
        assert float(serie.subs(X, point)) == pytest.approx(
            float(reference.subs(X, point))
        )


def test_la_methode_des_series_retrouve_l_exponentielle() -> None:
    """`y' - y = 0` : `a_n = a_0 / n!`."""
    a0, _ = INCONNUES
    solution = solution_en_serie((-1, 1), 5)
    assert sp.expand(solution - a0 * EXPONENTIELLE.somme_partielle(X, 5)) == 0


def test_la_methode_des_series_retrouve_le_cosinus_et_le_sinus() -> None:
    """`y'' + y = 0` : `a_0 cos x + a_1 sin x`."""
    a0, a1 = INCONNUES
    attendu = sp.series(a0 * sp.cos(X) + a1 * sp.sin(X), X, 0, 8).removeO()
    assert sp.expand(solution_en_serie((1, 0, 1), 7) - attendu) == 0


def test_un_point_singulier_est_refuse() -> None:
    """Page 82 : le coefficient de la plus haute derivee ne doit pas s'annuler."""
    with pytest.raises(ValueError, match="point singulier"):
        solution_en_serie((1, 0, X), 5)


def test_la_serie_des_cosinus_sur_n_cube_converge_uniformement() -> None:
    """Exemples 4.2 : `|cos(nx) / n^3| <= 1/n^3`, serie de Riemann d'exposant 3."""
    n = sp.Symbol("n", integer=True, positive=True)
    assert serie_majorante_convergente(1 / n**3, n)
    for rang in (1, 2, 5):
        terme = sp.cos(rang * X) / rang**3
        assert est_majoree_par(terme, R(1, rang**3), X, sp.Interval(0, 2 * sp.pi))


def test_une_majorante_divergente_ne_prouve_rien() -> None:
    """La serie harmonique diverge : `cos(nx)/n` n'est pas justiciable du critere."""
    n = sp.Symbol("n", integer=True, positive=True)
    assert not serie_majorante_convergente(1 / n, n)


def test_un_terme_qui_depasse_sa_majorante() -> None:
    assert not est_majoree_par(2 * sp.cos(X), 1, X, sp.Interval(0, sp.pi))


def test_l_integrale_de_la_serie_des_cosinus_se_calcule_terme_a_terme() -> None:
    """Exemples 4.2 : `int_0^(pi/2) cos(nx)/n^3 dx = sin(n pi/2) / n^4`."""
    for rang in range(1, 7):
        integrale = sp.integrate(sp.cos(rang * X) / rang**3, (X, 0, sp.pi / 2))
        assert integrale == sp.sin(rang * sp.pi / 2) / rang**4


def test_une_serie_qui_converge_simplement_mais_pas_uniformement() -> None:
    """Exemples 4.2 : `somme (x^(n-1) - x^n)` sur `[0, 1]`, de sommes `1 - x^n`.

    La limite vaut 1 sur `[0, 1[` et 0 en 1 ; l'ecart `x^n` garde 1 pour borne
    superieure quel que soit `n`.
    """
    for rang in (1, 5, 50):
        partielle = sum(X ** (k - 1) - X**k for k in range(1, rang + 1))
        assert sp.expand(partielle - (1 - X**rang)) == 0
        assert sp.limit(1 - X**rang, X, 1) == 0
        assert distance_uniforme(1 - X**rang, 1, X, sp.Interval.Ropen(0, 1)) == 1


def test_la_meme_serie_converge_uniformement_loin_de_un() -> None:
    """Sur `[0, 1/2]`, la distance vaut `2^(-n)` et tend vers 0."""
    for rang in (1, 5, 10):
        distance = distance_uniforme(1 - X**rang, 1, X, sp.Interval(0, R(1, 2)))
        assert distance == R(1, 2**rang)


def test_la_primitive_de_la_geometrique_converge_en_moins_un() -> None:
    """`somme x^n / n`, depuis `n = 1` : la serie harmonique alternee converge."""
    assert GEOMETRIQUE.primitive().domaine_de_convergence() == sp.Interval.Ropen(-1, 1)


def test_deriver_la_primitive_rend_la_serie_des_son_premier_terme() -> None:
    """La primitive commence au rang 1 ; sa derivee revient au rang 0."""
    retour = GEOMETRIQUE.primitive().derivee()
    assert [retour.coefficient(n) for n in range(3)] == [1, 1, 1]


def test_la_distance_uniforme_retient_le_plus_grand_ecart_des_deux_cotes() -> None:
    """`x^3` sur `[0, 2]` s'ecarte de 0 par le haut ; `-x^3`, par le bas."""
    intervalle = sp.Interval(0, 2)
    assert distance_uniforme(X**3, 0, X, intervalle) == 8
    assert distance_uniforme(-(X**3), 0, X, intervalle) == 8
