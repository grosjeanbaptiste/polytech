"""Syllabus de mathematiques appliquees, sections 8.2 a 8.5 : l'espace `L^2`.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 137 a 147. Les mesures des exemples 8.1, le produit scalaire
hermitien et sa norme, l'inegalite de Cauchy-Schwarz, la famille
orthonormee `e^(j n w t)` (exemple 8.2), la meilleure approximation et
l'inegalite de Bessel (8.3), les fonctions propres d'un systeme lineaire
(8.5.2).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.operateurs import D, OperateurDifferentiel
from polytech.maths.domain.fourier.approximation import (
    carre_de_la_distance,
    coefficients_de_fourier,
    est_orthonormee,
    meilleure_approximation,
    orthonormaliser,
)
from polytech.maths.domain.fourier.espace_l2 import EspaceL2
from polytech.maths.domain.fourier.fonctions_propres import (
    harmonique,
    reponse_a_un_harmonique,
    reponse_periodique,
)
from polytech.maths.domain.symboles import TEMPS as T
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational
J = sp.I
UNITE = EspaceL2.normalise(X, 0, 1)
SYMETRIQUE = EspaceL2(X, sp.Interval(-1, 1))


def test_la_mesure_de_lebesgue_sur_un_intervalle() -> None:
    """Exemples 8.1 : `int f dmu_1 = int_a^b f(x) dx`."""
    espace = EspaceL2(X, sp.Interval(0, 2))
    assert espace.integrale(X) == 2
    assert espace.integrale(sp.Integer(1)) == 2


def test_la_mesure_normalisee_donne_la_masse_un_a_l_intervalle() -> None:
    """Exemples 8.1 : `dmu_2 = dx / (b - a)`, « la mesure de `[a, b]` vaut 1 »."""
    espace = EspaceL2.normalise(X, 0, 2)
    assert espace.integrale(sp.Integer(1)) == 1
    assert espace.integrale(X) == 1


def test_la_densite_de_gauss_est_une_mesure_de_masse_un() -> None:
    """Exemples 8.1 : `dmu_3 = e^(-x^2/2) / racine(2 pi) dx` sur la droite."""
    gauss = EspaceL2(X, sp.S.Reals, sp.exp(-(X**2) / 2) / sp.sqrt(2 * sp.pi))
    assert gauss.integrale(sp.Integer(1)) == 1
    assert gauss.norme(X) == 1
    assert gauss.sont_orthogonales(X, sp.Integer(1))


def test_une_densite_negative_n_est_pas_une_mesure() -> None:
    """Page 137 : « une fonction `p(x)` positive »."""
    with pytest.raises(ValueError, match="densite negative"):
        EspaceL2(X, sp.Interval(0, 1), -X)


def test_la_norme_generalise_la_norme_euclidienne() -> None:
    """Definition 8.1 : `||f|| = racine(int |f|^2)`."""
    assert EspaceL2(X, sp.Interval(0, 1)).norme(X) == 1 / sp.sqrt(3)
    assert UNITE.norme(sp.exp(2 * sp.pi * J * X)) == 1


def test_une_fonction_qui_n_est_pas_de_carre_integrable() -> None:
    """`1/x` sur `]0, 1]` : `int 1/x^2` diverge."""
    with pytest.raises(ValueError, match="pas de carre integrable"):
        EspaceL2(X, sp.Interval(0, 1)).norme(1 / X)


def test_le_produit_scalaire_est_hermitien() -> None:
    """Definition 8.3 : `<g, f> = conj(<f, g>)` et `<f, a g> = conj(a) <f, g>`."""
    f, g = X + J * X**2, 1 - J * X
    direct = UNITE.produit_scalaire(f, g)
    assert UNITE.produit_scalaire(g, f) == sp.conjugate(direct)
    alpha = 2 + 3 * J
    assert sp.expand(UNITE.produit_scalaire(alpha * f, g) - alpha * direct) == 0
    assert (
        sp.expand(UNITE.produit_scalaire(f, alpha * g) - sp.conjugate(alpha) * direct)
        == 0
    )
    somme = UNITE.produit_scalaire(f, g + X)
    assert sp.expand(somme - direct - UNITE.produit_scalaire(f, X)) == 0


def test_l_inegalite_de_cauchy_schwarz() -> None:
    """Lemme 8.1 : `|<f, g>| <= ||f|| ||g||`, egalite pour des fonctions colineaires."""
    f, g = X, sp.exp(X)
    assert sp.Abs(UNITE.produit_scalaire(f, g)) < UNITE.norme(f) * UNITE.norme(g)
    colineaires = sp.Abs(UNITE.produit_scalaire(f, 3 * f))
    assert sp.simplify(colineaires - UNITE.norme(f) * UNITE.norme(3 * f)) == 0


def test_l_inegalite_triangulaire() -> None:
    """Lemme 8.2 : `||f + g|| <= ||f|| + ||g||` ; lemme 8.3 : `||a f|| = |a| ||f||`."""
    f, g = X, 1 - X**2
    assert UNITE.norme(f + g) <= UNITE.norme(f) + UNITE.norme(g)
    assert sp.simplify(UNITE.norme((3 - 4 * J) * f) - 5 * UNITE.norme(f)) == 0
    assert UNITE.distance(f, g) <= UNITE.distance(f, X**3) + UNITE.distance(X**3, g)
    assert UNITE.distance(f, g) == UNITE.distance(g, f)


def test_la_famille_des_exponentielles_est_orthonormee() -> None:
    """Exemple 8.2 : `<phi_i, phi_k> = delta_ik` pour `phi_n = e^(j n w t)`."""
    periode = sp.Symbol("T", positive=True)
    espace = EspaceL2.normalise(T, 0, periode)
    omega = 2 * sp.pi / periode
    famille = [harmonique(n, omega) for n in (-2, -1, 0, 1, 2)]
    assert est_orthonormee(famille, espace)
    assert espace.produit_scalaire(famille[0], famille[3]) == 0
    assert espace.produit_scalaire(famille[3], famille[3]) == 1


def test_une_famille_qui_n_est_pas_orthonormee() -> None:
    """`1` et `x` sur `[0, 1]` : leur produit scalaire vaut `1/2`."""
    assert not est_orthonormee([sp.Integer(1), X], UNITE)
    assert not est_orthonormee([2 * sp.Integer(1)], UNITE)


def test_les_polynomes_orthogonaux_par_orthonormalisation() -> None:
    """8.5.1 : `1, x, x^2` orthonormalises sur `[-1, 1]` : les polynomes de Legendre."""
    famille = orthonormaliser([sp.Integer(1), X, X**2], SYMETRIQUE)
    assert est_orthonormee(famille, SYMETRIQUE)
    legendre = [sp.legendre(n, X) for n in range(3)]
    for obtenu, reference in zip(famille, legendre, strict=True):
        norme = SYMETRIQUE.norme(reference)
        assert sp.simplify(obtenu - reference / norme) == 0


def test_les_coefficients_de_fourier_sont_les_projections() -> None:
    """8.3 : `c_k = <f, phi_k>`."""
    famille = orthonormaliser([sp.Integer(1), X], SYMETRIQUE)
    coefficients = coefficients_de_fourier(X**3, famille, SYMETRIQUE)
    assert coefficients[0] == 0
    assert (
        sp.simplify(coefficients[1] - SYMETRIQUE.produit_scalaire(X**3, famille[1]))
        == 0
    )
    assert (
        sp.simplify(meilleure_approximation(X**3, famille, SYMETRIQUE) - R(3, 5) * X)
        == 0
    )


def test_la_distance_est_minimale_pour_les_coefficients_de_fourier() -> None:
    """Page 144 : `d^2 = ||f||^2 + somme |a_k - c_k|^2 - somme |c_k|^2`."""
    famille = orthonormaliser([sp.Integer(1), X], SYMETRIQUE)
    fonction = X**3
    coefficients = coefficients_de_fourier(fonction, famille, SYMETRIQUE)
    minimale = carre_de_la_distance(fonction, coefficients, famille, SYMETRIQUE)
    meilleure = meilleure_approximation(fonction, famille, SYMETRIQUE)
    assert sp.simplify(minimale - SYMETRIQUE.distance(fonction, meilleure) ** 2) == 0
    for decalage in (R(1, 10), -R(1, 3)):
        autres = (coefficients[0] + decalage, coefficients[1])
        ailleurs = carre_de_la_distance(fonction, autres, famille, SYMETRIQUE)
        assert sp.simplify(ailleurs - minimale - decalage**2) == 0


def test_l_inegalite_de_bessel() -> None:
    """Page 144 : `somme |c_k|^2 <= ||f||^2`, stricte si la famille n'epuise pas `f`."""
    famille = orthonormaliser([sp.Integer(1), X, X**2], SYMETRIQUE)
    coefficients = coefficients_de_fourier(sp.exp(X), famille, SYMETRIQUE)
    energie = sum(sp.Abs(c) ** 2 for c in coefficients)
    assert sp.N(energie) < sp.N(SYMETRIQUE.norme(sp.exp(X)) ** 2)
    exacte = coefficients_de_fourier(1 + X**2, famille, SYMETRIQUE)
    assert sp.simplify(sum(c**2 for c in exacte) - SYMETRIQUE.norme(1 + X**2) ** 2) == 0


def test_une_exponentielle_est_fonction_propre_d_un_systeme_lineaire() -> None:
    """8.5.2 : `K(D) y = e^(j n w t)` a pour solution `e^(j n w t) / K(j n w)`."""
    operateur = OperateurDifferentiel(D**2 + 3 * D + 2)
    omega = sp.Symbol("omega", positive=True)
    for n in (0, 1, -2):
        entree = harmonique(n, omega)
        sortie = reponse_a_un_harmonique(operateur, n, omega)
        assert sp.simplify(operateur.appliquer(sortie) - entree) == 0
        assert sp.simplify(sortie / entree - 1 / operateur.en(J * n * omega)) == 0


def test_la_reponse_a_une_serie_est_la_serie_des_reponses() -> None:
    """Page 147 : `g = c_0 / K(0) + c_1 e^(jwt) / K(jw) + ...`."""
    operateur = OperateurDifferentiel(D + 1)
    coefficients = {0: R(1, 3), 1: 2 - J, -1: 2 + J, 2: R(1, 5)}
    entree = sum(c * harmonique(n, 2) for n, c in coefficients.items())
    sortie = reponse_periodique(operateur, coefficients, 2)
    assert sp.simplify(operateur.appliquer(sortie) - entree) == 0


def test_un_systeme_resonant_n_a_pas_de_reponse_periodique() -> None:
    """Le syllabus suppose `Re(b_i) < 0`, donc `K(j n w) != 0` ; sinon rien ne tient."""
    oscillateur = OperateurDifferentiel(D**2 + 4)
    with pytest.raises(ValueError, match="harmonique resonant"):
        reponse_a_un_harmonique(oscillateur, 1, 2)


def test_la_mesure_normalisee_sur_un_intervalle_qui_ne_part_pas_de_zero() -> None:
    """`[1, 3]` : `dmu = dx / 2`, de masse 1 ; la moyenne de `x` y vaut 2."""
    espace = EspaceL2.normalise(X, 1, 3)
    assert espace.integrale(sp.Integer(1)) == 1
    assert espace.integrale(X) == 2
