"""Syllabus d'algebre, chapitre X, sections 1 a 7 : valeurs propres et reductions.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 154 a 174, et
les solutions des pages 182 et 183. Valeurs et vecteurs propres,
multiplicites, diagonalisation, matrices symetriques, triangularisation en
une ou deux etapes, reduction ; les cinq exemples resolus, avec les
matrices de passage du syllabus, puis les reponses des exercices 1 a 18.

Deux ecarts du syllabus sont etablis :

* page 160, le polynome caracteristique de la matrice symetrique est
  imprime `(6 - t)^2 (t - 12)` ; c'est `(6 - t)^2 (12 - t)` -- une
  permutation de lignes a ete faite sans changer le signe ;
* page 165, la matrice `T = P^-1 A P` porte un zero en position (2, 3) ;
  le produit, avec le `P` et le `P^-1` imprimes, y donne `1/5`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.diagonalisation import (
    diagonaliser_orthogonalement,
    est_diagonalisable,
    puissance,
    sous_espace_propre,
)
from polytech.maths.domain.algebre.espaces_de_fonctions import matrice_d_un_operateur
from polytech.maths.domain.algebre.jordan import multiplicites
from polytech.maths.domain.algebre.matrices_particulieres import (
    est_diagonale,
    est_une_matrice_orthogonale,
    trace,
)
from polytech.maths.domain.algebre.rang_exact import rang_par_echelonnement
from polytech.maths.domain.algebre.reduction import (
    Nature,
    nature_de_la_reduction,
    reduire_au_maximum,
)
from polytech.maths.domain.algebre.sous_espaces import engendrent_le_meme_sous_espace
from polytech.maths.domain.algebre.spectre import INCONNUE, Spectre
from polytech.maths.domain.algebre.triangularisation import (
    est_triangulaire_superieure,
    est_triangularisable,
    semblable_par,
)
from polytech.maths.domain.symboles import TEMPS

M = sp.Matrix
R = sp.Rational
T = INCONNUE
DIAGONALISABLE = M([[3, 1, -3], [3, 2, -4], [5, -1, -3]])
SYMETRIQUE = M([[7, -2, 1], [-2, 10, -2], [1, -2, 7]])
UNE_ETAPE = M([[5, 10, -7], [-3, -5, 3], [-1, 2, -1]])
DEUX_ETAPES = M([[-3, -3, -2], [-3, -4, -3], [5, 6, 4]])
COMPLEXE = M([[-6, 4, 2, -1], [-10, 5, 0, 2], [-1, 2, 3, -3], [-3, 2, 0, 0]])


def _espace(
    matrice: sp.Matrix, valeur: object, *officielle: tuple[object, ...]
) -> bool:
    """Le sous-espace propre est celui qu'engendre la base officielle."""
    obtenue = sous_espace_propre(matrice, valeur)
    return len(obtenue) == len(officielle) and engendrent_le_meme_sous_espace(
        obtenue, officielle
    )


def test_l_exemple_de_diagonalisation_du_x_2_4() -> None:
    """`-(t-1)(t+1)(t-2)`, trois sous-espaces propres, `P` et `D = diag(1, -1, 2)`."""
    polynome = Spectre(DIAGONALISABLE).polynome_caracteristique()
    assert sp.expand(polynome + (T - 1) * (T + 1) * (T - 2)) == 0
    assert multiplicites(DIAGONALISABLE) == {-1: (1, 1), 1: (1, 1), 2: (1, 1)}
    assert _espace(DIAGONALISABLE, 1, (1, 1, 1))
    assert _espace(DIAGONALISABLE, -1, (5, 7, 9))
    assert _espace(DIAGONALISABLE, 2, (4, 5, 3))
    passage = M([[1, 5, 4], [1, 7, 5], [1, 9, 3]])
    assert semblable_par(DIAGONALISABLE, passage) == sp.diag(1, -1, 2)
    assert est_diagonalisable(DIAGONALISABLE)
    assert nature_de_la_reduction(DIAGONALISABLE) is Nature.DIAGONALISABLE


def test_la_dimension_d_un_sous_espace_propre() -> None:
    """X.1.1 : `dim S(lambda) = n - rg(A - lambda I)`."""
    for valeur in (1, -1, 2):
        decalee = DIAGONALISABLE - valeur * sp.eye(3)
        dimension = len(sous_espace_propre(DIAGONALISABLE, valeur))
        assert dimension == 3 - rang_par_echelonnement(decalee)
    with pytest.raises(ValueError, match="n'est pas une valeur propre"):
        sous_espace_propre(DIAGONALISABLE, 7)


def test_le_polynome_caracteristique_de_la_page_160_change_de_signe() -> None:
    """Imprime `(6 - t)^2 (t - 12)` ; le determinant vaut `(6 - t)^2 (12 - t)`."""
    polynome = Spectre(SYMETRIQUE).polynome_caracteristique()
    assert sp.expand(polynome - (6 - T) ** 2 * (12 - T)) == 0
    assert sp.expand(polynome - (6 - T) ** 2 * (T - 12)) != 0
    assert multiplicites(SYMETRIQUE) == {6: (2, 2), 12: (1, 1)}


def test_la_diagonalisation_orthogonale_de_la_page_161() -> None:
    """La base `{(2, 1, 0), (-1, 2, 5), (1, -2, 1)}` normee : `diag(6, 6, 12)`."""
    assert _espace(SYMETRIQUE, 6, (2, 1, 0), (-1, 2, 5))
    assert _espace(SYMETRIQUE, 12, (1, -2, 1))
    passage = M.hstack(
        M([2, 1, 0]) / sp.sqrt(5),
        M([-1, 2, 5]) / sp.sqrt(30),
        M([1, -2, 1]) / sp.sqrt(6),
    )
    assert est_une_matrice_orthogonale(passage)
    assert sp.simplify(passage.T * SYMETRIQUE * passage) == sp.diag(6, 6, 12)


def test_la_diagonalisation_orthogonale_automatique() -> None:
    """`P^-1 = P^t` et `P^t A P` diagonale, valeurs propres 6, 6, 12."""
    passage, diagonale = diagonaliser_orthogonalement(SYMETRIQUE)
    assert est_une_matrice_orthogonale(passage)
    assert est_diagonale(diagonale)
    assert sorted(diagonale[i, i] for i in range(3)) == [6, 6, 12]
    assert sp.simplify(passage.T * SYMETRIQUE * passage - diagonale) == sp.zeros(3, 3)


def test_seule_une_matrice_symetrique_se_diagonalise_orthogonalement() -> None:
    """X.2.5 : « les matrices symetriques reelles sont les seules »."""
    with pytest.raises(ValueError, match="matrice non symetrique"):
        diagonaliser_orthogonalement(DIAGONALISABLE)


def test_la_triangularisation_en_une_etape_du_x_3_4() -> None:
    """`-(t+2)^2 (t-3)`, `m_g(-2) = 1`, deux sous-espaces propres de dimension 1."""
    polynome = Spectre(UNE_ETAPE).polynome_caracteristique()
    assert sp.expand(polynome + (T + 2) ** 2 * (T - 3)) == 0
    assert multiplicites(UNE_ETAPE) == {-2: (2, 1), 3: (1, 1)}
    assert rang_par_echelonnement(UNE_ETAPE + 2 * sp.eye(3)) == 2
    assert _espace(UNE_ETAPE, -2, (1, 0, 1))
    assert _espace(UNE_ETAPE, 3, (-26, 15, 14))
    assert not est_diagonalisable(UNE_ETAPE)
    assert nature_de_la_reduction(UNE_ETAPE) is Nature.TRIANGULARISABLE
    assert reduire_au_maximum(UNE_ETAPE).etapes == 1


def test_la_matrice_t_de_la_page_165_porte_un_zero_de_trop() -> None:
    """Avec le `P` et le `P^-1` imprimes, l'element (2, 3) de `P^-1 A P` vaut `1/5`."""
    passage = M([[1, -26, 0], [0, 15, 0], [1, 14, 1]])
    inverse = M([[1, R(26, 15), 0], [0, R(1, 15), 0], [-1, -R(40, 15), 1]])
    assert passage.inv() == inverse
    obtenue = semblable_par(UNE_ETAPE, passage)
    imprimee = M([[-2, 0, -R(9, 5)], [0, 3, 0], [0, 0, -2]])
    assert obtenue == M([[-2, 0, -R(9, 5)], [0, 3, R(1, 5)], [0, 0, -2]])
    assert obtenue - imprimee == M([[0, 0, 0], [0, 0, R(1, 5)], [0, 0, 0]])
    assert est_triangulaire_superieure(obtenue)
    assert inverse * UNE_ETAPE * M([0, 0, 1]) == M([-R(9, 5), R(1, 5), -2])


def test_la_discussion_du_x_3_5_selon_a() -> None:
    """`-(t+1)^2 (t-a)` ; `a = -1` : valeur triple, `m_g = 1` ; sinon `m_g(-1) = 1`."""
    a = sp.Symbol("a")
    matrice = M([[-3, -3, -2], [-3, -4, -3], [a + 6, a + 7, a + 5]])
    polynome = Spectre(matrice).polynome_caracteristique()
    assert sp.expand(polynome + (T + 1) ** 2 * (T - a)) == 0
    assert multiplicites(DEUX_ETAPES) == {-1: (3, 1)}
    assert multiplicites(matrice.subs(a, 4)) == {-1: (2, 1), 4: (1, 1)}
    assert reduire_au_maximum(matrice.subs(a, 4)).etapes == 1


def test_la_triangularisation_en_deux_etapes_du_x_3_5() -> None:
    """`P_1`, `P_1^-1 A P_1`, `A_1`, `P_2`, puis `P` et la matrice triangulaire."""
    assert _espace(DEUX_ETAPES, -1, (1, 0, -1))
    p1 = M([[1, 0, 0], [0, 1, 0], [-1, 0, 1]])
    assert p1.inv() == M([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
    premiere = semblable_par(DEUX_ETAPES, p1)
    assert premiere == M([[-1, -3, -2], [0, -4, -3], [0, 3, 2]])
    a1 = premiere[1:, 1:]
    assert a1 == M([[-4, -3], [3, 2]])
    assert _espace(a1, -1, (1, -1))
    p2 = M([[1, 0], [-1, 1]])
    passage = p1 * sp.diag(1, p2)
    assert passage == M([[1, 0, 0], [0, 1, 0], [-1, -1, 1]])
    assert passage.inv() == M([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
    finale = semblable_par(DEUX_ETAPES, passage)
    assert finale == M([[-1, -1, -2], [0, -1, -3], [0, 0, -1]])
    assert reduire_au_maximum(DEUX_ETAPES).etapes == 2


def test_la_reduction_du_x_4_3() -> None:
    """`(t-1)^2 (t^2+1)` : ni diagonalisable ni triangularisable, deux etapes."""
    polynome = Spectre(COMPLEXE).polynome_caracteristique()
    assert sp.expand(polynome - (T - 1) ** 2 * (T**2 + 1)) == 0
    assert not est_triangularisable(COMPLEXE)
    assert nature_de_la_reduction(COMPLEXE) is Nature.NI_L_UN_NI_L_AUTRE
    assert _espace(COMPLEXE, 1, (1, 2, 0, 1))
    p1 = M([[1, 0, 0, 0], [2, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]])
    premiere = semblable_par(COMPLEXE, p1)
    assert premiere == M([[1, 4, 2, -1], [0, -3, -4, 4], [0, 2, 3, -3], [0, -2, -2, 1]])
    a1 = premiere[1:, 1:]
    assert _espace(a1, 1, (1, -1, 0))
    p2 = M([[1, 0, 0], [-1, 1, 0], [0, 0, 1]])
    passage = p1 * sp.diag(1, p2)
    assert passage == M([[1, 0, 0, 0], [2, 1, 0, 0], [0, -1, 1, 0], [1, 0, 0, 1]])
    assert passage.inv() == M(
        [[1, 0, 0, 0], [-2, 1, 0, 0], [-2, 1, 1, 0], [-1, 0, 0, 1]]
    )
    assert semblable_par(COMPLEXE, passage) == M(
        [[1, 2, 2, -1], [0, 1, -4, 4], [0, 0, -1, 1], [0, 0, -2, 1]]
    )


def test_la_reduction_automatique_laisse_un_bloc_sans_valeur_propre_reelle() -> None:
    """Deux etapes, un triangle d'ordre 2 de diagonale `1, 1`, puis un bloc `2 x 2`."""
    reduction = reduire_au_maximum(COMPLEXE)
    assert reduction.etapes == 2
    reduite = reduction.reduite
    assert semblable_par(COMPLEXE, reduction.passage) == reduite
    assert (reduite[0, 0], reduite[1, 1]) == (1, 1)
    assert reduite[1:, 0] == sp.zeros(3, 1)
    assert reduite[2:, 1] == sp.zeros(2, 1)
    assert sp.expand(reduite[2:, 2:].charpoly(T).as_expr() - (T**2 + 1)) == 0


def test_une_matrice_sans_valeur_propre_reelle_ne_se_reduit_pas() -> None:
    """La rotation d'un quart de tour : zero etape, et la matrice elle-meme."""
    rotation = M([[0, -1], [1, 0]])
    reduction = reduire_au_maximum(rotation)
    assert (reduction.etapes, reduction.reduite) == (0, rotation)
    assert reduction.passage == sp.eye(2)
    assert nature_de_la_reduction(rotation) is Nature.NI_L_UN_NI_L_AUTRE
    assert not est_diagonalisable(rotation)


def test_les_reponses_des_exercices_1_a_4() -> None:
    """Sous-espaces propres, matrice triangulaire, un vecteur propre, un rang."""
    un = M([[-15, 1, -9], [0, 6, 0], [4, 1, -3]])
    assert _espace(un, 6, (0, 9, 1))
    assert _espace(un, -9, (-3, 0, 2))
    deux = M([[5, 7, 14, -8], [0, 3, 2, 0], [0, 0, 0, 2], [0, 0, 0, 7]])
    assert Spectre(deux).valeurs_propres() == (0, 3, 5, 7)
    assert _espace(deux, 5, (1, 0, 0, 0))
    trois = M([[1, 2, 3], [3, 0, 3], [1, 4, 1]])
    assert trois * M([1, 1, 1]) == 6 * M([1, 1, 1])
    assert 5 - 2 > 2


@pytest.mark.parametrize(
    ("matrice", "nature", "valeurs", "etapes"),
    [
        ([[2, 2], [1, 3]], Nature.DIAGONALISABLE, {1: 1, 4: 1}, {1}),
        (
            [[2, 0, 1, 1], [-2, 2, 1, -1], [0, 0, 3, 1], [0, -2, -1, 3]],
            Nature.TRIANGULARISABLE,
            {2: 3, 4: 1},
            {1, 2},
        ),
        (
            [[-15, 1, -9], [0, 6, 0], [4, 1, -3]],
            Nature.TRIANGULARISABLE,
            {6: 1, -9: 2},
            {1},
        ),
        (
            [[1, -3, 3], [3, -5, 3], [6, -6, 4]],
            Nature.DIAGONALISABLE,
            {-2: 2, 4: 1},
            {1},
        ),
        (
            [[3, -1, 0, 0], [0, 3, 0, 0], [1, 0, 3, 1], [0, 1, 0, 3]],
            Nature.TRIANGULARISABLE,
            {3: 4},
            {1, 2},
        ),
        ([[2, 2, 1], [1, 3, 1], [1, 2, 2]], Nature.DIAGONALISABLE, {1: 2, 5: 1}, {1}),
    ],
    ids=["5", "6", "7", "8", "9", "11"],
)
def test_les_reponses_des_exercices_5_a_11(
    matrice: list[list[int]],
    nature: Nature,
    valeurs: dict[int, int],
    etapes: set[int],
) -> None:
    """La nature de la reduction, les valeurs propres, et le nombre d'etapes."""
    a = M(matrice)
    assert nature_de_la_reduction(a) is nature
    assert a.eigenvals() == valeurs
    reduction = reduire_au_maximum(a)
    assert reduction.etapes in etapes
    assert est_triangulaire_superieure(reduction.reduite)
    assert semblable_par(a, reduction.passage) == reduction.reduite


def test_la_reponse_de_l_exercice_10() -> None:
    """La matrice du X.4.3 : ni diagonalisable ni triangularisable, `1` double."""
    assert nature_de_la_reduction(COMPLEXE) is Nature.NI_L_UN_NI_L_AUTRE
    reelles = {v: m for v, m in COMPLEXE.eigenvals().items() if v.is_real}
    assert reelles == {1: 2}
    assert reduire_au_maximum(COMPLEXE).etapes in {1, 2}


def test_la_reponse_de_l_exercice_12() -> None:
    """Valeurs propres 2 et 8, et la base orthonormee proposee."""
    a = M([[4, 2, 2], [2, 4, 2], [2, 2, 4]])
    assert a.eigenvals() == {2: 2, 8: 1}
    officielle = M.hstack(
        M([1, 0, -1]) / sp.sqrt(2),
        M([1, -2, 1]) / sp.sqrt(6),
        M([1, 1, 1]) / sp.sqrt(3),
    )
    assert est_une_matrice_orthogonale(officielle)
    assert sp.simplify(officielle.T * a * officielle) == sp.diag(2, 2, 8)
    passage, diagonale = diagonaliser_orthogonalement(a)
    assert est_une_matrice_orthogonale(passage)
    assert sorted(diagonale[i, i] for i in range(3)) == [2, 2, 8]


def test_la_reponse_de_l_exercice_13() -> None:
    """`A^k` pour `[[1-a, a], [b, 1-b]]`, par diagonalisation."""
    a, b = sp.symbols("a b", positive=True)
    k = sp.Symbol("k", integer=True, positive=True)
    matrice = M([[1 - a, a], [b, 1 - b]])
    q = (1 - a - b) ** k
    officielle = M([[b + a * q, a - a * q], [b - b * q, a + b * q]]) / (a + b)
    assert sp.simplify(puissance(matrice, k) - officielle) == sp.zeros(2, 2)
    assert sp.simplify(officielle.subs(k, 1) - matrice) == sp.zeros(2, 2)
    assert sp.simplify(officielle.subs(k, 3) - matrice**3) == sp.zeros(2, 2)


def test_une_matrice_non_diagonalisable_n_a_pas_de_puissance_par_ce_chemin() -> None:
    with pytest.raises(ValueError, match="matrice non diagonalisable"):
        puissance(UNE_ETAPE, 3)


def test_la_reponse_de_l_exercice_14() -> None:
    """Determinant 6, trace 7, valeurs propres 1 et 3 connues : `diag(1, 1, 3, 2)`."""
    a = M([[3, 2, 2, -4], [2, 3, 2, -1], [1, 1, 2, -1], [2, 2, 2, -1]])
    assert a.det() == 6
    assert trace(a) == 7
    assert a * M([0, 3, 1, 2]) == 3 * M([0, 3, 1, 2])
    assert a.eigenvals() == {1: 2, 2: 1, 3: 1}
    assert est_diagonalisable(a)


def test_la_reponse_de_l_exercice_15() -> None:
    """Une racine carree de la matrice donnee, a valeurs propres positives."""
    a = M([[1, 0, 0], [-1, 1, -1], [1, 0, 2]])
    assert a**2 == M([[1, 0, 0], [-3, 1, -3], [3, 0, 4]])
    assert all(valeur > 0 for valeur in a.eigenvals())


def test_la_reponse_de_l_exercice_16() -> None:
    """`S(1)` et `S(-1)`, chacun de dimension 2 ; `(1, 1, 1, 1)` est dans `S(-1)`."""
    a = M([[1, -1, -1, -1], [-1, -1, -1, 1], [-1, -1, 1, -1], [-1, 1, -1, -1]]) / 2
    assert _espace(a, 1, (1, 0, -1, 0), (0, 1, -2, 1))
    assert _espace(a, -1, (1, 0, 1, 2), (0, 1, 0, -1))
    assert a * M([1, 1, 1, 1]) == -M([1, 1, 1, 1])
    assert a.eigenvals() == {1: 2, -1: 2}


def test_la_reponse_de_l_exercice_17() -> None:
    """La derivation sur `<sin t, cos t>` : valeurs propres `+-i`, `e^(+-it)`."""
    base = (sp.sin(TEMPS), sp.cos(TEMPS))
    derivation = matrice_d_un_operateur(lambda f: sp.diff(f, TEMPS), base, TEMPS)
    assert derivation == M([[0, -1], [1, 0]])
    assert derivation.eigenvals() == {sp.I: 1, -sp.I: 1}
    for k in (sp.I, -sp.I):
        (vecteur,) = sous_espace_propre(derivation, k)
        fonction = vecteur[0] * base[0] + vecteur[1] * base[1]
        assert sp.simplify(sp.diff(fonction, TEMPS) - k * fonction) == 0
        rapport = sp.simplify((fonction / sp.exp(k * TEMPS)).rewrite(sp.exp))
        assert TEMPS not in rapport.free_symbols


def test_la_reponse_de_l_exercice_18() -> None:
    """La projection sur le plan `2x + y - z = 0` parallelement a `(1, -1, 3)`."""
    normale, direction = M([[2, 1, -1]]), M([1, -1, 3])
    projection = sp.eye(3) - direction * normale / (normale * direction)[0]
    demi = R(1, 2)
    assert projection == M([[2, demi, -demi], [-1, demi, demi], [3, 3 * demi, -demi]])
    assert _espace(projection, 0, (1, -1, 3))
    assert _espace(projection, 1, (1, -2, 0), (0, 1, 1))
    assert projection * projection == projection
