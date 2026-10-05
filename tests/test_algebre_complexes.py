"""Syllabus d'algebre, chapitre I : les nombres complexes.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 1 a 16. Les
exemples resolus des paragraphes I.1.3, I.2.4, I.3.1, I.4.2 et I.5.2, puis
les « Solutions du 1er chapitre » : chaque reponse officielle des
exercices 1 a 18 est recalculee.

Une coquille du syllabus est etablie : page 3, l'enonce du deuxieme
exemple ecrit `(3 + 4i).(2 + 5i)` et calcule `(3 + 4i).(2 - 5i)`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.complexes_exacts import (
    forme_polaire,
    lieu_dans_le_plan_de_gauss,
    par_moivre,
    racines_carrees,
    racines_de_l_unite,
    racines_du_second_degre,
    racines_niemes_exactes,
)
from polytech.maths.domain.symboles import VARIABLE as X

I = sp.I  # noqa: E741 -- l'unite imaginaire, que le syllabus note i
R = sp.Rational
sqrt, pi, exp = sp.sqrt, sp.pi, sp.exp
Y = sp.Symbol("y", real=True)


def _memes(obtenus: tuple[sp.Expr, ...], attendus: tuple[sp.Expr, ...]) -> bool:
    """Deux listes de complexes egales a l'ordre pres."""
    restants = list(attendus)
    for valeur in obtenus:
        for candidat in restants:
            if sp.simplify(sp.expand_complex(valeur - candidat)) == 0:
                restants.remove(candidat)
                break
        else:
            return False
    return not restants


def test_les_parties_reelle_et_imaginaire_des_exemples_du_i_1_3() -> None:
    """Somme : `4 + 12i`. Quotient : `(7 - i)/(3 - 5i) = 13/17 + 16i/17`."""
    assert sp.expand((3 + 4 * I) + (2 + 5 * I) - (1 - 3 * I)) == 4 + 12 * I
    quotient = sp.simplify((7 - I) / (3 - 5 * I))
    assert (sp.re(quotient), sp.im(quotient)) == (R(13, 17), R(16, 17))


def test_le_produit_de_la_page_3_est_calcule_sur_un_autre_enonce() -> None:
    """L'enonce ecrit `(3 + 4i).(2 + 5i)` ; `26 - 7i` est le produit par `2 - 5i`."""
    assert sp.expand((3 + 4 * I) * (2 - 5 * I)) == 26 - 7 * I
    assert sp.expand((3 + 4 * I) * (2 + 5 * I)) == -14 + 23 * I


def test_le_conjugue_et_le_module() -> None:
    """I.1.2 et I.2.1 : `z z_barre = a^2 + b^2`, `z + z_barre = 2 Re z`, les modules."""
    a, b = sp.symbols("a b", real=True)
    z = a + b * I
    assert sp.expand(z * sp.conjugate(z)) == a**2 + b**2
    assert sp.expand(z + sp.conjugate(z)) == 2 * a
    w = 3 - 4 * I
    assert sp.Abs((1 + 2 * I) * w) == sp.Abs(1 + 2 * I) * sp.Abs(w)
    assert sp.Abs((1 + 2 * I) + w) <= sp.Abs(1 + 2 * I) + sp.Abs(w)


@pytest.mark.parametrize(
    ("z", "module", "argument"),
    [
        (sqrt(3) - I, 2, -pi / 6),
        (-1 + I, sqrt(2), 3 * pi / 4),
        (1 + sqrt(3) * I, 2, pi / 3),
        (2 * I, 2, pi / 2),
        (sp.Integer(-3), 3, pi),
        (1 / (1 + I), 1 / sqrt(2), -pi / 4),
    ],
)
def test_le_module_et_l_argument(
    z: sp.Expr, module: sp.Expr, argument: sp.Expr
) -> None:
    """Exemples I.2.4.1 et 2, exercice 3 a) a d)."""
    obtenu, angle = forme_polaire(z)
    assert sp.simplify(obtenu - module) == 0
    assert sp.simplify(angle - argument) == 0


def test_un_argument_qui_n_est_pas_remarquable() -> None:
    """Exemple I.2.4.3 : `-12 - 5i`, module 13, `theta = arctan(5/12) + pi`.

    Exercice 3 e) : `-5 + 4i`, module `racine(41)`, `theta = pi + arctan(-4/5)`.
    """
    module, angle = forme_polaire(-12 - 5 * I)
    assert module == 13
    assert sp.simplify(angle - (sp.atan(R(5, 12)) + pi) + 2 * pi) == 0
    module, angle = forme_polaire(-5 + 4 * I)
    assert module == sqrt(41)
    assert sp.simplify(angle - (pi + sp.atan(-R(4, 5)))) == 0


def test_l_argument_de_zero_n_est_pas_defini() -> None:
    with pytest.raises(ValueError, match="argument de zero"):
        forme_polaire(sp.Integer(0))


def test_le_produit_et_le_quotient_en_notation_exponentielle() -> None:
    """Exemples I.2.4.4 a 7 : les modules se multiplient, les arguments s'ajoutent."""
    z1, z2 = sqrt(3) - I, -1 + I
    produit = 2 * sqrt(2) * exp(I * 7 * pi / 12)
    assert sp.simplify(sp.expand_complex(z1 * z2 - produit)) == 0
    assert (
        sp.simplify(sp.expand_complex(par_moivre(z1, 4) - 2**4 * exp(-2 * I * pi / 3)))
        == 0
    )
    cinquieme = 2 ** R(5, 2) * exp(I * 15 * pi / 4)
    assert sp.simplify(sp.expand_complex(par_moivre(z2, 5) - cinquieme)) == 0
    quotient = 2 ** R(3, 2) * exp(-I * 53 * pi / 12)
    assert (
        sp.simplify(sp.expand_complex(par_moivre(z1, 4) / par_moivre(z2, 5) - quotient))
        == 0
    )


def test_la_formule_de_moivre_rend_la_puissance() -> None:
    """Exercice 5 : `(racine(3) + i)^4 = -8 + 8 racine(3) i`."""
    puissance = sp.expand_complex(par_moivre(sqrt(3) + I, 4))
    assert sp.simplify(puissance - (-8 + 8 * sqrt(3) * I)) == 0
    assert sp.expand((sqrt(3) + I) ** 4) == -8 + 8 * sqrt(3) * I


def test_les_reponses_des_exercices_1_et_11() -> None:
    """Solutions 1 a) a e) et 11 a) a c)."""
    assert sp.expand((1 + 2 * I) - (3 - 4 * I) + (5 - 6 * I)) == 3
    assert sp.expand((7 - 8 * I) * (9 + I)) == 71 - 65 * I
    assert sp.expand((4 + 3 * I) ** 2) == 7 + 24 * I
    assert sp.simplify((-2 + 3 * I) / (-3 + 4 * I) - (R(18, 25) - I / 25)) == 0
    assert I**5 == I
    assert sp.expand_complex(3 * I + 2 * exp(I * pi)) == -2 + 3 * I
    onze_b = sp.expand_complex(exp(I * pi / 3) - (3 + 2 * I))
    assert sp.simplify(onze_b - (-R(5, 2) + I * (sqrt(3) / 2 - 2))) == 0
    onze_c = 3 * exp(I * pi / 5) * (sqrt(3) - I)
    assert sp.simplify(sp.expand_complex(onze_c - 6 * exp(I * pi / 30))) == 0


def test_la_reponse_de_l_exercice_2() -> None:
    """`(x + iy)^4` reel plus grand que 16 : sur un axe, a 2 au moins de l'origine."""
    x = sp.Symbol("x", real=True)
    puissance = sp.expand((x + I * Y) ** 4)
    imaginaire = sp.factor(sp.im(puissance))
    assert sp.expand(imaginaire - 4 * x * Y * (x**2 - Y**2)) == 0
    assert sp.re(puissance).subs(Y, 0) == x**4
    assert sp.re(puissance).subs(x, 0) == Y**4
    assert sp.re(puissance).subs(Y, x) == -4 * x**4


def test_la_reponse_de_l_exercice_4() -> None:
    """`(1 + i)^3 / (1 + i racine(3))^4` : module `2^(-5/2)`, argument `-7 pi / 12`."""
    module, angle = forme_polaire((1 + I) ** 3 / (1 + I * sqrt(3)) ** 4)
    assert sp.simplify(module - 2 ** R(-5, 2)) == 0
    assert sp.simplify(angle + 7 * pi / 12) == 0


def test_la_reponse_de_l_exercice_6() -> None:
    """`(-1 + i racine 3)^15 / (1 - i)^20 + (-1 - i racine 3)^15 / (1 + i)^20 = -64`."""
    premier = par_moivre(-1 + I * sqrt(3), 15) / par_moivre(1 - I, 20)
    second = par_moivre(-1 - I * sqrt(3), 15) / par_moivre(1 + I, 20)
    assert sp.simplify(sp.expand_complex(premier + second)) == -64


def test_la_reponse_de_l_exercice_8() -> None:
    """`z1 = racine(3) - i`, `z2 = -3i` : produit de module 6, quotient de 2/3."""
    z1, z2 = sqrt(3) - I, -3 * I
    module, angle = forme_polaire(z1 * z2)
    assert module == 6
    assert sp.simplify(angle - (4 * pi / 3 - 2 * pi)) == 0
    module, angle = forme_polaire(z1 / z2)
    assert module == R(2, 3)
    assert sp.simplify(angle - (-5 * pi / 3 + 2 * pi)) == 0


def test_le_cercle_de_l_exemple_i_3_1() -> None:
    """`|z - 5i| = 3` : `x^2 + (y - 5)^2 = 9`, cercle de centre `(0, 5)`, de rayon 3."""
    lieu = lieu_dans_le_plan_de_gauss(lambda z: sp.Abs(z - 5 * I) ** 2 - 9)
    assert sp.expand(lieu - (X**2 + (Y - 5) ** 2 - 9)) == 0


def test_les_lieux_de_l_exercice_12() -> None:
    """Une droite, les deux bissectrices, l'axe `Oy` : solutions A, B et C."""
    droite = lieu_dans_le_plan_de_gauss(lambda z: sp.re(z) + sp.im(z) - 3)
    assert droite == X + Y - 3
    bissectrices = lieu_dans_le_plan_de_gauss(lambda z: sp.re(z**2))
    assert sp.factor(bissectrices) == (X - Y) * (X + Y)
    axe = lieu_dans_le_plan_de_gauss(lambda z: sp.re(z + sp.conjugate(z)))
    assert axe == 2 * X


def test_le_demi_plan_de_l_exercice_13() -> None:
    """`|(z - 1)/(z - 2)| < 1` : `|z - 1|^2 - |z - 2|^2 = 2x - 3 < 0`, `x < 3/2`."""
    ecart = lieu_dans_le_plan_de_gauss(
        lambda z: sp.Abs(z - 1) ** 2 - sp.Abs(z - 2) ** 2
    )
    assert sp.expand(ecart) == 2 * X - 3


def test_les_racines_quatriemes_de_l_exemple_i_4_2() -> None:
    """`-8 + 8 racine(3) i` : `racine(3) + i`, `-1 + racine(3) i`, et leurs opposes."""
    attendues = (sqrt(3) + I, -1 + sqrt(3) * I, -sqrt(3) - I, 1 - sqrt(3) * I)
    assert _memes(racines_niemes_exactes(-8 + 8 * sqrt(3) * I, 4), attendues)


def test_les_racines_de_l_unite() -> None:
    """Exemple I.4.2 et exercice 14 : `eta_l = e^(2 i l pi / n)`."""
    cinq = racines_de_l_unite(5)
    assert cinq == tuple(exp(2 * I * pi * k / 5) for k in range(5))
    demi, racine = R(1, 2), sqrt(3) / 2
    assert _memes(racines_de_l_unite(3), (1, -demi + racine * I, -demi - racine * I))
    assert _memes(racines_de_l_unite(4), (1, I, -1, -I))
    six = (
        1,
        demi + racine * I,
        -demi + racine * I,
        -1,
        -demi - racine * I,
        demi - racine * I,
    )
    assert _memes(racines_de_l_unite(6), six)


def test_les_proprietes_des_racines_de_l_unite() -> None:
    """`eta_l = eta_1^l`, `y_l = y_0 eta_l`, et (exercice 17 a) leur somme est nulle."""
    huit = racines_de_l_unite(8)
    for rang, racine in enumerate(huit):
        assert sp.simplify(sp.expand_complex(racine - huit[1] ** rang)) == 0
    assert sp.simplify(sp.expand_complex(sum(huit))) == 0
    racines = racines_niemes_exactes(-8 + 8 * sqrt(3) * I, 4)
    quatre = racines_de_l_unite(4)
    assert _memes(racines, tuple(racines[0] * eta for eta in quatre))


def test_la_somme_des_puissances_des_racines_de_l_unite() -> None:
    """Exercice 17 b) : `n` si `k` est multiple de `n`, 0 sinon."""
    six = racines_de_l_unite(6)
    for k, attendue in ((1, 0), (4, 0), (6, 6), (12, 6)):
        somme = sum(racine**k for racine in six)
        assert sp.simplify(sp.expand_complex(somme)) == attendue


def test_la_reponse_de_l_exercice_15() -> None:
    """Racines carrees de `-10i`, cubiques de `-2 + 2i`, quatriemes de 81, de -1."""
    assert _memes(
        racines_niemes_exactes(-10 * I, 2), (sqrt(5) * (-1 + I), sqrt(5) * (1 - I))
    )
    demi, racine = R(1, 2), sqrt(3) / 2
    cubiques = (
        1 + I,
        (-demi - racine) + I * (-demi + racine),
        (-demi + racine) + I * (-demi - racine),
    )
    assert _memes(racines_niemes_exactes(-2 + 2 * I, 3), cubiques)
    assert _memes(racines_niemes_exactes(sp.Integer(81), 4), (3, 3 * I, -3, -3 * I))
    cinquiemes = tuple(exp(I * pi * k / 5) for k in (1, 3, 5, 7, 9))
    assert _memes(racines_niemes_exactes(sp.Integer(-1), 5), cinquiemes)


def test_les_racines_carrees_du_discriminant_de_l_exemple_i_5_2() -> None:
    """`Delta = -12 + 16i` : `a^2 - b^2 = -12`, `2ab = 16`, d'ou `+-(2 + 4i)`."""
    assert _memes(racines_carrees(-12 + 16 * I), (2 + 4 * I, -2 - 4 * I))


def test_l_equation_du_second_degre_de_l_exemple_i_5_2() -> None:
    """`(1 - i) x^2 + 2x + 4 = 0` : `x_1 = -1 + i`, `x_2 = -2i`."""
    assert _memes(racines_du_second_degre(1 - I, 2, 4), (-1 + I, -2 * I))


def test_un_coefficient_dominant_nul_n_est_pas_un_second_degre() -> None:
    with pytest.raises(ValueError, match="coefficient dominant nul"):
        racines_du_second_degre(0, 2, 4)


def test_les_reponses_de_l_exercice_18() -> None:
    """b) `(5 +- racine(31) i)/4` ; c) `i` et `3/2 + i/2` ; a), d), e) substituees."""
    b = ((5 + sqrt(31) * I) / 4, (5 - sqrt(31) * I) / 4)
    assert _memes(racines_du_second_degre(-2, 5, -7), b)
    assert _memes(
        racines_du_second_degre(2 - 2 * I, -6, 2 + 4 * I), (I, R(3, 2) + I / 2)
    )
    for racine in (2 + I, 2 - I, -2 + I, -2 - I):
        assert sp.expand(racine**4 - 6 * racine**2 + 25) == 0
    septieme = X**7 + 3 * X**6 + 64 * X + 192
    a = (-3, sqrt(3) + I, 2 * I, -sqrt(3) + I, -sqrt(3) - I, -2 * I, sqrt(3) - I)
    for racine in a:
        assert sp.expand(septieme.subs(X, racine)) == 0
    cubique = X**3 + 2 * (I - 1) * X**2 + (3 - 4 * I) * X - 6
    for racine in (2, I, -3 * I):
        assert sp.expand(cubique.subs(X, racine)) == 0


def test_l_identite_de_l_exercice_7() -> None:
    """`(1 + i racine 3)(1 + i) e^(i phi) = 2 racine(2) e^(i (7 pi/12 + phi))`."""
    phi = sp.Symbol("phi", real=True)
    gauche = (1 + I * sqrt(3)) * (1 + I) * (sp.cos(phi) + I * sp.sin(phi))
    droite = 2 * sqrt(2) * exp(I * (7 * pi / 12 + phi))
    assert sp.simplify(sp.expand_complex(gauche - droite)) == 0


def test_les_reponses_de_l_exercice_9() -> None:
    """a) `x >= 0`, `y = 0` ; b) `y = 0` ; c) `x = 0`, `y = -pi + 2k pi`."""
    x = sp.Symbol("x", real=True)
    assert sp.Abs(sp.Integer(3)) == 3
    assert sp.Abs(sp.Integer(-3)) != -3
    assert sp.im(x + I * Y - sp.Abs(x + I * Y)) == Y
    assert sp.simplify((x + I * Y - x * exp(I * Y)).subs(Y, 0)) == 0
    for k in (-1, 0, 2):
        assert exp(0 + I * (-pi + 2 * k * pi)) == -1
    module = sp.Abs(exp(x + I * Y))
    assert sp.simplify(module - exp(x)) == 0


def test_la_reponse_de_l_exercice_10() -> None:
    """`z_barre = z^(n-1)` : pour `n > 2`, zero ou une racine `n`-ieme de l'unite."""
    for n in (3, 4, 5):
        for racine in (sp.Integer(0), *racines_de_l_unite(n)):
            ecart = sp.conjugate(racine) - racine ** (n - 1)
            assert sp.simplify(sp.expand_complex(ecart)) == 0
    assert sp.conjugate(sp.Integer(1)) == 1
    reel = sp.Symbol("r", real=True)
    assert sp.conjugate(reel) == reel
    assert sp.conjugate(2 * I) != (2 * I) ** 3


def test_la_reponse_de_l_exercice_16() -> None:
    """`z^7` et `1/z^2` conjugues : `z = e^(2 i k pi / 5)`, `k = 0, ..., 4`."""
    for z in racines_de_l_unite(5):
        ecart = sp.conjugate(z**7) - 1 / z**2
        assert sp.simplify(sp.expand_complex(ecart)) == 0
    hors = exp(I * pi / 3)
    assert sp.simplify(sp.expand_complex(sp.conjugate(hors**7) - 1 / hors**2)) != 0


def test_la_somme_ponderee_de_l_exercice_17_c() -> None:
    """`1 + 2 eps + ... + n eps^(n-1) = -n/(1 - eps)` si `eps != 1`."""
    n = 5
    for eps in racines_de_l_unite(n)[1:]:
        somme = sum((k + 1) * eps**k for k in range(n))
        assert sp.simplify(sp.expand_complex(somme + n / (1 - eps))) == 0
    assert sum(k + 1 for k in range(n)) == n * (n + 1) // 2
