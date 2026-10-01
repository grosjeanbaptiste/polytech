"""Le chapitre I du syllabus d'algebre : les nombres complexes.

Un chapitre entier que rien ne portait -- notation algebrique, conjugue,
notations trigonometrique et exponentielle, racines n-iemes, equations
polynomiales dans C.

Le syllabus dit lui-meme pourquoi ce chapitre ouvre le cours : « malgre
le role essentiel que joue l'ensemble des reels, celui-ci presente
certains inconvenients. Par exemple, le polynome `x^2 + 1` n'a pas de
racine reelle. Dans C, le polynome `x^2 + 1` possede deux racines. » La
motivation est donc exactement le tuple vide que
`racines_reelles_du_second_degre` rend sur ce polynome.

Les verifications ne recopient pas les formules : elles emploient les
**identites** que les complexes satisfont. `|z|^2 = z z_barre` relie le
module au conjugue sans passer par la racine carree ; chaque racine
n-ieme elevee a la puissance `n` doit redonner `z` ; et la somme des `n`
racines doit etre nulle pour `n >= 2`, parce qu'elles sont les sommets
d'un polygone regulier centre sur l'origine.
"""

from __future__ import annotations

import cmath
import math

import pytest
import sympy as sp

from polytech.maths.domain.algebre.complexes import (
    argument,
    conjugue,
    forme_exponentielle,
    module,
    racines_du_second_degre_dans_c,
    racines_niemes,
)


def test_le_module_au_carre_est_le_produit_par_le_conjugue() -> None:
    """`|z|^2 = z z_barre` : l'identite qui definit le conjugue.

    Elle relie les deux notions sans passer par la racine carree, donc
    sans perte de precision -- et une faute de signe sur le conjugue s'y
    verrait aussitot.
    """
    for z in (3 + 4j, -2 + 0.5j, 1j, -7 + 0j):
        assert module(z) ** 2 == pytest.approx((z * conjugue(z)).real)
        assert (z * conjugue(z)).imag == pytest.approx(0.0)


def test_le_conjugue_du_conjugue_rend_le_nombre() -> None:
    """Involution : deux fois le conjugue ne change rien."""
    for z in (3 + 4j, -2 + 0.5j, 1j):
        assert conjugue(conjugue(z)) == z


def test_la_forme_exponentielle_reconstruit_le_nombre() -> None:
    """`z = r e^{i theta}` : l'aller-retour doit etre exact.

    C'est tout l'interet de la notation du syllabus -- elle rend le
    produit et la puissance triviaux -- et elle ne vaut que si la
    reconstruction est fidele.
    """
    for z in (3 + 4j, -2 + 0.5j, 1j, -7 + 0j):
        r, theta = forme_exponentielle(z)
        assert r * cmath.exp(1j * theta) == pytest.approx(z)


def test_l_argument_concorde_avec_sympy() -> None:
    """Chemin independant : sympy calcule l'argument par sa propre voie."""
    for z in (3 + 4j, -2 + 0.5j, 1j, -7 + 0j):
        attendu = float(sp.arg(sp.sympify(z)))
        assert argument(z) == pytest.approx(attendu)


def test_les_racines_niemes_sont_au_nombre_de_n() -> None:
    """« Racines n-iemes d'un nombre complexe » : il y en a exactement `n`."""
    for n in (2, 3, 5, 8):
        assert len(racines_niemes(1 + 0j, n)) == n


def test_chaque_racine_nieme_elevee_a_n_redonne_le_nombre() -> None:
    """La verification qui ne recopie aucune formule.

    Si l'ecart angulaire entre racines etait faux -- `pi/n` au lieu de
    `2 pi/n`, l'erreur classique -- la moitie des racines ne redonnerait
    pas `z`, et ce test le dirait.
    """
    z = 16 * cmath.exp(1j * 0.7)
    for racine in racines_niemes(z, 4):
        assert racine**4 == pytest.approx(z, abs=1e-9)


def test_la_somme_des_racines_niemes_est_nulle() -> None:
    """Elles sont les sommets d'un polygone regulier centre sur l'origine.

    Vrai des que `n >= 2`, et c'est une garde bien plus forte qu'un
    controle de module : elle contraint les **phases** toutes ensemble.
    """
    for n in (2, 3, 5, 8):
        somme = sum(racines_niemes(81 - 12j, n))
        assert abs(somme) == pytest.approx(0.0, abs=1e-9)


def test_les_racines_niemes_ont_toutes_le_meme_module() -> None:
    """`|z|^(1/n)` pour chacune : elles ne different que par la phase."""
    z = 81 - 12j
    attendu = module(z) ** (1 / 5)
    for racine in racines_niemes(z, 5):
        assert module(racine) == pytest.approx(attendu)


def test_le_second_degre_dans_c_a_toujours_deux_racines() -> None:
    """« Dans C, le polynome x^2 + 1 possede deux racines », dit le syllabus.

    C'est la phrase qui motive tout le chapitre, et le test la prend au
    mot : meme la ou le discriminant reel est negatif, il y a deux
    racines, et elles sont conjuguees.
    """
    racines = racines_du_second_degre_dans_c(1, 0, 1)
    assert len(racines) == 2
    assert sorted(r.imag for r in racines) == pytest.approx([-1.0, 1.0])
    assert conjugue(racines[0]) == pytest.approx(racines[1])


def test_les_racines_dans_c_concordent_avec_sympy() -> None:
    """Sur les trois cas de discriminant : positif, nul, negatif."""
    for a, b, c in ((1, -3, 2), (1, -2, 1), (1, 0, 1), (2, 1, 3)):
        x = sp.Symbol("x")
        attendues = sorted(
            (complex(r) for r in sp.roots(a * x**2 + b * x + c, x)),
            key=lambda z: (z.real, z.imag),
        )
        obtenues = sorted(
            set(racines_du_second_degre_dans_c(a, b, c)),
            key=lambda z: (z.real, z.imag),
        )
        assert len(obtenues) == len(attendues)
        for obtenue, attendue in zip(obtenues, attendues, strict=True):
            assert obtenue == pytest.approx(attendue)


def test_une_racine_d_ordre_nul_est_refusee() -> None:
    """Il n'y a pas de racine zeroieme : l'equation `w^0 = z` n'a pas de sens."""
    with pytest.raises(ValueError, match="ordre de racine nul ou negatif"):
        racines_niemes(1 + 0j, 0)


def test_un_coefficient_dominant_nul_est_refuse_dans_c_aussi() -> None:
    """Sans terme en `x^2`, ce n'est pas un second degre, meme dans C."""
    with pytest.raises(ValueError, match="coefficient dominant nul"):
        racines_du_second_degre_dans_c(0, 1, 2)


def test_l_argument_de_zero_est_refuse() -> None:
    """Zero n'a pas d'argument : toutes les directions le representent.

    Le refus est preferable au zero par convention, qui ferait croire a
    une direction choisie.
    """
    with pytest.raises(ValueError, match="argument de zero"):
        argument(0j)
    assert module(0j) == pytest.approx(0.0)
    assert math.isfinite(module(0j))
