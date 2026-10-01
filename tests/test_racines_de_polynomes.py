"""Le chapitre IV.2 du syllabus d'algebre : les racines d'un polynome.

Trois methodes, que le syllabus donne en clair et que rien ne portait :

* le **second degre** -- `Delta = b^2 - 4ac`, racines `(-b +- racine)/2a`,
  et « si on travaille dans R, il y a des racines si et seulement si
  Delta >= 0 » ;
* les **bicarrees** -- « poser `x^2 = y`, resoudre `ay^2 + by + c = 0`,
  chaque `y` trouve donne deux valeurs de `x` (sauf `y = 0` : `x = 0`).
  Si on travaille dans R, seuls les `y` positifs sont a retenir » ;
* les **polynomes a coefficients entiers** -- « s'il existe une racine
  rationnelle `p/q`, alors `p` divise `a_0` et `q` divise `a_n` ».

Chaque resultat est verifie **par sympy**, c'est-a-dire par un chemin qui
ne partage aucune ligne avec le module : si la formule du syllabus etait
mal transcrite, les deux ne concorderaient pas.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.racines import (
    candidats_entiers,
    candidats_rationnels,
    discriminant,
    est_racine,
    racines_du_bicarre,
    racines_reelles_du_second_degre,
)


def test_le_discriminant_est_celui_du_syllabus() -> None:
    """`Delta = b^2 - 4ac`, et sympy le confirme par son propre chemin."""
    for a, b, c in ((1, -3, 2), (2, 4, 2), (1, 0, 1), (-3, 5, -1)):
        x = sp.Symbol("x")
        attendu = sp.discriminant(a * x**2 + b * x + c, x)
        assert discriminant(a, b, c) == attendu


def test_les_racines_du_second_degre_concordent_avec_sympy() -> None:
    """Et le test porte sur les **trois** cas : deux racines, une, aucune."""
    for a, b, c in ((1, -3, 2), (1, -2, 1), (1, 0, 1)):
        x = sp.Symbol("x")
        reelles = sorted(
            float(r) for r in sp.solve(a * x**2 + b * x + c, x) if r.is_real
        )
        assert sorted(racines_reelles_du_second_degre(a, b, c)) == reelles


def test_un_discriminant_negatif_ne_donne_aucune_racine_reelle() -> None:
    """« Il y a des racines si et seulement si Delta >= 0 », dit le syllabus.

    Le module rend donc un tuple **vide** plutot que de lever : l'absence
    de racine reelle est un resultat, pas une erreur. C'est d'ailleurs ce
    qui motive le chapitre I du meme syllabus -- `x^2 + 1` n'a pas de
    racine reelle, et c'est pourquoi on construit les complexes.
    """
    assert racines_reelles_du_second_degre(1, 0, 1) == ()
    assert discriminant(1, 0, 1) < 0


def test_une_racine_double_est_rendue_une_seule_fois() -> None:
    """`(x-1)^2` a une racine, de multiplicite deux.

    La rendre deux fois ferait croire a deux racines distinctes, ce qui
    change la forme de la solution d'une equation differentielle -- la ou
    ce resultat sert ensuite.
    """
    assert racines_reelles_du_second_degre(1, -2, 1) == (1.0,)


def test_le_bicarre_remonte_a_quatre_racines_par_la_substitution() -> None:
    """`x^4 - 5x^2 + 4` : les `y` valent 1 et 4, donc `x` vaut +-1 et +-2.

    sympy resout le quatrieme degre directement, sans substitution : la
    concordance verifie donc la methode du syllabus, et non seulement son
    arithmetique.
    """
    x = sp.Symbol("x")
    attendues = sorted(float(r) for r in sp.solve(x**4 - 5 * x**2 + 4, x) if r.is_real)
    assert sorted(racines_du_bicarre(1, -5, 4)) == attendues
    assert sorted(racines_du_bicarre(1, -5, 4)) == [-2.0, -1.0, 1.0, 2.0]


def test_un_bicarre_a_y_negatif_perd_ces_racines_dans_les_reels() -> None:
    """« Seuls les y positifs sont a retenir », dit le syllabus.

    `x^4 + 3x^2 - 4` donne `y = 1` et `y = -4` : le second ne rend aucun
    `x` reel, et il ne reste que deux racines au lieu de quatre.
    """
    assert sorted(racines_du_bicarre(1, 3, -4)) == [-1.0, 1.0]


def test_un_y_nul_ne_donne_qu_une_racine() -> None:
    """« Sauf y = 0 : x = 0 » -- la parenthese du syllabus, et elle compte.

    `x^4 - x^2` donne `y = 0` et `y = 1`. Le premier rend la seule racine
    zero, le second en rend deux : trois racines, non quatre.
    """
    assert sorted(racines_du_bicarre(1, -1, 0)) == [-1.0, 0.0, 1.0]


def test_les_candidats_rationnels_contiennent_les_racines_reelles() -> None:
    """`p` divise `a_0` et `q` divise `a_n` : le theoreme est une **borne**.

    Il ne donne pas les racines mais la liste finie ou les chercher, et
    c'est tout son interet. Le test verifie donc l'inclusion dans le bon
    sens : toute racine rationnelle est candidate.
    """
    coefficients = (2, -3, -3, 2)
    x = sp.Symbol("x")
    polynome = sum(c * x**i for i, c in enumerate(reversed(coefficients)))
    racines = [sp.Rational(r) for r in sp.solve(polynome, x) if r.is_rational]
    candidats = candidats_rationnels(coefficients)
    for racine in racines:
        assert float(racine) in candidats


def test_les_candidats_entiers_sont_les_diviseurs_du_terme_constant() -> None:
    """« S'il existe une racine entiere x, alors x divise a_0 ».

    Pour `x^3 - 6x^2 + 11x - 6`, dont les racines sont 1, 2 et 3, les
    candidats sont les diviseurs de 6 -- positifs et negatifs, « lorsqu'on
    parle de diviseurs » le syllabus les prend des deux signes.
    """
    candidats = candidats_entiers((1, -6, 11, -6))
    assert set(candidats) == {-6, -3, -2, -1, 1, 2, 3, 6}
    for racine in (1, 2, 3):
        assert racine in candidats


def test_une_racine_se_verifie_par_evaluation() -> None:
    """Le theoreme borne la recherche ; l'evaluation tranche."""
    coefficients = (1, -6, 11, -6)
    assert est_racine(coefficients, 2.0)
    assert not est_racine(coefficients, 4.0)


def test_un_terme_constant_nul_est_refuse_par_le_theoreme() -> None:
    """Si `a_0 = 0`, tout entier le divise -- la borne ne borne plus rien.

    Et le cas est facile a traiter autrement : un terme constant nul veut
    dire que zero est racine, donc qu'on peut factoriser par `x` et
    appliquer le theoreme au quotient. Le refus renvoie a ce chemin
    plutot que de rendre une liste infinie.
    """
    with pytest.raises(ValueError, match="terme constant nul"):
        candidats_entiers((1, -3, 2, 0))
    with pytest.raises(ValueError, match="terme constant nul"):
        candidats_rationnels((1, -3, 2, 0))
