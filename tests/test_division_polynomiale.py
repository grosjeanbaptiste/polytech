"""Le chapitre IV du syllabus d'algebre : division et relations de Viete.

IV.1 Division polynomiale -- « quelque soit la forme du diviseur, on peut
effectuer la division sous forme d'un calcul ecrit », et « si le diviseur
est du type `x - a`, on peut utiliser la methode d'Horner ». IV.3
Utilisation du lien entre racines et coefficients.

Le syllabus donne **ses propres oracles**, et c'est ce qui rend ce
fichier solide :

* il detaille Horner sur `4x^3 - 7x^2 + 3x - 5` divise par `x - 2`, pas a
  pas -- « on multiplie le premier coefficient par la racine du diviseur,
  on ecrit le resultat sous le deuxieme coefficient et on effectue la
  somme » ;
* il calcule, pour un polynome de degre cinq, deux fonctions
  symetriques de ses racines « sans calculer les racines » -- ce qui est
  tout l'interet des relations de Viete sur un degre cinq, dont les
  racines n'ont pas d'expression par radicaux en general.

Le second exemple ne peut cependant **pas** servir d'oracle : le
polynome y est donne en image, que la conversion n'a pas rendue. En
deduire le polynome depuis les deux reponses annoncees serait circulaire.
Les relations de Viete sont donc verifiees contre des polynomes dont
sympy donne les racines, en prenant la propriete dans l'autre sens.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.polynomes import (
    degre,
    derivee_formelle,
    division_de_horner,
    division_euclidienne,
    est_racine_multiple,
    evalue,
    factorise_par,
    fonctions_symetriques_des_racines,
    pgcd,
    reste_en,
)


def test_horner_retrouve_le_quotient_et_le_reste_du_syllabus() -> None:
    """`4x^3 - 7x^2 + 3x - 5` par `x - 2` : l'exemple detaille du syllabus.

    Les coefficients du quotient sont 4, 1, 5 et le reste vaut 5, ce qui
    suit le calcul que le syllabus deroule colonne par colonne.
    """
    quotient, reste = division_de_horner((4, -7, 3, -5), 2)
    assert quotient == (4.0, 1.0, 5.0)
    assert reste == pytest.approx(5.0)


def test_le_reste_de_la_division_par_x_moins_a_vaut_le_polynome_en_a() -> None:
    """Theoreme du reste, et il se verifie par deux chemins independants.

    Horner mene une division ; l'evaluation ne divise rien. Les deux
    doivent concorder, et c'est ce qui etablit le theoreme plutot que de
    le reciter.
    """
    coefficients = (4, -7, 3, -5)
    for a in (-2, 0, 1, 2, 3.5):
        _, reste = division_de_horner(coefficients, a)
        assert reste == pytest.approx(evalue(coefficients, a))
        assert reste_en(coefficients, a) == pytest.approx(reste)


def test_la_division_reconstruit_le_dividende() -> None:
    """`P = Q D + R` : l'identite qui definit la division.

    Elle vaut pour un diviseur quelconque, et c'est la forme generale que
    le syllabus annonce avant de donner Horner comme raccourci.
    """
    dividende = (3, -6, 1, 5, -3, -4)
    diviseur = (1, 0, -2)
    quotient, reste = division_euclidienne(dividende, diviseur)
    x = sp.Symbol("x")

    def en_sympy(c: tuple[float, ...]) -> sp.Expr:
        return sum(v * x**i for i, v in enumerate(reversed(c)))

    reconstruit = sp.expand(
        en_sympy(quotient) * en_sympy(diviseur) + en_sympy(reste),
    )
    assert sp.expand(en_sympy(dividende) - reconstruit) == 0


def test_le_reste_a_un_degre_strictement_inferieur_au_diviseur() -> None:
    """Condition sans laquelle la division ne serait pas unique."""
    _, reste = division_euclidienne((3, -6, 1, 5, -3, -4), (1, 0, -2))
    assert degre(reste) < degre((1, 0, -2))


def test_factoriser_par_une_racine_abaisse_le_degre_d_un_cran() -> None:
    """Et le reste est nul, ce qui est la definition d'une racine."""
    coefficients = (1, -6, 11, -6)
    quotient = factorise_par(coefficients, 1)
    assert degre(quotient) == degre(coefficients) - 1
    assert evalue(coefficients, 1) == pytest.approx(0.0)


def test_factoriser_par_un_non_diviseur_est_refuse() -> None:
    """Le reste ne serait pas nul : ce n'est pas une factorisation."""
    with pytest.raises(ValueError, match="n'est pas une racine"):
        factorise_par((1, -6, 11, -6), 4)


def test_la_derivee_formelle_concorde_avec_sympy() -> None:
    """Elle sert a detecter les racines multiples, et rien d'autre ici."""
    x = sp.Symbol("x")
    for coefficients in ((1, -6, 11, -6), (4, -7, 3, -5), (1, 0, 0)):
        polynome = sum(v * x**i for i, v in enumerate(reversed(coefficients)))
        attendue = sp.Poly(sp.diff(polynome, x), x).all_coeffs()
        assert list(derivee_formelle(coefficients)) == pytest.approx(
            [float(c) for c in attendue],
        )


def test_une_racine_multiple_annule_aussi_la_derivee() -> None:
    """Le critere, et il distingue ce que l'evaluation seule ne voit pas.

    `(x-1)^2 (x-2)` et `(x-1)(x-2)(x-3)` s'annulent tous deux en 1, mais
    seul le premier y a une racine double -- et c'est la derivee qui le
    dit.
    """
    double = (1, -4, 5, -2)
    simple = (1, -6, 11, -6)
    assert evalue(double, 1) == pytest.approx(0.0)
    assert evalue(simple, 1) == pytest.approx(0.0)
    assert est_racine_multiple(double, 1)
    assert not est_racine_multiple(simple, 1)


def test_le_pgcd_de_deux_polynomes_divise_les_deux() -> None:
    """Propriete definissante, verifiee par les deux restes nuls."""
    premier = (1, -3, 2)
    second = (1, -1, 0)
    diviseur = pgcd(premier, second)
    for coefficients in (premier, second):
        _, reste = division_euclidienne(coefficients, diviseur)
        assert all(c == pytest.approx(0.0) for c in reste)


def test_les_fonctions_symetriques_concordent_avec_sympy() -> None:
    """Les relations de Viete, verifiees contre les racines que sympy calcule.

    L'exemple du syllabus ne peut pas servir d'oracle : il annonce deux
    valeurs -- `24/6 = 4` et `-1/6` -- pour un polynome de degre cinq
    donne en **image**, que la conversion n'a pas rendue. Deduire le
    polynome de ses propres reponses serait circulaire, et c'est
    exactement ce que ce depot refuse.

    La verification passe donc par des polynomes dont sympy donne les
    racines : on calcule les fonctions symetriques depuis les
    coefficients d'un cote, depuis les racines de l'autre, et les deux
    doivent concorder. C'est la propriete que le syllabus emploie --
    « sans calculer les racines » -- prise dans l'autre sens.
    """
    x = sp.Symbol("x")
    for coefficients in ((1, -6, 11, -6), (2, -3, -3, 2), (1, 0, -5, 0, 4)):
        polynome = sum(v * x**i for i, v in enumerate(reversed(coefficients)))
        racines = [complex(r) for r in sp.Poly(polynome, x).all_roots()]
        symetriques = fonctions_symetriques_des_racines(coefficients)
        assert symetriques[1] == pytest.approx(sum(racines).real, abs=1e-9)
        produit_total = 1.0 + 0j
        for racine in racines:
            produit_total *= racine
        degre_du_polynome = len(coefficients) - 1
        assert symetriques[degre_du_polynome] == pytest.approx(
            produit_total.real,
            abs=1e-9,
        )


def test_les_fonctions_symetriques_concordent_avec_les_racines_calculees() -> None:
    """Sur un polynome dont on connait les racines, les deux voies coincident.

    `(x-1)(x-2)(x-3)` : somme 6, somme des produits deux a deux 11,
    produit 6. Les relations de Viete doivent les redonner depuis les
    seuls coefficients.
    """
    symetriques = fonctions_symetriques_des_racines((1, -6, 11, -6))
    assert symetriques[1] == pytest.approx(6.0)
    assert symetriques[2] == pytest.approx(11.0)
    assert symetriques[3] == pytest.approx(6.0)


def test_un_diviseur_nul_est_refuse() -> None:
    """La division par le polynome nul n'existe pas."""
    with pytest.raises(ValueError, match="diviseur nul"):
        division_euclidienne((1, 2, 3), (0,))


def test_un_polynome_vide_est_refuse() -> None:
    """Sans coefficient, il n'y a pas de polynome a diviser."""
    with pytest.raises(ValueError, match="polynome sans coefficient"):
        degre(())
