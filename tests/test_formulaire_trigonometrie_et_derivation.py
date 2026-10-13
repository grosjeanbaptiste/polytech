"""Le formulaire de mathematique, sections 2 et 3 : trigonometrie et derivation.

Source : `Mathematique/FORMULAIRES/Formulaire_Aide_memoire.pdf`, pages 3
et 4, lues sur l'image de la page.

Aucune formule n'est crue sur parole : chaque identite trigonometrique
est eprouvee en des points ou rien ne s'annule, chaque regle de derivation
est comparee a la derivee que `sympy` calcule sur la fonction entiere, et
chaque ligne de la table est redérivee.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.integration.derivees_du_formulaire import (
    ALPHA,
    BASE,
    DERIVEES_DU_FORMULAIRE,
    derivee_du_formulaire,
)
from polytech.maths.domain.integration.formules_trigonometriques import (
    ADDITION,
    CARNOT,
    HYPERBOLIQUES,
    SIMPSON,
    SIMPSON_INVERSES,
    Y,
    en_exponentielles,
    en_produit,
)
from polytech.maths.domain.integration.regles_de_derivation import (
    de_la_composee,
    de_la_puissance,
    de_la_reciproque,
    de_la_somme,
    du_multiple,
    du_produit,
    du_quotient,
)
from polytech.maths.domain.symboles import VARIABLE as X

POINTS = ({X: 0.3, Y: 1.1}, {X: -0.7, Y: 0.4}, {X: 2.3, Y: -1.9})
TOUTES = {**ADDITION, **CARNOT, **SIMPSON, **SIMPSON_INVERSES, **HYPERBOLIQUES}


def _egales(gauche: sp.Expr, droite: sp.Expr, **valeurs: float) -> bool:
    return all(
        abs(complex((gauche - droite).subs({**point, **valeurs}).evalf())) < 1e-12
        for point in POINTS
    )


def test_le_formulaire_porte_quatorze_identites() -> None:
    tailles = [len(t) for t in (ADDITION, CARNOT, SIMPSON, SIMPSON_INVERSES)]
    assert tailles == [2, 2, 4, 3]
    assert len(HYPERBOLIQUES) == 3
    assert len(TOUTES) == 14


@pytest.mark.parametrize("gauche", list(TOUTES), ids=str)
def test_chaque_identite_tient_en_trois_points(gauche: sp.Expr) -> None:
    assert _egales(gauche, TOUTES[gauche])


def test_une_identite_fausse_serait_vue() -> None:
    assert not _egales(sp.sin(X + Y), sp.sin(X) * sp.cos(Y) - sp.cos(X) * sp.sin(Y))


@pytest.mark.parametrize("somme", list(SIMPSON), ids=str)
def test_simpson_transforme_une_somme_en_produit(somme: sp.Expr) -> None:
    produit = en_produit(somme)
    assert produit == SIMPSON[somme]
    assert produit.is_Mul


def test_simpson_vaut_pour_d_autres_angles() -> None:
    produit = en_produit(sp.sin(3 * X) + sp.sin(X))
    assert sp.simplify(produit - 2 * sp.sin(2 * X) * sp.cos(X)) == 0
    produit = en_produit(sp.cos(5 * X) - sp.cos(X))
    assert sp.simplify(produit + 2 * sp.sin(3 * X) * sp.sin(2 * X)) == 0


def test_une_somme_qui_n_est_pas_de_simpson_leve() -> None:
    with pytest.raises(ValueError, match="ni une somme ni une difference"):
        en_produit(sp.sin(X) + sp.cos(Y))
    with pytest.raises(ValueError, match="ni une somme ni une difference"):
        en_produit(sp.sin(X) + 2 * sp.sin(Y))


def test_les_hyperboliques_s_ecrivent_par_l_exponentielle() -> None:
    assert en_exponentielles(sp.cosh(X)) == (sp.exp(X) + sp.exp(-X)) / 2
    assert en_exponentielles(sp.sinh(X)) == (sp.exp(X) - sp.exp(-X)) / 2
    assert sp.expand(en_exponentielles(sp.cosh(X) ** 2 - sp.sinh(X) ** 2)) == 1


F, G = sp.sin(X) + 2, sp.exp(2 * X)


def _meme(obtenue: sp.Expr, fonction: sp.Expr) -> bool:
    return bool(sp.simplify(obtenue - sp.diff(fonction, X)) == 0)


def test_les_quatre_regles_algebriques() -> None:
    assert _meme(de_la_somme(F, G), F + G)
    assert _meme(du_multiple(7, F), 7 * F)
    assert _meme(du_produit(F, G), F * G)
    assert _meme(du_quotient(F, G), F / G)


def test_la_derivee_d_une_composee() -> None:
    assert _meme(de_la_composee(sp.sin(X), X**2 + 1), sp.sin(X**2 + 1))


def test_la_derivee_de_la_reciproque() -> None:
    # (arcsin)' = 1 / (cos o arcsin) = 1 / racine(1 - x^2).
    obtenue = de_la_reciproque(sp.sin(X), sp.asin(X))
    assert sp.simplify(obtenue - 1 / sp.sqrt(1 - X**2)) == 0
    assert _meme(de_la_reciproque(sp.exp(X), sp.log(X)), sp.log(X))


def test_une_fausse_reciproque_leve() -> None:
    with pytest.raises(ValueError, match="n'est pas la reciproque"):
        de_la_reciproque(sp.sin(X), sp.acos(X))


def test_la_derivee_de_f_puissance_g() -> None:
    assert _meme(de_la_puissance(F, G), F**G)
    assert sp.simplify(de_la_puissance(X, X) - X**X * (sp.log(X) + 1)) == 0


def test_un_quotient_par_zero_leve() -> None:
    with pytest.raises(ValueError, match="denominateur nul"):
        du_quotient(F, 0)


def test_la_table_porte_les_vingt_lignes_du_formulaire() -> None:
    assert len(DERIVEES_DU_FORMULAIRE) == 20


@pytest.mark.parametrize("fonction", list(DERIVEES_DU_FORMULAIRE), ids=str)
def test_chaque_ligne_de_la_table_se_retrouve_en_derivant(fonction: sp.Expr) -> None:
    valeurs = {X: sp.Rational(3, 10), ALPHA: sp.Rational(5, 2), BASE: 3}
    if fonction in (sp.acosh(X), sp.acoth(X)):
        valeurs[X] = sp.Rational(13, 10)
    attendue = sp.diff(fonction, X).subs(valeurs)
    for forme in DERIVEES_DU_FORMULAIRE[fonction]:
        assert abs(float((forme.subs(valeurs) - attendue).evalf())) < 1e-12


def test_la_tangente_et_la_cotangente_ont_deux_ecritures() -> None:
    assert len(DERIVEES_DU_FORMULAIRE[sp.tan(X)]) == 2
    assert len(DERIVEES_DU_FORMULAIRE[sp.cot(X)]) == 2
    assert derivee_du_formulaire(sp.tan(X)) == 1 / sp.cos(X) ** 2


def test_une_fonction_hors_de_la_table_leve() -> None:
    with pytest.raises(ValueError, match="n'est pas dans la table du formulaire"):
        derivee_du_formulaire(sp.sin(2 * X))
