"""Discuter un systeme a parametre quand son determinant est toujours nul.

Les trois examens de mathematique posent la question « resoudre et
discuter en fonction des valeurs du parametre ». `discuter` examine les
valeurs qui annulent le determinant ; or le systeme de janvier 2025 a
deux colonnes egales, et son determinant est nul pour **tout** `a` :
`discuter` n'y trouve aucune valeur a examiner. C'est la condition de
compatibilite -- `rang(A) = rang(A | b)`, planche 19 de
`Mathematique/SLIDES/Slides_01_Algebre_lineaire_et_integrales.pdf` -- qui
tranche alors.

Oracle : le corrige officiel de janvier 2025
(`Mathematique/EXAMENS/Exam_2025_Janvier_Corrige.pdf`, page 2) -- pas
de solution si `a` n'est ni 0 ni 1/2 ; `{(1, y, -y)}` si `a = 0` ;
`{(1/2, y, -y)}` si `a = 1/2`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.algebre.determinants import Determinant
from polytech.maths.domain.algebre.discussion import discuter
from polytech.maths.domain.algebre.discussion_par_le_rang import discuter_par_le_rang
from polytech.maths.domain.algebre.solutions import (
    PARAMETRES,
    Nature,
    solution_generale,
)
from polytech.maths.domain.algebre.systemes import SystemeLineaire

A, M = sp.symbols("a m", real=True)
ALPHA = PARAMETRES[0]
DEMI = sp.Rational(1, 2)


def _systeme(lignes: list[list[object]]) -> SystemeLineaire:
    """La derniere colonne est le second membre."""
    complete = sp.Matrix(lignes)
    return SystemeLineaire(complete[:, :-1], complete[:, -1])


JANVIER_2025 = _systeme(
    [
        [A, 1 - A, 1 - A, A**2],
        [A, 1 + A, 1 + A, A - A**2],
        [1, 1, 1, 1 - A],
    ]
)
"""Aussi le systeme (6) de janvier 2021, propose au choix."""

JANVIER_2022 = _systeme([[1, 1, -1, 1], [3, 1, -1, 1], [1, -2, 2, M]])
"""Lu sur l'image du PDF : la conversion n'en a rendu que le texte."""

EXERCICE_20 = _systeme(
    [
        [A - 2, A - 2, 0, 2 * A - 2],
        [2, A - 1, A - 3, A + 5],
        [3 - A, 2 * A - 3, 2 * A - 2, A - 2],
    ]
)
"""Janvier 99, au syllabus d'algebre : le determinant s'annule en 2, 3 et 4."""


def test_le_determinant_de_janvier_2025_est_nul_pour_tout_a() -> None:
    """Deux colonnes egales : la regle de Cramer n'a rien a dire, jamais."""
    assert sp.expand(Determinant(JANVIER_2025.matrice).par_cofacteurs(0)) == 0
    assert discuter(JANVIER_2025, A) == {}


def test_janvier_2025_n_est_compatible_qu_en_zero_et_un_demi() -> None:
    discussion = discuter_par_le_rang(JANVIER_2025, A)
    assert discussion.generique is Nature.IMPOSSIBLE
    assert discussion.particulieres == {
        sp.Integer(0): Nature.INDETERMINE,
        DEMI: Nature.INDETERMINE,
    }


def test_les_deux_solutions_de_janvier_2025_sont_celles_du_corrige() -> None:
    """`{(1, y, -y)}` et `{(1/2, y, -y)}` : le corrige prend `y` libre, ici `z`."""
    en_zero = discuter_par_le_rang(JANVIER_2025, A).en(sp.Integer(0))
    en_un_demi = discuter_par_le_rang(JANVIER_2025, A).en(DEMI)
    assert solution_generale(en_zero) == (1, -ALPHA, ALPHA)
    assert solution_generale(en_un_demi) == (DEMI, -ALPHA, ALPHA)


def test_janvier_2022_n_est_compatible_qu_en_moins_deux() -> None:
    """Second chemin, a la main, par substitution.

    Les deux premieres lignes donnent `x = 0` et `y - z = 1` ; la troisieme
    vaut alors `-2 (y - z) = -2`.
    """
    discussion = discuter_par_le_rang(JANVIER_2022, M)
    assert discussion.generique is Nature.IMPOSSIBLE
    assert discussion.particulieres == {sp.Integer(-2): Nature.INDETERMINE}
    assert solution_generale(discussion.en(sp.Integer(-2))) == (0, ALPHA + 1, ALPHA)


def test_un_systeme_regulier_se_discute_comme_par_le_determinant() -> None:
    """Le lien entre les deux discussions : memes valeurs, memes natures."""
    discussion = discuter_par_le_rang(EXERCICE_20, A)
    assert discussion.generique is Nature.UNIQUE
    assert discussion.particulieres == discuter(EXERCICE_20, A)


def test_un_systeme_sans_parametre_n_a_aucun_cas_particulier() -> None:
    sans = _systeme([[1, 1, 3], [1, -1, 1]])
    discussion = discuter_par_le_rang(sans, A)
    assert (discussion.generique, discussion.particulieres) == (Nature.UNIQUE, {})


def test_une_valeur_ou_le_rang_tombe_sans_changer_la_nature_est_retenue() -> None:
    """La nature ne change pas, l'ensemble des solutions si.

    `a x = 0`, `a y = 0`, trois inconnues : une droite, sauf en `a = 0` ou
    c'est l'espace entier.
    """
    chute = discuter_par_le_rang(_systeme([[A, 0, 0, 0], [0, A, 0, 0]]), A)
    assert chute.generique is Nature.INDETERMINE
    assert chute.particulieres == {sp.Integer(0): Nature.INDETERMINE}


def test_une_valeur_ou_rien_ne_change_n_est_pas_un_cas_particulier() -> None:
    """`a x + a y = 0`, `x + y = 0` : le rang vaut 1 partout, meme en `a = 0`."""
    assert discuter_par_le_rang(_systeme([[A, A, 0], [1, 1, 0]]), A).particulieres == {}


def test_une_racine_complexe_n_est_pas_une_valeur_du_parametre() -> None:
    """`(a^2 + 1) x = 1` : rien ne s'annule sur les reels.

    Le symbole n'est pas declare reel, expres : `i` et `-i` sortent alors de
    la resolution, et c'est la discussion qui doit les ecarter.
    """
    libre = sp.Symbol("a")
    discussion = discuter_par_le_rang(_systeme([[libre**2 + 1, 1]]), libre)
    assert (discussion.generique, discussion.particulieres) == (Nature.UNIQUE, {})


def test_une_seule_equation_se_discute_sur_ses_mineurs_d_ordre_un() -> None:
    """`a x = 1` : impossible en `a = 0`, et aucun determinant d'ordre 2 a annuler."""
    discussion = discuter_par_le_rang(_systeme([[A, 1]]), A)
    assert discussion.generique is Nature.UNIQUE
    assert discussion.particulieres == {sp.Integer(0): Nature.IMPOSSIBLE}
