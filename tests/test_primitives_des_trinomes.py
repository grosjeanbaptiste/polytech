"""Formulaire, 5.4 et 5.7 (regles 3 a 7) : les radicaux d'un trinome.

Sources :

- `Mathematique/FORMULAIRES/Formulaire_Aide_memoire.pdf`, pages 6 et 7 :
  les cinq « cas particuliers » du 5.4, et les regles 3 a 7 du 5.7 ;
- `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, pages 56 a 59
  (document de travail, 3e partie, pages 5 a 8) : les exemples detailles
  des cas 3, 4 et 8, et la solution donnee page 59 pour
  `(2x + 1) / racine(4x^2 - 2x + 1)` ; pages 44, 47 et 51 pour les
  fractions sur un trinome sans racine reelle.

Une primitive ne s'ecrit pas d'une seule facon : chacune est controlee
par sa derivee, et celle du cours aussi.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.integration.changements_trigonometriques import (
    ANGLE,
    par_la_tangente,
    par_le_cosinus,
    par_le_sinus,
)
from polytech.maths.domain.integration.trinomes import (
    CAS_PARTICULIERS,
    PARAMETRE,
    completer_le_carre,
    decomposer_le_numerateur,
    primitive_de_l_inverse,
    primitive_de_la_racine,
    primitive_sur_la_racine,
)
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational
sqrt = sp.sqrt
POINTS = (R(3, 2), R(7, 3), R(23, 10))


def _derive_en(primitive: sp.Expr, integrande: sp.Expr, points: tuple) -> bool:
    ecart = sp.diff(primitive, X) - integrande
    return all(abs(complex(ecart.subs(X, p).evalf())) < 1e-12 for p in points)


def test_les_cinq_lignes_du_formulaire_font_six_cas() -> None:
    # La premiere ligne, `racine(x^2 +- p^2)`, en porte deux.
    assert len(CAS_PARTICULIERS) == 6


@pytest.mark.parametrize("integrande", list(CAS_PARTICULIERS), ids=str)
def test_chaque_cas_particulier_se_retrouve_en_derivant(integrande: sp.Expr) -> None:
    primitive = CAS_PARTICULIERS[integrande].subs(PARAMETRE, R(5, 4))
    assert _derive_en(primitive, integrande.subs(PARAMETRE, R(5, 4)), POINTS)


def test_completer_le_carre_du_cas_3() -> None:
    # 5x^2 - 4x + 1 = 5 (x - 2/5)^2 + 1/5.
    assert completer_le_carre(5 * X**2 - 4 * X + 1) == (5, R(2, 5), R(1, 5))
    assert completer_le_carre(5 - 4 * X - X**2) == (-1, -2, 9)


def test_un_premier_degre_n_a_pas_de_carre_a_completer() -> None:
    with pytest.raises(ValueError, match="n'est pas un trinome du second degre"):
        completer_le_carre(3 * X + 1)


def test_le_cas_3_la_racine_d_un_trinome() -> None:
    trinome = 5 * X**2 - 4 * X + 1
    u = 5 * (X - R(2, 5))
    du_cours = sqrt(5) / 25 * (u / 2 * sqrt(u**2 + 1) + sp.log(u + sqrt(u**2 + 1)) / 2)
    assert _derive_en(du_cours, sqrt(trinome), POINTS)
    assert _derive_en(primitive_de_la_racine(trinome), sqrt(trinome), POINTS)


def test_la_racine_d_un_trinome_tourne_vers_le_bas() -> None:
    trinome = 5 - 4 * X - X**2
    points = (R(-1, 2), R(1, 3), R(-3, 1))
    assert _derive_en(primitive_de_la_racine(trinome), sqrt(trinome), points)


def test_decomposer_le_numerateur_des_pages_47_et_57() -> None:
    assert decomposer_le_numerateur(5 * X + 4, X**2 - X + 1) == (R(5, 2), R(13, 2))
    assert decomposer_le_numerateur(X + 3, 5 - 4 * X - X**2) == (R(-1, 2), 1)


def test_le_cas_4_de_la_page_57() -> None:
    trinome = 5 - 4 * X - X**2
    du_cours = -sqrt(trinome) + sp.asin((X + 2) / 3)
    obtenue = primitive_sur_la_racine(X + 3, trinome)
    assert sp.simplify(obtenue - du_cours) == 0
    assert _derive_en(obtenue, (X + 3) / sqrt(trinome), (R(-1, 2), R(1, 3)))


def test_la_solution_de_la_page_59() -> None:
    trinome = 4 * X**2 - 2 * X + 1
    du_cours = sqrt(trinome) / 2 + R(3, 4) * sp.asinh(4 / sqrt(3) * (X - R(1, 4)))
    obtenue = primitive_sur_la_racine(2 * X + 1, trinome)
    assert _derive_en(du_cours, (2 * X + 1) / sqrt(trinome), POINTS)
    assert _derive_en(obtenue, (2 * X + 1) / sqrt(trinome), POINTS)
    assert (
        abs(float((obtenue - du_cours).subs(X, 2) - (obtenue - du_cours).subs(X, 5)))
        < 1e-12
    )


def test_une_racine_sous_laquelle_tout_est_negatif_leve() -> None:
    with pytest.raises(ValueError, match="jamais positif"):
        primitive_de_la_racine(-(X**2) - 1)


def test_l_inverse_d_un_trinome_sans_racine_reelle() -> None:
    # Page 44 : 3 / (x^2 + x + 1) ; page 51 : 1 / (x^2 + 4x + 5).
    attendue = 2 * sqrt(3) * sp.atan(2 / sqrt(3) * (X + R(1, 2)))
    assert sp.simplify(3 * primitive_de_l_inverse(X**2 + X + 1) - attendue) == 0
    assert primitive_de_l_inverse(X**2 + 4 * X + 5) == sp.atan(X + 2)


def test_un_trinome_qui_a_des_racines_se_decompose_autrement() -> None:
    with pytest.raises(ValueError, match="racines reelles"):
        primitive_de_l_inverse(X**2 - 4)


def test_la_regle_5_x_egale_b_sur_a_tangente() -> None:
    # racine(4x^2 + 9) avec x = (3/2) tg t : 3 / cos t, et dx = 3 dt / (2 cos^2 t).
    en_t = par_la_tangente(sqrt(4 * X**2 + 9), 2, 3)
    assert sp.simplify(en_t - R(9, 2) / sp.cos(ANGLE) ** 3) == 0


def test_la_regle_6_le_cas_8_de_la_page_59() -> None:
    # x = 2 / cos t : dx / (x racine(x^2 - 4)) devient dt / 2.
    en_t = par_le_cosinus(1 / (X * sqrt(X**2 - 4)), 1, 2)
    assert en_t == R(1, 2)
    primitive = sp.integrate(en_t, ANGLE).subs(ANGLE, sp.acos(2 / X))
    assert primitive == sp.acos(2 / X) / 2
    assert _derive_en(primitive, 1 / (X * sqrt(X**2 - 4)), (R(5, 2), R(7, 2)))


def test_la_regle_7_x_egale_b_sur_a_sinus() -> None:
    # racine(1 - x^2) avec x = sin t : cos^2 t.
    en_t = par_le_sinus(sqrt(1 - X**2), 1, 1)
    assert sp.simplify(en_t - sp.cos(ANGLE) ** 2) == 0


def test_un_changement_qui_laisse_un_radical_leve() -> None:
    with pytest.raises(ValueError, match="laisse un radical"):
        par_le_sinus(sqrt(1 + X**2), 1, 1)


def test_un_coefficient_nul_n_est_pas_un_changement() -> None:
    with pytest.raises(ValueError, match="strictement positifs"):
        par_la_tangente(sqrt(X**2 + 1), 0, 1)


def test_un_carre_parfait_n_a_pas_de_radical_a_primitiver() -> None:
    with pytest.raises(ValueError, match="carre parfait"):
        primitive_de_la_racine(X**2 - 2 * X + 1)


def test_la_racine_d_un_trinome_qui_a_deux_racines() -> None:
    trinome = X**2 - 2 * X - 3
    points = (R(7, 2), sp.Integer(5), R(13, 3))
    assert _derive_en(primitive_de_la_racine(trinome), sqrt(trinome), points)


def test_l_inverse_d_un_trinome_de_coefficient_dominant_deux() -> None:
    trinome = 2 * X**2 + 2 * X + 1
    assert _derive_en(primitive_de_l_inverse(trinome), 1 / trinome, POINTS)


def test_la_regle_6_avec_un_coefficient_devant_x() -> None:
    # x = 3 / (2 cos t) : dx / (x racine(4x^2 - 9)) devient dt / 3.
    assert par_le_cosinus(1 / (X * sqrt(4 * X**2 - 9)), 2, 3) == R(1, 3)


def test_les_regles_6_et_7_portent_le_rapport_b_sur_a_dans_dx() -> None:
    en_t = par_le_cosinus(sqrt(4 * X**2 - 9), 2, 3)
    assert sp.simplify(en_t - R(9, 2) * sp.sin(ANGLE) ** 2 / sp.cos(ANGLE) ** 3) == 0
    en_t = par_le_sinus(sqrt(9 - 4 * X**2), 2, 3)
    assert sp.simplify(en_t - R(9, 2) * sp.cos(ANGLE) ** 2) == 0
