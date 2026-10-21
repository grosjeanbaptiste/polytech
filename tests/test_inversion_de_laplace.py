"""L'inversion de Laplace par le theoreme des residus.

Oracle : planches de la transformee de Laplace,
`MathematiquesIngenieur3/SLIDES/Slides_11_Transformees_partie_2_Transformee_de_Laplace.pdf`,
planche 12, « deuxieme theoreme d'inversion » : si `F` est holomorphe hors
d'un nombre fini de singularites et que `|F(z)| <= M / |z|^xi` loin de
l'origine, `f(t) = somme_k Res_(z = z_k) (F(z) e^(z t))` pour `t > 0`.
Second chemin : les fractions simples (deuxieme regle de Heaviside,
`polytech.maths.domain.laplace.heaviside`). Les examens : janvier 2023,
question 4 c (`z / ((z - 1)^2 + 4)`), examen blanc 2, question 3.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.laplace.heaviside import par_fractions_simples
from polytech.maths.domain.laplace.inversion import original_par_les_residus
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T


def test_les_deux_poles_simples_de_janvier_2023() -> None:
    """`e^t (cos 2t + sin(2t) / 2)`, poles `1 + 2i` et `1 - 2i`."""
    attendu = sp.exp(T) * (sp.cos(2 * T) + sp.sin(2 * T) / 2)
    original = original_par_les_residus(P / ((P - 1) ** 2 + 4))
    assert sp.simplify(original - attendu) == 0


def test_un_pole_double_rend_t_e_moins_t() -> None:
    assert sp.simplify(original_par_les_residus(1 / (P + 1) ** 2) - T * sp.exp(-T)) == 0


@pytest.mark.parametrize(
    "image",
    [
        -(P**3 - 4 * P**2 + 3 * P + 3) / (P * (P - 3) * (P - 2) * (P + 1)),
        (P**2 + 1) / (P**3 * (P - 2)),
    ],
)
def test_les_residus_et_les_fractions_simples_concordent(image: sp.Expr) -> None:
    """Deux chemins independants vers le meme original."""
    ecart = original_par_les_residus(image) - par_fractions_simples(image)
    assert sp.simplify(sp.expand_complex(ecart)) == 0


def test_l_examen_blanc_2_question_3_resout_son_equation() -> None:
    """`u(3) - 4 u'' + u' + 6u = -3`, `u(0) = -1`, `u'(0) = 0`, `u''(0) = -2`.

    L'image vient de la transformee des derivees ; l'original se verifie
    dans l'equation et ses trois conditions, sans repasser par l'image.
    """
    image = -(P**3 - 4 * P**2 + 3 * P + 3) / (P * (P - 3) * (P - 2) * (P + 1))
    u = original_par_les_residus(image)
    equation = sp.diff(u, T, 3) - 4 * sp.diff(u, T, 2) + sp.diff(u, T) + 6 * u
    assert sp.simplify(equation) == -3
    assert [sp.diff(u, T, k).subs(T, 0) for k in range(3)] == [-1, 0, -2]


def test_une_image_qui_ne_s_eteint_pas_a_l_infini_est_refusee() -> None:
    """`F = p / (p + 1)` tend vers 1 : la condition du theoreme manque."""
    with pytest.raises(ValueError, match="ne tend pas vers 0"):
        original_par_les_residus(P / (P + 1))


def test_une_infinite_de_singularites_est_refusee() -> None:
    with pytest.raises(ValueError, match="nombre infini ou inconnu"):
        original_par_les_residus(1 / (P * sp.sinh(P)))
