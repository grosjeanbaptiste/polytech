"""Le residu lu sur la serie de Laurent : le coefficient d'indice -1.

Oracles : `MathematiquesIngenieur3/SYLLABUS/Syllabus_Feuillet_1_Analyse_complexe.pdf`,
section 9.2, methode 5, pages 28 et 29 (lues sur la page rendue).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.analyse_complexe.laurent import residu_par_laurent

Z = sp.Symbol("z")


@pytest.mark.parametrize("n", [2, 3, 6])
def test_un_nombre_fini_de_termes_singuliers(n: int) -> None:
    """Page 28, (115) et (116) : `Res((e^z - 1) / z^n, 0) = 1 / (n - 1)!`."""
    obtenu = residu_par_laurent((sp.exp(Z) - 1) / Z**n, Z, 0)
    assert obtenu == 1 / sp.factorial(n - 1)


def test_une_infinite_de_termes_singuliers() -> None:
    """Page 29 : `sin(1/z) = 1/z - 1/(3! z3) + ...`, donc `Res = 1`."""
    assert residu_par_laurent(sp.sin(1 / Z), Z, 0) == 1


def test_une_singularite_essentielle_hors_de_l_origine() -> None:
    """`z2 e^(1/(z-1))` en 1 : `e^(1/w) (1 + w)^2`, coefficient de `1/w`.

    `1/6 + 2/2 + 1 = 13/6` -- les termes `k = 3`, `2` et `1` de l'exponentielle.
    Garde contre une bibliotheque de calcul formel qui rend 0 ici.
    """
    obtenu = residu_par_laurent(Z**2 * sp.exp(1 / (Z - 1)), Z, 1)
    assert obtenu == sp.Rational(13, 6)


def test_un_point_regulier_a_un_residu_nul() -> None:
    """Page 27, methode 1 : un developpement de Taylor n'a pas de terme en `1/z`."""
    assert residu_par_laurent(sp.cos(Z), Z, 0) == 0


def test_une_serie_que_l_on_ne_sait_pas_ecrire_est_refusee() -> None:
    """Ni fini d'un cote ni de l'autre : la methode ne conclut pas, et le dit."""
    with pytest.raises(ValueError, match="serie de Laurent"):
        residu_par_laurent(sp.exp(1 / Z) / (1 - Z), Z, 0)
