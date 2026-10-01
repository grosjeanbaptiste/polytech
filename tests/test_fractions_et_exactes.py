"""Les deux gestes qui manquaient pour couvrir le chapitre.

La decomposition en fractions simples est le troisieme geste que le
fascicule nomme pour les primitives ; les equations exactes sont le
troisieme des quatre types d'EDO que les planches enumerent. Sans eux, se
declarer « couvert » sur ces deux notions aurait ete surestimer la
bibliotheque -- et c'est exactement ce que l'inventaire sert a empecher.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.exactes import FormeDifferentielle
from polytech.maths.domain.integration.fractions import FractionRationnelle
from polytech.maths.domain.symboles import VARIABLE as X

Y = sp.Symbol("y", real=True)


def test_la_decomposition_rend_les_fractions_une_a_une() -> None:
    """`1/(x^2-1)` donne deux fractions simples, et c'est ce qui se montre."""
    fraction = FractionRationnelle(numerateur=sp.Integer(1), denominateur=X**2 - 1)
    termes = fraction.termes()
    attendu = 2
    assert len(termes) == attendu
    assert sp.simplify(sum(termes) - 1 / (X**2 - 1)) == 0


def test_la_primitive_par_fractions_simples_se_verifie_en_la_derivant() -> None:
    fraction = FractionRationnelle(numerateur=sp.Integer(1), denominateur=X**2 - 1)
    derivee = sp.diff(fraction.primitive(), X)
    assert sp.simplify(derivee - 1 / (X**2 - 1)) == 0


def test_une_fraction_impropre_est_refusee_et_sa_partie_entiere_nommee() -> None:
    """Sauter la division donnerait une decomposition fausse, sans rien dire."""
    fraction = FractionRationnelle(numerateur=X**3, denominateur=X + 1)
    assert not fraction.est_propre()
    with pytest.raises(ValueError, match="fraction impropre"):
        fraction.termes()


def test_le_critere_de_schwarz_qualifie_une_forme_exacte() -> None:
    """`(2xy) dx + (x^2) dy` : dM/dy = 2x = dN/dx."""
    forme = FormeDifferentielle(facteur_en_x=2 * X * Y, facteur_en_y=X**2)
    assert forme.ecart_de_schwarz() == 0
    assert forme.est_exacte()


def test_le_potentiel_d_une_forme_exacte_a_les_bonnes_derivees() -> None:
    """Le controle intrinseque : `dg/dx = M` et `dg/dy = N`."""
    forme = FormeDifferentielle(facteur_en_x=2 * X * Y, facteur_en_y=X**2)
    potentiel = forme.potentiel()
    assert sp.simplify(sp.diff(potentiel, X) - 2 * X * Y) == 0
    assert sp.simplify(sp.diff(potentiel, Y) - X**2) == 0


def test_la_solution_est_implicite_comme_le_fascicule_l_ecrit() -> None:
    """« Elle admet alors comme solution implicite g(x, y) = c »."""
    forme = FormeDifferentielle(facteur_en_x=2 * X * Y, facteur_en_y=X**2)
    constante = sp.Symbol("c", real=True)
    assert forme.solution_implicite(constante).rhs == constante


def test_une_forme_non_exacte_est_refusee_et_son_ecart_nomme() -> None:
    """Integrer une forme inexacte rendrait un potentiel qui n'en est pas un."""
    forme = FormeDifferentielle(facteur_en_x=Y, facteur_en_y=X**2)
    assert not forme.est_exacte()
    with pytest.raises(ValueError, match="non exacte"):
        forme.potentiel()
