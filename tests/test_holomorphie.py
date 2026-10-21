"""L'holomorphie par Cauchy-Riemann, telle que le cours la pose.

Oracles :

* syllabus de Mathematiques III,
  `MathematiquesIngenieur3/SYLLABUS/Syllabus_Feuillet_1_Analyse_complexe.pdf`,
  page 6 : proposition 9 (`u_x = v_y`, `u_y = -v_x`), proposition 10
  (`f' = d_x f = -i d_y f`), proposition 11 (`u` et `v` harmoniques) ;
* serie 2, `MathematiquesIngenieur3/EXS/Exs_02_Fonctions_holomorphes.pdf` :
  exercice 1 (`z / conj(z)` n'est pas holomorphe), exercice 2 a
  (`u_x^2 + u_y^2 = v_x^2 + v_y^2`), exercice 8 b (Cauchy-Riemann pour `sin z`
  et `cosh z`) ;
* examen de janvier 2026, question 1 b) : `cos x cosh y - i sin x sinh y`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.analyse_complexe.holomorphie import (
    X,
    Y,
    Z,
    derivee_complexe,
    est_holomorphe,
    parties_reelle_et_imaginaire,
    sont_harmoniques,
    verifie_cauchy_riemann,
)


def test_les_parties_de_z_carre() -> None:
    assert parties_reelle_et_imaginaire(Z**2) == (X**2 - Y**2, 2 * X * Y)


def test_sin_et_cosh_verifient_cauchy_riemann() -> None:
    """Serie 2, exercice 8 b."""
    for f in (sp.sin(Z), sp.cosh(Z)):
        assert verifie_cauchy_riemann(*parties_reelle_et_imaginaire(f))


def test_z_sur_son_conjugue_n_est_pas_holomorphe() -> None:
    """Serie 2, exercice 1."""
    assert not est_holomorphe(Z / sp.conjugate(Z))


def test_le_conjugue_et_le_module_au_carre_ne_le_sont_pas() -> None:
    """Les contre-exemples que les examens de 2023 demandent de proposer."""
    assert not est_holomorphe(sp.conjugate(Z))
    assert not est_holomorphe(Z * sp.conjugate(Z))


def test_la_fonction_de_janvier_2026_est_holomorphe() -> None:
    """Question 1 b) : c'est cos z ecrit en x et y."""
    u = sp.cos(X) * sp.cosh(Y)
    v = -sp.sin(X) * sp.sinh(Y)
    assert verifie_cauchy_riemann(u, v)


def test_une_fonction_qui_ne_verifie_qu_une_relation_est_refusee() -> None:
    """u_x = v_y seule ne suffit pas : u = x, v = y + x."""
    assert not verifie_cauchy_riemann(X, Y + X)


def test_la_derivee_par_la_proposition_10() -> None:
    u, v = parties_reelle_et_imaginaire(Z**3)
    attendue = sp.expand(3 * (X + sp.I * Y) ** 2)
    assert sp.expand(derivee_complexe(u, v)) == attendue


def test_les_parties_d_une_fonction_holomorphe_sont_harmoniques() -> None:
    """Proposition 11 ; et x^2 + y^2 ne l'est pas."""
    assert sont_harmoniques(*parties_reelle_et_imaginaire(sp.exp(Z)))
    assert not sont_harmoniques(X**2 + Y**2, sp.Integer(0))


def test_les_gradients_de_u_et_v_ont_meme_norme() -> None:
    """Serie 2, exercice 2 a, sur sin z."""
    u, v = parties_reelle_et_imaginaire(sp.sin(Z))
    gradient = sp.diff(u, X) ** 2 + sp.diff(u, Y) ** 2
    assert sp.simplify(gradient - sp.diff(v, X) ** 2 - sp.diff(v, Y) ** 2) == 0


def test_les_deux_parties_doivent_etre_harmoniques() -> None:
    """La paire u = x, v = x^2 + y^2 ne l'est pas : v ne l'est pas."""
    assert not sont_harmoniques(X, X**2 + Y**2)
