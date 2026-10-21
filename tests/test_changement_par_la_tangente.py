"""Notes de 2011, pages 53 et 54 : poser `t = tg x`, et le mener au bout.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`.
`1 / (1 + cos^2 x)` est paire en `sin x` et en `cos x` ; on pose `t = tg x`
et l'on primitive `1 / (2 + t^2)` (page 53) ; la primitive vaut
`(1/sqrt 2) arctg(tg x / sqrt 2)` (page 54). Le formulaire (5.6, regle 3)
donne la meme regle.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.integration.regles_de_parite import (
    Changement,
    changement_conseille,
)
from polytech.maths.domain.integration.tangente import TANGENTE, en_tangente
from polytech.maths.domain.symboles import VARIABLE as X

S, C = sp.sin(X), sp.cos(X)
EXEMPLE = 1 / (1 + C**2)


def test_l_exemple_des_notes_designe_la_tangente() -> None:
    assert changement_conseille(EXEMPLE) is Changement.TANGENTE


def test_l_exemple_des_notes_pages_53_est_impair_en_sinus() -> None:
    assert changement_conseille(S * C**2 / (1 + S**2)) is Changement.COSINUS


def test_la_tangente_rend_l_integrande_des_notes() -> None:
    assert sp.simplify(en_tangente(EXEMPLE) - 1 / (2 + TANGENTE**2)) == 0


def test_la_primitive_des_notes_se_retrouve() -> None:
    """Second chemin : deriver la primitive de la page 54."""
    primitive = sp.atan(sp.tan(X) / sp.sqrt(2)) / sp.sqrt(2)
    assert sp.simplify(sp.diff(primitive, X) - EXEMPLE) == 0
    en_t = sp.integrate(en_tangente(EXEMPLE), TANGENTE)
    assert sp.simplify(en_t.subs(TANGENTE, sp.tan(X)) - primitive) == 0


def test_un_produit_sinus_cosinus_devient_rationnel_en_t() -> None:
    """`sin x cos x / (1 + sin^2 x)` : `t / ((1 + t^2)(1 + 2t^2))`."""
    attendu = TANGENTE / ((1 + TANGENTE**2) * (1 + 2 * TANGENTE**2))
    assert sp.simplify(en_tangente(S * C / (1 + S**2)) - attendu) == 0


def test_la_tangente_refuse_un_integrande_qui_n_est_pas_pair() -> None:
    with pytest.raises(ValueError, match="paire"):
        en_tangente(S / (1 + C**2))
