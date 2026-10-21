"""La transformee de Fourier d'une derivee, dans la convention du cours.

Oracle : planches de la transformee de Fourier,
`MathematiquesIngenieur3/SLIDES/Slides_11_Transformees_partie_1_Transformee_de_Fourier.pdf`,
planche 10, « les formules de derivation » : `F(f')(s) = i s F(f)(s)` si
`f` est continue et `f'` integrable. La formule ne depend pas du facteur de
normalisation, et `polytech.maths.domain.fourier.proprietes.de_la_derivee`
la porte pour toutes les conventions ; ce test la confronte a
`F(f')` calcule par les residus -- l'integrale directe, elle, ne se ferme
pas sous sympy et leve. Le cas de
l'examen d'aout 2023, question 2.3 : `g = f'`, `f = 1 / (1 + x^2)`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.analyse_complexe.jordan import integrale_oscillante
from polytech.maths.domain.fourier.proprietes import de_la_derivee
from polytech.maths.domain.fourier.transformee import PULSATION
from polytech.maths.domain.fourier.unitaire import (
    RACINE_DE_DEUX_PI,
    en_cosinus,
)
from polytech.maths.domain.symboles import VARIABLE as X

F = 1 / (1 + X**2)
G = sp.diff(F, X)
FREQUENCES = (sp.Rational(1, 2), 3, -2)


def _par_les_residus(fonction: sp.Expr, s: object) -> sp.Expr:
    """`(1 / racine(2 pi)) int f e^(-isx) dx`, par les residus."""
    return integrale_oscillante(fonction, X, -sp.sympify(s)) / RACINE_DE_DEUX_PI


def test_la_derivee_d_aout_2023_par_les_deux_chemins() -> None:
    """`F(f)` par sa demi-integrale en cosinus (f paire), `F(g)` par les residus."""
    par_la_propriete = de_la_derivee(en_cosinus(F))
    for s in FREQUENCES:
        ecart = par_la_propriete.subs(PULSATION, s) - _par_les_residus(G, s)
        assert sp.simplify(ecart) == 0


def test_la_transformee_de_g_vaut_i_s_racine_pi_sur_deux_e_moins_abs_s() -> None:
    """Le resultat de la question : `F(g)(s) = i s racine(pi / 2) e^(-|s|)`."""
    for s in FREQUENCES:
        attendue = sp.I * s * sp.sqrt(sp.pi / 2) * sp.exp(-abs(sp.sympify(s)))
        assert sp.simplify(_par_les_residus(G, s) - attendue) == 0
