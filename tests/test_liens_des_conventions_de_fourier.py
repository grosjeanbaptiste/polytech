"""Trois cours, trois conventions pour la transformee de Fourier : le lien.

- Mathematique (passerelle) et `polytech.maths` : `F(w) = int f e^(-j w x) dx`,
  le `1 / 2 pi` porte par l'inverse ;
- Signaux et systemes, syllabus section 2.1 : `F(f) = int f e^(-j 2 pi f t) dt`,
  aucun facteur ;
- Mathematiques III, section 11 partie 1, planche 8, definition (1) :
  `F(s) = (1 / racine(2 pi)) int f e^(-i s x) dx`, le facteur partage.

Ce ne sont pas les memes enonces : la convolution de la planche 9 porte un
`racine(2 pi)` que la bibliotheque n'a pas, et l'egalite energetique de la
planche 18 n'a pas le `1 / 2 pi` de `energie_frequentielle`. C'est pourquoi
`cursus/notions/MathematiquesIngenieur3.toml` rattache les titres de ce jeu a
`polytech.maths.domain.fourier.unitaire`, jamais a `transformee_de_fourier`.
Ces tests disent en quoi les trois s'accordent :
`F_M3 = F_maths / racine(2 pi)` et `F_signaux(f) = F_maths(2 pi f)`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.fourier.proprietes import (
    de_la_derivee,
    energie_frequentielle,
    energie_temporelle,
)
from polytech.maths.domain.fourier.transformee import (
    PULSATION,
    integrale_sur_la_droite,
    transformee_de_fourier,
)
from polytech.maths.domain.symboles import VARIABLE
from polytech.signaux.domain.frequence import (
    FREQUENCE,
    INSTANT,
    transformee_en_frequence,
    tri,
)

UNITAIRE = 1 / sp.sqrt(2 * sp.pi)
PORTE = sp.Piecewise((1, sp.Abs(VARIABLE) <= sp.Rational(1, 2)), (0, True))
CAUSALE = sp.exp(-VARIABLE) * sp.Heaviside(VARIABLE)
"""Ni paire ni impaire : un signe faux au noyau la trahit."""

FONCTIONS = [CAUSALE, PORTE, sp.exp(-sp.Abs(VARIABLE)), sp.exp(-(VARIABLE**2))]


def _de_mathematiques_3(fonction: sp.Expr) -> sp.Expr:
    """La definition (1) de la planche 8, integree telle que le cours l'ecrit.

    Hors de `s = 0`, point isole ou l'image se prolonge par continuite.
    """
    noyau = sp.exp(-sp.I * PULSATION * VARIABLE)
    image = integrale_sur_la_droite(fonction * noyau, VARIABLE)
    generique = image.args[0][0] if isinstance(image, sp.Piecewise) else image
    return UNITAIRE * generique


def _nul(ecart: sp.Expr) -> bool:
    return bool(sp.simplify(ecart.rewrite(sp.cos)) == 0)


@pytest.mark.parametrize("fonction", FONCTIONS)
def test_la_transformee_du_cours_est_celle_de_la_bibliotheque_a_un_facteur(
    fonction: sp.Expr,
) -> None:
    """Planche 8, (1) : `F_M3 = F_maths / racine(2 pi)`."""
    ecart = _de_mathematiques_3(fonction) - UNITAIRE * transformee_de_fourier(fonction)
    assert _nul(ecart)


@pytest.mark.parametrize("fonction", FONCTIONS[:2])
def test_l_egalite_energetique_du_cours_absorbe_le_facteur_de_la_bibliotheque(
    fonction: sp.Expr,
) -> None:
    """Planche 18 : `||f||^2 = ||F_M3||^2`, sans le `1 / 2 pi` de la bibliotheque."""
    image = transformee_de_fourier(fonction)
    carre = sp.simplify(sp.expand_complex(UNITAIRE**2 * image * sp.conjugate(image)))
    du_cours = integrale_sur_la_droite(carre, PULSATION)
    assert _nul(du_cours - energie_temporelle(fonction))
    assert _nul(du_cours - energie_frequentielle(image))


def test_la_convolution_ne_porte_le_facteur_que_dans_la_convention_du_cours() -> None:
    """Planche 9 : `F(f * g) = racine(2 pi) F(f) F(g)` ; la bibliotheque n'en a pas.

    Le triangle est la porte convoluee avec elle-meme.
    """
    triangle = tri(VARIABLE)
    porte = transformee_de_fourier(PORTE)
    assert _nul(transformee_de_fourier(triangle) - porte**2)
    du_cours = sp.sqrt(2 * sp.pi) * (UNITAIRE * porte) ** 2
    assert _nul(_de_mathematiques_3(triangle) - du_cours)


def test_la_derivation_ne_depend_pas_de_la_convention() -> None:
    """Planche 10 : `F(f') = i s F(f)`, le facteur se simplifie."""
    image = transformee_de_fourier(CAUSALE)
    assert _nul(de_la_derivee(image) - sp.I * PULSATION * image)


@pytest.mark.parametrize("fonction", FONCTIONS[:2])
def test_la_transformee_des_signaux_est_celle_des_maths_en_deux_pi_f(
    fonction: sp.Expr,
) -> None:
    """Syllabus de signaux, 2.1 : `F_signaux(f) = F_maths(w = 2 pi f)`."""
    en_hertz = transformee_en_frequence(fonction.subs(VARIABLE, INSTANT))
    attendue = transformee_de_fourier(fonction).subs(PULSATION, 2 * sp.pi * FREQUENCE)
    assert _nul(en_hertz - attendue)
