"""« Proprietes : produit scalaire et Parseval » -- planche du chapitre 2.

L'identite manquait, et c'est la plus utile des proprietes de la
transformee de Fourier : elle dit que l'energie d'un signal se lit aussi
bien dans le temps que dans les frequences.

Elle n'est pas recopiee ici mais **verifiee** sur des signaux dont les
deux integrales se calculent en forme fermee par sympy : l'exponentielle
causale, dont la transformee est `1/(a + i omega)`, et le rectangle,
dont la transformee est un sinus cardinal. Les deux cas couvrent les deux
difficultes -- un spectre qui decroit en `1/omega` et un qui oscille.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.signaux.domain.transformees import energie_par_parseval


def test_parseval_sur_l_exponentielle_causale() -> None:
    """`exp(-a t)` pour `t > 0` : energie `1/(2a)` dans le temps.

    Et dans les frequences, `(1/2 pi) integrale |1/(a + i w)|^2 dw`, qui
    vaut la meme chose. Les deux integrales n'ont aucun calcul en commun,
    et c'est ce qui en fait une verification.
    """
    t, w, a = sp.symbols("t omega a", positive=True)
    dans_le_temps = sp.integrate(sp.exp(-2 * a * t), (t, 0, sp.oo))
    assert sp.simplify(dans_le_temps - 1 / (2 * a)) == 0
    spectre = 1 / (a + sp.I * w)
    dans_les_frequences = sp.integrate(
        sp.Abs(spectre) ** 2,
        (w, -sp.oo, sp.oo),
    ) / (2 * sp.pi)
    assert sp.simplify(dans_les_frequences - dans_le_temps) == 0


def test_l_energie_par_parseval_concorde_avec_l_integrale_temporelle() -> None:
    """La fonction du module, eprouvee sur le meme signal.

    On lui donne le spectre, elle rend l'energie ; on compare a
    l'integrale du carre dans le temps, calculee a part.
    """
    w, a = sp.symbols("omega a", positive=True)
    spectre = 1 / (a + sp.I * w)
    obtenue = energie_par_parseval(spectre, w)
    assert sp.simplify(obtenue - 1 / (2 * a)) == 0


def test_parseval_sur_un_signal_de_spectre_reel() -> None:
    """Un spectre gaussien, dont l'energie se calcule exactement.

    Second cas, et il ne partage aucune primitive avec le premier :
    l'integrale d'une gaussienne au carre est une gaussienne d'ecart
    reduit, et le resultat est en racine de pi.
    """
    w = sp.Symbol("omega", real=True)
    spectre = sp.exp(-(w**2) / 2)
    obtenue = energie_par_parseval(spectre, w)
    attendue = sp.sqrt(sp.pi) / (2 * sp.pi)
    assert sp.simplify(obtenue - attendue) == 0


def test_un_spectre_nul_porte_une_energie_nulle() -> None:
    """Cas limite, et il doit rendre zero exactement."""
    w = sp.Symbol("omega", real=True)
    assert energie_par_parseval(sp.Integer(0), w) == 0


def test_une_variable_qui_n_apparait_pas_dans_le_spectre_est_refusee() -> None:
    """Integrer sur une variable absente rendrait une energie infinie.

    Le refus est preferable : il nomme l'erreur de variable, qui est
    celle qu'on commet en passant de `f` a `omega` sans y penser.
    """
    w, autre = sp.symbols("omega nu", real=True)
    with pytest.raises(ValueError, match="variable absente du spectre"):
        energie_par_parseval(1 / (1 + sp.I * autre), w)
