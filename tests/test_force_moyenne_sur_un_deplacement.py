"""Ce que vaut au juste le quotient `Delta W / Delta x` de l'exercice 6.

Ce fichier corrige une affirmation fausse, ecrite par l'agent dans le
corrige de l'exercice 6 du chapitre IV et dans la docstring de
`force_sur_une_armature` : « la capacite d'un condensateur plan varie
lineairement avec l'ecartement ». Elle varie en `1/e`. Elle est affine en
le **recouvrement** -- un glissement lateral --, pas en l'ecartement.

Le test qui pretendait justifier le `~=` de l'enonce etait juste en
lui-meme -- sur une capacite affine, la secante est la tangente -- mais il
ne prouvait rien sur l'exercice : il construisait la capacite affine dont
il avait besoin. L'enonce ne dit ni selon quelle direction l'armature se
deplace, ni que la capacite est affine.

Ce qui est vrai **sans rien supposer**, et que ce fichier etablit :

- le quotient `(1/2) U^2 Delta C / Delta x` est **exactement** la force
  **moyenne** sur le deplacement, quelle que soit la loi `C(x)`. C'est le
  theoreme fondamental de l'analyse, rien de plus ;
- il n'egale la force **instantanee** que si `C` est affine. Sur
  l'ecartement d'un condensateur plan, l'ecart relatif entre la force au
  depart et la force moyenne vaut `Delta x / (e + Delta x)` : il ne
  s'annule que pour un deplacement petit devant l'ecartement.

Le `~=` du manuscrit est donc bien une approximation dans le cas general,
et c'est la petitesse de `Delta x` **ou** la linearite de `C` qui la
justifie -- l'enonce ne permet pas de dire laquelle.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.energie_et_forces import (
    DEPLACEMENT,
    energie_d_un_condensateur,
    force_a_courant_constant,
)
from polytech.electromagnetisme.domain.force_moyenne import (
    ecart_relatif_sur_l_ecartement,
    force_moyenne,
    quotient_des_differences,
)

TENSION = sp.Symbol("U", positive=True)
ECARTEMENT = sp.Symbol("e", positive=True)
PAS = sp.Symbol("Delta", positive=True)
CONSTANTE = sp.Symbol("k", positive=True)

LOIS_DE_CAPACITE = (
    CONSTANTE / DEPLACEMENT,
    CONSTANTE * DEPLACEMENT,
    CONSTANTE * DEPLACEMENT**2,
)
"""Hyperbolique (l'ecartement), affine (le recouvrement), quadratique."""


def _energie(capacite: sp.Expr) -> sp.Expr:
    return energie_d_un_condensateur(capacite, TENSION)


def test_le_quotient_est_la_force_moyenne_quelle_que_soit_la_capacite() -> None:
    """L'egalite ne suppose rien sur `C(x)` : ni affine, ni petite variation."""
    for capacite in LOIS_DE_CAPACITE:
        energie = _energie(capacite)
        moyenne = force_moyenne(energie, ECARTEMENT, ECARTEMENT + PAS)
        quotient = quotient_des_differences(energie, ECARTEMENT, ECARTEMENT + PAS)
        assert sp.simplify(moyenne - quotient) == 0


def test_sur_une_capacite_affine_la_moyenne_est_la_force_instantanee() -> None:
    """Le cas du glissement lateral : la force ne depend pas de la position."""
    energie = _energie(CONSTANTE * DEPLACEMENT)
    instantanee = force_a_courant_constant(energie).subs(DEPLACEMENT, ECARTEMENT)
    moyenne = force_moyenne(energie, ECARTEMENT, ECARTEMENT + PAS)
    assert sp.simplify(moyenne - instantanee) == 0


def test_sur_l_ecartement_la_moyenne_n_est_pas_la_force_au_depart() -> None:
    """La capacite d'un condensateur plan est en `1/e`, donc non affine.

    C'est l'affirmation contraire que le corrige portait. La secante et
    la tangente y different, pour tout deplacement non nul.
    """
    energie = _energie(CONSTANTE / DEPLACEMENT)
    instantanee = force_a_courant_constant(energie).subs(DEPLACEMENT, ECARTEMENT)
    moyenne = force_moyenne(energie, ECARTEMENT, ECARTEMENT + PAS)
    assert sp.simplify(moyenne - instantanee) != 0


def test_l_ecart_relatif_sur_l_ecartement_vaut_le_pas_sur_l_ecartement_final() -> None:
    """`(F_depart - F_moyenne) / F_depart = Delta / (e + Delta)`.

    Oracle independant : avec `C = k/x`, la tangente vaut `-k/e^2` et la
    secante `-k/(e (e + Delta))` ; leur rapport est `e / (e + Delta)`.
    """
    obtenu = ecart_relatif_sur_l_ecartement(ECARTEMENT, PAS)
    assert sp.simplify(obtenu - PAS / (ECARTEMENT + PAS)) == 0


def test_l_ecart_relatif_s_annule_pour_un_deplacement_petit_devant_l_ecartement() -> (
    None
):
    """C'est la petitesse de `Delta x` qui justifie alors le `~=`."""
    ecart = ecart_relatif_sur_l_ecartement(ECARTEMENT, PAS)
    assert sp.limit(ecart, PAS, 0) == 0


def test_l_ecart_relatif_vaut_un_demi_quand_l_ecartement_double() -> None:
    """Un deplacement egal a l'ecartement : la moyenne est moitie du depart."""
    ecart = ecart_relatif_sur_l_ecartement(ECARTEMENT, ECARTEMENT)
    assert sp.simplify(ecart - sp.Rational(1, 2)) == 0


def test_un_deplacement_nul_est_refuse_par_les_deux_calculs() -> None:
    """Ni le quotient ni la moyenne n'ont de sens sur un intervalle vide."""
    energie = _energie(CONSTANTE / DEPLACEMENT)
    for calcul in (quotient_des_differences, force_moyenne):
        with pytest.raises(ValueError, match="deplacement nul"):
            calcul(energie, ECARTEMENT, ECARTEMENT)


def test_pour_les_donnees_de_l_enonce_l_ecart_serait_d_un_onzieme() -> None:
    """Dix millimetres d'ecartement, un de deplacement : `1/11`, soit 9 %.

    L'enonce du recueil donne `d = 10 mm` et un deplacement de `1 mm`.
    Ses armatures sont de revolution, non planes : la loi `C(d)` n'est
    pas connue. Mais **si** le condensateur etait plan, l'ecart entre la
    force au depart et la force moyenne serait d'un onzieme -- ordre de
    grandeur de ce que le « legerement » de l'enonce neglige.
    """
    ecart = ecart_relatif_sur_l_ecartement(sp.Rational(10, 1000), sp.Rational(1, 1000))
    assert ecart == sp.Rational(1, 11)
