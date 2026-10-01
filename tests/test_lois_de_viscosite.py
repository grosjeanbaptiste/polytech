"""La loi de Newton de la viscosite, et les fluides qui ne la suivent pas.

Toujours le chapitre I, et c'est son coeur : la planche « Relation pour
la viscosite » definit la viscosite par `tau = mu du/dy`, et la planche
suivante met en regard quatre comportements que la planche nomme --
newtonien, Bingham, pseudoplastique, dilatant.

La distinction n'est pas academique. Un fluide de Bingham ne coule pas du
tout sous une contrainte inferieure a son seuil : du dentifrice tient
dans sa brosse, une peinture ne goutte pas. Un pseudoplastique s'ecoule
plus facilement quand on le cisaille plus fort -- c'est ce qui fait qu'une
peinture s'etale au pinceau puis se figea. Et un dilatant fait l'inverse,
d'ou le sable mouille de la planche, ou l'on s'enfonce lentement mais qui
durcit sous un coup.
"""

from __future__ import annotations

import pytest

from polytech.fluides.domain.viscosite import (
    Comportement,
    contrainte_de_bingham,
    contrainte_de_newton,
    contrainte_en_loi_de_puissance,
    viscosite_apparente,
    viscosite_depuis_la_contrainte,
)


def test_la_contrainte_newtonienne_est_proportionnelle_au_gradient() -> None:
    """`tau = mu du/dy` : la definition meme de la viscosite."""
    assert contrainte_de_newton(viscosite=1.83e-5, gradient=1000.0) == pytest.approx(
        0.0183,
    )


def test_la_viscosite_se_deduit_de_la_contrainte_mesuree() -> None:
    """L'aller-retour doit etre exact : c'est le principe du viscosimetre.

    On impose un gradient connu, on mesure la contrainte -- par le couple
    sur un cylindre tournant -- et l'on en tire `mu`. Si la conversion
    n'etait pas exacte, la mesure ne vaudrait rien.
    """
    for viscosite in (1e-5, 1e-3, 0.8):
        contrainte = contrainte_de_newton(viscosite, gradient=250.0)
        assert viscosite_depuis_la_contrainte(contrainte, 250.0) == pytest.approx(
            viscosite,
        )


def test_un_fluide_de_bingham_ne_coule_pas_sous_son_seuil() -> None:
    """`tau - tau_0 = mu du/dy` : le seuil est ce qui le distingue.

    Sous le seuil, le gradient est nul quelle que soit la contrainte --
    c'est pourquoi le dentifrice tient dans sa brosse. La contrainte
    rendue vaut alors exactement le seuil, et non zero : le fluide
    s'oppose, il ne cede pas.
    """
    assert contrainte_de_bingham(seuil=50.0, viscosite=0.5, gradient=0.0) == (
        pytest.approx(50.0)
    )
    assert contrainte_de_bingham(seuil=50.0, viscosite=0.5, gradient=100.0) == (
        pytest.approx(100.0)
    )


def test_la_loi_de_puissance_distingue_pseudoplastique_et_dilatant() -> None:
    """`tau = k (du/dy)^n`, et c'est `n` qui decide du comportement.

    Sous l'unite le fluide se fluidifie quand on le cisaille plus fort ;
    au-dessus il s'epaissit. A `n = 1` on retrouve exactement Newton, ce
    qui est la verification que la loi generalise bien la precedente.
    """
    newtonien = contrainte_en_loi_de_puissance(
        constante=0.5, exposant=1.0, gradient=4.0
    )
    assert newtonien == pytest.approx(contrainte_de_newton(0.5, 4.0))
    pseudo = contrainte_en_loi_de_puissance(constante=0.5, exposant=0.5, gradient=4.0)
    dilatant = contrainte_en_loi_de_puissance(constante=0.5, exposant=1.5, gradient=4.0)
    assert pseudo < newtonien < dilatant


def test_la_viscosite_apparente_d_un_pseudoplastique_decroit_au_cisaillement() -> None:
    """Et c'est tout ce que « rheofluidifiant » veut dire.

    La viscosite apparente est le rapport `tau / (du/dy)`. Pour un
    exposant sous l'unite, elle decroit quand le gradient croit -- une
    peinture s'etale sous le pinceau puis se figea sur le mur. Pour un
    dilatant elle croit, d'ou le sable mouille qui durcit sous un coup.
    """
    douce = viscosite_apparente(constante=0.5, exposant=0.5, gradient=1.0)
    forte = viscosite_apparente(constante=0.5, exposant=0.5, gradient=100.0)
    assert forte < douce
    dilatant_doux = viscosite_apparente(constante=0.5, exposant=1.5, gradient=1.0)
    dilatant_fort = viscosite_apparente(constante=0.5, exposant=1.5, gradient=100.0)
    assert dilatant_fort > dilatant_doux


def test_le_comportement_se_nomme_depuis_l_exposant() -> None:
    """Les trois noms que la planche emploie, et leurs frontieres."""
    assert Comportement("newtonien") is Comportement.NEWTONIEN
    assert Comportement.depuis_l_exposant(1.0) is Comportement.NEWTONIEN
    assert Comportement.depuis_l_exposant(0.4) is Comportement.PSEUDOPLASTIQUE
    assert Comportement.depuis_l_exposant(1.6) is Comportement.DILATANT


def test_un_gradient_nul_refuse_la_viscosite_apparente() -> None:
    """Le rapport `tau / (du/dy)` y divise par zero.

    Et le refus porte un sens : au repos, un fluide non newtonien n'a pas
    de viscosite definie -- c'est precisement ce qui le distingue.
    """
    with pytest.raises(ValueError, match="gradient de vitesse nul"):
        viscosite_apparente(constante=0.5, exposant=0.5, gradient=0.0)


def test_un_gradient_nul_refuse_la_mesure_de_viscosite() -> None:
    """Sans cisaillement, le viscosimetre ne mesure rien."""
    with pytest.raises(ValueError, match="gradient de vitesse nul"):
        viscosite_depuis_la_contrainte(contrainte=1.0, gradient=0.0)


def test_un_exposant_negatif_est_refuse() -> None:
    """Un exposant negatif ne decrit aucun fluide que la planche nomme.

    La contrainte y decroitrait quand le cisaillement croit, et le fluide
    s'effondrerait de lui-meme.
    """
    with pytest.raises(ValueError, match="exposant negatif ou nul"):
        contrainte_en_loi_de_puissance(constante=0.5, exposant=-0.5, gradient=1.0)


def test_un_seuil_de_bingham_negatif_est_refuse() -> None:
    """Un seuil negatif ferait couler le fluide sans contrainte du tout."""
    with pytest.raises(ValueError, match="seuil de Bingham negatif"):
        contrainte_de_bingham(seuil=-1.0, viscosite=0.5, gradient=1.0)
