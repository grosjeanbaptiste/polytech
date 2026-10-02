"""La seance 2.1, exercice 2, questions 3 a 6 : les f.e.m. induites.

La spire de l'exercice se deplace, ou bien le courant du fil varie. Dans
les deux cas le manuscrit applique `e = -d(phi)/dt`.

Pour la translation, ce fichier ajoute la route que le manuscrit amorce
sans la finir : il ecrit le champ electromoteur `v ^ B` puis repart du
flux. Sa **circulation** le long de la spire donne la meme f.e.m. par un
chemin qui n'emploie aucun flux -- et les deux s'accordent, a l'inverse
du disque de l'exercice 12 ou le flux enlace ne varie pas.

Il etablit aussi ce que le manuscrit affirme pour la translation
oblique -- « il suffit de remplacer `v` par `v sin(alpha)` » -- en le
**deduisant** des deux cas purs : selon `x` la f.e.m. est entiere, selon
`y` elle est nulle.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.fem_de_la_spire import (
    fem_a_courant_sinusoidal,
    fem_de_translation,
    fem_par_le_champ_electromoteur,
)
from polytech.electromagnetisme.domain.magnetostatique.fil_et_spire import (
    SpireDevantUnFil,
)
from polytech.electromagnetisme.domain.symboles import PULSATION, TEMPS

PERMEABILITE = sp.Symbol("mu_0", positive=True)
COURANT = sp.Symbol("I_1", positive=True)
EFFICACE = sp.Symbol("I", positive=True)
LARGEUR = sp.Symbol("a", positive=True)
HAUTEUR = sp.Symbol("b", positive=True)
DISTANCE = sp.Symbol("d", positive=True)
VITESSE = sp.Symbol("v", positive=True)
ANGLE = sp.Symbol("alpha", positive=True)

SPIRE = SpireDevantUnFil(largeur=LARGEUR, hauteur=HAUTEUR, distance=DISTANCE)
FACTEUR = PERMEABILITE * COURANT * HAUTEUR * LARGEUR / (2 * sp.pi)


def _fem(vitesse_x: sp.Expr) -> sp.Expr:
    return fem_de_translation(SPIRE, COURANT, vitesse_x, PERMEABILITE)


def test_la_fem_de_translation_est_celle_qu_encadre_le_manuscrit() -> None:
    """`e = -(mu_0 I_1 a b v / 2 pi) / ((d + v t)(a + d + v t))`."""
    proche = DISTANCE + VITESSE * TEMPS
    attendue = -FACTEUR * VITESSE / (proche * (LARGEUR + proche))
    assert sp.simplify(_fem(VITESSE) - attendue) == 0


def test_la_circulation_du_champ_electromoteur_donne_la_meme_fem() -> None:
    """Route independante : `v ^ B` integre le long de la spire, sans flux."""
    par_circulation = fem_par_le_champ_electromoteur(
        SPIRE,
        COURANT,
        VITESSE,
        PERMEABILITE,
    )
    assert sp.simplify(par_circulation - _fem(VITESSE)) == 0


def test_la_fem_s_eteint_quand_la_spire_est_loin() -> None:
    """Le champ du fil devient uniforme a l'echelle de la spire, puis nul."""
    assert sp.limit(_fem(VITESSE), TEMPS, sp.oo) == 0


def test_une_translation_parallele_au_fil_n_induit_rien() -> None:
    """Question 4 : aucune vitesse selon `x`, donc aucun flux qui varie."""
    assert _fem(sp.Integer(0)) == 0


def test_la_translation_oblique_ne_compte_que_sa_composante_normale_au_fil() -> None:
    """Question 5 : remplacer `v` par `v sin(alpha)`, et c'est tout.

    L'oracle est le resultat du manuscrit, ecrit en toutes lettres.
    """
    proche = DISTANCE + VITESSE * sp.sin(ANGLE) * TEMPS
    attendue = -FACTEUR * VITESSE * sp.sin(ANGLE) / (proche * (LARGEUR + proche))
    assert sp.simplify(_fem(VITESSE * sp.sin(ANGLE)) - attendue) == 0


def test_la_fem_a_courant_sinusoidal_est_celle_qu_encadre_le_manuscrit() -> None:
    """`e = (mu_0 b I sqrt(2) omega / 2 pi) ln(d / (a + d)) sin(omega t)`."""
    attendue = (
        PERMEABILITE
        * HAUTEUR
        * EFFICACE
        * sp.sqrt(2)
        * PULSATION
        / (2 * sp.pi)
        * sp.log(DISTANCE / (LARGEUR + DISTANCE))
        * sp.sin(PULSATION * TEMPS)
    )
    obtenue = fem_a_courant_sinusoidal(SPIRE, EFFICACE, PERMEABILITE)
    assert sp.simplify(sp.expand_log(obtenue - attendue, force=True)) == 0


def test_la_fem_est_en_quadrature_avec_le_courant_qui_l_induit() -> None:
    """Le courant est en cosinus, la f.e.m. en sinus : un quart de periode.

    Elle est donc nulle quand le courant est extremal, et extremale quand
    il s'annule -- c'est la derivee qui compte, non la valeur.
    """
    fem = fem_a_courant_sinusoidal(SPIRE, EFFICACE, PERMEABILITE)
    assert sp.simplify(fem.subs(TEMPS, 0)) == 0
    quart = sp.pi / (2 * PULSATION)
    assert sp.simplify(fem.subs(TEMPS, quart)) != 0
