"""La seance 2.1, exercice 2 : un fil rectiligne et une spire rectangulaire.

Le fil porte `I_1` selon `+Oy`. La spire, de cotes `a` selon `x` et `b`
selon `y`, a son cote gauche a la distance `d` du fil ; elle porte `I_2`
dans le sens direct autour de `+u_z`, donc vers le bas sur son cote
gauche.

Le manuscrit traite les deux premieres questions par deux methodes -- la
force de Laplace cote par cote, puis les travaux virtuels -- et constate
leur accord. Ce fichier refait les deux, et ajoute ce que le manuscrit ne
dit pas :

- les cotes haut et bas ne « s'annulent » pas par decret : chacun porte
  une force non nulle, et c'est leur **somme** qui est nulle. Les quatre
  forces poussent vers l'interieur : la spire est comprimee ;
- l'inductance mutuelle est **negative**, et ce signe est une convention
  -- celle de la normale choisie --, pas une propriete physique ;
- au loin, la force est celle d'un **dipole magnetique** dans un champ
  non uniforme : moment fois gradient.

Les oracles sont les resultats encadres du manuscrit, transcrits dans
`transcriptions/Electromagnetisme/EXS/Exs_03_Stationnaire_2.1.md`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.fil_et_spire import (
    SpireDevantUnFil,
    force_cote_par_cote,
    force_par_les_travaux_virtuels,
    induction_du_fil,
    resultante,
)
from polytech.maths.domain.comparaison import egales, est_negative, est_positive

PERMEABILITE = sp.Symbol("mu_0", positive=True)
COURANT_DU_FIL = sp.Symbol("I_1", positive=True)
COURANT_DE_LA_SPIRE = sp.Symbol("I_2", positive=True)
LARGEUR = sp.Symbol("a", positive=True)
HAUTEUR = sp.Symbol("b", positive=True)
DISTANCE = sp.Symbol("d", positive=True)

SPIRE = SpireDevantUnFil(largeur=LARGEUR, hauteur=HAUTEUR, distance=DISTANCE)
FACTEUR = PERMEABILITE * COURANT_DU_FIL * COURANT_DE_LA_SPIRE * HAUTEUR / (2 * sp.pi)


def _forces() -> dict[str, tuple[sp.Expr, sp.Expr]]:
    return force_cote_par_cote(
        SPIRE,
        COURANT_DU_FIL,
        COURANT_DE_LA_SPIRE,
        PERMEABILITE,
    )


def test_l_induction_du_fil_entre_dans_la_page_du_cote_de_la_spire() -> None:
    """`B_z = -mu_0 I_1 / (2 pi x)` pour `x > 0` : porte par `-u_z`."""
    abscisse = sp.Symbol("x", positive=True)
    induction = induction_du_fil(COURANT_DU_FIL, abscisse, PERMEABILITE)
    attendue = -PERMEABILITE * COURANT_DU_FIL / (2 * sp.pi * abscisse)
    assert egales(induction, attendue)


def test_le_flux_est_celui_du_manuscrit() -> None:
    """`phi_21 = (mu_0 I_1 b / 2 pi) ln(d / (a + d))`."""
    attendu = (
        PERMEABILITE
        * COURANT_DU_FIL
        * HAUTEUR
        / (2 * sp.pi)
        * sp.log(DISTANCE / (LARGEUR + DISTANCE))
    )
    obtenu = SPIRE.flux(COURANT_DU_FIL, PERMEABILITE)
    assert egales(sp.expand_log(obtenu - attendu, force=True), 0)


def test_l_inductance_mutuelle_est_celle_qu_encadre_le_manuscrit() -> None:
    """`L_12 = (mu_0 b / 2 pi) ln(d / (a + d))`."""
    attendue = (
        PERMEABILITE * HAUTEUR / (2 * sp.pi) * sp.log(DISTANCE / (LARGEUR + DISTANCE))
    )
    obtenue = SPIRE.inductance_mutuelle(PERMEABILITE)
    assert egales(sp.expand_log(obtenue - attendue, force=True), 0)


def test_l_inductance_mutuelle_est_negative_avec_cette_normale() -> None:
    """Le signe vient de la normale `+u_z`, que le champ traverse a rebours.

    `d < a + d`, donc le logarithme est negatif. Choisir `-u_z` pour
    normale donnerait la meme valeur absolue et le signe oppose : c'est
    une convention, et le manuscrit dit bien « arbitraire ».
    """
    oppose = -SPIRE.inductance_mutuelle(PERMEABILITE)
    reference = PERMEABILITE * HAUTEUR / (2 * sp.pi) * sp.log(1 + LARGEUR / DISTANCE)
    echelle = 2 * sp.pi / (PERMEABILITE * HAUTEUR)
    assert egales(sp.exp(oppose * echelle), sp.exp(reference * echelle))
    assert reference.is_positive


def test_l_inductance_mutuelle_s_eteint_quand_la_spire_s_eloigne() -> None:
    """Loin du fil, plus aucun flux ne traverse la spire."""
    mutuelle = SPIRE.inductance_mutuelle(PERMEABILITE)
    assert sp.limit(mutuelle, DISTANCE, sp.oo) == 0


def test_la_force_sur_le_cote_gauche_repousse_la_spire() -> None:
    """Courants antiparalleles : `+ mu_0 I_1 I_2 b / (2 pi d)` selon `u_x`."""
    force_x, force_y = _forces()["gauche"]
    assert egales(force_x, FACTEUR / DISTANCE)
    assert force_y == 0


def test_la_force_sur_le_cote_droit_attire_la_spire() -> None:
    """Courants paralleles : `- mu_0 I_1 I_2 b / (2 pi (a + d))` selon `u_x`."""
    force_x, force_y = _forces()["droit"]
    assert egales(force_x, -(FACTEUR / (LARGEUR + DISTANCE)))
    assert force_y == 0


def test_les_cotes_haut_et_bas_portent_chacun_une_force_non_nulle() -> None:
    """Le manuscrit ecrit qu'ils « annulent leurs effets » : c'est la somme.

    Chacun subit une force selon `Oy`, de module
    `(mu_0 I_1 I_2 / 2 pi) ln((a + d) / d)`, et les deux sont opposees.
    """
    haut = _forces()["haut"]
    bas = _forces()["bas"]
    assert haut[0] == 0
    assert bas[0] == 0
    assert not egales(haut[1], 0)
    assert egales(haut[1] + bas[1], 0)


def test_les_quatre_forces_poussent_vers_l_interieur_de_la_spire() -> None:
    """La spire est **comprimee**, non etiree.

    Le cote haut est pousse vers le bas, le bas vers le haut, le gauche
    vers la droite et le droit vers la gauche. C'est ce qu'on attend
    d'une spire dont le flux propre s'oppose a celui du fil : elle tend
    a embrasser moins de ce flux.

    Le premier jet de ce fichier affirmait « etiree » dans une docstring,
    sans test : le signe n'avait pas ete regarde. Les quatre signes sont
    donc fixes ici, a `a = d`, ou le logarithme vaut `ln 2`.
    """
    a_egal_d = {LARGEUR: DISTANCE}
    forces = {
        cote: (f[0].subs(a_egal_d), f[1].subs(a_egal_d))
        for cote, f in _forces().items()
    }
    assert est_negative(forces["haut"][1])
    assert est_positive(forces["bas"][1])
    assert est_positive(forces["gauche"][0])
    assert est_negative(forces["droit"][0])


def test_la_resultante_est_celle_qu_encadre_le_manuscrit() -> None:
    """`F = (mu_0 I_1 I_2 b / 2 pi) a / (d (a + d))` selon `u_x`."""
    force_x, force_y = resultante(_forces())
    attendue = FACTEUR * LARGEUR / (DISTANCE * (LARGEUR + DISTANCE))
    assert egales(force_x, attendue)
    assert egales(force_y, 0)


def test_les_travaux_virtuels_redonnent_la_meme_force() -> None:
    """`F_x = I_1 I_2 dL_12/dd` et `F_y = 0` : la verification du manuscrit.

    Les deux routes ne partagent rien : l'une integre des forces de
    Laplace sur quatre cotes, l'autre derive une inductance.
    """
    directe = resultante(_forces())
    virtuelle = force_par_les_travaux_virtuels(
        SPIRE,
        COURANT_DU_FIL,
        COURANT_DE_LA_SPIRE,
        PERMEABILITE,
    )
    assert egales(directe[0], virtuelle[0])
    assert virtuelle[1] == 0


def test_la_force_repousse_la_spire_parce_que_le_cote_proche_l_emporte() -> None:
    """La resultante est positive : le cote gauche est plus pres du fil."""
    force_x, _ = resultante(_forces())
    assert est_positive(force_x)


def test_au_loin_la_force_est_le_moment_fois_le_gradient() -> None:
    """`F -> m dB/dx` avec `m = I_2 a b` : un dipole dans un champ non uniforme.

    Oracle independant : le moment magnetique de la spire, multiplie par
    la derivee du module de l'induction du fil. Aucune force de Laplace
    n'y intervient.
    """
    force_x, _ = resultante(_forces())
    moment = COURANT_DE_LA_SPIRE * LARGEUR * HAUTEUR
    gradient = PERMEABILITE * COURANT_DU_FIL / (2 * sp.pi * DISTANCE**2)
    assert sp.limit(force_x / (moment * gradient), DISTANCE, sp.oo) == 1
