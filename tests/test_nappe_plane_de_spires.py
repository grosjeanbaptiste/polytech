"""La seance 2.1, exercice 3 : l'induction au centre d'une nappe de spires.

Des spires jointives occupent une couronne plane, du rayon `R_1` au rayon
`R_2`, et portent ensemble le courant `I`. Le manuscrit decoupe la
couronne en **tubes de courant** de largeur `dr`, applique Biot et Savart
a chacun, et integre.

Son resultat encadre est juste. Trois choses lui manquent :

- le **controle par la limite** : une couronne infiniment mince doit
  redonner la spire unique, `mu_0 I / (2 R)` -- la formule que tout le
  monde connait, et le meilleur garde-fou contre un facteur perdu ;
- l'**encadrement** : la nappe fait moins bien que la meme intensite
  concentree sur son bord interieur, et mieux que sur son bord exterieur ;
- la lecture du logarithme : l'induction est `mu_0 I / 2` fois la
  **moyenne de `1/r`** sur la couronne.

Le signe du manuscrit -- `-u_z` -- vient du sens de courant de sa
figure. Les fonctions rendent la composante selon `u_z`, signe compris.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.nappe_plane import (
    induction_d_une_spire_au_centre,
    induction_de_la_nappe_au_centre,
    moyenne_de_l_inverse_du_rayon,
)

PERMEABILITE = sp.Symbol("mu_0", positive=True)
COURANT = sp.Symbol("I", positive=True)
INTERIEUR = sp.Symbol("R_1", positive=True)
ECART = sp.Symbol("e", positive=True)
EXTERIEUR = INTERIEUR + ECART
"""`R_2 = R_1 + e` : ecrit ainsi, `R_2 > R_1` est acquis pour sympy."""

NAPPE = induction_de_la_nappe_au_centre(COURANT, INTERIEUR, EXTERIEUR, PERMEABILITE)


def test_une_spire_seule_donne_mu_zero_i_sur_deux_r() -> None:
    """Biot et Savart integre sur un tour : la formule de reference."""
    rayon = sp.Symbol("R", positive=True)
    obtenue = induction_d_une_spire_au_centre(COURANT, rayon, PERMEABILITE)
    assert sp.simplify(obtenue + PERMEABILITE * COURANT / (2 * rayon)) == 0


def test_l_induction_de_la_nappe_est_celle_qu_encadre_le_manuscrit() -> None:
    """`B_z = -(mu_0 / 2) I / (R_2 - R_1) ln(R_2 / R_1)`."""
    attendue = (
        -PERMEABILITE
        / 2
        * COURANT
        / (EXTERIEUR - INTERIEUR)
        * sp.log(EXTERIEUR / INTERIEUR)
    )
    assert sp.simplify(sp.expand_log(NAPPE - attendue, force=True)) == 0


def test_une_couronne_infiniment_mince_redonne_la_spire_unique() -> None:
    """La limite `R_2 -> R_1` : le controle que le manuscrit ne fait pas."""
    spire = induction_d_une_spire_au_centre(COURANT, INTERIEUR, PERMEABILITE)
    assert sp.simplify(sp.limit(NAPPE, ECART, 0) - spire) == 0


def test_la_nappe_est_encadree_par_les_spires_de_ses_deux_bords() -> None:
    """Moins qu'au bord interieur, plus qu'au bord exterieur.

    Verifie sur les modules : `x - 1 > ln(x) > 1 - 1/x` pour `x > 1`,
    avec `x = R_2 / R_1`. Le test l'etablit par un developpement au
    voisinage de la couronne mince, ou l'ecart est du second ordre et de
    signe fixe.
    """
    au_bord_interieur = -induction_d_une_spire_au_centre(
        COURANT, INTERIEUR, PERMEABILITE
    )
    au_bord_exterieur = -induction_d_une_spire_au_centre(
        COURANT, EXTERIEUR, PERMEABILITE
    )
    module = -NAPPE
    valeurs = {PERMEABILITE: 1, COURANT: 1, INTERIEUR: 1, ECART: 1}
    assert float(au_bord_exterieur.subs(valeurs)) < float(module.subs(valeurs))
    assert float(module.subs(valeurs)) < float(au_bord_interieur.subs(valeurs))
    premier_ordre = sp.series(module - au_bord_interieur, ECART, 0, 2).removeO()
    assert sp.simplify(premier_ordre).is_negative


def test_le_logarithme_est_la_moyenne_de_l_inverse_du_rayon() -> None:
    """`B = (mu_0 I / 2) <1/r>` : chaque tube pese par l'inverse de son rayon."""
    moyenne = moyenne_de_l_inverse_du_rayon(INTERIEUR, EXTERIEUR)
    assert sp.simplify(NAPPE + PERMEABILITE * COURANT / 2 * moyenne) == 0


def test_l_induction_diverge_si_la_nappe_atteint_le_centre() -> None:
    """`R_1 -> 0` : le logarithme diverge, les spires centrales pesent sans borne."""
    exterieur = sp.Symbol("R_2", positive=True)
    interieur = sp.Symbol("R_1", positive=True)
    module = -induction_de_la_nappe_au_centre(
        COURANT, interieur, exterieur, PERMEABILITE
    )
    assert sp.limit(module, interieur, 0, "+") == sp.oo
