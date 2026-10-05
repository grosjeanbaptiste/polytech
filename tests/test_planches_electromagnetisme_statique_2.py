"""Planches d'electromagnetisme, chapitre II.2 : electrostatique avancee.

Source : `Electromagnetisme/SLIDES/Slides_02_Etats_statiques_partie_2.pdf`,
planches 2 a 19, lues sur les pages rendues.

Le fascicule resume les chapitres 3 et 4 du syllabus. Chaque relation
surlignee est eprouvee sur le cas que la planche dessine : le conducteur
cylindrique de la planche 5, la ligne au-dessus du sol des planches 7 et
8, le condensateur de la planche 12.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.coefficients import (
    residu_de_l_identite_de_gauss,
)
from polytech.electromagnetisme.domain.electrostatique.condensateurs import (
    charge_d_un_condensateur,
)
from polytech.electromagnetisme.domain.electrostatique.conducteurs import (
    champ_le_long_de_la_surface,
    champ_sur_la_surface,
)
from polytech.electromagnetisme.domain.electrostatique.cylindre import (
    PERMITTIVITE,
    divergence_cylindrique,
)
from polytech.electromagnetisme.domain.electrostatique.dielectriques import (
    charge_apparente,
    charge_fictive,
    polarisation_depuis_le_deplacement,
    susceptibilite,
)
from polytech.electromagnetisme.domain.electrostatique.droites_paralleles import (
    potentiel_de_deux_droites,
)
from polytech.electromagnetisme.domain.electrostatique.moments import SystemeDeCharges
from polytech.electromagnetisme.domain.symboles import ABSCISSE, ORDONNEE

RHO, RAYON = sp.symbols("rho R", positive=True)
LINEIQUE = sp.Symbol("lambda", positive=True)


def test_le_champ_sur_la_surface_est_la_moitie_du_champ_exterieur() -> None:
    """Planche 4 : `E_ext = sigma / eps_0` et `E_sur = E_ext / 2`."""
    sigma = sp.Symbol("sigma", positive=True)
    assert champ_le_long_de_la_surface(sigma) == sigma / PERMITTIVITE
    assert champ_sur_la_surface(sigma) == champ_le_long_de_la_surface(sigma) / 2


def test_le_conducteur_cylindrique_de_la_planche_5() -> None:
    """`V = (Q / l) ln(1 / rho) / (2 pi eps_0)`, `E = (Q / l) / (2 pi eps_0 rho)`.

    Sur le conducteur, la charge surfacique vaut `(Q / l) / (2 pi R)` et
    le champ la moitie du champ exterieur.
    """
    potentiel = LINEIQUE * sp.log(1 / RHO) / (2 * sp.pi * PERMITTIVITE)
    champ = -sp.diff(potentiel, RHO)
    assert sp.simplify(champ - LINEIQUE / (2 * sp.pi * PERMITTIVITE * RHO)) == 0
    surfacique = LINEIQUE / (2 * sp.pi * RAYON)
    sur_le_conducteur = champ_sur_la_surface(surfacique)
    assert sp.simplify(sur_le_conducteur - champ.subs(RHO, RAYON) / 2) == 0


def test_hors_du_conducteur_le_potentiel_suit_l_equation_de_laplace() -> None:
    """Planche 6 : le probleme fondamental, `laplacien V = 0` hors des conducteurs."""
    champ = LINEIQUE / (2 * sp.pi * PERMITTIVITE * RHO)
    assert divergence_cylindrique(champ, RHO) == 0


def test_le_sol_est_l_equipotentielle_zero_de_la_ligne_et_de_son_image() -> None:
    """Planches 7 et 8 : `Q / l` en `h`, son image en `-h`, plan mediateur nul."""
    hauteur = sp.Symbol("h", positive=True)
    potentiel = potentiel_de_deux_droites(LINEIQUE, hauteur)
    assert sp.simplify(potentiel.subs(ABSCISSE, 0)) == 0
    assert sp.simplify(potentiel.subs({ABSCISSE: hauteur / 2, ORDONNEE: 0})) != 0


def test_l_identite_de_gauss_definit_la_capacite() -> None:
    """Planches 11 et 12 : `Q' U = Q U'`, d'ou `Q = C U` et `C = 1 / U'` si `Q' = 1`."""
    charge, tension, unitaire = sp.symbols("Q_1 U_10 U_p", positive=True)
    residu = residu_de_l_identite_de_gauss([charge], [tension], [1], [unitaire])
    (solution,) = sp.solve(residu, charge)
    capacite = 1 / unitaire
    assert sp.simplify(solution - charge_d_un_condensateur(capacite, tension)) == 0


def test_un_atome_seul_et_le_dioxyde_de_carbone_ne_sont_pas_polaires() -> None:
    """Planche 14 : `p = 0` pour O=C=O, `p` non nul pour la molecule d'eau."""
    delta, longueur = sp.symbols("delta d", positive=True)
    carbone = SystemeDeCharges(
        (
            (2 * delta, (0, 0, 0)),
            (-delta, (longueur, 0, 0)),
            (-delta, (-longueur, 0, 0)),
        )
    )
    assert carbone.moment_electrique() == (0, 0, 0)
    eau = SystemeDeCharges(
        (
            (-2 * delta, (0, 0, 0)),
            (delta, (longueur, longueur, 0)),
            (delta, (-longueur, longueur, 0)),
        )
    )
    assert eau.moment_electrique() == (0, 2 * delta * longueur, 0)


def test_la_susceptibilite_et_la_polarisation_d_un_dielectrique() -> None:
    """Planches 16 et 18 : `chi = eps_r - 1`, `P = (1 - eps_0 / eps) D`."""
    relative, induction = sp.symbols("varepsilon_r D", positive=True)
    assert susceptibilite(4) == 3
    polarise = polarisation_depuis_le_deplacement(relative, induction)
    assert sp.simplify(polarise - (1 - 1 / relative) * induction) == 0


def test_les_charges_apparentes_sont_les_charges_reelles_divisees() -> None:
    """Planche 18 : `eta + eta' = (eps_0 / eps) eta`, et de meme en surface."""
    relative, reelle = sp.symbols("varepsilon_r eta", positive=True)
    fictive = charge_fictive(relative, reelle)
    assert sp.simplify(reelle + fictive - reelle / relative) == 0
    assert sp.simplify(charge_apparente(relative, reelle) - reelle / relative) == 0
    # Un isolant parfait sans charge reelle n'a pas de charge fictive en volume.
    assert charge_fictive(relative, 0) == 0


def test_remplacer_eps0_par_eps_multiplie_la_capacite() -> None:
    """Planche 19 : le potentiel est divise par `eps_r` a charges egales."""
    relative, charge, tension = sp.symbols("varepsilon_r Q U", positive=True)
    dans_le_vide = charge / tension
    apparente = charge_apparente(relative, charge)
    dans_le_dielectrique = charge / (apparente / dans_le_vide)
    assert sp.simplify(dans_le_dielectrique - relative * dans_le_vide) == 0
