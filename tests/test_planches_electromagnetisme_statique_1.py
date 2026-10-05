"""Planches d'electromagnetisme, chapitre II.1 : lois du champ electrostatique.

Source : `Electromagnetisme/SLIDES/Slides_02_Etats_statiques_partie_1.pdf`,
planches 2 a 22, lues sur les pages rendues.

Le fascicule resume le chapitre 2 du syllabus : rien n'y est neuf pour la
bibliotheque, et ces tests le disent planche par planche, en eprouvant
chaque relation surlignee sur un cas dont le resultat est connu.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.constantes import PERMITTIVITE_DU_VIDE
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.lois import est_electrostatique
from polytech.electromagnetisme.domain.electrostatique.moments import (
    SystemeDeCharges,
    moment_d_un_doublet,
)
from polytech.electromagnetisme.domain.electrostatique.passage import (
    residu_tangentiel,
    saut_normal,
)
from polytech.electromagnetisme.domain.electrostatique.polarisation import (
    charge_surfacique_fictive,
    charge_volumique_fictive,
    deplacement,
)
from polytech.electromagnetisme.domain.electrostatique.potentiel import (
    COULOMB,
    champ_du_potentiel,
    est_une_ligne_de_champ,
    potentiel_des_charges,
)
from polytech.electromagnetisme.domain.maxwell import residu_de_gauss_electrique
from polytech.electromagnetisme.domain.symboles import CARTESIENNES

X, Y, Z = CARTESIENNES
Q = sp.Symbol("q", positive=True)


def test_la_permittivite_du_vide_de_la_planche_7() -> None:
    """« 8,85 10^-12 F/m »."""
    assert pytest.approx(8.85e-12, rel=1e-3, abs=0) == PERMITTIVITE_DU_VIDE


def test_le_potentiel_et_le_champ_d_une_charge_ponctuelle() -> None:
    """Planche 8 : `V = q / (4 pi eps_0 r)` et `E = q u_r / (4 pi eps_0 r^2)`."""
    potentiel = potentiel_des_charges(SystemeDeCharges(((Q, (0, 0, 0)),)))
    sur_l_axe = potentiel.subs({Y: 0, Z: 0})
    distance = sp.Symbol("r", positive=True)
    assert sp.simplify(sur_l_axe.subs(X, distance) - COULOMB * Q / distance) == 0
    champ = champ_du_potentiel(potentiel)
    radial = champ[0].subs({Y: 0, Z: 0}).subs(X, distance)
    assert sp.simplify(radial - COULOMB * Q / distance**2) == 0


def test_le_champ_electrostatique_est_irrotationnel() -> None:
    """Planche 4 : la circulation sur tout contour ferme est nulle, `E = -grad V`."""
    charges = SystemeDeCharges(((Q, (1, 0, 0)), (-2 * Q, (0, 2, 0))))
    assert est_electrostatique(champ_du_potentiel(potentiel_des_charges(charges)))


def test_les_lignes_de_champ_d_une_charge_sont_des_rayons() -> None:
    """Planche 5 : orthogonales aux equipotentielles, elles partent de la charge."""
    champ = champ_du_potentiel(
        potentiel_des_charges(SystemeDeCharges(((Q, (0, 0, 0)),)))
    )
    parametre = sp.Symbol("s", positive=True)
    assert est_une_ligne_de_champ(champ, (parametre, 2 * parametre, 0), parametre)
    assert not est_une_ligne_de_champ(champ, (parametre, parametre**2, 0), parametre)


def test_le_gradient_de_l_inverse_de_la_distance() -> None:
    """Planche 7 : `grad(1 / R) = -R / R^3`, d'ou le champ de Coulomb."""
    distance = sp.sqrt(X**2 + Y**2 + Z**2)
    gradient = [sp.diff(1 / distance, axe) for axe in CARTESIENNES]
    attendu = [-axe / distance**3 for axe in CARTESIENNES]
    assert [sp.simplify(g - a) for g, a in zip(gradient, attendu, strict=True)] == [
        0
    ] * 3


def test_le_moment_d_un_doublet_ne_depend_pas_du_point_de_reference() -> None:
    """Planche 9 : `p = q d`, alors que le moment d'un systeme charge depend de `A`."""
    doublet = SystemeDeCharges(((Q, (1, 2, 0)), (-Q, (0, 0, 0))))
    reference = (sp.Symbol("a"), sp.Symbol("b"), sp.Symbol("c"))
    deplace = doublet.moment_electrique(reference)
    assert tuple(sp.expand(c) for c in deplace) == (Q, 2 * Q, 0)
    direct = moment_d_un_doublet(Q, negative=(0, 0, 0), positive=(1, 2, 0))
    assert direct == (Q, 2 * Q, 0)
    charge = SystemeDeCharges(((Q, (1, 2, 0)),))
    assert charge.moment_electrique(reference) != charge.moment_electrique()


def test_une_polarisation_uniforme_ne_laisse_de_charges_qu_en_surface() -> None:
    """Planche 12 : les dipoles voisins se compensent, sauf aux extremites."""
    polarise = (sp.Symbol("P_0", positive=True), 0, 0)
    assert charge_volumique_fictive(polarise) == 0
    assert charge_surfacique_fictive(polarise, normale_interieure=(1, 0, 0)) == (
        -polarise[0]
    )
    assert (
        charge_surfacique_fictive(polarise, normale_interieure=(-1, 0, 0))
        == (polarise[0])
    )


def test_une_polarisation_non_uniforme_fait_apparaitre_une_charge_volumique() -> None:
    """Planche 12 : si `div P` n'est pas nulle, `eta'` ne l'est pas non plus."""
    assert charge_volumique_fictive((3 * X, 0, 0)) == -3


def test_gauss_local_et_poisson_dans_une_boule_uniformement_chargee() -> None:
    """Planches 15 et 16 : `div D = eta` et `laplacien V = -eta / eps_0`.

    Dans la boule, `V = eta (3 a^2 - r^2) / (6 eps_0)`.
    """
    densite, rayon = sp.symbols("eta a", positive=True)
    carre = X**2 + Y**2 + Z**2
    potentiel = densite * (3 * rayon**2 - carre) / (6 * PERMITTIVITE)
    champ = champ_du_potentiel(potentiel)
    assert residu_de_gauss_electrique(deplacement(champ, (0, 0, 0)), densite) == 0
    laplacien = sum(sp.diff(potentiel, axe, 2) for axe in CARTESIENNES)
    assert sp.simplify(laplacien + densite / PERMITTIVITE) == 0


def test_la_continuite_au_droit_d_un_plan_charge() -> None:
    """Planches 18 a 22 : `E_t` continu, `D_n` saute de `sigma`, `V` continu.

    Un plan `z = 0` de charge surfacique `sigma` : `D = sigma / 2` de
    part et d'autre, en s'eloignant du plan.
    """
    sigma = sp.Symbol("sigma", positive=True)
    dessus, dessous = (0, 0, sigma / 2), (0, 0, -sigma / 2)
    assert saut_normal(dessus, dessous, normale_1=(0, 0, 1)) == sigma
    assert residu_tangentiel(dessus, dessous, normale_1=(0, 0, 1)) == (0, 0, 0)
    potentiel = -sigma * sp.Abs(Z) / (2 * PERMITTIVITE)
    assert sp.limit(potentiel, Z, 0, "+") == sp.limit(potentiel, Z, 0, "-")


def test_les_deux_voies_de_la_note_du_transparent_7_s_accordent() -> None:
    """Note manuscrite : `E = -grad V` par `(1 / x)' = -1 / x^2`, puis par composantes.

    La note ecrit `grad(r - r_i) = 1` ; ce qui vaut un, c'est la norme de
    `grad R_i`, le vecteur unitaire `u_R` de la voie 1.
    """
    position = (sp.Integer(1), sp.Integer(-2), sp.Integer(3))
    ecarts = [axe - p for axe, p in zip(CARTESIENNES, position, strict=True)]
    distance = sp.sqrt(sum(e**2 for e in ecarts))
    unitaire = [sp.diff(distance, axe) for axe in CARTESIENNES]
    assert sp.simplify(sum(u**2 for u in unitaire)) == 1
    par_la_chaine = [COULOMB * Q * u / distance**2 for u in unitaire]
    champ = champ_du_potentiel(
        potentiel_des_charges(SystemeDeCharges(((Q, position),)))
    )
    assert [sp.simplify(a - b) for a, b in zip(par_la_chaine, champ, strict=True)] == (
        [0, 0, 0]
    )
