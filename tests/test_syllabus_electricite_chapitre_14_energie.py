"""Syllabus d'electricite generale, chapitre 14 : courant total, energie, Poynting.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 14-7 a 14-22, lues sur les pages rendues.

Un seul nombre vient du cours : le temps de relaxation du cuivre,
`0,15 . 10^-18 s`. Le theoreme de Poynting est eprouve sur trois cas dont
la puissance est connue par ailleurs : un fil resistif (`R I^2`), un
cable coaxial parfait (`U I`) et une onde plane dans le vide.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.resistance import resistance_uniforme
from polytech.electromagnetisme.domain.circuits.resistivite import METAUX
from polytech.electromagnetisme.domain.constantes import (
    PERMEABILITE_DU_VIDE,
    PERMITTIVITE_DU_VIDE,
)
from polytech.electromagnetisme.domain.courant_total import (
    courant_de_deplacement,
    courant_total,
    rapport_du_deplacement_a_la_conduction,
)
from polytech.electromagnetisme.domain.courants import (
    courant_de_polarisation,
    intensite_a_travers,
)
from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
)
from polytech.electromagnetisme.domain.electrostatique.passage import (
    residu_tangentiel,
    saut_normal,
)
from polytech.electromagnetisme.domain.maxwell import (
    equation_de_continuite,
    residu_de_gauss_electrique,
    residus_d_ampere_maxwell,
    residus_de_faraday,
)
from polytech.electromagnetisme.domain.onde import (
    celerite_dans_le_vide,
    impedance_du_vide,
)
from polytech.electromagnetisme.domain.poynting import (
    puissance_emmagasinee,
    residu_de_poynting,
    vecteur_de_poynting,
)
from polytech.electromagnetisme.domain.relaxation import (
    charge_volumique_residuelle,
    duree_de_relaxation,
    temps_de_relaxation,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES, TEMPS

X, Y, Z = CARTESIENNES
NUL = (0, 0, 0)
CUIVRE = METAUX["cuivre"][0]


def test_le_courant_de_deplacement_est_la_derivee_du_deplacement() -> None:
    """(14.9.1) : `i_d = dD/dt`."""
    deplacement = (sp.sin(3 * TEMPS), X * TEMPS**2, 0)
    assert courant_de_deplacement(deplacement) == (
        3 * sp.cos(3 * TEMPS),
        2 * X * TEMPS,
        0,
    )


def test_dans_la_matiere_il_porte_le_courant_de_polarisation() -> None:
    """(14.18.1) : `i_d = eps_0 dE/dt + i_p`, `i_p` etant un vrai courant."""
    eps = sp.Symbol("varepsilon_0", positive=True)
    champ = (sp.Function("E")(TEMPS), 0, 0)
    polarise = (sp.Function("P")(TEMPS), 0, 0)
    deplacement = tuple(eps * e + p for e, p in zip(champ, polarise, strict=True))
    vide = courant_de_deplacement(tuple(eps * e for e in champ))
    matiere = courant_de_polarisation(polarise)
    attendu = tuple(v + m for v, m in zip(vide, matiere, strict=True))
    assert courant_de_deplacement(deplacement) == attendu


def test_le_courant_total_est_a_flux_conservatif() -> None:
    """(14.9.3) : `div i = 0`, par la continuite (7.6.4) et Gauss (14.6.1).

    Une charge qui s'evacue le long de `x` : le courant de conduction a
    une divergence, que le courant de deplacement compense.
    """
    charge = sp.exp(-TEMPS) * sp.cos(X)
    deplacement = (sp.exp(-TEMPS) * sp.sin(X), 0, 0)
    conduction = (sp.exp(-TEMPS) * sp.sin(X), 0, 0)
    assert residu_de_gauss_electrique(deplacement, charge) == 0
    assert equation_de_continuite(conduction, charge) == 0
    assert divergence_cartesienne(conduction) != 0
    total = courant_total(conduction, NUL, deplacement)
    assert divergence_cartesienne(total) == 0


def test_le_courant_d_un_condensateur_se_prolonge_entre_ses_armatures() -> None:
    """(14.21.6) : `dI_cd = dI_d,is`, sur un condensateur plan de section `S`."""
    aire = sp.Symbol("S", positive=True)
    charge = sp.Function("Q")(TEMPS)
    dans_l_isolant = courant_de_deplacement((0, 0, charge / aire))
    prolonge = intensite_a_travers(dans_l_isolant, (0, 0, 1), aire)
    assert sp.simplify(prolonge - sp.diff(charge, TEMPS)) == 0


def test_le_cuivre_relaxe_en_un_dixieme_d_attoseconde() -> None:
    """(14.20.1) et page 14-19 : `tau = eps_0 rho = 0,15 . 10^-18 s`."""
    tau = temps_de_relaxation(PERMITTIVITE_DU_VIDE, CUIVRE)
    assert tau == pytest.approx(0.15e-18, rel=5e-3)


def test_un_milieu_sans_resistivite_n_a_pas_de_temps_de_relaxation() -> None:
    with pytest.raises(ValueError, match="nulle ou negative"):
        temps_de_relaxation(PERMITTIVITE_DU_VIDE, 0.0)
    with pytest.raises(ValueError, match="nulle ou negative"):
        temps_de_relaxation(0.0, CUIVRE)


def test_la_charge_volumique_s_eteint_exponentiellement() -> None:
    """(14.20.3) `eta = eta_0 exp(-t / tau)` resout (14.20.2)."""
    initiale, tau = sp.symbols("eta_0 tau", positive=True)
    charge = charge_volumique_residuelle(initiale, TEMPS, tau)
    assert sp.simplify(tau * sp.diff(charge, TEMPS) + charge) == 0
    assert charge.subs(TEMPS, 0) == initiale


def test_le_dix_millieme_est_atteint_en_neuf_constantes_de_temps() -> None:
    """Page 14-19 : `tau ln 10 000`, « soit environ 9,2 tau »."""
    assert duree_de_relaxation(1e-4, 1.0) == pytest.approx(9.2, abs=0.02)
    atteinte = charge_volumique_residuelle(1, duree_de_relaxation(1e-4, 2.0), 2.0)
    assert float(atteinte) == pytest.approx(1e-4)


def test_une_fraction_hors_de_zero_un_est_refusee() -> None:
    with pytest.raises(ValueError, match="fraction hors de"):
        duree_de_relaxation(0.0, 1.0)
    with pytest.raises(ValueError, match="fraction hors de"):
        duree_de_relaxation(1.5, 1.0)


def test_dans_un_metal_le_courant_de_deplacement_est_negligeable() -> None:
    """Page 14-21 : le rapport des amplitudes vaut `2 pi f tau`."""
    tau = temps_de_relaxation(PERMITTIVITE_DU_VIDE, CUIVRE)
    rapport = float(rapport_du_deplacement_a_la_conduction(1e9, tau))
    assert rapport == pytest.approx(2 * math.pi * 1e9 * tau)
    assert rapport < 1e-8


def test_le_rapport_des_amplitudes_se_retrouve_sur_les_champs() -> None:
    """`D = eps E_m sin(2 pi f t)` contre `i_cd = E_m / rho`."""
    eps, rho, crete, frequence = sp.symbols("varepsilon rho E_m f", positive=True)
    deplacement = (eps * crete * sp.sin(2 * sp.pi * frequence * TEMPS), 0, 0)
    amplitude = courant_de_deplacement(deplacement)[0].subs(TEMPS, 0)
    attendu = rapport_du_deplacement_a_la_conduction(frequence, eps * rho)
    assert sp.simplify(amplitude / (crete / rho) - attendu) == 0


def test_un_fil_resistif_recoit_par_sa_surface_la_puissance_de_joule() -> None:
    """Page 14-15 : la composante normale de Poynting, entrante, vaut l'effet Joule.

    Au point `(a, 0, 0)` d'un fil d'axe `z` : `E = rho i` selon `z`,
    `H = I / (2 pi a)` selon `y`.
    """
    rayon, longueur, courant = 1e-3, 2.0, 5.0
    densite = courant / (math.pi * rayon**2)
    champ = (0.0, 0.0, CUIVRE * densite)
    magnetique = (0.0, courant / (2 * math.pi * rayon), 0.0)
    radial, tangent, axial = vecteur_de_poynting(champ, magnetique)
    assert radial < 0
    assert (tangent, axial) == (0, 0)
    entrante = -radial * 2 * math.pi * rayon * longueur
    resistance = resistance_uniforme(CUIVRE, longueur, math.pi * rayon**2)
    assert entrante == pytest.approx(resistance * courant**2)


def test_un_cable_coaxial_parfait_transporte_u_i_par_son_isolant() -> None:
    """Page 14-15 : entre conducteurs parfaits, Poynting leur est parallele.

    `E = U / (r ln(b / a))` radial, `H = I / (2 pi r)` azimutal : le flux
    du vecteur a travers la couronne vaut `U I`.
    """
    tension, courant, interieur, exterieur = sp.symbols("U I a b", positive=True)
    champ = (tension / (X * sp.log(exterieur / interieur)), 0, 0)
    magnetique = (0, courant / (2 * sp.pi * X), 0)
    radial, tangent, axial = vecteur_de_poynting(champ, magnetique)
    assert (radial, tangent) == (0, 0)
    flux = sp.integrate(axial * 2 * sp.pi * X, (X, interieur, exterieur))
    assert sp.simplify(flux - tension * courant) == 0


def test_des_charges_et_des_aimants_au_repos_ne_rayonnent_pas() -> None:
    """Remarque de la page 14-16 : Poynting non nul, mais de flux ferme nul."""
    distance = sp.sqrt(X**2 + Y**2 + Z**2)
    coulombien = tuple(axe / distance**3 for axe in CARTESIENNES)
    uniforme = (0, 0, 1)
    poynting = vecteur_de_poynting(coulombien, uniforme)
    assert poynting != NUL
    assert sp.simplify(divergence_cartesienne(poynting)) == 0


ONDE = sp.cos(sp.Symbol("omega", positive=True) * (TEMPS - X / celerite_dans_le_vide()))
CHAMP = (0, ONDE, 0)
MAGNETIQUE = (0, 0, ONDE / impedance_du_vide())
DEPLACEMENT = (
    PERMITTIVITE_DU_VIDE * CHAMP[0],
    PERMITTIVITE_DU_VIDE * CHAMP[1],
    PERMITTIVITE_DU_VIDE * CHAMP[2],
)
INDUCTION = (
    PERMEABILITE_DU_VIDE * MAGNETIQUE[0],
    PERMEABILITE_DU_VIDE * MAGNETIQUE[1],
    PERMEABILITE_DU_VIDE * MAGNETIQUE[2],
)


def _nul(expression: sp.Expr) -> bool:
    """Vrai si l'expression s'annule, aux arrondis des constantes du vide pres."""
    valeur = expression.subs(
        {TEMPS: 0.3e-9, X: 0.07, sp.Symbol("omega", positive=True): 2e9}
    )
    return abs(float(valeur)) < 1e-9


def test_l_onde_plane_satisfait_les_equations_de_maxwell() -> None:
    """(14.15.1a) et (14.15.1d) dans le vide, sans charge ni courant."""
    assert all(_nul(r) for r in residus_de_faraday(CHAMP, INDUCTION))
    assert all(_nul(r) for r in residus_d_ampere_maxwell(MAGNETIQUE, NUL, DEPLACEMENT))


def test_le_theoreme_de_poynting_sur_l_onde_plane() -> None:
    """(14.16.1) sous forme locale : `dw/dt + div(E ^ H) = 0` dans le vide."""
    residu = residu_de_poynting(CHAMP, MAGNETIQUE, DEPLACEMENT, INDUCTION, NUL)
    echelle = float(impedance_du_vide())
    assert _nul(residu * echelle)
    emmagasinee = puissance_emmagasinee(CHAMP, MAGNETIQUE, DEPLACEMENT, INDUCTION)
    assert not _nul(emmagasinee * echelle * 1e3)


def test_la_puissance_emmagasinee_derive_des_energies_volumiques() -> None:
    """(14.14.1) et (14.15.3) : en milieu lineaire, `d(eps E^2 / 2 + mu H^2 / 2)/dt`."""
    eps, mu = sp.symbols("varepsilon mu", positive=True)
    champ = (sp.Function("E")(TEMPS), 0, 0)
    magnetique = (0, sp.Function("H")(TEMPS), 0)
    puissance = puissance_emmagasinee(
        champ,
        magnetique,
        tuple(eps * e for e in champ),
        tuple(mu * h for h in magnetique),
    )
    energie = eps * champ[0] ** 2 / 2 + mu * magnetique[1] ** 2 / 2
    assert sp.simplify(puissance - sp.diff(energie, TEMPS)) == 0


def test_dans_un_conducteur_le_residu_compte_l_effet_joule() -> None:
    """(14.16.1) : un champ uniforme et constant dissipe `E . i_cd`, sans flux."""
    champ, courant = (0, 0, sp.Integer(3)), (0, 0, sp.Integer(4))
    residu = residu_de_poynting(champ, NUL, NUL, NUL, courant)
    assert residu == 12


def test_la_nappe_de_courant_fait_sauter_le_champ_tangentiel() -> None:
    """(14.17.1d) `u_n1 ^ H_1 + u_n2 ^ H_2 = i_S`, et (14.17.1b) pour `B`.

    Une nappe `i_S = K` selon `x` dans le plan `z = 0` : `H = -K / 2`
    selon `y` au-dessus, `+K / 2` au-dessous.
    """
    nappe = sp.Symbol("K", positive=True)
    dessus, dessous = (0, -nappe / 2, 0), (0, nappe / 2, 0)
    assert residu_tangentiel(dessus, dessous, normale_1=(0, 0, 1)) == (nappe, 0, 0)
    assert saut_normal(dessus, dessous, normale_1=(0, 0, 1)) == 0


def test_le_courant_total_ajoute_ses_trois_composantes() -> None:
    """(14.9.2) : conduction, convection, deplacement."""
    total = courant_total((1, 0, 0), (0, 2, 0), (0, 0, 3 * TEMPS))
    assert total == (1, 2, 3)


def test_la_fraction_entiere_est_la_charge_de_depart() -> None:
    """A l'instant zero, il reste tout : aucune duree n'est ecoulee."""
    assert duree_de_relaxation(1.0, 2.0) == 0
