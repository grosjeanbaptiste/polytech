"""Planches de mecanique des fluides, chapitre IV : equations fondamentales.

Source : `MecaniqueFluides/SLIDES/Slides_03_Equations_fondamentales.pdf`,
planches 2 a 54, lues sur les pages rendues (le fichier
`3_ITREE101_EquationsFondamentales.pdf` lui est identique, octet pour octet).

Les oracles du cours : l'ecoulement de Couette des planches 46 a 50, et
l'exercice du film d'huile de la planche 52, donne sans reponse et resolu
ici par les equations de Navier-Stokes. Les identites des planches -- la
derivee materielle de la masse, le rearrangement du terme visqueux, la
puissance des efforts internes -- sont eprouvees sur des champs choisis.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
    rotationnel_cartesien,
)
from polytech.fluides.domain.cinematique.espace import (
    tenseur_des_deformations,
    vorticite_dans_l_espace,
)
from polytech.fluides.domain.cinematique.symboles import T, X, Y, Z
from polytech.fluides.domain.fondamentales.contraintes import (
    contraintes_visqueuses,
    force_de_surface,
    tenseur_des_contraintes,
)
from polytech.fluides.domain.fondamentales.energie import (
    puissance_des_efforts_internes,
    residu_de_la_chaleur,
)
from polytech.fluides.domain.fondamentales.film_tombant import (
    debit_du_film,
    vitesse_du_film,
)
from polytech.fluides.domain.fondamentales.masse_locale import (
    residu_de_continuite,
    residu_en_derivee_materielle,
)
from polytech.fluides.domain.fondamentales.navier_stokes import (
    HAUTEUR,
    VISCOSITE,
    VITESSE_DE_PAROI,
    cisaillement,
    profil_de_couette,
)
from polytech.fluides.domain.fondamentales.quantite_de_mouvement import (
    residus_de_cauchy,
    residus_de_navier_stokes,
)
from polytech.maths.domain.analyse_vectorielle import (
    divergence,
    gradient,
    laplacien,
    rotationnel,
)

ESPACE = (X, Y, Z)
CHAMP = (X * Y + Z**2, Y * Z - X, X * Z + Y**2)
SCALAIRE = X**2 * Y + sp.sin(Z)
RHO, MU, G = sp.symbols("rho mu g", positive=True)


def test_le_gradient_la_divergence_et_le_rotationnel_des_rappels() -> None:
    """Planches 3 a 6, sur un champ et un scalaire choisis."""
    assert gradient(SCALAIRE, ESPACE) == (2 * X * Y, X**2, sp.cos(Z))
    assert divergence(CHAMP, ESPACE) == Y + Z + X
    assert rotationnel(CHAMP, ESPACE) == (2 * Y - Y, 2 * Z - Z, -1 - X)


def test_un_gradient_ne_tourne_pas_et_un_rotationnel_ne_diverge_pas() -> None:
    """Planche 6 : « ecoulement potentiel : rot V = 0 »."""
    assert rotationnel(gradient(SCALAIRE, ESPACE), ESPACE) == (0, 0, 0)
    assert divergence(rotationnel(CHAMP, ESPACE), ESPACE) == 0
    assert laplacien(SCALAIRE, ESPACE) == divergence(gradient(SCALAIRE, ESPACE), ESPACE)


def test_les_operateurs_s_accordent_d_un_cours_a_l_autre() -> None:
    """Un lien : l'electromagnetisme et la cinematique ecrivent les memes."""
    assert divergence(CHAMP, ESPACE) == divergence_cartesienne(CHAMP)
    assert rotationnel(CHAMP, ESPACE) == rotationnel_cartesien(CHAMP)
    assert rotationnel(CHAMP, ESPACE) == vorticite_dans_l_espace(CHAMP)


def test_le_theoreme_de_la_divergence_sur_un_cube() -> None:
    """Planche 4 : l'integrale de `div F` vaut le flux sortant de `F`."""
    volume = sp.integrate(divergence(CHAMP, ESPACE), (X, 0, 1), (Y, 0, 1), (Z, 0, 1))
    flux = sp.Integer(0)
    for rang, axe in enumerate(ESPACE):
        autres = [a for a in ESPACE if a != axe]
        sortant = CHAMP[rang].subs(axe, 1) - CHAMP[rang].subs(axe, 0)
        flux += sp.integrate(sortant, (autres[0], 0, 1), (autres[1], 0, 1))
    assert volume == flux


def test_un_systeme_de_coordonnees_incomplet_est_refuse() -> None:
    with pytest.raises(ValueError, match="trois coordonnees"):
        rotationnel(CHAMP, (X, Y))
    with pytest.raises(ValueError, match="autant de composantes"):
        divergence((X, Y), ESPACE)


def test_les_deux_ecritures_de_la_conservation_de_la_masse() -> None:
    """Planche 22 : `d(rho)/dt + div(rho v) = D(rho)/Dt + rho div v`."""
    masse = sp.Function("rho")(X, Y, Z, T)
    vitesse = tuple(sp.Function(n)(X, Y, Z, T) for n in ("u", "v", "w"))
    conservative = residu_de_continuite(masse, vitesse)
    materielle = residu_en_derivee_materielle(masse, vitesse)
    assert sp.simplify(conservative - materielle) == 0


def test_la_continuite_a_une_dimension() -> None:
    """Planche 22 : `d(rho)/dt + d(rho u)/dx = 0`, sur une onde de densite."""
    celerite = sp.Symbol("c", positive=True)
    onde = sp.Function("f")(X - celerite * T)
    assert residu_de_continuite(onde, (celerite, 0, 0)) == 0
    assert residu_de_continuite(onde, (2 * celerite, 0, 0)) != 0


def test_un_fluide_incompressible_a_une_vitesse_a_divergence_nulle() -> None:
    """Planche 38 : la masse volumique constante ne laisse que `div v = 0`."""
    assert residu_de_continuite(RHO, (X, -Y, 0)) == 0
    assert residu_de_continuite(RHO, (X, Y, 0)) == 2 * RHO


def test_le_tenseur_des_contraintes_separe_pression_et_viscosite() -> None:
    """Planche 36 : `sigma = -p I + tau`, avec `tau = 2 mu eps`."""
    pression = sp.Function("p")(X, Y, Z)
    visqueux = contraintes_visqueuses(CHAMP, MU)
    assert visqueux == 2 * MU * tenseur_des_deformations(CHAMP)
    total = tenseur_des_contraintes(pression, CHAMP, MU)
    assert total == -pression * sp.eye(3) + visqueux
    assert total == total.T


def test_en_statique_il_ne_reste_que_la_pression() -> None:
    """Planche 36 : « en statique des fluides », `sigma = -p I`."""
    pression = sp.Symbol("p", positive=True)
    assert tenseur_des_contraintes(pression, (0, 0, 0), MU) == -pression * sp.eye(3)


def test_la_force_de_surface_est_le_tenseur_applique_a_la_normale() -> None:
    """Planche 29 : `dF = sigma . n dA` ; en statique, une poussee normale."""
    pression = sp.Symbol("p", positive=True)
    statique = tenseur_des_contraintes(pression, (0, 0, 0), MU)
    assert force_de_surface(statique, (0, 0, 1), 2) == (0, 0, -2 * pression)
    cisaille = tenseur_des_contraintes(0, (Y, 0, 0), MU)
    # Couette : sur une facette de normale `y`, la force est tangentielle.
    assert force_de_surface(cisaille, (0, 1, 0), 1) == (MU, 0, 0)


def test_le_terme_visqueux_se_reduit_au_laplacien_en_incompressible() -> None:
    """Planches 39 et 40 : `div(2 mu eps) = mu laplacien(v) + mu grad(div v)`.

    Les residus de Cauchy et de Navier-Stokes ne different donc que du
    gradient de la divergence, nul pour un ecoulement incompressible.
    """
    pression = sp.Function("p")(X, Y, Z)
    pesanteur = (0, 0, -G)
    sigma = tenseur_des_contraintes(pression, CHAMP, MU)
    cauchy = residus_de_cauchy(RHO, CHAMP, sigma, pesanteur)
    navier = residus_de_navier_stokes(RHO, MU, CHAMP, pression, pesanteur)
    compressible = gradient(divergence(CHAMP, ESPACE), ESPACE)
    for c, n, reste in zip(cauchy, navier, compressible, strict=True):
        assert sp.simplify(c - n + MU * reste) == 0
    incompressible = (X * Y, -(Y**2) / 2 + Z, X - Z * Y + Y * Z)
    assert divergence(incompressible, ESPACE) == 0
    sigma = tenseur_des_contraintes(pression, incompressible, MU)
    ecarts = [
        sp.simplify(c - n)
        for c, n in zip(
            residus_de_cauchy(RHO, incompressible, sigma, pesanteur),
            residus_de_navier_stokes(RHO, MU, incompressible, pression, pesanteur),
            strict=True,
        )
    ]
    assert ecarts == [0, 0, 0]


def test_l_ecoulement_de_couette_resout_navier_stokes() -> None:
    """Planches 46 a 49 : `u = V y / h`, `p = p_0 - rho g z`."""
    vitesse = (profil_de_couette(), 0, 0)
    pression = sp.Symbol("p_0", positive=True) - RHO * G * Z
    assert residu_de_continuite(RHO, vitesse) == 0
    assert residus_de_navier_stokes(RHO, MU, vitesse, pression, (0, 0, -G)) == (0, 0, 0)
    # Sans pression hydrostatique, la composante verticale ne s'equilibre pas.
    desequilibre = residus_de_navier_stokes(RHO, MU, vitesse, 0, (0, 0, -G))
    assert desequilibre[2] != 0


def test_la_contrainte_de_couette_est_mu_v_sur_h() -> None:
    """Planche 50 : `tau_xy = tau_yx = mu V / h`, et `F / A = mu V / h`."""
    visqueux = contraintes_visqueuses((profil_de_couette(), 0, 0), VISCOSITE)
    attendue = VISCOSITE * VITESSE_DE_PAROI / HAUTEUR
    assert sp.simplify(visqueux[0, 1] - attendue) == 0
    assert visqueux[0, 1] == visqueux[1, 0]
    assert visqueux[0, 0] == 0
    assert sp.simplify(cisaillement() - attendue) == 0


def test_le_terme_d_advection_equilibre_la_pression_d_un_tourbillon() -> None:
    """Planche 44 : `rho (v . grad) v = -grad p`, sans viscosite ni gravite.

    Rotation en bloc `v = omega (-y, x, 0)` : l'acceleration est
    centripete, et la pression croit vers l'exterieur comme
    `rho omega^2 r^2 / 2`.
    """
    omega = sp.Symbol("omega", positive=True)
    vitesse = (-omega * Y, omega * X, 0)
    pression = RHO * omega**2 * (X**2 + Y**2) / 2
    assert residus_de_navier_stokes(RHO, MU, vitesse, pression, (0, 0, 0)) == (0, 0, 0)
    assert residus_de_navier_stokes(RHO, MU, vitesse, -pression, (0, 0, 0)) != (
        0,
        0,
        0,
    )
    sigma = tenseur_des_contraintes(pression, vitesse, MU)
    assert residus_de_cauchy(RHO, vitesse, sigma, (0, 0, 0)) == (0, 0, 0)


EPAISSEUR = sp.Symbol("h", positive=True)
FILM = vitesse_du_film(X, EPAISSEUR, RHO, MU, G)


def test_le_film_d_huile_de_la_planche_52_resout_navier_stokes() -> None:
    """La paroi en `x = 0`, l'air en `x = h`, la gravite selon `-z`.

    Aucune pression n'est appliquee : `P = P_atm` partout, et l'huile
    tombe « par simple gravite ».
    """
    vitesse = (0, 0, FILM)
    atmosphere = sp.Symbol("P_atm", positive=True)
    assert residu_de_continuite(RHO, vitesse) == 0
    assert residus_de_navier_stokes(RHO, MU, vitesse, atmosphere, (0, 0, -G)) == (
        0,
        0,
        0,
    )


def test_le_film_adhere_a_la_paroi_et_glisse_sans_frottement_sous_l_air() -> None:
    """Planche 45 : non-glissement a la paroi ; l'air n'exerce aucun cisaillement."""
    assert FILM.subs(X, 0) == 0
    assert sp.diff(FILM, X).subs(X, EPAISSEUR) == 0
    surface = sp.simplify(FILM.subs(X, EPAISSEUR))
    assert sp.simplify(surface + RHO * G * EPAISSEUR**2 / (2 * MU)) == 0
    assert surface.is_negative


def test_le_profil_normalise_du_film_est_une_demi_parabole() -> None:
    """« Dessinez le profil de vitesse normalise » : `w / w_max = 2 s - s^2`."""
    reduit = sp.Symbol("s", positive=True)
    surface = FILM.subs(X, EPAISSEUR)
    normalise = sp.simplify(FILM.subs(X, reduit * EPAISSEUR) / surface)
    assert sp.simplify(normalise - (2 * reduit - reduit**2)) == 0


def test_le_debit_du_film_croit_comme_le_cube_de_l_epaisseur() -> None:
    """`q = rho g h^3 / (3 mu)` par unite de largeur, vers le bas."""
    debit = debit_du_film(EPAISSEUR, RHO, MU, G)
    par_integration = -sp.integrate(FILM, (X, 0, EPAISSEUR))
    assert sp.simplify(debit - par_integration) == 0
    assert sp.simplify(debit - RHO * G * EPAISSEUR**3 / (3 * MU)) == 0


def test_une_epaisseur_ou_une_viscosite_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="nulle ou negative"):
        vitesse_du_film(X, 0, RHO, MU, G)
    with pytest.raises(ValueError, match="nulle ou negative"):
        debit_du_film(EPAISSEUR, RHO, 0, G)


def test_la_puissance_des_efforts_internes_ne_voit_que_la_deformation() -> None:
    """Planche 57 : `(grad v) : sigma = eps : sigma`, `sigma` etant symetrique."""
    pression = sp.Function("p")(X, Y, Z)
    sigma = tenseur_des_contraintes(pression, CHAMP, MU)
    jacobienne = sp.Matrix(3, 3, lambda i, j: sp.diff(CHAMP[i], ESPACE[j]))
    complet = sum(jacobienne[i, j] * sigma[i, j] for i in range(3) for j in range(3))
    assert sp.simplify(puissance_des_efforts_internes(CHAMP, sigma) - complet) == 0


def test_la_dissipation_visqueuse_d_un_fluide_incompressible_est_positive() -> None:
    """`eps : sigma = 2 mu eps : eps` si `div v = 0` : la pression ne travaille pas."""
    vitesse = (profil_de_couette(), 0, 0)
    sigma = tenseur_des_contraintes(sp.Symbol("p", positive=True), vitesse, VISCOSITE)
    dissipation = puissance_des_efforts_internes(vitesse, sigma)
    attendue = VISCOSITE * (VITESSE_DE_PAROI / HAUTEUR) ** 2
    assert sp.simplify(dissipation - attendue) == 0


def test_l_equation_de_la_chaleur_d_un_fluide_au_repos() -> None:
    """Planche 61 : `rho c_p DT/Dt = div(lambda grad T)`.

    Au repos, c'est l'equation de diffusion : une sinusoide s'y eteint.
    """
    chaleur, conduction, onde = sp.symbols("c_p lambda k", positive=True)
    diffusivite = conduction / (RHO * chaleur)
    temperature = sp.exp(-diffusivite * onde**2 * T) * sp.sin(onde * X)
    repos = (0, 0, 0)
    assert residu_de_la_chaleur(RHO, chaleur, conduction, temperature, repos) == 0


def test_la_chaleur_est_advectee_par_l_ecoulement() -> None:
    """`DT/Dt` porte le terme `v . grad T` : un profil transporte sans se deformer."""
    chaleur, vitesse = sp.symbols("c_p U", positive=True)
    temperature = sp.Function("f")(X - vitesse * T)
    sans_conduction = residu_de_la_chaleur(
        RHO, chaleur, 0, temperature, (vitesse, 0, 0)
    )
    assert sans_conduction == 0
