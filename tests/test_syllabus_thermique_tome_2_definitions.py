"""Le syllabus de thermique, tome 2 : les definitions de la convection.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_2_2013.pdf`, pages 7
a 20 (premiere partie, chapitre 1), lues sur les pages rendues.

Les oracles : les equations (1.1) a (1.11), l'exemple de l'air ralenti
depuis 100 m/s (5 degres), et les constantes des rappels de mecanique des
fluides -- 4,64, 72,94, 2320, 0,03, 82 et 110 degres --, chacune reliee
au contexte des fluides quand il la porte.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.fluides.domain.aerodynamique import Ecoulement, SurfacePortante
from polytech.fluides.domain.fondamentales.bernoulli_generalise import (
    pression_d_arret,
)
from polytech.fluides.domain.fondamentales.continuite import vitesse_dans_la_section
from polytech.fluides.domain.fondamentales.masse_locale import residu_de_continuite
from polytech.fluides.domain.fondamentales.quantite_de_mouvement import (
    residus_de_navier_stokes,
)
from polytech.fluides.domain.grandeurs import MasseVolumique
from polytech.fluides.domain.profil_de_couche_limite import (
    quantite_de_mouvement_reduite,
)
from polytech.fluides.domain.regimes import est_incompressible, nombre_de_mach
from polytech.fluides.domain.turbulence import Decomposition
from polytech.thermique.domain.conduction.fourier import (
    ESPACE,
    INSTANT,
    residu_de_fourier_kirchhoff,
)
from polytech.thermique.domain.convection.couches_limites import (
    REYNOLDS_CRITIQUE_EN_TUBE,
    REYNOLDS_CRITIQUE_SUR_PLAQUE,
    angle_de_decollement,
    epaisseur_de_sous_couche,
    epaisseur_laminaire,
    longueur_d_etablissement,
)
from polytech.thermique.domain.convection.definitions import (
    coefficient_local,
    coefficient_moyen,
    densite_de_flux_turbulente,
    echauffement_cinetique,
    residu_de_l_energie,
)

X, Y, Z = ESPACE


def test_le_coefficient_local_est_une_pente_a_la_paroi() -> None:
    """(1.1) et (1.2) : `phi = K (t_p - t_f)` ; `K` est une pente a la paroi."""
    conductivite, epaisseur = sp.symbols("lambda delta", positive=True)
    paroi, fluide = sp.symbols("t_p t_f", real=True)
    reduit = Y / epaisseur
    champ = fluide + (paroi - fluide) * (1 - sp.Rational(3, 2) * reduit + reduit**3 / 2)
    local = coefficient_local(champ, Y, conductivite, paroi=paroi, fluide=fluide)
    assert sp.simplify(local - 3 * conductivite / (2 * epaisseur)) == 0
    flux = -conductivite * sp.diff(champ, Y).subs(Y, 0)
    assert sp.simplify(flux - local * (paroi - fluide)) == 0


def test_le_coefficient_moyen_d_une_couche_qui_s_epaissit() -> None:
    """(1.3) : `Phi = K_moyen S (t_p - t_f)` ; si `K` varie en `x^-1/2`, le double."""
    longueur = sp.Symbol("L", positive=True)
    abscisse = sp.Symbol("x", positive=True)
    local = 7 / sp.sqrt(abscisse)
    moyen = coefficient_moyen(local, abscisse, longueur)
    assert sp.simplify(moyen - 2 * local.subs(abscisse, longueur)) == 0
    assert coefficient_moyen(sp.Integer(12), abscisse, longueur) == 12


def test_la_continuite_d_un_fluide_incompressible() -> None:
    """(1.4) : `div u = 0`, la forme que prend la conservation de la masse."""
    tourbillon = (-Y, X, sp.Integer(0))
    assert residu_de_continuite(sp.Integer(1000), tourbillon) == 0
    source = (X, Y, Z)
    assert residu_de_continuite(sp.Integer(1000), source) == 3000


def test_l_equation_de_navier_stokes_au_repos_est_l_hydrostatique() -> None:
    """(1.7) : `rho Du/Dtau = -grad p + mu laplacien(u) + rho g`."""
    rho, mu, g = sp.symbols("rho mu g", positive=True)
    repos = (sp.Integer(0),) * 3
    pesanteur = (sp.Integer(0), sp.Integer(0), -g)
    residus = residus_de_navier_stokes(rho, mu, repos, -rho * g * Z, pesanteur)
    assert residus == (0, 0, 0)
    sans_pression = residus_de_navier_stokes(rho, mu, repos, sp.Integer(0), pesanteur)
    assert sans_pression[2] == rho * g


def test_l_equation_de_l_energie_au_repos_est_celle_de_fourier_kirchhoff() -> None:
    """(1.9) : `Dt/Dtau = a laplacien(t)` ; la derivee historique remplace `d/dtau`."""
    a = sp.Symbol("a", positive=True)
    mode = sp.exp(-a * INSTANT) * sp.sin(X)
    repos = (sp.Integer(0),) * 3
    assert residu_de_l_energie(mode, repos, a) == 0
    assert residu_de_l_energie(X**2, repos, a) == residu_de_fourier_kirchhoff(X**2, a)


def test_un_champ_emporte_par_l_ecoulement_diffuse_dans_son_repere() -> None:
    """Le meme mode, vu d'un fluide qui avance a la vitesse `u`."""
    a, u = sp.symbols("a u", positive=True)
    uniforme = (u, sp.Integer(0), sp.Integer(0))
    emporte = sp.exp(-a * INSTANT) * sp.sin(X - u * INSTANT)
    assert residu_de_l_energie(emporte, uniforme, a) == 0
    assert residu_de_fourier_kirchhoff(emporte, a) != 0
    stationnaire = sp.exp(u * X / a)
    assert residu_de_l_energie(stationnaire, uniforme, a) == 0


def test_l_air_ralenti_depuis_100_metres_par_seconde_s_echauffe_de_5_degres() -> None:
    """Page 13 : `u_o^2 / 2 = Dp / rho = c_p Dt`, d'ou `Dt ~ 5 degres`, `Ma ~ 0,3`."""
    assert echauffement_cinetique(100.0, 1005.0) == pytest.approx(5.0, abs=0.05)
    surpression = pression_d_arret(0.0, 1.2, 100.0)
    assert surpression / 1.2 == pytest.approx(1005.0 * 4.975, rel=1e-3)
    mach = nombre_de_mach(100.0, 340.0)
    assert mach == pytest.approx(0.3, abs=0.01)
    assert est_incompressible(mach)
    with pytest.raises(ValueError, match="chaleur massique nulle ou negative"):
        echauffement_cinetique(100.0, 0.0)


def test_la_turbulence_ajoute_un_transport_d_agitation() -> None:
    """Page 14 : `phi = -lambda dt_moy/dy + rho c_p moyenne(v_ag t_ag)`."""
    vitesse = Decomposition((0.2, -0.2, 0.1, -0.1))
    temperature = Decomposition((60.5, 59.5, 60.25, 59.75))
    turbulent = densite_de_flux_turbulente(
        1.2, 1005.0, vitesse, temperature, conductivite=0.026, gradient=0.0
    )
    assert turbulent == pytest.approx(1.2 * 1005.0 * 0.0625)
    moleculaire = densite_de_flux_turbulente(
        1.2,
        1005.0,
        vitesse,
        Decomposition((60.0,) * 4),
        conductivite=0.026,
        gradient=-500.0,
    )
    assert moleculaire == pytest.approx(13.0)
    assert turbulent > 5 * moleculaire


def test_la_couche_limite_laminaire_en_4_64() -> None:
    """Page 16 : `delta / x = 4,64 Re_x^-0,5`, la solution integrale du profil cubique.

    Le bilan de von Karman, `tau_p = rho U^2 d(theta)/dx`, donne
    `delta / x = sqrt(2 g'(0) / (theta/delta)) Re_x^-0,5`.
    """
    s = sp.Symbol("s", nonnegative=True)
    cubique = sp.Rational(3, 2) * s - s**3 / 2
    theta = quantite_de_mouvement_reduite(cubique, s)
    assert theta == sp.Rational(39, 280)
    constante = sp.sqrt(2 * sp.diff(cubique, s).subs(s, 0) / theta)
    assert float(constante) == pytest.approx(4.64, abs=0.005)
    assert epaisseur_laminaire(0.1, 6.7e4) == pytest.approx(
        4.64 * 0.1 / math.sqrt(6.7e4)
    )
    assert epaisseur_laminaire(0.1, 6.7e4) == pytest.approx(1.8e-3, rel=0.02)
    with pytest.raises(ValueError, match="couche limite turbulente"):
        epaisseur_laminaire(1.0, 6e5)


def test_la_transition_et_la_sous_couche_visqueuse() -> None:
    """(1.10) : `Re_crit = 3e5 a 5e5` ; `delta_l / x = 72,94 Re_x^-0,9`."""
    assert REYNOLDS_CRITIQUE_SUR_PLAQUE == (3e5, 5e5)
    sous_couche = epaisseur_de_sous_couche(1.0, 1e6)
    assert sous_couche == pytest.approx(72.94 * 1e6**-0.9)
    turbulente = 0.376 * 1e6**-0.2
    assert sous_couche / turbulente == pytest.approx(194 * 1e6**-0.7, rel=1e-3)
    assert sous_couche < turbulente / 50
    with pytest.raises(ValueError, match="couche limite encore laminaire"):
        epaisseur_de_sous_couche(1.0, 1e5)


def test_l_etablissement_dans_un_tube() -> None:
    """(1.11) : laminaire si `Re_d < 2320` ; `L_e / d = 0,03 Re_d`, soit 70."""
    assert REYNOLDS_CRITIQUE_EN_TUBE == 2320.0
    assert longueur_d_etablissement(0.02, 2320.0) / 0.02 == pytest.approx(70.0, abs=0.5)
    assert longueur_d_etablissement(0.02, 1000.0) == pytest.approx(0.6)
    with pytest.raises(ValueError, match="ecoulement turbulent"):
        longueur_d_etablissement(0.02, 5000.0)
    section = math.pi * 0.02**2 / 4
    moyenne = vitesse_dans_la_section(0.05 / 998.0, section)
    assert moyenne == pytest.approx(0.05 / (998.0 * section))


def test_le_decollement_et_la_chute_de_trainee() -> None:
    """Pages 18 et 19 : 82 degres en laminaire, 110 au-dela de `Re = 4e5`."""
    assert angle_de_decollement(1e4) == 82.0
    assert angle_de_decollement(1e6) == 110.0
    cylindre = SurfacePortante(Ecoulement(MasseVolumique(1.2), 10.0), 0.05 * 1.0)
    avant, apres = cylindre.force(1.2), cylindre.force(0.3)
    assert avant == pytest.approx(1.2 * 0.05 * 1.2 * 10.0**2 / 2)
    assert apres == pytest.approx(avant / 4)
    with pytest.raises(ValueError, match="nombre de Reynolds nul ou negatif"):
        angle_de_decollement(0.0)
