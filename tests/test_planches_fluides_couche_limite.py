"""Planches de mecanique des fluides, chapitre IX : la couche limite.

Source : `MecaniqueFluides/SLIDES/Slides_08_Aerodynamique.pdf`, planches 3
a 18, lues sur les pages rendues.

Les oracles du cours : `delta / L` en `1 / racine(Re)` (planche 5), les
poids `1 / Re` des equations adimensionnelles (planche 6), l'epaisseur a
99 pour cent en `eta = 5` (planche 11), et la relation de compatibilite
a la paroi `nu u_yy = -U dU/dx = (1 / rho) dP/dx` (planche 17).
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.fluides.domain.blasius import (
    resoudre_blasius,
    variable_de_similitude,
)
from polytech.fluides.domain.cinematique.symboles import X, Y
from polytech.fluides.domain.couche_limite import (
    COEFFICIENT_DE_FROTTEMENT_DE_BLASIUS,
    CONSTANTE_DE_BLASIUS,
    RAPPORT_DE_DEPLACEMENT,
    RAPPORT_DE_QUANTITE_DE_MOUVEMENT,
    coefficient_de_frottement_local,
)
from polytech.fluides.domain.lubrification import vitesse_dans_le_film
from polytech.fluides.domain.prandtl import (
    Echelles,
    courbure_a_la_paroi,
    epaisseur_relative_de_couche_limite,
    gradient_de_pression_exterieur,
    nature_du_gradient_de_pression,
    residu_de_prandtl,
)
from polytech.fluides.domain.profil_de_couche_limite import (
    coefficient_de_frottement,
    deplacement_reduit,
    frottement_parietal,
    quantite_de_mouvement_reduite,
)

S = sp.Symbol("s", nonnegative=True)
U, L, NU, RHO = sp.symbols("U L nu rho", positive=True)


def test_l_epaisseur_de_la_couche_limite_decroit_comme_la_racine_du_reynolds() -> None:
    """Planche 5 : advection et diffusion du meme ordre, `delta ~ L / racine(Re)`."""
    assert epaisseur_relative_de_couche_limite(1e4) == pytest.approx(0.01)
    advection = U**2 / L
    delta = sp.Symbol("delta", positive=True)
    (solution,) = sp.solve(sp.Eq(advection, NU * U / delta**2), delta)
    assert sp.simplify(solution / L - 1 / sp.sqrt(U * L / NU)) == 0
    with pytest.raises(ValueError, match="Reynolds nul ou negatif"):
        epaisseur_relative_de_couche_limite(0.0)


def test_les_poids_des_equations_adimensionnelles_de_la_planche_6() -> None:
    """`1 / Re` devant `d2u*/dx*2`, rien devant `d2u*/dy*2`."""
    echelles = Echelles(vitesse=U, longueur=L, viscosite=NU)
    reynolds = echelles.reynolds()
    assert reynolds == U * L / NU
    reduit = sp.exp(-Y) * sp.sin(X)
    reel = echelles.longitudinale(reduit)
    norme = L / U**2
    etoile = {X: X / L, Y: Y * sp.sqrt(reynolds) / L}
    selon_x = sp.diff(reduit, X, 2).subs(etoile, simultaneous=True) / reynolds
    selon_y = sp.diff(reduit, Y, 2).subs(etoile, simultaneous=True)
    assert sp.simplify(NU * sp.diff(reel, X, 2) * norme - selon_x) == 0
    assert sp.simplify(NU * sp.diff(reel, Y, 2) * norme - selon_y) == 0


def test_la_vitesse_transversale_est_plus_petite_d_un_facteur_racine_de_re() -> None:
    """`v* = (v / U) racine(Re)` : la continuite garde ses deux termes."""
    echelles = Echelles(vitesse=U, longueur=L, viscosite=NU)
    courant = sp.exp(-Y) * sp.sin(X)
    u_reduit, v_reduit = sp.diff(courant, Y), -sp.diff(courant, X)
    reel_u = echelles.longitudinale(u_reduit)
    reel_v = echelles.transversale(v_reduit)
    assert sp.simplify(sp.diff(reel_u, X) + sp.diff(reel_v, Y)) == 0
    rapport = sp.simplify(reel_v / echelles.longitudinale(v_reduit))
    assert rapport == 1 / sp.sqrt(echelles.reynolds())


def test_le_profil_d_aspiration_resout_l_equation_de_prandtl() -> None:
    """Une solution exacte : `u = U (1 - exp(-v_0 y / nu))`, `v = -v_0`."""
    aspiration = sp.Symbol("v_0", positive=True)
    u = U * (1 - sp.exp(-aspiration * Y / NU))
    assert residu_de_prandtl(u, -aspiration, U, NU) == 0
    assert residu_de_prandtl(u, aspiration, U, NU) != 0


def test_au_bord_de_la_couche_prandtl_se_reduit_a_bernoulli() -> None:
    """Hors de la couche, `u = U_e(x)` et `v = -y U_e'` : le residu s'annule."""
    exterieure = U * (1 + X / L)
    transversale = -Y * sp.diff(exterieure, X)
    assert residu_de_prandtl(exterieure, transversale, exterieure, NU) == 0


def test_bernoulli_donne_le_gradient_de_pression_exterieur() -> None:
    """Planche 10 : `p_e + rho U_e^2 / 2` constant, donc `dp_e/dx = -rho U_e U_e'`."""
    exterieure = U * (1 + X / L)
    pression = sp.Symbol("p_0") - RHO * exterieure**2 / 2
    gradient = gradient_de_pression_exterieur(RHO, exterieure)
    assert sp.simplify(gradient - sp.diff(pression, X)) == 0


def test_la_courbure_a_la_paroi_suit_le_gradient_de_pression() -> None:
    """Planche 17 : `nu u_yy(0) = -U dU/dx = (1 / rho) dP/dx`."""
    exterieure = U * (1 + X / L)
    courbure = courbure_a_la_paroi(exterieure, NU)
    gradient = gradient_de_pression_exterieur(RHO, exterieure)
    assert sp.simplify(NU * courbure - gradient / RHO) == 0
    assert sp.simplify(NU * courbure + exterieure * sp.diff(exterieure, X)) == 0


def test_le_film_lubrifie_verifie_la_meme_compatibilite() -> None:
    """Un lien : `mu u_yy = dp/dx` dans le film du chapitre VII."""
    epaisseur, viscosite, pente = sp.symbols("h mu G", positive=True)
    profil = vitesse_dans_le_film(Y, epaisseur, U, viscosite, pente)
    assert sp.simplify(viscosite * sp.diff(profil, Y, 2).subs(Y, 0) - pente) == 0


def test_les_cinq_cas_de_la_planche_17() -> None:
    assert nature_du_gradient_de_pression(-2.0) == "favorable"
    assert nature_du_gradient_de_pression(0.0) == "nul"
    assert nature_du_gradient_de_pression(3.0) == "adverse"


BLASIUS = resoudre_blasius()


def test_la_variable_de_similitude_de_blasius() -> None:
    """Planche 11 : `eta = y racine(U / (nu x))`."""
    assert variable_de_similitude(0.002, 2.0, 1e-6, 0.5) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="abscisse nulle ou negative"):
        variable_de_similitude(0.002, 2.0, 1e-6, 0.0)


def test_l_epaisseur_a_99_pour_cent_tombe_en_eta_egal_cinq() -> None:
    """Planche 11 : « 99% boundary layer thickness » a `eta = 5`."""
    assert BLASIUS.epaisseur_a_99_pour_cent() == pytest.approx(4.91, abs=0.01)
    assert BLASIUS.epaisseur_a_99_pour_cent() == pytest.approx(
        CONSTANTE_DE_BLASIUS, abs=0.1
    )
    assert BLASIUS.vitesse_reduite(5.0) == pytest.approx(0.99, abs=0.002)
    assert BLASIUS.vitesse_reduite(0.0) == 0
    assert BLASIUS.vitesse_reduite(9.0) == pytest.approx(1.0, abs=1e-5)


def test_la_table_de_blasius_s_interpole_et_refuse_ce_qu_elle_ne_porte_pas() -> None:
    """Pres de la paroi, `f' = f''(0) eta` : la « slope at the wall » de la figure."""
    pente = BLASIUS.courbure_a_la_paroi()
    assert BLASIUS.vitesse_reduite(0.01) == pytest.approx(0.01 * pente, rel=1e-3)
    with pytest.raises(ValueError, match="eta hors de la table"):
        BLASIUS.vitesse_reduite(10.5)
    with pytest.raises(ValueError, match="eta hors de la table"):
        BLASIUS.vitesse_reduite(-0.5)


def test_la_solution_resout_l_equation_de_blasius() -> None:
    """`2 f''' + f f'' = 0`, verifiee par differences sur la table."""
    pas = BLASIUS.pas
    for rang in (50, 200, 450):
        _, f, _, courbure = BLASIUS.table[rang]
        avant, apres = BLASIUS.table[rang - 1][3], BLASIUS.table[rang + 1][3]
        troisieme = (apres - avant) / (2 * pas)
        assert 2 * troisieme + f * courbure == pytest.approx(0.0, abs=1e-4)


def test_blasius_rend_les_constantes_de_la_bibliotheque() -> None:
    """Un lien : 0,664, 0,344 et 0,133 sortent de l'equation, pas d'une table."""
    assert BLASIUS.courbure_a_la_paroi() == pytest.approx(0.332, abs=5e-4)
    frottement = 2 * BLASIUS.courbure_a_la_paroi()
    assert frottement == pytest.approx(COEFFICIENT_DE_FROTTEMENT_DE_BLASIUS, abs=1e-3)
    deplacement = BLASIUS.deplacement() / CONSTANTE_DE_BLASIUS
    assert deplacement == pytest.approx(RAPPORT_DE_DEPLACEMENT, abs=1e-3)
    quantite = BLASIUS.quantite_de_mouvement() / CONSTANTE_DE_BLASIUS
    assert quantite == pytest.approx(RAPPORT_DE_QUANTITE_DE_MOUVEMENT, abs=1e-3)
    assert BLASIUS.quantite_de_mouvement() == pytest.approx(frottement, abs=1e-3)


def test_le_frottement_de_blasius_par_la_definition_de_la_planche_16() -> None:
    """`C_f = tau_w / (rho U^2 / 2)` avec `tau_w = mu du/dy` a la paroi."""
    vitesse, abscisse, cinematique, masse = 0.5, 0.5, 1e-6, 1000.0
    pente = vitesse * BLASIUS.courbure_a_la_paroi()
    pente *= math.sqrt(vitesse / (cinematique * abscisse))
    contrainte = masse * cinematique * pente
    local = coefficient_de_frottement(contrainte, masse, vitesse)
    reynolds = vitesse * abscisse / cinematique
    assert local == pytest.approx(coefficient_de_frottement_local(reynolds), rel=1e-3)


def test_les_epaisseurs_integrales_d_un_profil_lineaire() -> None:
    """Planche 15 : `delta*` integre `1 - g`, `theta` integre `g (1 - g)`."""
    assert deplacement_reduit(S, S) == sp.Rational(1, 2)
    assert quantite_de_mouvement_reduite(S, S) == sp.Rational(1, 6)


def test_les_epaisseurs_integrales_d_un_profil_parabolique() -> None:
    profil = 2 * S - S**2
    assert deplacement_reduit(profil, S) == sp.Rational(1, 3)
    assert quantite_de_mouvement_reduite(profil, S) == sp.Rational(2, 15)
    pas = 10_000
    somme = sum(1 - (2 * k / pas - (k / pas) ** 2) for k in range(pas)) / pas
    assert somme == pytest.approx(1 / 3, abs=1e-4)


def test_le_frottement_parietal_d_un_profil_semblable() -> None:
    """Planche 16 : `tau_w = mu du/dy` en `y = 0`, soit `mu U_e g'(0) / delta`."""
    profil = 2 * S - S**2
    contrainte = frottement_parietal(profil, S, viscosite=1e-3, vitesse=2.0)
    assert contrainte(0.01) == pytest.approx(1e-3 * 2.0 * 2 / 0.01)
    with pytest.raises(ValueError, match="epaisseur nulle ou negative"):
        contrainte(0.0)


def test_un_profil_qui_ne_part_pas_de_zero_est_refuse() -> None:
    with pytest.raises(ValueError, match="condition de non-glissement"):
        deplacement_reduit(1 - S / 2, S)
    with pytest.raises(ValueError, match="vitesse exterieure"):
        quantite_de_mouvement_reduite(S / 2, S)
    with pytest.raises(ValueError, match="vitesse nulle ou negative"):
        coefficient_de_frottement(1.0, 1000.0, 0.0)
