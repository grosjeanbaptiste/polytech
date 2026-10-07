"""Les treize notions d'electromagnetisme, et leurs quatre ancrages.

Trois viennent des transcriptions manuscrites du depot -- les seuls
documents chiffres lisibles de la matiere, les fascicules d'origine etant
des scans dont l'OCR ne tirait que du charabia :

* Thevenin/Norton : `110 V` en serie avec `1 ohm`, en parallele avec
  `10 A`, donne `120 V` a vide. Le detail qui fait l'exercice est que le
  courant de la source traverse `R` en sens **oppose**, d'ou `+10` ;
* circuit magnetique : `B = 1,6 T` lu sur la courbe donne `H = 70 A/m`,
  `Phi = 1,6 x 8.10^-4 = 0,00128 Wb`, et
  `70 x 2 pi x 0,16 / 1000 = 0,07037 A` ;
* Poisson en cylindriques, facteur `1/rho` compris.

Le quatrieme est intrinseque, et plus fort que les trois : les constantes
du vide sont declarees **independamment**, donc
`c = 1 / racine(epsilon_0 mu_0)` est une verification et non une
definition. Une faute de transcription sur l'une des trois se verrait la,
et nulle part ailleurs.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain import energie_et_forces, maxwell, onde
from polytech.electromagnetisme.domain.circuits import (
    sinusoidaux,
    stationnaires,
    transitoires,
)
from polytech.electromagnetisme.domain.constantes import VITESSE_DE_LA_LUMIERE
from polytech.electromagnetisme.domain.effet_pelliculaire import (
    epaisseur_de_peau,
    est_negligeable,
    rapport_des_epaisseurs,
)
from polytech.electromagnetisme.domain.electrostatique import avancee, lois
from polytech.electromagnetisme.domain.magnetostatique import (
    circuits_magnetiques,
    couplage,
    induction,
)
from polytech.electromagnetisme.domain.magnetostatique import lois as magnetiques
from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    PERMEABILITE,
)
from polytech.electromagnetisme.domain.symboles import (
    ABSCISSE,
    COTE,
    ORDONNEE,
    PULSATION,
    RAYON,
    TEMPS,
)

ZERO = sp.Integer(0)

# --- EM-II.1 : lois fondamentales du champ electrostatique -----------------


def test_la_divergence_cylindrique_porte_son_facteur_un_sur_rho() -> None:
    """L'oublier ne se voit pas : l'expression reste homogene et devient fausse.

    Sur `E_rho = 1/rho`, la bonne divergence est nulle -- c'est le champ
    du fil charge. Sans le facteur `1/rho`, on trouverait `-1/rho^2`.
    """
    champ = (1 / RAYON, ZERO, ZERO)
    assert lois.divergence_cylindrique(champ) == 0


def test_un_champ_radial_uniforme_a_une_divergence_non_nulle() -> None:
    """Controle de non-degenerescence : la formule ne rend pas zero partout."""
    assert lois.divergence_cylindrique((RAYON, ZERO, ZERO)) == 2


def test_un_champ_electrostatique_a_un_rotationnel_nul() -> None:
    """Seconde propriete fondamentale : c'est elle qui cree le potentiel."""
    assert lois.est_electrostatique((ABSCISSE, ORDONNEE, COTE))


def test_un_champ_a_rotationnel_non_nul_n_est_pas_electrostatique() -> None:
    """Le dire avant de chercher un potentiel qui n'existe pas."""
    assert not lois.est_electrostatique((-ORDONNEE, ABSCISSE, ZERO))


def test_le_champ_de_coulomb_decroit_comme_l_inverse_du_carre() -> None:
    proche = lois.champ_de_coulomb(1e-9, 0.1)
    loin = lois.champ_de_coulomb(1e-9, 0.2)
    assert float(proche / loin) == pytest.approx(4.0)


def test_une_distance_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="distance nulle"):
        lois.champ_de_coulomb(1e-9, 0.0)


# --- EM-II.2 : electrostatique avancee -------------------------------------


def test_un_potentiel_harmonique_satisfait_laplace() -> None:
    """`V = x^2 - y^2` : le cas d'ecole des planches."""
    assert avancee.satisfait_laplace(ABSCISSE**2 - ORDONNEE**2)


def test_poisson_relie_le_laplacien_a_la_densite() -> None:
    """Un potentiel quadratique implique une charge uniforme."""
    densite = sp.Symbol("eta", real=True)
    equation = avancee.equation_de_poisson(ABSCISSE**2, densite)
    assert sp.simplify(equation.lhs - 2) == 0


def test_la_charge_image_annule_le_potentiel_sur_le_plan() -> None:
    """C'est la condition aux limites que la methode sert a satisfaire.

    Et par unicite -- « probleme fondamental » des planches -- la
    satisfaire suffit : c'est **la** solution, pas une solution.
    """
    assert avancee.potentiel_sur_le_plan(1e-9, 0.05) == pytest.approx(0.0)


def test_un_dielectrique_renforce_la_capacite() -> None:
    nue = avancee.capacite_plane(0.01, 1e-3)
    garnie = avancee.capacite_plane(0.01, 1e-3, relative=4.0)
    assert garnie == pytest.approx(4 * nue, rel=1e-9, abs=0)


def test_une_permittivite_relative_sous_l_unite_est_refusee() -> None:
    """Elle affaiblirait la capacite : c'est une donnee mal lue."""
    with pytest.raises(ValueError, match="permittivite relative sous l'unite"):
        avancee.permittivite(0.5)


def test_une_geometrie_de_condensateur_invalide_est_refusee() -> None:
    with pytest.raises(ValueError, match="geometrie invalide"):
        avancee.capacite_plane(0.0, 1e-3)


def test_un_potentiel_sur_le_plan_a_distance_nulle_est_refuse() -> None:
    with pytest.raises(ValueError, match="distance nulle"):
        avancee.potentiel_sur_le_plan(1e-9, 0.0)


# --- EM-III.1 : lois fondamentales du champ magnetique ---------------------


def test_le_flux_magnetique_est_toujours_conservatif() -> None:
    """`div B = 0` : l'inverse exact de l'electrostatique."""
    assert magnetiques.est_un_champ_magnetique((-ORDONNEE, ABSCISSE, ZERO))


def test_un_champ_a_divergence_non_nulle_n_est_pas_magnetique() -> None:
    """Il impliquerait une charge magnetique, qui n'existe pas."""
    assert not magnetiques.est_un_champ_magnetique((ABSCISSE, ZERO, ZERO))


def test_l_induction_du_fil_decroit_comme_l_inverse_de_la_distance() -> None:
    """Et non comme l'inverse du carre : c'est la difference avec Coulomb."""
    proche = magnetiques.induction_du_fil_infini(10.0, 0.01)
    loin = magnetiques.induction_du_fil_infini(10.0, 0.02)
    assert proche / loin == pytest.approx(2.0)


def test_une_distance_au_fil_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="distance au fil nulle"):
        magnetiques.induction_du_fil_infini(10.0, 0.0)


def test_une_section_nulle_n_a_pas_de_flux() -> None:
    with pytest.raises(ValueError, match="section nulle"):
        magnetiques.flux_magnetique(1.0, 0.0)


def test_une_permeabilite_relative_sous_l_unite_est_refusee() -> None:
    with pytest.raises(ValueError, match="permeabilite relative sous l'unite"):
        magnetiques.champ_depuis_l_induction(1.0, relative=0.5)


# --- EM-III.2 : induction ---------------------------------------------------


def test_la_loi_de_faraday_porte_le_signe_de_lenz() -> None:
    """`e = -dPhi/dt` : l'effet s'oppose a la cause."""
    flux = sp.Symbol("Phi_0", positive=True) * TEMPS
    assert induction.force_electromotrice(flux) < 0


def test_un_flux_constant_n_induit_rien() -> None:
    assert induction.force_electromotrice(sp.Integer(5)) == 0


def test_la_fem_d_un_entrefer_mobile_decroit_avec_le_temps() -> None:
    """Interrogation cotee 2022-2023 (Charleroi), question 2, derniere ligne.

    `E(t) = mu_0 v N^2 I S / (2 (l_e + v t)^2)` : l'entrefer s'ouvre a
    vitesse constante, mais le flux varie comme l'inverse de l'entrefer.
    La force electromotrice n'est donc **pas** constante.
    """
    n, i, s, le, v = sp.symbols("N I S l_e v", positive=True)
    fem = induction.fem_d_un_entrefer_mobile(n, i, s, le, v)
    attendue = PERMEABILITE * v * n**2 * i * s / (2 * (le + v * TEMPS) ** 2)
    assert sp.simplify(fem - attendue) == 0
    assert not induction.est_constante(fem)
    pente = sp.diff(fem, TEMPS).subs(TEMPS, sp.Symbol("tau", positive=True))
    assert sp.simplify(pente).is_negative
    assert fem.subs(TEMPS, 0) == PERMEABILITE * v * n**2 * i * s / (2 * le**2)


def test_la_fem_d_un_entrefer_mobile_vient_du_flux() -> None:
    """Second chemin : `-N dPhi/dt`, avec deux entrefers de `l_e + v t`."""
    n, i, s, le, v = sp.symbols("N I S l_e v", positive=True)
    flux = induction.flux_a_entrefers_dominants(n, i, s, 2 * (le + v * TEMPS))
    par_faraday = induction.force_electromotrice(n * flux)
    fem = induction.fem_d_un_entrefer_mobile(n, i, s, le, v)
    assert sp.simplify(fem - par_faraday) == 0
    assert sp.simplify(fem.subs(v, 0)) == 0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0, 1.0, 8e-4, 1e-3, 0.5), "nombre de spires"),
        ((10, 1.0, 0.0, 1e-3, 0.5), "section nulle"),
        ((10, 1.0, 8e-4, 0.0, 0.5), "entrefer initial nul"),
    ],
)
def test_une_geometrie_d_entrefer_invalide_est_refusee(
    arguments: tuple[float, float, float, float, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        induction.fem_d_un_entrefer_mobile(*arguments)


def test_une_longueur_de_barre_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="longueur de barre"):
        induction.fem_de_deplacement(1.0, 0.0, 1.0)


# --- EM-III.3 : Thevenin et Norton -----------------------------------------


def test_le_dipole_de_l_exercice_un_donne_bien_cent_vingt_volts() -> None:
    """`110 + 1 x 10 = 120 V` : le courant traverse R en sens oppose."""
    dipole = stationnaires.source_de_tension_et_source_de_courant(110.0, 1.0, 10.0)
    assert dipole.tension_a_vide == pytest.approx(120.0)


def test_le_courant_de_court_circuit_de_l_exercice_un() -> None:
    """`110/1 + 10 = 120 A`, et la resistance equivalente vaut bien 1 ohm."""
    dipole = stationnaires.source_de_tension_et_source_de_courant(110.0, 1.0, 10.0)
    assert dipole.courant_de_court_circuit() == pytest.approx(120.0)
    assert dipole.tension_a_vide / dipole.courant_de_court_circuit() == pytest.approx(
        1.0,
    )


def test_thevenin_et_norton_sont_reciproques() -> None:
    """Convertir dans un sens puis dans l'autre doit rendre le dipole initial."""
    depart = stationnaires.Thevenin(tension_a_vide=120.0, resistance=1.0)
    retour = depart.en_norton().en_thevenin()
    assert retour.tension_a_vide == pytest.approx(depart.tension_a_vide)
    assert retour.resistance == pytest.approx(depart.resistance)


def test_la_tension_sur_une_charge_tend_vers_la_tension_a_vide() -> None:
    """Une charge infinie, c'est le circuit ouvert."""
    dipole = stationnaires.Thevenin(tension_a_vide=120.0, resistance=1.0)
    assert dipole.tension_sur(1e9) == pytest.approx(120.0, rel=1e-6)
    assert dipole.tension_sur(0.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "fabrique",
    [
        lambda: stationnaires.Thevenin(tension_a_vide=1.0, resistance=0.0),
        lambda: stationnaires.Norton(courant_de_court_circuit=1.0, resistance=0.0),
    ],
)
def test_une_resistance_nulle_est_refusee(fabrique: object) -> None:
    """Un dipole de resistance nulle n'a pas d'equivalent dual."""
    assert callable(fabrique)
    with pytest.raises(ValueError, match="nulle ou negative"):
        fabrique()


def test_une_charge_negative_est_refusee() -> None:
    dipole = stationnaires.Thevenin(tension_a_vide=1.0, resistance=1.0)
    with pytest.raises(ValueError, match="resistance de charge negative"):
        dipole.tension_sur(-1.0)


# --- EM-III.4 : couplage magnetique ----------------------------------------


def test_le_coefficient_de_couplage_reste_sous_l_unite() -> None:
    paire = couplage.CouplageMagnetique(premiere=1e-3, seconde=4e-3, mutuelle=1e-3)
    assert paire.coefficient() == pytest.approx(0.5)


def test_un_couplage_parfait_est_reconnu() -> None:
    paire = couplage.CouplageMagnetique(premiere=1e-3, seconde=4e-3, mutuelle=2e-3)
    assert paire.est_parfait()


def test_les_deux_sens_d_enroulement_donnent_deux_inductances() -> None:
    """`L1 + L2 +- 2M` : le sens n'est pas une convention tacite."""
    paire = couplage.CouplageMagnetique(premiere=1e-3, seconde=4e-3, mutuelle=1e-3)
    assert paire.en_serie(aidant=True) == pytest.approx(7e-3)
    assert paire.en_serie(aidant=False) == pytest.approx(3e-3)


def test_un_couplage_superieur_a_l_unite_est_refuse() -> None:
    """Il donnerait une inductance equivalente negative."""
    with pytest.raises(ValueError, match="couplage superieur a l'unite"):
        couplage.CouplageMagnetique(premiere=1e-3, seconde=1e-3, mutuelle=2e-3)


def test_une_inductance_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="inductance premiere"):
        couplage.CouplageMagnetique(premiere=0.0, seconde=1e-3, mutuelle=0.0)


# --- EM-III.5 : circuits magnetiques ---------------------------------------

BRANCHE = circuits_magnetiques.BrancheFerromagnetique(
    induction=1.6,
    champ_lu=70.0,
    section=8e-4,
    longueur=2 * math.pi * 0.16,
)


def test_le_flux_de_la_branche_est_celui_du_corrige() -> None:
    """`1,6 x (2 x 4).10^-4 = 0,00128 Wb`."""
    assert BRANCHE.flux() == pytest.approx(0.00128)


def test_le_courant_de_la_branche_est_celui_du_corrige() -> None:
    """`70 x 2 pi x 0,16 = 1000 I` donne `I = 0,07037 A`."""
    assert BRANCHE.courant_pour(1000) == pytest.approx(0.07037, abs=1e-5)


def test_la_reluctance_croit_avec_la_longueur_et_decroit_avec_la_section() -> None:
    longue = circuits_magnetiques.reluctance(0.2, 1e-4)
    courte = circuits_magnetiques.reluctance(0.1, 1e-4)
    large = circuits_magnetiques.reluctance(0.1, 2e-4)
    assert longue == pytest.approx(2 * courte)
    assert large == pytest.approx(courte / 2)


def test_hopkinson_est_l_analogue_de_la_loi_d_ohm() -> None:
    flux = circuits_magnetiques.flux_par_hopkinson(1000.0, 2e6)
    assert flux == pytest.approx(5e-4)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0.0, 1e-4), "geometrie invalide"),
        ((0.1, 0.0), "geometrie invalide"),
    ],
)
def test_une_geometrie_de_reluctance_invalide_est_refusee(
    arguments: tuple[float, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        circuits_magnetiques.reluctance(*arguments)


def test_une_permeabilite_relative_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="permeabilite relative nulle"):
        circuits_magnetiques.reluctance(0.1, 1e-4, relative=0.0)


def test_une_reluctance_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="reluctance nulle"):
        circuits_magnetiques.flux_par_hopkinson(1000.0, 0.0)


def test_une_branche_de_geometrie_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="geometrie invalide"):
        circuits_magnetiques.BrancheFerromagnetique(
            induction=1.6,
            champ_lu=70.0,
            section=0.0,
            longueur=1.0,
        )


def test_une_branche_sans_spire_est_refusee() -> None:
    with pytest.raises(ValueError, match="nombre de spires"):
        BRANCHE.courant_pour(0)


# --- EM-IV.1 : les equations de Maxwell ------------------------------------


def test_la_conservation_de_la_charge_se_deduit_de_maxwell() -> None:
    """`div J + d eta/dt = 0` sur un couple qui la satisfait.

    Avec `eta = eta_0 x t`, la charge croit en tout point, donc le courant
    doit converger : `div J = -eta_0 x`, ce que donne
    `J_x = -eta_0 x^2 / 2`.
    """
    amplitude = sp.Symbol("eta_0", real=True)
    densite = amplitude * ABSCISSE * TEMPS
    courant = (-amplitude * ABSCISSE**2 / 2, ZERO, ZERO)
    assert maxwell.conserve_la_charge(courant, densite)


def test_un_courant_incompatible_avec_sa_charge_est_detecte() -> None:
    """La garde la plus utile du chapitre : elle echoue si un terme est faux."""
    assert not maxwell.conserve_la_charge((ABSCISSE, ZERO, ZERO), sp.sympify(ABSCISSE))


def test_le_flux_magnetique_de_maxwell_est_toujours_nul() -> None:
    assert maxwell.residu_de_gauss_magnetique((-ORDONNEE, ABSCISSE, ZERO)) == 0


def test_faraday_rend_un_residu_par_composante() -> None:
    """Un champ nul et une induction qui varie : Faraday est viole."""
    attendu = 3
    residus = maxwell.residus_de_faraday((ZERO, ZERO, ZERO), (ZERO, ZERO, TEMPS))
    assert len(residus) == attendu
    assert residus == (0, 0, 1)


def test_ampere_maxwell_porte_le_courant_de_deplacement() -> None:
    """Le dernier terme `dD/dt` est l'apport propre de Maxwell.

    Avec un champ magnetique nul et un deplacement qui croit en `t`, le
    residu vaut `-1` : c'est bien le courant de deplacement qui manque.
    """
    residus = maxwell.residus_d_ampere_maxwell(
        (ZERO, ZERO, ZERO),
        (ZERO, ZERO, ZERO),
        (TEMPS, ZERO, ZERO),
    )
    assert sp.simplify(residus[0] + 1) == 0


def test_gauss_electrique_relie_la_divergence_a_la_charge() -> None:
    """Un deplacement `D_x = x` implique une densite unite."""
    assert (
        maxwell.residu_de_gauss_electrique((ABSCISSE, ZERO, ZERO), sp.Integer(1)) == 0
    )


# --- EM-IV.2 : energie et forces -------------------------------------------


def test_les_deux_methodes_de_calcul_de_force_s_opposent() -> None:
    """A courant constant et a flux constant : les signes sont inverses.

    C'est le piege du chapitre, et le verifier evite de le reinventer.
    """
    inductance = 1e-3 / (1 + energie_et_forces.DEPLACEMENT)
    energie = energie_et_forces.energie_d_une_inductance(inductance, sp.Integer(2))
    assert energie_et_forces.les_deux_signes_s_opposent(energie)


def test_l_energie_d_une_inductance_croit_comme_le_carre_du_courant() -> None:
    energie = energie_et_forces.energie_d_une_inductance(sp.Integer(2), sp.Integer(3))
    assert sp.simplify(energie - 9) == 0


def test_l_energie_d_un_condensateur_croit_comme_le_carre_de_la_tension() -> None:
    energie = energie_et_forces.energie_d_un_condensateur(sp.Integer(2), sp.Integer(3))
    assert sp.simplify(energie - 9) == 0


@pytest.mark.parametrize(
    ("fonction", "message"),
    [
        (energie_et_forces.densite_d_energie_electrique, "permittivite nulle"),
        (energie_et_forces.densite_d_energie_magnetique, "permeabilite nulle"),
    ],
)
def test_une_constante_de_milieu_nulle_est_refusee(
    fonction: object,
    message: str,
) -> None:
    assert callable(fonction)
    with pytest.raises(ValueError, match=message):
        fonction(0.0, 1.0)


# --- EM-IV.3 : regimes transitoires ----------------------------------------


def test_la_reponse_du_premier_ordre_satisfait_son_equation() -> None:
    """Controle intrinseque : la formule generale est reinjectee."""
    constante = sp.Symbol("tau", positive=True)
    reponse = transitoires.reponse_du_premier_ordre(
        sp.Integer(0),
        sp.Integer(5),
        constante,
    )
    assert transitoires.satisfait_son_equation(reponse, constante, sp.Integer(5))


def test_la_reponse_part_de_sa_valeur_initiale_et_tend_vers_la_finale() -> None:
    constante = sp.Symbol("tau", positive=True)
    reponse = transitoires.reponse_du_premier_ordre(
        sp.Integer(2),
        sp.Integer(5),
        constante,
    )
    assert sp.simplify(reponse.subs(TEMPS, 0) - 2) == 0
    assert sp.limit(reponse, TEMPS, sp.oo) == 5


def test_les_deux_constantes_de_temps_du_chapitre() -> None:
    assert transitoires.constante_de_temps_rl(10.0, 1e-3) == pytest.approx(1e-4)
    assert transitoires.constante_de_temps_rc(10.0, 1e-6) == pytest.approx(1e-5)


def test_l_amortissement_et_la_pulsation_propre_du_rlc() -> None:
    assert float(transitoires.pulsation_propre(1e-3, 1e-6)) == pytest.approx(
        1 / math.sqrt(1e-9),
    )
    assert float(transitoires.amortissement_rlc(2.0, 1e-3, 1e-6)) == pytest.approx(
        math.sqrt(1e-6 / 1e-3),
    )


@pytest.mark.parametrize(
    ("fonction", "arguments"),
    [
        (transitoires.constante_de_temps_rl, (0.0, 1e-3)),
        (transitoires.constante_de_temps_rc, (10.0, 0.0)),
        (transitoires.pulsation_propre, (0.0, 1e-6)),
        (transitoires.amortissement_rlc, (1.0, 0.0, 1e-6)),
    ],
)
def test_un_composant_nul_est_refuse(
    fonction: object,
    arguments: tuple[float, ...],
) -> None:
    assert callable(fonction)
    with pytest.raises(ValueError, match="positif"):
        fonction(*arguments)


# --- EM-IV.4 : circuits sinusoidaux ----------------------------------------


def test_la_resonance_serie_annule_la_partie_imaginaire() -> None:
    """C'est la definition du cours, et elle se verifie exactement."""
    inductance, capacite = sp.Rational(1, 1000), sp.Rational(1, 1000000)
    serie = sinusoidaux.en_serie(
        sinusoidaux.impedance_resistive(sp.Integer(10)),
        sinusoidaux.impedance_inductive(inductance),
        sinusoidaux.impedance_capacitive(capacite),
    )
    resonance = sinusoidaux.pulsation_de_resonance_serie(inductance, capacite)
    assert sinusoidaux.est_resonant(serie, resonance)


def test_hors_resonance_l_impedance_garde_une_partie_imaginaire() -> None:
    serie = sinusoidaux.en_serie(
        sinusoidaux.impedance_resistive(sp.Integer(10)),
        sinusoidaux.impedance_inductive(sp.Rational(1, 1000)),
    )
    assert not sinusoidaux.est_resonant(serie, sp.Integer(1000))


def test_deux_resistances_en_parallele_donnent_la_moitie() -> None:
    parallele = sinusoidaux.en_parallele(sp.Integer(10), sp.Integer(10))
    assert sp.simplify(parallele - 5) == 0


def test_l_impedance_inductive_croit_avec_la_pulsation() -> None:
    impedance = sinusoidaux.impedance_inductive(sp.Rational(1, 1000))
    assert sp.simplify(impedance / (sp.I * PULSATION)) == sp.Rational(1, 1000)


def test_le_facteur_de_qualite_croit_quand_la_resistance_decroit() -> None:
    fort = sinusoidaux.facteur_de_qualite(sp.Integer(1), sp.Integer(1), sp.Integer(1))
    faible = sinusoidaux.facteur_de_qualite(
        sp.Integer(10), sp.Integer(1), sp.Integer(1)
    )
    assert fort > faible


def test_une_capacite_nulle_n_a_pas_d_impedance() -> None:
    """Elle serait infinie a toute pulsation : le dire plutot que la composer."""
    with pytest.raises(ValueError, match="capacite nulle"):
        sinusoidaux.impedance_capacitive(sp.Integer(0))


def test_une_mise_en_serie_vide_est_refusee() -> None:
    """Zero impedance en serie rendrait zero, c'est-a-dire un court-circuit."""
    with pytest.raises(ValueError, match="aucune impedance"):
        sinusoidaux.en_serie()


def test_une_mise_en_parallele_vide_est_refusee() -> None:
    """Et celle-ci diviserait par zero."""
    with pytest.raises(ValueError, match="aucune impedance"):
        sinusoidaux.en_parallele()


def test_un_facteur_de_qualite_sans_resistance_est_refuse() -> None:
    """Il serait infini : un circuit sans perte n'amortit jamais."""
    with pytest.raises(ValueError, match="resistance nulle"):
        sinusoidaux.facteur_de_qualite(sp.Integer(0), sp.Integer(1), sp.Integer(1))


# --- EM-T21 : l'equation d'onde ---------------------------------------------


def test_la_celerite_deduite_des_deux_constantes_vaut_bien_c() -> None:
    """Le recoupement intrinseque : trois constantes declarees separement.

    `c = 1 / racine(epsilon_0 mu_0)` n'est donc pas une definition mais
    une verification, et une faute de transcription sur l'une des trois se
    verrait ici -- et nulle part ailleurs.
    """
    assert onde.celerite_dans_le_vide() == pytest.approx(
        VITESSE_DE_LA_LUMIERE,
        rel=1e-9,
    )


def test_l_impedance_du_vide_vaut_trois_cent_soixante_dix_sept_ohms() -> None:
    """Second recoupement des memes constantes, par leur rapport."""
    assert onde.impedance_du_vide() == pytest.approx(376.730, abs=0.001)


def test_une_onde_progressive_satisfait_l_equation_d_onde() -> None:
    celerite = sp.Symbol("c", positive=True)
    pulsation = sp.Symbol("w", positive=True)
    amplitude = onde.onde_progressive(celerite, pulsation)
    assert onde.est_une_onde(amplitude, celerite)


def test_une_fonction_quelconque_n_est_pas_une_onde() -> None:
    celerite = sp.Symbol("c", positive=True)
    assert not onde.est_une_onde(ABSCISSE * TEMPS**2, celerite)


def test_la_longueur_d_onde_decroit_avec_la_frequence() -> None:
    assert onde.longueur_d_onde(3e8, 1e6) == pytest.approx(300.0)


def test_une_frequence_nulle_n_a_pas_de_longueur_d_onde() -> None:
    with pytest.raises(ValueError, match="frequence nulle"):
        onde.longueur_d_onde(3e8, 0.0)


# --- EM-PEL : effet pelliculaire -------------------------------------------


def test_l_epaisseur_de_peau_du_cuivre_a_cinquante_hertz() -> None:
    """Environ 9 mm : du meme ordre que le rayon des conducteurs industriels.

    C'est le point de la note du cours -- l'effet n'est pas une curiosite
    de haute frequence, il dimensionne les jeux de barres.
    """
    assert epaisseur_de_peau(50.0) == pytest.approx(9.2e-3, abs=0.2e-3)


def test_l_epaisseur_decroit_comme_la_racine_de_la_frequence() -> None:
    cent_fois = epaisseur_de_peau(50.0) / epaisseur_de_peau(5000.0)
    assert cent_fois == pytest.approx(10.0)
    assert rapport_des_epaisseurs(50.0, 5000.0) == pytest.approx(10.0)


def test_l_effet_est_negligeable_sur_un_fil_fin_a_basse_frequence() -> None:
    assert est_negligeable(50.0, 1e-3)


def test_l_effet_n_est_pas_negligeable_sur_une_barre_a_cinquante_hertz() -> None:
    """Un rayon de 2 cm : l'epaisseur de peau ne couvre pas la section."""
    assert not est_negligeable(50.0, 0.02)


def test_une_frequence_nulle_n_a_pas_d_epaisseur_de_peau() -> None:
    """La pulsation serait nulle, et l'epaisseur infinie."""
    with pytest.raises(ValueError, match="frequence nulle"):
        epaisseur_de_peau(0.0)


def test_un_rapport_de_frequences_nulles_est_refuse() -> None:
    with pytest.raises(ValueError, match="frequences nulles"):
        rapport_des_epaisseurs(0.0, 50.0)


def test_un_rayon_nul_ne_permet_pas_de_juger_l_effet() -> None:
    with pytest.raises(ValueError, match="rayon nul"):
        est_negligeable(50.0, 0.0)


def test_une_conductivite_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="conductivite nulle"):
        epaisseur_de_peau(50.0, conductivite=0.0)
