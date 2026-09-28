"""Chaque garde du contexte electronique est reellement declenchee.

Un `raise` que nul test n'atteint est du code jamais execute : le jour ou
il se declenche, rien ne garantit qu'il fasse ce qu'on croit.
"""

import pytest
import sympy as sp

from polytech.electronique.domain.acquisition import (
    bits_necessaires,
    cadence_du_multiplexeur,
    densite_de_bruit_maximale,
    gain_de_chaine,
    nyquist,
    puissance_de_bruit_en_courant,
)
from polytech.electronique.domain.amplification import (
    diviseur_d_entree,
    gain_inverseur,
    gain_maximal_par_etage,
    gain_non_inverseur,
    nombre_d_etages,
)
from polytech.electronique.domain.biquad import TowThomas
from polytech.electronique.domain.chebyshev import elements_de_chebyshev
from polytech.electronique.domain.conversion import ConvertisseurAnalogiqueNumerique
from polytech.electronique.domain.filtre import PrototypeButterworth
from polytech.electronique.domain.friis import (
    facteur_d_une_cascade,
    facteur_en_cascade,
    facteur_maximal_du_second_etage,
)
from polytech.electronique.domain.induction import (
    BoucleDeMesure,
    BoucleRectangulaire,
)
from polytech.electronique.domain.laplace import est_stable
from polytech.electronique.domain.nodale import (
    systeme_avec_transconductance,
    systeme_en_echelle_lc,
    transfert_avec_transconductance,
    transfert_de_sortie,
)
from polytech.electronique.domain.pole_zero import pente_asymptotique
from polytech.electronique.domain.quadripole import (
    admittance_d_entree_a_vide,
    cellule_en_t,
    matrice_y_en_l,
    parametres_g_depuis_z,
    transfert_a_vide_depuis_y,
)
from polytech.electronique.domain.retroaction import (
    gain_en_boucle_fermee,
    parametres_h_amplificateur,
    parametres_h_reseau_resistif,
)
from polytech.electronique.domain.sallen_key import CelluleSallenKey
from polytech.electronique.domain.second_ordre import PasseBasCanonique


@pytest.mark.parametrize("bits", [0, -12])
def test_une_resolution_non_positive_est_refusee(bits: int) -> None:
    with pytest.raises(ValueError, match="resolution non strictement positive"):
        ConvertisseurAnalogiqueNumerique(bits=bits, pleine_echelle_volts=15.0)


def test_une_pleine_echelle_nulle_est_refusee() -> None:
    """Sans etendue, le pas de quantification n'existe pas."""
    with pytest.raises(ValueError, match="pleine echelle non positive"):
        ConvertisseurAnalogiqueNumerique(bits=12, pleine_echelle_volts=0.0)


@pytest.mark.parametrize(("gbw", "bande"), [(0.0, 100e3), (2.5e6, 0.0)])
def test_un_produit_gain_bande_ou_une_bande_nulle_est_refuse(
    gbw: float,
    bande: float,
) -> None:
    with pytest.raises(ValueError, match="doivent etre positifs"):
        gain_maximal_par_etage(gbw, bande)


def test_un_etage_qui_n_amplifie_pas_est_signale() -> None:
    """Gain unitaire par etage : aucun nombre d'etages n'atteindrait la cible.

    Le calcul logarithmique divergerait en silence ; mieux vaut le dire.
    """
    with pytest.raises(ValueError, match="n'amplifie pas a cette bande"):
        nombre_d_etages(500.0, 1e5, 1e5)


@pytest.mark.parametrize("montage", [gain_non_inverseur, gain_inverseur])
def test_une_resistance_nulle_est_refusee(montage: object) -> None:
    """Elle donnerait un gain infini ou une division par zero."""
    with pytest.raises(ValueError, match="resistances non positives"):
        montage(0.0, 20e3)  # type: ignore[operator]


def test_un_prototype_vide_est_refuse() -> None:
    with pytest.raises(ValueError, match="prototype vide"):
        PrototypeButterworth(elements=(), impedance=50.0)


def test_une_impedance_non_positive_est_refusee() -> None:
    with pytest.raises(ValueError, match="impedance non positive"):
        PrototypeButterworth(elements=(1.0,), impedance=0.0)


def test_un_element_non_positif_du_prototype_est_refuse() -> None:
    """Une inductance nulle n'est pas un filtre, c'est un fil."""
    with pytest.raises(ValueError, match="element non positif"):
        PrototypeButterworth(elements=(1.0, -2.0), impedance=50.0)


def test_une_pulsation_de_coupure_nulle_est_refusee() -> None:
    proto = PrototypeButterworth(elements=(1.0,), impedance=50.0)
    with pytest.raises(ValueError, match="pulsation de coupure non positive"):
        proto.en_passe_haut(0.0)


def test_une_cellule_sans_impedance_shunt_est_refusee() -> None:
    """Elle court-circuiterait sa sortie : la matrice n'aurait pas de sens."""
    with pytest.raises(ValueError, match="court-circuite"):
        cellule_en_t(sp.Symbol("R"), sp.Integer(0), sp.Integer(0))


@pytest.mark.parametrize(("visee", "capteur"), [(0.0, 1e-5), (5.0, 0.0)])
def test_un_gain_de_chaine_sur_des_tensions_nulles_est_refuse(
    visee: float,
    capteur: float,
) -> None:
    with pytest.raises(ValueError, match="tensions non positives"):
        gain_de_chaine(visee, capteur)


def test_une_frequence_maximale_nulle_n_a_pas_de_cadence_de_nyquist() -> None:
    with pytest.raises(ValueError, match="frequence maximale non positive"):
        nyquist(0.0)


def test_un_multiplexeur_sans_voie_est_refuse() -> None:
    with pytest.raises(ValueError, match="nombre de voies non positif"):
        cadence_du_multiplexeur(400.0, 0)


@pytest.mark.parametrize(("etendue", "pas"), [(0.0, 1e-7), (8e-5, 0.0)])
def test_une_etendue_ou_un_pas_nul_est_refuse(etendue: float, pas: float) -> None:
    with pytest.raises(ValueError, match="etendue ou pas non positif"):
        bits_necessaires(etendue, pas)


def test_une_bande_nulle_ne_porte_aucune_puissance_de_bruit() -> None:
    with pytest.raises(ValueError, match="bande non positive"):
        puissance_de_bruit_en_courant(20e-15, 1e3, 0.0)


@pytest.mark.parametrize(("budget", "bande"), [(0.0, 200.0), (7.5e-7, 0.0)])
def test_un_budget_de_bruit_nul_est_refuse(budget: float, bande: float) -> None:
    with pytest.raises(ValueError, match="budget ou bande non positif"):
        densite_de_bruit_maximale(budget, bande)


def test_une_pulsation_de_cellule_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="pulsation non positive"):
        TowThomas(pulsation=0.0, qualite=0.7, gain=2.0)


def test_un_facteur_de_qualite_nul_est_refuse() -> None:
    """Le motif est ecrit en toutes lettres : l'audit ne lit pas les variables.

    Une premiere ecriture passait le message par une parametrisation ; le
    test s'executait, mais l'audit transversal ne pouvait pas voir que la
    garde etait couverte, et la signalait a raison comme orpheline.
    """
    with pytest.raises(ValueError, match="facteur de qualite non positif"):
        TowThomas(pulsation=1e4, qualite=0.0, gain=2.0)


def test_un_gain_de_cellule_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="gain non positif"):
        TowThomas(pulsation=1e4, qualite=0.7, gain=0.0)


@pytest.mark.parametrize(("r_min", "c_max"), [(0.0, 1e-6), (100.0, 0.0)])
def test_des_contraintes_de_composants_nulles_sont_refusees(
    r_min: float,
    c_max: float,
) -> None:
    """Sans borne, le facteur d'echelle n'aurait rien a satisfaire."""
    cellule = TowThomas(pulsation=1e4, qualite=0.7, gain=2.0)
    with pytest.raises(ValueError, match="non positive"):
        cellule.facteur_d_echelle(r_min, c_max)


def test_une_boucle_collee_au_fil_est_refusee() -> None:
    """A distance nulle le champ diverge : l'integrale n'a pas de valeur."""
    with pytest.raises(ValueError, match="boucle collee au fil"):
        BoucleDeMesure(hauteur=1.0, distance=0, largeur=1.0)


@pytest.mark.parametrize(("largeur", "hauteur"), [(0, 1.0), (1.0, 0)])
def test_une_boucle_d_aire_nulle_est_refusee(
    largeur: float,
    hauteur: float,
) -> None:
    """Sans aire, aucun flux ne la traverse et la question perd son sens."""
    with pytest.raises(ValueError, match="boucle d'aire nulle"):
        BoucleDeMesure(hauteur=hauteur, distance=1.0, largeur=largeur)


def test_une_pulsation_de_coupure_nulle_est_refusee_en_passe_bas() -> None:
    """La mise a l'echelle divise par elle : zero ferait exploser le filtre."""
    prototype = PrototypeButterworth(elements=(1.414, 1.414), impedance=73.0)
    with pytest.raises(ValueError, match="pulsation de coupure non positive"):
        prototype.en_passe_bas(0.0)


def test_une_impedance_d_entree_nulle_interdit_la_matrice_g() -> None:
    """`g_11 = 1/Z_11` : la matrice hybride inverse n'existe alors pas."""
    with pytest.raises(ValueError, match="impedance d'entree nulle"):
        parametres_g_depuis_z(sp.Matrix([[0, 1], [1, 1]]))


def test_une_cellule_sallen_key_sans_second_ordre_est_refusee() -> None:
    """Sans `C_1`, le denominateur tombe au premier ordre : rien a identifier."""
    cellule = CelluleSallenKey(r1=1.0, r2=1.0, c1=0, c2=1.0)
    with pytest.raises(ValueError, match="pas du second ordre"):
        cellule.pulsation_propre()


def test_un_transfert_sans_pole_est_refuse() -> None:
    """Un denominateur constant ne decrit aucun systeme dynamique."""
    variable = sp.Symbol("s")
    with pytest.raises(ValueError, match="denominateur sans pole"):
        est_stable(sp.Integer(1) / 2, variable)


def test_des_poles_incalculables_interdisent_de_conclure() -> None:
    """Sur une quintique generale, sympy ne rend aucune racine : on ne ment pas."""
    variable = sp.Symbol("s")
    with pytest.raises(ValueError, match="poles non calculables"):
        est_stable(1 / (variable**5 - variable + 1), variable)


@pytest.mark.parametrize(
    ("resistance", "inductance", "capacite"),
    [(0, 1, 1), (1, 0, 1), (1, 1, 0)],
)
def test_un_composant_nul_dans_l_echelle_est_refuse(
    resistance: float,
    inductance: float,
    capacite: float,
) -> None:
    """Un element nul court-circuite ou ouvre l'echelle : plus de filtre."""
    with pytest.raises(ValueError, match="composant nul dans l'echelle"):
        systeme_en_echelle_lc(resistance, inductance, capacite, sp.Symbol("s"))


def test_un_systeme_nodal_singulier_est_refuse() -> None:
    """A `s = -1 + i` et composants unitaires, le determinant s'annule."""
    with pytest.raises(ValueError, match="systeme nodal singulier"):
        transfert_de_sortie(1, 1, 1, -1 + sp.I)


def test_une_resistance_de_sortie_nulle_interdit_la_matrice_h() -> None:
    """Sans `R_o`, la maille de sortie n'a plus de loi a deriver."""
    with pytest.raises(ValueError, match="resistance de sortie nulle"):
        parametres_h_amplificateur(1e6, 0, 1e5)


@pytest.mark.parametrize(("masse", "reaction"), [(0, 1e6), (1e6, 0)])
def test_une_resistance_nulle_dans_le_pont_est_refusee(
    masse: float,
    reaction: float,
) -> None:
    """Un pont a resistance nulle court-circuite la retroaction."""
    with pytest.raises(ValueError, match="resistance nulle dans le pont"):
        parametres_h_reseau_resistif(masse, reaction)


def test_une_matrice_h_degeneree_est_refusee() -> None:
    """`h_11 h_22` nul rend le rapport de boucle infini : rien a conclure."""
    with pytest.raises(ValueError, match="matrice H degeneree"):
        gain_en_boucle_fermee(sp.Matrix([[0, 1], [1, 1]]))


def test_un_gain_de_premier_etage_nul_est_refuse_par_friis() -> None:
    """Friis divise par le carre de ce gain : zero ferait exploser la formule."""
    with pytest.raises(ValueError, match="gain du premier etage non positif"):
        facteur_en_cascade(1.5, 10.0, 0.0)


def test_un_facteur_global_sous_un_est_refuse() -> None:
    """Un quadripole n'enleve pas de bruit : sous 1, la question n'a pas de sens."""
    with pytest.raises(ValueError, match="facteur global sous la borne physique"):
        facteur_maximal_du_second_etage(0.5, 4.0)


@pytest.mark.parametrize(("zeros", "poles"), [(-1, 2), (2, -1)])
def test_des_comptes_negatifs_de_poles_ou_zeros_sont_refuses(
    zeros: int,
    poles: int,
) -> None:
    """Une carte ne porte pas un nombre negatif de croix."""
    with pytest.raises(ValueError, match="comptes negatifs"):
        pente_asymptotique(zeros=zeros, poles=poles)


def test_une_pulsation_nulle_prive_le_filtre_de_reference() -> None:
    """Sans `omega_0`, la forme canonique n'a plus d'echelle."""
    with pytest.raises(ValueError, match="pulsation nulle"):
        PasseBasCanonique(1.0, 0, 0.7)


def test_un_facteur_de_qualite_nul_est_refuse_par_la_forme_canonique() -> None:
    """`omega_0 / Q` deviendrait infini : le terme en s n'existe plus."""
    with pytest.raises(ValueError, match="facteur de qualite nul"):
        PasseBasCanonique(1.0, 1000.0, 0)


@pytest.mark.parametrize(
    ("source", "entree"),
    [(-1.0, 1000.0), (1000.0, 0.0)],
)
def test_un_diviseur_d_entree_mal_defini_est_refuse(
    source: float,
    entree: float,
) -> None:
    """Une source de resistance negative ou une entree nulle n'ont pas de sens."""
    with pytest.raises(ValueError, match="resistance de source negative"):
        diviseur_d_entree(source, entree)


def test_une_impedance_serie_nulle_confond_les_deux_acces() -> None:
    """Sans serie, la matrice admittance diverge."""
    with pytest.raises(ValueError, match="impedance serie nulle"):
        matrice_y_en_l(0, 1)


def test_une_admittance_de_sortie_nulle_interdit_le_transfert_a_vide() -> None:
    """`-y_21/y_22` n'existe pas si `y_22` est nul."""
    with pytest.raises(ValueError, match="admittance de sortie nulle"):
        transfert_a_vide_depuis_y(sp.Matrix([[1, 1], [1, 0]]))


def test_une_cascade_vide_n_a_pas_de_facteur_de_bruit() -> None:
    """Sans etage, la formule de Friis n'a rien a sommer."""
    with pytest.raises(ValueError, match="cascade vide"):
        facteur_d_une_cascade(())


@pytest.mark.parametrize(
    ("contre_reaction", "charge"),
    [(0, 1.0), (1.0, 0)],
)
def test_une_resistance_nulle_dans_l_etage_est_refusee(
    contre_reaction: float,
    charge: float,
) -> None:
    """Une conductance infinie court-circuiterait un noeud entier."""
    with pytest.raises(ValueError, match="resistance nulle dans l'etage"):
        systeme_avec_transconductance(1.0, 1.0, contre_reaction, charge, 1.0)


def test_une_transconductance_qui_annule_le_determinant_est_refusee() -> None:
    """A `g_m = -5` et resistances unitaires, le systeme devient singulier."""
    with pytest.raises(ValueError, match="systeme singulier"):
        transfert_avec_transconductance(1, 1, 1, 1, -5)


def test_une_admittance_de_sortie_nulle_interdit_l_admittance_d_entree() -> None:
    """A vide, `V_2` se deduit de `y_21/y_22` : nul, rien ne se deduit."""
    with pytest.raises(ValueError, match="admittance de sortie nulle"):
        admittance_d_entree_a_vide(sp.Matrix([[1, 1], [1, 0]]))


@pytest.mark.parametrize(("longueur", "largeur"), [(0, 1.0), (1.0, 0)])
def test_une_boucle_rectangulaire_d_aire_nulle_est_refusee(
    longueur: float,
    largeur: float,
) -> None:
    """Sans aire, le champ uniforme n'y depose aucun flux."""
    with pytest.raises(ValueError, match="boucle d'aire nulle"):
        BoucleRectangulaire(longueur=longueur, largeur=largeur)


@pytest.mark.parametrize("ondulation", [0.0, -0.1])
def test_une_ondulation_non_positive_est_refusee(ondulation: float) -> None:
    """Sans ondulation, le filtre n'est plus un Chebyshev."""
    with pytest.raises(ValueError, match="ondulation non positive"):
        elements_de_chebyshev(ondulation, 3)


def test_un_ordre_nul_est_refuse() -> None:
    """Un filtre d'ordre zero n'a aucun element a tabuler."""
    with pytest.raises(ValueError, match="ordre inferieur a un"):
        elements_de_chebyshev(0.1, 0)
