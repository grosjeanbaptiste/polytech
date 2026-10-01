"""Le dernier lot : ce qui manquait aux dix-sept derniers fascicules.

Chaque bloc de ce fichier reprend un titre precis que les planches
enoncent et que rien ne portait. Les ancrages sont de trois natures --
une valeur exacte quand le cours en donne une, une propriete d'ordre
quand il n'en donne pas, et un recoupement entre deux routes quand les
deux existent.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    potentiel_scalaire_magnetique,
)
from polytech.electromagnetisme.domain.magnetostatique.lois import (
    potentiel_vecteur_du_fil_infini,
)
from polytech.electronique.domain.boucle_fermee import (
    gain_vu_de_la_charge,
    gain_vu_de_la_source,
    impedance_vue_en_boucle_ouverte,
)
from polytech.electronique.domain.bruit import (
    FAMILLES_DE_BRUIT,
    bruit_de_grenaille_quantique,
    est_blanc,
)
from polytech.electronique.domain.des_champs_aux_circuits import (
    tension_de_defaut_terre_neutre,
)
from polytech.electronique.domain.facteur_de_bruit import (
    facteur_de_bruit_ponctuel,
    tension_de_bruit_depuis_une_fiche_technique,
)
from polytech.electronique.domain.pole_zero import (
    carte_des_poles_et_zeros,
    harmonique_filtree,
)
from polytech.signaux.domain.harmonique import parties_reelle_et_imaginaire
from polytech.signaux.domain.reponse_indicielle import minimum_de_resonance
from polytech.signaux.domain.stabilite import critere_simplifie_de_nyquist
from polytech.signaux.domain.systemes import comparaison_des_modeles
from polytech.signaux.domain.temporel import est_du_troisieme_ordre
from polytech.signaux.domain.transformees import (
    TRANSFORMEES_USUELLES,
    transformee_inverse_en_pratique,
)
from polytech.signaux.domain.transmittance import Transmittance
from polytech.thermique.domain.convection.forcee import (
    coefficient_depuis_le_nusselt,
    effet_de_la_turbulence,
)
from polytech.thermique.domain.convection.naturelle import (
    nusselt_d_un_espace_ferme,
    nusselt_du_cylindre_horizontal,
)
from polytech.thermique.domain.rayonnement.corps_noir import proprietes_d_un_solide_reel
from polytech.thermique.domain.rayonnement.facteurs_de_forme import (
    aire_d_echange_direct,
    facteur_entre_deux_plaques_paralleles,
)

P = sp.Symbol("p")
OMEGA = sp.Symbol("omega", positive=True)


# --- Bruit : les familles que le fascicule enumere -------------------------


def test_le_fascicule_enumere_cinq_familles_de_bruit() -> None:
    """Thermique, grenaille, scintillement, popcorn, quantique."""
    attendu = 5
    assert len(FAMILLES_DE_BRUIT) == attendu
    assert "popcorn" in FAMILLES_DE_BRUIT


def test_seuls_le_thermique_et_la_grenaille_sont_blancs() -> None:
    """Le scintillement est rose, le popcorn impulsionnel : ni l'un ni l'autre.

    La distinction decide de l'integration : un bruit blanc s'integre comme
    la largeur de bande, un bruit rose comme son logarithme.
    """
    assert est_blanc("thermique")
    assert est_blanc("grenaille")
    assert not est_blanc("scintillement")
    assert not est_blanc("popcorn")


def test_une_famille_inconnue_est_refusee() -> None:
    with pytest.raises(ValueError, match="famille de bruit inconnue"):
        est_blanc("rose bonbon")


def test_le_bruit_quantique_est_proportionnel_a_la_frequence() -> None:
    """`2 h f` : negligeable en electronique, dominant en optique.

    C'est pourquoi le fascicule le mentionne sans l'employer -- a un
    gigahertz il est quinze ordres de grandeur sous le bruit thermique.
    """
    basse = bruit_de_grenaille_quantique(1e9)
    haute = bruit_de_grenaille_quantique(2e9)
    assert haute == pytest.approx(2 * basse)
    assert basse < 1e-23


def test_le_facteur_de_bruit_ponctuel_se_mesure_a_une_frequence() -> None:
    """« Also called spot noise factor », dit la planche.

    Il depend de la frequence, et c'est pourquoi une fiche technique en
    donne une courbe et non un nombre.
    """
    a_mille = facteur_de_bruit_ponctuel(densite_totale=4e-18, densite_de_source=2e-18)
    assert a_mille == pytest.approx(2.0)


def test_une_densite_de_source_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="densite de la source"):
        facteur_de_bruit_ponctuel(densite_totale=4e-18, densite_de_source=0.0)


def test_une_fiche_technique_donne_une_densite_que_l_on_integre() -> None:
    """« Reminder: how to read a datasheet (OPA140) » : la planche le fait.

    Une fiche donne `nV/racine(Hz)` ; la tension efficace s'en tire par la
    racine de la bande. C'est le geste que tout dimensionnement commence.
    """
    efficace = tension_de_bruit_depuis_une_fiche_technique(5.0, bande=10_000.0)
    assert efficace == pytest.approx(5e-9 * math.sqrt(10_000.0))


def test_une_bande_nulle_ne_donne_aucune_tension_de_bruit() -> None:
    with pytest.raises(ValueError, match="bande nulle"):
        tension_de_bruit_depuis_une_fiche_technique(5.0, bande=0.0)


# --- Convection ------------------------------------------------------------


def test_le_coefficient_de_convection_se_tire_du_nusselt() -> None:
    """`K = Nu lambda / L` : le retour du nombre sans dimension au physique."""
    assert coefficient_depuis_le_nusselt(100.0, 0.026, 0.5) == pytest.approx(5.2)


def test_la_turbulence_augmente_l_echange() -> None:
    """« Effet de la turbulence » : elle melange, donc elle transporte mieux.

    Le rapport des exposants de Reynolds le dit : `0,8` en turbulent contre
    `0,5` en laminaire, donc l'echange croit plus vite avec la vitesse.
    """
    laminaire, turbulent = effet_de_la_turbulence()
    assert turbulent > laminaire
    assert laminaire == pytest.approx(0.5)
    assert turbulent == pytest.approx(0.8)


def test_le_cylindre_horizontal_a_sa_propre_correlation() -> None:
    """`Nu = 0,53 Ra^(1/4)` en laminaire : la geometrie compte."""
    assert nusselt_du_cylindre_horizontal(1e6) == pytest.approx(0.53 * 1e6**0.25)


def test_un_rayleigh_nul_est_refuse_pour_le_cylindre() -> None:
    with pytest.raises(ValueError, match="Rayleigh nul"):
        nusselt_du_cylindre_horizontal(0.0)


def test_un_espace_ferme_conduit_avant_de_convecter() -> None:
    """Recoupement avec la lame de fluide : les deux doivent concorder.

    Sous le seuil de mise en mouvement, `Nu = 1` -- le fluide ne bouge pas
    et ne fait que conduire.
    """
    assert nusselt_d_un_espace_ferme(1000.0) == pytest.approx(1.0)
    assert nusselt_d_un_espace_ferme(1e5) > 1.0


# --- Rayonnement -----------------------------------------------------------


def test_un_solide_reel_a_ses_trois_proprietes_qui_somment_a_un() -> None:
    """`alpha + rho + tau = 1` : c'est un bilan, donc une egalite exacte.

    Le fascicule consacre une planche aux « proprietes des solides
    reels », et cette somme est ce qui les contraint.
    """
    absorbe, reflechit, transmet = proprietes_d_un_solide_reel(0.7, 0.2)
    assert absorbe + reflechit + transmet == pytest.approx(1.0)
    assert transmet == pytest.approx(0.1)


def test_un_solide_opaque_ne_transmet_rien() -> None:
    _, _, transmet = proprietes_d_un_solide_reel(0.7, 0.3)
    assert transmet == pytest.approx(0.0)


def test_une_somme_superieure_a_un_est_refusee() -> None:
    """Elle violerait la conservation de l'energie incidente."""
    with pytest.raises(ValueError, match="somme des proprietes"):
        proprietes_d_un_solide_reel(0.7, 0.5)


def test_le_facteur_entre_deux_plaques_paralleles_tend_vers_un() -> None:
    """Deux plaques infiniment proches ne voient qu'elles-memes.

    C'est la limite de l'expression analytique, et le cas tabule « deux
    plans gris infinis paralleles » en decoule. La convergence est **lente**
    -- a un rapport de cent le facteur vaut encore 0,98, et il faut un
    rapport de dix mille pour atteindre 0,9998 -- ce qui dit combien
    l'hypothese « plans infinis » est exigeante en pratique.

    Ce test est ne d'une erreur : j'avais ecrit le radical `racine(1+X^2/2)`
    au lieu de `racine(1+X^2)`, et la limite plafonnait a 0,86 au lieu de
    tendre vers un. La limite exacte est ce qui a revele la faute.
    """
    proche = facteur_entre_deux_plaques_paralleles(cote=1.0, distance=1e-4)
    assert proche == pytest.approx(1.0, abs=3e-4)
    lointain = facteur_entre_deux_plaques_paralleles(cote=1.0, distance=100.0)
    assert lointain < 0.01


def test_une_distance_nulle_entre_plaques_est_refusee() -> None:
    with pytest.raises(ValueError, match="distance nulle"):
        facteur_entre_deux_plaques_paralleles(cote=1.0, distance=0.0)


def test_l_aire_d_echange_direct_est_symetrique() -> None:
    """`S_i F_ij = S_j F_ji` : la reciprocite, ecrite en aires.

    C'est la forme que le fascicule emploie pour les echanges totaux,
    parce qu'elle rend la symetrie visible.
    """
    une = aire_d_echange_direct(aire=2.0, facteur=0.5)
    autre = aire_d_echange_direct(aire=1.0, facteur=1.0)
    assert une == pytest.approx(autre)


# --- Boucle fermee et performances -----------------------------------------


def test_l_impedance_vue_en_boucle_ouverte_est_la_somme_des_series() -> None:
    """« Open-loop quantities » : avant retroaction, tout s'additionne."""
    assert impedance_vue_en_boucle_ouverte(1e3, 1e6) == pytest.approx(1e6 + 1e3)


def test_une_resistance_de_source_reduit_le_gain_vu() -> None:
    """« Impact of source resistance » : le diviseur d'entree mange du gain."""
    sans = gain_vu_de_la_source(100.0, source=0.0, entree=1e6)
    avec = gain_vu_de_la_source(100.0, source=1e6, entree=1e6)
    assert avec == pytest.approx(sans / 2)


def test_une_resistance_de_charge_reduit_le_gain_vu() -> None:
    """« Impact of load resistance » : le diviseur de sortie aussi."""
    sans = gain_vu_de_la_charge(100.0, sortie=0.0, charge=1e3)
    avec = gain_vu_de_la_charge(100.0, sortie=1e3, charge=1e3)
    assert avec == pytest.approx(sans / 2)


def test_une_impedance_d_entree_nulle_annule_le_gain_vu() -> None:
    with pytest.raises(ValueError, match="impedance d'entree nulle"):
        gain_vu_de_la_source(100.0, source=1e3, entree=0.0)


def test_le_minimum_de_resonance_existe_pour_un_systeme_a_deux_pics() -> None:
    """« En cas de presence d'un |M|min » : la planche le prevoit.

    Il n'existe pas toujours -- un second ordre n'en a pas --, et le dire
    vaut mieux que de rendre une valeur qui n'en est pas une.
    """
    assert minimum_de_resonance((1.2, 0.8, 1.1)) == pytest.approx(0.8)
    with pytest.raises(ValueError, match="aucun minimum"):
        minimum_de_resonance((1.2, 1.1, 1.0))


# --- Stabilite et modeles --------------------------------------------------


def test_le_critere_simplifie_regarde_si_la_courbe_entoure_moins_un() -> None:
    """La forme que le fascicule emploie en pratique.

    « Critere simplifie d'application : stable en BF car n'entoure pas »
    -- et elle ne vaut que pour un systeme stable en boucle ouverte, ce
    que la garde exige.
    """
    assert critere_simplifie_de_nyquist(abscisse_du_croisement=-0.5)
    assert not critere_simplifie_de_nyquist(abscisse_du_croisement=-6.67)


def test_un_croisement_positif_est_refuse_par_le_critere_simplifie() -> None:
    """La courbe de Nyquist croise l'axe reel du cote negatif, par construction."""
    with pytest.raises(ValueError, match="croisement positif"):
        critere_simplifie_de_nyquist(abscisse_du_croisement=0.5)


def test_les_parties_reelle_et_imaginaire_se_separent() -> None:
    """« Calcul de Re et Im » : l'etape que le trace de Nyquist demande."""
    reelle, imaginaire = parties_reelle_et_imaginaire(1 / (1 + sp.I * OMEGA))
    assert sp.simplify(reelle - 1 / (1 + OMEGA**2)) == 0
    assert sp.simplify(imaginaire + OMEGA / (1 + OMEGA**2)) == 0


def test_la_comparaison_des_modeles_nomme_leurs_deux_portees() -> None:
    """« Comparaison des modeles » : le fascicule met les deux en regard."""
    global_, local = comparaison_des_modeles()
    assert "partout" in global_
    assert "voisinage" in local


def test_un_systeme_du_troisieme_ordre_est_reconnu() -> None:
    """Le fascicule lui consacre une partie : ses reponses ont trois modes."""
    assert est_du_troisieme_ordre(Transmittance(expression=1 / (1 + P) ** 3))
    assert not est_du_troisieme_ordre(Transmittance(expression=1 / (1 + P)))


# --- Transformees, cartes, et le reste -------------------------------------


def test_les_transformees_usuelles_du_fascicule_sont_tabulees() -> None:
    """« Transformees de signaux usuels » : une table, donc une donnee."""
    minimum = 4
    assert len(TRANSFORMEES_USUELLES) >= minimum
    assert "echelon" in TRANSFORMEES_USUELLES


def test_la_transformee_inverse_en_pratique_passe_par_les_fractions_simples() -> None:
    """« En pratique : transformee inverse » : le fascicule donne la recette."""
    etapes = transformee_inverse_en_pratique()
    assert "fractions simples" in etapes[0]
    assert len(etapes) >= 2


def test_la_carte_des_poles_et_zeros_les_separe() -> None:
    """« Pole-zero map analysis » : le fascicule y consacre cinq planches."""
    poles, zeros = carte_des_poles_et_zeros(P / ((P + 1) * (P + 2)), P)
    assert zeros == (sp.Integer(0),)
    assert poles == (sp.Integer(-2), sp.Integer(-1))


def test_une_harmonique_est_attenuee_selon_le_module_du_filtre() -> None:
    """« Filtering digital signals — harmonics » : chaque harmonique son sort."""
    attenuee = harmonique_filtree(amplitude=1.0, module_du_filtre=0.1)
    assert attenuee == pytest.approx(0.1)


def test_un_module_de_filtre_negatif_est_refuse() -> None:
    with pytest.raises(ValueError, match="module du filtre"):
        harmonique_filtree(amplitude=1.0, module_du_filtre=-0.1)


def test_le_potentiel_vecteur_du_fil_decroit_logarithmiquement() -> None:
    """`-(mu_0 I / 2 pi) ln r` : l'expression macroscopique du fascicule."""
    proche = potentiel_vecteur_du_fil_infini(10.0, 0.01)
    loin = potentiel_vecteur_du_fil_infini(10.0, 0.1)
    assert proche > loin


def test_une_distance_nulle_n_a_pas_de_potentiel_vecteur() -> None:
    with pytest.raises(ValueError, match="distance nulle"):
        potentiel_vecteur_du_fil_infini(10.0, 0.0)


def test_le_potentiel_scalaire_magnetique_est_la_fmm_partielle() -> None:
    """`H l` entre deux points : l'analogue du potentiel electrique.

    Il n'existe que dans les regions sans courant -- sinon `H` n'est pas a
    circulation nulle et aucun potentiel ne le decrit.
    """
    assert potentiel_scalaire_magnetique(champ=70.0, longueur=0.5) == pytest.approx(
        35.0
    )


def test_une_tension_de_defaut_apparait_entre_terre_et_neutre() -> None:
    """« Earth and neutral wire » : ils ne sont pas au meme potentiel.

    Le courant de retour dans le neutre y cree une chute, et c'est la
    raison pour laquelle une masse reliee au neutre n'est pas une masse.
    """
    tension = tension_de_defaut_terre_neutre(courant=16.0, resistance=0.5)
    assert tension == pytest.approx(8.0)


def test_un_courant_negatif_est_refuse_dans_le_defaut() -> None:
    with pytest.raises(ValueError, match="courant negatif"):
        tension_de_defaut_terre_neutre(courant=-16.0, resistance=0.5)
