"""Deux silences que j'avais declares a tort, et qui se ferment.

L'inventaire tenait pour « sans calcul » le chapitre d'introduction de la
thermique et celui de l'electronique. Pour le second c'etait presque
juste -- il est administratif, hormis le plan du cours. Pour le premier
c'etait **faux** : il porte la loi de refroidissement de Newton, les trois
modes de convection, et la superposition des modes en « cas reel ». Rien
de cela n'est administratif.

L'ancrage le plus net est la superposition : trois modes agissant en
parallele sur une meme paroi additionnent leurs **coefficients**, pas
leurs resistances -- l'inverse exact de trois parois en serie. Confondre
les deux est l'erreur que le chapitre previent d'entree.
"""

from __future__ import annotations

import pytest

from polytech.electronique.domain.plan_du_cours import (
    CHAPITRES,
    chapitre_du_sujet,
    nombre_de_chapitres,
)
from polytech.thermique.domain.echelle_de_newton import (
    celsius_depuis_degres_newton,
    degres_newton_depuis_celsius,
)
from polytech.thermique.domain.modes import (
    MODES_DE_TRANSMISSION,
    MODES_FONDAMENTAUX,
    RAPPORT_DE_DOMINANCE,
    ModeDeConvection,
    coefficient_superpose,
    flux_de_newton,
    gain_de_la_convection,
    mode_de_convection,
    mode_dominant,
    puissance_metabolique_au_repos,
    resistance_depuis_l_echange,
)
from polytech.thermique.domain.modes_fondamentaux import ModeFondamental
from polytech.thermique.domain.validite_de_newton import (
    ECART_MAXIMAL_DE_NEWTON,
    echange_lineaire,
    loi_de_newton_valide,
)


def test_la_loi_de_newton_est_proportionnelle_a_l_ecart() -> None:
    """`phi = K S (t_p - t_f)` : doubler l'ecart double le flux.

    C'est la loi de 1701, et elle est **lineaire** -- ce qui la distingue
    du rayonnement, en puissance quatrieme. Le chapitre les met en regard
    pour cette raison.
    """
    simple = flux_de_newton(echange=10.0, aire=1.0, ecart=20.0)
    double = flux_de_newton(echange=10.0, aire=1.0, ecart=40.0)
    assert double == pytest.approx(2 * simple)
    assert simple == pytest.approx(200.0)


def test_un_ecart_nul_ne_transfere_rien() -> None:
    """L'equilibre thermique : pas de flux sans difference de temperature."""
    assert flux_de_newton(echange=10.0, aire=1.0, ecart=0.0) == pytest.approx(0.0)


def test_un_ecart_negatif_inverse_le_flux() -> None:
    """Le signe dit le sens, et le conserver evite de perdre une direction."""
    assert flux_de_newton(echange=10.0, aire=1.0, ecart=-20.0) < 0


def test_un_coefficient_d_echange_negatif_est_refuse() -> None:
    with pytest.raises(ValueError, match="coefficient d'echange"):
        flux_de_newton(echange=-10.0, aire=1.0, ecart=20.0)


def test_une_aire_nulle_ne_transfere_rien_et_est_refusee() -> None:
    with pytest.raises(ValueError, match="aire nulle"):
        flux_de_newton(echange=10.0, aire=0.0, ecart=20.0)


@pytest.mark.parametrize(
    ("force", "naturel", "attendu"),
    [
        (100.0, 1.0, ModeDeConvection.FORCEE),
        (1.0, 100.0, ModeDeConvection.NATURELLE),
        (10.0, 10.0, ModeDeConvection.MIXTE),
    ],
)
def test_les_trois_modes_de_convection_du_chapitre(
    force: float,
    naturel: float,
    attendu: ModeDeConvection,
) -> None:
    """Le chapitre les distingue par ce qui est negligeable devant quoi.

    Planche 22 : en convection mixte, les forces d'Archimede sont « non
    negligeables » a cote du mouvement impose. Le cours ne chiffre pas ce
    « negligeable » : le rapport de dominance se donne. Avec dix, la
    convention de la bibliotheque, un regime porte le nom du coefficient
    qui domine l'autre d'un facteur dix.
    """
    regime = mode_de_convection(
        force=force, naturel=naturel, dominance=RAPPORT_DE_DOMINANCE
    )
    assert regime == attendu


def test_le_rapport_de_dominance_deplace_la_frontiere() -> None:
    """Le meme couple est mixte ou force selon ce qu'on tient pour negligeable."""
    assert pytest.approx(10.0) == RAPPORT_DE_DOMINANCE
    assert (
        mode_de_convection(force=30.0, naturel=10.0, dominance=10.0)
        is ModeDeConvection.MIXTE
    )
    assert (
        mode_de_convection(force=30.0, naturel=10.0, dominance=3.0)
        is ModeDeConvection.FORCEE
    )
    assert (
        mode_de_convection(force=10.0, naturel=30.0, dominance=3.0)
        is ModeDeConvection.NATURELLE
    )


def test_un_coefficient_de_convection_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="coefficient de convection"):
        mode_de_convection(force=0.0, naturel=10.0, dominance=10.0)
    with pytest.raises(ValueError, match="rapport de dominance non superieur a un"):
        mode_de_convection(force=5.0, naturel=10.0, dominance=1.0)


def test_les_modes_en_parallele_additionnent_leurs_coefficients() -> None:
    """`K = K_cv + K_ray` : c'est l'inverse de trois parois en serie.

    Trois parois en serie additionnent leurs **resistances** ; trois modes
    sur une meme paroi additionnent leurs **coefficients**. Confondre les
    deux est l'erreur que le chapitre previent d'entree, et c'est pourquoi
    il parle de « superposition des modes ».
    """
    assert coefficient_superpose((8.0, 5.0, 2.0)) == pytest.approx(15.0)


def test_un_seul_mode_vaut_lui_meme() -> None:
    """Cas limite, et controle de coherence de la superposition."""
    assert coefficient_superpose((8.0,)) == pytest.approx(8.0)


def test_une_superposition_vide_est_refusee() -> None:
    """Aucun mode, aucun transfert : le dire plutot que rendre zero."""
    with pytest.raises(ValueError, match="aucun mode"):
        coefficient_superpose(())


def test_un_coefficient_negatif_dans_la_superposition_est_refuse() -> None:
    """Il ferait **diminuer** le transfert total : impossible."""
    with pytest.raises(ValueError, match="coefficient negatif"):
        coefficient_superpose((8.0, -5.0))


def test_le_mode_dominant_est_celui_du_plus_grand_coefficient() -> None:
    """La question que le chapitre pose sur chacune de ses applications."""
    assert mode_dominant({"convection": 8.0, "rayonnement": 2.0}) == "convection"


def test_deux_modes_a_egalite_sont_refuses() -> None:
    """Aucun ne domine, et designer l'un des deux serait arbitraire."""
    with pytest.raises(ValueError, match="aucun mode ne domine"):
        mode_dominant({"convection": 5.0, "rayonnement": 5.0})


def test_un_dictionnaire_vide_n_a_pas_de_mode_dominant() -> None:
    with pytest.raises(ValueError, match="aucun mode"):
        mode_dominant({})


def test_le_plan_du_cours_d_electronique_a_quatre_chapitres() -> None:
    """La planche 0 les annonce : circuits, AOP, filtrage, instrumentation.

    C'est le seul contenu non administratif du chapitre, et le porter
    comme donnee vaut mieux que de le declarer absent.
    """
    attendu = 4
    assert nombre_de_chapitres() == attendu
    assert CHAPITRES[0].startswith("circuits")


def test_chaque_sujet_du_cours_se_rattache_a_son_chapitre() -> None:
    """Ce qui rend le plan utile : il repond a « ou cela se trouve-t-il ? »."""
    assert chapitre_du_sujet("analyse nodale") == CHAPITRES[0]
    assert chapitre_du_sujet("Sallen-Key") == CHAPITRES[2]


def test_un_sujet_etranger_au_cours_est_nomme() -> None:
    """Le dire plutot que de le rattacher au hasard au premier chapitre."""
    with pytest.raises(ValueError, match="sujet hors du cours"):
        chapitre_du_sujet("thermodynamique des gaz reels")


def test_les_deux_modes_fondamentaux_sont_nommes() -> None:
    """Planche 19 de l'introduction : « 2 modes fondamentaux de transfert ».

    « 1. Conductibilite thermique, 2. Rayonnement. » La bibliotheque en
    portait trois, convection comprise ; le syllabus (tome 1, page 10) dit
    de celle-ci qu'elle « n'est qu'un cas particulier de la conductibilite ».
    """
    assert MODES_FONDAMENTAUX == ("conductibilite", "rayonnement")
    assert tuple(mode.name.lower() for mode in ModeFondamental) == MODES_FONDAMENTAUX


def test_les_trois_transmissions_restent_nommees() -> None:
    """Tome 1, page 10 : « conductibilite, convection et rayonnement »."""
    assert MODES_DE_TRANSMISSION == ("conductibilite", "convection", "rayonnement")
    assert set(MODES_FONDAMENTAUX) < set(MODES_DE_TRANSMISSION)


def test_la_resistance_se_deduit_du_coefficient_d_echange() -> None:
    """`R = 1 / (K S)` : c'est ce qui rend l'analogie electrique possible.

    Et c'est pourquoi la suite du cours parle en resistances et non en
    coefficients -- elles s'additionnent en serie, ce que les coefficients
    ne font pas.
    """
    assert resistance_depuis_l_echange(echange=10.0, aire=2.0) == pytest.approx(0.05)


def test_une_resistance_depuis_un_echange_nul_est_refusee() -> None:
    """Elle serait infinie : une paroi parfaitement isolante, pas un resultat."""
    with pytest.raises(ValueError, match="echange nul"):
        resistance_depuis_l_echange(echange=0.0, aire=2.0)


def test_la_loi_de_newton_ne_vaut_que_dans_un_domaine_restreint() -> None:
    """« Remarques - validite de la loi de Newton », planche d'introduction.

    « La loi n'est donc valide que dans un domaine restreint de
    temperatures. » Le cours ne chiffre pas ce domaine : il se donne, et
    la bibliotheque n'en invente plus un.
    """
    assert loi_de_newton_valide(ecart=20.0, domaine=50.0)
    assert loi_de_newton_valide(ecart=50.0, domaine=50.0)
    assert not loi_de_newton_valide(ecart=60.0, domaine=50.0)
    with pytest.raises(ValueError, match="domaine nul ou negatif"):
        loi_de_newton_valide(ecart=20.0, domaine=0.0)


def test_le_seuil_de_deux_cents_kelvins_n_est_qu_une_convention() -> None:
    """Le cours ne le donne pas : il reste nomme, et se passe explicitement."""
    assert loi_de_newton_valide(ecart=200.0, domaine=ECART_MAXIMAL_DE_NEWTON)
    assert not loi_de_newton_valide(ecart=200.5, domaine=ECART_MAXIMAL_DE_NEWTON)


def test_un_ecart_negatif_ne_change_pas_la_validite() -> None:
    """C'est son module qui compte : refroidir ou chauffer est symetrique."""
    assert loi_de_newton_valide(ecart=-20.0, domaine=50.0)
    assert not loi_de_newton_valide(ecart=-60.0, domaine=50.0)


def test_seule_la_convection_forcee_est_a_peu_pres_lineaire() -> None:
    """Meme planche : la convection forcee l'est, pas la naturelle.

    « C'est a peu pres le cas pour la convection forcee mais ce n'est pas
    le cas pour le rayonnement, ni pour la convection naturelle. »
    """
    assert echange_lineaire(ModeDeConvection.FORCEE)
    assert not echange_lineaire(ModeDeConvection.NATURELLE)
    assert not echange_lineaire(ModeFondamental.RAYONNEMENT)
    for muet in (ModeDeConvection.MIXTE, ModeFondamental.CONDUCTIBILITE):
        with pytest.raises(ValueError, match="le cours ne se prononce pas"):
            echange_lineaire(muet)


def test_la_convection_transporte_mieux_que_la_conduction_pure() -> None:
    """« Conductibilite pure / convection » : le chapitre les met en regard.

    A epaisseur egale, un fluide en mouvement transporte un ordre de
    grandeur de plus qu'un fluide au repos -- et c'est tout l'interet de
    le mettre en mouvement.
    """
    rapport = gain_de_la_convection(conduction=0.6, convection=600.0)
    assert rapport == pytest.approx(1000.0)


def test_un_gain_de_convection_inferieur_a_un_est_refuse() -> None:
    """La convection ne peut pas transporter moins que la conduction seule."""
    with pytest.raises(ValueError, match="gain de convection"):
        gain_de_la_convection(conduction=600.0, convection=0.6)


def test_l_echelle_de_newton_place_la_glace_a_zero_et_le_corps_a_douze() -> None:
    """« Scala graduum caloris » (1701) : Newton a defini son echelle ainsi.

    Deux points fixes -- la glace fondante et le corps humain -- et c'est
    le coup de genie que la planche souligne : il a ensuite extrapole
    au-dela du domaine de son thermometre, par une loi de refroidissement.
    """
    assert degres_newton_depuis_celsius(0.0) == pytest.approx(0.0)
    assert degres_newton_depuis_celsius(37.0) == pytest.approx(12.0, abs=0.1)


def test_l_echelle_de_newton_est_lineaire_et_inversible() -> None:
    """L'aller-retour doit etre exact : sinon la conversion perdrait l'echelle."""
    for celsius in (0.0, 37.0, 100.0, 250.0):
        newton = degres_newton_depuis_celsius(celsius)
        assert celsius_depuis_degres_newton(newton) == pytest.approx(celsius)


def test_une_temperature_sous_le_zero_de_l_echelle_est_refusee() -> None:
    """Newton n'a pas gradue sous la glace fondante : son echelle y est muette."""
    with pytest.raises(ValueError, match="sous le zero de l'echelle"):
        degres_newton_depuis_celsius(-10.0)


def test_le_corps_humain_evacue_sa_chaleur_par_plusieurs_modes() -> None:
    """« Physiologie thermique » : le chapitre s'en sert comme application.

    Un corps au repos dissipe environ cent watts, et la superposition des
    modes est ce qui le permet -- aucun seul n'y suffirait a temperature
    ambiante.
    """
    puissance = puissance_metabolique_au_repos()
    assert 80.0 <= puissance <= 120.0
    aire = 1.8
    echange = puissance / (aire * 12.0)
    assert coefficient_superpose((echange,)) == pytest.approx(echange)


def test_un_degre_newton_negatif_est_refuse() -> None:
    """La garde de retour, que le premier jet laissait sans test.

    L'audit des gardes l'a trouvee : `celsius_depuis_degres_newton`
    refusait un argument negatif sans qu'aucun test ne le declenche --
    donc sans preuve que le refus arrive, ni que son message nomme la
    cause. Une garde non eprouvee est un echec silencieux en attente.
    """
    with pytest.raises(ValueError, match="sous le zero de son echelle"):
        celsius_depuis_degres_newton(-1.0)
