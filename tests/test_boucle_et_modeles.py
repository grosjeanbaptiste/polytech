"""Le reste de la stabilite, des performances et des modeles de systemes.

**Stabilite** : les deux grandes questions que le fascicule pose -- la
boucle est-elle stable, et quelles performances a-t-elle --, la
distinction asservissement / regulation, la chaine directe et la chaine de
reaction, et le critere de Nyquist par **encerclements**.

**Performances** : le pic de resonance et son minimum, la bande passante
en boucle fermee, et l'estimation de la reponse indicielle a partir des
caracteristiques frequentielles -- « et si on pouvait estimer la reponse
indicielle BF ?! », demande la planche.

**Modeles** : le modele global contre le modele local, le schema-bloc, et
la decomposition en sous-systemes.

L'ancrage le plus solide est le critere de Nyquist : pour un systeme
stable en boucle ouverte, la boucle fermee est stable si et seulement si
la courbe n'encercle pas le point `-1`. Zero encerclement, stable ; un
encerclement, un pole instable. C'est un compte entier, donc exact.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.signaux.domain.architecture import (
    NatureDeLaBoucle,
    chaine_de_reaction,
    chaine_directe,
    nature_de_la_boucle,
    questions_de_la_boucle,
)
from polytech.signaux.domain.reponse_indicielle import (
    estimation_du_depassement_depuis_le_pic,
    pic_de_resonance_du_second_ordre,
    temps_de_montee_depuis_la_bande,
)
from polytech.signaux.domain.stabilite import (
    poles_instables_par_nyquist,
)
from polytech.signaux.domain.systemes import (
    Portee,
    composer_en_cascade,
    portee_du_modele,
)
from polytech.signaux.domain.transmittance import Transmittance

P = sp.Symbol("p")


def test_le_fascicule_pose_deux_grandes_questions() -> None:
    """La stabilite d'abord, les performances ensuite -- et dans cet ordre.

    L'ordre n'est pas indifferent : parler des performances d'une boucle
    instable n'a aucun sens, et c'est pourquoi le fascicule traite la
    stabilite au chapitre 7 et les performances au 8.
    """
    attendu = 2
    questions = questions_de_la_boucle()
    assert len(questions) == attendu
    assert "stable" in questions[0]
    assert "performances" in questions[1]


def test_une_consigne_variable_donne_un_asservissement() -> None:
    """Un asservissement doit **suivre** : ses specifications portent sur l'ecart."""
    attendu = NatureDeLaBoucle.ASSERVISSEMENT
    assert nature_de_la_boucle(consigne_variable=True) == attendu


def test_une_consigne_fixe_donne_une_regulation() -> None:
    """Une regulation doit **rejeter** : ses specifications portent sur la perturbation.

    Le fascicule les distingue parce que les deux ne demandent pas le meme
    correcteur, et c'est pourquoi ce sont deux tests et non un parametre.
    """
    attendu = NatureDeLaBoucle.REGULATION
    assert nature_de_la_boucle(consigne_variable=False) == attendu


def test_la_chaine_directe_et_la_chaine_de_reaction_se_composent() -> None:
    """`G` et `H` : leur produit est le gain de boucle, et il est sans dimension."""
    correcteur = Transmittance(expression=sp.Integer(10))
    procede = Transmittance(expression=1 / (1 + P))
    capteur = Transmittance(expression=sp.Rational(1, 2))
    directe = chaine_directe((correcteur, procede))
    reaction = chaine_de_reaction(capteur)
    boucle = directe.en_serie_avec(reaction)
    assert sp.simplify(boucle.expression - 5 / (1 + P)) == 0


def test_une_chaine_directe_vide_est_refusee() -> None:
    with pytest.raises(ValueError, match="chaine directe vide"):
        chaine_directe(())


def test_le_critere_de_nyquist_compte_les_encerclements_du_point_moins_un() -> None:
    """Zero encerclement : stable. Un encerclement : un pole instable.

    C'est un compte **entier**, donc exact -- et c'est ce qui fait la force
    du critere sur un systeme a retard, qu'aucun polynome ne representerait.
    """
    assert poles_instables_par_nyquist(encerclements=0, poles_ouverts=0) == 0
    assert poles_instables_par_nyquist(encerclements=1, poles_ouverts=0) == 1


def test_un_pole_instable_en_ouvert_se_compense_par_un_encerclement() -> None:
    """`Z = N + P` : c'est pourquoi une boucle peut stabiliser un instable.

    Un pole instable en ouvert exige exactement un encerclement **dans le
    bon sens** pour que la boucle fermee soit stable.
    """
    assert poles_instables_par_nyquist(encerclements=-1, poles_ouverts=1) == 0


def test_un_nombre_de_poles_ouverts_negatif_est_refuse() -> None:
    with pytest.raises(ValueError, match="nombre de poles"):
        poles_instables_par_nyquist(encerclements=0, poles_ouverts=-1)


def test_le_pic_de_resonance_diverge_quand_l_amortissement_s_annule() -> None:
    """`1 / (2 zeta racine(1 - zeta^2))` : il n'existe que sous `1/racine(2)`.

    Au-dela, la reponse frequentielle est monotone et il n'y a pas de pic
    -- le dire vaut mieux que de rendre un nombre qui ne correspond a rien.
    """
    faible = pic_de_resonance_du_second_ordre(0.1)
    fort = pic_de_resonance_du_second_ordre(0.5)
    assert faible > fort > 1.0
    with pytest.raises(ValueError, match="pas de pic"):
        pic_de_resonance_du_second_ordre(0.8)


def test_le_depassement_s_estime_depuis_le_pic_de_resonance() -> None:
    """« Et si on pouvait estimer la reponse indicielle BF ?! », demande la planche.

    L'estimation passe par l'amortissement : le pic donne `zeta`, et
    `zeta` donne le depassement. Les deux relations sont exactes pour un
    second ordre, et approchees au-dela -- ce qui est dit.
    """
    depassement = estimation_du_depassement_depuis_le_pic(2.0)
    assert 0.0 < depassement < 1.0
    plus_resonant = estimation_du_depassement_depuis_le_pic(5.0)
    assert plus_resonant > depassement


def test_un_pic_sous_l_unite_est_refuse() -> None:
    """Un pic de resonance ne descend pas sous un : ce serait une attenuation."""
    with pytest.raises(ValueError, match="pic sous l'unite"):
        estimation_du_depassement_depuis_le_pic(0.5)


def test_le_temps_de_montee_est_inversement_proportionnel_a_la_bande() -> None:
    """`t_m omega_BP ~ constante` : la relation que le fascicule emploie.

    Elle encadre, et ne donne pas une valeur unique -- d'ou un couple de
    bornes, et non un nombre.
    """
    basse, haute = temps_de_montee_depuis_la_bande(0.45)
    assert basse < haute
    deux_fois, _ = temps_de_montee_depuis_la_bande(0.90)
    assert deux_fois == pytest.approx(basse / 2)


def test_une_bande_passante_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="bande passante nulle"):
        temps_de_montee_depuis_la_bande(0.0)


def test_un_modele_lineaire_vaut_partout() -> None:
    """« Deux modeles pour un meme systeme », dit la planche."""
    assert portee_du_modele(lineaire=True) == Portee.GLOBALE


def test_un_modele_linearise_ne_vaut_qu_au_voisinage() -> None:
    """La distinction n'est pas academique : elle dit jusqu'ou aller.

    Un modele local employe loin de son point de fonctionnement donne un
    resultat faux sans le signaler.
    """
    assert portee_du_modele(lineaire=False) == Portee.LOCALE


def test_la_composition_en_cascade_multiplie_les_transmittances() -> None:
    """La decomposition en sous-systemes du fascicule, et son recollement."""
    sous_systemes = (
        Transmittance(expression=sp.Integer(2)),
        Transmittance(expression=1 / (1 + P)),
        Transmittance(expression=sp.Integer(3)),
    )
    total = composer_en_cascade(sous_systemes)
    assert sp.simplify(total.expression - 6 / (1 + P)) == 0


def test_une_cascade_vide_est_refusee() -> None:
    with pytest.raises(ValueError, match="cascade vide"):
        composer_en_cascade(())
