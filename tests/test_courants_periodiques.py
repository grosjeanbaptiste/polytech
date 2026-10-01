"""Le chapitre 17 du syllabus d'electricite generale : courants periodiques.

Il n'etait pas porte : la bibliotheque traitait le sinusoidal, et le
periodique **non** sinusoidal est autre chose -- c'est la que la
puissance deformante apparait, et elle n'existe pas en sinusoidal pur.

Le syllabus est explicite sur les trois puissances et leur lien, et il
avertit lui-meme que la reactive « fait l'objet dans la litterature de
definitions differentes ». Il en donne deux, et c'est la seconde qui
introduit `D` : la puissance apparente se decompose alors en active,
reactive du fondamental, et deformante.

La verification la plus forte est l'identite qui relie les trois :
`S^2 = P^2 + Q_1^2 + D^2`. Elle doit tenir exactement, et elle enferme
les trois definitions d'un coup -- une faute sur l'une se verrait.
"""

from __future__ import annotations

import math

import pytest

from polytech.electromagnetisme.domain.circuits.periodiques import (
    courant_actif,
    courant_reactif,
    facteur_de_puissance,
    puissance_apparente,
    puissance_deformante,
    puissance_reactive_globale,
    taux_de_distorsion,
    valeur_efficace_des_harmoniques,
)


def test_la_valeur_efficace_compose_la_continue_et_les_harmoniques() -> None:
    """`I^2 = I_0^2 + somme I_n^2` : les contributions s'ajoutent en carres.

    C'est l'orthogonalite des harmoniques qui l'autorise, et c'est le
    piege du chapitre : les valeurs efficaces ne s'additionnent pas. Un
    courant de 3 A continu portant un fondamental de 4 A efficaces vaut
    5 A efficaces, et non 7.
    """
    assert valeur_efficace_des_harmoniques(3.0, (4.0,)) == pytest.approx(5.0)


def test_une_grandeur_alternative_pure_n_a_pas_de_composante_continue() -> None:
    """Sa valeur efficace est alors celle de ses seuls harmoniques."""
    efficace = valeur_efficace_des_harmoniques(0.0, (10.0, 3.0, 1.0))
    assert efficace == pytest.approx(math.sqrt(100 + 9 + 1))


def test_la_puissance_apparente_est_le_produit_des_valeurs_efficaces() -> None:
    """`S = U I`, « reste definie par le produit des valeurs efficaces ».

    Citation du syllabus : la definition ne change pas en periodique, et
    c'est justement ce qui fait apparaitre un ecart avec la puissance
    active que le sinusoidal n'avait pas.
    """
    assert puissance_apparente(230.0, 10.0) == pytest.approx(2300.0)


def test_le_facteur_de_puissance_ne_depasse_jamais_l_unite() -> None:
    """Le syllabus le demontre : « on a toujours -lambda <= 1 ».

    La garde porte donc un resultat du cours, et non une precaution
    d'implementation : une puissance active superieure a l'apparente
    serait une faute de mesure ou de calcul, jamais un circuit.
    """
    assert facteur_de_puissance(1800.0, 2300.0) == pytest.approx(0.7826, abs=1e-4)
    with pytest.raises(ValueError, match="facteur de puissance au-dela de l'unite"):
        facteur_de_puissance(2500.0, 2300.0)


def test_la_reactive_globale_complete_l_active_dans_l_apparente() -> None:
    """`Q^2 = S^2 - P^2` : la premiere des deux definitions du syllabus."""
    assert puissance_reactive_globale(2300.0, 1800.0) == pytest.approx(
        math.sqrt(2300.0**2 - 1800.0**2),
    )


def test_la_deformante_est_ce_que_le_fondamental_n_explique_pas() -> None:
    """`Q_d^2 = Q^2 - Q_1^2`, ou l'indice 1 designe le fondamental.

    C'est la definition du syllabus, et son sens : la reactive globale
    comprend ce que les harmoniques apportent ; en retirer la part du
    fondamental laisse exactement la deformante.
    """
    globale, fondamentale = 1000.0, 600.0
    assert puissance_deformante(globale, fondamentale) == pytest.approx(800.0)


def test_une_charge_sinusoidale_pure_n_a_aucune_puissance_deformante() -> None:
    """Et c'est la propriete qui definit le chapitre par difference.

    En sinusoidal pur, la reactive globale **est** celle du fondamental :
    il n'y a pas d'autre harmonique. La deformante est donc nulle, et
    toute valeur non nulle signale un courant non sinusoidal.
    """
    assert puissance_deformante(850.0, 850.0) == pytest.approx(0.0)


def test_l_identite_des_trois_puissances_tient_exactement() -> None:
    """`S^2 = P^2 + Q_1^2 + D^2` : les trois definitions ensemble.

    C'est la verification la plus forte du module : elle enferme la
    decomposition entiere, et une faute sur l'une des trois puissances
    s'y verrait. On part de valeurs efficaces et d'un fondamental, on
    calcule les trois, et la somme des carres doit redonner l'apparente.
    """
    tension, courant = 230.0, 10.0
    apparente = puissance_apparente(tension, courant)
    active = 1700.0
    globale = puissance_reactive_globale(apparente, active)
    fondamentale = 1200.0
    deformante = puissance_deformante(globale, fondamentale)
    somme = active**2 + fondamentale**2 + deformante**2
    assert somme == pytest.approx(apparente**2)


def test_les_deux_composantes_du_courant_se_deduisent_des_puissances() -> None:
    """`P = U I_p` et `Q = U I_q` : la decomposition en phase et en quadrature.

    Le syllabus la pose explicitement -- `i_p` est « l'homologue du
    courant actif (ou watte) » et `i_q` celui du « courant reactif (ou
    dewatte) » des circuits sinusoidaux. Les deux courants se composent
    en carres pour redonner le courant total, l'un etant orthogonal a
    l'autre.
    """
    tension, courant = 230.0, 10.0
    active = 1700.0
    globale = puissance_reactive_globale(puissance_apparente(tension, courant), active)
    actif = courant_actif(active, tension)
    reactif = courant_reactif(globale, tension)
    assert math.hypot(actif, reactif) == pytest.approx(courant)


def test_le_taux_de_distorsion_rapporte_les_harmoniques_au_fondamental() -> None:
    """Nul sur un fondamental seul, et croissant avec les harmoniques.

    C'est la grandeur que les normes de qualite de reseau emploient, et
    elle se lit directement : 30 % de distorsion sur un fondamental de
    10 A veut dire 3 A repartis dans les harmoniques.
    """
    assert taux_de_distorsion(10.0, ()) == pytest.approx(0.0)
    assert taux_de_distorsion(10.0, (3.0,)) == pytest.approx(0.30)
    assert taux_de_distorsion(10.0, (3.0, 4.0)) == pytest.approx(0.5)


def test_une_apparente_nulle_refuse_le_facteur_de_puissance() -> None:
    """Sans puissance apparente, le facteur n'est pas defini."""
    with pytest.raises(ValueError, match="puissance apparente nulle ou negative"):
        facteur_de_puissance(0.0, 0.0)


def test_une_fondamentale_superieure_a_la_globale_est_refusee() -> None:
    """La deformante serait imaginaire : la part ne depasse pas le tout."""
    with pytest.raises(ValueError, match="reactive du fondamental superieure"):
        puissance_deformante(600.0, 1000.0)


def test_une_active_superieure_a_l_apparente_est_refusee() -> None:
    """Meme raison, un cran plus haut : la reactive serait imaginaire."""
    with pytest.raises(ValueError, match="active superieure a l'apparente"):
        puissance_reactive_globale(1000.0, 1800.0)


def test_un_fondamental_nul_refuse_le_taux_de_distorsion() -> None:
    """Le taux y divise par zero, et un signal sans fondamental n'en a pas."""
    with pytest.raises(ValueError, match="fondamental nul ou negatif"):
        taux_de_distorsion(0.0, (1.0,))
