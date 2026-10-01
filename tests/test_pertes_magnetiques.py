"""La section 17.7 du syllabus : cycle dynamique et pertes magnetiques.

Le syllabus n'en donne pas de formule de Steinmetz -- aucun exposant,
aucune constante de materiau. Ce qu'il donne est plus fondamental, et se
teste : l'aire du cycle d'hysteresis **est** le travail volumique
dissipe, « comme dans le cas d'une energie dissipee pour vaincre des
frottements, cette energie se retrouve dans la matiere sous forme de
chaleur ».

De la suit tout le reste. La puissance dissipee est cette aire multipliee
par la frequence et le volume -- un cycle par periode. Et le cycle
**dynamique** est plus large que le statique, parce que « l'energie
dissipee augmente, par suite notamment de l'apparition de courants
induits au sein du materiau (courants de Foucault) ». L'inegalite est
donc un resultat du cours, et elle garde le calcul.
"""

from __future__ import annotations

import pytest

from polytech.electromagnetisme.domain.magnetostatique.pertes import (
    MECANISMES_DE_PERTE,
    energie_du_cycle,
    pertes_par_hysteresis,
    surcout_du_regime_dynamique,
)


def test_l_energie_dissipee_par_cycle_est_l_aire_fois_le_volume() -> None:
    """« Le travail volumique qu'il faut fournir », dit le syllabus.

    L'aire du cycle est en joules par metre cube ; multipliee par le
    volume du materiau, elle donne les joules dissipes a chaque cycle.
    """
    assert energie_du_cycle(aire=350.0, volume=0.002) == pytest.approx(0.7)


def test_les_pertes_croissent_lineairement_avec_la_frequence() -> None:
    """Un cycle par periode : doubler la frequence double les pertes.

    C'est vrai de l'hysteresis seule. Les courants de Foucault, eux, y
    ajoutent une part qui croit plus vite -- ce que le syllabus annonce
    sans la chiffrer, et que ce module ne chiffre donc pas non plus.
    """
    cinquante = pertes_par_hysteresis(aire=350.0, frequence=50.0, volume=0.002)
    cent = pertes_par_hysteresis(aire=350.0, frequence=100.0, volume=0.002)
    assert cinquante == pytest.approx(35.0)
    assert cent == pytest.approx(2 * cinquante)


def test_un_cycle_plus_large_dissipe_davantage() -> None:
    """L'aire est la grandeur qui compte, et elle se lit sur le graphe.

    C'est pourquoi un materiau « doux » -- a cycle etroit -- est celui
    des transformateurs, et un materiau « dur » -- a cycle large -- celui
    des aimants permanents. Le meme graphe sert aux deux usages et se lit
    en sens inverse.
    """
    etroit = pertes_par_hysteresis(aire=120.0, frequence=50.0, volume=0.001)
    large = pertes_par_hysteresis(aire=1200.0, frequence=50.0, volume=0.001)
    assert large == pytest.approx(10 * etroit)


def test_le_regime_dynamique_coute_plus_que_le_statique() -> None:
    """« L'energie dissipee augmente » : le surcout est positif, toujours.

    Le syllabus l'attribue aux courants de Foucault et au trainage. Le
    module ne les chiffre pas -- il n'en a pas les correlations -- mais il
    refuse un cycle dynamique plus etroit que le statique, qui
    contredirait le cours.
    """
    surcout = surcout_du_regime_dynamique(statique=300.0, dynamique=420.0)
    assert surcout == pytest.approx(120.0)
    assert surcout_du_regime_dynamique(statique=300.0, dynamique=300.0) == (
        pytest.approx(0.0)
    )


def test_un_cycle_dynamique_plus_etroit_que_le_statique_est_refuse() -> None:
    """Il faudrait que les courants induits **rendent** de l'energie."""
    with pytest.raises(ValueError, match="cycle dynamique plus etroit"):
        surcout_du_regime_dynamique(statique=300.0, dynamique=250.0)


def test_les_quatre_mecanismes_que_le_syllabus_nomme() -> None:
    """Hysteresis, courants de Foucault, trainage, hysteresis tournante.

    Les deux derniers sont ceux qu'on oublie. Le trainage est le retard
    de l'aimantation sur le champ -- « l'aimantation ne prend sa valeur
    d'equilibre qu'apres un certain temps » -- et l'hysteresis tournante
    apparait quand le champ garde sa grandeur mais tourne, ce qui est le
    cas dans une machine.
    """
    assert len(MECANISMES_DE_PERTE) == 4
    assert "Foucault" in " ".join(MECANISMES_DE_PERTE)
    assert "trainage" in " ".join(MECANISMES_DE_PERTE)


def test_une_aire_negative_est_refusee() -> None:
    """Un cycle d'aire negative serait parcouru en sens generateur."""
    with pytest.raises(ValueError, match="aire de cycle negative"):
        energie_du_cycle(aire=-10.0, volume=0.001)


def test_un_volume_negatif_est_refuse() -> None:
    """Pas de materiau, pas de pertes -- mais pas de volume negatif non plus."""
    with pytest.raises(ValueError, match="volume negatif"):
        energie_du_cycle(aire=350.0, volume=-0.001)


def test_une_frequence_negative_est_refusee() -> None:
    """Un cycle parcouru a frequence negative n'a pas de sens physique.

    Le signe du temps ne renverse pas la dissipation : un materiau ne
    restitue pas l'energie de son hysteresis.
    """
    with pytest.raises(ValueError, match="frequence negative"):
        pertes_par_hysteresis(aire=350.0, frequence=-50.0, volume=0.001)
