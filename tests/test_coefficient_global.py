"""Le chapitre 1 du tome 2 du syllabus : le coefficient global `K`.

Absent de la bibliotheque pour la meme raison que les deux precedents --
la mesure ne regardait que les planches. C'est pourtant la clef de voute
du tome 2 : tout dimensionnement d'echangeur, de paroi ou d'isolation s'y
ramene.

Le morceau le plus utile est la **linearisation du rayonnement**, et le
syllabus en donne l'argument complet en prose : on developpe en serie de
Mac Laurin, et « si on avait mis `T_a^4` en evidence, on aurait trouve
`4 T_a^3 (T_p - T_a)`. Il est donc plus exact de prendre » la temperature
**moyenne**. Cette superiorite n'est pas prise sur parole ici : les tests
la mesurent, en montrant que l'erreur de la moyenne est d'ordre deux en
l'ecart quand celle de la temperature ambiante n'est que d'ordre un.

Son usage, le syllabus le dit aussi : « juger par un calcul rapide si le
rayonnement est important ou negligeable par rapport a un autre mode de
transmission ».
"""

from __future__ import annotations

import pytest

from polytech.thermique.domain.coefficient_global import (
    CONSTANTE_DE_STEFAN,
    coefficient_de_rayonnement_exact,
    coefficient_de_rayonnement_linearise,
    coefficient_global_plan,
    temperature_de_paroi,
)


def test_sans_ecart_de_temperature_les_deux_coefficients_concordent() -> None:
    """A l'ecart nul la linearisation est exacte : c'est son point de tangence."""
    exact = coefficient_de_rayonnement_exact(0.9, 300.0, 300.0)
    linearise = coefficient_de_rayonnement_linearise(0.9, 300.0, 300.0)
    assert linearise == pytest.approx(exact)


def test_le_coefficient_linearise_suit_le_cube_de_la_temperature_moyenne() -> None:
    """« proportionnel a la troisieme puissance de la temperature moyenne ».

    Citation du syllabus. Doubler la temperature moyenne multiplie donc le
    coefficient par **huit** -- c'est pourquoi le rayonnement, negligeable
    a temperature ambiante, domine dans un four.
    """
    simple = coefficient_de_rayonnement_linearise(1.0, 300.0, 300.0)
    double = coefficient_de_rayonnement_linearise(1.0, 600.0, 600.0)
    assert double == pytest.approx(8 * simple)


def test_a_temperature_ambiante_le_coefficient_vaut_quelques_watts() -> None:
    """Environ `5,5 W/(m^2 K)` pour `epsilon = 0,9` a 293 K.

    Le syllabus en fait son premier exemple d'application -- radiateurs
    et tuyauteries entoures par les parois d'un local -- et c'est l'ordre
    de grandeur qui permet de le comparer d'un coup d'oeil a la
    convection naturelle, qui vaut quelques unites elle aussi. Les deux
    sont donc du meme ordre, ce qui est le point du chapitre.
    """
    coefficient = coefficient_de_rayonnement_linearise(0.9, 293.0, 293.0)
    assert 5.0 < coefficient < 6.0


def test_la_temperature_moyenne_est_plus_exacte_que_l_ambiante() -> None:
    """L'argument meme du syllabus, mesure plutot que cru.

    Prendre `4 sigma T_a^3` -- ce qu'on obtient en mettant `T_a^4` en
    evidence -- laisse une erreur **proportionnelle** a l'ecart ; prendre
    la moyenne la rend proportionnelle a son **carre**. Diviser l'ecart
    par deux divise donc la premiere erreur par deux et la seconde par
    quatre, et c'est ce que ce test observe.
    """
    ambiante = 300.0

    def erreurs(ecart: float) -> tuple[float, float]:
        paroi = ambiante + ecart
        exact = coefficient_de_rayonnement_exact(1.0, paroi, ambiante)
        par_la_moyenne = coefficient_de_rayonnement_linearise(1.0, paroi, ambiante)
        par_l_ambiante = 4 * CONSTANTE_DE_STEFAN * ambiante**3
        return abs(par_la_moyenne - exact), abs(par_l_ambiante - exact)

    moyenne_large, ambiante_large = erreurs(20.0)
    moyenne_fine, ambiante_fine = erreurs(10.0)
    assert moyenne_large / moyenne_fine == pytest.approx(4.0, rel=0.05)
    assert ambiante_large / ambiante_fine == pytest.approx(2.0, rel=0.05)


def test_le_coefficient_global_est_l_inverse_de_la_somme_des_resistances() -> None:
    """`1/K = 1/h_i + e/lambda + 1/h_e` : des resistances en serie.

    Ce sont les **resistances** qui s'ajoutent, et non les coefficients :
    c'est l'erreur que le chapitre previent, et elle est du cote
    optimiste -- additionner les coefficients donnerait un `K` bien plus
    grand, donc une paroi qu'on croirait bien plus deperditive qu'elle
    n'est.
    """
    global_ = coefficient_global_plan(
        echange_interieur=8.0,
        epaisseur=0.2,
        conductivite=1.0,
        echange_exterieur=25.0,
    )
    attendu = 1 / (1 / 8.0 + 0.2 / 1.0 + 1 / 25.0)
    assert global_ == pytest.approx(attendu)
    assert global_ < min(8.0, 25.0)


def test_la_resistance_la_plus_grande_commande_le_coefficient_global() -> None:
    """Ameliorer une resistance faible ne sert a rien : le chapitre y insiste.

    On double l'echange exterieur, qui est la plus petite des trois
    resistances : le coefficient global ne bouge presque pas. C'est le
    raisonnement qui decide ou mettre l'isolant.
    """
    avant = coefficient_global_plan(8.0, 0.2, 1.0, 25.0)
    apres = coefficient_global_plan(8.0, 0.2, 1.0, 50.0)
    assert apres / avant < 1.06


def test_la_temperature_de_paroi_se_deduit_du_flux_et_de_l_echange() -> None:
    """`t_p = t_fluide - phi / h` : la temperature superficielle de paroi.

    C'est la « determination des temperatures superficielles » du
    paragraphe 1.4. Elle sert a verifier qu'un isolant ne depasse pas sa temperature
    admissible, et c'est la raison pour laquelle le chapitre la traite.
    """
    assert temperature_de_paroi(fluide=293.0, flux=80.0, echange=8.0) == pytest.approx(
        283.0,
    )


def test_un_echange_nul_est_refuse_pour_la_temperature_de_paroi() -> None:
    """La paroi serait a temperature infinie : la division n'a pas de sens."""
    with pytest.raises(ValueError, match="coefficient d'echange nul ou negatif"):
        temperature_de_paroi(fluide=293.0, flux=80.0, echange=0.0)


def test_une_conductivite_nulle_est_refusee() -> None:
    """Une paroi parfaitement isolante a une resistance infinie, non nulle."""
    with pytest.raises(ValueError, match="conductivite nulle ou negative"):
        coefficient_global_plan(8.0, 0.2, 0.0, 25.0)


def test_une_emissivite_hors_de_zero_un_est_refusee() -> None:
    """Aucun corps ne rayonne plus que le corps noir."""
    with pytest.raises(ValueError, match="emissivite hors de"):
        coefficient_de_rayonnement_linearise(1.4, 300.0, 300.0)


def test_une_temperature_absolue_negative_est_refusee() -> None:
    """Les lois du rayonnement sont en kelvins, et le cube changerait de signe."""
    with pytest.raises(ValueError, match="temperature absolue negative"):
        coefficient_de_rayonnement_linearise(0.9, -5.0, 300.0)
