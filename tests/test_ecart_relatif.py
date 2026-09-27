"""L'ecart relatif est rapporte a la reference, jamais a la mesure.

La question n'est pas rhetorique. Le corrige affichait un ecart rapporte a
la valeur *calculee*, tout en le commentant comme un ecart a la reponse
*publiee*. Les deux conventions coincidaient a l'affichage -- 0,27 % dans
les deux cas -- si bien qu'aucun chiffre ne trahissait l'ambiguite. Ces
tests fixent la convention et la rendent visible.
"""

import pytest

from polytech.fluides.domain.ecart import ecart_relatif


def test_l_ecart_se_rapporte_a_la_reference_et_non_a_la_mesure() -> None:
    """Cas ou les deux conventions divergent franchement.

    Rapportee a la reference, la difference vaut la moitie ; rapportee a la
    mesure, elle vaudrait le double. Aucun arrondi ne peut les confondre.
    """
    assert ecart_relatif(mesure=50.0, reference=100.0) == pytest.approx(0.5)


def test_l_ecart_ne_depend_pas_du_signe_de_la_difference() -> None:
    """Depasser la reference ou lui rester en deca se compte pareil."""
    en_dessous = ecart_relatif(mesure=90.0, reference=100.0)
    au_dessus = ecart_relatif(mesure=110.0, reference=100.0)
    assert en_dessous == pytest.approx(au_dessus)


def test_une_mesure_egale_a_la_reference_ne_montre_aucun_ecart() -> None:
    assert ecart_relatif(mesure=130.0, reference=130.0) == 0.0


def test_une_reference_nulle_est_refusee_et_non_contournee() -> None:
    """Diviser par zero rendrait un infini qui passerait pour un resultat."""
    with pytest.raises(ValueError, match="reference nulle"):
        ecart_relatif(mesure=1.0, reference=0.0)


def test_l_ecart_du_manometre_vaut_bien_ce_que_le_corrige_affiche() -> None:
    """Le cas reel : 129,65 kPa calcules contre 130 kPa publies."""
    assert ecart_relatif(mesure=129.65, reference=130.0) == pytest.approx(
        0.35 / 130.0,
    )
