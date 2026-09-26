"""Les etapes de raisonnement du corrige sont derivees par la bibliotheque."""

import pytest
import sympy as sp

from polytech.fluides.domain.cinematique.champ import ChampDeVitesse
from polytech.fluides.domain.cinematique.symboles import T, X, Y

B = ChampDeVitesse(u=X * T, v=X - Y)
C = ChampDeVitesse(u=2 * X, v=-2 * Y)


def _egal(obtenu: sp.Expr, attendu: sp.Expr) -> bool:
    return bool(sp.simplify(obtenu - attendu) == 0)


def test_les_six_derivees_partielles_de_b_sont_derivees() -> None:
    d = B.derivees_partielles()
    assert _egal(d.du_dt, X)
    assert _egal(d.du_dx, T)
    assert _egal(d.du_dy, sp.Integer(0))
    assert _egal(d.dv_dt, sp.Integer(0))
    assert _egal(d.dv_dx, sp.Integer(1))
    assert _egal(d.dv_dy, sp.Integer(-1))


def test_le_gradient_temporel_de_b_vaut_x_selon_ex() -> None:
    dudt, dvdt = B.gradient_temporel()
    assert _egal(dudt, X)
    assert _egal(dvdt, sp.Integer(0))


def test_la_loi_de_volume_de_b_est_integree_depuis_la_divergence() -> None:
    """V(t)/V0 = exp(t^2/2 - t), obtenu en integrant div V."""
    assert _egal(B.loi_de_volume(), sp.exp(T**2 / 2 - T))


def test_le_volume_de_c_reste_constant() -> None:
    assert _egal(C.loi_de_volume(), sp.Integer(1))


def test_l_ecart_materiel_de_c_suit_les_exponentielles_de_l_etirement() -> None:
    assert _egal(C.ecart_materiel(axe=0), sp.exp(2 * T))
    assert _egal(C.ecart_materiel(axe=1), sp.exp(-2 * T))


def test_les_lignes_de_courant_de_c_sont_l_hyperbole_xy() -> None:
    equation = C.equation_lignes_de_courant()
    assert _egal(equation.lhs, X * Y)
    assert equation.rhs == sp.Symbol("C")


def test_un_axe_invalide_est_refuse() -> None:
    with pytest.raises(ValueError, match="axe"):
        C.ecart_materiel(axe=2)


def test_les_lignes_de_courant_de_b_dependent_du_temps() -> None:
    """Sans fonction de courant, il reste l'integration directe de dy/dx.

    C'est ce qui prouve que le reseau se deforme : la famille depend
    explicitement de t, alors que celle de C n'en depend pas.
    """
    y = B.lignes_de_courant_explicites()
    constante = sp.Symbol("C")
    assert y.has(T)
    assert y.has(constante)
    # la solution satisfait bien dy/dx = v/u
    residu = sp.simplify(sp.diff(y, X) - (X - y) / (X * T))
    assert residu == 0


def test_les_lignes_de_courant_de_c_ne_dependent_pas_du_temps() -> None:
    y = C.lignes_de_courant_explicites()
    assert not y.has(T)


@pytest.mark.parametrize("champ", [B, C])
def test_le_produit_des_ecarts_redonne_la_loi_de_volume(champ: ChampDeVitesse) -> None:
    """Coherence interne : l'aire est le produit des deux dimensions.

    Les ecarts materiels s'integrent depuis le tenseur, la loi de volume
    depuis la divergence -- deux chemins distincts qui doivent se rejoindre.
    """
    produit = champ.ecart_materiel(axe=0) * champ.ecart_materiel(axe=1)
    assert _egal(produit, champ.loi_de_volume())


def test_la_particule_de_b_s_etire_puis_se_comprime() -> None:
    """delta_x = e^{t^2/2} croit toujours ; delta_y = e^{-t} decroit toujours."""
    assert _egal(B.ecart_materiel(axe=0), sp.exp(T**2 / 2))
    assert _egal(B.ecart_materiel(axe=1), sp.exp(-T))
