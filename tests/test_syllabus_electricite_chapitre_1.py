"""Syllabus d'electricite generale, chapitre 1 : « Notions de base ».

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 1-1 a 1-14, lues sur les pages rendues (le texte converti perd les
formules). Les paragraphes 1.3 et 1.4 sont deja portes par
`tests/test_charges_de_la_matiere.py` ; ce fichier porte 1.2, 1.5, 1.6 et
1.7.

Deux ecarts, etablis ici :

* exemple 1.5.2, page 1-12 : le syllabus imprime `35 . 10^27 C/m^3` pour un
  noyau de 92 protons et de 1 fm de rayon. `Q / (4 pi R^3 / 3)` donne
  `3,5 . 10^27` : un facteur dix ;
* relation (1.6.5), page 1-12 : « lorsque f(x) ne possede que des poles
  simples » -- ce sont les **zeros** simples de `f` qui portent les pics.

La figure 1.5.1 a ete rendue a 230 points par pouce pour en lire les
valeurs : chaque intervalle `[k, k+1[` porte `k^2` charges equidistantes
(« n = x^2 »), et `lambda / e` est le nombre de charges de la fenetre
divise par sa largeur.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.charges import (
    PARTICULES,
    charge_d_un_systeme,
    nombre_de_charges_elementaires,
)
from polytech.electromagnetisme.domain.constantes import CHARGE_ELEMENTAIRE
from polytech.electromagnetisme.domain.electrons_lies import moyenne_temporelle
from polytech.electromagnetisme.domain.moyennes_macroscopiques import (
    charge_lineique_d_un_filet,
    charge_moyenne,
    charge_surfacique_d_une_couche,
    charge_volumique_d_une_sphere,
    charge_volumique_des_charges_ponctuelles,
    charges_de_la_fenetre,
)
from polytech.electromagnetisme.domain.symboles import (
    CARTESIENNES,
    PULSATION,
    TEMPS,
)
from polytech.maths.domain.fourier.dirac import (
    appliquer_la_derivee_du_dirac,
    dirac_d_une_fonction,
)
from polytech.maths.domain.fourier.distributions import appliquer_le_dirac
from polytech.maths.domain.symboles import VARIABLE

LIGNE = [k + Fraction(j, k**2) for k in range(1, 5) for j in range(k**2)]
"""Figure 1.5.1 (a) : une charge en 1, quatre de 2 a 3, neuf de 3 a 4, seize ensuite."""

PERIODE = 2 * sp.pi / PULSATION


def _lambda_sur_e(debut: Fraction | int, largeur: Fraction | int) -> Fraction:
    fenetre = charges_de_la_fenetre(LIGNE, debut, largeur)
    return charge_moyenne([1] * len(fenetre), largeur)


# --- 1.2 Charges electriques -------------------------------------------------


def test_la_charge_elementaire_du_syllabus() -> None:
    """(1.2.1) : e = 1,602 177 33 . 10^-19 C, soit 1,6 . 10^-19 C."""
    assert math.isclose(CHARGE_ELEMENTAIRE, 1.60217733e-19, rel_tol=1e-6)
    assert math.isclose(CHARGE_ELEMENTAIRE, 1.6e-19, rel_tol=2e-3)


def test_la_charge_d_un_systeme_est_un_multiple_entier() -> None:
    """(1.2.2) : Q = N e, « ou N est un nombre entier positif ou negatif »."""
    assert charge_d_un_systeme(3) == 3 * CHARGE_ELEMENTAIRE
    assert charge_d_un_systeme(-2) == -2 * CHARGE_ELEMENTAIRE
    assert charge_d_un_systeme(0) == 0


def test_le_nombre_de_charges_d_une_charge_donnee() -> None:
    """Le noyau d'uranium de l'exemple 1.5.2 : Q = 92 e."""
    assert nombre_de_charges_elementaires(92 * CHARGE_ELEMENTAIRE) == 92
    assert nombre_de_charges_elementaires(-2 * CHARGE_ELEMENTAIRE) == -2
    assert nombre_de_charges_elementaires(0.0) == 0


def test_une_charge_fractionnaire_n_existe_pas() -> None:
    """« Ne peut varier que par multiples entiers d'une charge elementaire »."""
    with pytest.raises(ValueError, match="pas un multiple entier de e"):
        nombre_de_charges_elementaires(1.5 * CHARGE_ELEMENTAIRE)
    with pytest.raises(TypeError, match="un nombre entier de charges"):
        charge_d_un_systeme(1.5)


def test_le_tableau_1_3_1_des_particules() -> None:
    """Page 1-3 : l'electron porte -e, le proton +e, le neutron et le photon 0."""
    charges = {nom: particule.charges for nom, particule in PARTICULES.items()}
    assert charges == {"proton": 1, "electron": -1, "neutron": 0, "photon": 0}
    assert PARTICULES["photon"].masse == 0
    assert PARTICULES["photon"].spin == 1
    assert {PARTICULES[n].spin for n in ("proton", "electron", "neutron")} == {
        sp.Rational(1, 2)
    }


def test_le_proton_est_1836_fois_plus_lourd_que_l_electron() -> None:
    """Page 1-3, sous le tableau : « 1 836 fois plus lourd »."""
    rapport = PARTICULES["proton"].masse / PARTICULES["electron"].masse
    assert math.isclose(rapport, 1836, rel_tol=1e-3)
    assert PARTICULES["neutron"].masse > PARTICULES["proton"].masse


def test_la_charge_d_une_particule_en_coulombs() -> None:
    """Colonne « Charge (C) » : 1,602 . 10^-19, son oppose, et zero."""
    assert PARTICULES["proton"].charge == CHARGE_ELEMENTAIRE
    assert PARTICULES["electron"].charge == -CHARGE_ELEMENTAIRE
    assert PARTICULES["neutron"].charge == 0


# --- 1.5 Moyennes macroscopiques ---------------------------------------------


def test_la_ligne_de_l_exemple_1_5_1_porte_trente_charges() -> None:
    """« n = x^2 » : 1 + 4 + 9 + 16 charges, la premiere en x = 1."""
    assert len(LIGNE) == 30
    assert LIGNE[:5] == [1.0, 2.0, 2.25, 2.5, 2.75]


@pytest.mark.parametrize(("debut", "attendu"), [(1, 1), (2, 4), (3, 9), (4, 16)])
def test_la_figure_1_5_1_d_pour_une_fenetre_de_largeur_1(
    debut: int, attendu: int
) -> None:
    """« La valeur correcte » : lambda/e = x^2, soit 1, 4, 9, 16."""
    assert _lambda_sur_e(debut, 1) == attendu


def test_la_figure_1_5_1_c_pour_une_fenetre_de_largeur_un_demi() -> None:
    """De 1 a 4,5 par pas de 0,5 : 2, 0, 4, 4, 10, 8, 16, 16."""
    demi = Fraction(1, 2)
    lus = [_lambda_sur_e(1 + rang * demi, demi) for rang in range(8)]
    assert lus == [2, 0, 4, 4, 10, 8, 16, 16]


def test_la_figure_1_5_1_b_fluctue_pour_une_fenetre_trop_petite() -> None:
    """Largeur 0,1 : la courbe saute entre 0, 10 et 20."""
    dixieme = Fraction(1, 10)
    lus = [_lambda_sur_e(1 + rang * dixieme, dixieme) for rang in range(40)]
    assert set(lus) == {0, 10, 20}
    assert lus[:10] == [10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert max(lus[:30]) == 10
    assert min(lus[30:]) == 10


def test_la_figure_1_5_1_e_et_f_pour_des_fenetres_trop_grandes() -> None:
    """Largeur 2 : 2,5 puis 12,5 ; largeur 4 : 7,5 -- la variation est gommee."""
    assert [_lambda_sur_e(1, 2), _lambda_sur_e(3, 2)] == [2.5, 12.5]
    assert _lambda_sur_e(1, 4) == 7.5


def test_la_fenetre_contient_son_debut_et_non_sa_fin() -> None:
    """La charge placee en x = 2 compte dans `[2, 3[`, pas dans `[1, 2[`."""
    assert charges_de_la_fenetre([1.0, 2.0], 1, 1) == [1.0]
    assert charges_de_la_fenetre([1.0, 2.0], 2, 1) == [2.0]


def test_la_charge_moyenne_est_la_somme_des_charges_sur_l_etendue() -> None:
    """(1.5.1), (1.5.5), (1.5.8) : la meme moyenne, en volume, en aire, en longueur."""
    assert charge_moyenne([1.0, -3.0, 4.0], 0.5) == 4.0
    assert charge_moyenne([], 2.0) == 0.0


@pytest.mark.parametrize("etendue", [0.0, -1.0])
def test_une_etendue_nulle_ou_negative_est_refusee(etendue: float) -> None:
    """« Si d tend vers zero, le rapport tend vers zero ou l'infini »."""
    with pytest.raises(ValueError, match="etendue strictement positive attendue"):
        charge_moyenne([1.0], etendue)
    with pytest.raises(ValueError, match="etendue strictement positive attendue"):
        charges_de_la_fenetre([1.0], 0.0, etendue)


def test_la_couche_et_le_filet_electriques() -> None:
    """(1.5.4) : sigma = eta a ; (1.5.7) : lambda = eta s."""
    assert charge_surfacique_d_une_couche(2.0, 1e-3) == 2e-3
    assert charge_lineique_d_un_filet(2.0, 1e-6) == 2e-6


def test_le_noyau_d_uranium_de_l_exemple_1_5_2() -> None:
    """Q = 92 e sur une sphere de 1 fm : 3,5 . 10^27 C/m^3, et non 35 . 10^27."""
    eta = charge_volumique_d_une_sphere(charge_d_un_systeme(92), 1e-15)
    assert math.isclose(eta, 3.5e27, rel_tol=0.01)
    assert not math.isclose(eta, 35e27, rel_tol=0.5)


def test_la_charge_volumique_rend_la_charge_de_la_sphere() -> None:
    """(1.5.3) : Q = integrale de eta sur le volume."""
    eta = charge_volumique_d_une_sphere(3.0, 2.0)
    assert math.isclose(eta * 4 / 3 * math.pi * 2.0**3, 3.0)
    with pytest.raises(ValueError, match="rayon strictement positif attendu"):
        charge_volumique_d_une_sphere(3.0, 0.0)


# --- 1.6 Notions avancees : la distribution de Dirac --------------------------


def test_le_dirac_evalue_la_fonction() -> None:
    """(1.6.3) : integrale de f(x) delta(x - a) = f(a)."""
    assert appliquer_le_dirac(2, VARIABLE**2 + 1) == 5


def test_la_derivee_du_dirac_rend_l_oppose_de_la_derivee() -> None:
    """(1.6.4) : integrale de f(x) delta'(x - a) = -f'(a)."""
    assert appliquer_la_derivee_du_dirac(2, VARIABLE**2) == -4
    brut = sp.integrate(
        VARIABLE**2 * sp.DiracDelta(VARIABLE - 2, 1), (VARIABLE, -sp.oo, sp.oo)
    )
    assert brut == -4


def test_le_dirac_d_une_fonction_a_deux_zeros() -> None:
    """(1.6.5) : delta(x^2 - 1) = (delta(x - 1) + delta(x + 1)) / 2."""
    assert dirac_d_une_fonction(VARIABLE**2 - 1) == {
        sp.Integer(-1): sp.Rational(1, 2),
        sp.Integer(1): sp.Rational(1, 2),
    }


def test_le_dirac_d_une_fonction_recoupe_par_l_integrale() -> None:
    """Somme des f(x_i) / |g'(x_i)|, comparee a l'integrale que sympy calcule."""
    fonction, interieur = sp.cos(VARIABLE), VARIABLE**2 - 1
    pics = dirac_d_une_fonction(interieur)
    somme = sum(poids * appliquer_le_dirac(x, fonction) for x, poids in pics.items())
    integrale = sp.integrate(
        fonction * sp.DiracDelta(interieur), (VARIABLE, -sp.oo, sp.oo)
    )
    assert sp.simplify(somme - integrale) == 0


def test_un_changement_d_echelle_divise_le_poids() -> None:
    """delta(2x - 4) = delta(x - 2) / 2."""
    assert dirac_d_une_fonction(2 * VARIABLE - 4) == {sp.Integer(2): sp.Rational(1, 2)}


def test_un_zero_double_est_refuse() -> None:
    """(1.6.5) ne vaut que pour des zeros simples : en un zero double, f' s'annule."""
    with pytest.raises(ValueError, match="zero multiple"):
        dirac_d_une_fonction((VARIABLE - 1) ** 2)


def test_une_fonction_sans_zero_reel_ne_porte_aucun_pic() -> None:
    """delta(x^2 + 1) est nulle : la somme est vide."""
    assert dirac_d_une_fonction(VARIABLE**2 + 1) == {}


def test_des_charges_ponctuelles_s_ecrivent_comme_une_charge_volumique() -> None:
    """(1.6.8) : eta(r) = somme des q_i delta(r - r_i), dont l'integrale rend Q."""
    eta = charge_volumique_des_charges_ponctuelles({(0, 0, 0): 2, (1, 2, 3): -5})
    bornes = [(axe, -sp.oo, sp.oo) for axe in CARTESIENNES]
    assert sp.integrate(eta, *bornes) == -3
    dans_la_boite = [
        (axe, sp.Rational(-1, 2), sp.Rational(1, 2)) for axe in CARTESIENNES
    ]
    assert sp.integrate(eta, *dans_la_boite) == 2


# --- 1.7 Mouvement des electrons lies -----------------------------------------


def test_la_position_moyenne_de_l_electron_lie_est_celle_du_noyau() -> None:
    """(1.7.1) : sur une orbite fermee, la moyenne de r_A' est r_A."""
    grand_axe, noyau = sp.Symbol("a", positive=True), sp.Symbol("x_A", real=True)
    position = noyau + grand_axe * sp.cos(PULSATION * TEMPS)
    assert moyenne_temporelle(position, PERIODE) == noyau


def test_la_vitesse_moyenne_de_l_electron_lie_est_celle_du_noyau() -> None:
    """(1.7.2) et (1.7.3) : le mouvement relatif, ferme, ne contribue pas."""
    rayon, vitesse = sp.symbols("a v_A", positive=True)
    position = vitesse * TEMPS + rayon * sp.sin(PULSATION * TEMPS)
    assert moyenne_temporelle(sp.diff(position, TEMPS), PERIODE) == vitesse


def test_la_moyenne_d_un_produit_n_est_pas_le_produit_des_moyennes() -> None:
    """Remarque de la page 1-14 : x' et v' pris ensemble se moyennent a part."""
    rayon = sp.Symbol("a", positive=True)
    abscisse = rayon * sp.cos(PULSATION * TEMPS)
    vitesse = sp.diff(rayon * sp.sin(PULSATION * TEMPS), TEMPS)
    produit = moyenne_temporelle(abscisse * vitesse, PERIODE)
    assert sp.simplify(produit - rayon**2 * PULSATION / 2) == 0
    assert moyenne_temporelle(abscisse, PERIODE) == 0
    assert moyenne_temporelle(vitesse, PERIODE) == 0


def test_une_duree_d_observation_nulle_est_refusee() -> None:
    """(1.7.4) : la moyenne se prend sur une duree T, strictement positive."""
    with pytest.raises(ValueError, match="duree strictement positive attendue"):
        moyenne_temporelle(TEMPS, 0)
