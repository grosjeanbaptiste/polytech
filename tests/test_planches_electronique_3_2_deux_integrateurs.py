"""Les planches d'electronique : biquads a deux integrateurs (chapitre 3, II).

Source : `ElectroniqueInstrumentation/SLIDES/Slides_3_part2.pdf`, planches
44 a 66, lues sur les pages rendues -- leurs formules sont des images.

Chaque fonction de transfert imprimee est confrontee a l'analyse nodale du
montage dessine, AOP ideaux : le sommateur differentiel (planche 48), le
biquad KHN (49 a 52), le Tow-Thomas (55 a 57), le Fleischer-Tow (59, 60).
Suivent la mise en cascade (62 a 64) et le choix d'une technologie (66).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electronique.domain.aop import aop, sortie_ideale
from polytech.electronique.domain.aop_ideal import tensions_a_aop_ideaux
from polytech.electronique.domain.biquad import TowThomas
from polytech.electronique.domain.choix_de_filtre import (
    Section,
    Technologie,
    ordonner_les_sections,
    technologies_possibles,
)
from polytech.electronique.domain.deux_integrateurs import (
    fleischer_tow,
    khn,
    sommateur_differentiel,
    tow_thomas,
)
from polytech.electronique.domain.dimensionnement_khn import (
    gain_du_khn,
    rapports_du_khn,
    regles_de_fleischer_tow,
)
from polytech.electronique.domain.netliste import Admittance, SourceDeTension

S = sp.Symbol("s")
R = sp.symbols("R_1:7", positive=True)
C1, C2, RR = sp.symbols("C_1 C_2 R", positive=True)


def test_le_sommateur_differentiel_generalise() -> None:
    """Planche 48 : deux entrees sur `+`, une sur `-`, contre-reaction `R_f`."""
    r1, r2, r3, rf = sp.symbols("R_1 R_2 R_3 R_f", positive=True)
    k1, k2, k3 = sommateur_differentiel(r1, r2, r3, rf)
    assert sp.simplify(k1 - (rf + r3) / r3 * r2 / (r1 + r2)) == 0
    assert sp.simplify(k2 - (rf + r3) / r3 * r1 / (r1 + r2)) == 0
    assert sp.simplify(k3 + rf / r3) == 0


def test_le_khn_passe_bas() -> None:
    """Planche 49 : `E_2 / E_1`, numerateur `R_4 (R_5 + R_6) / (R_5 (R_3 + R_4))`."""
    r1, r2, r3, r4, r5, r6 = R
    bas, _, _ = khn(R, (C1, C2), S)
    produit = r1 * r2 * C1 * C2
    numerateur = r4 / r5 * (r5 + r6) / (r3 + r4) / produit
    lineaire = r3 / r5 * (r5 + r6) / (r3 + r4) / (r1 * C1)
    planche = numerateur / (S**2 + lineaire * S + r6 / r5 / produit)
    assert sp.simplify(bas - planche) == 0


def test_le_khn_donne_aussi_un_passe_bande_et_un_passe_haut() -> None:
    """Planche 52 : `E_3 = -s R_2 C_2 E_2`, `E_4 = -s R_1 C_1 E_3`."""
    r1, r2 = R[0], R[1]
    bas, bande, haut = khn(R, (C1, C2), S)
    assert sp.simplify(bande + S * r2 * C2 * bas) == 0
    assert sp.simplify(haut + S * r1 * C1 * bande) == 0
    r3, r4, r5, r6 = R[2:]
    a_l_infini = sp.limit(haut, S, sp.oo)
    assert sp.simplify(a_l_infini - r4 * (r5 + r6) / (r5 * (r3 + r4))) == 0
    assert bas.subs(S, 0) != 0
    assert bande.subs(S, 0) == 0
    assert haut.subs(S, 0) == 0


def test_la_pulsation_la_qualite_et_le_gain_du_khn() -> None:
    """Planche 50 : `w_0`, `Q` et `G` identifies sur la forme canonique."""
    r1, r2, r3, r4, r5, r6 = R
    bas, _, _ = khn(R, (C1, C2), S)
    pulsation = sp.sqrt(r6 / r5 / (r1 * r2 * C1 * C2))
    qualite = r5 / r3 * (r3 + r4) / (r5 + r6) * sp.sqrt(r6 / r5 * r1 * C1 / (r2 * C2))
    gain = r4 / r6 * (r5 + r6) / (r3 + r4)
    canonique = gain * pulsation**2 / (S**2 + pulsation / qualite * S + pulsation**2)
    assert sp.simplify(bas - canonique) == 0
    assert sp.simplify(gain_du_khn(r3, r4, r5, r6) - gain) == 0


def test_la_methode_de_dimensionnement_du_khn() -> None:
    """Planche 51 : `R_6 / R_5 = w_0^2`, puis `R_4 / R_3` d'apres `Q`."""
    pulsation, qualite = sp.Integer(3), sp.Rational(5, 2)
    six_sur_cinq, quatre_sur_trois = rapports_du_khn(pulsation, qualite)
    assert six_sur_cinq == 9
    assert quatre_sur_trois == qualite * (1 + 9) / 3 - 1
    resistances = (1, 1, 1, quatre_sur_trois, 1, six_sur_cinq)
    bas, _, _ = khn(resistances, (1, 1), S)
    denominateur = sp.Poly(sp.denom(sp.cancel(bas)), S).monic()
    assert denominateur.all_coeffs() == [1, pulsation / qualite, pulsation**2]
    with pytest.raises(ValueError, match="facteur de qualite trop faible"):
        rapports_du_khn(3, sp.Rational(1, 10))
    for w, q in ((0, 1), (1, 0)):
        with pytest.raises(ValueError, match="nul ou negatif"):
            rapports_du_khn(w, q)


def test_le_tow_thomas() -> None:
    """Planche 55 : un passe-bas inverseur en `E_2`, un passe-bande en `E_3`."""
    r1, r2, r3, r4 = R[:4]
    bas, bande = tow_thomas((r1, r2, r3, r4), (C1, C2), S)
    denominateur = S**2 + S / (r1 * C1) + 1 / (r2 * r3 * C1 * C2)
    assert sp.simplify(bas + 1 / (r2 * r4 * C1 * C2) / denominateur) == 0
    assert sp.simplify(bande + S / (r4 * C1) / denominateur) == 0


def test_le_dimensionnement_du_tow_thomas() -> None:
    """Planche 57 : `R_1 = Q / w_0`, `R_2 = R_3 = 1 / w_0`, `R_4 = 1 / (G w_0)`."""
    cellule = TowThomas(pulsation=4.0, qualite=3.0, gain=2.0)
    r1, r2, r4 = (
        cellule.r1_normalisee(),
        cellule.r2_normalisee(),
        cellule.r4_normalisee(),
    )
    exacts = tuple(sp.nsimplify(r) for r in (r1, r2, r2, r4))
    bas, _ = tow_thomas(exacts, (1, 1), S)
    attendu = -2 * 16 / (S**2 + sp.Rational(4, 3) * S + 16)
    assert sp.simplify(bas - attendu) == 0


def test_le_fleischer_tow() -> None:
    """Planche 59 : un numerateur complet, le denominateur du Tow-Thomas."""
    r1, r2, r3, r4, r5, r6 = R
    transfert = fleischer_tow(R, RR, (C1, C2), S)
    numerateur = (
        RR / r5 * S**2 + (RR / r5 - r1 / r4) / (r1 * C1) * S + 1 / (r3 * r6 * C1 * C2)
    )
    denominateur = S**2 + S / (r1 * C1) + 1 / (r2 * r3 * C1 * C2)
    assert sp.simplify(transfert + numerateur / denominateur) == 0


@pytest.mark.parametrize(
    ("nature", "coefficients"),
    [("passe-bas", (0, 0, 1)), ("passe-haut", (1, 0, 0)), ("passe-bande", (0, -1, 0))],
)
def test_les_regles_du_fleischer_tow(
    nature: str, coefficients: tuple[int, int, int]
) -> None:
    """Planche 60 : `R_4`, `R_5`, `R_6` selon le type, l'infini retirant la branche.

    Le numerateur attendu est `-G (a s^2 + b b_1 s + c b_0)` : le passe-bas
    et le passe-haut sortent inverses ; le passe-bande, non.
    """
    pulsation, qualite, gain, r = sp.Integer(2), sp.Integer(5), sp.Integer(3), 7
    r1, r2 = qualite / pulsation, 1 / pulsation
    r4, r5, r6 = regles_de_fleischer_tow(nature, gain, r1=r1, r2=r2, r=r)
    transfert = fleischer_tow((r1, r2, r2, r4, r5, r6), r, (1, 1), S)
    b0, b1 = pulsation**2, pulsation / qualite
    carre, lineaire, constant = coefficients
    numerateur = gain * (carre * S**2 + lineaire * b1 * S + constant * b0)
    assert sp.simplify(transfert + numerateur / (S**2 + b1 * S + b0)) == 0


def test_les_regles_refusent_un_type_inconnu() -> None:
    """Le coupe-bande demande `a_2` et `a_0` : la planche ne le regle pas seule."""
    with pytest.raises(ValueError, match="type de filtre inconnu"):
        regles_de_fleischer_tow("coupe-bande", 1, r1=1, r2=1, r=1)
    with pytest.raises(ValueError, match="gain nul ou negatif"):
        regles_de_fleischer_tow("passe-bas", 0, r1=1, r2=1, r=1)


def test_l_ordre_des_sections_d_une_cascade() -> None:
    """Planche 64 : le passe-bas en tete, le passe-haut en queue."""
    sections = (Section.PASSE_HAUT, Section.PASSE_BANDE, Section.PASSE_BAS)
    assert ordonner_les_sections(sections) == (
        Section.PASSE_BAS,
        Section.PASSE_BANDE,
        Section.PASSE_HAUT,
    )
    doubles = (Section.PASSE_BANDE, Section.PASSE_BAS, Section.PASSE_BAS)
    assert ordonner_les_sections(doubles)[:2] == (Section.PASSE_BAS, Section.PASSE_BAS)
    with pytest.raises(ValueError, match="aucune section"):
        ordonner_les_sections(())


def test_deux_decoupages_d_un_meme_passe_bande() -> None:
    """Planche 63 : deux passe-bande, ou un passe-haut suivi d'un passe-bas."""
    premier = S**2 + sp.Rational(573, 10000) * S + sp.Rational(10937, 10000)
    second = S**2 + sp.Rational(524, 10000) * S + sp.Rational(9144, 10000)
    deux_bandes = (S / premier) * (S / second)
    haut_puis_bas = (S**2 / premier) * (1 / second)
    bas_puis_haut = (1 / premier) * (S**2 / second)
    assert sp.simplify(deux_bandes - haut_puis_bas) == 0
    assert sp.simplify(deux_bandes - bas_puis_haut) == 0


def test_la_technologie_selon_la_frequence() -> None:
    """Planche 66 : ordres de grandeur « tres, tres grossiers »."""
    assert technologies_possibles(1e3) == (
        Technologie.CAPACITES_COMMUTEES,
        Technologie.ACTIF_A_AOP,
    )
    assert technologies_possibles(1e6) == (Technologie.ACTIF_A_AOP,)
    assert technologies_possibles(100e6) == (
        Technologie.TRANSCONDUCTANCE,
        Technologie.ECHELLE_LC,
    )
    assert technologies_possibles(900e6) == (Technologie.ECHELLE_LC,)
    assert technologies_possibles(5e9) == (
        Technologie.ECHELLE_LC,
        Technologie.MICRORUBAN,
    )
    assert technologies_possibles(50e9) == (Technologie.MICRORUBAN,)
    assert technologies_possibles(100e3) == (Technologie.ACTIF_A_AOP,)
    assert technologies_possibles(10e6) == (
        Technologie.TRANSCONDUCTANCE,
        Technologie.ECHELLE_LC,
    )
    with pytest.raises(ValueError, match="frequence negative"):
        technologies_possibles(-1.0)


def test_le_zero_virtuel_pose_vaut_le_zero_virtuel_etabli() -> None:
    """Un integrateur a fuite : l'AOP ideal pose rend la limite du gain fini."""
    r1, rf, c = sp.symbols("R_1 R_f C", positive=True)
    branches = [Admittance(1, 2, 1 / r1), Admittance(2, 3, 1 / rf + S * c)]
    (_, moins, pose) = tensions_a_aop_ideaux(branches, [(0, 2, 3)], {1: 1}, 3)
    source = SourceDeTension(1, 0, sp.Integer(1), sp.Symbol("I_e"))
    etabli = sortie_ideale([source, *branches, *aop(0, 2, 3)], 3, 3)
    assert sp.simplify(pose - etabli) == 0
    assert sp.simplify(pose + rf / (r1 * (1 + S * rf * c))) == 0
    assert moins == 0


def test_un_suiveur_et_un_non_inverseur_par_le_zero_virtuel() -> None:
    """L'entree `+` peut etre un noeud : `1 + R_2 / R_1`, et 1 sans pont."""
    r1, r2 = sp.symbols("R_1 R_2", positive=True)
    pont = [Admittance(2, 0, 1 / r1), Admittance(2, 3, 1 / r2)]
    tensions = tensions_a_aop_ideaux(pont, [(1, 2, 3)], {1: 1}, 3)
    assert sp.simplify(tensions[2] - (1 + r2 / r1)) == 0
    assert tensions[1] == 1


def test_le_zero_virtuel_refuse_un_circuit_mal_pose() -> None:
    """Sans contre-reaction, ou avec une sortie imposee, rien ne se resout."""
    with pytest.raises(ValueError, match="sans solution unique"):
        tensions_a_aop_ideaux([Admittance(1, 2, 1)], [(0, 2, 3)], {1: 1}, 3)
    with pytest.raises(ValueError, match="partagee, ou imposee"):
        tensions_a_aop_ideaux([Admittance(1, 2, 1)], [(0, 2, 1)], {1: 1}, 2)
    with pytest.raises(ValueError, match="partagee, ou imposee"):
        tensions_a_aop_ideaux([], [(0, 1, 3), (0, 2, 3)], {1: 1}, 3)
