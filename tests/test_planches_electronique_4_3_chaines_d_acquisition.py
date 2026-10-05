"""Les planches d'electronique : cascades, convertisseurs, multiplexeurs (4, III).

Source : `ElectroniqueInstrumentation/SLIDES/Slides_4_part3.pdf`, 52
planches lues sur leur couche de texte.

Les oracles : la formule de Friis et son etablissement a deux etages
(planches 5 a 18), `LSB = 2 V_r / (2^n - 1)` et le convertisseur quatre
bits sur +-5 V echantillonnant 5 sin(2 pi 100 t) a 2 kHz (planches 24 a
27), le multiplexage de deux signaux a 200 Hz (planche 41) et les cinq
architectures types (planches 46 a 50).

Deux ecarts : la planche 21 ecrit « outside the frequency range [0, f_s] »
la ou son integrale s'arrete a `f_s / 2` ; la planche 52 annonce un
passe-bas « of cutoff frequency 100 Hz » dont la fonction de transfert,
`2 pi 200 / (s + 2 pi 200)`, coupe a 200 Hz.
"""

from __future__ import annotations

import itertools
import math

import pytest
import sympy as sp

from polytech.electronique.domain.acquisition import cadence_du_multiplexeur, nyquist
from polytech.electronique.domain.bruit_thermique import tension_efficace_thermique
from polytech.electronique.domain.chaines_types import Architecture, composants
from polytech.electronique.domain.conversion import (
    ConvertisseurAnalogiqueNumerique,
    EchantillonnageDeLEtendue,
    etat_de_l_echantillonnage,
)
from polytech.electronique.domain.facteur_de_bruit import facteur_de_bruit
from polytech.electronique.domain.friis import facteur_d_une_cascade, facteur_en_cascade
from polytech.electronique.domain.multiplexage import (
    convertisseurs_economises,
    instants_de_la_voie,
    paires_multiplexeur_convertisseur,
    residu_de_commutation,
)
from polytech.electronique.domain.quantification import (
    Quantificateur,
    echantillon_a_ouverture,
)
from polytech.signaux.domain.echantillonnage import frequence_apparente
from polytech.signaux.domain.processus_aleatoires import part_de_puissance_repliee

SOURCE, BANDE, T0 = 50.0, 1e4, 290.0
THERMIQUE = tension_efficace_thermique(SOURCE, T0, BANDE) ** 2
PREMIER, SECOND = (20e-9) ** 2 * BANDE, (200e-9) ** 2 * BANDE
"""Les puissances de bruit en tension de deux etages, sur la bande."""


def _rapport_en_sortie(
    gains: tuple[float, float], bruits: tuple[float, float]
) -> float:
    """Planches 9 et 10 : le bruit de chaque etage, amplifie par ce qui le suit."""
    a1, a2 = gains
    signal = (1e-6) ** 2
    bruit = a1**2 * a2**2 * (THERMIQUE + bruits[0]) + a2**2 * bruits[1]
    return a1**2 * a2**2 * signal / bruit


def test_le_rapport_signal_sur_bruit_d_une_cascade() -> None:
    """Planche 12 : le gain du dernier etage n'y contribue pas."""
    reference = _rapport_en_sortie((10.0, 5.0), (PREMIER, SECOND))
    assert _rapport_en_sortie((10.0, 500.0), (PREMIER, SECOND)) == pytest.approx(
        reference, rel=1e-9, abs=0
    )
    attendu = (1e-6) ** 2 / (THERMIQUE + PREMIER + SECOND / 10.0**2)
    assert reference == pytest.approx(attendu, rel=1e-9, abs=0)


def test_la_formule_de_friis_a_deux_etages() -> None:
    """Planches 14 a 17 : `F = F1 + (F2 - 1) / A1^2`."""
    f1, f2 = 1 + PREMIER / THERMIQUE, 1 + SECOND / THERMIQUE
    entree = (1e-6) ** 2 / THERMIQUE
    sortie = _rapport_en_sortie((10.0, 5.0), (PREMIER, SECOND))
    global_ = facteur_de_bruit(entree, sortie)
    assert global_ == pytest.approx(facteur_en_cascade(f1, f2, 10.0), rel=1e-9, abs=0)
    assert global_ == pytest.approx(f1 + (f2 - 1) / 100, rel=1e-9, abs=0)


def test_l_amplificateur_le_moins_bruyant_passe_en_premier() -> None:
    """Planches 5 et 17 : l'ordre compte, et le premier etage decide."""
    f1, f2 = 1 + PREMIER / THERMIQUE, 1 + SECOND / THERMIQUE
    assert f1 < f2
    bon = facteur_d_une_cascade(((f1, 10.0), (f2, 10.0)))
    mauvais = facteur_d_une_cascade(((f2, 10.0), (f1, 10.0)))
    assert bon < mauvais
    assert bon - f1 < 0.02 * (f2 - 1)


def test_la_formule_de_friis_generalisee() -> None:
    """Planche 18 : `F1 + (F2 - 1)/A1^2 + (F3 - 1)/(A1 A2)^2`, ou en gains `G = A^2`."""
    etages = ((1.5, 10.0), (4.0, 5.0), (10.0, 2.0))
    en_tension = 1.5 + (4.0 - 1) / 10.0**2 + (10.0 - 1) / (10.0 * 5.0) ** 2
    assert facteur_d_une_cascade(etages) == pytest.approx(en_tension, rel=1e-9, abs=0)
    g1, g2 = 10.0**2, 5.0**2
    assert en_tension == pytest.approx(
        1.5 + 3.0 / g1 + 9.0 / (g1 * g2), rel=1e-9, abs=0
    )


def test_le_theoreme_d_echantillonnage() -> None:
    """Planche 21 : `f_s >= 2 f_max`, sinon le spectre se replie."""
    assert nyquist(500.0) == 1000.0
    assert frequence_apparente(100.0, 2000.0) == pytest.approx(100.0, rel=1e-9, abs=0)
    assert frequence_apparente(1700.0, 2000.0) == pytest.approx(300.0, rel=1e-9, abs=0)


def test_le_repliement_est_negligeable_si_la_puissance_au_dela_l_est() -> None:
    """Planche 21 : `int_0^(f_s/2) S >> int_(f_s/2)^oo S`, et non « [0, f_s] »."""
    frequence = sp.Symbol("f", real=True)
    densite = 1 / (1 + (frequence / 100) ** 2)
    large = part_de_puissance_repliee(densite, frequence, 20000)
    serree = part_de_puissance_repliee(densite, frequence, 200)
    attendue = (sp.pi / 2 - sp.atan(100)) / sp.atan(100)
    assert sp.simplify(large - attendue) == 0
    assert float(large) < 0.01
    assert serree == 1
    jusqu_a_la_cadence = (sp.pi / 2 - sp.atan(200)) / sp.atan(200)
    assert float(jusqu_a_la_cadence) < float(large) / 1.9


QUATRE_BITS = Quantificateur(reference=5.0, bits=4)


def test_le_pas_de_quantification() -> None:
    """Planche 24 : `LSB = 2 V_r / (2^n - 1)` ; 0,667 V sur +-5 V et quatre bits."""
    assert QUATRE_BITS.pas() == pytest.approx(10 / 15, rel=1e-9, abs=0)
    lie = ConvertisseurAnalogiqueNumerique(bits=4, pleine_echelle_volts=10.0)
    assert QUATRE_BITS.pas() == pytest.approx(lie.pas_exact(), rel=1e-9, abs=0)
    assert QUATRE_BITS.code(-5.0) == 0
    assert QUATRE_BITS.code(5.0) == 15
    assert QUATRE_BITS.niveau(15) == pytest.approx(5.0, rel=1e-9, abs=0)
    assert QUATRE_BITS.niveau(8) - QUATRE_BITS.niveau(7) == pytest.approx(
        10 / 15, rel=1e-9, abs=0
    )
    with pytest.raises(ValueError, match="code hors de"):
        QUATRE_BITS.niveau(16)


def _codes(amplitude: float) -> list[int]:
    """Le signal `A sin(2 pi 100 t)` echantillonne a 2 kHz pendant 20 ms."""
    phases = (2 * math.pi * 100 * k / 2000 for k in range(40))
    return [QUATRE_BITS.code(amplitude * math.sin(phase)) for phase in phases]


def test_un_signal_qui_couvre_la_pleine_echelle() -> None:
    """Planche 25 : 5 V d'amplitude, tous les codes extremes sont atteints."""
    codes = _codes(5.0)
    assert (min(codes), max(codes)) == (0, 15)
    for rang, code in enumerate(codes):
        exact = 5.0 * math.sin(2 * math.pi * 100 * rang / 2000)
        assert abs(QUATRE_BITS.niveau(code) - exact) <= QUATRE_BITS.pas() / 2 + 1e-12
    assert etat_de_l_echantillonnage(5.0, 5.0) is EchantillonnageDeLEtendue.CORRECT


def test_un_signal_qui_ne_couvre_pas_la_pleine_echelle() -> None:
    """Planche 26 : une faible amplitude n'emploie que quelques codes."""
    codes = _codes(1.0)
    assert set(codes) == {6, 7, 8, 9}
    assert len(set(_codes(5.0))) > 2 * len(set(codes))
    sous = etat_de_l_echantillonnage(1.0, 5.0)
    assert sous is EchantillonnageDeLEtendue.SOUS_EXPLOITEE


def test_un_signal_qui_depasse_la_pleine_echelle_sature() -> None:
    """Planche 27 : « ADCs saturate if their full-scale range is exceeded »."""
    assert QUATRE_BITS.code(7.0) == 15
    assert QUATRE_BITS.code(-7.0) == 0
    assert QUATRE_BITS.sature(7.0)
    assert QUATRE_BITS.sature(-7.0)
    assert not QUATRE_BITS.sature(5.0)
    codes = _codes(7.0)
    assert codes.count(15) > _codes(5.0).count(15)
    assert etat_de_l_echantillonnage(7.0, 5.0) is EchantillonnageDeLEtendue.DEPASSEE


def test_le_temps_d_ouverture_moyenne_le_signal() -> None:
    """Planche 29 : l'echantillon vaut a peu pres la moyenne sur `t_A`."""
    t = sp.Symbol("t", real=True)
    signal = 5 * sp.sin(2 * sp.pi * 100 * t)
    instantane = float(signal.subs(t, 0))
    court = float(echantillon_a_ouverture(signal, t, instant=0, ouverture=1e-6))
    long = float(echantillon_a_ouverture(signal, t, instant=0, ouverture=1e-3))
    assert court == pytest.approx(instantane, abs=2e-3)
    assert long == pytest.approx(
        5 * (1 - math.cos(0.2 * math.pi)) / (0.2 * math.pi), rel=1e-9, abs=0
    )
    assert 0 < long < float(signal.subs(t, 1e-3))
    with pytest.raises(ValueError, match="temps d'ouverture nul ou negatif"):
        echantillon_a_ouverture(signal, t, instant=0, ouverture=0)


def test_le_multiplexeur_mutualise_les_convertisseurs() -> None:
    """Planches 35 et 43 : `m - 1` convertisseurs de moins, a `m f_s` hertz."""
    assert convertisseurs_economises(32) == 31
    assert convertisseurs_economises(2) == 1
    assert cadence_du_multiplexeur(1000.0, 2) == 2000.0
    with pytest.raises(ValueError, match="au moins une voie"):
        convertisseurs_economises(0)


def test_les_voies_multiplexees_sont_echantillonnees_a_des_instants_decales() -> None:
    """Planches 39 a 41 : deux signaux a 200 Hz, chacun regulierement espace."""
    premiere = instants_de_la_voie(0, voies=2, cadence=200.0, nombre=4)
    seconde = instants_de_la_voie(1, voies=2, cadence=200.0, nombre=4)
    assert premiere == pytest.approx((0.0, 0.005, 0.010, 0.015), rel=1e-9, abs=0)
    assert seconde == pytest.approx((0.0025, 0.0075, 0.0125, 0.0175), rel=1e-9, abs=0)
    for instants in (premiere, seconde):
        ecarts = {round(b - a, 12) for a, b in itertools.pairwise(instants)}
        assert ecarts == {0.005}
    with pytest.raises(ValueError, match="voie hors de"):
        instants_de_la_voie(2, voies=2, cadence=200.0, nombre=4)


def test_les_cinq_architectures_types() -> None:
    """Planches 46 a 50 : du tout-dedie au tout-mutualise."""
    un = composants(Architecture.UN, capteurs=8)
    assert (un.faible_bruit, un.filtres, un.usage_general, un.convertisseurs) == (
        8,
        16,
        8,
        8,
    )
    deux = composants(Architecture.DEUX, capteurs=8)
    assert (deux.filtres, deux.usage_general) == (8, 8)
    trois = composants(Architecture.TROIS, capteurs=8)
    assert (trois.faible_bruit, trois.filtres, trois.usage_general) == (8, 8, 0)
    quatre = composants(Architecture.QUATRE, capteurs=64, voies_par_multiplexeur=32)
    assert (quatre.faible_bruit, quatre.multiplexeurs, quatre.convertisseurs) == (
        64,
        2,
        2,
    )
    cinq = composants(Architecture.CINQ, capteurs=64, voies_par_multiplexeur=64)
    assert (cinq.faible_bruit, cinq.usage_general, cinq.filtres) == (0, 1, 64)
    assert (cinq.multiplexeurs, cinq.convertisseurs) == (1, 1)
    assert un.multiplexeurs == 0
    assert paires_multiplexeur_convertisseur(64, 32) == 2
    assert paires_multiplexeur_convertisseur(65, 32) == 3


def test_un_filtre_apres_le_multiplexeur_lisse_les_commutations() -> None:
    """Planches 51 et 52 : le saut d'une voie a l'autre n'a pas fini de s'eteindre."""
    duree = 1 / cadence_du_multiplexeur(200.0, 2)
    par_la_fonction_de_transfert = residu_de_commutation(200.0, duree)
    assert par_la_fonction_de_transfert == pytest.approx(
        math.exp(-math.pi), rel=1e-9, abs=0
    )
    assert par_la_fonction_de_transfert == pytest.approx(0.043, abs=5e-4)
    par_la_legende = residu_de_commutation(100.0, duree)
    assert par_la_legende == pytest.approx(0.208, abs=5e-4)
    s = sp.Symbol("s")
    transfert = 2 * sp.pi * 200 / (s + 2 * sp.pi * 200)
    coupure = sp.solve(sp.Eq(sp.denom(transfert), 0), s)[0]
    assert -coupure / (2 * sp.pi) == 200


def test_les_gardes_de_la_chaine_d_acquisition() -> None:
    """Une reference nulle, zero bit, zero capteur, une cadence nulle sont refuses."""
    with pytest.raises(ValueError, match="tension de reference nulle ou negative"):
        Quantificateur(reference=0.0, bits=4)
    with pytest.raises(ValueError, match="resolution inferieure a un bit"):
        Quantificateur(reference=5.0, bits=0)
    with pytest.raises(ValueError, match="comptes invalides"):
        paires_multiplexeur_convertisseur(0, 32)
    with pytest.raises(ValueError, match="comptes invalides"):
        paires_multiplexeur_convertisseur(8, 0)
    with pytest.raises(ValueError, match="cadence par voie nulle ou negative"):
        instants_de_la_voie(0, voies=2, cadence=0.0, nombre=4)
    with pytest.raises(ValueError, match="invalide"):
        residu_de_commutation(0.0, 1e-3)
    with pytest.raises(ValueError, match="invalide"):
        residu_de_commutation(100.0, -1e-3)
    with pytest.raises(ValueError, match="au moins un capteur"):
        composants(Architecture.UN, capteurs=0)
    with pytest.raises(ValueError, match="cadence d'echantillonnage nulle ou negative"):
        part_de_puissance_repliee(sp.Integer(1), sp.Symbol("f", real=True), 0)
