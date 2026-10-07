"""Macros de « la modelisation », problemes II, III, VI, VII et VIII."""

from __future__ import annotations

from polytech.optimisation.application import modele_decoupe as de
from polytech.optimisation.application import modele_decoupe_bornes as bo
from polytech.optimisation.application import modele_depistage as dp
from polytech.optimisation.application import modele_gare as ga
from polytech.optimisation.application import modele_gare_categories as ca
from polytech.optimisation.application import modele_maintenance as ma
from polytech.optimisation.application import modele_maintenance_conditions as co
from polytech.optimisation.application import modele_meubles as me
from polytech.optimisation.application import modele_meubles_plan as pl
from polytech.optimisation.application.formatage import arrondi, exact, point

_LETTRES = "ABCD"


def _decoupe() -> dict[str, str]:
    return {
        "ModeleDecoupeMotifs": str(len(de.MOTIFS)),
        "ModeleDecoupeLargeur": arrondi(bo.largeur_commandee(), 1),
        "ModeleDecoupeRouleaux": str(bo.rouleaux_au_moins()),
        "ModeleDecoupePerte": arrondi(bo.perte_au_moins(), 1),
        "ModeleDecoupeDecoupes": str(bo.decoupes_au_moins()),
        "ModeleDecoupeRelaxRouleaux": exact(de.relaxation("rouleaux")),
        "ModeleDecoupeRelaxDecoupes": exact(de.relaxation("decoupes")),
        "ModeleDecoupeGrandes": str(bo.grandes_pieces()),
        "ModeleDecoupePieces": str(sum(de.COMMANDES)),
        "ModeleDecoupeSansChute": str(bo.rouleaux_sans_chute_au_plus()),
    }


def _gare() -> dict[str, str]:
    cout, equipes = ga.optimum(ga.BESOINS, "salaire")
    _, deux, trois = ga.cout_par_definition()
    _, delta, gamma = ga.choix_du_modele()
    categories, plan = ca.optimum()
    separes = ",\\ ".join(exact(v) for v in plan[: ga.PERIODES])
    coupes = ",\\ ".join(exact(v) for v in plan[ga.PERIODES :])
    return {
        "ModeleGareEffectif": exact(ga.optimum(ga.BESOINS, "effectif")[0]),
        "ModeleGareImpaires": str(ga.besoin_des_periodes(ga.BESOINS, (1, 3, 5))),
        "ModeleGarePaires": str(ga.besoin_des_periodes(ga.BESOINS, (2, 4, 6))),
        "ModeleGareCout": exact(cout),
        "ModeleGareEquipes": point(equipes),
        "ModeleGareCoutDeux": exact(deux),
        "ModeleGareCoutTrois": exact(trois),
        "ModeleGareDelta": exact(delta),
        "ModeleGareGamma": exact(gamma),
        "ModeleGareContreExemple": arrondi(ca.cout(ca.CONTRE_EXEMPLE), 1),
        "ModeleGareVide": exact(ca.presents(ca.CONTRE_EXEMPLE)[0]),
        "ModeleGareCategories": arrondi(categories, 1),
        "ModeleGarePlan": f"({separes} \\mid {coupes})",
    }


def _meubles() -> dict[str, str]:
    imprime = tuple(me.STOCKS_IMPRIMES.items())
    selon_l_enonce = pl.plan(tuple(me.STOCKS_DE_L_ENONCE.items()))
    plafonds = [exact(pl.plafond_hebdomadaire(m)) for m in me.MEUBLES]
    return {
        "ModeleMeublesGainImprime": str(pl.plan(imprime).gain),
        "ModeleMeublesGainEnonce": str(selon_l_enonce.gain),
        "ModeleMeublesDeLuxe": point(selon_l_enonce.production["L"]),
        "ModeleMeublesStandard": point(selon_l_enonce.production["S"]),
        "ModeleMeublesPlafonds": " \\text{ ou } ".join(plafonds),
        "ModeleMeublesGainExtension": exact(
            pl.plan(imprime, 4).gain - pl.plan(imprime).gain
        ),
    }


def _maintenance() -> dict[str, str]:
    cout, debuts = ma.meilleur_planning()
    imprime, _, _ = ma.classement(imprimes=True)[0]
    pointe, plannings = ma.plus_petite_pointe()
    macros = {
        "ModeleMaintenancePlannings": str(len(ma.plannings())),
        "ModeleMaintenanceCout": str(cout),
        "ModeleMaintenanceDebuts": point(debuts),
        "ModeleMaintenanceBesoins": point(ma.besoins(debuts)),
        "ModeleMaintenanceImprime": str(imprime),
        "ModeleMaintenanceBesoinsImprimes": point(ma.besoins(debuts, imprimes=True)),
        "ModeleMaintenancePointe": str(pointe),
        "ModeleMaintenancePlanningsPointe": str(len(plannings)),
    }
    for lettre, condition in zip(_LETTRES, co.CONDITIONS, strict=True):
        ligne = " & ".join(str(v) for v in co.bilan(condition))
        macros[f"ModeleMaintenanceLigne{lettre}"] = ligne
    return macros


def _depistage() -> dict[str, str]:
    suffisants = dp.choix_suffisants()
    return {
        "ModeleDepistageCout": str(suffisants[0][0]),
        "ModeleDepistageChoix": point(suffisants[0][1]),
        "ModeleDepistageSuffisants": str(len(suffisants)),
        "ModeleDepistageSecond": str(suffisants[1][0]),
    }


def macros_des_ateliers() -> dict[str, str]:
    """Les valeurs des problemes II, III, VI, VII et VIII."""
    return {**_decoupe(), **_gare(), **_meubles(), **_maintenance(), **_depistage()}
