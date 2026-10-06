r"""Macros des exercices 2 de mathematique, seance II : transformees de Laplace."""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import laplace_exs_02 as la
from polytech.maths.application import problemes_exs_02 as pb
from polytech.maths.application.formatage import formule
from polytech.maths.application.macros_exs_01 import NOMBRES
from polytech.maths.application.macros_reponses_exs_02 import macros_des_reponses
from polytech.maths.domain.laplace.definition import integrale_tronquee
from polytech.maths.domain.laplace.heaviside import fractions_simples
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T

DECOMPOSEES = (4, 5, 6)
"""II.2 : les images dont le corrige montre les fractions simples."""


def terme_a_terme(fonction: sp.Expr) -> tuple[sp.Expr, ...]:
    """L'image de chaque terme de la fonction, sans les reduire au meme denominateur."""
    termes = sp.Add.make_args(sp.expand(fonction))
    return tuple(Transformee.de(terme).image for terme in termes)


def somme(termes: tuple[sp.Expr, ...]) -> str:
    """Des fractions simples, laissees telles quelles : `sympy` les regrouperait."""
    return formule(sp.Add(*termes, evaluate=False))


def _images_et_originaux() -> dict[str, str]:
    macros = {
        f"MathDeuxImage{lettre.upper()}": formule(image)
        for lettre, image in la.images().items()
    }
    macros["MathDeuxImageB"] = somme(terme_a_terme(la.FONCTIONS["b"]))
    tronquee, retardee = la.cube_tronque_et_cube_retarde()
    macros |= {
        "MathDeuxCubeTronque": formule(sp.apart(tronquee, P)),
        "MathDeuxCubeRetarde": formule(retardee),
        "MathDeuxMotifScie": formule(sp.expand(integrale_tronquee(T, sp.Integer(1)))),
        "MathDeuxMotifCreneau": formule(
            integrale_tronquee(sp.Integer(1), sp.Integer(1))
        ),
    }
    for numero, inverser in la.ORIGINAUX.items():
        image, original = inverser()
        nom = NOMBRES[numero - 1]
        macros[f"MathDeuxOriginal{nom}"] = formule(sp.expand(original))
        if numero in DECOMPOSEES:
            macros[f"MathDeuxFractions{nom}"] = somme(fractions_simples(image))
    return macros


def _problemes() -> dict[str, str]:
    image, termes = pb.image_de_la_convolution()
    par_l_image, _ = pb.convolution()
    avant, apres = pb.rampe_ecretee()
    return {
        "MathDeuxFinaleFormule": formule(pb.valeur_par_la_formule()),
        "MathDeuxFinaleOriginale": formule(sp.expand(pb.originale_du_3())),
        "MathDeuxFinaleStable": formule(pb.cas_stable()[1]),
        "MathDeuxConvolutionImage": formule(image),
        "MathDeuxConvolutionFractions": somme(termes),
        "MathDeuxConvolution": formule(par_l_image),
        "MathDeuxCauchyUnImage": formule(sp.factor(pb.probleme_1().image())),
        "MathDeuxCauchyUnFractions": somme(fractions_simples(pb.probleme_1().image())),
        "MathDeuxCauchyUn": formule(sp.expand(pb.probleme_1().solution())),
        "MathDeuxCauchyDeuxImage": formule(sp.factor(pb.probleme_2().image())),
        "MathDeuxCauchyDeuxFractions": somme(
            fractions_simples(pb.probleme_2().image())
        ),
        "MathDeuxCauchyDeux": formule(pb.probleme_2().solution()),
        "MathDeuxRampeAvant": formule(avant),
        "MathDeuxRampeApres": formule(apres),
    }


def macros_de_la_seance_deux() -> dict[str, str]:
    """Les formules et les courbes de la seance II."""
    return {**_images_et_originaux(), **_problemes(), **macros_des_reponses()}
