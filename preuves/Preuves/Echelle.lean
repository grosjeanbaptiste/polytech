import Mathlib
import Preuves.Etapes

/-!
# L'échelle des bases : des entiers naturels au théorème de Pythagore, et au-delà

Consigne du 2026-10-19 : reconstruire les mathématiques « sur base des bases
de géométrie, en démontrant toutes les étapes permettant d'accéder aux
nouvelles formules, de façon récursive ».

Le graphe des étapes (`Preuves.Etapes`) compte des milliers de constantes pour
un seul théorème : c'est la preuve entière, pas un cours. L'échelle en garde
les marches qu'un cours enseigne, chacune portée par des constantes de
Mathlib, et chacune appuyée sur des marches plus basses.

Un appui ne se déclare pas, il se **constate** : `#echelle` vérifie que
l'amont des constantes d'une marche contient au moins une constante de
chaque marche qu'elle dit employer. Une marche qui citerait un appui que sa
preuve n'emploie pas fait échouer la mesure.
-/

open Lean

namespace Preuves

/-- Une marche : son identifiant, les constantes qui la portent, ses appuis. -/
structure Marche where
  id : String
  constantes : List Name
  appuis : List String

/-- L'échelle, du plus bas au plus haut, dans l'ordre que ses appuis
constatés imposent ; distance, Gram-Schmidt, déterminant, produits
vectoriel et mixte, complexes, argument et Moivre sont les notions de
`polytech/` déjà ancrées (`Preuves.Ancrages`). Les textes (énoncé, raison, figure)
sont dans `cursus/echelle.toml` ; les noms ici sont vérifiés par Lean. -/
def echelle : List Marche := [
  ⟨"naturels", [``Nat, ``Nat.add_comm, ``Nat.mul_comm, ``Nat.rec],
    []⟩,
  ⟨"entiers", [``Int, ``Int.add_comm, ``Int.mul_comm, ``Int.add_left_neg],
    ["naturels"]⟩,
  ⟨"rationnels", [``Rat, ``Rat.add_comm, ``Rat.mul_comm],
    ["entiers"]⟩,
  ⟨"ordre", [``mul_self_nonneg, ``sq_nonneg, ``abs_add_le],
    ["naturels"]⟩,
  ⟨"reels", [``Real, ``Real.exists_isLUB],
    ["rationnels", "ordre"]⟩,
  ⟨"racine", [``Real.sqrt, ``Real.sq_sqrt, ``Real.sqrt_sq],
    ["reels"]⟩,
  ⟨"vecteurs", [``EuclideanSpace, ``smul_add, ``add_smul],
    ["reels"]⟩,
  ⟨"produit-scalaire", [``inner, ``real_inner_comm, ``real_inner_self_eq_norm_sq],
    ["racine"]⟩,
  ⟨"norme", [``norm_add_le, ``norm_smul],
    ["reels"]⟩,
  ⟨"cauchy-schwarz", [``real_inner_le_norm, ``abs_real_inner_le_norm],
    ["produit-scalaire"]⟩,
  ⟨"pythagore-vectoriel", [``norm_add_sq_eq_norm_sq_add_norm_sq_iff_real_inner_eq_zero,
    ``norm_sub_sq_eq_norm_sq_add_norm_sq_iff_real_inner_eq_zero],
    ["produit-scalaire"]⟩,
  ⟨"gram-schmidt", [``InnerProductSpace.gramSchmidt, ``InnerProductSpace.gramSchmidt_orthogonal],
    ["vecteurs", "produit-scalaire", "norme"]⟩,
  ⟨"determinant", [``Matrix.det, ``Matrix.det_mul],
    ["vecteurs"]⟩,
  ⟨"produit-vectoriel", [``crossProduct, ``cross_anticomm, ``dot_self_cross],
    ["vecteurs"]⟩,
  ⟨"produit-mixte", [``triple_product_eq_det],
    ["determinant", "produit-vectoriel"]⟩,
  ⟨"complexes", [``Complex, ``Complex.conj_re, ``Complex.conj_im],
    ["reels"]⟩,
  ⟨"trigonometrie", [``Real.cos, ``Real.sin, ``Real.sin_sq_add_cos_sq, ``Real.pi],
    ["racine", "vecteurs", "norme", "complexes"]⟩,
  ⟨"derivee", [``deriv, ``HasDerivAt, ``Real.hasDerivAt_sin],
    ["trigonometrie"]⟩,
  ⟨"angle", [``InnerProductGeometry.angle, ``InnerProductGeometry.cos_angle],
    ["cauchy-schwarz", "trigonometrie"]⟩,
  ⟨"pythagore", [``EuclideanGeometry.dist_sq_eq_dist_sq_add_dist_sq_iff_angle_eq_pi_div_two],
    ["pythagore-vectoriel", "angle"]⟩,
  ⟨"loi-des-cosinus", [``EuclideanGeometry.law_cos],
    ["angle"]⟩,
  ⟨"argument", [``Complex.arg],
    ["trigonometrie"]⟩,
  ⟨"integrale", [``intervalIntegral, ``intervalIntegral.integral_eq_sub_of_hasDerivAt],
    ["produit-scalaire", "derivee", "argument"]⟩,
  ⟨"distance", [``EuclideanSpace.dist_eq],
    ["argument"]⟩,
  ⟨"moivre", [``Complex.cos_add_sin_mul_I_pow],
    ["racine", "norme", "complexes"]⟩,
  ⟨"limites", [``Filter.Tendsto, ``Metric.tendsto_atTop, ``tendsto_nhds_unique],
    ["reels"]⟩,
  ⟨"exponentielle", [``Real.exp, ``Real.exp_add, ``Real.hasDerivAt_exp],
    ["derivee", "complexes"]⟩,
  ⟨"polynomes-et-racines", [``Polynomial, ``Polynomial.IsRoot,
    ``Polynomial.mul_divByMonic_eq_iff_isRoot, ``Polynomial.card_roots', ``Complex.isAlgClosed],
    ["reels", "complexes"]⟩,
  ⟨"equations-differentielles", [``ODE_solution_unique, ``HasDerivAt],
    ["derivee", "limites"]⟩,
  ⟨"logarithme", [``Real.log, ``Real.exp_log, ``Real.log_mul],
    ["exponentielle"]⟩,
  ⟨"systemes-lineaires-et-rang", [``Matrix.rank, ``Matrix.rank_mul_le,
    ``Matrix.mulVec_injective_iff_isUnit, ``Matrix.det_ne_zero_of_left_inverse],
    ["determinant", "vecteurs"]⟩,
  ⟨"transformee-de-laplace", [``integral_exp_mul_complex_Ioi, ``integral_exp_neg_Ioi,
    ``MeasureTheory.integral_Ioi_of_hasDerivAt_of_tendsto],
    ["integrale", "exponentielle", "complexes"]⟩
]

/-- Les constantes de l'amont réuni d'une liste de constantes. -/
def amontReuni (env : Environment) (noms : List Name) : Std.HashSet Name :=
  noms.foldl (fun acc n => (amont env n).fold (fun a k _ => a.insert k) acc) {}

/-- Pour chaque appui déclaré, l'une de ses constantes est-elle dans l'amont ? -/
def appuisConstates (env : Environment) (m : Marche) : List (String × Bool) :=
  let am := amontReuni env m.constantes
  m.appuis.map fun a =>
    match echelle.find? (·.id == a) with
    | none => (a, false)
    | some base => (a, base.constantes.any (am.contains ·))

/-- `#echelle "echelle.json"` : chaque marche, ses constantes, ses appuis tels
que constatés, la taille de son amont et les axiomes atteints. Une marche
inconnue ou un appui non constaté est une erreur. -/
elab "#echelle " chemin:str : command => do
  let env ← getEnv
  let mut lignes : Array Json := #[]
  let mut fautes : Array String := #[]
  for m in echelle do
    let constates := appuisConstates env m
    for (a, ok) in constates do
      unless ok do fautes := fautes.push s!"{m.id} : appui « {a} » non constaté"
    let am := amontReuni env m.constantes
    let axiomes := am.toList.filter (nature env · == "axiome") |>.map (·.toString)
    -- Toutes les marches dont une constante est dans l'amont : ce que la preuve
    -- emploie réellement, d'où se choisissent les appuis déclarés.
    let constates := echelle.filter (fun b => b.id != m.id && b.constantes.any (am.contains ·))
      |>.map (·.id)
    lignes := lignes.push <| Json.mkObj [
      ("id", Json.str m.id),
      ("constantes", Json.arr (m.constantes.toArray.map (Json.str ·.toString))),
      ("appuis", Json.arr (m.appuis.toArray.map Json.str)),
      ("constates", Json.arr (constates.toArray.map Json.str)),
      ("amont", Json.num am.size),
      ("axiomes", Json.arr ((axiomes.toArray.qsort (· < ·)).map Json.str))]
  IO.FS.writeFile chemin.getString (Json.arr lignes).pretty
  if fautes.isEmpty then
    logInfo m!"{echelle.length} marches, chaque appui constaté"
  else
    throwError m!"{String.intercalate "\n" fautes.toList}"

end Preuves
