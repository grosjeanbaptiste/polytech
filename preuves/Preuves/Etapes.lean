import Preuves.Amont

/-!
# Les étapes d'un résultat

Le graphe amont (`Preuves.Amont`) compte tout ce que le noyau a vérifié :
pour le théorème de Pythagore, 16592 constantes, dont des milliers de
lemmes auxiliaires, de récurseurs et d'instances qu'aucun cours ne nomme.

Une étape est une constante qu'un cours pourrait nommer. Ce module en donne
la règle (`visible`) et le graphe contracté (`etapes`) : chaque étape y cite
les étapes qu'elle emploie, directement ou à travers des constantes cachées.
-/

open Lean

namespace Preuves

/-- Les derniers composants d'un nom qui trahissent un lemme d'équation ou un
morceau de preuve engendré par Lean. -/
def suffixesMecaniques : List String :=
  ["eq_def", "injEq", "sizeOf_spec", "inj", "noConfusionType", "below", "brecOn",
    "binductionOn", "casesOn", "recOn", "ndrec", "ndrecOn"]

/-- Les espaces de noms de la machinerie des tactiques : `omega`, `ring`,
`norm_num`… y rangent leurs certificats, qui font un calcul à la place de
l'auteur. Un cours dit « en développant », il ne nomme pas le certificat. -/
def machinerie : List Name :=
  [`Lean, `Std, `Mathlib.Tactic, `Mathlib.Meta, `Aesop, `Qq, `Batteries.Tactic]

/-- Une constante est une étape si un cours pourrait la nommer : ni nom
interne, ni récurseur ou constructeur engendré, ni filtrage compilé, ni
instance, ni lemme d'équation, ni projection de structure (le champ se lit
par la structure), ni machinerie des tactiques. -/
def visible (env : Environment) (c : Name) : Bool :=
  match env.find? c with
  | none => false
  | some info =>
    let dernier := match c with
      | .str _ s => s
      | _ => ""
    !c.isInternal && !isAuxRecursor env c && !isNoConfusion env c
      && !(Meta.isMatcherCore env c) && !(Meta.isInstanceCore env c)
      && !(dernier.startsWith "eq_" && (dernier.drop 3).all Char.isDigit)
      && !(dernier.startsWith "proof_") && !(dernier.startsWith "match_")
      && !(suffixesMecaniques.contains dernier)
      && !(info matches .recInfo _) && !(info matches .ctorInfo _)
      && !(info matches .quotInfo _)
      && (env.getProjectionFnInfo? c).isNone
      && !(machinerie.any (·.isPrefixOf c))

/-- Les étapes visibles qu'atteint `c` en traversant les constantes cachées,
avec mémoire, et une garde contre une boucle (déclarations mutuelles). -/
partial def visiblesSous (env : Environment) (c : Name) :
    StateM (Std.HashMap Name (Array Name)) (Array Name) := do
  if let some r := (← get).get? c then return r
  modify (·.insert c #[])  -- en cours : un retour sur soi n'ajoute rien
  let mut acc : Array Name := #[]
  for d in directes env c do
    if d == c then continue
    if visible env d then
      acc := acc.push d
    else
      acc := acc ++ (← visiblesSous env d)
  let r := acc.toList.eraseDups.toArray
  modify (·.insert c r)
  return r

/-- Le graphe des étapes de `racine` : chaque étape, et les étapes qu'elle
emploie. La racine y est toujours, même cachée. -/
partial def etapes (env : Environment) (racine : Name) : Std.HashMap Name (Array Name) :=
  let rec go (file : List Name) (g : Std.HashMap Name (Array Name))
      (memo : Std.HashMap Name (Array Name)) : Std.HashMap Name (Array Name) :=
    match file with
    | [] => g
    | c :: reste =>
      if g.contains c then go reste g memo else
        let (ups, memo) := (visiblesSous env c).run (memo.erase c)
        go (reste ++ ups.toList) (g.insert c ups) memo
  go [racine] {} {}

/-- `#etapes nom "fichier.json"` : écrit le graphe des étapes de `nom`, et dit
sa taille au regard du graphe amont complet. -/
elab "#etapes " id:ident chemin:str : command => do
  let env ← getEnv
  let nom ← Elab.Command.liftCoreM <| Elab.realizeGlobalConstNoOverloadWithInfo id
  let g := etapes env nom
  IO.FS.writeFile chemin.getString (enJson env g).compress
  logInfo m!"{nom} : {g.size} étapes, sur {(amont env nom).size} constantes en amont"

end Preuves
