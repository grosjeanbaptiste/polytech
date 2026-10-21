import Preuves.Etapes

/-!
# Les étapes : le graphe amont, sans la mécanique de Lean

Un petit monde fait à la main, sans Mathlib : une définition, deux
théorèmes dont l'un passe par un lemme auxiliaire, une instance. Chaque
`#guard` échoue à la compilation si la règle qu'il nomme est rompue.
-/

namespace Essai

def double (n : Nat) : Nat := n + n

theorem double_zero : double 0 = 0 := rfl

/-- Un lemme de passage, au nom interne : il ne doit pas paraître. -/
theorem _passage (n : Nat) : double n = n + n := rfl

theorem double_un : double 1 = 2 := by rw [_passage]

class Pointe (α : Type) where pt : α

instance : Pointe Nat := ⟨0⟩

def origine : Nat := Pointe.pt

structure Paire where
  gauche : Nat
  droite : Nat

def somme (p : Paire) : Nat := p.gauche + p.droite

end Essai

/-- Un résultat rangé sous `Lean` : la machinerie des tactiques, qu'un cours
ne nomme pas (certificats de `omega`, de `ring`…). -/
theorem Lean.EssaiMachinerie.certificat : True := trivial

theorem Essai.vrai : True := Lean.EssaiMachinerie.certificat

open Lean Preuves in
run_meta do
  let env ← getEnv
  -- La mécanique se cache : nom interne, instance, récurseur.
  guard <| !(visible env `Essai._passage)
  guard <| !(visible env `Essai.instPointeNat)
  guard <| !(visible env `Nat.rec)
  -- Ce qu'un cours nommerait reste.
  guard <| visible env `Essai.double
  guard <| visible env `Essai.double_un
  -- Une étape cite ce qu'elle emploie, y compris à travers un passage caché.
  let g := etapes env `Essai.double_un
  guard <| (g.get? `Essai.double_un).any (·.contains `Essai.double)
  guard <| !(g.contains `Essai._passage)
  -- À travers une instance cachée, l'étape voit ce que l'instance emploie.
  let o := etapes env `Essai.origine
  guard <| !(o.contains `Essai.instPointeNat)
  guard <| (o.get? `Essai.origine).any (·.contains `Essai.Pointe)
  -- Une projection de structure se cache ; la structure, elle, reste.
  guard <| !(visible env `Essai.Paire.gauche)
  guard <| visible env `Essai.Paire
  guard <| (( etapes env `Essai.somme).get? `Essai.somme).any (·.contains `Essai.Paire)
  -- La machinerie des tactiques se cache.
  guard <| !(visible env `Lean.EssaiMachinerie.certificat)
  guard <| !((etapes env `Essai.vrai).contains `Lean.EssaiMachinerie.certificat)
  -- Chaque étape citée est elle-même une entrée du graphe : la récursion est close.
  for (_, ups) in g.toList do
    for u in ups do guard (g.contains u)
