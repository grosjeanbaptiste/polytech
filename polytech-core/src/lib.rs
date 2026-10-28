//! Noyaux numeriques purs de la bibliotheque `polytech`, exposes a Python.
//!
//! Chaque module Rust est pur (flottants en entree et en sortie, erreurs en
//! `String`) ; `liaison` ne fait que les traduire en `ValueError`.

pub mod bandes;
pub mod comparaison;
pub mod compressibilite;
pub mod conduction;
pub mod electronique;
pub mod gravitation;
mod liaison;
pub mod texte;
