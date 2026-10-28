//! Liaison capacitive et tensions de decalage en sortie d'un AOP.

use crate::texte::repr;
use std::f64::consts::PI;

/// `C = 1 / (2 pi R f_c)`.
///
/// # Errors
/// Resistance ou frequence non strictement positive.
pub fn capacite_de_liaison(resistance: f64, frequence_de_coupure: f64) -> Result<f64, String> {
    if resistance <= 0.0 || frequence_de_coupure <= 0.0 {
        return Err(format!(
            "non positive : R = {}, f = {}",
            repr(resistance),
            repr(frequence_de_coupure)
        ));
    }
    Ok(1.0 / (2.0 * PI * resistance * frequence_de_coupure))
}

/// Gain `1 + R_f / R_m` d'un non-inverseur.
fn gain_de_bruit(vers_la_masse: f64, contre_reaction: f64) -> Result<f64, String> {
    if vers_la_masse <= 0.0 || contre_reaction < 0.0 {
        return Err(format!(
            "resistances non positives : {}, {}",
            repr(vers_la_masse),
            repr(contre_reaction)
        ));
    }
    Ok(1.0 + contre_reaction / vers_la_masse)
}

/// Les trois defauts d'un AOP : `e_0`, `I_+`, `I_-`.
#[derive(Clone, Copy)]
pub struct Defauts {
    pub tension_de_decalage: f64,
    pub courant_plus: f64,
    pub courant_moins: f64,
}

/// `G (e_0 - R_s I_+) + R_f I_-`, `G = 1 + R_f / R_m`.
///
/// # Errors
/// Resistances invalides (`R_m <= 0`, `R_f < 0`, `R_s < 0`).
pub fn decalage_du_non_inverseur(
    d: Defauts,
    vers_la_masse: f64,
    contre_reaction: f64,
    source: f64,
) -> Result<f64, String> {
    let gain = gain_de_bruit(vers_la_masse, contre_reaction)?;
    if source < 0.0 {
        return Err(format!("resistance negative : {} ohm", repr(source)));
    }
    Ok(
        d.tension_de_decalage * gain - gain * (d.courant_plus * source)
            + d.courant_moins * contre_reaction,
    )
}

/// `-(R_f / R_e) V_e + (1 + R_f / R_e) e_0 + R_f I_-`.
///
/// # Errors
/// Resistances invalides (`R_e <= 0`, `R_f < 0`).
pub fn decalage_de_l_inverseur(
    entree: f64,
    d: Defauts,
    resistance_entree: f64,
    contre_reaction: f64,
) -> Result<f64, String> {
    let bruit = gain_de_bruit(resistance_entree, contre_reaction)?;
    Ok(-contre_reaction / resistance_entree * entree
        + d.tension_de_decalage * bruit
        + d.courant_moins * contre_reaction)
}

#[cfg(test)]
mod tests {
    use super::{Defauts, capacite_de_liaison, decalage_de_l_inverseur, decalage_du_non_inverseur};

    const D: Defauts = Defauts {
        tension_de_decalage: 1e-3,
        courant_plus: 2e-9,
        courant_moins: 1e-9,
    };

    #[test]
    fn liaison_nominale() {
        let c = capacite_de_liaison(10e3, 20.0).unwrap();
        assert!((c - 1.0 / (2.0 * std::f64::consts::PI * 10e3 * 20.0)).abs() < 1e-18);
    }

    #[test]
    fn liaison_refuse_zero_et_negatif() {
        assert_eq!(
            capacite_de_liaison(0.0, 20.0).unwrap_err(),
            "non positive : R = 0.0, f = 20.0"
        );
        assert!(capacite_de_liaison(1.0, -1.0).is_err());
    }

    #[test]
    fn non_inverseur_somme_des_defauts() {
        let v = decalage_du_non_inverseur(D, 1e3, 9e3, 1e3).unwrap();
        let attendu = 10.0 * (1e-3 - 1e3 * 2e-9) + 9e3 * 1e-9;
        assert!((v - attendu).abs() < 1e-15);
    }

    #[test]
    fn non_inverseur_gardes() {
        let e = decalage_du_non_inverseur(D, 0.0, 9e3, 1e3).unwrap_err();
        assert_eq!(e, "resistances non positives : 0.0, 9000.0");
        assert!(decalage_du_non_inverseur(D, 1e3, -1.0, 1e3).is_err());
        let e = decalage_du_non_inverseur(D, 1e3, 9e3, -5.0).unwrap_err();
        assert_eq!(e, "resistance negative : -5.0 ohm");
    }

    #[test]
    fn inverseur_somme_des_defauts() {
        let v = decalage_de_l_inverseur(0.1, D, 1e3, 10e3).unwrap();
        let attendu = -10.0 * 0.1 + 11.0 * 1e-3 + 10e3 * 1e-9;
        assert!((v - attendu).abs() < 1e-12);
    }

    #[test]
    fn inverseur_gardes() {
        assert!(decalage_de_l_inverseur(0.1, D, 0.0, 1.0).is_err());
        assert!(decalage_de_l_inverseur(0.1, D, 1.0, -1.0).is_err());
    }
}
