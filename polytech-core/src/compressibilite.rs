//! Variation de masse volumique rapportee a la valeur d'arret (isentropique).

use crate::texte::repr;

/// `1 - (1 + (gamma - 1) Ma^2 / 2)^(-1/(gamma - 1))` ; `gamma` vient de l'appelant.
///
/// # Errors
/// Nombre de Mach negatif.
pub fn variation_relative_a_l_arret(mach: f64, gamma: f64) -> Result<f64, String> {
    if mach < 0.0 {
        return Err(format!("nombre de Mach negatif : {}", repr(mach)));
    }
    let base = 1.0 + (gamma - 1.0) * mach.powi(2) / 2.0;
    Ok(1.0 - base.powf(-1.0 / (gamma - 1.0)))
}

#[cfg(test)]
mod tests {
    use super::variation_relative_a_l_arret as v;

    #[test]
    fn nulle_a_l_arret() {
        assert!(v(0.0, 1.4).unwrap().abs() < f64::MIN_POSITIVE);
    }

    #[test]
    fn quatre_virgule_quatre_pour_cent_a_mach_0_3() {
        assert!((v(0.3, 1.4).unwrap() - 0.044).abs() < 5e-4);
    }

    #[test]
    fn croissante_en_mach() {
        assert!(v(0.5, 1.4).unwrap() > v(0.3, 1.4).unwrap());
    }

    #[test]
    fn mach_negatif_refuse() {
        assert_eq!(v(-0.1, 1.4).unwrap_err(), "nombre de Mach negatif : -0.1");
    }
}
