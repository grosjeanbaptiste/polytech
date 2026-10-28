//! `F = G m1 m2 / r^2` ; `G` est passee par l'appelant (elle vit dans Python).

use crate::texte::repr;

/// # Errors
/// Masse negative ou distance non strictement positive.
pub fn force_gravitationnelle(
    constante: f64,
    masse_1: f64,
    masse_2: f64,
    distance: f64,
) -> Result<f64, String> {
    if masse_1 < 0.0 || masse_2 < 0.0 {
        return Err(format!(
            "masse negative : {}, {}",
            repr(masse_1),
            repr(masse_2)
        ));
    }
    if distance <= 0.0 {
        return Err(format!(
            "distance non strictement positive : {}",
            repr(distance)
        ));
    }
    Ok(constante * masse_1 * masse_2 / distance.powi(2))
}

#[cfg(test)]
mod tests {
    use super::force_gravitationnelle as f;

    #[test]
    fn nominal() {
        let v = f(2.0, 3.0, 5.0, 2.0).unwrap();
        assert!((v - 7.5).abs() < 1e-15);
    }

    #[test]
    fn masse_nulle_donne_zero() {
        assert!(f(2.0, 0.0, 5.0, 1.0).unwrap().abs() < f64::MIN_POSITIVE);
    }

    #[test]
    fn masse_negative_refusee() {
        let e = f(1.0, -1.0, 5.0, 1.0).unwrap_err();
        assert_eq!(e, "masse negative : -1.0, 5.0");
        assert!(f(1.0, 1.0, -5.0, 1.0).is_err());
    }

    #[test]
    fn distance_nulle_ou_negative_refusee() {
        let e = f(1.0, 1.0, 1.0, 0.0).unwrap_err();
        assert_eq!(e, "distance non strictement positive : 0.0");
        assert!(f(1.0, 1.0, 1.0, -1.0).is_err());
    }
}
