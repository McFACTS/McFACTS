# Changelog

<!---
Last updated: 12/5/2024
-->

# Version 0.2.0  

## Enhancements
  - Refined `dynamics` module, including updates to `circular_binaries_encounters_ecc_prograde` and `circular_binaries_encounters_circ_prograde`.
  - All functions converted to an object-oriented structure. (#247)
  - Added a flag variable to enable the generation of a galaxy runs directory. (#243)
  - Introduced a new method for estimating the eccentricity of each component in an ionized binary.  

## Bug Fixes
  - Fixed bugs in functions related to eccentricity handling. (#256)  
  - Improved checks for cases where `orb_a > disk_radius_outer`.
  - Resolved issues with `excluded_angles` in `bin_spheroid_encounter`. (#251)
  - Fixed bugs in `type1_migration` and `retro_bh_orb_disk_evolve`. (#250)
  - Updated `evolve.bin_reality_check` to handle  now checks more things (such as ionization due to eccentricity > 1), removing the need for a separate reality check.

## Testing and Documentation
  - Added unit tests and integrated `pytest` workflow. (#255, #254)   
  - Added terminal text feedback after the setup command and adjusted Python version requirements. (#249)
  - Updated IO documentation for clarity.  