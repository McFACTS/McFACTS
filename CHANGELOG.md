# Changelog

<!---
Last updated: 2025-07-25
-->

## [0.3.0] - 2025-06-16
### Added

  - Output columns moved for easier importing. [#305](https://github.com/McFACTS/McFACTS/pull/305)
  - Data files for new disk parameters. [#272](https://github.com/McFACTS/McFACTS/pull/272)
  - Initial code for shock and jet luminosity. [#270](https://github.com/McFACTS/McFACTS/pull/270)
  - Option for Gaussian sampling of ΔE in strong 2+1 interactions. [#269](https://github.com/McFACTS/McFACTS/pull/269)
  - Log file reading capability. [#278](https://github.com/McFACTS/McFACTS/pull/278)
  - Torque prescription updates and `phenom_turb`. [#277](https://github.com/McFACTS/McFACTS/pull/277)
  - Big stars module. [#271](https://github.com/McFACTS/McFACTS/pull/271)
  - Boolean switch to select velocity models. [#276](https://github.com/McFACTS/McFACTS/pull/276)


### Changed

  - BBH functions updated to use array inputs/outputs. [#293](https://github.com/McFACTS/McFACTS/pull/293)
  - Torque interpolators moved into their own function. [#289](https://github.com/McFACTS/McFACTS/pull/289)
  - Populated disks with BHs using power law probability. [#274](https://github.com/McFACTS/McFACTS/pull/274)
  - Dynamics updated; default run set to 0.5 Myr. [#262](https://github.com/McFACTS/McFACTS/pull/262)
  - Renamed `emily_plots` to `em_plots`. [#294](https://github.com/McFACTS/McFACTS/pull/294)


### Fixed

  - `orb_a` values not updating inside if-statement. [#310](https://github.com/McFACTS/McFACTS/pull/310)
  - Bug causing repeated ID numbers. [#301](https://github.com/McFACTS/McFACTS/pull/301)
  - Redundant slow `np.argsort` call removed; random numbers generated once. [#300](https://github.com/McFACTS/McFACTS/pull/300)
  - `assert` bug in `bin_sep`. [#299](https://github.com/McFACTS/McFACTS/pull/299)
  - `argparse` bug. [#297](https://github.com/McFACTS/McFACTS/pull/297)
  - Galaxy number restored to 1. [#295](https://github.com/McFACTS/McFACTS/pull/295)
  - `test_ReadInputs` fixed (correct branch). [#303](https://github.com/McFACTS/McFACTS/pull/303)
  - Velocity math, kick velocity histogram, and pressure gradient interpolator bugs. [#294](https://github.com/McFACTS/McFACTS/pull/294)
  - Vectorized `disk_capture.retro_bh_orb_disk_evolve` and fixed related bug. [#298](https://github.com/McFACTS/McFACTS/pull/298)
  - Bug initializing BHs and stars with `galaxy = 0`. [#275](https://github.com/McFACTS/McFACTS/pull/275)
  - Disk mass gain bug. [#273](https://github.com/McFACTS/McFACTS/pull/273)
  - Torque calculation and `CubicSpline` issues. [#282](https://github.com/McFACTS/McFACTS/pull/282), [#279](https://github.com/McFACTS/McFACTS/pull/279)
  - Behavior in `ReadLog` function. [#287](https://github.com/McFACTS/McFACTS/pull/287)
  - Initial star number code. [#281](https://github.com/McFACTS/McFACTS/pull/281)
  - Minor point mass bug. [#285](https://github.com/McFACTS/McFACTS/pull/285)
  - Migration outer edge catches. [#283](https://github.com/McFACTS/McFACTS/pull/283)
  - Fix emri and lvk gw_freq evolution and plotting [[686d32d](https://github.com/McFACTS/McFACTS/commit/686d32dd1898018ad8af2504efe46d108fd5d868)]


### Removed

  - Unused luminosity-related module `mock_phot.py`. [#294](https://github.com/McFACTS/McFACTS/pull/294)
  - Verbose migration time printouts. [#267](https://github.com/McFACTS/McFACTS/pull/267)
  - Extra unused files. [#268](https://github.com/McFACTS/McFACTS/pull/268)


### Other / Misc / Merged

  - Merged BH fix and assert statements. [#286](https://github.com/McFACTS/McFACTS/pull/286)
  - Main development branches merged: [#292](https://github.com/McFACTS/McFACTS/pull/292), [#290](https://github.com/McFACTS/McFACTS/pull/290), [#284](https://github.com/McFACTS/McFACTS/pull/284), [#265](https://github.com/McFACTS/McFACTS/pull/265)
  - Paper 3 work and other merges. [#288](https://github.com/McFACTS/McFACTS/pull/288)
  - Main-dev synced with main. [#263](https://github.com/McFACTS/McFACTS/pull/263)
  - General update PRs. [#261](https://github.com/McFACTS/McFACTS/pull/261)
  - Barry's "nuclear" merge of migration changes [#266](https://github.com/McFACTS/McFACTS/pull/266)



## Version 0.2.0  

### Enhancements
  - Refined `dynamics` module, including updates to `circular_binaries_encounters_ecc_prograde` and `circular_binaries_encounters_circ_prograde`.
  - All functions converted to an object-oriented structure. (#247)
  - Added a flag variable to enable the generation of a galaxy runs directory. (#243)
  - Introduced a new method for estimating the eccentricity of each component in an ionized binary.  

### Bug Fixes
  - Fixed bugs in functions related to eccentricity handling. (#256)  
  - Improved checks for cases where `orb_a > disk_radius_outer`.
  - Resolved issues with `excluded_angles` in `bin_spheroid_encounter`. (#251)
  - Fixed bugs in `type1_migration` and `retro_bh_orb_disk_evolve`. (#250)
  - Updated `evolve.bin_reality_check` to handle  now checks more things (such as ionization due to eccentricity > 1), removing the need for a separate reality check.

### Testing and Documentation
  - Added unit tests and integrated `pytest` workflow. (#255, #254)   
  - Added terminal text feedback after the setup command and adjusted Python version requirements. (#249)
  - Updated IO documentation for clarity.  