# Changelog

<!---
Last updated: 2025-07-25
-->
## [0.4.0] - 2026-02-02

### Summary
#### Added
- `precession` and `sxs` options for determining merger remnant properties
- new star plots and movies tracking stellar objects
- NSC capture of stars
- [Contributing guidelines for PRs](https://mcfacts.readthedocs.io/en/main-dev/commitmsg.html)
- [Sphynx installation instructions](https://mcfacts.readthedocs.io/en/main-dev/MM_documentation.html)
- Immortal stars
- Thermal emission model for jet luminosities observable with LSST.
#### Changed
- Many performance improvements, including external rust library ([mcfast](https://pypi.org/project/mcfast/))
- Improvements to spin and kick modelling
#### Fixed
- dynamics.py star-star and BH-star interactions now conserve angular momentum and energy
- many hill sphere interactions
- Retrograde order of operations, flips, and eccentricity limits

### Individual PRs (summaries) in the order they were merged into main-dev
#### [307: Surrogate Model Implementation](https://github.com/McFACTS/McFACTS/pull/307)
- Implemented way to use SxS surrogate models to predict merger properties
- Added SxS package dependencies to pyproject.toml
- Added `flag_use_surrogate` to IOdocumentation
#### [315: Major stars update: NSC capture, dynamics](https://github.com/McFACTS/McFACTS/pull/315)
-   delta_energy_strong renamed to delta_energy_strong_mu with delta_energy_strong_sigma parameter added to reflect that for the star-star and star-BH interactions this is a Gaussian distribution
-   stars_plots.py updated to include a bunch of new plots
-   star_bh_movie_frames.py updates
-   accretion.accrete_star_mass() includes the orb_ecc in the Hill radius calculation
-   evolve.bin_contact_check() added .value to the contact_condition line to speed up calculations
-   dynamics.py star-star and star-BH interactions now properly conserve angular momentum and energy
-   Changes to agnobject.py
-   Changes to mcfacts_sim.py
 --  output files are appended to at the end of each galaxy, instead written all at once at the end. Columns to write and filenames have been moved to above the galaxy loop.
 --  NSC star capture implemented
 --  Stars and BHs unbound from the disk via dynamical interactions are recorded to an output file
 --  Inner disk stars and BHs are included if snapshots are turned on
 --  changes to prograde/retrograde star treatment and hill sphere interactions
 --  If two stars interact and are within the mutual Hill sphere they merge
 --  Print statements at end of each galaxy updated to be more clear about stars vs BHs
#### [318: Changed variable names, removed usage of unused variable](https://github.com/McFACTS/McFACTS/pull/318)
#### [319: Proposing sweep function and adding tests](https://github.com/McFACTS/McFACTS/pull/319)
- Added proposed sweep variant for the circular_singles_encounters_prograde function, along with testing infrastructure for it.
#### [320: dynamics: added flag_dynamics_sweep](https://github.com/McFACTS/McFACTS/pull/320)
- Changes the default behavior of model_choice_old.ini to use the new sweep implementation created by Nico, and provides a way to turn it on/off (flag_dynamics_sweep).
#### [321: Cube Optimizations](https://github.com/McFACTS/McFACTS/pull/321)
- By changing from using np.poly.root() to find the cube roots of various items, we can instead make use of a closed form analytical solution. This leads to an ~1.6x runtime speed improvement over the original function.
#### [322: bugfix](https://github.com/McFACTS/McFACTS/pull/322)
- Fixing issue with previous PR where the cardano kwarg for circular_singles_encounters_prograde_stars was not properly set in sim.py. Also renaming cardano to fast_cube for informativity.
#### [323: Paper gw231123](https://github.com/McFACTS/McFACTS/pull/323)
- Added new IMF options
- Implemented `precession` alternative for computing merger and remnant quantities
#### [325: Updated documentation and ReadInputs.py ](https://github.com/McFACTS/McFACTS/pull/325)
- Added a description of the precession option to ReadInputs.py and created earlier error handling for precession module.
#### [327: fixing circular_singles_encounters_prograde edge check](https://github.com/McFACTS/McFACTS/pull/327)
- Fix the edge case error which occurs when a particle is exactly on the edge of the disk radius, such that it is not kicked back in yet does not pass the assert test.
#### [331: Updating spin calculations and plots](https://github.com/McFACTS/McFACTS/pull/331)
- Updated spins to the correct T&M '08 calculations/reference frame.
- Created a spin_check.py file to reset the spins to a specific range in order to better match the NRsurrogate.
- Created new plots to show a histogram of the number of spins in the spin_v_kick figure.
#### [336: EM Models for LSST](https://github.com/McFACTS/McFACTS/pull/336)
- Added a script to calculate the EM model of a the thermal emission of a BBH merger in AGN disk with the velocity of forward shock. Also added data files for LSST filters that are being used here
#### [337: Some new print statements and minor tweak to disk mass/gained file writing](https://github.com/McFACTS/McFACTS/pull/337)
#### [330: Contributing Guidelines](https://github.com/McFACTS/McFACTS/pull/330)
- Updated the pull request documentation: https://mcfacts.readthedocs.io/en/main-dev/commitmsg.html
#### [338: Saving immortal star files, update parameter for star coalescence](https://github.com/McFACTS/McFACTS/pull/338)
- Created arrays `stars_all_id_nums`, `stars_masses_initial`, and `stars_orb_a_initial`
- Created new `AGNImmortalStar` object
- Added checks and output files for immortal stars
#### [348: adding documentation.rst](https://github.com/McFACTS/McFACTS/pull/348)
- Adding a rough page with instructions on installing sphinx/compiling documentation pages.
- https://mcfacts.readthedocs.io/en/main-dev/MM_documentation.html
#### [342: Eliminating unnecessary operations from si_from_r_g](https://github.com/McFACTS/McFACTS/pull/342)
-  Rewriting `si_from_r_g` to optionally accept a pre-computed value of `r_g` in meter units, allowing the function to skip the expensive initialization of this value in most cases.
#### [341: Created spin_check function to make the default T&M08 more accurate](https://github.com/McFACTS/McFACTS/pull/341)
- Spin check system for the default model has been updated
- Labels for the compare_plots.py file have been corrected
#### [340: Changes to lum.py](https://github.com/McFACTS/McFACTS/pull/340)
- Removed unused variables in jet_luminosity function of lum.py and made appropriate syntax change to called function in merge.py. Removed si to cgs conversion so si units carry through, without changing the result in the shock_luminosity and jet_luminosity functions.
#### [352: Retrograde Math Fix](https://github.com/McFACTS/McFACTS/pull/352)
- Correcting the order of operations issue in the tau_semi_lat method slows down retrograde evolution.
- Correcting the check for things flipping prograde to only look at the orbital inclination. It was previously setup to let anything with zero eccentricity flip.
- Setting the lower limit of retrograde eccentricities to the critical eccentricity (0.01). Having an eccentricity of exactly 0 causes some issues with the analytical model. Since retrogrades were incorrectly flipped, this was previously not an error we ran into.
#### [353: Update paper citations with ApJ links](https://github.com/McFACTS/McFACTS/pull/353)
- This pull request moves the citations from `README.md` to `docs/source/cite.rst` and adds the new hyperlinks to the fully published ApJ articles.
#### [354: Renamed exploded stars to disrupted](https://github.com/McFACTS/McFACTS/pull/354)
#### [351: Optimization: Accelerating `tau_*_dyn` functions with Rust + Pedantry](https://github.com/McFACTS/McFACTS/pull/351)
- We introduce the first set of Rust accelerants for McFACTS, as part of the [McFAST](https://pypi.org/project/mcfast/) helper module, now on PyPI.
- McFAST surfaces two functions, `tau_ecc_dyn_helper`, `tau_inc_dyn_helper`, which take over the vectorized computations performed by NumPy with optimized Rust functions.
#### [358: Updating star visualizations](https://github.com/McFACTS/McFACTS/pull/358)
- Updated scripts for star visualizations for more relevant plots and animations. These now match what is present in my dissertation and are formatted with the McFACTS plot styling.
#### [356: fixing tau_dyn test file](https://github.com/McFACTS/McFACTS/pull/356)
- Fixed malformed structure of test file that caused pytest to fail when running tau_dyn test.
#### [355: Nojulia](https://github.com/McFACTS/McFACTS/pull/355)
- Created explicit NumPy/Python version requirements for McFACTS
- removed SxS dependencies from pyproject.toml and enforced imports in ReadInputs for `flag_use_surrogate`
#### [359: docs: include instructions for validation steps and optional testing](https://github.com/McFACTS/McFACTS/pull/359)
- Updates the PULL_REQUEST_TEMPLATE.md file to add instructions for running and installing pytest
- Clarify a few items in the checklist


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

