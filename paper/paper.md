---
title: 'McFACTS: Monte carlo For AGN Channel Testing and Simulation'
tags:
  - Python
  - astronomy
  - active galactic nuclei
  - population synth
authors:
  - name: Jake Postiglione
    orcid: 0000-0003-0738-8186
    corresponding: true
    affiliation: "1, 2"
  - name: Miranda McCarthy
    orcid: 0009-0005-9964-4790
    affiliation: "1, 2"
  - name: Nicolas Posner
    orcid: 0009-0004-4600-5074
    affiliation: "9"
  - name: K.E. Saavik Ford
    orcid: 0000-0002-5956-851X
    affiliation: "1, 2, 3, 7"
  - name: Barry McKernan
    orcid: 0000-0002-9726-0508
    affiliation: "1, 2, 3, 7"
  - name: Harrison E. Cook
    orcid: 0000-0001-7163-8712
    affiliation: "4"
  - name: Vera Delfavero
    orcid: 0000-0001-7099-765X
    affiliation: "8"
  - name: Emily McPike
    orcid: 0009-0008-5622-6857
    affiliation: "1, 2"
  - name: Kaila Nathaniel
    orcid: 0000-0003-2430-9515
    affiliation: "5"
  - name: Rosalba Perna
    orcid: 0000-0002-3635-5677
    affiliation: "6"
  - name: Varun Pritmani
    orcid: 0009-0000-3666-0586
    affiliation: "2"
  - name: Shawn Ray
    orcid: 0009-0005-5038-3171
    affiliation: "1, 2"
  - name: Richard O'Shaughnessy
    orcid: 0000-0001-5832-8517
    affiliation: "5"
affiliations:
  - name: Graduate Center, City University of New York, 365 5th Avenue, New York, NY 10016, USA
    index: 1
    ror: "00awd9g61"
  - name: Department of Astrophysics, American Museum of Natural History, New York, NY 10024, USA
    index: 2
    ror: "03thb3e06"
  - name: Department of Science, BMCC, City University of New York, New York, NY 10007, USA
    index: 3
    ror: "040hwr020"
  - name: New Mexico State University, Department of Astronomy, PO Box 30001 MSC 4500, Las Cruces, NM 88003, USA
    index: 4
    ror: "01kbvt179"
  - name: Center for Computational Relativity and Gravitation, Rochester Institute of Technology, Rochester, New York 14623, USA
    index: 5
  - name: Department of Physics and Astronomy, Stony Brook University, Stony Brook, NY, 11794, USA
    index: 6
    ror: "05qghxh33"
  - name: Center for Computational Astrophysics, Flatiron Institute, 162 5th Ave, New York, NY 10010, USA
    index: 7
    ror: "00sekdz59"
  - name: Canadian Institute for Theoretical Astrophysics, University of Toronto, 60 St George St, Toronto, ON M5S 3H8, Canada
    index: 8
    ror: "03dbr7087"
  - name: Data Science Institute, University of Chicago, 5801 S Ellis Ave, Chicago, IL 60637, USA
    index: 9
    ror: "024mw5h28"
date: 01 July 2026
bibliography: paper.bib
---

# Summary

Active Galactic Nuclei (AGN) are gas-rich and dynamically active structures at
the center of some galaxies. At the core of an AGN exists a Supermassive Black
Hole (SMBH), creating an immense gravitational well hosting a diverse populations 
of black holes and stars. A portion of the gravitational waves events detected 
by the LIGO-Virgo-KAGRA (LVK) collaboration are expected to originate
from the merging of binary black holes (BBHs) embedded in the accretion disk of 
AGN. This "AGN channel" is also expected to host binaries that will be 
detectable by the Laser Interferometer Space Antenna (LISA). Several features of 
this population, which appear in current observables, strongly depend on the 
properties of the accretion disk, and nuclear star cluster (NSC). Some of these 
properties can be difficult to directly measure; however, through population 
synthesis we can place constraints on some of these features, expanding out 
understanding of the AGN channel.

`McFACTS` (Monte carlo For AGN Channel Testing and Simulation) is the leading
public and open-source population synthesis code that models the AGN channel for 
LVK-detectable BBH mergers [@mckernan:2025]. Given a model of an AGN disk and 
distributions of object properties in the NSC, `McFACTS` seeds a population of 
single black holes and stars which are then allowed to evolve through a
variety of physical effects and form binary objects which in-turn can ionize.
The gas in the disk can drive accretion, migration, and eccentricity dampening.
Different populations within the disk (e.g. circular vs eccentric orbiters), can
have dynamical encounters and exchange energy. Stellar binaries, BBH, and 
Extreme Mass Ratio Inspirals (EMRIs) are also allowed to evolve through
gravitational wave decay. `McFACTS` makes use of several long-standing
packages making up the scientific python ecosystem, such as `Astropy`
[@astropy:2022], `NumPy` [@harris:2020],  and `SciPy` [@scipy:2020]. The AGN
disk structure and properties are obtained through the `pAGN` [@gangardt:2024]
package, supporting both the Sirko and Goodman [@sg:2003] and Thompson,
Quataert and Murry [@tqm:2005] disk models.

This paper serves to document a substantial re-structure of `McFACTS`,
relative to the version first presented in @mckernan:2025. The code has been
significantly reorganized around a modular simulation framework, allowing for 
easy and intuitive composition of simulation timelines. Several key methods have
also been vectorized, and the most computationally demanding algorithms have
been offloaded to a companion library called `McFAST`, built in Rust. On the
astrophysics side of things, the simulation has been expanded to include a 
full stellar population [@nathenial:2026, in prep], 
electromagnetic-counterpart models [@mcpike:2026; @mcpike:2026b, in prep],
numerical-relativity surrogate models for merger remnants [@ray:2026, in prep],
and semi-analytical gas evolution models for binary objects 
[@postiglione:2026, in prep].

# Statement of need

Of the proposed formation channels for compact-object mergers, the AGN channel
remains poorly constrained. A population synthesis code for this channel must
therefore make it possible to easily vary physical assumptions and 
prescriptions, while remaining fast and performant. The original release of
`McFACTS` demonstrated the scientific value of an open-source AGN channel code,
but its simulation logic and bookkeeping was concentrated into one main script.
This made it difficult for new contributions to be added, or for different
combinations of physical processes to be tested. As the `McFACTS` collaboration 
and its scientific goals grew, these limitations became the main obstacle in
maintaining and upgrading the code.

The restructured version of `McFACTS`, documented in this paper, address these
needs directly. Physical process have been separated into modular, and 
independent reusable units. Population objects are now stored and managed in a 
centralized location with automatic consistency checking against the entire 
datastore. The input/output structure has been revamped, using interfaces to
allow for the handling of different file types and interoperability with other
pipelines and codes. The result of these changes is a code that is easier to
extend, faster to run, and more easily adoptable by students and researchers
looking to make contributions to the functionality of `McFACTS` or integrate 
it into their own work.

# State of the field            

Several population synthesis codes targeting compact object formation and
evolution currently exist, include `COSMIC` [@breivik:2020], 
`COMPAS` [@riley:2022], `CMC` [@rodriguez:2022], and `cogsworth` [@wagg:2024].
These codes focus on formation channels for isolated binaries and dense
stellar clusters, but do not model the gas-driven dynamics of an AGN disk.
Codes that focused on the AGN channel have historically been geared towards
specific features, such as focusing on the mergers found at migration traps, or 
are not open source and have not been released publicly. `McFACST` distinguishes
itself from its predecessors by being public, open source, and tooled to
simulate the full AGN channel across a broad range of the AGN parameter space.

# Software design

The restructure of `McFACTS` separates the framework from any particular
simulation or model assembled out of its building blocks. This new framework
is best understood through a top-down lens, where each level orchestrates the
level beneath it. A `Galaxy` holds its populations in a `FilingCabinet` of 
`AGNObjectArray`s. The `AGNOBjectArray`s are evolved according to a
`SimulationTimeline`, and timeline is composed of individual `TimelineActor`s.
The population is seeded with `Populator` objects.

## Galaxy

The `Galaxy` object sits at the core of the simulation framework. A single
`Galaxy` object represents one synthetic galaxy, with its own population,
deterministic random state using `Philox`, and the history of simulation
timelines that have been run. A `Galaxy` is first populated using `Populator`
objects. Once the `Galaxy` is seeded, a `SimulationTimeline` holding several
`TimelineActor` objects is run.

## Filing Cabinet and AGN Object Array

A `Galaxy` stores its population in a `FilingCabinet`, a typed container that
organizes different populations with consistency checks to ensure that an
object does not appear in two categories at once. Each entry in the cabinet
takes the form of an `AGNObjectArray`, which stores every property (orbital
parameters, mass, spin and spin angle, etc.) as parallel `NumPy` `arrays. Every 
object carries a unique identifier and the ids of its parents, should they 
exist. Subclasses for different population categories, such as single black 
holes, binary black holes, merged black holes, and so forth, allows for quick
object-type validation when needed, and for the inheritance of shared parameters.

## Simulation Timeline, Timeline Actors, and Populators

Evolution of `Galaxy` populations is scheduled through a `SimulationTimeline` 
object. The timeline holds an ordered list of different operations that are
applied to the population over some fixed number of timesteps. A single `Galaxy`
may run several timelines sequentially, for example a brief timeline that
classifies objects at the start of a run, followed by the main physics timeline
evolving the population in the disk. The `TimelineActor` object encapsulates the
implementation of physical processes through a common `perform` interface, with 
each actor generally representing a single physica process (migration, 
accretion, eccentricity damping, binary formation, merger processing, etc.).
Each actor is passed a reference to the `FilingCabinet` of that galaxy, and can
read and write directly to it. A `Populator` is structured similarly to a
`TimelineActor`, but distinctly plays the role of seeding the initial 
populations, constructing the starting `AGNObjectArray` objects passed to the
filling cabinet. `Populators` get run by the galaxy before any
`SimulationTimeline` objects execute `TimelineActor`s.

## Crosscutting Utilities

Several utilities are used across the entire hierarchy of the simulation 
framework. The `SettingsManager` acts as a central repo for built-in defaults
and user overridable settings. Internal constants are protected from
modification by the user, and custom settings can be passed for use with
third-party modules. The `SnapshotHandler` provide an interface for the 
simulation framework to load and save the `SettingsManager` and `FilingCabinet`
objects. By default, we support both `.txt` and `.ini` for the loading and
saving of the `SettingsManger` and support `.txt` files for the `FilingCabinet`.
Plans exist to implement the `.h5` format for HDF5 support and `.db, .sqlite`
with SQLite support.

## Extension: McFAST

`mcfast` is a custom-built extension to the `McFACTS` codebase written in Rust,
with Python bindings produced by `pyo3`. It provides optimized variants of
several `McFACTS` functions, including vectorized, single-pass variants of
computation-heavy functions (e.g. `shock_luminosity`,
`analytical_kick_velocity`) and tailored variants of unit conversion functions
(e.g. `si_from_r_g`, `r_g_from_units`) which bypass AstroPy's allocation- and
string-heavy unit conversion operations.

### Motivation

As of the prior version v0.3.0, `McFACTS`'s most time-intensive operations fell
into three categories:
 1. Long chains of `numpy` array operations on many arrays of equal length.
 2. Inefficient general-purpose code in Python (eigenvalue rootfinding vs
 Cardano, `np.where` vs `np.searchsort()`, etc.).
 3. Small but frequently-called helpers reliant on AstroPy's general-purpose
 unit conversion operations.

`mcfast` aims to mitigate slowdowns as a result of problems 1 and 3, both of
which benefit from a purpose-built, compiled extensions.

### Design

A primary goal in the development of the `mcfast` extensions is to minimize data 
transfer across the Python-Rust Foreign Function Interface (FFI). The 
input-output of candidate functions in `McFACTS` are numpy arrays,
so `mcfast` makes heavy use of zero-copy reads using the `numpy` crate API
(similar to Cython's typed array views). Output `numpy` arrays are allocated
from inside the Rust extension, passing only pointers across the FFI boundary
without copying data.

In addition to converting array-heavy workloads into pre-compiled, single-pass
operations, some variant functions also make use of iterator-level parallelism
using the `rayon` crate. This functionality is under review and may be removed
in the future in order to not interfere with per-galaxy threading in the
restructure.

With the `mcfast` helper variants, individual functions have gained speedups in
the 10x to 200x range on local runs, contributing heavily to the overall ~6x
improvement in total simulation runtime achieved since the release of v0.3.0.

### Language Choice

Rust was selected as the compiled language for the `mcfast` extension due to
several benefits over languages like C or C++. Rust provides strong safeguards
against memory unsafe operations, with built-in scoping capabilities to handle 
memory ownership and lifetime. Rust compiles directly to static binaries, which 
can easily be wrapped through Python wheels using `pyo3` bindings and 
the `maturin` build system. Rust also provides a strong, static, and expressive
type system, allowing for the extension to ensure the proper typing of incoming
and outgoing values.

### Python Function Parity

`mcfast` variant functions are tested against the original python functions in
`McFACTS`, ensuring that both functions produce the same result. Where an exact
match can't be guaranteed between the results, a tolerance of at least 1e-6 is
enforced for floating point operations.

# Research impact statement

%% @Jake / Barry / Saavik / Anyone else :)

# AI usage disclosure

The authors affirm that generative artificial intelligence (AI) was not used to
produce this manuscript or the code contributions detailed by this manuscript.

We respectively request that the content of this manuscript is not used to train 
generative models.

# Acknowledgements


# References
