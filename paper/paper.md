---
title: 'McFACTS: Monte carlo For AGN Channel Testing and Simulation'
tags:
  - Python
  - astronomy
  - active galactic nuclei
  - population synthesis
authors:
  - name: Jake Postiglione
    orcid: 0000-0003-0738-8186
    corresponding: true
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
  - name: Miranda McCarthy
    orcid: 0009-0005-9964-4790
    affiliation: "1, 2"
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
    ror: "00v4yb702"
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
date: 20 Nov 2026
bibliography: paper.bib
---

# Summary

`McFACTS` (Monte carlo For AGN Channel Testing and Simulation) is the first
open-source population synthesis code for modeling Active Galactic 
Nuclei (AGN) [@mckernan:2025]. AGN are gas-rich and dynamically complex 
structures at the center of some galaxies. The core an AGN contain 
supermassive black holes (SMBH), holding together a diverse swarm of black 
holes and stars that collectively form a nuclear star cluster (NSC). A 
portion of the gravitational wave (GW) events detected by the 
LIGO-Virgo-KAGRA (LVK) are expected to originate from binary black holes 
(BBHs) embedded in AGN disks. Some of these BBHs are expected to be detectable 
by the Laser Interferometer Space Antenna (LISA) GW observatory launching in 
the 2030s. 

Given an AGN disk model and several properties such as the SMBH and NSC mass, 
`McFACTS` generates an initial population of black holes (BH) and stars, 
allowing them to evolve through a variety of physical effects. Some of the 
objects form binaries and eventually merge into remnant BH which can in turn 
become part of another binary. Other objects will experience chance 
encounters with each other throughout the simulation. Single objects 
orbiting the SMBH with a mass ratio ($q$) of $10^{-5}$ to $10^{-7}$ are 
called Extreme Mass Ratio Inspirals (EMRIs) and evolve through GW decay.

`McFACTS` makes use of several packages commonly used in the 
scientific Python ecosystem, such as `AstroPy` [@astropy:2022], 
`NumPy` [@harris:2020], and `SciPy` [@scipy:2020]. The AGN disk structure 
and properties are obtained through the `pAGN` [@gangardt:2024] package, 
supporting both the Sirko and Goodman [-@sg:2003]. and Thompson, Quataert and 
Murray [-@tqm:2005] disk models.


# Statement of Need

Of the proposed BBH formation channels, AGN remain poorly constrained. 
A population synthesis code for the AGN channel must therefore make it possible 
to easily vary physical assumptions and prescriptions, while remaining 
performant. The original release of `McFACTS` demonstrated these 
capabilities and established the scientific value of an open-source AGN channel 
code where one did not previously exist. However, this inaugural version 
concentrated the simulation logic and general bookkeeping into one main 
script, making it difficult to maintain and hard for end users and 
contributors to modify the existing physical assumptions. This obstacle in 
maintain and upgrading the code on grew worse as the size of the `McFACTS` 
collaboration increased.

The restructured version of `McFACTS`, documented in this paper, addresses these
needs directly. Utilities and code for handling physical processes now 
exist as independent and reusable modular units. Instances of new classes,
representing the astrophysical objects within a simulation, are stored and 
managed in a central location with duplicate and consistency 
checks against the entire datastore. The input/output is centralized and 
upgraded to use exchangeable handlers, each supporting different file types 
and improving interoperability with other codes and data processing 
pipelines. Several compute-heavy functions run as external methods 
implemented in Rust, with verified data parity. The result of all of these 
changes is a version of `McFACTS` that is fast to run, simple to support, and 
easier to adopt by collaborators looking to extend the functionality of 
`McFACTS` or integrate it into their work. `McFACTS` is indented to be used by 
researchers who model GW populations or simulate AGN disks, and observers 
that need a synthetic catalog of events from the AGN channel.

Several new astrophysical features are expanded on in works by members of the 
`McFACTS` collaboration and are not fully discussed in this paper: a. Models 
and methods for evaluating the electromagnetic counterparts of mergers 
[@mcpike:2026; @mcpike:2026b, in prep]. b. Determination of merger remnant 
properties via a numerical relativity surrogate model [@ray:2026]. c. 
Application of semi-analytical gas evolution models for binary objects 
[@postiglione:2026, in prep]. 


# State of The Field            

Several population synthesis codes targeting BBH formation and
evolution currently exist, including `COSMIC` [@breivik:2020], 
`COMPAS` [@riley:2022], `CMC` [@rodriguez:2022], and `cogsworth` [@wagg:2024].
These codes focus on formation channels for BBHs in gas-poor environments
(e.g. isolated binaries, dense stellar clusters, etc.), but do not model the 
critical gas-rich dynamics of an AGN disk. Codes that focused on the AGN 
channel have historically been geared towards specific features, such as 
focusing on the mergers found at migration traps, or are not open source 
and have not been released publicly. `McFACTS` distinguishes itself from 
its predecessors by being public, open-source, and tooled to simulate the 
AGN channel for BBH mergers across a broad range of the AGN parameter space.

# Software Design

The restructure of `McFACTS` separates the framework from any particular
simulation or model assembled out of its building blocks. This new framework
is best understood through a top-down lens, where each level orchestrates the
level beneath it. A `Galaxy` holds its populations in a `FilingCabinet` of 
`AGNObjectArray`s. The `AGNObjectArray`s are evolved according to a
`SimulationTimeline`, and each timeline is composed of individual 
`TimelineActor`s (graphically represented in \autoref{diag}). The population 
is seeded with `GalaxyPopulator` objects.

![Diagram showing a high level overview of the relationship between different object types used by the restructured version of `McFACTS`.](restructure-diag.png){#diag height="750px"}

## Galaxy

The `Galaxy` object sits at the core of the simulation framework. A single
`Galaxy` object represents one synthetic galaxy, with its own population,
deterministic random state using a Philox pseudo-random number generator, and the history of simulation
timelines that have been run. A 
`Galaxy` is first populated using `GalaxyPopulator` objects. Once the 
`Galaxy` is seeded, a `SimulationTimeline` holding several `TimelineActor` 
objects is run.

## Filing Cabinet and AGN Object Array

A `FilingCabinet` is a flexible container that stores the population for 
a single `Galaxy` object or an entire universe of galaxies. The cabinet stores a
dictionary of `AGNObjectArray`s and provides a storage interface for global 
variables that need to be accessed by independent modules. Consistency 
checks ensure an object with the same `UUID` does not appear in multiple arrays 
stored in the `FilingCabinet`. The `AGNObjectArray` object allows for the 
association of attributes representing physical properties of objects, while 
maintaining the vectorization convention of storing object properties 
(orbital parameters, mass, spin and spin angle, etc.) in parallel `NumPy` 
arrays. Subclasses for different population categories, such as single black 
holes, BBHs, merged black holes, and so forth, allow for quick object-type 
validation, and the inheritance of shared attributes.


## Simulation Timeline, Timeline Actors, and Galaxy Populators

Evolution of the population in a `Galaxy` is scheduled through a 
`SimulationTimeline` object. The timeline holds an ordered list of different 
operations that are applied to the population over some fixed number of 
timesteps. A single `Galaxy` can run several timelines sequentially, 
processing the population and other parameters in distinct stages. A 
`TimelineActor` object encapsulates the implementation of 
physical processes (migration, binary formation, etc.) through a common 
`perform` interface. Each actor is passed a reference to the `FilingCabinet` 
of that galaxy, and can read and write directly to it. A `GalaxyPopulator` 
is structured similarly to a `TimelineActor`, but distinctly plays the role 
of seeding the initial populations, constructing the starting 
`AGNObjectArray` objects passed to the `FilingCabinet`.


## Scaling with SMBH Mass

The primary executable in the restructured workflow supports scaling 
several properties of the AGN with changes to input parameters. When enabled,
scaling can adjust the NSC mass, the outer disk radius, the boundary between 
the inner/outer disk, the type-I migration trap radius, and capture rates 
from the NSC, following the scaling relations of @delfavero:2025. Prior to 
the restructure, these parameters needed to be tuned by hand or modified by 
a script using regular expressions. Because this step happens during 
initialization, the settings written and logged in the output directory are 
the final scaled settings used in the run. As a result, it is easier to 
simulate AGN of different types and sizes.


## Crosscutting Utilities

The `SettingsManager` acts as a central repository for built-in 
defaults and user-overridable settings. Internal constants are protected from
modification by the user, and custom settings can be passed for use with
third-party modules. The `SnapshotHandler` provides an interface for the 
simulation framework to load and save `SettingsManager` and `FilingCabinet`
objects. By default, we support both `.txt` and `.ini` for the loading and
saving of the `SettingsManager` and support `.txt` files for the 
`FilingCabinet`. Members of the `McFACTS` collaboration have developed 
tooling to support `.hdf5` files. There are also plans to support `.parquet` 
files using the Apache Parquet format and `.db`, `.sqlite` files through 
SQLite. IO formatting is highly customizable, allowing for `McFACTS` to be 
easily implemented into third-party data processing pipelines and software 
stacks.


## Extension: mcfast

`mcfast` is a custom-built extension to the `McFACTS` codebase written in Rust,
with Python bindings produced by `pyo3`. It provides optimized variants of
several `McFACTS` functions, including vectorized, single-pass variants of
computationally heavy functions (e.g. `shock_luminosity`,
`analytical_kick_velocity`) and tailored variants of unit conversion 
functions (e.g. `si_from_r_g`, `r_g_from_units`) which bypass AstroPy's unit 
conversion operations.


### Motivation, Design, and Parity

The most time-intensive operations included long chains of `NumPy` array 
operations on several arrays of equal length, and frequently-called helper 
functions reliant on `AstroPy` unit conversion operations. Several of the 
`mcfast` function variants have measured a 10x to 200x performance increase 
over their Python counterparts. Overall, these individual speedups have 
contributed to an overall ~6x improvement in the total runtime of the 
simulation.

The primary goal of `mcfast` is to minimize data transfer across the 
Python-Rust Foreign Function Interface (FFI). The inputs and outputs of 
candidate functions in `McFACTS` are `NumPy` arrays, so `mcfast` makes heavy 
use of zero-copy reads using the `NumPy` crate API. Output `NumPy` arrays 
are allocated from inside the Rust extension, passing only pointers across 
the FFI without copying data.

`mcfast` variant functions are tested against the original Python functions in
`McFACTS`, ensuring that both functions produce the same result. Where an exact
match can't be guaranteed between the results, a deviation of at most 1e-6 is
enforced for as a tolerance for floating point operations.


# Research Impact Statement

Since its initial release, `McFACTS` has been used as the primary research 
tool in several publications aimed at studying the AGN channel. As of 
November 2026, these publications have collectively been cited over 100 times, 
and provided a wealth of insight 
into how the AGN channel contributes to the population of detectable sources
for LVK and LISA. Papers published by the `McFACTS` 
collaboration study the effects of AGN on the  $q$–$\chi_{\rm eff}$
distribution of LVK detectable BBH mergers [@cook:2025], show the effects 
of NSC properties, galaxy mass, and redshift across an entire population of 
galaxies [@delfavero:2025], demonstrate the effects of the AGN channel on 
electromagnetic signatures and remnant kicks of embedded BBH mergers 
[@mcpike:2026], and explore the AGN channel's effects on the resulting 
properties of BBH merger remnants [@ray:2026] through numerical relativity 
surrogate models. 


# AI Usage Disclosure

The authors affirm that generative artificial intelligence (AI) was not used to
produce this manuscript or the code contributions detailed by this manuscript.

We respectfully request that the content of this manuscript is not used to 
train generative models.


# Acknowledgements

Support for JP was provided by Schmidt Sciences, LLC.
JP, NP, KESF, BM, EM, SR and MM would like to thank the Center for 
Computational Astrophysics at the Flatiron Institute for their hospitality 
while a portion of this research was carried out. The Flatiron Institute is a 
division of the Simons Foundation.
JP would like to thank Michael Zingale for their helpful 
presentation and conversation about code structuring and numerical methods.


# References
