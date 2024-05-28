# Declarations
.PHONY: all clean

all: clean tests plots vera_plots
tests: mcfacts_sim

######## Definitions ########
#### Package ####
# Version number for the repository
# This is where you change the version number by hand. Not anywhere else.
# Alpha begins at 0.1.0
# Feature-complete Alpha begins at 0.2.0
# Beta begins at 0.3.0
VERSION=0.0.0

### Should work for everyone ###
# Current directory
#HERE=$(shell pwd)
HERE=./

#### Scripts ####
MCFACTS_SIM_EXE = ${HERE}/scripts/mcfacts_sim.py
POPULATION_PLOTS_EXE = ${HERE}/scripts/population_plots.py
VERA_PLOTS_EXE = ${HERE}/scripts/vera_plots.py

######## Instructions ########
#### Install ####

version: clean
	echo "__version__ = '${VERSION}'" > __version__.py
	echo "__version__ = '${VERSION}'" > src/mcfacts/__version__.py

install: clean version
	pip install -e .

#### Test one thing at a time ####

mcfacts_sim: clean
	python ${MCFACTS_SIM_EXE} \
		--fname-log out.log \

plots:  mcfacts_sim
	python ${POPULATION_PLOTS_EXE} 

vera_plots: mcfacts_sim
	python ${VERA_PLOTS_EXE} \
		--cdf chi_eff chi_p M gen1 gen2 t_merge \
		--verbose

#### CLEAN ####
clean:
	rm -rf run*
	rm -rf output_mergers_population.dat
	rm -rf m1m2.png
	rm -rf merger_mass_v_radius.png
	rm -rf q_chi_eff.png
	rm -rf time_of_merger.png
	rm -rf merger_remnant_mass.png
	rm -rf gw_strain.png
	rm -rf out.log
	rm -rf mergers_cdf*.png
	rm -rf mergers_nal*.png
