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
MSTAR_RUNS_EXE = ${HERE}/scripts/vera_mstar_bins.py
MSTAR_PLOT_EXE = ${HERE}/src/mcfacts/outputs/plot_mcfacts_handler_quantities.py

#### Setup ####
SEED=3456789012
FNAME_INI= ${HERE}/recipes/model_choice.ini
MSTAR_RUNS_WKDIR = ${HERE}/runs_mstar_bins
# NAL files might not exist unless you download them from
# https://gitlab.com/xevra/nal-data
# scripts that use NAL files might not work unless you install
# gwalk (pip3 install gwalk)
FNAME_GWTC2_NAL = ${HOME}/Repos/nal-data/GWTC-2.nal.hdf5

######## Instructions ########
#### Install ####

version: clean
	echo "__version__ = '${VERSION}'" > __version__.py
	echo "__version__ = '${VERSION}'" > src/mcfacts/__version__.py

install: clean version
	pip3 install -e .

#### Test one thing at a time ####

mcfacts_sim: clean
	python3 ${MCFACTS_SIM_EXE} \
		--n_iterations 10 \
		--fname-ini ${FNAME_INI} \
		--fname-log out.log \
		--seed ${SEED}

plots:  mcfacts_sim
	python3 ${POPULATION_PLOTS_EXE} 

vera_plots: mcfacts_sim
	python3 ${VERA_PLOTS_EXE} \
		--cdf chi_eff chi_p M gen1 gen2 t_merge \
		--verbose

mstar_runs:
	python3 ${MSTAR_RUNS_EXE} \
		--fname-ini ${FNAME_INI} \
		--number_of_timesteps 100 \
		--n_iterations 10 \
		--dynamics \
		--feedback \
		--mstar-min 1e9 \
		--mstar-max 1e13 \
		--nbins 9 \
		--scrub \
		--fname-nal ${FNAME_GWTC2_NAL} \
		--wkdir ${MSTAR_RUNS_WKDIR}
	python3 ${MSTAR_PLOT_EXE} --run-directory ${MSTAR_RUNS_WKDIR}/early
	python3 ${MSTAR_PLOT_EXE} --run-directory ${MSTAR_RUNS_WKDIR}/late
		

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
