# Declarations
.PHONY: all clean

all: clean tests plots #vera_plots
tests: mcfacts_sim

######## Definitions ########
#### Package ####
# Version number for the repository
# This is where you change the version number by hand. Not anywhere else.
# Alpha begins at 0.1.0
# Feature-complete Alpha begins at 0.2.0
# Beta begins at 0.3.0
VERSION=0.1.0

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
STARS_PLOTS = ${HERE}/scripts/stars_plots.py

#### Setup ####
SEED=3456789108
#FNAME_INI= ${HERE}/recipes/p1_thompson.ini
FNAME_INI= ${HERE}/recipes/model_choice_old.ini
FNAME_INI_MSTAR= ${HERE}/recipes/p3_pAGN_on.ini
MSTAR_RUNS_WKDIR = ${HERE}/runs_mstar_bins_pAGN
# NAL files might not exist unless you download them from
# https://gitlab.com/xevra/nal-data
# scripts that use NAL files might not work unless you install
# gwalk (pip3 install gwalk)
FNAME_GWTC2_NAL = ${HOME}/Repos/nal-data/GWTC-2.nal.hdf5
#Set this to change your working directory
wd=${HERE}

######## Instructions ########
#### Install ####

version: clean
	echo "__version__ = '${VERSION}'" > __version__.py
	echo "__version__ = '${VERSION}'" > src/mcfacts/__version__.py

install: clean version
	python -m pip install --editable .

setup: clean version
	source ~/.bash_profile && \
	conda activate base && \
	conda remove -n mcfacts-dev --all -y && \
	conda create --name mcfacts-dev "python>=3.10.4,<3.13" pip "pytest" -c conda-forge -c defaults -y && \
	conda activate mcfacts-dev && \
	python -m pip install --editable .
	@echo "\n"
	@echo "Run 'conda activate mcfacts-dev' to switch to the correct conda environment."
	@echo "\n"
	@echo "Want to keep up-to-date with future McUpdates and announcements? Sign up for our mailing list!"
	@echo "https://docs.google.com/forms/d/e/1FAIpQLSeupzj8ledPslYc0bHbnJHKB7_LKlr8SY3SfbEVyL5AfeFlVg/viewform"
	@echo "\n"

unit_test: clean version
	source ~/.bash_profile && \
	conda activate mcfacts-dev && \
	pytest

DIST=dist/mcfacts-${VERSION}.tar.gz
build-install: clean version
	python3 -m build
	python3 -m pip install $(DIST)

test-build: build-install
	mkdir test-build
	cp ${DIST} test-build
	cp ${MCFACTS_SIM_EXE} test-build
	cd test-build; pip install ${notdir ${DIST}}
	cd test-build; python3 ${notdir ${MCFACTS_SIM_EXE}}

#### Test one thing at a time ####

# do not put linebreaks between any of these lines. Your run will call a different .ini file
mcfacts_sim: clean
	mkdir -p runs
	cd runs; \
		python ../${MCFACTS_SIM_EXE} \
		--galaxy_num 100 \
		--fname-ini ../${FNAME_INI} \
		--fname-log out.log \
		--seed ${SEED}


plots: mcfacts_sim
	cd runs; \
	python ../${POPULATION_PLOTS_EXE} --fname-mergers ${wd}/output_mergers_population.dat --plots-directory ${wd}

just_plots:
	cd runs; \
	python ../${POPULATION_PLOTS_EXE} --fname-mergers ${wd}/output_mergers_population.dat --plots-directory ${wd}

vera_plots: mcfacts_sim
	python ${VERA_PLOTS_EXE} \
		--cdf-fields chi_eff chi_p final_mass gen1 gen2 time_merge \
		--verbose

mstar_runs:
	python ${MSTAR_RUNS_EXE} \
		--fname-ini ${FNAME_INI_MSTAR} \
		--timestep_num 1000 \
		--bin_num_max 10000 \
		--nbins 33 \
		--galaxy_num 100 \
		--mstar-min 1e9 \
		--mstar-max 1e13 \
		--scrub \
		--fname-nal ${FNAME_GWTC2_NAL} \
		--wkdir ${MSTAR_RUNS_WKDIR} \
		--truncate-opacity
		#--nbins 33 
		#--timestep_num 1000 \
	#python3 ${MSTAR_PLOT_EXE} --run-directory ${MSTAR_RUNS_WKDIR}

kaila_stars: plots
	cd runs; \
	python ../${STARS_PLOTS} \
	--runs-directory ${wd} \
	--fname-stars ${wd}/output_mergers_stars.dat \
	--plots-directory ${wd}
		

#### CLEAN ####

#TODO: Create an IO class that wraps the standard IO. This wrapper will keep a persistent log of all of the
#instantaneous files created. The wrapper would have a cleanup function, and can also report metrics :^)
#Plus, if we use a standard python IO library, we don't have to worry about rm / del and wildcards!

clean:
	rm -rf ${wd}/run*
	rm -rf ${wd}/runs/*
	rm -rf ${wd}/output_mergers*.dat
	rm -rf ${wd}/m1m2.png
	rm -rf ${wd}/merger_mass_v_radius.png
	rm -rf ${wd}/q_chi_eff.png
	rm -rf ${wd}/time_of_merger.png
	rm -rf ${wd}/merger_remnant_mass.png
	rm -rf ${wd}/gw_strain.png
	rm -rf ${wd}/out.log
	rm -rf ${wd}/mergers_cdf*.png
	rm -rf ${wd}/mergers_nal*.png
	rm -rf ${wd}/r_chi_p.png
	rm -rf ${wd}/dist
	rm -rf ${wd}/test-build

clean_win:
	for /d %%i in (.\run*) do rd /s /q "%%i"
	for /d %%i in (.\output_mergers*.dat) do rd /s /q "%%i"
	del /q .\m1m2.png
	del /q .\merger_mass_v_radius.png
	del /q .\q_chi_eff.png
	del /q .\time_of_merger.png
	del /q .\merger_remnant_mass.png
	del /q .\gw_strain.png
	del /q .\out.log
	for /d %%i in (.\mergers_cdf*.png) do rd /s /q "%%i"
	for /d %%i in (.\mergers_nal*.png) do rd /s /q "%%i"
	del /q .\r_chi_p.png
