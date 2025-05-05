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
VERSION=0.2.1

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
DISK_MASS_PLOTS = ${HERE}/scripts/disk_mass_plots.py
ORBA_MASS_FRAMES = ${HERE}/scripts/star_bh_movie_frames.py
EM_PLOTS = ${HERE}/scripts/em_plots.py

#### Setup ####
SEED=3456789108 # put an 8 here
#FNAME_INI= ${HERE}/recipes/p1_thompson.ini
FNAME_INI= ${HERE}/recipes/model_choice_old.ini
FNAME_INI_MSTAR_SCALE= ${HERE}/recipes/paper_3/p3_scale.ini
FNAME_INI_MSTAR_FIXED= ${HERE}/recipes/paper_3/p3_fixed.ini
MSTAR_RUNS_WKDIR_SCALE = ${HERE}/runs_mstar_bins_scale
MSTAR_RUNS_WKDIR_FIXED = ${HERE}/runs_mstar_bins_fixed
# NAL files might not exist unless you download them from
# https://gitlab.com/xevra/nal-data
# scripts that use NAL files might not work unless you install
# gwalk (pip3 install gwalk)
FNAME_GWTC2_NAL = ${HOME}/Repos/nal-data/GWTC-2.nal.hdf5
#Set this to change your working directory
wd=${HERE}

## Setup for dumb parallelization
MBINS_FIXED := \
	FIXED_00 FIXED_01 FIXED_02 FIXED_03 FIXED_04 \
	FIXED_05 FIXED_06 FIXED_07 FIXED_08 FIXED_09 \
	FIXED_10 FIXED_11 FIXED_12 FIXED_13 FIXED_14 \
	FIXED_15 FIXED_16 FIXED_17 FIXED_18 FIXED_19 \
	FIXED_20 FIXED_21 FIXED_22 FIXED_23 FIXED_24 \
	FIXED_25 FIXED_26 FIXED_27 FIXED_28 FIXED_29 \
	FIXED_30 FIXED_31 FIXED_32
MBINS_SCALE := \
	SCALE_00 SCALE_01 SCALE_02 SCALE_03 SCALE_04 \
	SCALE_05 SCALE_06 SCALE_07 SCALE_08 SCALE_09 \
	SCALE_10 SCALE_11 SCALE_12 SCALE_13 SCALE_14 \
	SCALE_15 SCALE_16 SCALE_17 SCALE_18 SCALE_19 \
	SCALE_20 SCALE_21 SCALE_22 SCALE_23 SCALE_24 \
	SCALE_25 SCALE_26 SCALE_27 SCALE_28 SCALE_29 \
	SCALE_30 SCALE_31 SCALE_32

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
		--fname-log mcfacts.log \
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

kaila_stars: plots
	cd runs; \
	python ../${STARS_PLOTS} \
	--runs-directory ${wd} \
	--fname-stars ${wd}/output_mergers_stars_population.dat \
	--fname-stars-merge ${wd}/output_mergers_stars_merged.dat \
	--fname-stars-explode ${wd}/output_mergers_stars_exploded.dat \
	--plots-directory ${wd}

kaila_stars_movie: clean
	mkdir -p runs
	cd runs; \
		python ../${MCFACTS_SIM_EXE} \
		--galaxy_num 100 \
		--fname-ini ../${FNAME_INI} \
		--fname-log mcfacts.log \
		--seed ${SEED} \
		--save-snapshots

kaila_stars_make_movie: kaila_stars_plots
	cd runs; \
	python ../${ORBA_MASS_FRAMES} \
	--fpath-snapshots ${wd}/gal000/ \
	--num-timesteps 50 \
	--plots-directory ${wd}/gal000
	rm -fv ${wd}/runs/orba_mass_movie.mp4
	ffmpeg -f image2 -framerate 5 -i ${wd}/runs/gal000/orba_mass_movie_timestep_%02d_log.png -vcodec libx264 -pix_fmt yuv420p -crf 22 ${wd}/runs/orba_mass_movie.mp4

kaila_stars_plots: just_plots
	cd runs; \
	python ../${STARS_PLOTS} \
	--runs-directory ${wd} \
	--fname-stars ${wd}/output_mergers_stars_population.dat \
	--fname-stars-merge ${wd}/output_mergers_stars_merged.dat \
	--fname-stars-explode ${wd}/output_mergers_stars_exploded.dat \
	--plots-directory ${wd}

disk_mass_plots:
	cd runs; \
	python ../${DISK_MASS_PLOTS} \
	--runs-directory ${wd} \
	--fname-disk ${wd}/output_diskmasscycled.dat \
	--plots-directory ${wd}		
		
em_plots: 
	cd runs; \
	python ../${EM_PLOTS} \
	--runs-directory ${wd} \
	--fname-emris ${wd}/output_mergers_emris.dat \
	--fname-mergers ${wd}/output_mergers_population.dat \
	--fname-lvk ${wd}/output_mergers_lvk.dat \
	--plots-directory ${wd}

#### Vera's mstar_runs ####

# Define the setup for mstar_runs for the scaled inifile
setup_mstar_runs_scale:
	python ${MSTAR_RUNS_EXE} \
		--fname-ini ${FNAME_INI_MSTAR_SCALE} \
		--timestep_num 1000 \
		--bin_num_max 10000 \
		--galaxy_num 100 \
		--mbins ${MBINS_SCALE} \
		--mstar-min 1e9 \
		--mstar-max 1e13 \
		--scrub \
		--fname-nal ${FNAME_GWTC2_NAL} \
		--wkdir ${MSTAR_RUNS_WKDIR_SCALE} \
		--truncate-opacity
		#--nbins 33 
		#--timestep_num 1000 \
	#python3 ${MSTAR_PLOT_EXE} --run-directory ${MSTAR_RUNS_WKDIR}

# Define the setup for mstar_runs with the fixed inifile
setup_mstar_runs_fixed:
	python ${MSTAR_RUNS_EXE} \
		--fname-ini ${FNAME_INI_MSTAR_FIXED} \
		--timestep_num 1000 \
		--bin_num_max 10000 \
		--galaxy_num 100 \
		--mbins ${MBINS_FIXED} \
		--mstar-min 1e9 \
		--mstar-max 1e13 \
		--scrub \
		--fname-nal ${FNAME_GWTC2_NAL} \
		--wkdir ${MSTAR_RUNS_WKDIR_FIXED}
		#--nbins 33 
		#--timestep_num 1000 \
	#python3 ${MSTAR_PLOT_EXE} --run-directory ${MSTAR_RUNS_WKDIR}
		
# Define an individual job for the fixed inifile
%.run_fixed: setup_mstar_runs_fixed
	bash runs_mstar_bins_fixed/early/$(basename $@)/p3_fixed.sh
	bash runs_mstar_bins_fixed/late/$(basename $@)/p3_fixed.sh
# Define an individual job for the scaled inifile
%.run_scale: setup_mstar_runs_scale
	bash runs_mstar_bins_scale/early/$(basename $@)/p3_scale.sh
	bash runs_mstar_bins_scale/late/$(basename $@)/p3_scale.sh

## You can't handle the truth!
# Seriously, I am lucky every time I can get this to work at all
# Pattern Rules are truly the dark arts
mstar_runs_scale: $(MBINS_SCALE)
$(MBINS_SCALE): %: %.run_scale
mstar_runs_fixed: $(MBINS_FIXED)
$(MBINS_FIXED): %: %.run_fixed


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
	rm -rf ${wd}/mcfacts.log
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
	del /q .\mcfacts.log
	for /d %%i in (.\mergers_cdf*.png) do rd /s /q "%%i"
	for /d %%i in (.\mergers_nal*.png) do rd /s /q "%%i"
	del /q .\r_chi_p.png
