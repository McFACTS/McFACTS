<h1 align="center">
    <br>
    <a href="https://github.com/McFACTS/McFACTS"><img src="branding/logo/mcfacts_logo.png" alt="Markdownify" width="500"></a>
    <br>
    <span style="font-weight: normal">
        <b>M</b>onte <b>c</b>arlo <b>F</b>or <b>A</b>GN <b>C</b>hannel <b>T</b>esting and <b>S</b>imulations
    </span>  
    <br>
</h1>

<h4 align="center">A python package that does the AGN channel for you! </h4>

McFACTS is the first public, open source, population synthesis code modeling the *full* AGN channel for LVK detectable BBH mergers.

## Documentation

You can find more information about McFACTS as well as contact and office hour information at our [website](https://saavikford.wixsite.com/saavik/general-7). It's a work in progress, so please be patient!

Opt-in to everything McFACTS: click [here](https://docs.google.com/forms/d/e/1FAIpQLSeupzj8ledPslYc0bHbnJHKB7_LKlr8SY3SfbEVyL5AfeFlVg/viewform) to join our mailing list.

You can find documentation for our code and modules at our [Read the Docs](https://mcfacts.readthedocs.io).

Input and outputs are documented in [`IOdocumentation.txt`](https://github.com/McFACTS/McFACTS/blob/main/IOdocumentation.txt). 

Want build or browse the docs locally? Run the following:

```bash
# Switch to the mcfacts-dev environment and install required packages to build the docs
$ conda activate mcfacts-dev
$ conda install sphinx sphinx_rtd_theme sphinx-autoapi sphinx-automodapi

# Switch to the docs directory
$ cd docs

# Clean up any previous generated docs 
$ make clean

# Generate the html version of the docs in ./docs/build/html
$ make html
```

## Installation

To clone and run this application, you'll need [Git](https://git-scm.com).

The latest development version is available directly from our [GitHub Repo](https://github.com/McFACTS/McFACTS).
To start, clone the repository:

```bash
$ git clone https://github.com/McFACTS/McFACTS
$ cd McFACTS
```

### User Setup

Navigate to the `McFACTS/` directory and run

```bash
pip install [OPTIONS] .
```

The option `--editable` is commonly used when installing with development in mind.
See [pip install documentation](https://pip.pypa.io/en/stable/cli/pip_install/) for additional details.

### Developer Setup

Navigate to the `McFACTS/` directory and run

```bash
python -m pip install --editable .
```

## Running McFACTS

Try a default McFACTS run and make some visualizations:

```bash
# Using the Makefile
$ make plots

# Invoking the Python Script
$ python scripts/mcfacts_sim.py --fname-ini ./recipes/model_choice.ini --seed 3456789108
```

Output and figures will be placed in `McFACTS/runs/`.

### Changing things up

Our default inputs are located at `./recipes/model_choice.ini`. Edit this file or create your own `my_model.ini` file with your chosen values.

If using the `Makefile`, modify the `FNAME_INI` variable with the correct path to your ini file. Run `make plots` again.

If invoking the python script directly, pass the correct path to your file to the `--fname-ini` options:

```bash
python scripts/mcfacts_sim.py --fname-ini <./path/to/my_model.ini>
```

For reproducibility, choose and pass an integer to the `--seed` flag.


> **Hint:**
> You can see all availalbe commandline options with this command:
>```bash
>python scripts/mcfacts_sim.py -h
>```


## Output Files

Output files will appear in `runs`. For each timestep there will be a folder called `gal$N`, with `$N` as the run number. Inside that folder will be `initial_params_bh.dat` which records the initial parameters of all black holes in the simulation and `output_mergers.dat` which records the details of every merger throughout that run. If you are trying to get distributions of merger properties, you probably only need `output_mergers.dat`. If you are trying to track the history of individual mergers or want to know e.g. the state of the nuclear star cluster after an AGN of some duration, you might want to enable the command line option `--save_snapshots`, which will then write `output_bh_single_$ts.dat` and `output_bh_binary_$ts.dat` where `$ts` is the timestep number (0-N)---these files track every single/binary in the simulation at that timestep.

Once all the runs are finished, McFACTS will write the following data files:

* `out.log` records all input parameters for the runs
* `output_mergers_population.dat` details of all BBH mergers
* `output_mergers_survivors.dat` details of all BHs (single and binary) at the end of the AGN
* `output_mergers_lvk.dat` BBH with GW frequencies in the LISA/LVK bands
* `output_mergers_emris.dat` properties of EMRIs

McFACTS will also generate the following plots:

* `gw_strain.png` GW frequency characteristic strain per frequency per year vs average GW frequency
* `m1m2.png` $M_1$ vs $M_2$ as a function of BH generation
* `merger_mass_v_radius.png` Mass of BH merger mass as a function of disk radius and generation
* `merger_remnant_mass.png` Number of BH mergers as a function of remnant mass and BH generation
* `q_chi_eff.png` BH mass ratio ($q=M_{2}/M_{1}$) as a function of $\chi_{\rm eff}$ and BH generation
* `r_chi_p.png` $\chi_\mathrm{p}$ as a function of disk radius and generation
* `time_of_merger.png` Mass of BH merger mass against time of merger and BH generation

## Contributing

Want to contribute? Great! We've got a lot of stuff for you to work on. Please read our [Contributor's Guide](https://github.com/McFACTS/McFACTS/blob/main/docs/source/contribute.rst) for details on our process.

## Citing McFACTS

Please see [Citing McFACTS](https://github.com/McFACTS/McFACTS/blob/main/docs/source/cite.rst) to acknowledge this code.
