# mpm
Multiple properties Probabilistic systems Model checker

Mpm builds upon already created model checkers for probabilistic systems -- [PARAM](https://depend.cs.uni-saarland.de/tools/param/publications/bibitem.php?key=HahnHWZ10), [PRISM](http://www.prismmodelchecker.org), [Prophesy](https://moves.rwth-aachen.de/research/tools/prophesy/), and [Storm](http://www.stormchecker.org/) .
It extends those to solve multiple property cases with higher efficiency.

In advance it provides data-informed analysis to constrain rational functions which are result of symbolic model checking. 

The constraints are solved using CEGAR like method - Space refinement. 

Besides that we provide sampling and other techniques to have overview of the rational functions or the parameter space.

For a brief overview please you can see our poster (CMSB 19) and for more information please read [[1]](#one).
Feel free to leave response either via issues or email.
*****
## HOW TO INSTALL

### 1. INSTALL DEPENDENCIES:

* [Python](https://www.python.org/) 3
	* Windows - just python
	* Ubuntu/Debian - Python header files should also be installed, please use `sudo apt install python3-dev`
	* Fedora/CentOS - Python header files should also be installed, please use`sudo dnf install python3-devel`
* [PRISM](http://www.prismmodelchecker.org) 4.4
* install tkinter library 
  * Windows - already done
  * otherwise go [here](https://tkdocs.com/tutorial/install.html) 
* other missing python packages 
  * in the main directory `MYmpm` run `pip3 install -v .`
* [Jupyter Notebook](https://jupyter.org/install) (optional)
* [Storm](http://www.stormchecker.org/) (optional, advanced) 
* [Prophesy](https://moves.rwth-aachen.de/research/tools/prophesy/) (optional, advanced)

Are you having trouble with z3? Read `MYmpm\README-z3.md`. Still having trouble? Please contact us.

****
### 2. SETUP CONFIG (OPTIONAL)

In the main folder there is `config.ini` file. Please fill in required paths.

[mandatory_paths]
* `prism_path`: path to PRISM `PRISM\bin\`
* `cwd`: path to the main folder `MYmpm`

[paths]
* `models`: path to [PRISM models](http://www.prismmodelchecker.org/tutorial/die.php) eg. `MYmpm/models`
* `properties`: path to [PRISM properties](https://www.prismmodelchecker.org/manual/PropertySpecification/Introduction) eg. `MYmpm/properties`
* `data`: path to data eg. `MYmpm/data`
* `results`: path to save results (all the results are saved in the subfolders) eg. `MYmpm/results`
* `tmp`: path to save temporal/intermidiate files  eg. `MYmpm/tmp`

*****
## HOW TO RUN

*****
Now you can import the code as library, run the tool with GUI, or use Jupyter notebook. 

### Tool
\- open command line in the main mpm directory (on Win - please open it with admin privileges to ensure changing the PRISM setting does not fail on permission denied)

`>> cd src`

`>> python gui.py`

Graphical User Interface should appear now (With some output return in command line). 
We are currently working on the manual, by that time you have to manage on your own.

### Jupyter notebook
\- open command line in the main mpm directory (on Win - please open it with admin privileges to ensure changing the PRISM setting does not fail on permission denied)

`>> cd ipython`

`>> jupyter notebook`

Several notebooks appear:
* `create\_models\_and\_properties` can be used to automatically create models, properties for population models presented in [[1]](#one).
* `synth_params` serves to synthesise rational function using PRISM 
* `sample_n_visualise` samples and visualises result rational functions
* `generate_data` generate synthetic data by simulating the model
* `direct_param_synth` creates commands to be used for "direct" constrain solving using PRISM and Storm without deriving rational functions.
* `analysis` employs parameter space refinement using z3 solver or interval arithmetic to solve computed constraints    

to follow workflow of the paper [[1]](#one) just run the notebooks in this order. The documentation and the source code of the used functions is in `mpm\src`. When you are familiar with the notebooks, try your input files or even adapt the notebooks.  



*****
## HOW TO USE

*****
To briefly present the main worflow of the tool using graphical user interface please see `tutorial.pdf`.

More information on how to use the tool can be obtained from the paper [[1]](#one) (if not reachable, please write us an email.)

Manual in progress - see `manual.pdf`



*****
## HOW TO CITE US

<a name="one"> </a>
[1] M. Hajnal, M. Nouvian, D. Šafránek, and T. Petrov, Data-informed parameter synthesis for population Markov chains In: International Workshop on Hybrid Systems Biology. Springer, 2019. To appear.


*****
## ACKNOWLEDGMENT

I want to thank the following people who helped me with the code issues:
* [Michal Ovčiarik](https://github.com/bargulg)
* [Nhat-Huy Phung](https://github.com/huypn12)
* [Denis Repin](https://github.com/dennerepin)

and I want to thank the following people for user testing and feedback:
* [Denis Repin](https://github.com/dennerepin)
* [Samuel Pastva](https://github.com/daemontus)
* [Stefano Tognazzi](https://github.com/stefanotognazzi)
* Morgane Nouvian
* [Matej Troják](https://github.com/xtrojak)
