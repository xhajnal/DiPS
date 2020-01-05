# DiPS Data-informed Parameter Synthesiser

## A Tool for Data-informed Parameter Synthesis for Discrete-Time Stochastic Processes from Multiple-Property Specifications


DiPS builds upon already created model checkers for probabilistic systems -- [PARAM](https://depend.cs.uni-saarland.de/tools/param/publications/bibitem.php?key=HahnHWZ10), [PRISM](http://www.prismmodelchecker.org), [Prophesy](https://moves.rwth-aachen.de/research/tools/prophesy/), and [Storm](http://www.stormchecker.org/) .
It extends those to solve multiple property cases with higher efficiency.

In advance, it provides data-informed analysis to constrain rational functions which are the result of symbolic model checking.

The constraints are solved using:

* space refinement - CEGAR like method splitting the parameter space. In each iteration, the result is provided by:
    * SMT solver - Z3 or dreal
    * interval arithmetic - scipy
* space sampling - checking satisfaction of constraints in selected points 
* optimisation - searching for least violating point and 
* Metropolis-Hastings - searching for most probable parameter points.

To have an overview of the rational functions, we provide visualisation based on sampling.

For a brief summary, please you can see our poster (CMSB 19), and for more information, please read [[1]](#one).
Feel free to leave response either via issues or email.
*****
## HOW TO INSTALL

### 1. INSTALL DEPENDENCIES:

* [Python](https://www.python.org/) 3
    * Windows - just python
    * Ubuntu/Debian - Python header files should also be installed, please use `sudo apt install python3-dev`
    * Fedora/CentOS - Python header files should also be installed, please use `sudo dnf install python3-devel`
* [PRISM](http://www.prismmodelchecker.org) 4.4
* install tkinter library 
  * Windows - already done
  * otherwise go [here](https://tkdocs.com/tutorial/install.html) 
* other missing python packages 
  * in the main directory `MyDiPS` run `pip3 install -v .`
* [Jupyter Notebook](https://jupyter.org/install) (optional)
* [Storm](http://www.stormchecker.org/) (optional, advanced) 
* [Prophesy](https://moves.rwth-aachen.de/research/tools/prophesy/) (optional, advanced)

Are you having trouble with z3? Read `MyDiPS\README-z3.md`. Still having trouble? Please contact us.

****
### 2. SETUP CONFIG (OPTIONAL)

In the main folder, there is `config.ini` file. Please fill in required paths.

[mandatory_paths]
* `prism_path`: path to PRISM `PRISM\bin\`
* `cwd`: path to the main folder `MyDiPS`

[paths]
* `models`: path to [PRISM models](http://www.prismmodelchecker.org/tutorial/die.php) eg. `MyDiPS/models`
* `properties`: path to [PRISM properties](https://www.prismmodelchecker.org/manual/PropertySpecification/Introduction) eg. `MyDiPS/properties`
* `data`: path to data eg. `MyDiPS/data`
* `results`: path to save results (all the results are saved in the subfolders) eg. `MyDiPS/results`
* `tmp`: path to save temporal/intermidiate files  eg. `MyDiPS/tmp`

*****
## HOW TO RUN

*****
Now you can import the code as a library, run the tool with GUI, or use Jupyter notebook. 

### Tool
\- open command line in the main DiPS directory (on Win - please open it with admin privileges to ensure changing the PRISM setting does not fail on permission denied)

`>> cd src`

`>> python gui.py`

Graphical User Interface should appear now (With some output return in the command line). 
We are currently working on the manual; by that time you have to manage on your own.

### Jupyter notebook
\- open command line in the main DiPS directory (on Win - please open it with admin privileges to ensure changing the PRISM setting does not fail on permission denied)

`>> cd ipython`

`>> jupyter notebook`

Several notebooks appear:
* `create\_models\_and\_properties` can be used to automatically create models, properties for population models presented in [[1]](#one).
* `synth_params` serves to synthesise rational function using PRISM 
* `sample_n_visualise` samples and visualises result rational functions
* `generate_data` generate synthetic data by simulating the model
* `direct_param_synth` creates commands to be used for "direct" constrain solving using PRISM and Storm without deriving rational functions.
* `analysis` employs parameter space refinement using z3 solver or interval arithmetic to solve computed constraints    

to follow workflow of the paper [[1]](#one) just run the notebooks in this order. The documentation and the source code of the used functions is in `MyDiPS\src`. When you are familiar with the notebooks, try your input files or even adapt the notebooks.  



*****
## HOW TO USE

*****
To briefly present the main workflow of the tool using the graphical user interface, please see `tutorial.pdf`.

More information on how to use the tool can be obtained from the paper [[1]](#one) (if not reachable, please write us an email.)

Manual in progress - see `manual.pdf`



*****
## HOW TO CITE US

<a name="one"> </a>
[1] Hajnal, M., Nouvian, M., Šafránek, D., Petrov, T.: Data-informed parameter synthesis for population markov chains. In: Češka, M., Paoletti, N. (eds.) Hybrid Systems Biology. pp. 147{164. Springer International Publishing, Cham (2019)


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
