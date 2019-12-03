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
* [Jupyter Notebook](https://jupyter.org/install) (optional)
* [z3](https://github.com/Z3Prover/z3/releases) 4.6.0 - make sure you use same (32/64bit) version as the Python
* [PRISM](http://www.prismmodelchecker.org) 4.4
* [Storm](http://www.stormchecker.org/) (optional) 
* [Prophesy](https://moves.rwth-aachen.de/research/tools/prophesy/) (optional)
* install tkinter library 
  * Windows - already done
  * otherwise go [here](https://tkdocs.com/tutorial/install.html) 
* other missing python packages 
  * in the main directory run `pip install -v .`

****
### 2. SETUP CONFIG

In the main folder there is `config.ini` file. Please fill in required paths.

[mandatory_paths]
* `prism_path`: path to PRISM `PRISM\bin\`
* `z3_path`: path to z3 `MYZ3\bin\python`. 
* `cwd`: path to ipython folder `MYmpm/ipython`

[paths]
* `models`: path to [PRISM models](http://www.prismmodelchecker.org/tutorial/die.php) 
* `properties`: path to [PRISM properties](https://www.prismmodelchecker.org/manual/PropertySpecification/Introduction) 
* `data`: path to data
* `results`: path to save results
* `tmp`: path to save temporal/intermidiate files

*****
## HOW TO RUN

*****

### Jupyter notebook
\- open command line in the main mpm directory (on Win - please open it with admin privileges to ensure changing the PRISM setting does not fail on permission denied)

`>> cd ipython`

`>> jupyter notebook`

Several notebooks appear:
 
* `create_data_and_synth_params` can be used to automatically create models, properties, and synthesise rational function using PRISM for population models used in [[1]](#one).
* `sample_n_visualise` samples and visualises result rational functions
* `generate_data` generate synthetic data by simulating the model
* `direct_param_synth` creates commands to be used for "direct" constrain solving using PRISM and Storm without deriving rational functions.
* `analysis` our method - parameter space refinement using z3 solver or interval arithmetic to solve constraints    

to run thought the workflow of the paper just run the notebooks in this order. The documentation and the source code of the used functions is in `mpm\src`. When you are familiar with the notebooks, try your input files or even adapt the notebooks.  

### Tool
\- open command line in the main mpm directory (on Win - please open it with admin privileges to ensure changing the PRISM setting does not fail on permission denied)

`>> cd src`

`>> python gui.py`

Graphical User Interface should appear now. We are currently working on the manual, by that time you have to manage on your own.

*****
## HOW TO USE

*****
Manual in progress ...

The main worflow described in the paper (below) (if not reachable, please write us an email.)

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
