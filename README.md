# mpm
Multiple properties Probabilistic systems Model checker

Mpm builds upon already created model checkers for probabilistic systems -- [PARAM](https://depend.cs.uni-saarland.de/tools/param/publications/bibitem.php?key=HahnHWZ10), [PRISM](http://www.prismmodelchecker.org), [Prophesy](https://moves.rwth-aachen.de/research/tools/prophesy/), and [Storm](http://www.stormchecker.org/) .
It extends those to solve multiple property cases with higher efficiency.

For more information please read [[1]](#one).
Feel free to leave response either via issues or email.
*****
## HOW TO INSTALL

### 1. INSTALL DEPENDENCIES:

* [Python](https://www.python.org/)
* [Jupyter Notebook](https://jupyter.org/install)
* [z3](https://github.com/Z3Prover/z3/releases) - make sure you use same (32/64bit) version as the Python
* [PRISM](http://www.prismmodelchecker.org)
* [Storm](http://www.stormchecker.org/) (optional)
* [Prophesy](https://moves.rwth-aachen.de/research/tools/prophesy/) (optional)
* missing python packages 

****
### 2. SETUP CONFIG

in the main folder there is `config.ini` file. Please fill in required paths.

* `prism_path`: path to PRISM `PRISM\bin\`
* `z3_path`: path to z3 `MYZ3\bin\python`. 
* `cwd`: path to ipython folder `MYmpm/ipython`
* `models`: path to [PRISM models](http://www.prismmodelchecker.org/tutorial/die.php) 
* `properties`: path to [PRISM properties](https://www.prismmodelchecker.org/manual/PropertySpecification/Introduction) 
* `data`: path to data
* `prism_results`: path to prism results
* `storm_results`: path to storm results
* `refinement_results`: path to refinement results 
* `refine_timeout`: timeout for a refinement (in seconds)


*****
## HOW TO RUN

*****
\- open command line in the main mpm directory (on Win - please open it with admin privileges to ensure changing the PRISM setting does not fail on permission denied)

`>> cd ipython`

`>> jupyter notebook`

Several notebooks appears:
 
* `create_data_and_synth_params` can be used to automatically create models, properties, and synthetise rational function using PRISM for population models used in [[1]](#one).
* `sample_n_visualise` samples and visualises result rational functions
* `generate_data` generate synthetic data by simulating the model
* `direct_param_synth` creates commands to be used for "direct" constrain solving using PRISM and Storm without deriving rational functions.
* `analysis` our method - parameter space refinement using z3 solver or interval arithmetic to solve constraints    

to run thought the workflow of the paper just run the notebooks in this order. The documentation and the source code of the used functions is in `mpm\src`. When you are familiar with the notebooks, try your input files or even adapt the notebooks.  


*****
## HOW TO CITE US

<a name="one"> </a>
[1] M. Hajnal, M. Nouvian, D. Safranek, and T. Petrov, Data-informed parameter synthesis for population Markov chains In: International Workshop on Hybrid Systems Biology. Springer, 2019. To appear.
