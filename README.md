# mpm
Multiple properties Probabilistic systems Model checker

Mpm builds upon already created model checkers for probabilistic systems -- [PARAM](https://depend.cs.uni-saarland.de/tools/param/publications/bibitem.php?key=HahnHWZ10), [PRISM](http://www.prismmodelchecker.org), [Prophesy](https://moves.rwth-aachen.de/research/tools/prophesy/), and [Storm](http://www.stormchecker.org/) .
It extends those to solve multiple property cases with higher efficiency.

For more information please read [[1]](#one).

*****
## HOW TO INSTALL

###INSTALL DEPENDENCIES:

* [Jupyter Notebook](https://jupyter.org/install)
* [z3](https://github.com/Z3Prover/z3/releases) 
* [PRISM](http://www.prismmodelchecker.org)
* [Storm](http://www.stormchecker.org/) (optional)
* [Prophesy](https://moves.rwth-aachen.de/research/tools/prophesy/) (optional)
* missing python packages 

****
### SETUP CONFIG

in the main folder there is config.ini file. Please fill in required paths.

* `prism_path`: path to PRISM `PRISM\bin\`
* `z3_path`: path to z3 `MYZ3\bin\python`. 
* `cwd`: path to ipython folder `MYmpm/ipython`
* `models`: path to [PRISM models](http://www.prismmodelchecker.org/tutorial/die.php) 
* `properties`: path to [PRISM properties](https://www.prismmodelchecker.org/manual/PropertySpecification/Introduction) 
* `data`: path to data
* `prism_results`: path to prism results
* `storm_results`: path to storm results


*****
## HOW TO RUN

*****
`>> cd ipython`

`>> jupyter notebook`

Several notebooks appears:
 
* `create_data_and_synth_params` can be used to automatically create models, properties, and synthetise rational function using PRISM for population models used in [1]
* `sample_n_visualise` samples and visualise result rational functions
* `generate_data` generate synthetic data by simulating the model
* `direct_param_synth` creates commands to be used for "direct" constrain solving using PRISM and Storm without deriving rational functions.
* `analysis` our method - parameter space refinement using z3 solver or interval arithmetic to solve constraints    


*****
## HOW TO CITE US

<a name="one"> </a>
[1] M. Hajnal, T. Petrov, D. Safranek, and M. Nouvian, Data-informed parameter synthesis for population Markov chains In: International Workshop on Hybrid Systems Biology. Springer, Cham, 2019. To appear.
