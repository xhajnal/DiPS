# mpm
Multiple properties Probabilistic systems Model checker

Mpm builds upon already created model checkers for probabilistic systems -- [PARAM](https://depend.cs.uni-saarland.de/tools/param/publications/bibitem.php?key=HahnHWZ10), [PRISM](http://www.prismmodelchecker.org), [Prophesy](https://moves.rwth-aachen.de/research/tools/prophesy/), and [Storm](http://www.stormchecker.org/) .
It extends those to solve multiple property cases with higher efficiency.

*****
## HOW TO INSTALL

###INSTALL DEPENDENCIES:

* [z3](https://github.com/Z3Prover/z3/releases)
* [PRISM](http://www.prismmodelchecker.org)
* [Jupyter Notebook](https://jupyter.org/install)
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
