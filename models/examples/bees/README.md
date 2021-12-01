# Honeybee mass-stinging defence response
README author: Matej Hajnal, Julia Klein

## Contents
1. [Introduction](#Introduction)
2. [Model mechanics](#Model-mechanics)
3. [Model assuptions](#Model-assuptions)
4. [Temporal properties](#Temporal-properties)
5. [Parametric model checking and rational function](#Parametric-model-checking-and-rational-function)
6. [Data](#Data)
7. [Analysis](#Analysis)

## Introduction 
Honeybees live in densely populated nests, in which they also gather resources in the form of pollen and honey. 
This makes their colonies very attractive troves of nutrients for many predators, including large mammals such as bears and humans.
To fend them off, the bees have to band together into a collective stinging attack.
This defensive reaction is typically initiated by (transiently) specialised bees termed guard bees, who monitor the colony's surroundings.
They react to disturbances by stinging the intruder or by running inside the nest with their stinger extruded and their wings fanning.
In both cases, their behaviour causes the release of the sting alarm pheromone (SAP), a complex pheromonal blend carried directly on the stinger.
This chemical signal arouses nearby bees and recruits them to the site of the disturbance, where they decide whether to participate or not in the defensive effort by stinging or otherwise harassing the predator.
Hence, the SAP plays a major role in amplifying the defensive reaction of the colony so that it reaches critical mass.

In this model, our aim is to better understand how honeybees use the SAP to communicate during defence, in particular by quantifying its effect on the likelihood to sting of individual bees. 
To do so, we observed the behaviour of groups of bees confronted with a fake predator (a rotating dummy) inside an arena and quantified their defensive reaction by simply counting the number of stingers embedded in the dummy at the end of a trial.
We propose a mathematical model of the group dynamics, which transparently links the probabilistic choice to sting of a single bee in response to given alarm pheromone concentrations to the collective outcome observed in the experiment. Concretely, each honeybee is modelled as a Markovian agent potentially triggered into stinging at a given alarm pheromone concentration, and which releases more alarm pheromone upon doing so. Each stinging bee thus modifies the environment so that more bees may be triggered into stinging, leading to a chain of reactions that stops when no additional bee is recruited (steady-state reached).



| ![space-1.jpg](https://user-images.githubusercontent.com/23458520/133645103-4735cec6-8490-406c-8ce5-dce59de7c471.png) |
|:--:|
| <b> Model of three bees as a parametrised Markov Chain. In each state, a vector depicting a state of the population is shown. Each bee agent tracks its state -init before the stimulus, 1 stinging, and !i not stinging when i amount of pheromone is present, i.e. A < θ_i. Transitions, updates of bee states whether to sting or not, are parametrised by the probability of stinging when i amount of pheromone is present - r_i. When multiple bees are updating their state, we consider this as an independent decision and hence multiply respective terms. Terms such as (r_i−r_k)/(1−r_k) interpret the conditional probability P(A > θ_i \| A < θ_k). </b> |



## Model mechanics
To create a model of a given population size, we created a Python script, `src/create_models_properties.py`, which is a part of the tool package. 
It first creates state space, a set of reachable states. 
For each number of stinging bees k ∈ [0,...,n], a state with k bees in state 1, we compute all combinations with repetition of n−k elements from [0,...,k], possible levels of concentration of alarm pheromone for which respective bee did not sting. 
In the second step, to create the set of transitions in each step, we compute the number of bees which can be updated at the current state - non-stinging bees with the lower state !i than the number of already stinging bees. 
We multiply terms ![image](https://user-images.githubusercontent.com/23458520/133643818-c20082c4-5ed2-43b1-afe4-38a8130cc251.png)
or ![image](https://user-images.githubusercontent.com/23458520/133643966-3de6fa3a-92ab-4df9-bc8a-f214b5831c0e.png) for each unique !i in the state to create all possible decisions of each bee either to sting or not to sting respectively.
We multiply each of the terms by a combination number picking a number of bees not stinging at alarm pheromone level, !i, to be updated in this transition from all such bees in the state. 
 
We leverage abstract description of PRISM language, which applies a transition to all possible states, e.g. in a state where there are two bees stinging, the other two bees did not sting at pheromone level 0, another bee did not sting at level 1, and the rest of the bees decided not to sting at level 2, there are three bees to be updated in total, those who did not sting at level 0 and level 1. 
Each bee decides to either sting or not. 
Let's say one of the two bees which did not sting at level 0 decides to sting, and the second does not.
The third bee decides to sting as well. 
Hence we have a transition to a state with 4 bees stinging and one additional bee which does not sting at alarm pheromone at level 2.
Transition probability is equal to:

![image](https://user-images.githubusercontent.com/23458520/133643137-901ad2d7-4e21-46c4-baa6-98da0f754b35.png)



## Model assumptions
When creating the model, we assumed several facts thanks to the observations and expertise of the biologists:

### Degradation of SAP
The half-life of the SAP is much higher than the time of a bee to respond; hence we do not model its degradation.

### Non-decreasing parameter values
In the transitions of the model, the term encoding the probability of a bee stinging when i amount of pheromone given the bee did not sting at k amount of pheromone:
(r_i−r_k)/(1−r_k) see Figure above, assumes that the later probability, r_i, is not smaller than in the previous state -r_k, r_i ≥ r_k, otherwise, this term describing a conditional probability 
should evaluate to 0 and not negative value.
Since not PRISM nor Storm is capable of working with if-then clause nor max(0, (r_i − r_k)/(1 − r_k))) we constrain the parameter values of the result rational functions.
We have adapted DiPS for this purpose, with the implementation visible on the branch `non-decreasing`. Another option is to introduce fresh parameters for each of these terms.
However, this would increase the number of parameters quadratically as there is up to n·(n−1)/2 parameter pairs to be substituted.

### Synchronisation
At each point, more than a single bee can sting, different update functions can be applied. 
For instance, we can assume every time, only a single bee can decide at a time. This would model a bee that reacts infinitely fast right after sensing the required SAP level to sting. 
We refer to this as asynchronous mechanics.
However, from the observations, it looks like the opposite is correct, and all the bees are able to sting at the time sting as we consider time to decide to sting much longer than the time to spread the SAP to other bees.
We refer to this as synchronous mechanics, and we use this mechanic in the paper and analysis.
Finally, there is yet another option we have considered, and it is a combination of these two mechanics - semisynchronous mechanics.
In the first step, when the SAP level is 0, all the bees decide to sting all at once, and after that, asynchronous mechanics starts.
For simplicity, we speak only about synchronous mechanics in the rest of this README and in the paper, however you can create models with other mechanics as well using the script.


## Temporal properties
In the data, we are interested in the distribution among the number of stinging bees, hence the distribution of reachability of BSCCs in the model.
We use PCTL properties such as P=?[F(a0 = 0)&(a1 = 0)&(a2 = 0)&...&(a9 = 0)&(b= 1)], where (a0 = 0)&(a1 = 0)&(a2 = 0)&...&(a9 = 0)&(b= 1) is a state where count of a_i indicates number of stinging bees and b=1 is a flag of a "final" state.
F is the temporal Future operator, which encodes reachability, and prefix P=? queries the probability of the wrapped term. 
This way, we query the probability of reaching individual BSCC. 

## Parametric model checking and rational function 
As the model is parametric, the probabilities of reaching respective BSCC are algebraic expressions parametrised with the model parameters. 
These expressions are in the form of a rational function in the case of standard pMCs.
Model checkers such as PRISM and Storm are capable of computing these rational functions.
The procedure is referred to as parametric model checking.

## Model selection - agnostic, linear, and sigmoidal model
We perform model selection between two biologically plausible hypotheses - that the parameters follow either a linear or a sigmoidal trend.
We refer to the model with all parametrisation as agnostic as it does not propose any dependence on the parameter values besides the non-decreasing trend.
 
Linear and Sigmoidal models were created by replacing r_i with respective dependence, decreasing the dimensionality of parameter space, which enables us to run space sampling and refinement.
 
### Linear model
 r_i = r_0 + i·∆ , 
 where r_0 is the basal probability to sting at SAP level zero and ∆ is the additional probability to sting for each SAP level added. 
 
### Sigmoidal model
 The sigmoidal model is depicted as a dependence of parameter values using Hill function. Hence, the value of respective r_i can be expressed
as: r_i = r_0 + (Vmax−r_0) / (1+( Km/i)^n).
 This transformation decreases the number of parameters to four: r_0 - basal level, Vmax - saturation level, Km - value at which the hill
function is at half of the slope, and n - Hill coefficient indicates the slope of the curve.

## Data
 We have conducted an experiment counting number of stinging bees in the arena as a result of 92 trials. 
 We calculate the probability of x number of stinging bees as # of observed trials with x stinging bees / # all trials. 
 As `data.txt` shows the result distribution as 0 stinging bees, 1 stinging bee, ... is 
 
 [0.2391304348, 0.152173913, 0.2065217391, 0.1195652174, 0.04347826087, 0.1086956522, 0.08695652174, 0.02173913043, 0.02173913043, 0, 0] 
 
## Analysis
Finally, we provide a step-by-step analysis to reproduce the observation presented in the paper while each step is described in more detail in the tutorial - `tutorial.pdf`.
Framework installation instructions can be found in the `README.md`.
Both files are available in the DiPS tool main folder or at https://github.com/xhajnal/DiPS.

### Agnostic model
- First, we select the `non-decreasing` branch - `>> git checkout -b non-decreasing`. 
- Next, we open DiPS' Graphical User Interface. 
- In the very first tab, the model and temporal properties are loaded - `10_synchronous.pm` and `prop_10_bees.pctl`.
- In the second tab, we compute the rational functions:
	- We select installed parametric model checker - PRISM or Storm - and choose to factor the result functions or not.  
 	- We press Run parametric model checking button, and after the procedure is complete, the result rational functions are shown in the editor. 
- Alternatively to the previous two points, you can load the rational functions, `rational_functions/agnostic_10_bees_factorised.txt`, in the second tab.
- We follow the fourth tab Data & Intervals, 
	- Here, we load the data file, `data.txt`, and select 92 samples.
	- We choose 0.9 as the confidence level and Agresti-Coull as a method to compute confidence intervals. 
	- To actually compute them, we press Compute Intervals button. 
- To compute the optimised point: 
	- We select Apply non-decreasing params checkbox and press Optimise parameters button. 
	- After the procedure is complete, a new window with the results is shown. 
- To calculate the constraints, 
	- we continue to the next tab, and we press Calculate constraints button.
	- The constraints will be shown in the editor.
- Now, we follow to the last tab, where the rest of the analysis of the agnostic model is done. 
- Sampling and Refinement analysis is here, but due to the dimensionality of the agnostic model, we won't probably be able to obtain reasonable results.
- Finally, the Metropolis-Hastings analysis - 
	- select reasonable number of iteration (10,000,000 used), 
	- select Apply non-decreasing params checkbox, and 
	- press the Metropolis-Hastings button to run the analysis.
	- First, two metadata plots are shown. You can read more about these in the tutorial. 
 - Finally, the main result is shown in the main window of DiPS showing the set of accepted points as a scatter-line plot.
 - We strongly recommend running the analysis with a very low number of iterations to extrapolate the expected time to finish first.

### Linear and Sigmoidal model
Here, we describe only the parts which are different from the analysis of the agnostic model.
For this part of the analysis, we can either use `non-decreasing` branch or the `default` branch.
- During the analysis, unselect each `apply non-decreasing params` checkbox.
- First, skip loading of the model and properties and directly load the rational functions from a file in the second tab - `rational_functions/lin_10_bees_factorised.txt` and `rational_functions/sig_4_param_partially_factorised.txt` for the linear and sigmoidal version respectively.
- Load data and follow the settings in the fourth tab, Data & Intervals, as previously. 
- Compute the optimised point (unselect the `apply non-decreasing params` checkbox) and calculate the constraints as previously, you need to press the button even when new rational functions are loaded.
- Repeat the Metropolis-Hastings analysis while unselecting the `apply non-decreasing params` checkbox and choosing the right number of iterations (we used 30,000,000 for the linear model and 358,287 for the sigmoidal model)
 
 
### Model selection
Model selection is implemented in the R script `model_selection.R`. When sourced, it automatically runs all analyses, prints the outputs in the console, and saves the plots in the same folder.
 
As an input, the script uses three files. The first one, `dat_parameters.txt`, contains the optimised parameter values r_i of the agnostic model, parameters r_0 and ∆ of the linear model, and parameters Km, Vmax, n, and r_0 of the sigmoidal model. From these optimised points, the parameter values r_i are computed according to the linear and sigmoidal models described above. The second input file, `dat_functions.txt`, contains the true rational function values, as well as the function values produced by the agnostic, linear, and sigmoidal model.
 
The first step in model selection is to compute the residual sum of squares (RSS) for both, linear and sigmoidal, models. The values are compared to real data for the RSS of rational function values and compared to agnostic values for the RSS of parameter values. A third input file `mh_ranges.txt` provides the ranges of values obtained from Metropolis-Hastings to normalise the residuals. The weight for each parameter value is computed with min-max normalisation and multiplied with the according residual.
 
The script then outputs the Akaike Information Criterion (AIC), AIC = n log(RSS/n) + 2k for n observations and k free parameters, for rational function values, parameter values, and normalised parameter values of both models.
 
We consider the linear model as the better fitting one because of its lower AIC score and validate its absolute quality next. The script computes the coefficient of determination, R^2 = 1 - RSS/TSS with TSS being the total sum of squares, to test the model's predictions. R^2 is only evaluated for parameter values and normalised parameter values. Residual plots and Q-Q plots are created and saved to confirm the normality of residuals.



Now, you should be able to replicate all the results.
This Reproducibility protocol is available on Zenodo including output Metropolis-Hastings result files. 

