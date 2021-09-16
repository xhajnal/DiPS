// Honeybee mass stinging model. A population of bees a_1, ..., a_n defending the hive decide to sting or not. 
// Published in Hajnal et al., Data-informed parameter synthesis for population Markov chains, HSB 2019 
// Semisynchronous semantics, 2-params 
dtmc 
 
const double p;  //probability to sting at initial condition
const double q;  //probability to sting after sensing the alarm pheromone

module two_param_agents_3
       // ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
       // where success denotes decision to sting, failure the opposite
       // b = 1: 'final'/leaf/BSCC state flag
       a0 : [-1..2] init -1; 
       a1 : [-1..2] init -1; 
       a2 : [-1..2] init -1; 
       b : [0..1] init 0; 

       //  initial transition
       []   a0 = -1 & a1 = -1  & a2 = -1 -> 1.0*p*p*p: (a0'=1) & (a1'=1) & (a2'=1) + 3.0*p*p*(1-p): (a0'=1) & (a1'=1) & (a2'=2) + 3.0*p*(1-p)*(1-p): (a0'=1) & (a1'=2) & (a2'=2) + 1.0*(1-p)*(1-p)*(1-p): (a0'=2) & (a1'=2) & (a2'=2);

       // some ones, some zeros transitions
       []   a0 = 0 & a1 = 0 & a2 = 0 -> (a0'= 0) & (a1'= 0) & (a2'= 0) & (b'=1);
       []   a0 = 1 & a1 = 0 & a2 = 0 -> (a0'= 1) & (a1'= 0) & (a2'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 0 -> (a0'= 1) & (a1'= 1) & (a2'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 1 -> (a0'= 1) & (a1'= 1) & (a2'= 1) & (b'=1);

       // some ones, some twos transitions
       []   a0 = 1 & a1 = 2 & a2 = 2 -> q:(a0'= 1) & (a1'= 1) & (a2'= 2) + 1-q:(a0'= 1) & (a1'= 2) & (a2'= 0);
       []   a0 = 1 & a1 = 1 & a2 = 2 -> q:(a0'= 1) & (a1'= 1) & (a2'= 1) + 1-q:(a0'= 1) & (a1'= 1) & (a2'= 0);

       // some ones, some twos, some zeros transitions
       []   a0 = 1 & a1 = 2 & a2 = 0 -> q: (a0' = 1) & (a1'= 1) & (a2'= 0) + 1-q: (a0' = 1) & (a1'= 0) & (a2'= 0);

       // all twos transition
       []   a0 = 2 & a1 = 2  & a2 = 2 -> (a0'= 0) & (a1'= 0) & (a2'= 0);
endmodule 

rewards "mean" 
       a0 = 0 & a1 = 0 & a2 = 0:0;
       a0 = 1 & a1 = 0 & a2 = 0:1;
       a0 = 1 & a1 = 1 & a2 = 0:2;
       a0 = 1 & a1 = 1 & a2 = 1:3;
endrewards 
rewards "mean_squared" 
       a0 = 0 & a1 = 0 & a2 = 0:0;
       a0 = 1 & a1 = 0 & a2 = 0:1;
       a0 = 1 & a1 = 1 & a2 = 0:4;
       a0 = 1 & a1 = 1 & a2 = 1:9;
endrewards 
rewards "mean_cubed" 
       a0 = 0 & a1 = 0 & a2 = 0:0;
       a0 = 1 & a1 = 0 & a2 = 0:1;
       a0 = 1 & a1 = 1 & a2 = 0:8;
       a0 = 1 & a1 = 1 & a2 = 1:27;
endrewards 
