// Honeybee mass stinging model. A population of bees a_1, ..., a_n defending the hive decide to sting or not.
// Semisynchronous semantics, multiparametric 
// Published in Hajnal et al., Data-informed parameter synthesis for population Markov chains, HSB 2019 
dtmc 
 
const double p;  //probability to sting at initial condition
const double q1;
const double q2;
const double q3;

module multi_param_agents_4
       // ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
       // where success denotes decision to sting, failure the opposite
       // b = 1: 'final'/leaf/BSCC state flag
       a0 : [-1..2] init -1; 
       a1 : [-1..2] init -1; 
       a2 : [-1..2] init -1; 
       a3 : [-1..2] init -1; 
       b : [0..1] init 0; 

       //  initial transition
       []   a0 = -1 & a1 = -1  & a2 = -1  & a3 = -1 -> 1.0*p*p*p*p: (a0'=1) & (a1'=1) & (a2'=1) & (a3'=1) + 4.0*p*p*p*(1-p): (a0'=1) & (a1'=1) & (a2'=1) & (a3'=2) + 6.0*p*p*(1-p)*(1-p): (a0'=1) & (a1'=1) & (a2'=2) & (a3'=2) + 4.0*p*(1-p)*(1-p)*(1-p): (a0'=1) & (a1'=2) & (a2'=2) & (a3'=2) + 1.0*(1-p)*(1-p)*(1-p)*(1-p): (a0'=2) & (a1'=2) & (a2'=2) & (a3'=2);

       // some ones, some zeros transitions
       []   a0 = 0 & a1 = 0 & a2 = 0 & a3 = 0 -> (a0'= 0) & (a1'= 0) & (a2'= 0) & (a3'= 0) & (b'=1);
       []   a0 = 1 & a1 = 0 & a2 = 0 & a3 = 0 -> (a0'= 1) & (a1'= 0) & (a2'= 0) & (a3'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 0 & a3 = 0 -> (a0'= 1) & (a1'= 1) & (a2'= 0) & (a3'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 0 -> (a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 -> (a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (b'=1);

       // some ones, some twos transitions
       []   a0 = 1 & a1 = 2 & a2 = 2 & a3 = 2 -> q1:(a0'= 1) & (a1'= 1) & (a2'= 2) & (a3'= 2) + 1-q1:(a0'= 1) & (a1'= 2) & (a2'= 2) & (a3'= 0);
       []   a0 = 1 & a1 = 1 & a2 = 2 & a3 = 2 -> q2:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 2) + 1-q2:(a0'= 1) & (a1'= 1) & (a2'= 2) & (a3'= 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 2 -> q3:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) + 1-q3:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 0);

       // some ones, some twos, some zeros transitions
       []   a0 = 1 & a1 = 2 & a2 = 0 & a3 = 0 -> q1: (a0' = 1) & (a1' = 1) & (a2' = 0) & (a3' = 0) + 1-q1: (a0' = 1) & (a1' = 0) & (a2' = 0) & (a3' = 0);
       []   a0 = 1 & a1 = 2 & a2 = 2 & a3 = 0 -> q1: (a0' = 1) & (a1' = 1) & (a2' = 2) & (a3' = 0) + 1-q1: (a0' = 1) & (a1' = 2) & (a2' = 0) & (a3' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 2 & a3 = 0 -> q2: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 0) + 1-q2: (a0' = 1) & (a1' = 1) & (a2' = 0) & (a3' = 0);

       // all twos transition
       []   a0 = 2 & a1 = 2  & a2 = 2  & a3 = 2 -> (a0'= 0) & (a1'= 0) & (a2'= 0) & (a3'= 0);
endmodule 

rewards "mean" 
       a0 = 0 & a1 = 0 & a2 = 0 & a3 = 0:0;
       a0 = 1 & a1 = 0 & a2 = 0 & a3 = 0:1;
       a0 = 1 & a1 = 1 & a2 = 0 & a3 = 0:2;
       a0 = 1 & a1 = 1 & a2 = 1 & a3 = 0:3;
       a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1:4;
endrewards 
rewards "mean_squared" 
       a0 = 0 & a1 = 0 & a2 = 0 & a3 = 0:0;
       a0 = 1 & a1 = 0 & a2 = 0 & a3 = 0:1;
       a0 = 1 & a1 = 1 & a2 = 0 & a3 = 0:4;
       a0 = 1 & a1 = 1 & a2 = 1 & a3 = 0:9;
       a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1:16;
endrewards 
rewards "mean_cubed" 
       a0 = 0 & a1 = 0 & a2 = 0 & a3 = 0:0;
       a0 = 1 & a1 = 0 & a2 = 0 & a3 = 0:1;
       a0 = 1 & a1 = 1 & a2 = 0 & a3 = 0:8;
       a0 = 1 & a1 = 1 & a2 = 1 & a3 = 0:27;
       a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1:64;
endrewards 
