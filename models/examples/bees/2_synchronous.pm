// New Honeybee mass stinging model. A population of bees a_1, ..., a_n defending the hive decide to sting or not.
// Synchronous semantics, multiparametric 
dtmc 
 
r_i - probability to sting when i amount of pheromone present 
const double r_0;
const double r_1;

module multi_param_bee_agents_2
       // Two types of states are present, b=0 and b=1, where b=0 flags inner/decision/nonleaf state and b=1 represents 'final'/leaf/BSCC state
       // b = 0 => ai - state of agent i: 3:init, 1:success, (stinging), -j: failure (not stinging) when j amount of pheromone present 
       // b = 1 => ai - state of the agent i: 0: not stinging, 1:stinging
       // this duality of ai meaning serves back-compatibility with properties from the old models
       a0 : [-1..3] init 3; 
       a1 : [-1..3] init 3; 
       b : [0..1] init 0; 

       //  initial transition
       []   a0 = 3 & a1 = 3  & b = 0 -> 1.0*r_0*r_0: (a0'=1) & (a1'=1) + 2.0*r_0*(1-r_0): (a0'=1) & (a1'=0) + 1.0*(1-r_0)*(1-r_0): (a0'=0) & (a1'=0);

       // some ones, some nonpositive final transitions
       []   a0 = 0 & a1 = 0 & b = 0  -> (a0'= 0) & (a1'= 0) & (b'=1);
       []   a0 = 1 & a1 = -1 & b = 0  -> (a0'= 1) & (a1'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & b = 0  -> (a0'= 1) & (a1'= 1) & (b'=1);

       // some ones, some nonpositive transitions
       []   a0 = 1 & a1 = 0 & b = 0  -> 1.0*(1-(r_1 - r_0)/(1 - r_0)): (a0'=1) & (a1'=-1) + 1.0* ((r_1 - r_0)/(1 - r_0)): (a0'=1) & (a1'=1);

       // self-loops in BSCCs
       []   b=1 -> 1:(b'=1);

endmodule 

rewards "mean" 
       a0 = 0 & a1 = 0:0;
       a0 = 1 & a1 = -1:1;
       a0 = 1 & a1 = 1:2;
endrewards 
rewards "mean_squared" 
       a0 = 0 & a1 = 0:0;
       a0 = 1 & a1 = -1:1;
       a0 = 1 & a1 = 1:4;
endrewards 
rewards "mean_cubed" 
       a0 = 0 & a1 = 0:0;
       a0 = 1 & a1 = -1:1;
       a0 = 1 & a1 = 1:8;
endrewards 
