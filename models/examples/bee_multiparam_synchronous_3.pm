dtmc 
 
const double r_0;
const double r_1;
const double r_2;

module multi_param_bee_agents_3
       // Two types of states are present, b=0 and b=1, where b=0 flags inner/decision/nonleaf state and b=1 represents 'final'/leaf/BSCC state
       // b = 0 => ai - state of agent i: 3:init, 1:success, (stinging), -j: failure (not stinging) when j amount of pheromone present 
       // b = 1 => ai - state of the agent i: 0: not stinging, 1:stinging
       // this duality of ai meaning serves back-compatibility with properties from the old models
       a0 : [-2..3] init 3; 
       a1 : [-2..3] init 3; 
       a2 : [-2..3] init 3; 
       b : [0..1] init 0; 

       //  initial transition
       []   a0 = 3 & a1 = 3  & a2 = 3  & b = 0 -> 1.0*r_0*r_0*r_0: (a0'=1) & (a1'=1) & (a2'=1) + 3.0*r_0*r_0*(1-r_0): (a0'=1) & (a1'=1) & (a2'=0) + 3.0*r_0*(1-r_0)*(1-r_0): (a0'=1) & (a1'=0) & (a2'=0) + 1.0*(1-r_0)*(1-r_0)*(1-r_0): (a0'=0) & (a1'=0) & (a2'=0);

       // some ones, some nonpositive final transitions
       []   a0 = 0 & a1 = 0 & a2 = 0 & b = 0  -> (a0'= 0) & (a1'= 0) & (a2'= 0) & (b'=1);
       []   a0 = 1 & a1 = -1 & a2 = -1 & b = 0  -> (a0'= 1) & (a1'= 0) & (a2'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = -2 & b = 0  -> (a0'= 1) & (a1'= 1) & (a2'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 1 & b = 0  -> (a0'= 1) & (a1'= 1) & (a2'= 1) & (b'=1);

       // some ones, some nonpositive transitions
       []   a0 = 1 & a1 = 0 & a2 = 0 & b = 0  -> 1.0*(1-(r_1 - r_0)/(1 - r_0))*(1-(r_1 - r_0)/(1 - r_0)): (a0'=1) & (a1'=-1) & (a2'=-1) + 2.0* ((r_1 - r_0)/(1 - r_0))*(1-(r_1 - r_0)/(1 - r_0)): (a0'=1) & (a1'=1) & (a2'=-1) + 1.0* ((r_1 - r_0)/(1 - r_0))* ((r_1 - r_0)/(1 - r_0)): (a0'=1) & (a1'=1) & (a2'=1);
       []   a0 = 1 & a1 = 1 & a2 = 0 & b = 0  -> 1.0*(1-(r_2 - r_0)/(1 - r_0)): (a0'=1) & (a1'=1) & (a2'=-2) + 1.0* ((r_2 - r_0)/(1 - r_0)): (a0'=1) & (a1'=1) & (a2'=1);
       []   a0 = 1 & a1 = 1 & a2 = -1 & b = 0  -> 1.0*(1-(r_2 - r_1)/(1 - r_1)): (a0'=1) & (a1'=1) & (a2'=-2) + 1.0* ((r_2 - r_1)/(1 - r_1)): (a0'=1) & (a1'=1) & (a2'=1);

       // selfloops in BSCCs
       []   b=1 -> 1:(b'=1);

endmodule 

rewards "mean" 
       a0 = 0 & a1 = 0 & a2 = 0:0;
       a0 = 1 & a1 = -1 & a2 = -1:1;
       a0 = 1 & a1 = 1 & a2 = -2:2;
       a0 = 1 & a1 = 1 & a2 = 1:3;
endrewards 
rewards "mean_squared" 
       a0 = 0 & a1 = 0 & a2 = 0:0;
       a0 = 1 & a1 = -1 & a2 = -1:1;
       a0 = 1 & a1 = 1 & a2 = -2:4;
       a0 = 1 & a1 = 1 & a2 = 1:9;
endrewards 
rewards "mean_cubed" 
       a0 = 0 & a1 = 0 & a2 = 0:0;
       a0 = 1 & a1 = -1 & a2 = -1:1;
       a0 = 1 & a1 = 1 & a2 = -2:8;
       a0 = 1 & a1 = 1 & a2 = 1:27;
endrewards 
