dtmc 
 
const double p;
const double q;

module two_param_agents_3
       // ai - state of agent i:  -1:init 0:total_failure 1:success 2:failure_after_first_attempt
       a0 : [-1..2] init -1; 
       a1 : [-1..2] init -1; 
       a2 : [-1..2] init -1; 
       b : [0..1] init 0; 

       //  initial transitions
       // some -1, some 1 transitions
       []   a0 = -1 & a1 = -1  & a2 = -1 -> p: (a0' = -1) & (a1' = -1) & (a2' = 1) + 1-p: (a0' = -1) & (a1'= -1) & (a2' = 2);
       []   a0 = -1 & a1 = -1  & a2 = 1 -> p: (a0' = -1) & (a1' = 1) & (a2' = 1) + 1-p: (a0' = -1) & (a1' = 1) & (a2' = 2);
       []   a0 = -1 & a1 = 1  & a2 = 1 -> p: (a0' = 1) & (a1' = 1) & (a2' = 1) + 1-p: (a0' = 1) & (a1' = 1) & (a2' = 2);

       // some -1, some 2 transitions
       []   a0 = -1 & a1 = -1 & a2 = 2-> p: (a0' = -1) & (a1' = 1) & (a2' = 2) + 1-p: (a0' = -1) & (a1' = 2) & (a2' = 2);
       []   a0 = -1 & a1 = 2 & a2 = 2-> p: (a0' = 1) & (a1' = 2) & (a2' = 2) + 1-p: (a0' = 2) & (a1' = 2) & (a2' = 2);

       // some -1, some 1, some 2 transitions
       []   a0 = -1 & a1 = 1 & a2 = 2 -> p: (a0' = 1) & (a1'= 1) & (a2'= 2) + 1-p: (a0' = 1) & (a1'= 2) & (a2'= 2);

       //  not initial transitions
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
