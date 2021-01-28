// Knuth's model of a fair die using three coins
// author Tatjana Petrov
dtmc

const double p1;
const double p2;
const double p3=0.5;


module die

	// local state
	s : [0..7] init 0;
	// value of the dice
	d : [0..6] init 0;
	// fake variable to provide reachability rewards
	r : [0..1] init 0; 
	
	[] s=0 -> p1 : (s'=1) + (1-p1) : (s'=2);
	[] s=1 -> p2 : (s'=3) + (1-p2) : (s'=4);
	[] s=2 -> p2 : (s'=5) + (1-p2) : (s'=6);
	[] s=3 -> p3 : (s'=1) + (1-p3) : (s'=7) & (d'=1);
	[] s=4 -> p3 : (s'=7) & (d'=2) + (1-p3) : (s'=7) & (d'=3);
	[] s=5 -> p3 : (s'=7) & (d'=4) + (1-p3) : (s'=7) & (d'=5);
	[] s=6 -> p3 : (s'=2) + (1-p3) : (s'=7) & (d'=6);
	// fake transition to provide reachability rewards
	[r] s=7 -> 1: (r'=1);
	[]  r=1 -> 1: (r'=1);
	
endmodule

rewards "side_rolled"
	[r] d=1 : 1;
	[r] d=2 : 2;
	[r] d=3 : 3;
	[r] d=4 : 4;
	[r] d=5 : 5;
	[r] d=6 : 6;
endrewards

label "one" = s=7&d=1;
label "two" = s=7&d=2;
label "three" = s=7&d=3;
label "four" = s=7&d=4;
label "five" = s=7&d=5;
label "six" = s=7&d=6;
label "done" = s=7;





