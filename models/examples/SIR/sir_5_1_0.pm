// SIR (Susceptible, Infected, Recovered) model
// @author Huy Phung 
// @edited Matej Hajnal
// available from https://github.com/huypn12/bbeess-py/tree/main/data/prism
// generated using https://github.com/huypn12/bbeess-py/blob/main/data/model_generator/sir_generator.py

dtmc
  const double alpha;
  const double beta;

module sir_5_1_0 
	s : [0..6] init 5;
	i : [0..6] init 1;
	r : [0..6] init 0;
 
	[] s=5 & i=1 & r=0 -> ((5*alpha)/(9*alpha+6*beta)):(s'=4) & (i'=2) & (r'=0) + ((1*beta)/(9*alpha+6*beta)):(s'=5) & (i'=0) & (r'=1) + ((4*alpha+5*beta)/(9*alpha+6*beta)):(s'=5) & (i'=1) & (r'=0);
	[] s=4 & i=2 & r=0 -> ((8*alpha)/(9*alpha+6*beta)):(s'=3) & (i'=3) & (r'=0) + ((2*beta)/(9*alpha+6*beta)):(s'=4) & (i'=1) & (r'=1) + ((1*alpha+4*beta)/(9*alpha+6*beta)):(s'=4) & (i'=2) & (r'=0);
	[] s=3 & i=3 & r=0 -> ((9*alpha)/(9*alpha+6*beta)):(s'=2) & (i'=4) & (r'=0) + ((3*beta)/(9*alpha+6*beta)):(s'=3) & (i'=2) & (r'=1) + ((3*beta)/(9*alpha+6*beta)):(s'=3) & (i'=3) & (r'=0);
	[] s=2 & i=4 & r=0 -> ((8*alpha)/(9*alpha+6*beta)):(s'=1) & (i'=5) & (r'=0) + ((4*beta)/(9*alpha+6*beta)):(s'=2) & (i'=3) & (r'=1) + ((1*alpha+2*beta)/(9*alpha+6*beta)):(s'=2) & (i'=4) & (r'=0);
	[] s=1 & i=5 & r=0 -> ((5*alpha)/(9*alpha+6*beta)):(s'=0) & (i'=6) & (r'=0) + ((5*beta)/(9*alpha+6*beta)):(s'=1) & (i'=4) & (r'=1) + ((4*alpha+1*beta)/(9*alpha+6*beta)):(s'=1) & (i'=5) & (r'=0);
	[] s=0 & i=6 & r=0 -> ((6*beta)/(9*alpha+6*beta)):(s'=0) & (i'=5) & (r'=1) + ((9*alpha)/(9*alpha+6*beta)):(s'=0) & (i'=6) & (r'=0);
	[] s=0 & i=5 & r=1 -> ((5*beta)/(9*alpha+6*beta)):(s'=0) & (i'=4) & (r'=2) + ((9*alpha+1*beta)/(9*alpha+6*beta)):(s'=0) & (i'=5) & (r'=1);
	[] s=0 & i=4 & r=2 -> ((4*beta)/(9*alpha+6*beta)):(s'=0) & (i'=3) & (r'=3) + ((9*alpha+2*beta)/(9*alpha+6*beta)):(s'=0) & (i'=4) & (r'=2);
	[] s=0 & i=3 & r=3 -> ((3*beta)/(9*alpha+6*beta)):(s'=0) & (i'=2) & (r'=4) + ((9*alpha+3*beta)/(9*alpha+6*beta)):(s'=0) & (i'=3) & (r'=3);
	[] s=0 & i=2 & r=4 -> ((2*beta)/(9*alpha+6*beta)):(s'=0) & (i'=1) & (r'=5) + ((9*alpha+4*beta)/(9*alpha+6*beta)):(s'=0) & (i'=2) & (r'=4);
	[] s=0 & i=1 & r=5 -> ((1*beta)/(9*alpha+6*beta)):(s'=0) & (i'=0) & (r'=6) + ((9*alpha+5*beta)/(9*alpha+6*beta)):(s'=0) & (i'=1) & (r'=5);
	[] s=1 & i=4 & r=1 -> ((4*alpha)/(9*alpha+6*beta)):(s'=0) & (i'=5) & (r'=1) + ((4*beta)/(9*alpha+6*beta)):(s'=1) & (i'=3) & (r'=2) + ((5*alpha+2*beta)/(9*alpha+6*beta)):(s'=1) & (i'=4) & (r'=1);
	[] s=1 & i=3 & r=2 -> ((3*alpha)/(9*alpha+6*beta)):(s'=0) & (i'=4) & (r'=2) + ((3*beta)/(9*alpha+6*beta)):(s'=1) & (i'=2) & (r'=3) + ((6*alpha+3*beta)/(9*alpha+6*beta)):(s'=1) & (i'=3) & (r'=2);
	[] s=1 & i=2 & r=3 -> ((2*alpha)/(9*alpha+6*beta)):(s'=0) & (i'=3) & (r'=3) + ((2*beta)/(9*alpha+6*beta)):(s'=1) & (i'=1) & (r'=4) + ((7*alpha+4*beta)/(9*alpha+6*beta)):(s'=1) & (i'=2) & (r'=3);
	[] s=1 & i=1 & r=4 -> ((1*alpha)/(9*alpha+6*beta)):(s'=0) & (i'=2) & (r'=4) + ((1*beta)/(9*alpha+6*beta)):(s'=1) & (i'=0) & (r'=5) + ((8*alpha+5*beta)/(9*alpha+6*beta)):(s'=1) & (i'=1) & (r'=4);
	[] s=2 & i=3 & r=1 -> ((6*alpha)/(9*alpha+6*beta)):(s'=1) & (i'=4) & (r'=1) + ((3*beta)/(9*alpha+6*beta)):(s'=2) & (i'=2) & (r'=2) + ((3*alpha+3*beta)/(9*alpha+6*beta)):(s'=2) & (i'=3) & (r'=1);
	[] s=2 & i=2 & r=2 -> ((4*alpha)/(9*alpha+6*beta)):(s'=1) & (i'=3) & (r'=2) + ((2*beta)/(9*alpha+6*beta)):(s'=2) & (i'=1) & (r'=3) + ((5*alpha+4*beta)/(9*alpha+6*beta)):(s'=2) & (i'=2) & (r'=2);
	[] s=2 & i=1 & r=3 -> ((2*alpha)/(9*alpha+6*beta)):(s'=1) & (i'=2) & (r'=3) + ((1*beta)/(9*alpha+6*beta)):(s'=2) & (i'=0) & (r'=4) + ((7*alpha+5*beta)/(9*alpha+6*beta)):(s'=2) & (i'=1) & (r'=3);
	[] s=3 & i=2 & r=1 -> ((6*alpha)/(9*alpha+6*beta)):(s'=2) & (i'=3) & (r'=1) + ((2*beta)/(9*alpha+6*beta)):(s'=3) & (i'=1) & (r'=2) + ((3*alpha+4*beta)/(9*alpha+6*beta)):(s'=3) & (i'=2) & (r'=1);
	[] s=3 & i=1 & r=2 -> ((3*alpha)/(9*alpha+6*beta)):(s'=2) & (i'=2) & (r'=2) + ((1*beta)/(9*alpha+6*beta)):(s'=3) & (i'=0) & (r'=3) + ((6*alpha+5*beta)/(9*alpha+6*beta)):(s'=3) & (i'=1) & (r'=2);
	[] s=4 & i=1 & r=1 -> ((4*alpha)/(9*alpha+6*beta)):(s'=3) & (i'=2) & (r'=1) + ((1*beta)/(9*alpha+6*beta)):(s'=4) & (i'=0) & (r'=2) + ((5*alpha+5*beta)/(9*alpha+6*beta)):(s'=4) & (i'=1) & (r'=1);
	[] s=0 & i=0 & r=6 -> (s'=0) & (i'=0) & (r'=6);
	[] s=1 & i=0 & r=5 -> (s'=1) & (i'=0) & (r'=5);
	[] s=2 & i=0 & r=4 -> (s'=2) & (i'=0) & (r'=4);
	[] s=3 & i=0 & r=3 -> (s'=3) & (i'=0) & (r'=3);
	[] s=4 & i=0 & r=2 -> (s'=4) & (i'=0) & (r'=2);
	[] s=5 & i=0 & r=1 -> (s'=5) & (i'=0) & (r'=1); 
endmodule

label "bscc_0_0_6" = s=0 & i=0 & r=6 ;
label "bscc_1_0_5" = s=1 & i=0 & r=5 ;
label "bscc_2_0_4" = s=2 & i=0 & r=4 ;
label "bscc_3_0_3" = s=3 & i=0 & r=3 ;
label "bscc_4_0_2" = s=4 & i=0 & r=2 ;
label "bscc_5_0_1" = s=5 & i=0 & r=1 ;

// Number of states: 27
// Number of BSCCs: 6