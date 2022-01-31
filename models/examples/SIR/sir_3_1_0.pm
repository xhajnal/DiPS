// SIR (Susceptible, Infected, Recovered) model
// @author Huy Phung 
// available from https://github.com/huypn12/bbeess-py/tree/main/data/prism
// generated using https://github.com/huypn12/bbeess-py/blob/main/data/model_generator/sir_generator.py

dtmc
  const double alpha;
  const double beta;

module sir_3_1_0 
	s : [0..4] init 3;
	i : [0..4] init 1;
	r : [0..4] init 0;
 
	[] s=3 & i=1 & r=0 -> ((3*alpha)/(3*alpha+4*beta)):(s'=2) & (i'=2) & (r'=0) + ((1*beta)/(3*alpha+4*beta)):(s'=3) & (i'=0) & (r'=1) + ((3*beta)/(3*alpha+4*beta)):(s'=3) & (i'=1) & (r'=0);
	[] s=2 & i=2 & r=0 -> ((2*alpha)/(3*alpha+4*beta)):(s'=1) & (i'=3) & (r'=0) + ((2*beta)/(3*alpha+4*beta)):(s'=2) & (i'=1) & (r'=1) + ((1*alpha+2*beta)/(3*alpha+4*beta)):(s'=2) & (i'=2) & (r'=0);
	[] s=1 & i=3 & r=0 -> ((1*alpha)/(3*alpha+4*beta)):(s'=0) & (i'=4) & (r'=0) + ((3*beta)/(3*alpha+4*beta)):(s'=1) & (i'=2) & (r'=1) + ((2*alpha+1*beta)/(3*alpha+4*beta)):(s'=1) & (i'=3) & (r'=0);
	[] s=0 & i=4 & r=0 -> ((4*beta)/(3*alpha+4*beta)):(s'=0) & (i'=3) & (r'=1) + ((3*alpha)/(3*alpha+4*beta)):(s'=0) & (i'=4) & (r'=0);
	[] s=0 & i=3 & r=1 -> ((3*beta)/(3*alpha+4*beta)):(s'=0) & (i'=2) & (r'=2) + ((3*alpha+1*beta)/(3*alpha+4*beta)):(s'=0) & (i'=3) & (r'=1);
	[] s=0 & i=2 & r=2 -> ((2*beta)/(3*alpha+4*beta)):(s'=0) & (i'=1) & (r'=3) + ((3*alpha+2*beta)/(3*alpha+4*beta)):(s'=0) & (i'=2) & (r'=2);
	[] s=0 & i=1 & r=3 -> ((1*beta)/(3*alpha+4*beta)):(s'=0) & (i'=0) & (r'=4) + ((3*alpha+3*beta)/(3*alpha+4*beta)):(s'=0) & (i'=1) & (r'=3);
	[] s=1 & i=2 & r=1 -> ((1*alpha)/(3*alpha+4*beta)):(s'=0) & (i'=3) & (r'=1) + ((2*beta)/(3*alpha+4*beta)):(s'=1) & (i'=1) & (r'=2) + ((2*alpha+2*beta)/(3*alpha+4*beta)):(s'=1) & (i'=2) & (r'=1);
	[] s=1 & i=1 & r=2 -> ((1*alpha)/(3*alpha+4*beta)):(s'=0) & (i'=2) & (r'=2) + ((1*beta)/(3*alpha+4*beta)):(s'=1) & (i'=0) & (r'=3) + ((2*alpha+3*beta)/(3*alpha+4*beta)):(s'=1) & (i'=1) & (r'=2);
	[] s=2 & i=1 & r=1 -> ((2*alpha)/(3*alpha+4*beta)):(s'=1) & (i'=2) & (r'=1) + ((1*beta)/(3*alpha+4*beta)):(s'=2) & (i'=0) & (r'=2) + ((1*alpha+3*beta)/(3*alpha+4*beta)):(s'=2) & (i'=1) & (r'=1);
	[] s=0 & i=0 & r=4 -> (s'=0) & (i'=0) & (r'=4);
	[] s=1 & i=0 & r=3 -> (s'=1) & (i'=0) & (r'=3);
	[] s=2 & i=0 & r=2 -> (s'=2) & (i'=0) & (r'=2);
	[] s=3 & i=0 & r=1 -> (s'=3) & (i'=0) & (r'=1); 
endmodule

label "bscc_0_0_4" = s=0 & i=0 & r=4 ;
label "bscc_1_0_3" = s=1 & i=0 & r=3 ;
label "bscc_2_0_2" = s=2 & i=0 & r=2 ;
label "bscc_3_0_1" = s=3 & i=0 & r=1 ;

// Number of states: 14
// Number of BSCCs: 4