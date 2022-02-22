dtmc

const double p;
const double q;
module zeroconf4
  // local state
  s : [0..8] init 0;

  [] s=0 -> q : (s'=1) + (1-q) : (s'=7);
  [] s=1 -> p : (s'=2) + (1-p) : (s'=0);
  [] s=2 -> p : (s'=3) + (1-p) : (s'=0);
  [] s=3 -> p : (s'=4) + (1-p) : (s'=0);
  [] s=4 -> p : (s'=5) + (1-p) : (s'=0);
  [] s=5 -> (s'=6);
  [] s=6 -> (s'=6);
  [] s=7 -> (s'=8);
  [] s=8 -> (s'=8);
  
endmodule

label "err" = s=6;
label "ok" = s=8;