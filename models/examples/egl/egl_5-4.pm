// randomized protocol for signing contracts Even, Goldreich and Lempel

dtmc

// we now let B to makes his/her choices based on what he/she knows 
// to do this I have added non-determinism to the previous version
// and changed the modules so that only "B's view" is visible
// then reveal the values when B thinks he has an advantage

// to model the non-deterministic behaviour of corrupt party (party B)
// we have a set of possible initial states corresponding to what messages
// he/she tries to over hear when sending - we could do this with nondeterminism 
// but it will just make the model less structured and B has to make the choices 
// at the start anyway since B's view at this point should tell him nothing
// (we use the new construct init...endinit to specify the set of initial states)

// note that certain variables that belong to a party appear in the other party's module
// as this leads to a more structured model - without this PRISM runs out of memory

// note we have included the case when B stops if he/she thinks that the protocol has reached 
// a state where he/she has an advantage

// currently, this model only works for N up to 20

const double p;
const int N=5; // number of pairs of secrets the party sends
const int L=4; // number of bits in each secret

module counter
	
	b : [1..L]; // counter for current bit to be send (used in phases 2 and 3)
	n : [0..max(N-1,1)]; // counter as parties send N messages in a row
	phase : [1..5]; // phase of the protocol
	party : [1..2]; // which party moves
	// 1 first phase of the protocol (sending messages of the form OT(.,.,.,.)
	// 2 and 3 - second phase of the protocol (sending secretes 1..n and n+1..2n respectively)
	// 4 finished the protocol
	
	// FIRST PHASE
	[receiveB] phase=1 & party=1 -> (party'=2); // first A sends a message then B does
	[receiveA] phase=1 & party=2 & n<N-1 -> (party'=1) & (n'=n+1); // after B sends a message we move onto the next message
	[receiveA] phase=1 & party=2 & n=N-1 -> (party'=1) & (phase'=2) & (n'=0); // B has sent his final message - move to next phase
	// SECOND AND THIRD PHASES
	// when A sends
	[receiveB] ((phase)>=(2)&(phase)<=(3))& party=1 & n=0-> (party'=2); // A transmits bth bit of secrets 1..N or N=1..2N
	[receiveA] ((phase)>=(2)&(phase)<=(3))& party=2 & n<N-1-> (n'=n+1); // A transmits bth bit of secrets 1..N or N=1..2N
	[receiveA] ((phase)>=(2)&(phase)<=(3))& party=2 & n=N-1 -> (party'=1) & (n'=1); // finished for party A now move to party B
	// when A sends
	[receiveB] ((phase)>=(2)&(phase)<=(3))& party=1 & n<N-1 & n>0 -> (n'=n+1); // B transmits bth bit of secrets 1..N or N=1..2N
	[receiveB] ((phase)>=(2)&(phase)<=(3))& party=1 & n=N-1 & b<L -> (party'=1) & (n'=0) & (b'=b+1); // finished for party B move to next bit
	[receiveB] phase=2 & party=1 & n=N-1 & b=L -> (phase'=3) & (party'=1) & (n'=0) & (b'=1); // finished for party B move to next phase
	[receiveB] phase=3 & party=1 & n=N-1 & b=L -> (phase'=4); // finished protocol (reveal values)
	
	// FINISHED
	[] phase=4 -> (phase'=4); // loop
	
endmodule

// party A
module partyA
	
	// bi the number of bits of B's ith secret A knows 
	// (keep pairs of secrets together to give a more structured model)
	b0  : [0..L]; b20 : [0..L];
	b1  : [0..L]; b21 : [0..L];
	b2  : [0..L]; b22 : [0..L];
	b3  : [0..L]; b23 : [0..L];
	b4  : [0..L]; b24 : [0..L];
	
	
	// first step (get either secret i or (N-1)+i with equal probability)
	[receiveA] phase=1 & n=0  -> p : (b0'=L)  + (1-p) : (b20'=L);
	[receiveA] phase=1 & n=1  -> p : (b1'=L)  + (1-p) : (b21'=L);
	[receiveA] phase=1 & n=2  -> p : (b2'=L)  + (1-p) : (b22'=L);
	[receiveA] phase=1 & n=3  -> p : (b3'=L)  + (1-p) : (b23'=L);
	[receiveA] phase=1 & n=4  -> p : (b4'=L)  + (1-p) : (b24'=L);
	// second step (secrets 0,...,N-1)
	[receiveA] phase=2 & n=0  -> (b0'=min(b0+1,L));
	[receiveA] phase=2 & n=1  -> (b1'=min(b1+1,L));
	[receiveA] phase=2 & n=2  -> (b2'=min(b2+1,L));
	[receiveA] phase=2 & n=3  -> (b3'=min(b3+1,L));
	[receiveA] phase=2 & n=4  -> (b4'=min(b4+1,L));
	// second step (secrets N,...,2N-1)
	[receiveA] phase=3 & n=0  -> (b20'=min(b20+1,L));
	[receiveA] phase=3 & n=1  -> (b21'=min(b21+1,L));
	[receiveA] phase=3 & n=2  -> (b22'=min(b22+1,L));
	[receiveA] phase=3 & n=3  -> (b23'=min(b23+1,L));
	[receiveA] phase=3 & n=4  -> (b24'=min(b24+1,L));

endmodule

// construct module for party B through renaming
module partyB=partyA[b0 =a0 ,b1 =a1 ,b2 =a2 ,b3 =a3 ,b4 =a4,
                     b20=a20,b21=a21,b22=a22,b23=a23,b24=a24,
                     receiveA=receiveB] 
endmodule

// formulae
formula kB = ( (a0=L  & a20=L)
			 | (a1=L  & a21=L)
			 | (a2=L  & a22=L)
			 | (a3=L  & a23=L)
			 | (a4=L  & a24=L));

formula kA = ( (b0=L  & b20=L)
			 | (b1=L  & b21=L)
			 | (b2=L  & b22=L)
			 | (b3=L  & b23=L)
			 | (b4=L  & b24=L));

// labels
label "knowB" = kB;
label "knowA" = kA;

// reward structures

// messages from B that A needs to knows a pair once B knows a pair
rewards "messages_A_needs"
	[receiveA] kB & !kA : 1;
endrewards

// messages from A that B needs to knows a pair once A knows a pair
rewards "messages_B_needs"
	[receiveA] kA & !kB : 1;
endrewards
