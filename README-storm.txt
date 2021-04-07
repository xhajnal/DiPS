If you have tried to install Storm (with stormpy) on Ubuntu and failed you can try to follow these instructions.
Disclaimer: This is not official Storm manual. 

Prerequisities: 
- python 3.8
- git
- conda (optional)


0. (Optional) Use virtual enviroment for python 
	0.1 Install conda
		https://docs.conda.io/en/latest/
	0.2. Create a python virtual enviroment
		conda create --name py38 python=3.8
	0.3. make alias (use gedit or other text editor you prefer instead of sublime)
		$ sublime ~/.bashrc

		insert this line:
		alias python38="conda activate py38"

	0.4. activate env
		python38

1. switch gcc and g++ compiler to 6 (Storm recommended setting)
	$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
	$ sudo apt-get update
	$ sudo apt install gcc-6
	$ sudo apt-get install g++-6
	$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 6
	$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 6
	$ sudo update-alternatives --config gcc
	# select 6
	
	$ sudo update-alternatives --config g++
	# select 6

	$ gcc --version
	gcc (Ubuntu 6.5.0-2ubuntu1~18.04) 6.5.0 20181026
	$ g++ --version
	g++ (Ubuntu 6.5.0-2ubuntu1~18.04) 6.5.0 20181026

2. OBTAIN PREREQUISITIES
	2.0 $ python38
	2.1 related packages
		$ sudo apt-get install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev


3. Obtain carl (http://smtrat.github.io/carl/ubuntu_1404_lts.html) with configuration for Prophesy https://moves-rwth.github.io/prophesy/installation.html
	$ git clone https://github.com/smtrat/carl.git 
	$ cd carl/

	## check you have master branch
	$ git branch
	* master14

	$ mkdir build
	$ cd build/
	$ cmake .. -DUSE_CLN_NUMBERS=ON -DUSE_GINAC=ON
	$ make lib_carl 
	$ make 
	$ make test
	$ make doc


4. Obtain carl-parser master14 (https://github.com/ths-rwth/carl-parser)
	4.1 obtain carl-parser prerequisities
	$ sudo apt install maven
	$ sudo apt install uuid-dev

	$ git clone -b master14 https://github.com/ths-rwth/carl-parser.git
	$ cd carl-parser
	
	## check you have stable branch
	$ git branch
	* master14

	$ mkdir build && cd build
	$ cmake ..
	$ make


5. Obtain pycarl
	$ git clone https://github.com/moves-rwth/pycarl
	$ cd pycarl
	$ python setup.py develop


6. Finally, obtain Storm - compile (https://www.stormchecker.org/documentation/obtain-storm/dependencies.html)
	6.1 obtain Storm prerequisities
		Already done in step 2.

	6.2 obtain Carl
		Already done in step 3

	6.3 obtain Boost (https://github.com/boostorg/wiki/wiki/Getting-Started%3A-Overview)
		# probably already done
		# download boost -> Git  from (https://www.boost.org/users/history/version_1_75_0.html)
		# unzip boost
		$ cd boost
		$ ./bootstrap.sh
		$ ./b2

	6.4 build Storm
		$ git clone -b stable https://github.com/moves-rwth/storm.git
		$ cd storm

		## check you have stable branch
		$ git branch
		* stable

		$ mkdir build && cd build
		$ cmake ..
		
		## use number of cores (int) instead of <cores>, eg. 4
		$ make -j <cores>

		$ sublime ~/.bashrc
		# add this line, where you replace STORM_DIR with path to Storm
		# export PATH=$PATH:STORM_DIR/build/bin
		
		$ make check
		## reopen terminal or $source ~/.bashrc
		$ storm --version
		
		## USE STORM - (please correct the paths to models)
		$ time storm-pars --prism /home/DiPS/models/examples/bee/semisynchronous_2_bees.pm --prop 'P>0.0  [ F (a0=1)&(a1=0)&(b=1)]' --region '0.0<=p<=1.0,0.0<=q<=1.0' --refine 0.05
		$ time storm-pars --prism /home/DiPS/models/examples/bee/semisynchronous_2_bees.pm --prop 'P>0.0  [ F (a0=1)&(a1=0)&(b=1)]' --region '0.0<=p<=1.0,0.0<=q<=1.0' --refine 0.05 --region:engine "validatingpl" --region:engine "validatingpl"
		

7. (Optional) Obtain stormpy - (https://moves-rwth.github.io/stormpy/installation.html#installation)
	$ git clone https://github.com/moves-rwth/stormpy.git 
	$ cd stormpy

	## check you have master branch
	$ git branch
	* master

	$ python setup.py test
	$ python
	>> import stormpy
