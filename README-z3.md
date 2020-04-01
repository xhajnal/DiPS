1. pip installation
    
    If you didn't install all the requirements, please do.  
    Otherwise, please
    - uninstall the z3 libraries `pip uninstall z3-solver`, `pip uninstall z3` and 
    - install them again in this order z3 first, z3-solver second - `pip install z3`, `pip install z3-solver`
    
2. manual installation 
	if the problem remains,  
	- uninstall z3 libraries `pip uninstall z3-solver`, `pip uninstall z3` again and install respective library manually - prebuild version [here](https://github.com/Z3Prover/z3/releases), or to be build version [here](https://github.com/Z3Prover/z3).
	- add the path of the subfolder `bin/python` to item z3_path, in section paths in config.ini in the main folder of this project

3. custom installation, 
	if the problem still remains, please create an issue, or contact us.

<!---
The Z3 Python frontend directory must be in your PYTHONPATH environment variable. Z3Py will automatically search for the Z3 library (z3.dll (Windows), libz3.so (Linux), or libz3.dylib (OSX)). You may also initialize Z3Py manually using the command:

# init("z3.dll")
-->
