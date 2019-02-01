import sys, os, platform,struct
cwd = os.getcwd()

# z3_path = "/home/matej/z3/build" #Freya


# os.environ["PATH"]=os.environ["PATH"]":/home/matej/z3/"


z3_path = "C:/z3py/z3-4.6.0-x64-win/bin/python"

os.environ["PYTHONPATH"]=z3_path
os.environ["Z3_LIBRARY_PATH"]=z3_path
os.environ["Z3_LIBRARY_DIRS"]=z3_path


os.chdir(z3_path)
print(os.getcwd())
from z3.z3 import Real
os.chdir(cwd) 
p = Real('p')




import sys, os, platform,struct
cwd = os.getcwd()

# z3_path = "/home/matej/z3/build" #Freya
z3_path = "C:/z3py/z3-4.6.0-x64-win/bin"  #SPICY



# os.environ["PATH"]=os.environ["PATH"]:"/home/matej/z3/" #Freya
os.environ["PATH"]=os.environ["PATH"]+";"+z3_path  #SPICY
z3_path = z3_path+"/python"
os.environ["PYTHONPATH"]=z3_path
os.environ["Z3_LIBRARY_PATH"]=z3_path
os.environ["Z3_LIBRARY_DIRS"]=z3_path

os.chdir(z3_path)
print(os.getcwd())
from z3 import *
os.chdir(cwd) 
p = Real('p')



PATH/LD_LIBRARY_PATH/DYLD_LIBRARY_PATH