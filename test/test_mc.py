import unittest
from src.mc import *
import test.admin as admin
cwd = os.getcwd()

try:
    os.mkdir("tmp")
except FileExistsError:
    pass
try:
    os.mkdir("tmp/results")
except FileExistsError:
    pass
try:
    os.mkdir("tmp/results/prism_results")
except FileExistsError:
    pass
try:
    os.mkdir("tmp/results/storm_results")
except FileExistsError:
    pass

model_dir = os.path.join(cwd, "models")
properties_dir = os.path.join(cwd, "properties")
prism_results = os.path.join(cwd, "tmp/results/prism_results")
storm_results = os.path.join(cwd, "tmp/results/storm_results")


class MyTestCase(unittest.TestCase):
    def test_changing_javaheap(self):
        print(colored('Test_changing_javaheap on Windows', 'blue'))
        # if not admin.isUserAdmin():
        #   admin.runAsAdmin()
        if system().startswith("win"):
            a = (set_java_heap_win("9g"))
            print("previous memory:", a)
            set_java_heap_win(a)
        else:
            print("Skipping this test since not on windows")

    def test_storm_single_file(self):
        print(colored('Test storm call with single file', 'blue'))
        agents_quantities = [2, 3]
        for population in agents_quantities:
            call_storm("asynchronous_{}.pm prop_{}.pctl".format(population, population), model_path=model_dir, properties_path=properties_dir)
        print(colored('Test storm call with single file with timer', 'blue'))
        for population in agents_quantities:
            call_storm("asynchronous_{}.pm prop_{}.pctl".format(population, population), model_path=model_dir, properties_path=properties_dir, time=True)

    def test_storm_multiple_files(self):
        print(colored('Test storm call multiple files', 'blue'))
        agents_quantities = [2, 3]
        call_storm_files("asyn*_", agents_quantities, model_path=os.path.join(cwd, "models"), properties_path=properties_dir, output_path=storm_results)

    def test_storm_multiple_files_specified_props(self):
        print(colored('Test storm call multiple files with specified files', 'blue'))
        agents_quantities = [2, 3]
        call_storm_files("asyn*_", agents_quantities, property_file="moments.pctl", model_path=os.path.join(cwd, "models"), properties_path=properties_dir, output_path=storm_results)

    def test_prism_easy(self):
        print(colored('Test prism call with single file', 'blue'))
        agents_quantities = [2, 3]

        ## Simulating the path
        call_prism(
            f"{os.path.join(cwd,'models/asynchronous_2.pm')} -const p=0.028502714675268215,q=0.5057623641293089 -simpath 2 dummy_path.txt",
            silent=True, prism_output_path=prism_results)

        print(colored('Testing not existing input file', 'blue'))
        call_prism(
            'fake.pm -const p=0.028502714675268215,q=0.5057623641293089 -simpath 2 '
            'path_synchronous__2_3500_0.028502714675268215_0.5057623641293089.txt', prism_output_path=cwd)

        ## Model checking
        print(colored('Testing simple model checking', 'blue'))
        for population in agents_quantities:
            path1 = os.path.join(cwd, 'models/asynchronous_2.pm')
            path2 = os.path.join(cwd, 'properties/prop_2.pctl')
            call_prism(f"{path1} {path2} -param p=0:1,q=0:1,alpha=0:1", seq=False, std_output_path=prism_results)

    def test_call_prism_files(self):
        agents_quantities = [2, 3]
        ## call_prism_files
        print(colored('Call_prism_files', 'blue'))
        call_prism_files("asyn*_", agents_quantities, model_path=model_dir, output_path=prism_results)

    def test_prism_heavy_load(self):
        agents_quantities = [20, 40]
        ## TODO comment next line
        agents_quantities = []
        ## 20 should pass
        ## This will require noprobcheck for 40
        call_prism_files(f"{os.path.join(model_dir,'syn*_')}", agents_quantities, properties_path=properties_dir, output_path=prism_results)
        ## This will require seq for 40
        call_prism_files(f"{os.path.join(model_dir,'semi*_')}", agents_quantities, properties_path=properties_dir, output_path=prism_results)
        ## This will require seq with adding the memory for 40
        call_prism_files(f"{os.path.join(model_dir,'asyn*_')}", agents_quantities, properties_path=properties_dir, output_path=prism_results)


if __name__ == "__main__":
    unittest.main()
