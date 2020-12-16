import unittest
import warnings

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
            call_storm(os.path.join(model_dir, f"asynchronous_{population}.pm"), [],  os.path.join(properties_dir, f"prop_{population}.pctl"), os.path.join(storm_results, f"storm_no_time_{population}.txt"))
        print(colored('Test storm call with single file with timer', 'blue'))
        for population in agents_quantities:
            call_storm(os.path.join(model_dir, f"asynchronous_{population}.pm"), [],  os.path.join(properties_dir, f"prop_{population}.pctl"), os.path.join(storm_results, f"storm_with_time_{population}.txt"), time=True)

    def test_simulation(self):
        print(colored('Test prism call with single file', 'blue'))
        ## Simulating a path
        call_prism(
            f"{os.path.join(cwd,'models/asynchronous_2.pm')} -const p=0.028502714675268215,q=0.5057623641293089 -simpath 2 dummy_path.txt",
            silent=True, prism_output_path=prism_results)

    def test_fake_file(self):
        print(colored('Testing not existing input file', 'blue'))
        a, b = call_prism(
            'fake.pm -const p=0.028502714675268215,q=0.5057623641293089 -simpath 2 '
            'path_synchronous__2_3500_0.028502714675268215_0.5057623641293089.txt', model_path=model_dir, prism_output_path=cwd)
        self.assertEqual(a, 404)
        self.assertTrue(re.match(r"model.*file.*not found", b))

    def test_simple_model_checking(self):
        ## Model checking
        print(colored('Testing simple model checking', 'blue'))
        agents_quantities = [2, 3]
        for population in agents_quantities:
            path1 = os.path.join(cwd, 'models/asynchronous_2.pm')
            path2 = os.path.join(cwd, 'properties/prop_2.pctl')
            call_prism(f"{path1} {path2} -param p=0:1,q=0:1,alpha=0:1", seq=False, std_output_path=prism_results)

    def test_call_prism_files(self):
        agents_quantities = [2, 3]
        ## call_prism_files
        print(colored('Call_prism_files', 'blue'))
        call_prism_files("asyn*_", agents_quantities, model_path=model_dir, properties_path=properties_dir, output_path=prism_results)

    def test_prism_heavy_load(self, heavy=False):
        if heavy:
            agents_quantities = [20, 40]
        else:
            warnings.warn("Following test is computationaly demanding, please follow instructions to run it", RuntimeWarning)
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
