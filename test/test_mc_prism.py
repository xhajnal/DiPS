import unittest
from src.mc_prism import *


class MyTestCase(unittest.TestCase):
    def test_changing_javaheap(self):
        print(colored('Test_changing_javaheap on Windows', 'blue'))
        if sys.platform.startswith("win"):
            a = (set_javaheap_win("9g"))
            print("previous memory:", a)
            set_javaheap_win(a)
        else:
            print("Skipping this test since not on windows")

    def test_storm_single_file(self):
        print(colored('Test storm call with single file', 'blue'))
        agents_quantities = [2, 3]
        for population in agents_quantities:
            call_storm("semisynchronous_{}.pm prop_{}.pctl".format(population, population), std_output_path=os.path.join(cwd, "test"))
        print(colored('Test storm call with single file with timer', 'blue'))
        for population in agents_quantities:
            call_storm("semisynchronous_{}.pm prop_{}.pctl".format(population, population), std_output_path=os.path.join(cwd, "test"), time=True)

    def test_storm_multiple_files(self):
        print(colored('Test storm call multiple files', 'blue'))
        agents_quantities = [2, 3]
        call_storm_files("syn*_", agents_quantities, output_path=os.path.join(cwd, "test"))
        print(colored('Test storm call multiple files with specified files', 'blue'))
        call_storm_files("syn*_", agents_quantities, property_file="moments.pctl", output_path=os.path.join(cwd, "test"))

    def test_prism_easy(self):
        print(colored('Test prism call with single file', 'blue'))
        agents_quantities = [2, 3]
        try:
            os.mkdir("test")
        except FileExistsError:
            print("folder src/test probably already exists, if not this will fail")
        os.chdir("test")

        call_prism(
            "synchronous_10.pm -const p=0.028502714675268215,q=0.5057623641293089 -simpath 2 dummy_path1550773616.0244777.txt",
            silent=True, prism_output_path="/home/matej/Git/mpm/src/test", std_output_path=None)

        ## Model checking
        print(colored('Testing simple model checking', 'blue'))
        for population in agents_quantities:
            call_prism("semisynchronous_{}.pm prop_{}.pctl -param p=0:1,q=0:1,alpha=0:1"
                       .format(population, population), seq=False, std_output_path=os.path.join(cwd, "test"))

        ## Simulating the path
        print(colored('Testing simulation', 'blue'))
        call_prism(
            'synchronous_2.pm -const p=0.028502714675268215,q=0.5057623641293089 -simpath 2 '
            'path11.txt', prism_output_path=os.path.join(cwd, "test"), std_output_path=None)

        print(colored('Test simulation change the path of the path files output', 'blue'))
        print(colored('This should produce a file in ', 'blue'))
        call_prism(
            'synchronous_2.pm -const p=0.028502714675268215,q=0.5057623641293089 -simpath 2 '
            'path12.txt', prism_output_path="../src/test", std_output_path=None)

        # print(colored('testing simulation with stdout', 'blue'))
        # file = open("path_synchronous__2_3500_0.028502714675268215_0.5057623641293089.txt", "w+")
        # print(colored('testing not existing input file', 'blue'))
        # call_prism(
        #    'fake.pm -const p=0.028502714675268215,q=0.5057623641293089 -simpath 2 '
        #    'path_synchronous__2_3500_0.028502714675268215_0.5057623641293089.txt', prism_output_path=cwd,std_output_path=None)

        ## call_prism_files
        print(colored('Call_prism_files', 'blue'))
        call_prism_files("syn*_", agents_quantities)

        print(colored('Call_prism_files2', 'blue'))
        call_prism_files("multiparam_syn*_", agents_quantities)

    def test_prism_heavy_load(self):
        agents_quantities = [20, 40]
        ## 20 should pass
        ## This will require noprobcheck for 40
        call_prism_files("syn*_", agents_quantities, output_path=os.path.join(cwd, "test"))
        ## This will require seq for 40
        call_prism_files("semi*_", agents_quantities, output_path=os.path.join(cwd, "test"))
        ## This will require seq with adding the memory for 40
        call_prism_files("asyn*_", agents_quantities, output_path=os.path.join(cwd, "test"))


if __name__ == "__main__":
    unittest.main()

