import platform
from os.path import isfile
from tkinter import *
import webbrowser
import pickle
import os
from pathlib import Path
from tkinter import filedialog, ttk

import configparser
config = configparser.ConfigParser()
workspace = os.path.dirname(__file__)
sys.path.append(workspace)
from load import create_intervals, load_all_functions
import space
from synthetise import *
from mc_prism import *
cwd = os.getcwd()


class Gui:
    def __init__(self, root):
        ## Variables
        self.model_path = ""
        self.properties_path = ""
        self.data_path = ""
        self.prism_results = ""  ## Path to prism results
        self.storm_results = ""  ## Path to Storm results
        self.refinement_results = ""  ## Path to refinement results
        self.figures = ""  ## Path to saved figures
        self.load_config()  ## Load the config file

        self.model = StringVar()  ## Model file / Model as a string
        self.property = StringVar()  ## Property file / property as a string
        self.data = StringVar()  ## Data file / data as a dictionary of/ list of numbers
        self.functions_file = StringVar()  ## Rational functions file
        self.functions = ""  ## Model checking results
        self.intervals = ""  ## Computed intervals
        self.space = StringVar()  ## Space file / class
        self.props = ""  ## Derived properties

        ## Settings
        self.version = "alpha"  ## version of the gui

        ## Settings/data
        self.alpha = ""  ## confidence
        self.n_samples = ""  ## number of samples
        self.program = ""  ## prism/storm
        self.max_depth = ""  ## max recursion depth
        self.coverage = ""  ## coverage threshold
        self.epsilon = ""  ## rectangle size threshold
        self.alg = ""  ## refinement alg. number

        self.size_q = ""  ## number of samples
        self.save = ""  ## True if saving on

        ## GUI INIT
        root.title('mpm')
        root.minsize(400, 300)

        ## DESIGN
        ## DESIGN - STATUS
        frame = Frame(root)
        frame.pack(fill=X)
        Label(frame, text=f"Loaded model:", anchor=W, justify=LEFT).pack(side=LEFT)
        self.model_label = Label(frame, textvariable=self.model, anchor=W, justify=LEFT)
        self.model_label.pack(side=LEFT, fill=X)
        # label1.grid(row=1, column=0, sticky=W)

        frame = Frame(root)
        frame.pack(fill=X)
        Label(frame, text=f"Loaded property:", anchor=W, justify=LEFT).pack(side=LEFT)
        self.property_label = Label(frame, textvariable=self.property, anchor=W, justify=LEFT)
        # property_label.grid(row=2, column=0)
        self.property_label.pack(side=TOP, fill=X)

        frame = Frame(root)
        frame.pack(fill=X)
        Label(frame, text=f"Loaded functions:", anchor=W, justify=LEFT).pack(side=LEFT)
        self.functions_label = Label(frame, textvariable=self.functions_file, anchor=W, justify=LEFT)
        # functions_label.grid(row=3, column=0)
        self.functions_label.pack(side=TOP, fill=X)

        frame = Frame(root)
        frame.pack(fill=X)
        Label(frame, text=f"Loaded data:", anchor=W, justify=LEFT).pack(side=LEFT)
        self.data_label = Label(frame, textvariable=self.data, anchor=W, justify=LEFT)
        # data_label.grid(row=4, column=0)
        self.data_label.pack(side=TOP, fill=X)

        frame = Frame(root)
        frame.pack(fill=X)
        Label(frame, text=f"Loaded space:", anchor=W, justify=LEFT).pack(side=LEFT)
        self.space_label = Label(frame, textvariable=self.space, anchor=W, justify=LEFT)
        # space_label.grid(row=5, column=0)
        self.space_label.pack(side=TOP, fill=X)

        ## DESIGN - TABS
        # Defines and places the notebook widget
        nb = ttk.Notebook(root, height=500, width=500)
        nb.pack(fill=BOTH)

        page1 = ttk.Frame(nb)  # Adds tab 1 of the notebook
        nb.add(page1, text='Edit')
        Label(page1, text=f"Loaded model:", anchor=W, justify=LEFT).pack(side=LEFT)
        Label(page1, text=f"Loaded property:", anchor=W, justify=LEFT).pack(side=LEFT)
        ## TBD ADD THE TEXT OF THE MODELS
        ## TBD ADD THE TEXT OF THE PROPERTY

        # Adds tab 2 of the notebook
        page2 = ttk.Frame(nb)
        nb.add(page2, text='Synthesise')
        ## TBD ADD checkbox to choose the program to synthesise rational functions
        ## TBD ADD THE TEXT TO SHOW THE FILE / RATIONAL FUNCTIONS

        page3 = ttk.Frame(nb)
        nb.add(page3, text='Conversion data + functions to properties')
        ## TBD ADD setting for creating  intervals - alpha, n_samples

        page4 = ttk.Frame(nb)
        nb.add(page4, text='Refine')
        ## TBD ADD setting for creating refinement -  max_dept, coverage, epsilon, alg

        ## MENU
        main_menu = Menu(root)
        root.config(menu=main_menu)

        ## MENU-FILE
        file_menu = Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="File", menu=file_menu)

        ## MENU-FILE-LOAD
        load_menu = Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Load", menu=load_menu, underline=0)
        load_menu.add_command(label="Load model", command=self.load_model)
        load_menu.add_command(label="Load property", command=self.load_property)
        load_menu.add_command(label="Load rational functions", command=self.load_functions)
        load_menu.add_command(label="Load data", command=self.load_data)
        load_menu.add_command(label="Load space", command=self.load_space)
        file_menu.add_separator()

        ## MENU-FILE-SAVE
        save_menu = Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Save", menu=save_menu, underline=0)
        save_menu.add_command(label="Save model", command=self.save_model)
        save_menu.add_command(label="Save property", command=self.save_property)
        # save_menu.add_command(label="Save rational functions", command=self.save_functions())  ## MAYBE IN THE FUTURE
        save_menu.add_command(label="Save data", command=self.save_data)
        save_menu.add_command(label="Save space", command=self.save_space)
        file_menu.add_separator()

        ## MENU-FILE-EXIT
        file_menu.add_command(label="Exit", command=root.quit)

        ## MENU-EDIT
        edit_menu = Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Edit", menu=edit_menu)

        ## MENU-SHOW
        show_menu = Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Show", menu=show_menu)
        show_menu.add_command(label="Space", command=self.show_space)

        ## MENU-ANALYSIS
        analysis_menu = Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Synthesise parameters", command=self.synth_params)
        analysis_menu.add_command(label="Create intervals", command=self.create_intervals)
        analysis_menu.add_command(label="Sample space", command=self.sample_space)
        analysis_menu.add_command(label="Refine space", command=self.refine_space)

        ## MENU-SETTINGS
        settings_menu = Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Edit config", command=self.edit_config)

        ## MENU-HELP
        help_menu = Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Help", command=self.show_help)
        help_menu.add_command(label="Check for updates", command=self.checkupdates)
        help_menu.add_command(label="About", command=self.printabout)

        ## STATUS BAR
        self.status = Label(root, text="", bd=1, relief=SUNKEN, anchor=W)
        self.status.pack(side=BOTTOM, fill=X)

    def load_config(self):
        os.chdir(workspace)
        config.read(os.path.join(workspace, "../config.ini"))

        self.model_path = Path(config.get("paths", "models"))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.properties_path = Path(config.get("paths", "properties"))
        if not os.path.exists(self.properties_path):
            os.makedirs(self.properties_path)

        self.data_path = config.get("paths", "data")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.prism_results = config.get("paths", "prism_results")
        if not os.path.exists(self.prism_results):
            os.makedirs(self.prism_results)

        self.storm_results = config.get("paths", "storm_results")
        if not os.path.exists(self.storm_results):
            os.makedirs(self.storm_results)

        self.refinement_results = config.get("paths", "refinement_results")
        if not os.path.exists(self.refinement_results):
            os.makedirs(self.refinement_results)

        self.figures = config.get("paths", "figures")
        if not os.path.exists(self.figures):
            os.makedirs(self.figures)

        os.chdir(cwd)

    ## LOGIC
    ## FILE
    def load_model(self):
        self.status_set("Please select the model to be loaded.")
        self.model.set(filedialog.askopenfilename(initialdir=self.model_path, title="Model loading - Select file", filetypes=(("pm files", "*.pm"), ("all files", "*.*"))))
        self.status_set("Model loaded.")

    def load_property(self):
        self.status_set("Please select the property to be loaded.")
        self.property.set(filedialog.askopenfilename(initialdir=self.properties_path, title="Property loading - Select file", filetypes=(("property files", "*.pctl"), ("all files", "*.*"))))
        # print(self.property)
        self.status_set("Property loaded.")

    def load_data(self):
        self.status_set("Please select the data to be loaded.")
        self.data.set(filedialog.askopenfilename(initialdir=self.data_path, title="Data loading - Select file", filetypes=(("pickled files", "*.p"), ("all files", "*.*"))))
        # print(self.data)
        self.status_set("Data loaded.")

    def load_functions(self):
        self.status_set("Please select the prism/storm symbolic results to be loaded.")
        self.functions_file.set(filedialog.askopenfilename(initialdir=self.prism_results, title="Rational functions loading - Select file", filetypes=(("text files", "*.txt"), ("all files", "*.*"))))
        # print(self.functions)
        self.functions, rewards = load_all_functions( self.functions_file.get(), tool="unknown", factorize=True, agents_quantities=False, rewards_only=False, f_only=False)
        # print("self.functions", self.functions)
        # print("self.rewards", self.rewards)
        self.functions.update(rewards)
        print(self.functions)
        self.status_set(f"{len(self.functions.keys())} rational functions loaded")

    def load_space(self):
        self.status_set("Please select the space to be loaded.")
        self.space.set(filedialog.askopenfilename(initialdir=self.data_path, title="Space loading - Select file", filetypes=(("pickled files", "*.p"), ("all files", "*.*"))))
        # print(self.space)
        self.status_set("Space loaded")

    def save_model(self):
        if self.model is "":
            self.status_set("There is no model to be saved.")
            return

        self.status_set("Please select folder to store the model in.")
        save_model = filedialog.asksaveasfilename(initialdir=self.model_path, title="Model saving - Select file", filetypes=(("pm files", "*.pm"), ("all files", "*.*")))
        print(save_model)

        if isfile(self.model.get()):
            print(self.model)
            ## os.copy the file
        else:
            with open(save_model, "w") as file:
                for line in self.model:
                    file.write(line)
        self.status_set("Model saved.")

    def save_property(self):
        if self.property is "":
            self.status_set("There is no property to be saved.")
            return

        self.status_set("Please select folder to store the property in.")
        save_property = filedialog.asksaveasfilename(initialdir=self.model_path, title="Property saving - Select file", filetypes=(("pctl files", "*.pctl"), ("all files", "*.*")))
        print(save_property)
        if isfile(self.property.get()):
            print(self.property)
            ## os.copy the file
        else:
            with open(save_property, "w") as file:
                for line in self.property:
                    file.write(line)
        self.status_set("Property saved.")

    ## MAYBE IN THE FUTURE
    def save_functions(self):
        if self.functions is "" and self.rewards is "":
            self.status_set("There are no rational functions to be saved.")
            return

        ## TBD choose to save rewards or normal functions

        self.status_set("Please select folder to store the rational functions in.")
        if self.program is "prism":
            save_functions = filedialog.asksaveasfilename(initialdir=self.prism_results, title="Rational functions saving - Select file", filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
        if self.program is "storm":
            save_functions = filedialog.asksaveasfilename(initialdir=self.storm_results, title="Rational functions saving - Select file", filetypes=(("pickle files", "*.p"), ("all files", "*.*")))

        print(save_functions)
        if isfile(self.property.get()):
            print(self.property)
            ## os.copy the file
        else:
            with open(save_functions, "w") as file:
                for line in self.property:
                    file.write(line)
        self.status_set("Property saved.")

    def save_data(self):
        if self.data is "":
            self.status_set("There is no data to be saved.")
            return

        self.status_set("Please select folder to store the data in.")
        save_data = filedialog.asksaveasfilename(initialdir=self.data_path, title="Data saving - Select file", filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
        print(save_data)
        pickle.dump(self.data, open(save_data, 'wb'))
        self.status_set("Data saved.")

    def save_space(self):
        if self.space is "":
            self.status_set("There is no space to be saved.")
            return
        self.status_set("Please select folder to store the space in.")
        save_space = filedialog.asksaveasfilename(initialdir=self.data_path, title="Space saving - Select file", filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
        print(save_space)
        pickle.dump(self.data, open(save_space, 'wb'))
        self.status_set("Space saved.")

    ## EDIT

    ## SHOW
    def show_space(self):
        self.status_set("Please select which parts to be shown.")
        ## TBD choose what to show
        self.space.show(self, title="", green=True, red=True, sat_samples=False, unsat_samples=False, save=False)

    ## ANALYSIS
    def synth_params(self):
        self.status_set("Parameter synthesis running ...")
        ## TBD solve agents_quantities

        if self.model.get() is "":
            self.load_model()

        if self.property.get() is "":
            self.load_property()

        ## TBD Window where to choose between PRISM and Storm
        self.program = "prism"

        if self.program.lower() is "prism":
            call_prism_files(self.model, agents_quantities, param_intervals=False, seq=False, noprobchecks=False, memory="", model_path=self.model_path, properties_path=self.properties_path, property_file=False, output_path=prism_results)
            self.status_set("Parameter synthesised. Output here: {}", [os.path.join(self.prism_results, filename)])
            return

        if self.program.lower() is "storm":
            call_storm_files(self.model, agents_quantities, model_path=model_path, properties_path=properties_path, property_file=False, output_path=storm_results, time=False)
            self.status_set("Parameter synthesised. Output here: {}", [os.path.join(self.prism_results, filename)])
            return
        self.status_set("Selected program not recognised")

    def create_intervals(self):
        self.status_set("Intervals are being created ...")
        if self.data is "":
            self.load_data()
        ## TBD DESIGN THIS POPUP WINDOW AFTER CLICK to set alpha, n_samples
        self.intervals = create_intervals(self.alpha, self.n_samples, self.data)
        self.status_set("Intervals created.")

    def sample_space(self):
        self.status_set("Space sampling running ...")
        ## TBD DESIGN THIS POPUP WINDOW AFTER CLICK
        ## TBD LOAD props, size_q
        self.space.grid_sample(self.props, self.size_q, silent=False, save=self.save)
        self.status_set("Space sampling done.")

    def refine_space(self):
        self.status_set("Space refinement running ...")
        print("refine_space")

        ## TBD DESIGN THIS POPUP WINDOW AFTER CLICK to set max_depth, epsilon, coverage, algorithm

        if self.intervals is "":
            ## TBD Error window, compute the intervals beforehead
            print("Intervals not computed, properties cannot be computed")

        if self.props is "":

            self.props = ineq_to_props(self.functions, self.intervals, silent=True)
            ## TBD
            print("Properties not computed")

        ## TBD LOAD props, n, epsilon, coverage
        self.space = check_deeper(self.space, self.props, self.max_depth, self.epsilon, self.coverage, silent=True, version=self.alg, size_q=False, debug=False, save=False, title="")
        self.status_set("Space refinement done.")

    ## SETTINGS
    def edit_config(self):
        if "wind" in platform.system().lower():
            ## TBD TEST THIS ON WINDOWS
            os.startfile(f'{os.path.join(workspace, "../config.ini")}')
        else:
            os.system(f'gedit {os.path.join(workspace, "../config.ini")}')
        self.load_config()  ## Reloading the config file after change
        self.status_set("Config file saved.")

    ## HELP
    def show_help(self):
        print("show_help")
        webbrowser.open_new("https://github.com/xhajnal/mpm#mpm")

    def checkupdates(self):
        print("check updates")
        webbrowser.open_new("https://github.com/xhajnal/mpm/releases")

    def printabout(self):
        top2 = Toplevel(root)
        top2.title("About")
        top2.resizable(0, 0)
        explanation = f" Mpm version: {self.version} \n More info here: https://github.com/xhajnal/mpm \n Powered by University of Constance and Masaryk University"
        Label(top2, justify=LEFT, text=explanation).pack(padx=13, pady=20)
        top2.transient(root)
        top2.grab_set()
        root.wait_window(top2)

        print("Mpm version alpha")
        print("More info here: https://github.com/xhajnal/mpm")
        print("Powered by University of Constance and Masaryk University")

    ## STATUS BAR
    def status_set(self, text, *args):
        self.status.config(text=text.format(args))
        self.status.update_idletasks()

    def status_clear(self):
        self.status.config(text="")
        self.status.update_idletasks()


root = Tk()
spam = Gui(root)
root.mainloop()

# root = Tk()
# # theLabel = Label(root, text="Hello")
# # theLabel.pack()
#
# topFrame = Frame(root)
# topFrame.pack()
#
# bottomFrame = Frame(root)
# bottomFrame.pack(side=BOTTOM)
#
# button1 = Button(topFrame, text="Button 1", fg="red")
# button2 = Button(topFrame, text="Button 2", fg="blue")
# button3 = Button(topFrame, text="Button 3", fg="green")
# button4 = Button(bottomFrame, text="Button 4", fg="purple")
#
#
# button1.pack(side=LEFT)
# button2.pack(side=LEFT)
# button3.pack(side=LEFT)
# button4.pack()
#
# root.mainloop()
