import platform
from os.path import isfile
from tkinter import *
from tkinter import scrolledtext, messagebox
import webbrowser
import pickle
import os
from pathlib import Path
from tkinter import filedialog, ttk

import configparser

config = configparser.ConfigParser()
workspace = os.path.dirname(__file__)
sys.path.append(workspace)
from load import create_intervals, load_all_functions, find_param
import space
from synthetise import ineq_to_props, check_deeper
from mc_prism import call_prism_files, call_storm_files

cwd = os.getcwd()


class MyDialog:

    def __init__(self, parent, text):
        top = self.top = Toplevel(parent)
        Label(top, text=text).pack()
        self.e.pack(padx=5)

        b = Button(top, text="OK", command=quit)
        b.pack(pady=5)


class Gui:
    def __init__(self, root):

        ## Variables
        ## Directories
        self.model_dir = ""
        self.properties_dir = ""
        self.data_dir = ""
        self.prism_results = ""  ## Path to prism results
        self.storm_results = ""  ## Path to Storm results
        self.refinement_results = ""  ## Path to refinement results
        self.figures_dir = ""  ## Path to saved figures
        self.load_config()  ## Load the config file

        ## Files
        self.model_file = StringVar()  ## Model file
        self.property_file = StringVar()  ## Property file
        self.data_file = StringVar()  ## Data file
        self.functions_file = StringVar()  ## Rational functions file
        self.space_file = StringVar()  ## Space file

        ## True Variables
        self.model = ""
        self.property = ""
        self.data = ""
        self.functions = ""  ## Model checking results
        self.intervals = ""  ## Computed intervals
        self.space = ""  ## Instance of a space Class
        self.props = ""  ## Derived properties

        ## Settings
        self.version = "alpha"  ## version of the gui

        ## Settings/data
        # self.alpha = ""  ## confidence
        # self.n_samples = ""  ## number of samples
        self.program = StringVar()  ## prism/storm
        self.max_depth = ""  ## max recursion depth
        self.coverage = ""  ## coverage threshold
        self.epsilon = ""  ## rectangle size threshold
        self.alg = ""  ## refinement alg. number

        self.size_q = ""  ## number of samples
        self.save = ""  ## True if saving on

        ## GUI INIT
        root.title('mpm')
        root.minsize(1000, 300)

        ## DESIGN

        ## STATUS BAR
        self.status = Label(root, text="", bd=1, relief=SUNKEN, anchor=W)
        self.status.pack(side=BOTTOM, fill=X)

        ## DESIGN - STATUS
        frame = Frame(root)
        frame.pack(fill=X)
        Label(frame, text=f"Loaded model:", anchor=W, justify=LEFT).pack(side=LEFT)
        self.model_label = Label(frame, textvariable=self.model_file, anchor=W, justify=LEFT)
        self.model_label.pack(side=LEFT, fill=X)
        # label1.grid(row=1, column=0, sticky=W)

        frame = Frame(root)
        frame.pack(fill=X)
        Label(frame, text=f"Loaded property:", anchor=W, justify=LEFT).pack(side=LEFT)
        self.property_label = Label(frame, textvariable=self.property_file, anchor=W, justify=LEFT)
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
        self.data_label = Label(frame, textvariable=self.data_file, anchor=W, justify=LEFT)
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
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=1)

        ## TAB EDIT
        page1 = ttk.Frame(nb, width=600, height=200, name="edit")  # Adds tab 1 of the notebook
        ## TBD CHANGE THE STATE OF THE TAB WHILE RUNNING
        # style = ttk.Style()
        # style.configure("BW.TLabel", foreground="black", background="white")
        # print("self.model.get()", self.model_file.get())
        # print("self.property.get()", self.property_file.get())
        # print(self.model_file.get() is "")
        # state = ("disabled", "normal")[(self.model_file.get() is not "") or (self.property_file.get() is not "")]
        # print("state", state)
        # lambdaaa = lambda: "disabled" if (self.model_file.get() is "") else "normal"
        # print("lambdaaa", lambdaaa())
        nb.add(page1, text='Edit', state="normal", sticky="nsew")

        # self.model_file.set("dsada")
        # nb.update()
        # page1.update()
        # print("lambdaaa", lambdaaa())

        frame_left = Frame(page1, width=600, height=200)
        frame_left.pack(side=LEFT, fill=X)

        Button(frame_left, text='Load model', command=self.load_model).grid(row=0, column=0, sticky=W, pady=4)  # pack(anchor=W)
        Button(frame_left, text='Save model', command=self.save_model).grid(row=0, column=1, sticky=W, pady=4)  # pack(anchor=W)
        Label(frame_left, text=f"Loaded model:", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, pady=4)  # pack(anchor=W)

        self.model_text = scrolledtext.ScrolledText(frame_left, height=100)
        # self.model_text.config(state="disabled")
        self.model_text.grid(row=2, column=0, columnspan=16, sticky=W+E+N+S, pady=4)  # pack(anchor=W, fill=X, expand=True)

        frame_right = Frame(page1)
        frame_right.pack(side=RIGHT, fill=X)

        Button(frame_right, text='Load property', command=self.load_property).grid(row=0, column=0, sticky=W, pady=4)  # pack(anchor=W)
        Button(frame_right, text='Save property', command=self.save_property).grid(row=0, column=1, sticky=W, pady=4)  # pack(anchor=W)
        Label(frame_right, text=f"Loaded property:", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, pady=4)  # pack(anchor=W)

        self.property_text = scrolledtext.ScrolledText(frame_right, height=100)
        # self.property_text.config(state="disabled")
        self.property_text.grid(row=2, column=0, columnspan=16, rowspan=2, sticky=W+E+N+S, pady=4)  # pack(anchor=W, fill=X)

        print(nb.select(0), type(nb.select(0)))
        # print(page1, type(page1))

        # page1.state(("normal",))
        # page1.s

        ## TBD ADD THE TEXT OF THE MODELS
        ## TBD ADD THE TEXT OF THE PROPERTY

        ## TAB SYNTHESISE
        page2 = ttk.Frame(nb, width=400, height=200, name="synthetise")  # Adds tab 2 of the notebook
        nb.add(page2, text='Synthesise')

        ## SELECTING THE PROGRAM
        self.program.set(1)
        Label(page2, text="Select the program: ", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, pady=4)
        Radiobutton(page2, text="Prism", variable=self.program, value="prism").grid(row=1, column=1, sticky=W, pady=4)
        Radiobutton(page2, text="Storm", variable=self.program, value="storm").grid(row=1, column=2, sticky=W, pady=4)
        Button(page2, text='Run parameter synthesis', command=self.synth_params).grid(row=2, column=0, sticky=W, pady=4)
        Button(page2, text='Load results', command=self.load_functions).grid(row=2, column=1, sticky=W, pady=4)
        ## TBD ADD THE TEXT TO SHOW THE FILE / RATIONAL FUNCTIONS


        ## TAB DATA CONVERSION
        page3 = ttk.Frame(nb, width=400, height=200, name="conversion")
        nb.add(page3, text='Conversion data + functions to properties')

        Button(page3, text='Load data', command=self.load_data).grid(row=0, column=0, sticky=W, pady=4)

        ## SET THE INTERVAL COMPUTATION SETTINGS
        Label(page3, text="Set alpha, the confidence:", anchor=W, justify=LEFT).grid(row=1)
        Label(page3, text="Set n_samples, number of samples: ", anchor=W, justify=LEFT).grid(row=2)

        self.alpha_entry = Entry(page3)
        self.n_samples_entry = Entry(page3)

        self.alpha_entry.grid(row=1, column=1)
        self.n_samples_entry.grid(row=2, column=1)

        ## TBD ADD setting for creating  intervals - alpha, n_samples
        Button(page3, text='Create intervals', command=self.create_intervals).grid(row=4, column=0, sticky=W, pady=4)

        Label(page3, text=f"Intervals:", anchor=W, justify=LEFT).grid(row=5, column=0, sticky=W, pady=4)

        self.data_text = Text(page3, height=10, width=30)
        # self.data_text.config(state="disabled")
        self.data_text.grid(row=5, column=0, sticky=W, pady=4)

        ## TAB DATA REFINEMENT
        page4 = ttk.Frame(nb, width=400, height=200, name="refine")
        nb.add(page4, text='Refine')

        Label(page4, text="Set max_dept: ", anchor=W, justify=LEFT).grid(row=0)
        Label(page4, text="Set coverage: ", anchor=W, justify=LEFT).grid(row=1)
        Label(page4, text="Set epsilon: ", anchor=W, justify=LEFT).grid(row=2)
        Label(page4, text="Set algorithm: ", anchor=W, justify=LEFT).grid(row=3)

        self.max_dept_entry = Entry(page4)
        self.coverage_entry = Entry(page4)
        self.epsilon_entry = Entry(page4)
        self.algorithm_entry = Entry(page4)

        self.max_dept_entry.grid(row=0, column=1)
        self.coverage_entry.grid(row=1, column=1)
        self.epsilon_entry.grid(row=2, column=1)
        self.algorithm_entry.grid(row=3, column=1)

        Button(page4, text='Refine space', command=self.refine_space).grid(row=4, column=0, sticky=W, pady=4)
        Button(page4, text='Load space', command=self.load_space).grid(row=4, column=1, sticky=W, pady=4)

        page5 = ttk.Frame(nb, name="testy")
        # page5.pack(expand=True)
        nb.add(page5, text='testy')

        self.testy_text = Text(page5, height=100)
        # self.testy_text.config(state="disabled")
        # self.testy_text.grid(row=0, column=0, sticky=W + E + N + S, pady=4)
        self.testy_text.pack(fill="both")

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

    def load_config(self):
        os.chdir(workspace)
        config.read(os.path.join(workspace, "../config.ini"))

        self.model_dir = Path(config.get("paths", "models"))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.properties_dir = Path(config.get("paths", "properties"))
        if not os.path.exists(self.properties_dir):
            os.makedirs(self.properties_dir)

        self.data_dir = config.get("paths", "data")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.prism_results = config.get("paths", "prism_results")
        if not os.path.exists(self.prism_results):
            os.makedirs(self.prism_results)

        self.storm_results = config.get("paths", "storm_results")
        if not os.path.exists(self.storm_results):
            os.makedirs(self.storm_results)

        self.refinement_results = config.get("paths", "refinement_results")
        if not os.path.exists(self.refinement_results):
            os.makedirs(self.refinement_results)

        self.figures_dir = config.get("paths", "figures")
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)

        os.chdir(cwd)

    ## LOGIC
    ## FILE
    def load_model(self):
        self.status_set("Please select the model to be loaded.")
        self.model_file.set(filedialog.askopenfilename(initialdir=self.model_dir, title="Model loading - Select file",
                                                       filetypes=(("pm files", "*.pm"), ("all files", "*.*"))))
        self.model_text.delete('1.0', END)
        self.model_text.insert('end', open(self.model_file.get(), 'r').read())
        self.status_set("Model loaded.")
        # print("self.model", self.model.get())

    def load_property(self):
        self.status_set("Please select the property to be loaded.")
        self.property_file.set(
            filedialog.askopenfilename(initialdir=self.properties_dir, title="Property loading - Select file",
                                       filetypes=(("property files", "*.pctl"), ("all files", "*.*"))))
        self.property_text.delete('1.0', END)
        self.property_text.insert('end', open(self.property_file.get(), 'r').read())
        self.status_set("Property loaded.")
        # print("self.property", self.property.get())

    def load_data(self):
        self.status_set("Please select the data to be loaded.")
        self.data_file.set(filedialog.askopenfilename(initialdir=self.data_dir, title="Data loading - Select file",
                                                      filetypes=(("pickled files", "*.p"), ("all files", "*.*"))))
        if ".p" in self.data_file.get():
            self.data = pickle.load(open(self.data_file.get(), "rb"))
        else:
            print()
            ## TBD
            # self.data = PARSE THE DATA
        # print(self.data)
        self.status_set("Data loaded.")

    def load_functions(self):
        self.status_set("Please select the prism/storm symbolic results to be loaded.")

        print("self.program.get()", self.program.get())
        if self.program.get() == "prism":
            self.functions_file.set(filedialog.askopenfilename(initialdir=self.prism_results, title="Rational functions loading - Select file", filetypes=(("text files", "*.txt"), ("all files", "*.*"))))
        else:
            self.functions_file.set(filedialog.askopenfilename(initialdir=self.storm_results, title="Rational functions loading - Select file", filetypes=(("text files", "*.txt"), ("all files", "*.*"))))
        # print(self.functions)
        self.functions, rewards = load_all_functions(self.functions_file.get(), tool="unknown", factorize=True,
                                                     agents_quantities=False, rewards_only=False, f_only=False)
        # print("self.functions", self.functions)
        # print("self.rewards", self.rewards)
        self.functions.update(rewards)
        print(self.functions)
        self.status_set(f"{len(self.functions.keys())} rational functions loaded")

    def load_space(self):
        self.status_set("Please select the space to be loaded.")
        self.space_file.set(filedialog.askopenfilename(initialdir=self.data_dir, title="Space loading - Select file", filetypes=(("pickled files", "*.p"), ("all files", "*.*"))))
        # print(self.space)

        ## pickle load
        self.space = pickle.load(open(self.space_file.get(), "rb"))
        print(self.space)
        self.status_set("Space loaded")

    def save_model(self):
        ## CHECK IF THE MODEL IS NON EMPTY
        # if len(self.model_text.get('1.0', END)) <= 1:
        #    self.status_set("There is no model to be saved.")
        #    return

        self.status_set("Please select folder to store the model in.")
        save_model = filedialog.asksaveasfilename(initialdir=self.model_dir, title="Model saving - Select file",
                                                  filetypes=(("pm files", "*.pm"), ("all files", "*.*")))
        if "." not in save_model:
            save_model = save_model + ".pm"
        # print("save_model", save_model)

        with open(save_model, "w") as file:
            file.write(self.model_text.get(1.0, END))

        self.status_set("Model saved.")

    def save_property(self):
        ## CHECK IF THE PROPERTY IS NON EMPTY
        # if len(self.property_text.get('1.0', END)) <= 1:
        #    self.status_set("There is no property to be saved.")
        #    return

        self.status_set("Please select folder to store the property in.")
        save_property = filedialog.asksaveasfilename(initialdir=self.properties_dir, title="Property saving - Select file",
                                                     filetypes=(("pctl files", "*.pctl"), ("all files", "*.*")))
        if "." not in save_property:
            save_property = save_property + ".pctl"
        # print("save_property", save_property)

        with open(save_property, "w") as file:
            file.write(self.property_text.get(1.0, END))

        self.status_set("Property saved.")

    ## MAYBE IN THE FUTURE
    def save_functions(self):
        if self.functions is "":
            self.status_set("There are no rational functions to be saved.")
            return

        ## TBD choose to save rewards or normal functions

        self.status_set("Please select folder to store the rational functions in.")
        if self.program is "prism":
            save_functions = filedialog.asksaveasfilename(initialdir=self.prism_results,
                                                          title="Rational functions saving - Select file",
                                                          filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
        elif self.program is "storm":
            save_functions = filedialog.asksaveasfilename(initialdir=self.storm_results,
                                                          title="Rational functions saving - Select file",
                                                          filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
        else:
            save_functions = "save_functions Error"
        print(save_functions)
        if isfile(self.property_file.get()):
            print(self.property_file)
            ## os.copy the file
        else:
            with open(save_functions, "w") as file:
                for line in self.property:
                    file.write(line)
        self.status_set("Property saved.")

    def save_data(self):
        if self.data_file is "":
            self.status_set("There is no data to be saved.")
            return

        self.status_set("Please select folder to store the data in.")
        save_data = filedialog.asksaveasfilename(initialdir=self.data_dir, title="Data saving - Select file",
                                                 filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
        print(save_data)
        pickle.dump(self.data_file, open(save_data, 'wb'))
        self.status_set("Data saved.")

    def save_space(self):
        if self.space is "":
            self.status_set("There is no space to be saved.")
            return
        self.status_set("Please select folder to store the space in.")
        save_space = filedialog.asksaveasfilename(initialdir=self.data_dir, title="Space saving - Select file",
                                                  filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
        print(save_space)
        pickle.dump(self.data_file, open(save_space, 'wb'))
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

        ## If model file not selected load model
        if self.model_file.get() is "":
            self.load_model()

        ## If property file not selected load property
        if self.property_file.get() is "":
            self.load_property()

        if self.program.get().lower() == "prism":
            call_prism_files(self.model_file.get(), [], param_intervals=False, seq=False, noprobchecks=False, memory="",
                             model_path="", properties_path=self.properties_dir, property_file=self.property_file.get(),
                             output_path=self.prism_results)
            # self.status_set("Parameter synthesised. Output here: {}", [os.path.join(self.prism_results, filename)])
            return
        elif self.program.get().lower() == "storm":
            call_storm_files(self.model_file.get(), [], model_path=self.model_dir, properties_path=self.properties_dir,
                             property_file=self.property_file.get(), output_path=self.storm_results, time=False)
            # self.status_set("Parameter synthesised. Output here: {}", [os.path.join(self.prism_results, filename)])
            return
        else:
            ## Show window to inform to select the program
            messagebox.showinfo("Synthesise", "Select a program for parameter synthesis first.")

    def create_intervals(self):
        self.status_set("Intervals are being created ...")
        if self.data_file.get() is "":
            self.load_data()

        print(self.data_file.get())
        ## TBD DESIGN THIS POPUP WINDOW AFTER CLICK to set alpha, n_samples
        self.intervals = create_intervals(self.alpha_entry.get(), self.n_samples_entry.get(), self.data_file)
        self.data_text.delete('1.0', END)
        self.data_text.insert('end', self.intervals)
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

        if self.intervals == "":
            ## TBD Error window, compute the intervals beforehead
            print("Intervals not computed, properties cannot be computed")

        if self.props == "":
            self.props = ineq_to_props(self.functions, self.intervals, silent=True)
            ## TBD
            print("Properties not computed")

        if self.space == "":
            print("space is empty creating new one")
            parameters = globals()["parameters"]
            for polynome in self.props:
                parameters.update(load.find_param(polynome))
            self.space = space.RefinedSpace(region, parameters, types=None,  rectangles_sat=False, rectangles_unsat=False, rectangles_unknown=None, sat_samples=None, unsat_samples=None, true_point=False, title=False, proxy_params=False, decoding=False)

        ## TBD LOAD props, n, epsilon, coverage
        self.space = check_deeper(self.space, self.props, self.max_depth, self.epsilon, self.coverage, silent=True,
                                  version=self.alg, size_q=False, debug=False, save=False, title="")
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

## ON UBUNTU
# root.attributes('-zoomed', True)
root.state('zoomed')
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
