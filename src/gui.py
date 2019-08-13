import platform
from os.path import isfile
from time import sleep
from tkinter import *
from tkinter import scrolledtext, messagebox
import webbrowser
import pickle
import os
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.messagebox import askyesno

import matplotlib.pyplot as pyplt
import matplotlib
matplotlib.use("TKAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import configparser

config = configparser.ConfigParser()
workspace = os.path.dirname(__file__)
sys.path.append(workspace)

from load import create_intervals, load_all_functions, find_param
import space
from synthetise import ineq_to_props, check_deeper
from mc_prism import call_prism_files, call_storm_files
from sample_n_visualise import sample_fun, visualise, eval_and_show, get_param_values

cwd = os.getcwd()


class Gui(Tk):
    def __init__(self,  *args, **kwargs):

        super().__init__(*args, **kwargs)
        ## Variables
        ## Directories
        self.model_dir = ""
        self.property_dir = ""
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
        self.props_file = StringVar()  ## Props file
        self.space_file = StringVar()  ## Space file

        ## Checking the change
        self.model_changed = False
        self.property_changed = False
        self.functions_changed = False
        self.data_changed = False
        self.intervals_changed = False
        self.props_changed = False
        self.space_changed = False

        ## True Variables
        self.model = ""
        self.property = ""
        self.data = ""
        self.functions = ""  ## Model checking results
        self.intervals = ""  ## Computed intervals
        self.parameters = ""  ##  Parsed parameters
        self.space = ""  ## Instance of a space Class
        self.props = ""  ## Derived properties

        ## Settings
        self.version = "1.2.1"  ## version of the gui

        ## Settings/data
        # self.alpha = ""  ## confidence
        # self.n_samples = ""  ## number of samples
        self.program = StringVar()  ## prism/storm
        self.max_depth = ""  ## max recursion depth
        self.coverage = ""  ## coverage threshold
        self.epsilon = ""  ## rectangle size threshold
        self.alg = ""  ## refinement alg. number

        self.factor = BooleanVar()  ## Flag for factorising rational functions
        self.size_q = ""  ## number of samples
        self.save = ""  ## True if saving on

        ## OTHER SETTINGS
        self.button_pressed = BooleanVar()

        ## GUI INIT
        self.title('mpm')
        self.minsize(1000, 300)

        ## DESIGN

        ## STATUS BAR
        self.status = Label(self, text="", bd=1, relief=SUNKEN, anchor=W)
        self.status.pack(side=BOTTOM, fill=X)

        ## DESIGN - STATUS
        frame = Frame(self)
        frame.pack(fill=X)
        Label(frame, text=f"Loaded model:", anchor=W, justify=LEFT).pack(side=LEFT)
        self.model_label = Label(frame, textvariable=self.model_file, anchor=W, justify=LEFT)
        self.model_label.pack(side=LEFT, fill=X)
        # label1.grid(row=1, column=0, sticky=W)

        frame = Frame(self)
        frame.pack(fill=X)
        Label(frame, text=f"Loaded property:", anchor=W, justify=LEFT).pack(side=LEFT)
        self.property_label = Label(frame, textvariable=self.property_file, anchor=W, justify=LEFT)
        # property_label.grid(row=2, column=0)
        self.property_label.pack(side=TOP, fill=X)

        frame = Frame(self)
        frame.pack(fill=X)
        Label(frame, text=f"Loaded functions:", anchor=W, justify=LEFT).pack(side=LEFT)
        self.functions_label = Label(frame, textvariable=self.functions_file, anchor=W, justify=LEFT)
        # functions_label.grid(row=3, column=0)
        self.functions_label.pack(side=TOP, fill=X)

        frame = Frame(self)
        frame.pack(fill=X)
        Label(frame, text=f"Loaded data:", anchor=W, justify=LEFT).pack(side=LEFT)
        self.data_label = Label(frame, textvariable=self.data_file, anchor=W, justify=LEFT)
        # data_label.grid(row=4, column=0)
        self.data_label.pack(side=TOP, fill=X)

        frame = Frame(self)
        frame.pack(fill=X)
        Label(frame, text=f"Loaded space:", anchor=W, justify=LEFT).pack(side=LEFT)
        self.space_label = Label(frame, textvariable=self.space_file, anchor=W, justify=LEFT)
        # space_label.grid(row=5, column=0)
        self.space_label.pack(side=TOP, fill=X)

        ## DESIGN - TABS
        # Defines and places the notebook widget
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=1)


        ## TAB EDIT
        page1 = ttk.Frame(nb, width=600, height=200, name="model_properties")  # Adds tab 1 of the notebook
        nb.add(page1, text='Model & Properties', state="normal", sticky="nsew")

        # page1.rowconfigure(5, weight=1)
        # page1.columnconfigure(6, weight=1)

        frame_left = Frame(page1, width=600, height=200)
        frame_left.rowconfigure(3, weight=1)
        frame_left.columnconfigure(6, weight=1)
        frame_left.pack(side=LEFT, fill=X)

        Button(frame_left, text='Load model', command=self.load_model).grid(row=0, column=0, sticky=W, pady=4, padx=4)  # pack(anchor=W)
        Button(frame_left, text='Save model', command=self.save_model).grid(row=0, column=1, sticky=W, pady=4)  # pack(anchor=W)
        Label(frame_left, text=f"Loaded model:", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, pady=4)  # pack(anchor=W)

        self.model_text = scrolledtext.ScrolledText(frame_left, height=100)
        # self.model_text.config(state="disabled")
        self.model_text.grid(row=2, column=0, columnspan=16, rowspan=2, sticky=W+E+N+S, pady=4)  # pack(anchor=W, fill=X, expand=True)

        frame_right = Frame(page1)
        frame_right.rowconfigure(3, weight=1)
        frame_right.columnconfigure(6, weight=1)
        frame_right.pack(side=RIGHT, fill=X)

        Button(frame_right, text='Load property', command=self.load_property).grid(row=0, column=0, sticky=W, pady=4, padx=4)  # pack(anchor=W)
        Button(frame_right, text='Save property', command=self.save_property).grid(row=0, column=1, sticky=W, pady=4)  # pack(anchor=W)
        Label(frame_right, text=f"Loaded property:", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, pady=4)  # pack(anchor=W)

        self.property_text = scrolledtext.ScrolledText(frame_right, height=100)
        # self.property_text.config(state="disabled")
        self.property_text.grid(row=2, column=0, columnspan=16, rowspan=2, sticky=W+E+N+S, pady=4)  # pack(anchor=W, fill=X)

        # print(nb.select(0), type(nb.select(0)))
        # print(page1, type(page1))


        ## TAB SYNTHESISE
        page2 = ttk.Frame(nb, width=400, height=100, name="synthetise")  # Adds tab 2 of the notebook
        nb.add(page2, text='Synthesise')

        page2.rowconfigure(5, weight=1)
        page2.columnconfigure(6, weight=1)

        ## SELECTING THE PROGRAM
        self.program.set(1)
        Label(page2, text="Select the program: ", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, pady=4)
        Radiobutton(page2, text="Prism", variable=self.program, value="prism").grid(row=1, column=1, sticky=W, pady=4)
        Radiobutton(page2, text="Storm", variable=self.program, value="storm").grid(row=1, column=2, sticky=W, pady=4)
        Button(page2, text='Run parameter synthesis', command=self.synth_params).grid(row=2, column=0, sticky=W, pady=4, padx=4)
        Button(page2, text='Load results', command=self.load_functions).grid(row=2, column=1, sticky=W, pady=4)

        Label(page2, text=f"Loaded function file:", anchor=W, justify=LEFT).grid(row=3, column=0, sticky=W, pady=4)

        self.functions_text = scrolledtext.ScrolledText(page2, height=100)
        self.functions_text.grid(row=4, column=0, columnspan=16, rowspan=2, sticky=W, pady=4)

        Label(page2, text=f"Show function:", anchor=W, justify=LEFT).grid(row=2, column=17, sticky=W, pady=4)
        Radiobutton(page2, text="Original", variable=self.factor, value=False).grid(row=2, column=18, sticky=W, pady=4)
        Radiobutton(page2, text="Factorised", variable=self.factor, value=True).grid(row=2, column=19, sticky=W, pady=4)
        Label(page2, text=f"Parsed functions:", anchor=W, justify=LEFT).grid(row=3, column=17, sticky=W, pady=4)
        self.functions_parsed_text = scrolledtext.ScrolledText(page2, height=100)
        self.functions_parsed_text.grid(row=4, column=17, columnspan=16, rowspan=2, sticky=W, pady=4)


        ## TAB SAMPLE AND VISUALISE
        page3 = ttk.Frame(nb, width=400, height=200, name="sampling")
        nb.add(page3, text='Sampling functions')

        page3.rowconfigure(5, weight=1)
        page3.columnconfigure(6, weight=1)

        Label(page3, text="Set size_q, number of samples per dimension:", anchor=W, justify=LEFT).grid(row=1, column=0, padx=4, pady=4)
        self.fun_size_q_entry = Entry(page3)
        self.fun_size_q_entry.grid(row=1, column=1)

        Button(page3, text='Sample functions', command=self.sample_fun).grid(row=2, column=0, sticky=W, pady=4, padx=4)

        Label(page3, text=f"Sampled functions:", anchor=W, justify=LEFT).grid(row=3, column=0, sticky=W, pady=4)

        self.sampled_functions_text = scrolledtext.ScrolledText(page3, height=100)
        self.sampled_functions_text.grid(row=4, column=0, columnspan=16, rowspan=2, sticky=W, pady=4)

        Button(page3, text='Show functions in a single point', command=self.show_funs).grid(row=2, column=17, sticky=W, pady=4, padx=4)
        Button(page3, text='Show all sampled points', command=self.show_funs_in_all_points).grid(row=3, column=17, sticky=W, pady=4, padx=4)
        self.Next_sample = Button(page3, text="Next", state="disabled", command=lambda: self.button_pressed.set(True))
        self.Next_sample.grid(row=3, column=18, sticky=W, pady=4, padx=4)

        self.page3_plotframe = Frame(page3)
        self.page3_plotframe.grid(row=4, column=17, columnspan=2, sticky=W, pady=4, padx=4)
        self.page3_figure = pyplt.figure()
        self.page3_a = self.page3_figure.add_subplot(111)
        # print("type a", type(self.a))

        self.page3_canvas = FigureCanvasTkAgg(self.page3_figure, master=self.page3_plotframe)  # A tk.DrawingArea.
        self.page3_canvas.draw()
        self.page3_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.page3_toolbar = NavigationToolbar2Tk(self.page3_canvas, self.page3_plotframe)
        self.page3_toolbar.update()
        self.page3_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.page3_figure_locked = BooleanVar()
        self.page3_figure_locked.set(False)


        ## TAB DATA CONVERSION
        page4 = ttk.Frame(nb, width=400, height=200, name="conversion")
        nb.add(page4, text='Data conversion')
        # page4.columnconfigure(0, weight=1)
        # page4.rowconfigure(2, weight=1)
        # page4.rowconfigure(7, weight=1)

        Button(page4, text='Load data', command=self.load_data).grid(row=0, column=0, sticky=W, pady=4)

        Label(page4, text=f"Data:", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, pady=4)

        self.data_text = Text(page4, height=12)  # , height=10, width=30
        # self.data_text.config(state="disabled")
        self.data_text.grid(row=2, column=0, columnspan=2, sticky=W, pady=4)

        ## SET THE INTERVAL COMPUTATION SETTINGS
        Label(page4, text="Set alpha, the confidence:", anchor=W, justify=LEFT).grid(row=3)
        Label(page4, text="Set n_samples, number of samples: ", anchor=W, justify=LEFT).grid(row=4)

        self.alpha_entry = Entry(page4)
        self.n_samples_entry = Entry(page4)

        self.alpha_entry.grid(row=3, column=1)
        self.n_samples_entry.grid(row=4, column=1)

        self.alpha_entry.insert(END, '0.95')
        self.n_samples_entry.insert(END, '100')

        ## TBD ADD setting for creating  intervals - alpha, n_samples
        Button(page4, text='Create intervals', command=self.create_intervals).grid(row=5, column=0, sticky=W, pady=4)

        Label(page4, text=f"Intervals:", anchor=W, justify=LEFT).grid(row=6, column=0, sticky=W, pady=4)

        self.interval_text = Text(page4, height=12)  #height=10, width=30
        # self.interval_text.config(state="disabled")
        self.interval_text.grid(row=7, column=0, columnspan=2, sticky=W, pady=4)


        ## TAB PROPS
        page5 = ttk.Frame(nb, width=400, height=200, name="props")
        nb.add(page5, text='Props')

        Button(page5, text='Show props', command=self.show_props).grid(row=0, column=0, sticky=W, pady=4)
        Button(page5, text='Load props', command=self.load_props).grid(row=0, column=1, sticky=W, pady=4)
        Button(page5, text='Append props', command=self.append_props).grid(row=0, column=2, sticky=W, pady=4)

        self.props_text = scrolledtext.ScrolledText(page5, height=100)
        self.props_text.grid(row=1, column=0, columnspan=16, rowspan=2, sticky=W+E+N+S, pady=4)  # pack(anchor=W, fill=X)


        ## TAB SAMPLE AND REFINEMENT
        page6 = ttk.Frame(nb, width=400, height=200, name="refine")
        nb.add(page6, text='Sample & Refine')

        frame_left = Frame(page6, width=200, height=200)
        frame_left.pack(side=LEFT, fill=X)

        Button(frame_left, text='Load space', command=self.load_space).grid(row=0, column=0, sticky=W, pady=4)
        Button(frame_left, text='Delete space', command=self.refresh_space).grid(row=0, column=1, sticky=W, pady=4)

        ttk.Separator(frame_left, orient=HORIZONTAL).grid(row=1, column=0, columnspan=7, sticky='nwe', pady=8)

        Label(frame_left, text="Set size_q: ", anchor=W, justify=LEFT).grid(row=1, pady=16)

        self.size_q_entry = Entry(frame_left)
        self.size_q_entry.grid(row=1, column=1)
        self.size_q_entry.insert(END, '5')

        Button(frame_left, text='Sample space', command=self.sample_space).grid(row=6, column=0, sticky=W, padx=4, pady=4)

        ttk.Separator(frame_left, orient=VERTICAL).grid(column=2, row=1, rowspan=6, sticky='ns', padx=10, pady=8)

        Label(frame_left, text="Set max_dept: ", anchor=W, justify=LEFT).grid(row=1, column=3, padx=10)
        Label(frame_left, text="Set coverage: ", anchor=W, justify=LEFT).grid(row=2, column=3, padx=10)
        Label(frame_left, text="Set epsilon: ", anchor=W, justify=LEFT).grid(row=3, column=3, padx=10)
        Label(frame_left, text="Set algorithm: ", anchor=W, justify=LEFT).grid(row=4, column=3, padx=10)

        self.max_dept_entry = Entry(frame_left)
        self.coverage_entry = Entry(frame_left)
        self.epsilon_entry = Entry(frame_left)
        self.algorithm_entry = Entry(frame_left)

        self.max_dept_entry.grid(row=1, column=4)
        self.coverage_entry.grid(row=2, column=4)
        self.epsilon_entry.grid(row=3, column=4)
        self.algorithm_entry.grid(row=4, column=4)

        self.max_dept_entry.insert(END, '5')
        self.coverage_entry.insert(END, '0.95')
        self.epsilon_entry.insert(END, '0')
        self.algorithm_entry.insert(END, '4')

        self.save_sample = BooleanVar()
        c = Checkbutton(frame_left, text="Save results", variable=self.save_sample)
        c.grid(row=5, column=0, sticky=W, padx=4, pady=4)

        self.save_refinement = BooleanVar()
        c = Checkbutton(frame_left, text="Save results", variable=self.save_refinement)
        c.grid(row=5, column=3, sticky=W, pady=4, padx=10)

        Button(frame_left, text='Refine space', command=self.refine_space).grid(row=6, column=3, sticky=W, pady=4, padx=10)

        ttk.Separator(frame_left, orient=HORIZONTAL).grid(row=7, column=0, columnspan=7, sticky='nwe', pady=4)

        frame_left.rowconfigure(13, weight=1)
        frame_left.columnconfigure(16, weight=1)

        self.space_text = scrolledtext.ScrolledText(frame_left, height=100)
        self.space_text.grid(row=12, column=0, columnspan=16, rowspan=2, sticky=W, pady=4)  # pack(anchor=W, fill=X)

        frame_right = Frame(page6, width=200, height=200)
        frame_right.pack(side=TOP, fill=X)

        self.page6_plotframe = Frame(frame_right)
        self.page6_plotframe.pack(fill=X)
        self.page6_figure = pyplt.figure()

        # print("type a", type(self.a))

        self.page6_canvas = FigureCanvasTkAgg(self.page6_figure, master=self.page6_plotframe)  # A tk.DrawingArea.
        self.page6_canvas.draw()
        self.page6_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.page6_toolbar = NavigationToolbar2Tk(self.page6_canvas, self.page6_plotframe)
        self.page6_toolbar.update()
        self.page6_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.page6_a = self.page6_figure.add_subplot(111)

        # page7 = ttk.Frame(nb, name="testy")
        # # page7.pack(expand=True)
        # nb.add(page7, text='testy')
        #
        # self.testy_text = scrolledtext.ScrolledText(page7, height=100)
        # # self.testy_text.config(state="disabled")
        # # self.testy_text.grid(row=0, column=0, sticky=W + E + N + S, pady=4)
        # Button(page7, text='Load results', command=self.load_functions).pack()
        # self.testy_text.pack()
        #
        # page7 = ttk.Frame(nb, name="testyy")
        # # page7.pack(expand=True)
        # nb.add(page7, text='testy')
        #
        # self.testy_text2 = scrolledtext.ScrolledText(page7, height=100)
        # # self.testy_text.config(state="disabled")
        # # self.testy_text.grid(row=0, column=0, sticky=W + E + N + S, pady=4)
        # Button(page7, text='Load results', command=self.load_functions).grid()
        # self.testy_text2.grid()

        ## MENU
        main_menu = Menu(self)
        self.config(menu=main_menu)

        ## MENU-FILE
        file_menu = Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="File", menu=file_menu)

        ## MENU-FILE-LOAD
        # load_menu = Menu(file_menu, tearoff=0)
        # file_menu.add_cascade(label="Load", menu=load_menu, underline=0)
        # load_menu.add_command(label="Load model", command=self.load_model)
        # load_menu.add_command(label="Load property", command=self.load_property)
        # load_menu.add_command(label="Load rational functions", command=self.load_functions)
        # load_menu.add_command(label="Load data", command=self.load_data)
        # load_menu.add_command(label="Load space", command=self.load_space)
        # file_menu.add_separator()

        ## MENU-FILE-SAVE
        # save_menu = Menu(file_menu, tearoff=0)
        # file_menu.add_cascade(label="Save", menu=save_menu, underline=0)
        # save_menu.add_command(label="Save model", command=self.save_model)
        # save_menu.add_command(label="Save property", command=self.save_property)
        # # save_menu.add_command(label="Save rational functions", command=self.save_functions())  ## TBD MAYBE IN THE FUTURE
        # save_menu.add_command(label="Save data", command=self.save_data)
        # save_menu.add_command(label="Save space", command=self.save_space)
        # file_menu.add_separator()

        ## MENU-FILE-EXIT
        file_menu.add_command(label="Exit", command=self.quit)

        ## MENU-EDIT
        # edit_menu = Menu(main_menu, tearoff=0)
        # main_menu.add_cascade(label="Edit", menu=edit_menu)

        ## MENU-SHOW
        # show_menu = Menu(main_menu, tearoff=0)
        # main_menu.add_cascade(label="Show", menu=show_menu)
        # show_menu.add_command(label="Space", command=self.show_space)

        ## MENU-ANALYSIS
        # analysis_menu = Menu(main_menu, tearoff=0)
        # main_menu.add_cascade(label="Analysis", menu=analysis_menu)
        # analysis_menu.add_command(label="Synthesise parameters", command=self.synth_params)
        # analysis_menu.add_command(label="Create intervals", command=self.create_intervals)
        # analysis_menu.add_command(label="Sample space", command=self.sample_space)
        # analysis_menu.add_command(label="Refine space", command=self.refine_space)

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

    def report_callback_exception(self, exc, val, tb):
        """Report callback exception on sys.stderr.

        Applications may want to override this internal function, and
        should when sys.stderr is None."""
        import traceback
        print("Exception in Tkinter callback", file=sys.stderr)
        sys.last_type = exc
        sys.last_value = val
        sys.last_traceback = tb
        traceback.print_exception(exc, val, tb)
        messagebox.showerror("Error", message=str(val))

    def load_config(self):
        os.chdir(workspace)
        config.read(os.path.join(workspace, "../config.ini"))

        self.model_dir = Path(config.get("paths", "models"))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.property_dir = Path(config.get("paths", "properties"))
        if not os.path.exists(self.property_dir):
            os.makedirs(self.property_dir)

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
    ## FILE - LOAD AND SAVE
    def load_model(self):
        print("Loading model ...")
        if self.model_changed:
            if not askyesno("Loading model", "Previously obtained model will be lost. Do you want to proceed?"):
                return
        self.status_set("Please select the model to be loaded.")

        spam = filedialog.askopenfilename(initialdir=self.model_dir, title="Model loading - Select file",
                                          filetypes=(("pm files", "*.pm"), ("all files", "*.*")))
        ## If no file selected
        if spam == "":
            self.status_set("No file selected.")
            return
        else:
            self.model_changed = True
            self.model_file.set(spam)
            self.model_text.delete('1.0', END)
            self.model_text.insert('end', open(self.model_file.get(), 'r').read())

            self.status_set("Model loaded.")
            # print("self.model", self.model.get())

    def load_property(self):
        print("Loading properties ...")
        if self.property_changed:
            if not askyesno("Loading properties", "Previously obtained properties will be lost. Do you want to proceed?"):
                return
        self.status_set("Please select the property to be loaded.")

        spam = filedialog.askopenfilename(initialdir=self.property_dir, title="Property loading - Select file",
                                          filetypes=(("property files", "*.pctl"), ("all files", "*.*")))
        ## If no file selected
        if spam == "":
            self.status_set("No file selected.")
            return
        else:
            self.property_changed = True
            self.property_file.set(spam)
            self.property_text.delete('1.0', END)
            self.property_text.insert('end', open(self.property_file.get(), 'r').read())
            self.status_set("Property loaded.")
            # print("self.property", self.property.get())

    def load_data(self):
        print("Loading data ...")
        if self.data_changed:
            if not askyesno("Loading data", "Previously obtained data will be lost. Do you want to proceed?"):
                return

        self.status_set("Please select the data to be loaded.")

        spam = filedialog.askopenfilename(initialdir=self.data_dir, title="Data loading - Select file",
                                          filetypes=(("pickled files", "*.p"), ("all files", "*.*")))
        ## If no file selected
        if spam == "":
            self.status_set("No file selected.")
            return
        else:
            self.data_changed = True
            self.data_file.set(spam)

            if ".p" in self.data_file.get():
                self.data = pickle.load(open(self.data_file.get(), "rb"))
                self.unfold_data()
            else:
                print()
                ## TBD
                # self.data = PARSE THE DATA
            # print(self.data)
            self.status_set("Data loaded.")

    def unfold_data(self):
        """" unfolds the data dictionary into a single list"""
        if isinstance(self.data, dict):
            ## TBD Maybe rewrite this as key and pass the argument to unfold_data2
            self.key = StringVar()
            self.status_set(
                "Loaded data are in a form of dictionary, please select which item you would like to choose:")
            self.new_window = Toplevel(self)
            ## SCROLABLE WINDOW
            canvas = Canvas(self.new_window)
            canvas.pack(side=LEFT)
            self.new_window.maxsize(800, 800)

            scrollbar = Scrollbar(self.new_window, command=canvas.yview)
            scrollbar.pack(side=LEFT, fill='y')

            canvas.configure(yscrollcommand=scrollbar.set)

            def on_configure(event):
                canvas.configure(scrollregion=canvas.bbox('all'))

            canvas.bind('<Configure>', on_configure)
            frame = Frame(canvas)
            canvas.create_window((0, 0), window=frame, anchor='nw')

            label = Label(frame, text="Loaded data are in a form of dictionary, please select which item you would like to choose:")
            label.pack()
            self.key.set(" ")

            first = True
            for key in self.data.keys():
                spam = Radiobutton(frame, text=key, variable=self.key, value=key)
                spam.pack(anchor=W)
                if first:
                    spam.select()
                    first = False
            spam = Button(frame, text="OK", command=self.unfold_data2)
            spam.pack()
        else:
            self.data_text.delete('1.0', END)
            self.data_text.insert('end', str(self.data))

    def unfold_data2(self):
        """" dummy method of unfold_data"""
        try:
            self.data = self.data[self.key.get()]
        except KeyError:
            self.data = self.data[eval(self.key.get())]

        print(self.data)
        self.new_window.destroy()
        self.unfold_data()

    def load_functions(self, file=False):
        """ Load functions

        Args
        -------------
        file (Path/String): direct path to load the function file
        """
        print("Loading functions ...")
        if self.functions_changed:
            if not askyesno("Loading functions", "Previously obtained functions will be lost. Do you want to proceed?"):
                return

        self.status_set("Loading functions - checking inputs")

        print("self.program.get()", self.program.get())
        if self.program.get() == "prism":
            initialdir = self.prism_results
        elif self.program.get() == "storm":
            initialdir = self.storm_results
        else:
            messagebox.showwarning("Load functions", "Select a program for which you want to load data.")
            return

        ## If file to load is NOT preselected
        print(file)
        if not file:
            self.status_set("Please select the prism/storm symbolic results to be loaded.")
            spam = filedialog.askopenfilename(initialdir=initialdir, title="Rational functions loading - Select file",
                                              filetypes=(("text files", "*.txt"), ("all files", "*.*")))
        else:
            if os.path.isfile(file):
                spam = str(file)
            else:
                spam = ""

        ## If no file / not a file selected
        if spam == "":
            self.status_set("No file selected.")
            return
        # print("self.functions_file.get() ", self.functions_file.get())
        self.functions_file.set(spam)
        # print("self.functions_file.get() ", self.functions_file.get())
        if not self.functions_file.get() is "":
            self.functions_changed = True
            # self.model_changed = False
            # self.property_changed = False
        # print("self.functions_changed", self.functions_changed)

        # print("self.factor", self.factor.get())
        self.functions, rewards = load_all_functions(self.functions_file.get(), tool=self.program.get(),
                                                     factorize=self.factor.get(), agents_quantities=False,
                                                     rewards_only=False, f_only=False)
        ## Merge functions and rewards
        # print("self.functions", self.functions)
        # print("rewards", rewards)
        for key in self.functions.keys():
            if key in rewards.keys():
                self.functions[key].extend(rewards[key])
        print("self.functions", self.functions)

        self.unfold_functions()

        self.status_set(f"{len(self.functions.keys())} rational functions loaded")

        ## Show loaded functions
        self.functions_text.delete('1.0', END)
        self.functions_text.insert('1.0', open(self.functions_file.get(), 'r').read())

        # self.testy_text.delete('1.0', END)
        # self.testy_text.insert('1.0', open(self.functions_file.get(), 'r').read())
        #
        # self.testy_text2.delete('1.0', END)
        # self.testy_text2.insert('1.0', open(self.functions_file.get(), 'r').read())

    def unfold_functions(self):
        """" unfolds the function dictionary into a single list """
        if isinstance(self.functions, dict):
            ## TBD Maybe rewrite this as key and pass the argument to unfold_functions2
            ## NO because dunno how to send it to the function as a argument
            self.key = StringVar()
            self.status_set(
                "Loaded functions are in a form of dictionary, please select which item you would like to choose:")
            self.functions_window = Toplevel(self)
            label = Label(self.functions_window,
                          text="Loaded functions are in a form of dictionary, please select which item you would like to choose:")
            label.pack()
            self.key.set(" ")

            first = True
            for key in self.functions.keys():
                spam = Radiobutton(self.functions_window, text=key, variable=self.key, value=key)
                spam.pack(anchor=W)
                if first:
                    spam.select()
                    first = False
            spam = Button(self.functions_window, text="OK", command=self.unfold_functions2)
            spam.pack()
        else:
            self.functions_parsed_text.delete('1.0', END)
            self.functions_parsed_text.insert('end', str(self.functions))

    def unfold_functions2(self):
        """" dummy method of unfold_functions"""
        try:
            self.functions = self.functions[self.key.get()]
        except KeyError:
            self.functions = self.functions[eval(self.key.get())]

        print(self.functions)
        self.functions_window.destroy()
        self.unfold_functions()

    def show_props(self):
        self.validate_props(position="Props")

    def load_props(self, append=False):
        print("Loading props ...")
        if self.props_changed and not append:
            if not askyesno("Loading props", "Previously obtained props will be lost. Do you want to proceed?"):
                return
        self.status_set("Please select the props to be loaded.")
        spam = filedialog.askopenfilename(initialdir=self.data_dir, title="Props loading - Select file", filetypes=(("text files", "*.txt"), ("all files", "*.*")))

        print("old props", self.props)
        print("old props type", type(self.props))
        print("loaded props file", spam)

        ## If no file selected
        if spam == "":
            self.status_set("No file selected.")
            return
        else:
            self.props_changed = True
            self.props_file.set(spam)

            if append:
                if self.props == "":
                    print("was here")
                    self.props = []
                    print("old props", self.props)
                    print("old props type", type(self.props))
                with open(self.props_file.get(), 'r') as file:
                    for line in file:
                        print(line)
                        self.props.append(line)
            else:
                self.props = []
                with open(self.props_file.get(), 'r') as file:
                    for line in file:
                        print(line[:-1])
                        self.props.append(line[:-1])
            print("self.props", self.props)

            self.props_text.delete('1.0', END)
            self.props_text.insert('end', str(self.props))
            self.status_set("Props loaded")

    def append_props(self):
        self.load_props(append=True)

    def load_space(self):
        print("Loading space ...")
        if self.space_changed:
            if not askyesno("Loading space", "Previously obtained space will be lost. Do you want to proceed?"):
                return
        self.status_set("Please select the space to be loaded.")
        spam = filedialog.askopenfilename(initialdir=self.data_dir, title="Space loading - Select file", filetypes=(("pickled files", "*.p"), ("all files", "*.*")))
        # print(self.space)

        ## If no file selected
        if spam == "":
            self.status_set("No file selected.")
            return
        else:
            self.space_changed = True
            self.space_file.set(spam)

            self.space = pickle.load(open(self.space_file.get(), "rb"))

            ## Show the space as niceprint()
            print("space", self.space)
            print()
            print("space nice print \n", self.space.nice_print())
            self.space_text.delete('1.0', END)
            self.space_text.insert('end', self.space.nice_print())
            self.status_set("Space loaded")

    def save_model(self):
        ## TBD CHECK IF THE MODEL IS NON EMPTY
        # if len(self.model_text.get('1.0', END)) <= 1:
        #    self.status_set("There is no model to be saved.")
        #    return

        print("Saving the model ...")
        self.status_set("Please select folder to store the model in.")
        save_model = filedialog.asksaveasfilename(initialdir=self.model_dir, title="Model saving - Select file",
                                                  filetypes=(("pm files", "*.pm"), ("all files", "*.*")))
        if save_model == "":
            self.status_set("No file selected.")
            return

        if "." not in save_model:
            save_model = save_model + ".pm"
        # print("save_model", save_model)

        with open(save_model, "w") as file:
            file.write(self.model_text.get(1.0, END))

        self.status_set("Model saved.")

    def save_property(self):
        print("Saving the property ...")
        ## TBD CHECK IF THE PROPERTY IS NON EMPTY
        # if len(self.property_text.get('1.0', END)) <= 1:
        #    self.status_set("There is no property to be saved.")
        #    return

        self.status_set("Please select folder to store the property in.")
        save_property = filedialog.asksaveasfilename(initialdir=self.property_dir, title="Property saving - Select file",
                                                     filetypes=(("pctl files", "*.pctl"), ("all files", "*.*")))
        if save_property == "":
            self.status_set("No file selected.")
            return

        if "." not in save_property:
            save_property = save_property + ".pctl"
        # print("save_property", save_property)

        with open(save_property, "w") as file:
            file.write(self.property_text.get(1.0, END))

        self.status_set("Property saved.")

    ## TBD MAYBE IN THE FUTURE
    def save_functions(self):
        print("Saving the functions ...")
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
            self.status_set("Error - Selected program not recognised.")
            save_functions = "Error - Selected program not recognised."
        print(save_functions)

        if save_functions == "":
            self.status_set("No file selected.")
            return

        with open(save_functions, "w") as file:
            for line in self.props:
                file.write(line)
        self.status_set("Property saved.")

    def save_data(self):
        print("Saving the data ...")
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
        print("Saving the space ...")
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
        print("Synthesising parameters ...")
        proceed = True
        if self.functions_changed:
            proceed = askyesno("Parameter synthesis", "Synthesising the parameters will overwrite current functions. Do you want to proceed?")

        if proceed:
            self.status_set("Parameter synthesis - checking inputs")

            if self.model_changed:
                messagebox.showwarning("Parameter synthesis", "The model for parameter synthesis has changed in the mean time, please consider that.")
            if self.property_changed:
                messagebox.showwarning("Parameter synthesis", "The properties for parameter synthesis have changed in the mean time, please consider that.")
            ## If model file not selected load model
            if self.model_file.get() is "":
                self.status_set("Load model for parameter synthesis")
                self.load_model()

            ## If property file not selected load property
            if self.property_file.get() is "":
                self.status_set("Load property for parameter synthesis")
                self.load_property()

            if self.program.get().lower() == "prism":
                self.status_set("Parameter synthesis is running ...")
                call_prism_files(self.model_file.get(), [], param_intervals=False, seq=False, noprobchecks=False, memory="",
                                 model_path="", properties_path=self.property_dir, property_file=self.property_file.get(),
                                 output_path=self.prism_results)
                ## Deriving output file
                self.functions_file.set(str(os.path.join(Path(self.prism_results), str(Path(self.model_file.get()).stem)+"_"+str(Path(self.property_file.get()).stem)+".txt")))
                self.status_set("Parameter synthesised finished. Output here: {}", self.functions_file.get())
                self.load_functions(self.functions_file.get())
                # self.functions_text.delete('1.0', END)
                # self.functions_text.insert('1.0', open(self.functions_file.get(), 'r').read())

            elif self.program.get().lower() == "storm":
                self.status_set("Parameter synthesis running ...")
                call_storm_files(self.model_file.get(), [], model_path="", properties_path=self.property_dir,
                                 property_file=self.property_file.get(), output_path=self.storm_results, time=False)
                ## Deriving output file
                self.functions_file.set(str(os.path.join(Path(self.storm_results), str(Path(self.model_file.get()).stem) + "_" + str(Path(self.property_file.get()).stem) + ".cmd")))
                self.status_set("Command to run the parameter synthesis saved here: {}", self.functions_file.get())
                self.load_functions(self.functions_file.get())
                # self.functions_text.delete('1.0', END)
                # self.functions_text.insert('1.0', open(self.functions_file.get(), 'r').read())
            else:
                ## Show window to inform to select the program
                self.status_set("Program for parameter synthesis not selected")
                messagebox.showwarning("Synthesise", "Select a program for parameter synthesis first.")
                return

    def sample_fun(self):
        """Sampling rational functions"""
        print("Sampling rational functions ...")
        self.status_set("Sampling rational functions. - checking inputs")
        if self.fun_size_q_entry.get() == "":
            messagebox.showwarning("Sampling rational functions", "Choose size_q, number of samples per dimension.")
            return
        if self.functions == "":
            messagebox.showwarning("Sampling rational functions", "Load the functions first, please")
            return

        self.status_set("Sampling rational functions.")

        ## TBD If self.functions got more than one entry
        self.sampled_functions = sample_fun(self.functions, int(self.fun_size_q_entry.get()), debug=True)
        self.sampled_functions_text.delete('1.0', END)
        self.sampled_functions_text.insert('1.0', "rational function index, [parameter values], function value: \n")
        spam = ""
        for item in self.sampled_functions:
            spam = spam + str(item) + ",\n"
        self.sampled_functions_text.insert('2.0', spam[:-2])

    def show_funs(self):
        """Showing sampled rational functions in a single point"""
        print("Showing sampled rational functions ...")
        self.status_set("Showing sampled rational functions.")

        ## Disable overwriting the plot by show_funs_in_all_points
        if self.page3_figure_locked.get():
            if not askyesno("Show functions in a single point", "The result plot is currently in use. Do you want override?"):
                return
        self.page3_figure_locked.set(True)

        if self.functions == "":
            messagebox.showwarning("Sampling rational functions", "Load the functions first, please")
            return

        ## TBD Maybe rewrite this as key and pass the argument to load_param_intervals
        self.key = StringVar()
        self.status_set("Choosing parameters value:")
        self.new_window = Toplevel(self)
        label = Label(self.new_window,
                      text="Please choose value of respective parameter:")
        label.grid(row=0)
        self.key.set(" ")

        globals()["parameters"] = set()
        for polynome in self.functions:
            globals()["parameters"].update(find_param(polynome))
        globals()["parameters"] = sorted(list(globals()["parameters"]))
        self.parameters = globals()["parameters"]
        print("self.parameters", self.parameters)

        ## Parse parameters values
        self.parameter_values = []
        i = 1
        ## For each param create an entry
        for param in self.parameters:
            Label(self.new_window, text=param, anchor=W, justify=LEFT).grid(row=i, column=0)
            spam = Entry(self.new_window)
            spam.grid(row=i, column=1)
            spam.insert(END, '0')
            self.parameter_values.append(spam)
            i = i + 1

        ## To be used to wait until the button is pressed
        self.button_pressed.set(False)
        load_param_values_button = Button(self.new_window, text="OK", command=self.load_param_values)
        load_param_values_button.grid(row=i)

        ## Waiting for the pop-up window closing
        load_param_values_button.wait_variable(self.button_pressed)
        print("key pressed")

        ## TBD If self.functions got more than one entry
        ## Getting the plot values instead of the plot itself
        ax = eval_and_show(self.functions, self.parameter_values, give_back=True)

        self.page3_figure.clf()
        self.page3_a = self.page3_figure.add_subplot(111)
        # print("setting values", ax)
        self.page3_a.set_ylabel(ax[3])
        self.page3_a.set_xlabel(ax[4])
        self.page3_a.set_title(ax[5])
        self.page3_a.bar(ax[0], ax[1], ax[2], color='b')

        self.page3_figure.canvas.draw()
        self.page3_figure.canvas.flush_events()

    def show_funs_in_all_points(self):
        """Showing sampled rational functions in all sampled points"""
        def onclick(event):
            self.button_pressed.set(True)

        print("Showing sampled rational functions ...")
        self.status_set("Showing sampled rational functions.")

        if self.page3_figure_locked.get():
            if not askyesno("Show all sampled points", "The result plot is currently in use. Do you want override?"):
                return
        self.page3_figure_locked.set(False)

        if self.fun_size_q_entry.get() == "":
            messagebox.showwarning("Sampling rational functions", "Choose size_q, number of samples per dimension.")
            return

        if self.functions == "":
            messagebox.showwarning("Sampling rational functions", "Load the functions first, please")
            return

        ## TBD Maybe rewrite this as key and pass the argument to load_param_intervals

        ## TBD If self.functions got more than one entry
        self.button_pressed.set(False)
        self.page3_figure.canvas.mpl_connect('button_press_event', onclick)

        if not self.parameters:
            globals()["parameters"] = set()
            for polynome in self.functions:
                globals()["parameters"].update(find_param(polynome))
            globals()["parameters"] = sorted(list(globals()["parameters"]))
            self.parameters = globals()["parameters"]
            print("self.parameters", self.parameters)

        self.Next_sample.config(state="normal")
        for parameter_point in get_param_values(self.parameters, self.fun_size_q_entry.get(), False):
            ## If
            if self.page3_figure_locked.get():
                return
            ax = eval_and_show(self.functions, parameter_point, give_back=True)

            self.page3_figure.clf()

            self.page3_a = self.page3_figure.add_subplot(111)
            # print("setting values", ax)
            self.page3_a.set_ylabel(ax[3])
            self.page3_a.set_xlabel(ax[4])
            self.page3_a.set_title(ax[5])
            self.page3_a.bar(ax[0], ax[1], ax[2], color='b')

            self.page3_figure.canvas.draw()
            self.page3_figure.canvas.flush_events()

            self.Next_sample.wait_variable(self.button_pressed)
        self.Next_sample.config(state="disabled")
        self.page3_figure_locked.set(False)

    def create_intervals(self):
        """Creates intervals from data"""
        print("Creating intervals ...")
        self.status_set("Create interval - checking inputs")
        if self.alpha_entry.get() == "":
            messagebox.showwarning("Creating intervals", "Choose alpha, the confidence measure before creating intervals.")
            return

        if self.n_samples_entry.get() == "":
            messagebox.showwarning("Creating intervals", "Choose n_samples, number of experimental samples before creating intervals")
            return

        ## If data file not selected load data
        if self.data_file.get() is "":
            self.load_data()

        print(self.data_file.get())
        self.status_set("Intervals are being created ...")
        self.intervals = create_intervals(float(self.alpha_entry.get()), float(self.n_samples_entry.get()), self.data)
        self.interval_text.delete('1.0', END)
        self.interval_text.insert('end', self.intervals)
        self.status_set("Intervals created.")

        self.intervals_changed = True

    def validate_props(self, position=False):
        """ Validating created properties

        Args:
        position: (String) Name of the place from which is being called e.g. "Refine Space"/"Sample space"
        """
        print("Validating properties ...")
        if position is False:
            position = "Validating props"
        ## If props empty create props
        if self.props == "" or self.functions_changed or self.intervals_changed:
            print("Validating props")
            print("self.functions", self.functions)
            print("self.intervals", self.intervals)
            ## If functions empty raise an error (return False)
            if self.functions == "":
                print("No functions loaded nor not computed to create properties")
                messagebox.showwarning(position, "Load or synthesise functions first.")
                return False
            ## If intervals empty raise an error (return False)
            if self.intervals == "":
                print("Intervals not computed, properties cannot be computed")
                messagebox.showwarning(position, "Compute intervals first.")
                return False

            if self.functions_changed:
                self.functions_changed = False

            if self.intervals_changed:
                self.intervals_changed = False

            ## Create props
            self.props = ineq_to_props(self.functions, self.intervals, silent=True)
            self.props_changed = True

            self.props_text.delete('1.0', END)
            self.props_text.insert('end', str(self.props))
            print("self.props", self.props)
        return True

    def refresh_space(self):
        """Unloads space"""
        if askyesno("Sample & Refine", "Data of the space, its text representation, and the plot will be lost. Do you want to proceed?"):
            self.space = ""
            self.space_changed = False
            self.space_text.delete('1.0', END)
            self.page6_figure.clf()
            self.page6_a = self.page6_figure.add_subplot(111)
            self.page6_figure.canvas.draw()
            self.page6_figure.canvas.flush_events()
            self.status_set("Space deleted.")

    def validate_space(self, position=False):
        """ Checking validity of the space

        Args:
        position: (String) Name of the place from which is being called e.g. "Refine Space"/"Sample space"
        """
        print("Checking space ...")
        if position is False:
            position = "Validating space"
        ## If the space is empty create a new one
        if self.space == "":
            print("space is empty - creating a new one")
            ## Parse params from props
            globals()["parameters"] = set()
            for polynome in self.props:
                globals()["parameters"].update(find_param(polynome))
            globals()["parameters"] = sorted(list(globals()["parameters"]))
            self.parameters = globals()["parameters"]
            print("self.parameters", self.parameters)

            ## TBD Maybe rewrite this as key and pass the argument to load_param_intervals
            self.key = StringVar()
            self.status_set("Choosing ranges of parameters:")
            self.new_window = Toplevel(self)
            label = Label(self.new_window,
                          text="Please choose intervals of the parameters to be used:")
            label.grid(row=0)
            self.key.set(" ")

            ## Parse parameters intervals - region
            self.parameter_intervals = []
            i = 1
            ## For each param create an entry
            for param in self.parameters:
                Label(self.new_window, text=param, anchor=W, justify=LEFT).grid(row=i, column=0)
                spam_low = Entry(self.new_window)
                spam_high = Entry(self.new_window)
                spam_low.grid(row=i, column=1)
                spam_high.grid(row=i, column=2)
                spam_low.insert(END, '0')
                spam_high.insert(END, '1')
                self.parameter_intervals.append([spam_low, spam_high])
                i = i + 1

            ## To be used to wait until the button is pressed
            self.button_pressed.set(False)
            load_param_intervals_button = Button(self.new_window, text="OK", command=self.load_param_intervals)
            load_param_intervals_button.grid(row=i)

            load_param_intervals_button.wait_variable(self.button_pressed)
            print("key pressed")
        else:
            if self.props_changed:
                ## TBD show warning that you are using old space for computation ## HERE
                messagebox.showwarning(position, "Using previously created space with new props. Consider using fresh new space.")
                ## Check if the properties and data are valid
                globals()["parameters"] = set()
                for polynome in self.props:
                    globals()["parameters"].update(find_param(polynome))
                globals()["parameters"] = sorted(list(globals()["parameters"]))
                self.parameters = globals()["parameters"]

                if not len(self.space.params) == len(self.parameters):
                    messagebox.showerror(position, "Cardinality of the space does not correspond to the props. Consider using fresh space.")
                    return False
                elif not sorted(self.space.params) == sorted(self.parameters):
                    messagebox.showerror(position, f"Parameters of the space - {self.space.params} - does not correspond to the one in props - {self.parameters}. Consider using fresh space.")
                    return False
        return True

    # def key_pressed_callback(self):
    #     self.load_param_intervals()

    def load_param_intervals(self):
        region = []
        for param_index in range(len(self.parameters)):
            ## Getting the values from each entry, low = [0], high = [1]
            region.append([float(self.parameter_intervals[param_index][0].get()), float(self.parameter_intervals[param_index][1].get())])
        print("region", region)
        del self.key
        self.new_window.destroy()
        del self.new_window
        # del self.parameter_intervals
        self.space = space.RefinedSpace(region, self.parameters)
        self.button_pressed.set(True)
        print("self.space", self.space)

    def load_param_values(self):
        for param_index in range(len(self.parameter_values)):
            ## Getting the values from each entry, low = [0], high = [1]
            self.parameter_values[param_index] = float(self.parameter_values[param_index].get())
        del self.key
        self.new_window.destroy()
        del self.new_window
        self.button_pressed.set(True)
        print("self.parameter_values", self.parameter_values)

    def sample_space(self):
        print("Sampling space ...")
        self.status_set("Space sampling - checking inputs")
        ## Getting values from entry boxes
        self.size_q = int(self.size_q_entry.get())

        ## Checking if all entries filled
        if self.size_q == "":
            messagebox.showwarning("Refine space", "Choose size_q, number of samples before sampling.")
            return

        ## If no space loaded check properties
        if self.space == "":
            if not self.validate_props("Sample Space"):
                return

        if not self.validate_space("Sample Space"):
            return

        self.status_set("Space sampling is running ...")
        print("self.space.params", self.space.params)
        print("self.props", self.props)
        print("self.size_q", self.size_q)
        print("self.save_sample.get()", self.save_sample.get())
        self.space.sample(self.props, self.size_q, silent=False, save=self.save_sample.get())

        ## Show the space as niceprint()
        print("space", self.space)
        print()
        print("space nice print \n", self.space.nice_print())
        self.space_text.delete('1.0', END)
        self.space_text.insert('end', self.space.nice_print())
        self.status_set("Space sampling done.")

        spam, egg = self.space.show(sat_samples=True, unsat_samples=True, red=False, green=False,
                                    save=self.save_sample.get(), where=[self.page6_figure, self.page6_a])
        ## if no plot provided
        if spam is None:
            messagebox.showinfo("Sample Space", egg)
        else:
            self.page6_figure = spam
            self.page6_a = egg
            self.page6_figure.canvas.draw()
            self.page6_figure.canvas.flush_events()

    def refine_space(self):
        print("Refining space ...")
        self.status_set("Space refinement - checking inputs")

        ## Getting values from entry boxes
        self.max_depth = int(self.max_dept_entry.get())
        self.coverage = float(self.coverage_entry.get())
        self.epsilon = float(self.epsilon_entry.get())
        self.alg = int(self.algorithm_entry.get())

        ## Checking if all entries filled
        if self.max_depth == "":
            messagebox.showwarning("Refine space", "Choose max recursion depth before refinement.")
            return

        if self.coverage == "":
            messagebox.showwarning("Refine space", "Choose coverage, nonwhite fraction to reach before refinement.")
            return

        if self.epsilon == "":
            messagebox.showwarning("Refine space", "Choose epsilon, min rectangle size before refinement.")
            return

        if self.alg == "":
            messagebox.showwarning("Refine space", "Pick algorithm for the refinement before running.")
            return

        ## If no space loaded check properties
        if self.space == "":
            if not self.validate_props("Refine Space"):
                return

        if not self.validate_space("Refine Space"):
            return

        self.status_set("Space refinement is running ...")
        spam = check_deeper(self.space, self.props, self.max_depth, self.epsilon, self.coverage, silent=False,
                            version=self.alg, size_q=False, debug=False, save=self.save_refinement.get(),
                            title="", where=[self.page6_figure, self.page6_a])
        ## If the visualisation of the space did not succeed
        if isinstance(spam, tuple):
            self.space = spam[0]
            messagebox.showinfo("Space refinement", spam[1])
        else:
            self.space = spam
        self.page6_figure.canvas.draw()
        self.page6_figure.canvas.flush_events()
        ## Show the space as niceprint()
        print("space", self.space)
        print()
        print("space nice print \n", self.space.nice_print())
        self.space_text.delete('1.0', END)
        self.space_text.insert('end', self.space.nice_print())
        self.status_set("Space refinement done.")

    ## SETTINGS
    def edit_config(self):
        print("Editing config ...")
        if "wind" in platform.system().lower():
            ## TBD TEST THIS ON WINDOWS
            os.startfile(f'{os.path.join(workspace, "../config.ini")}')
        else:
            os.system(f'gedit {os.path.join(workspace, "../config.ini")}')
        self.load_config()  ## Reloading the config file after change
        self.status_set("Config file saved.")

    ## HELP
    def show_help(self):
        print("Showing help ...")
        webbrowser.open_new("https://github.com/xhajnal/mpm#mpm")

    def checkupdates(self):
        print("Checking for updates ...")
        self.status_set("Checking for updates ...")
        webbrowser.open_new("https://github.com/xhajnal/mpm/releases")

    def printabout(self):
        print("Printing about ...")
        top2 = Toplevel(self)
        top2.title("About")
        top2.resizable(0, 0)
        explanation = f" Mpm version: {self.version} \n More info here: https://github.com/xhajnal/mpm \n Powered by University of Constance and Masaryk University"
        Label(top2, justify=LEFT, text=explanation).pack(padx=13, pady=20)
        top2.transient(self)
        top2.grab_set()
        self.wait_window(top2)

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


gui = Gui()
## System dependent fullscreen setting
if "wind" in platform.system().lower():
    gui.state('zoomed')
else:
    gui.attributes('-zoomed', True)
gui.mainloop()
