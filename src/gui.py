import platform
from copy import deepcopy
from tkinter import *
from tkinter import scrolledtext, messagebox
import webbrowser
import pickle
import os
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.messagebox import askyesno
from tkinter.ttk import Progressbar

import matplotlib.pyplot as pyplt
import matplotlib

from common.convert import ineq_to_constraints
from common.z3 import is_this_z3_function, translate_z3_function, is_this_exponential_function

matplotlib.use("TKAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import configparser

config = configparser.ConfigParser()
workspace = os.path.dirname(__file__)
sys.path.append(workspace)

from mc_informed import general_create_data_informed_properties
from load import load_functions, find_param, load_data, find_param_old
from common.math import create_intervals
import space
from synthetise import check_deeper
from mc import call_prism_files, call_storm_files
from sample_n_visualise import sample_list_funs, eval_and_show, get_param_values, heatmap
from optimize import optimize


cwd = os.getcwd()


## class copied from https://stackoverflow.com/questions/20399243/display-message-when-hovering-over-something-with-mouse-cursor-in-python/20399283
class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        """ Display text in tooltip window """

        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=self.text, justify=LEFT,
                      background="#ffffe0", relief=SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


## copied from https://stackoverflow.com/questions/20399243/display-message-when-hovering-over-something-with-mouse-cursor-in-python/20399283
def createToolTip(widget, text):
    tool_tip = ToolTip(widget)

    def enter(event):
        tool_tip.showtip(text)

    def leave(event):
        tool_tip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


## Callback function (but can be used also inside the GUI class)
def show_message(typee, where, message):
    if typee == 1 or str(typee).lower() == "error":
        messagebox.showerror(where, message)
    if typee == 2 or str(typee).lower() == "warning":
        messagebox.showwarning(where, message)
    if typee == 3 or str(typee).lower() == "info":
        messagebox.showinfo(where, message)


class Gui(Tk):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        ## Trying to configure pyplot
        # pyplt.autoscale()
        pyplt.autoscale(tight=True)

        ## Variables
        ## Directories
        self.model_dir = ""  ## Path to model
        self.property_dir = ""  ## Path to temporal properties
        self.data_dir = ""  ## Path to data
        self.prism_results = ""  ## Path to prism results
        self.storm_results = ""  ## Path to Storm results
        self.refinement_results = ""  ## Path to refinement results
        self.figures_dir = ""  ## Path to saved figures
        self.optimisation_results_dir = ""  ## Path to saved optimisation results
        self.tmp_dir = ""  ## Path for tmp folder
        self.load_config()  ## Load the config file

        ## Files
        self.model_file = StringVar()  ## Model file
        self.property_file = StringVar()  ## Property file
        self.data_informed_property_file = StringVar()  ## Data informed property file
        self.data_file = StringVar()  ## Data file
        self.functions_file = StringVar()  ## Rational functions file
        self.constraints_file = StringVar()  ## constraints file
        self.space_file = StringVar()  ## Space file

        ## Checking the change
        self.model_changed = False
        self.property_changed = False
        self.functions_changed = False
        self.data_changed = False
        self.data_intervals_changed = False
        self.constraints_changed = False
        self.space_changed = False

        ## True Variables
        # self.model = ""
        # self.property = ""
        self.data = ""
        self.data_informed_property = ""  ## Property containing the interval boundaries from the data
        self.functions = ""  ## Parameter synthesis results (rational functions)
        self.z3_functions = ""  ## functions with z3 expressions inside
        self.data_intervals = ""  ## Computed intervals
        self.parameters = ""  ##  Parsed parameters
        self.parameter_domains = []  ## Parameters intervals
        self.constraints = ""  ## Computed or loaded constrains
        self.z3_constraints = ""  ## Constrains with z3 expressions inside
        self.space = ""  ## Instance of a space Class

        ## Results
        self.sampled_functions = []
        self.optimised_param_point = ""  ## List of parameter values with least distance
        self.optimised_function_value = ""  ## List of functions values with least distance
        self.optimised_distance = ""  ## The actual distance between functions and data

        ## Space visualisation settings
        self.show_samples = None
        self.show_refinement = None
        self.show_true_point = None

        ## Settings
        self.version = "1.8.2"  ## Version of the gui
        self.silent = BooleanVar()  ## Sets the command line output to minimum
        self.debug = BooleanVar()  ## Sets the command line output to maximum

        ## Settings/data
        # self.alpha = ""  ## Confidence
        # self.n_samples = ""  ## Number of samples
        self.program = StringVar()  ## "prism"/"storm"
        self.max_depth = ""  ## Max recursion depth
        self.coverage = ""  ## Coverage threshold
        self.epsilon = ""  ## Rectangle size threshold
        self.alg = ""  ## Refinement alg. number

        self.solver = ""  ## SMT solver - z3 or dreal
        self.delta = 0.01  ## dreal setting

        self.factor = BooleanVar()  ## Flag for factorising rational functions
        self.size_q = ""  ## Number of samples
        self.save = ""  ## True if saving on

        ## OTHER SETTINGS
        self.button_pressed = BooleanVar()  ## Inner variable to close created window
        self.python_recursion_depth = 1000  ## Inner python setting

        ## GUI INIT
        self.title('Mpm')
        self.minsize(1000, 300)

        ## Temporal gui features
        self.progress_bar = None
        self.new_window = None

        ## DESIGN

        ## STATUS BAR
        self.status = Label(self, text="", bd=1, relief=SUNKEN, anchor=W)
        self.status.pack(side=BOTTOM, fill=X)

        ## DESIGN - STATUS
        frame = Frame(self)  ## Upper frame
        frame.pack(fill=X)

        Label(frame, text=f"Model file:", anchor=W, justify=LEFT).grid(row=0, column=0, sticky=W, padx=4)
        self.model_label = Label(frame, textvariable=self.model_file, anchor=W, justify=LEFT)
        self.model_label.grid(row=0, column=1, sticky=W, padx=4)

        Label(frame, text=f"Property file:", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, padx=4)
        self.property_label = Label(frame, textvariable=self.property_file, anchor=W, justify=LEFT)
        self.property_label.grid(row=1, column=1, sticky=W, padx=4)

        Label(frame, text=f"Functions file:", anchor=W, justify=LEFT).grid(row=2, column=0, sticky=W, padx=4)
        self.functions_label = Label(frame, textvariable=self.functions_file, anchor=W, justify=LEFT)
        self.functions_label.grid(row=2, column=1, sticky=W, padx=4)

        Label(frame, text=f"Data file:", anchor=W, justify=LEFT).grid(row=3, column=0, sticky=W, padx=4)
        self.data_label = Label(frame, textvariable=self.data_file, anchor=W, justify=LEFT)
        self.data_label.grid(row=3, column=1, sticky=W, padx=4)

        Label(frame, text=f"Constraints file:", anchor=W, justify=LEFT).grid(row=4, column=0, sticky=W, padx=4)
        self.constraints_label = Label(frame, textvariable=self.constraints_file, anchor=W, justify=LEFT)
        self.constraints_label.grid(row=4, column=1, sticky=W, padx=4)

        Label(frame, text=f"Space file:", anchor=W, justify=LEFT).grid(row=5, column=0, sticky=W, padx=4)
        self.space_label = Label(frame, textvariable=self.space_file, anchor=W, justify=LEFT)
        self.space_label.grid(row=5, column=1, sticky=W, padx=4)

        show_print_checkbutton = Checkbutton(frame, text="Hide print in command line", variable=self.silent)
        show_print_checkbutton.grid(row=5, column=9, sticky=E, padx=4)
        debug_checkbutton = Checkbutton(frame, text="Extensive command line print", variable=self.debug)
        debug_checkbutton.grid(row=5, column=10, sticky=E, padx=4)
        # print("self.silent", self.silent.get())

        ## DESIGN - TABS
        # Defines and places the notebook widget
        nb = ttk.Notebook(self)  ## Tab part of the GUI
        nb.pack(fill="both", expand=1)


        ## TAB EDIT
        page1 = ttk.Frame(nb, width=600, height=200, name="model_properties")  # Adds tab 1 of the notebook
        nb.add(page1, text='Model & Properties', state="normal", sticky="nsew")

        # page1.rowconfigure(5, weight=1)
        # page1.columnconfigure(6, weight=1)

        frame_left = Frame(page1, width=400, height=100)  ## Model part
        # for i in range(4):
        #    frame_left.rowconfigure(i, weight=1)
        frame_left.rowconfigure(2, weight=1)
        frame_left.columnconfigure(6, weight=1)
        # for i in range(7):
        #     frame_left.columnconfigure(i, weight=1)
        frame_left.pack(side=LEFT, fill=X)

        Button(frame_left, text='Open model', command=self.load_model).grid(row=0, column=0, sticky=W, padx=4,
                                                                            pady=4)  # pack(anchor=W)
        Button(frame_left, text='Save model', command=self.save_model).grid(row=0, column=1, sticky=W, padx=4,
                                                                            pady=4)  # pack(anchor=W)
        Label(frame_left, text=f"Loaded model file:", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, padx=4,
                                                                                   pady=4)  # pack(anchor=W)

        self.model_text = scrolledtext.ScrolledText(frame_left, height=100)
        # self.model_text.config(state="disabled")
        self.model_text.grid(row=2, column=0, columnspan=16, rowspan=2, sticky=W, padx=4,
                             pady=4)  # pack(anchor=W, fill=X, expand=True)

        frame_right = Frame(page1)  ## Property part
        for i in range(7):
            frame_left.columnconfigure(i, weight=1)
        frame_right.rowconfigure(3, weight=1)
        # frame_right.columnconfigure(6, weight=1)
        frame_right.pack(side=RIGHT, fill=X)

        Button(frame_right, text='Open property', command=self.load_property).grid(row=0, column=0, sticky=W, pady=4,
                                                                                   padx=4)  # pack(anchor=W)
        Button(frame_right, text='Save property', command=self.save_property).grid(row=0, column=1, sticky=W, pady=4)  # pack(anchor=W)
        Label(frame_right, text=f"Loaded property file:", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W,
                                                                                       pady=4)  # pack(anchor=W)

        self.property_text = scrolledtext.ScrolledText(frame_right, height=100)
        # self.property_text.config(state="disabled")
        self.property_text.grid(row=2, column=0, columnspan=16, rowspan=2, sticky=W + E + N + S, pady=4)  # pack(anchor=W, fill=X)

        # print(nb.select(0), type(nb.select(0)))
        # print(page1, type(page1))


        ## TAB SYNTHESISE
        page2 = ttk.Frame(nb, width=400, height=100, name="synthesise")  # Adds tab 2 of the notebook
        nb.add(page2, text='Synthesise functions')

        page2.rowconfigure(5, weight=1)
        page2.columnconfigure(6, weight=1)

        ## SELECTING THE PROGRAM
        self.program.set("prism")
        Label(page2, text="Select the program: ", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, padx=4,
                                                                               pady=4)
        Radiobutton(page2, text="Prism", variable=self.program, value="prism").grid(row=1, column=1, sticky=W, pady=4)
        radio = Radiobutton(page2, text="Storm", variable=self.program, value="storm")
        radio.grid(row=1, column=2, sticky=W, pady=4)
        createToolTip(radio,
                      text='This option results in a command that would produce desired output. (If you installed Storm, open command line and insert the command. Then load output file.)')

        Label(page2, text=f"Show function(s):", anchor=W, justify=LEFT).grid(row=2, column=0, sticky=W, padx=4, pady=4)
        Radiobutton(page2, text="Original", variable=self.factor, value=False).grid(row=2, column=1, sticky=W, pady=4)
        Radiobutton(page2, text="Factorised", variable=self.factor, value=True).grid(row=2, column=2, sticky=W, pady=4)

        Button(page2, text='Run parameter synthesis', command=self.synth_params).grid(row=3, column=0, sticky=W, padx=4, pady=4)
        Button(page2, text='Open Prism/Storm output file', command=self.load_mc_output_file).grid(row=3, column=1, sticky=W, pady=4)

        Label(page2, text=f"Loaded Prism/Storm output file:", anchor=W, justify=LEFT).grid(row=4, column=0, sticky=W, padx=4, pady=4)

        self.functions_text = scrolledtext.ScrolledText(page2, height=100, state=DISABLED)
        self.functions_text.grid(row=5, column=0, columnspan=16, rowspan=2, sticky=W, padx=4, pady=4)

        Label(page2, text=f"Rational functions section.", anchor=W, justify=LEFT).grid(row=1, column=17, sticky=W, padx=4, pady=4)
        Button(page2, text='Open functions', command=self.load_parsed_functions).grid(row=3, column=17, sticky=W, padx=4, pady=4)
        Button(page2, text='Save functions', command=self.save_functions).grid(row=3, column=18, sticky=W, pady=4)

        Label(page2, text=f"Parsed function(s):", anchor=W, justify=LEFT).grid(row=4, column=17, sticky=W, padx=4, pady=4)
        self.functions_parsed_text = scrolledtext.ScrolledText(page2, height=100, state=DISABLED)
        self.functions_parsed_text.grid(row=5, column=17, columnspan=16, rowspan=2, sticky=W, pady=4)


        ## TAB SAMPLE AND VISUALISE
        self.page3 = ttk.Frame(nb, width=400, height=200, name="sampling")
        nb.add(self.page3, text='Sample functions')



        Label(self.page3, text="Set number of samples per variable (grid size):", anchor=W, justify=LEFT).grid(row=1, column=0, padx=4, pady=4)
        self.fun_size_q_entry = Entry(self.page3)
        self.fun_size_q_entry.grid(row=1, column=1)

        Button(self.page3, text='Sample functions', command=self.sample_fun).grid(row=2, column=0, sticky=W, padx=4, pady=4)

        Label(self.page3, text=f"Values of sampled points:", anchor=W, justify=LEFT).grid(row=3, column=0, sticky=W, padx=4, pady=4)

        self.sampled_functions_text = scrolledtext.ScrolledText(self.page3, height=100, state=DISABLED)
        self.sampled_functions_text.grid(row=4, column=0, columnspan=8, rowspan=2, sticky=W, padx=4, pady=4)

        Label(self.page3, text=f"Rational functions visualisation", anchor=W, justify=CENTER).grid(row=1, column=14, columnspan=3, pady=4)
        Button(self.page3, text='Plot functions in a given point', command=self.show_funs_in_single_point).grid(row=2, column=14, padx=4, pady=4)
        Button(self.page3, text='Plot all sampled points', command=self.show_funs_in_all_points).grid(row=2, column=15, padx=4, pady=4)
        Button(self.page3, text='Heatmap', command=self.show_heatmap).grid(row=2, column=16, padx=4, pady=4)
        self.Next_sample_button = Button(self.page3, text="Next plot", state="disabled",
                                         command=lambda: self.button_pressed.set(True))
        self.Next_sample_button.grid(row=3, column=15, padx=4, pady=4)

        self.page3_figure = None
        # self.page3_figure = pyplt.figure()
        # self.page3_a = self.page3_figure.add_subplot(111)
        # print("type a", type(self.a))

        self.page3.rowconfigure(5, weight=1)
        self.page3.columnconfigure(18, weight=1)

        self.page3_figure_in_use = StringVar()
        self.page3_figure_in_use.set("")


        ## TAB DATA
        page4 = ttk.Frame(nb, width=400, height=200, name="data")
        nb.add(page4, text='Data & Intervals')
        # page4.columnconfigure(0, weight=1)
        # page4.rowconfigure(2, weight=1)
        # page4.rowconfigure(7, weight=1)

        Button(page4, text='Open data file', command=self.load_data).grid(row=0, column=0, sticky=W, padx=4, pady=4)
        Button(page4, text='Save data', command=self.save_data).grid(row=0, column=1, sticky=W, padx=4)

        label10 = Label(page4, text=f"Data:", anchor=W, justify=LEFT)
        label10.grid(row=1, column=0, sticky=W, padx=4, pady=4)
        createToolTip(label10, text='For each rational function exactly one data point should be assigned.')

        self.data_text = Text(page4, height=11)  # , height=10, width=30
        ## self.data_text.bind("<FocusOut>", self.parse_data)
        # self.data_text = Text(page4, height=12, state=DISABLED)  # , height=10, width=30
        # self.data_text.config(state="disabled")
        self.data_text.grid(row=2, column=0, columnspan=16, sticky=W, padx=4, pady=4)

        ## SET THE INTERVAL COMPUTATION SETTINGS
        Button(page4, text='Optimize parameters', command=self.optimize).grid(row=3, column=0, sticky=W, padx=4, pady=4)

        label42 = Label(page4, text="Set alpha, the confidence:", anchor=W, justify=LEFT)
        label42.grid(row=4)
        createToolTip(label42, text='confidence')
        label43 = Label(page4, text="Set n_samples, number of samples: ", anchor=W, justify=LEFT)
        label43.grid(row=5)
        createToolTip(label43, text='number of samples')

        self.alpha_entry = Entry(page4)
        self.n_samples_entry = Entry(page4)

        self.alpha_entry.grid(row=4, column=1)
        self.n_samples_entry.grid(row=5, column=1)

        self.alpha_entry.insert(END, '0.90')
        self.n_samples_entry.insert(END, '60')

        Button(page4, text='Create intervals', command=self.create_data_intervals).grid(row=6, column=0, sticky=W, padx=4, pady=4)

        Label(page4, text=f"Intervals:", anchor=W, justify=LEFT).grid(row=7, column=0, sticky=W, padx=4, pady=4)

        self.data_intervals_text = Text(page4, height=11, state=DISABLED)  # height=10, width=30
        # self.data_intervals_text.config(state="disabled")
        self.data_intervals_text.grid(row=8, column=0, rowspan=2, columnspan=16, sticky=W, padx=4, pady=4)

        ttk.Separator(page4, orient=VERTICAL).grid(row=0, column=17, rowspan=10, sticky='ns', padx=50, pady=10)
        Label(page4, text=f"Data informed property section.", anchor=W, justify=LEFT).grid(row=0, column=18, sticky=W, padx=5, pady=4)
        Label(page4, text=f"Loaded property file:", anchor=W, justify=LEFT).grid(row=1, column=18, sticky=W, padx=5, pady=4)

        self.property_text2 = scrolledtext.ScrolledText(page4, height=4, state=DISABLED)
        # self.property_text2.config(state="disabled")
        self.property_text2.grid(row=2, column=18, columnspan=16, rowspan=2, sticky=W + E + N + S, padx=5, pady=4)
        Button(page4, text='Generate data informed properties', command=self.generate_data_informed_properties).grid(row=4, column=18, sticky=W, padx=5, pady=4)

        self.data_informed_property_text = scrolledtext.ScrolledText(page4, height=4, state=DISABLED)
        self.data_informed_property_text.grid(row=5, column=18, columnspan=16, rowspan=4, sticky=W + E + N + S, padx=5, pady=10)

        Button(page4, text='Save data informed properties', command=self.save_data_informed_properties).grid(row=9, column=18, sticky=W, padx=5, pady=4)


        ## TAB constraints
        page5 = ttk.Frame(nb, width=400, height=200, name="constraints")
        nb.add(page5, text='Constraints')

        page5.rowconfigure(2, weight=1)
        page5.columnconfigure(16, weight=1)

        Button(page5, text='Recalculate constraints', command=self.recalculate_constraints).grid(row=0, column=0, sticky=W, padx=4, pady=4)

        self.constraints_text = scrolledtext.ScrolledText(page5, height=100, state=DISABLED)
        self.constraints_text.grid(row=1, column=0, columnspan=16, rowspan=2, sticky=W, padx=4, pady=4)

        Label(page5, text=f"Import/Export:", anchor=W, justify=LEFT).grid(row=3, column=0, sticky=W, padx=4, pady=4)
        Button(page5, text='Open constraints', command=self.load_constraints).grid(row=3, column=1, sticky=W, pady=4)
        Button(page5, text='Append constraints', command=self.append_constraints).grid(row=3, column=2, sticky=W, pady=4)
        Button(page5, text='Save constraints', command=self.save_constraints).grid(row=3, column=3, sticky=W, pady=4)


        ## TAB SAMPLE AND REFINEMENT
        page6 = ttk.Frame(nb, width=1000, height=400, name="refine")
        nb.add(page6, text='Sample & Refine space')

        frame_left = Frame(page6, width=500, height=200)
        frame_left.pack(side=LEFT, expand=False)

        # Button(frame_left, text='Create space', command=self.validate_space).grid(row=0, column=0, sticky=W, padx=4, pady=4)

        ttk.Separator(frame_left, orient=HORIZONTAL).grid(row=1, column=0, columnspan=15, sticky='nwe', padx=10, pady=8)

        label61 = Label(frame_left, text="Set size_q: ", anchor=W, justify=LEFT, padx=10)
        label61.grid(row=1, pady=16)
        createToolTip(label61, text='number of samples per dimension')

        self.size_q_entry = Entry(frame_left)
        self.size_q_entry.grid(row=1, column=1)
        self.size_q_entry.insert(END, '5')

        Button(frame_left, text='Grid sampling', command=self.sample_space).grid(row=7, column=0, columnspan=2, sticky=W, padx=10, pady=4)

        ttk.Separator(frame_left, orient=VERTICAL).grid(row=1, column=2, rowspan=8, sticky='ns', padx=0, pady=25)

        label71 = Label(frame_left, text="Set N_obs: ", anchor=W, justify=LEFT)
        label71.grid(row=1, column=3)
        createToolTip(label71, text='number of samples to be used for sampling - subset of all samples')

        self.N_obs_entry = Entry(frame_left)
        self.N_obs_entry.grid(row=1, column=4)
        self.N_obs_entry.insert(END, '500')

        label71 = Label(frame_left, text="Set MH_samples: ", anchor=W, justify=LEFT)
        label71.grid(row=2, column=3)
        createToolTip(label71, text='number of iterations')

        self.MH_samples_entry = Entry(frame_left)
        self.MH_samples_entry.grid(row=2, column=4)
        self.MH_samples_entry.insert(END, '500')

        label71 = Label(frame_left, text="Set eps: ", anchor=W, justify=LEFT)
        label71.grid(row=3, column=3)
        createToolTip(label71, text='very small value used as probability of non-feasible values in prior')

        self.eps_entry = Entry(frame_left)
        self.eps_entry.grid(row=3, column=4)
        self.eps_entry.insert(END, '0')

        # N: int, N_obs: int, MH_samples: int, eps

        Button(frame_left, text='Metropolis-Hastings', command=self.hastings).grid(row=7, column=3, columnspan=2, sticky=W, pady=4)

        ttk.Separator(frame_left, orient=VERTICAL).grid(row=1, column=5, rowspan=8, sticky='ns', padx=10, pady=25)

        label62 = Label(frame_left, text="Set max_dept: ", anchor=W, justify=LEFT)
        label62.grid(row=1, column=8, padx=10)
        createToolTip(label62, text='Maximal number of splits')
        label63 = Label(frame_left, text="Set coverage: ", anchor=W, justify=LEFT)
        label63.grid(row=2, column=8, padx=10)
        createToolTip(label63, text='Proportion of the nonwhite area to be reached')
        label64 = Label(frame_left, text="Set epsilon: ", anchor=W, justify=LEFT)
        label64.grid(row=3, column=8, padx=10)
        createToolTip(label64,
                      text='Minimal size of the rectangle to be checked (if 0 all rectangles are being checked)')
        label65 = Label(frame_left, text="Set algorithm: ", anchor=W, justify=LEFT)
        label65.grid(row=4, column=8, padx=10)
        createToolTip(label65, text='Choose from algorithms:\n 1-4 - using SMT solvers \n 1 - DFS search \n 2 - BFS search \n 3 - BFS search with example propagation \n 4 - BFS with example and counterexample propagation \n 5 - interval algorithmic')

        label66 = Label(frame_left, text="Set solver: ", anchor=W, justify=LEFT)
        label66.grid(row=5, column=8, padx=10)
        createToolTip(label66, text='When using SMT solver (alg 1-4), two options are possible, z3 or dreal (with delta complete decision procedures)')

        label67 = Label(frame_left, text="Set delta: ", anchor=W, justify=LEFT)
        label67.grid(row=6, column=8, padx=10)
        createToolTip(label67, text='When using dreal solver, delta is used to set solver error boundaries for satisfiability.')

        self.max_dept_entry = Entry(frame_left)
        self.coverage_entry = Entry(frame_left)
        self.epsilon_entry = Entry(frame_left)
        self.alg = ttk.Combobox(frame_left, values=('1', '2', '3', '4', '5'))
        self.solver = ttk.Combobox(frame_left, values=('z3', 'dreal'))
        self.delta_entry = Entry(frame_left)

        self.max_dept_entry.grid(row=1, column=9)
        self.coverage_entry.grid(row=2, column=9)
        self.epsilon_entry.grid(row=3, column=9)
        self.alg.grid(row=4, column=9)
        self.solver.grid(row=5, column=9)
        self.delta_entry.grid(row=6, column=9)

        self.max_dept_entry.insert(END, '5')
        self.coverage_entry.insert(END, '0.95')
        self.epsilon_entry.insert(END, '0')
        self.alg.current(3)
        self.solver.current(0)
        self.delta_entry.insert(END, '0.01')

        Button(frame_left, text='Refine space', command=self.refine_space).grid(row=7, column=8, columnspan=2, sticky=W, pady=4, padx=10)

        ttk.Separator(frame_left, orient=HORIZONTAL).grid(row=8, column=0, columnspan=15, sticky='nwe', padx=10, pady=4)

        self.space_text = scrolledtext.ScrolledText(frame_left, height=200, width=100, state=DISABLED)
        self.space_text.grid(row=12, column=0, columnspan=15, rowspan=2, sticky=W, padx=10)  # pack(anchor=W, fill=X)

        frame_left.rowconfigure(13, weight=1)
        frame_left.columnconfigure(15, weight=1)

        Button(frame_left, text='Open space', command=self.load_space).grid(row=14, column=2, sticky=S, padx=4, pady=4)
        Button(frame_left, text='Save space', command=self.save_space).grid(row=14, column=3, sticky=S, padx=4, pady=4)
        Button(frame_left, text='Delete space', command=self.refresh_space).grid(row=14, column=4, sticky=S, padx=4, pady=4)

        frame_right = Frame(page6, width=500, height=200)
        frame_right.pack(side=TOP, fill=BOTH, expand=True)

        Button(frame_right, text='Edit True point', command=self.edit_true_point).pack(side=TOP)

        Label(frame_right, text=f"Space Visualisation", anchor=W, justify=CENTER).pack(side=TOP)

        ##################################################### UPPER PLOT ###############################################
        self.page6_plotframe = Frame(frame_right)
        self.page6_plotframe.pack(side=TOP, fill=Y, expand=True)
        self.page6_figure = pyplt.figure(figsize=(8, 2))
        self.page6_figure.tight_layout()  ## By huypn

        self.page6_canvas = FigureCanvasTkAgg(self.page6_figure, master=self.page6_plotframe)  # A tk.DrawingArea.
        self.page6_canvas.draw()
        self.page6_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.page6_toolbar = NavigationToolbar2Tk(self.page6_canvas, self.page6_plotframe)
        self.page6_toolbar.update()
        self.page6_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.page6_a = self.page6_figure.add_subplot(111)

        ##################################################### LOWER PLOT ###############################################
        self.page6_plotframe2 = Frame(frame_right)
        self.page6_plotframe2.pack(side=TOP, fill=Y, expand=True)
        self.page6_figure2 = pyplt.figure(figsize=(8, 2))
        self.page6_figure2.tight_layout()  ## By huypn

        self.page6_canvas2 = FigureCanvasTkAgg(self.page6_figure2, master=self.page6_plotframe2)  # A tk.DrawingArea.
        self.page6_canvas2.draw()
        self.page6_canvas2.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.page6_toolbar2 = NavigationToolbar2Tk(self.page6_canvas2, self.page6_plotframe2)
        self.page6_toolbar2.update()
        self.page6_canvas2.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.page6_b = self.page6_figure2.add_subplot(111)
        #################################################### /PLOTS ####################################################

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
        # # save_menu.add_command(label="Save rational functions", command=self.save_functions())
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
        help_menu.add_command(label="Check for updates", command=self.check_updates)
        help_menu.add_command(label="About", command=self.print_about)

        self.autoload()

    def load_config(self):
        """ Loads variables from the config file """
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
        
        self.optimisation_results_dir = config.get("paths", "optimisation")
        if not os.path.exists(self.optimisation_results_dir):
            os.makedirs(self.optimisation_results_dir)

        self.tmp_dir = config.get("paths", "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
            
        os.chdir(cwd)

    ## LOGIC
    ## FILE - LOAD, PARSE, SHOW, AND SAVE
    def load_model(self, file=False):
        """ Loads model from a text file.

        Args:
            file (path/string): direct path to load the function file
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading model ...")
            ## If some model previously loaded
            if len(self.model_text.get('1.0', END)) > 1:
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
            ## If some model previously loaded
            if len(self.model_text.get('1.0', END)) > 1:
                self.model_changed = True
            self.model_file.set(spam)
            # self.model_text.configure(state='normal')
            self.model_text.delete('1.0', END)
            self.model_text.insert('end', open(self.model_file.get(), 'r').read())
            # self.model_text.configure(state='disabled')
            self.status_set("Model loaded.")
            # print("self.model", self.model.get())

            ## Autosave
            if not file:
                self.save_model(os.path.join(self.tmp_dir, "model"))

    def load_property(self, file=False):
        """ Loads temporal properties from a text file.
        Args:
            file (path/string): direct path to load the function file
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading properties ...")
            ## If some property previously loaded
            if len(self.property_text.get('1.0', END)) > 1:
                if not askyesno("Loading properties",
                                "Previously obtained properties will be lost. Do you want to proceed?"):
                    return
            self.status_set("Please select the property to be loaded.")

            spam = filedialog.askopenfilename(initialdir=self.property_dir, title="Property loading - Select file",
                                              filetypes=(("property files", "*.pctl"), ("all files", "*.*")))
        ## If no file selected
        if spam == "":
            self.status_set("No file selected.")
            return
        else:
            ## If some property previously loaded
            if len(self.property_text.get('1.0', END)) > 1:
                self.property_changed = True
            self.property_file.set(spam)
            self.property_text.configure(state='normal')
            self.property_text.delete('1.0', END)
            self.property_text.insert('end', open(self.property_file.get(), 'r').read())
            # self.property_text.configure(state='disabled')

            self.property_text2.configure(state='normal')
            self.property_text2.delete('1.0', END)
            self.property_text2.insert('end', open(self.property_file.get(), 'r').read())
            # self.property_text2.configure(state='disabled')
            self.status_set("Property loaded.")
            # print("self.property", self.property.get())

            ## Autosave
            if not file:
                self.save_property(os.path.join(self.tmp_dir, "properties"))

    def load_mc_output_file(self, file=False):
        """ Loads parameter synthesis output text file

        Args:
            file (path/string): direct path to load the function file
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
            if "prism" in str(file):
                self.program.set("prism")
            elif "storm" in str(file):
                self.program.set("storm")
            else:
                print(f"Error while loading file {file}")
                return
        else:
            print("Loading functions ...")

            if self.functions_changed:
                if not askyesno("Loading functions", "Previously obtained functions will be lost. Do you want to proceed?"):
                    return

            self.status_set("Loading functions - checking inputs")

            if not self.silent.get():
                print("Used program: ", self.program.get())
            if self.program.get() == "prism":
                initial_dir = self.prism_results
            elif self.program.get() == "storm":
                initial_dir = self.storm_results
            else:
                messagebox.showwarning("Load functions", "Select a program for which you want to load data.")
                return

            self.status_set("Please select the prism/storm symbolic results to be loaded.")
            spam = filedialog.askopenfilename(initialdir=initial_dir, title="Rational functions loading - Select file",
                                              filetypes=(("text files", "*.txt"), ("all files", "*.*")))

        ## If no file / not a file selected
        if spam == "" or spam == ():
            self.status_set("No file selected.")
            return

        self.functions_file.set(spam)
        # print("self.functions_file.get() ", self.functions_file.get())
        if not self.functions_file.get() is "":
            self.functions_changed = True
            # self.model_changed = False
            # self.property_changed = False
        # print("self.functions_changed", self.functions_changed)

        # print("self.factor", self.factor.get())
        self.functions, rewards = load_functions(self.functions_file.get(), tool=self.program.get(),
                                                 factorize=self.factor.get(), rewards_only=False, f_only=False)
        ## Merge functions and rewards
        # print("self.functions", self.functions)
        # print("rewards", rewards)
        for expression in rewards:
            self.functions.append(expression)

        if not self.silent.get():
            print("Unparsed functions: ", self.functions)

        self.unfold_functions()

        if isinstance(self.functions, dict):
            self.status_set(f"{len(self.functions.keys())} rational functions loaded")
        elif isinstance(self.functions, list):
            self.status_set(f"{len(self.functions)} rational functions loaded")
        else:
            raise Exception("Loading parameter synthesis results",
                            f"Expected type of the functions is dict or list, got {type(self.functions)}")

        if not self.silent.get():
            print("Parsed list of functions: ", self.functions)

        self.z3_functions = ""
        for function in self.functions:
            if is_this_z3_function(function):
                self.store_z3_functions()
                messagebox.showinfo("Loading functions", "Some of the functions contains z3 expressions, these are being stored and used only for z3 refinement, shown functions are translated into python expressions.")
                break

        ## Show loaded functions
        self.functions_text.configure(state='normal')
        self.functions_text.delete('1.0', END)
        self.functions_text.insert('1.0', open(self.functions_file.get(), 'r').read())
        # self.functions_text.configure(state='disabled')
        ## Resetting parsed intervals
        self.parameters = []
        self.parameter_domains = []

        ## Check whether loaded
        if not self.functions:
            messagebox.showwarning("Loading functions", "No functions loaded. Please check input file.")
        else:
            pass
            ## Autosave
            ## TODO
            # if not file:
            #   self.save_functions(os.path.join(self.tmp_dir, f"functions_{self.program.get()}"))

    def store_z3_functions(self):
        """ Stores a copy of functions as a self.z3_functions """
        self.z3_functions = deepcopy(self.functions)
        for index, function in enumerate(self.functions):
            self.functions[index] = translate_z3_function(function)

    def store_z3_constraints(self):
        """ Stores a copy of constraints as a self.z3_constraints """
        self.z3_constraints = deepcopy(self.constraints)
        for index, constraint in enumerate(self.constraints):
            self.constraints[index] = translate_z3_function(constraint)

    def unfold_functions(self):
        """" Unfolds the function dictionary into a single list """

        if isinstance(self.functions, dict):
            ## TODO Maybe rewrite this as key and pass the argument to unfold_functions2
            ## NO because dunno how to send it to the function as a argument
            if len(self.functions.keys()) == 1:
                for key in self.functions.keys():
                    self.functions = self.functions[key]
                    break
                self.unfold_functions()
                return

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
            spam.focus()
            spam.bind('<Return>', self.unfold_functions2)
        else:
            functions = ""
            for function in self.functions:
                functions = f"{functions},\n{function}"
            functions = functions[2:]

            self.functions_parsed_text.configure(state='normal')
            self.functions_parsed_text.delete('1.0', END)
            self.functions_parsed_text.insert('end', functions)
            # self.functions_parsed_text.configure(state='disabled')

    def unfold_functions2(self):
        """" Dummy method of unfold_functions """

        try:
            self.functions = self.functions[self.key.get()]
        except KeyError:
            self.functions = self.functions[eval(self.key.get())]

        if not self.silent.get():
            print("Parsed list of functions: ", self.functions)
        self.functions_window.destroy()
        self.unfold_functions()

    def load_parsed_functions(self, file=False):
        """ Loads parsed rational functions from a pickled file.
        Args:
            file (path/string): direct path to load the function file
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading parsed rational functions ...")
            if self.data_changed:
                if not askyesno("Loading parsed rational functions",
                                "Previously obtained functions will be lost. Do you want to proceed?"):
                    return

            self.status_set("Please select the parsed rational functions to be loaded.")

            if not self.silent.get():
                print("self.program.get()", self.program.get())
            if self.program.get() == "prism":
                initial_dir = self.prism_results
            elif self.program.get() == "storm":
                initial_dir = self.storm_results
            else:
                messagebox.showwarning("Load functions", "Select a program for which you want to load data.")
                return

            spam = filedialog.askopenfilename(initialdir=initial_dir,
                                              title="Rational functions saving - Select file",
                                              filetypes=(("pickle files", "*.p"), ("all files", "*.*")))

        ## If no file selected
        if spam == "":
            self.status_set("No file selected.")
            return
        else:
            self.functions = False
            self.functions_changed = True
            self.functions_file.set(spam)
            self.z3_functions = ""

            if ".p" in self.functions_file.get():
                self.functions = pickle.load(open(self.functions_file.get(), "rb"))

            print("loaded functions", self.functions)
            if not self.functions:
                messagebox.showwarning("Loading functions", "No functions loaded. Please check input file.")
                self.status_set("No rational functions loaded.")
                return

            for function in self.functions:
                if is_this_z3_function(function):
                    self.store_z3_functions()
                    messagebox.showinfo("Loading functions",
                                        "Some of the functions contains z3 expressions, these are being stored and used only for z3 refinement, shown functions are translated into python expressions.")
                    break

            functions = ""

            for function in self.functions:
                functions = f"{functions},\n{function}"
            functions = functions[2:]

            self.functions_parsed_text.configure(state='normal')
            self.functions_parsed_text.delete('1.0', END)
            self.functions_parsed_text.insert('end', functions)
            # self.functions_parsed_text.configure(state='disabled')

            ## Resetting parsed intervals
            self.parameters = []
            self.parameter_domains = []

            ## Autosave
            if not file:
                self.save_functions(os.path.join(self.tmp_dir, f"functions.p"))

            self.status_set("Parsed rational functions loaded.")

    def load_data(self, file=False):
        """ Loads data from a file. Either pickled list or comma separated values in one line
        Args:
            file (path/string): direct path to load the data file
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
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
            self.data = False
            self.data_changed = True
            self.data_file.set(spam)

            if ".p" in self.data_file.get():
                self.data = pickle.load(open(self.data_file.get(), "rb"))
                self.unfold_data()
            else:
                self.data = load_data(self.data_file.get(), silent=self.silent.get(), debug=not self.silent.get())
                if not self.data:
                    messagebox.showerror("Loading data", f"Error, No data loaded.")
                    self.status_set("Data not loaded properly.")
                    return
                self.unfold_data()
            if not self.silent.get():
                print("Loaded data: ", self.data)

            ## Autosave
            if not file:
                self.save_data(os.path.join(self.tmp_dir, "data"))

            self.status_set("Data loaded.")
            # self.parse_data_from_window()

    def unfold_data(self):
        """" Unfolds the data dictionary into a single list """
        if isinstance(self.data, dict):
            ## TODO Maybe rewrite this as key and pass the argument to unfold_data2
            self.key = StringVar()
            self.status_set(
                "Loaded data are in a form of dictionary, please select which item you would like to choose:")
            self.new_window = Toplevel(self)
            ## SCROLLABLE WINDOW
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

            label = Label(frame,
                          text="Loaded data are in a form of dictionary, please select which item you would like to choose:")
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
            spam.focus()
            spam.bind('<Return>', self.unfold_data2)
        else:
            # self.data_text.configure(state='normal')
            self.data_text.delete('1.0', END)
            spam = ""
            for item in self.data:
                spam = f"{spam},\n{item}"
            spam = spam[2:]
            self.data_text.insert('end', spam)
            # self.data_text.configure(state='disabled')

    def unfold_data2(self):
        """" Dummy method of unfold_data """
        try:
            self.data = self.data[self.key.get()]
        except KeyError:
            self.data = self.data[eval(self.key.get())]

        if not self.silent.get():
            print("self.data", self.data)
        self.new_window.destroy()
        self.unfold_data()

    def load_data_intervals(self, file=False):
        """ Loads intervals from a given file
        Args:
            file (path/string): direct path to load the intervals file
        """
        ## TODO
        pass

    def recalculate_constraints(self):
        """ Merges rational functions and intervals into constraints. Shows it afterwards. """
        ## If there is some constraints
        if len(self.constraints_text.get('1.0', END)) > 1:
            proceed = messagebox.askyesno("Recalculate constraints",
                                          "Previously obtained constraints will be lost. Do you want to proceed?")
        else:
            proceed = True
        if proceed:
            self.constraints = ""
            self.z3_constraints = ""
            self.validate_constraints(position="constraints")
            ## Autosave
            self.save_constraints(os.path.join(self.tmp_dir, "constraints"))
        self.status_set("constraints recalculated and shown.")

    def load_constraints(self, append=False, file=False):
        """ Loads constraints from a pickled file.
        Args:
            append (bool): if True, loaded constraints are appended to previous
            file (path/string): direct path to load the constraint file
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading constraints ...")

            if self.constraints_changed and not append:
                if not askyesno("Loading constraints", "Previously obtained constraints will be lost. Do you want to proceed?"):
                    return
            self.status_set("Please select the constraints to be loaded.")
            spam = filedialog.askopenfilename(initialdir=self.data_dir, title="constraints loading - Select file",
                                              filetypes=(("text files", "*.p"), ("all files", "*.*")))

        if self.debug.get():
            print("old constraints", self.constraints)
            print("old constraints type", type(self.constraints))
            print("loaded constraints file", spam)

        ## If no file selected
        if spam == "":
            self.status_set("No file selected.")
            return
        else:
            self.constraints_changed = True
            self.constraints_file.set(spam)
            self.z3_constraints = ""

            if append:
                if self.constraints == "":
                    self.constraints = []
                spam = pickle.load(open(self.constraints_file.get(), "rb"))
                self.constraints.extend(spam)
            else:
                try:
                    self.constraints = pickle.load(open(self.constraints_file.get(), "rb"))
                except pickle.UnpicklingError:
                    messagebox.showerror("Loading constraints", "Error, no constraints loaded")
                    return
                # self.constraints = []
                #
                # with open(self.constraints_file.get(), 'r') as file:
                #     for line in file:
                #         print(line[:-1])
                #         self.constraints.append(line[:-1])
            if self.debug.get():
                print("self.constraints", self.constraints)

            for constraint in self.constraints:
                if is_this_z3_function(constraint):
                    self.store_z3_constraints()
                    messagebox.showinfo("Loading constraints",
                                        "Some of the constraints contains z3 expressions, these are being stored and used only for z3 refinement, shown constraints are translated into python expressions.")
                    break

            constraints = ""
            for prop in self.constraints:
                constraints = f"{prop},\n{constraints}"
            constraints = constraints[:-2]

            self.constraints_text.configure(state='normal')
            self.constraints_text.delete('1.0', END)
            self.constraints_text.insert('end', constraints)
            # self.constraints_text.configure(state='disabled')

            ## Resetting parsed intervals
            self.parameters = []
            self.parameter_domains = []

            if not self.constraints:
                messagebox.showwarning("Loading constraints", "No constraints loaded. Please check input file.")
                self.status_set("No constraints loaded.")
            else:
                ## Autosave
                if not file:
                    self.save_constraints(os.path.join(self.tmp_dir, "constraints"))
                self.status_set("Constraints loaded.")

    def append_constraints(self):
        """ Appends loaded constraints from a pickled file to previously obtained constraints. """
        self.load_constraints(append=True)
        self.status_set("constraints appended.")

    def load_space(self, file=False):
        """ Loads space from a pickled file.
        Args:
            file (path/string): direct path to load the space file
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading space ...")

            if self.space:
                if not askyesno("Loading space", "Previously obtained space will be lost. Do you want to proceed?"):
                    return
            ## Delete previous space
            self.refresh_space()

            self.status_set("Please select the space to be loaded.")
            spam = filedialog.askopenfilename(initialdir=self.data_dir, title="Space loading - Select file",
                                              filetypes=(("pickled files", "*.p"), ("all files", "*.*")))

        ## If no file selected
        if spam == "":
            self.status_set("No file selected.")
            return
        else:
            self.space_changed = True
            self.space_file.set(spam)

            self.space = pickle.load(open(self.space_file.get(), "rb"))

            ## Show the space as niceprint()
            self.print_space()

            ## Ask if you want to visualise the space
            # self.show_samples = messagebox.askyesno("Loaded space", "Do you want to visualise samples?")
            self.show_samples = True
            # self.show_refinement = messagebox.askyesno("Loaded space", "Do you want to visualise refinement (safe & unsafe regions)?")
            self.show_refinement = True
            if self.space.true_point is not None:
                self.show_true_point = True
                # self.show_true_point = messagebox.askyesno("Loaded space", "Do you want to show the true point?")
            else:
                self.show_true_point = False
            self.show_space(self.show_refinement, self.show_samples, self.show_true_point, show_all=True)

            self.space_changed = True

            if not self.space:
                messagebox.showwarning("Loading space", "No space loaded. Please check input file.")
                self.status_set("No space loaded.")
            else:
                ## Autosave
                if not file:
                    self.save_space(os.path.join(self.tmp_dir, "space"))
                self.status_set("Space loaded.")

    def print_space(self, clear=False):
        """ Print the niceprint of the space into space text window.

        Args:
            clear (bool): if True the text is cleared
        """
        if not self.space == "":
            if not self.silent.get() and not clear:
                print("space", self.space)
                print()
                print("space nice print \n", self.space.nice_print())

            self.space_text.configure(state='normal')
            self.space_text.delete('1.0', END)
            if not clear:
                self.space_text.insert('end', self.space.nice_print())
            # self.space_text.configure(state='disabled')

    def show_space(self, show_refinement, show_samples, show_true_point, clear=False, show_all=False):
        """ Visualises the space in the plot.

        Args:
            show_refinement (bool): if True refinement is shown
            show_samples (bool): if True samples are shown
            show_true_point (bool): if True the true point is shown
            clear (bool): if True the plot is cleared
            show_all (bool):  if True, not only newly added rectangles are shown
        """
        if not self.space == "":
            if not clear:
                figure, axis = self.space.show(green=show_refinement, red=show_refinement, sat_samples=show_samples,
                                               unsat_samples=show_samples, true_point=show_true_point, save=False,
                                               where=[self.page6_figure, self.page6_a], show_all=show_all)

                ## If no plot provided
                if figure is None:
                    messagebox.showinfo("Load Space", axis)
                else:
                    self.page6_figure = figure
                    self.page6_a = axis
                    self.page6_figure.tight_layout()  ## By huypn
                    self.page6_figure.canvas.draw()
                    self.page6_figure.canvas.flush_events()
            else:
                self.page6_figure.clf()
                self.page6_a = self.page6_figure.add_subplot(111)
                self.page6_figure.tight_layout()  ## By huypn
                self.page6_figure.canvas.draw()
                self.page6_figure.canvas.flush_events()

    def edit_true_point(self):
        """ Sets the true point of the space """

        if self.space is "":
            print("No space loaded. Cannot set the true_point.")
            messagebox.showwarning("Edit True point", "Load space first.")
            return
        else:
            print(self.space.nice_print())
            self.create_window_to_load_param_point(parameters=self.space.params)
            self.space.true_point = self.parameter_values
            print(self.space.nice_print())

            self.print_space()
            self.page6_a.cla()
            self.show_space(self.show_refinement, self.show_samples, True, show_all=True)

    def parse_data_from_window(self):
        """ Parses data from the window. """
        # print("Parsing data ...")

        data = self.data_text.get('1.0', END)
        # print("parsed data as a string", data)
        data = data.split()
        for i in range(len(data)):
            if "," in data[i]:
                data[i] = float(data[i][:-1])
            else:
                data[i] = float(data[i])
        # print("parsed data as a list", data)
        self.data = data

    def save_model(self, file=False):
        """ Saves obtained model as a file.
        Args:
            file: file to save the model
        """
        ## TODO CHECK IF THE MODEL IS NON EMPTY
        # if len(self.model_text.get('1.0', END)) <= 1:
        #    self.status_set("There is no model to be saved.")
        #    return

        if file:
            save_model_file = file
        else:
            print("Saving the model ...")
            self.status_set("Please select folder to store the model in.")
            save_model_file = filedialog.asksaveasfilename(initialdir=self.model_dir, title="Model saving - Select file",
                                                           filetypes=(("pm files", "*.pm"), ("all files", "*.*")))
            if save_model_file == "":
                self.status_set("No file selected to store the model.")
                return

        if "." not in save_model_file:
            save_model_file = save_model_file + ".pm"
        # print("save_model_file", save_model_file)

        with open(save_model_file, "w") as file:
            file.write(self.model_text.get(1.0, END))

        if not file:
            self.status_set("Model saved.")

    def save_property(self, file=False):
        """ Saves obtained temporal properties as a file.

        Args:
            file: file to save the property
        """
        print("Saving the property ...")
        ## TODO CHECK IF THE PROPERTY IS NON EMPTY
        # if len(self.property_text.get('1.0', END)) <= 1:
        #    self.status_set("There is no property to be saved.")
        #    return

        if file:
            save_property_file = file
        else:
            self.status_set("Please select folder to store the property in.")
            save_property_file = filedialog.asksaveasfilename(initialdir=self.property_dir,
                                                              title="Property saving - Select file",
                                                              filetypes=(("pctl files", "*.pctl"), ("all files", "*.*")))
            if save_property_file == "":
                self.status_set("No file selected to store the property.")
                return

        if "." not in save_property_file:
            save_property_file = save_property_file + ".pctl"
        # print("save_property_file", save_property_file)

        with open(save_property_file, "w") as file:
            file.write(self.property_text.get(1.0, END))

        if not file:
            self.status_set("Property saved.")

    def generate_data_informed_properties(self):
        """ Generates Data informed property from temporal properties and data. Prints it. """
        if self.property_file.get() is "":
            messagebox.showwarning("Data informed property generation", "No property file loaded.")
            return False

        if self.data_intervals == "":
            print("Intervals not computed, properties cannot be generated")
            messagebox.showwarning("Data informed property generation", "Compute intervals first.")
            return False

        # general_create_data_informed_properties(prop_file, intervals, output_file=False)
        self.data_informed_property = general_create_data_informed_properties(self.property_file.get(), self.data_intervals, silent=self.silent.get())
        self.data_informed_property_text.configure(state='normal')
        self.data_informed_property_text.delete('1.0', END)
        spam = ""
        for item in self.data_informed_property:
            spam = spam + str(item) + ",\n"
        self.data_informed_property_text.insert('end', spam)

        ## Autosave
        self.save_data_informed_properties(os.path.join(self.tmp_dir, "data_informed_properties"))
        # self.data_informed_property_text.configure(state='disabled')

    def save_data_informed_properties(self, file=False):
        """ Saves computed data informed property as a text file.

        Args:
            file: file to save the data_informed_properties
        """
        print("Saving data informed property ...")
        ## TODO CHECK IF THE PROPERTY IS NON EMPTY
        # if len(self.property_text.get('1.0', END)) <= 1:
        #    self.status_set("There is no property to be saved.")
        #    return

        if file:
            save_data_informed_property_file = file
        else:
            self.status_set("Please select folder to store data informed property in.")
            save_data_informed_property_file = filedialog.asksaveasfilename(initialdir=self.property_dir,
                                                                            title="Data informed property saving - Select file",
                                                                            filetypes=(("pctl files", "*.pctl"), ("all files", "*.*")))
            if save_data_informed_property_file == "":
                self.status_set("No file selected to store data informed property.")
                return

        if "." not in save_data_informed_property_file:
            save_data_informed_property_file = save_data_informed_property_file + ".pctl"
        # print("save_property_file", save_property_file)

        with open(save_data_informed_property_file, "w") as file:
            file.write(self.data_informed_property_text.get('1.0', END))

        if not file:
            self.status_set("Data informed property saved.")

    ## TODO MAYBE IN THE FUTURE
    def save_mc_output_file(self, file=False):
        """ Saves parsed functions as a pickled file.

        Args:
            file: file to save the functions
        """
        print("Saving the rational functions ...")

        if self.functions is "":
            self.status_set("There are no rational functions to be saved.")
            return

        ## TODO choose to save rewards or normal functions
        if file:
            save_functions_file = file
        else:
            self.status_set("Please select folder to store the rational functions in.")
            if self.program is "prism":
                save_functions_file = filedialog.asksaveasfilename(initialdir=self.prism_results,
                                                                   title="Rational functions saving - Select file",
                                                                   filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
            elif self.program is "storm":
                save_functions_file = filedialog.asksaveasfilename(initialdir=self.storm_results,
                                                                   title="Rational functions saving - Select file",
                                                                   filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
            else:
                self.status_set("Error - Selected program not recognised.")
                save_functions_file = "Error - Selected program not recognised."
            if not self.silent.get():
                print("Saving functions in file: ", save_functions_file)

            if save_functions_file == "":
                self.status_set("No file selected to store the rational functions.")
                return

        if "." not in save_functions_file:
            save_functions_file = save_functions_file + ".txt"

        with open(save_functions_file, "w") as file:
            for line in self.functions:
                file.write(line)

        if not file:
            self.status_set("Rational functions saved.")

    def save_functions(self, file=False):
        """ Saves parsed rational functions as a pickled file.

        Args:
            file: file to save the parsed functions
        """

        if self.functions is "":
            self.status_set("There is no functions to be saved.")
            return

        if file:
            save_functions_file = file
        else:
            print("Saving the parsed functions ...")
            # print("self.program.get()", self.program.get())
            if self.program.get() == "prism":
                initial_dir = self.prism_results
            elif self.program.get() == "storm":
                initial_dir = self.storm_results
            else:
                messagebox.showwarning("Save parsed rational functions",
                                       "Select a program for which you want to save functions.")
                return

            save_functions_file = filedialog.asksaveasfilename(initialdir=initial_dir,
                                                               title="Rational functions saving - Select file",
                                                               filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
            if save_functions_file == "":
                self.status_set("No file selected to store the parsed rational functions.")
                return

        if "." not in save_functions_file:
            save_functions_file = save_functions_file + ".p"

        if not self.silent.get() and not file:
            print("Saving parsed functions as a file:", save_functions_file)

        pickle.dump(self.functions, open(save_functions_file, 'wb'))
        self.status_set("Parsed functions saved.")

    def save_data(self, file=False):
        """ Saves data as a pickled file.

        Args:
            file (string):  file to save the data
        """
        self.parse_data_from_window()

        if file:
            save_data_file = file
        else:
            print("Saving the data ...")

            if not self.data:
                messagebox.showwarning("Saving data", "There is no data to be saved.")
                self.status_set("There is no data to be saved.")
                return

            self.status_set("Please select folder to store the data in.")
            save_data_file = filedialog.asksaveasfilename(initialdir=self.data_dir, title="Data saving - Select file",
                                                          filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
            if save_data_file == "":
                self.status_set("No file selected to store the data.")
                return

        if "." not in save_data_file:
            save_data_file = save_data_file + ".p"

        if not self.silent.get():
            print("Saving data as a file:", save_data_file)

        pickle.dump(self.data, open(save_data_file, 'wb'))

        if not file:
            self.status_set("Data saved.")

    def save_intervals(self, file=False):
        """ Saves constraints as a pickled file.

        Args:
            file (string):  file to save the constraints
        """
        ## TODO
        pass

    def save_constraints(self, file=False):
        """ Saves constraints as a pickled file.

        Args:
            file (string):  file to save the constraints
        """

        if file:
            save_constraints_file = file
        else:
            print("Saving the constraints ...")
            if self.constraints is "":
                self.status_set("There is no constraints to be saved.")
                return

            self.status_set("Please select folder to store the constraints in.")
            save_constraints_file = filedialog.asksaveasfilename(initialdir=self.data_dir, title="constraints saving - Select file",
                                                                 filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
            if save_constraints_file == "":
                self.status_set("No file selected to store the constraints.")
                return

        if "." not in save_constraints_file:
            save_constraints_file = save_constraints_file + ".p"

        if not self.silent.get():
            print("Saving constraints as a file:", save_constraints_file)

        pickle.dump(self.constraints, open(save_constraints_file, 'wb'))
        if not file:
            self.status_set("constraints saved.")

    def save_space(self, file=False):
        """ Saves space as a pickled file.

        Args:
            file (string):  file to save the space
        """

        if file:
            save_space_file = file
        else:
            print("Saving the space ...")
            if self.space is "":
                self.status_set("There is no space to be saved.")
                return
            self.status_set("Please select folder to store the space in.")
            save_space_file = filedialog.asksaveasfilename(initialdir=self.data_dir, title="Space saving - Select file",
                                                           filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
            if save_space_file == "":
                self.status_set("No file selected to store the space in.")
                return

        if "." not in save_space_file:
            save_space_file = save_space_file + ".p"

        if not self.silent.get():
            print("Saving space as a file:", save_space_file)

        pickle.dump(self.space, open(save_space_file, 'wb'))
        if not file:
            self.status_set("Space saved.")

    ## ANALYSIS
    def synth_params(self):
        """ Computes rational functions from model and temporal properties. Saves output as a text file. """
        print("Synthesising parameters ...")
        self.status_set("Synthesising parameters.")
        proceed = True
        if self.functions_changed:
            proceed = askyesno("Parameter synthesis",
                               "Synthesising the parameters will overwrite current functions. Do you want to proceed?")

        if proceed:
            self.status_set("Parameter synthesis - checking inputs")

            if self.model_changed:
                messagebox.showwarning("Parameter synthesis",
                                       "The model for parameter synthesis has changed in the mean time, please consider that.")
            if self.property_changed:
                messagebox.showwarning("Parameter synthesis",
                                       "The properties for parameter synthesis have changed in the mean time, please consider that.")
            ## If model file not selected load model
            if self.model_file.get() is "":
                self.status_set("Load model for parameter synthesis")
                self.load_model()

            ## If property file not selected load property
            if self.property_file.get() is "":
                self.status_set("Load property for parameter synthesis")
                self.load_property()

            try:
                if self.program.get().lower() == "prism":
                    self.cursor_toggle_busy(True)
                    self.status_set("Parameter synthesis is running ...")
                    call_prism_files(self.model_file.get(), [], param_intervals=False, seq=False, noprobchecks=False,
                                     memory="", model_path="", properties_path=self.property_dir,
                                     property_file=self.property_file.get(), output_path=self.prism_results,
                                     gui=show_message, silent=self.silent.get())
                    ## Deriving output file
                    self.functions_file.set(str(os.path.join(Path(self.prism_results),
                                                             str(Path(self.model_file.get()).stem) + "_" + str(
                                                                 Path(self.property_file.get()).stem) + ".txt")))
                    self.status_set("Parameter synthesised finished. Output here: {}", self.functions_file.get())
                    self.load_mc_output_file(self.functions_file.get())

                elif self.program.get().lower() == "storm":
                    self.cursor_toggle_busy(True)
                    self.status_set("Parameter synthesis running ...")
                    call_storm_files(self.model_file.get(), [], model_path="", properties_path=self.property_dir,
                                     property_file=self.property_file.get(), output_path=self.storm_results, time=False)
                    ## Deriving output file
                    self.functions_file.set(str(os.path.join(Path(self.storm_results),
                                                             str(Path(self.model_file.get()).stem) + "_" + str(
                                                                 Path(self.property_file.get()).stem) + ".cmd")))
                    self.status_set("Command to run the parameter synthesis saved here: {}", self.functions_file.get())
                    self.load_mc_output_file(self.functions_file.get())
                else:
                    ## Show window to inform to select the program
                    self.status_set("Program for parameter synthesis not selected")
                    messagebox.showwarning("Synthesise", "Select a program for parameter synthesis first.")
                    return
            finally:
                self.cursor_toggle_busy(False)

            self.model_changed = False
            self.property_changed = False
            ## Resetting parsed intervals
            self.parameters = []
            self.parameter_domains = []

            # self.save_parsed_functions(os.path.join(self.tmp_dir, "parsed_functions"))
            self.cursor_toggle_busy(False)

    def sample_fun(self):
        """ Samples rational functions. Prints the result. """
        print("Sampling rational functions ...")
        self.status_set("Sampling rational functions. - checking inputs")
        if self.fun_size_q_entry.get() == "":
            messagebox.showwarning("Sampling rational functions", "Choose size_q, number of samples per dimension.")
            return
        if self.functions == "":
            messagebox.showwarning("Sampling rational functions", "Load the functions first, please")
            return

        self.status_set("Sampling rational functions.")
        self.validate_parameters(where=self.functions)

        try:
            self.cursor_toggle_busy(True)
            self.sampled_functions = sample_list_funs(self.functions, int(self.fun_size_q_entry.get()),
                                                      parameters=self.parameters, intervals=self.parameter_domains,
                                                      debug=self.debug.get(), silent=self.silent.get())
        finally:
            self.cursor_toggle_busy(False)
        self.sampled_functions_text.configure(state='normal')
        self.sampled_functions_text.delete('1.0', END)
        self.sampled_functions_text.insert('1.0', "rational function index, [parameter values], function value: \n")
        spam = ""
        for item in self.sampled_functions:
            spam = spam + str(item[0]+1) + ", ["
            for index in range(1, len(item)-1):
                spam = spam + str(item[index]) + ", "
            spam = spam[:-2]
            spam = spam + "], " + str(item[-1]) + ",\n"
        self.sampled_functions_text.insert('2.0', spam[:-2])
        # self.sampled_functions_text.configure(state='disabled')
        self.status_set("Sampling rational functions finished.")

    def show_funs_in_single_point(self):
        """ Plots rational functions in a given point. """
        print("Ploting rational functions in a given point ...")
        self.status_set("Ploting rational functions in a given point.")

        if self.functions == "":
            messagebox.showwarning("Ploting rational functions in a given point.", "Load the functions first, please.")
            return

        ## Disable overwriting the plot by show_funs_in_all_points
        if self.page3_figure_in_use.get():
            if not askyesno("Ploting rational functions in a given point",
                            "The result plot is currently in use. Do you want override?"):
                return
        self.page3_figure_in_use.set("1")

        self.validate_parameters(where=self.functions, intervals=False)

        self.status_set("Choosing parameters value:")
        self.create_window_to_load_param_point(parameters=self.parameters)

        self.reinitialise_plot()

        ## Getting the plot values instead of the plot itself

        #     self.initialise_plot(what=self.page3_figure, where=self.page3_plotframe)
        # else:
        #     pyplt.close()
        #     self.page3_figure = pyplt.figure()
        #     self.page3_a = self.page3_figure.add_subplot(111)
        # print("self.parameter_values", self.parameter_values)
        spam, egg = eval_and_show(self.functions, self.parameter_values, data=self.data, data_intervals=self.data_intervals,
                                  debug=self.debug.get(), where=[self.page3_figure, self.page3_a])

        if spam is None:
            messagebox.showinfo("Plots rational functions in a given point.", egg)
        else:
            self.page3_figure = spam
            self.page3_a = egg
            self.initialise_plot(what=self.page3_figure)
            self.page3_a.autoscale(enable=False)
            self.page3_figure.tight_layout()  ## By huypn
            # self.page3_figure.canvas.draw()
            # self.page3_figure.canvas.flush_events()

        if not self.silent.get():
            print(f"Using point", self.parameter_values)
        self.status_set("Sampling rational functions done.")

    def show_funs_in_all_points(self):
        """ Shows sampled rational functions in all sampled points. """
        print("Plotting sampled rational functions ...")
        self.status_set("Plotting sampled rational functions.")

        if self.page3_figure_in_use.get():
            if not askyesno("Show all sampled points", "The result plot is currently in use. Do you want override?"):
                return

        if self.functions == "":
            messagebox.showwarning("Sampling rational functions", "Load the functions first, please")
            return

        if self.fun_size_q_entry.get() == "":
            messagebox.showwarning("Sampling rational functions", "Choose size_q, number of samples per dimension.")
            return
        self.page3_figure_in_use.set("2")

        self.validate_parameters(where=self.functions)

        ## To be used to wait until the button is pressed
        self.button_pressed.set(False)
        self.Next_sample_button.config(state="normal")
        self.reinitialise_plot(set_onclick=True)

        for parameter_point in get_param_values(self.parameters, self.fun_size_q_entry.get(), False):
            if self.page3_figure_in_use.get() is not "2":
                return

            # print("parameter_point", parameter_point)
            self.page3_a.cla()
            spam, egg = eval_and_show(self.functions, parameter_point, data=self.data, data_intervals=self.data_intervals,
                                      debug=self.debug.get(), where=[self.page3_figure, self.page3_a])

            if spam is None:
                messagebox.showinfo("Plots rational functions in a given point.", egg)
            else:
                spam.tight_layout()
                self.page3_figure = spam
                self.page3_a = egg

                self.initialise_plot(what=self.page3_figure)
                # self.page3_a.autoscale(enable=False)
                # self.page3_figure.canvas.draw()
                # self.page3_figure.canvas.flush_events()

            self.Next_sample_button.wait_variable(self.button_pressed)
        # self.Next_sample_button.config(state="disabled")
        self.status_set("Plotting sampled rational functions finished.")

    def show_heatmap(self):
        """ Shows heatmap - sampling of a rational function in all sampled points. """
        print("Plotting heatmap of rational functions ...")
        self.status_set("Plotting heatmap of rational functions.")

        if self.page3_figure_in_use.get():
            if not askyesno("Plot heatmap", "The result plot is currently in use. Do you want override?"):
                return

        if self.functions == "":
            messagebox.showwarning("Plot heatmap", "Load the functions first, please")
            return

        if self.fun_size_q_entry.get() == "":
            messagebox.showwarning("Plot heatmap", "Choose size_q, number of samples per dimension.")
            return

        self.validate_parameters(where=self.functions)

        if len(self.parameters) is not 2:
            messagebox.showerror("Plot heatmap",
                                 f"Could not show this 2D heatmap. Parsed function(s) contain {len(self.parameters)} parameter(s), expected 2.")
            return

        self.page3_figure_in_use.set("3")
        ## To be used to wait until the button is pressed
        self.button_pressed.set(False)
        self.Next_sample_button.config(state="normal")

        self.reinitialise_plot(set_onclick=True)

        i = 0
        for function in self.functions:
            if self.page3_figure_in_use.get() is not "3":
                return
            i = i + 1
            self.page3_figure = heatmap(function, self.parameter_domains,
                                        [int(self.fun_size_q_entry.get()), int(self.fun_size_q_entry.get())],
                                        posttitle=f"Function number {i}: {function}", where=True,
                                        parameters=self.parameters)
            self.initialise_plot(what=self.page3_figure)

            self.Next_sample_button.wait_variable(self.button_pressed)
        # self.Next_sample_button.config(state="disabled")
        # self.page3_figure_locked.set(False)
        # self.update()
        self.status_set("Ploting sampled rational functions finished.")

    def optimize(self):
        """ Search for parameter values minimizing the distance of function to data. """
        print("Optimizing the distance between functions and data ...")
        self.status_set("Optimizing the distance between functions and data.")

        if self.functions == "":
            messagebox.showwarning("Optimize functions", "Load the functions first, please")
            return

        if self.data == "":
            messagebox.showwarning("Optimize functions", "Load the data first, please")
            return

        self.validate_parameters(where=self.functions)

        print("self.parameters", self.parameters)
        print("self.parameter_domains", self.parameter_domains)

        try:
            self.cursor_toggle_busy(True)
            ## TODO - tweak - update this to actually show the progress
            self.new_window = Toplevel(self)
            Label(self.new_window, text="Refinement in progress", anchor=W, justify=LEFT).pack()
            pb_hD = ttk.Progressbar(self.new_window, orient='horizontal', mode='indeterminate')
            pb_hD.pack(expand=True, fill=BOTH, side=TOP)
            pb_hD.start(50)
            self.update()

            result = optimize(self.functions, self.parameters, self.parameter_domains, self.data)
        except Exception as err:
            messagebox.showerror("Optimize", f"Error occurred during Optimization: {err}")
            raise err
            return
        finally:
            self.cursor_toggle_busy(False)
            self.new_window.destroy()

        self.optimised_param_point = result[0]
        self.optimised_function_value = result[1]
        self.optimised_distance = result[2]

        window = Toplevel(self)
        window.title('Result of optimisation')
        window.state('normal')
        width = max(len(str(result[0])), len(str(result[1])), len(str(result[2])))

        window.minsize(400, width+20)
        window.resizable(False, False)
        Label(window, text=f"Parameter point: ").grid(row=1)
        Label(window, text=f"Function values: ").grid(row=2)
        Label(window, text=f"Distance: ").grid(row=3)

        var = StringVar()
        var.set(result[0])
        ent = Entry(window, state='readonly', textvariable=var, width=width, relief='flat', readonlybackground='white', fg='black')
        ent.grid(row=1, column=1)

        var = StringVar()
        var.set(result[1])
        ent = Entry(window, state='readonly', textvariable=var, width=width, relief='flat', readonlybackground='white', fg='black')
        ent.grid(row=2, column=1)

        var = StringVar()
        var.set(result[2])
        ent = Entry(window, state='readonly', textvariable=var, width=width, relief='flat', readonlybackground='white', fg='black')
        ent.grid(row=3, column=1)

        save_optimisation_button = Button(window, text="Save Result", command=self.save_optimisation_result)
        save_optimisation_button.grid(row=4, column=1)

        ## Autosave
        self.save_optimisation_result(os.path.join(self.tmp_dir, "optimisation_results"))

        print("parameter point", self.optimised_param_point)
        print("function values", self.optimised_function_value)
        print("distance", self.optimised_distance)

    def save_optimisation_result(self, file=False):
        """ Stores optimisation results as a file

        Args:
            file (string):  file to store the optimisation results
        """
        if file:
            save_opt_result_file = file
        else:
            self.status_set("Please select folder to store the optimisation result.")

            save_opt_result_file = filedialog.asksaveasfilename(initialdir=self.optimisation_results_dir,
                                                                title="optimisation result saving - Select file",
                                                                filetypes=(("text file", "*.txt"), ("all files", "*.*")))
            if save_opt_result_file == "":
                self.status_set("No file selected to store the optimisation results.")
                return

        if "." not in save_opt_result_file:
            save_opt_result_file = save_opt_result_file + ".txt"

        with open(save_opt_result_file, "w") as file:
            file.write(f"parameter point {self.optimised_param_point} \n")
            file.write(f"function values {self.optimised_function_value} \n")
            file.write(f"distance {self.optimised_distance} \n")

    def create_data_intervals(self):
        """ Creates intervals from data. """
        print("Creating intervals ...")
        self.status_set("Create interval - checking inputs")
        if self.alpha_entry.get() == "":
            messagebox.showwarning("Creating intervals",
                                   "Choose alpha, the confidence measure before creating intervals.")
            return

        if self.n_samples_entry.get() == "":
            messagebox.showwarning("Creating intervals",
                                   "Choose n_samples, number of experimental samples before creating intervals")
            return

        ## If data file not selected load data
        if self.data_file.get() is "":
            self.load_data()
        # print("self.data_file.get()", self.data_file.get())

        ## Refresh the data from the window
        self.parse_data_from_window()

        self.status_set("Intervals are being created ...")
        self.data_intervals = create_intervals(float(self.alpha_entry.get()), float(self.n_samples_entry.get()), self.data)

        intervals = ""
        if not self.silent.get():
            print("Created intervals", self.data_intervals)
        for interval in self.data_intervals:
            intervals = f"{intervals},\n({interval.inf}, {interval.sup})"
        # print("intervals", intervals)
        intervals = intervals[2:]
        self.data_intervals_text.configure(state='normal')
        self.data_intervals_text.delete('1.0', END)
        self.data_intervals_text.insert('end', intervals)
        # self.data_intervals_text.configure(state='disabled')

        self.data_intervals_changed = True

        ## Autosave
        self.save_intervals(os.path.join(self.tmp_dir, "intervals"))

        self.status_set("Intervals created.")

    def sample_space(self):
        """ Samples (Parameter) Space. Plots the results. """
        print("Sampling space ...")
        self.status_set("Space sampling - checking inputs")
        ## Getting values from entry boxes
        self.size_q = int(self.size_q_entry.get())

        ## Checking if all entries filled
        if self.size_q == "":
            messagebox.showwarning("Sample space", "Choose size_q, number of samples before sampling.")
            return

        if self.constraints == "":
            messagebox.showwarning("Sample space", "Load or calculate constraints before refinement.")
            return

        ## Check space
        if not self.validate_space("Sample Space"):
            return

        self.status_set("Space sampling is running ...")
        if not self.silent.get():
            print("space.params", self.space.params)
            print("constraints", self.constraints)
            print("size_q", self.size_q)

        try:
            self.cursor_toggle_busy(True)

            self.new_window = Toplevel(self)
            Label(self.new_window, text="Sampling progress", anchor=W, justify=LEFT).pack()
            self.progress_bar = Progressbar(self.new_window, orient=HORIZONTAL, length=100, mode='determinate')
            self.progress_bar.pack()
            self.update()

            ## This progress is passed as whole to update the thing inside the called function
            self.space.grid_sample(self.constraints, self.size_q, silent=self.silent.get(), save=False, progress=self.update_progress_bar)
        finally:
            self.new_window.destroy()
            del self.new_window
            self.cursor_toggle_busy(False)

        self.print_space()

        self.show_space(show_refinement=False, show_samples=True, show_true_point=self.show_true_point)

        self.space_changed = False
        self.constraints_changed = False

        ## Autosave
        self.save_space(os.path.join(self.tmp_dir, "space"))

        self.status_set("Space sampling finished.")

    def hastings(self):
        """ Samples (Parameter) Space using Metropolis hastings """
        print("Space Metropolis-Hastings ...")
        self.status_set("Space Metropolis-Hastings - checking inputs")

        if self.constraints:
            messagebox.showwarning("Metropolis Hastings", "Data and functions are being used to run Metropolis Hasting, make sure they are in accordance with computed constrains.")

        ## TODO transformation back to data and functions from constraints #Hard_task
        if self.functions == "":
            messagebox.showwarning("Space Metropolis-Hastings", "Load functions before Metropolis-Hastings.")
            return

        if self.data == "":
            messagebox.showwarning("Space Metropolis-Hastings", "Load data before Metropolis-Hastings.")
            return

        ## Check functions / Get function parameters
        self.validate_parameters(where=self.functions)

        self.status_set("Space sampling using Metropolis Hastings is running ...")
        if not self.silent.get():
            print("functions", self.functions)
            print("function params", self.parameters)
            print("data", self.data)

        if not self.validate_space("Space Metropolis-Hastings"):
            return

        self.create_window_to_load_param_point(parameters=self.space.params)

        self.page6_figure2.clf()
        self.page6_b = self.page6_figure2.add_subplot(111)
        self.page6_figure2.tight_layout()  ## By huypn
        self.page6_figure2.canvas.draw()
        self.page6_figure2.canvas.flush_events()

        from metropolis_hastings import initialise_sampling

        try:
            self.cursor_toggle_busy(True)

            self.new_window = Toplevel(self)
            Label(self.new_window, text="Metropolis hastings progress", anchor=W, justify=LEFT).pack()
            self.progress_bar = Progressbar(self.new_window, orient=HORIZONTAL, length=100, mode='determinate')
            self.progress_bar.pack()
            self.update()

            ## This progress is passed as whole to update the thing inside the called function
            self.page6_figure2, self.page6_b = initialise_sampling(self.space, self.data, self.functions,
                                                                   int(self.n_samples_entry.get()),
                                                                   int(self.N_obs_entry.get()),
                                                                   int(self.MH_samples_entry.get()),
                                                                   float(self.eps_entry.get()),
                                                                   theta_init=self.parameter_values,
                                                                   where=[self.page6_figure2, self.page6_b],
                                                                   progress=self.update_progress_bar)
        finally:
            self.new_window.destroy()
            del self.new_window
            self.cursor_toggle_busy(False)

        # try:
        #     self.cursor_toggle_busy(True)
        #     initialise_sampling(self.space, self.data, self.functions, int(self.n_samples_entry.get()), int(self.N_obs_entry.get()), int(self.MH_samples_entry.get()), float(self.eps_entry.get()), where=[self.page6_figure2, self.page6_b])
        # except:
        #     messagebox.showerror(sys.exc_info()[1], "Try to check whether the data, functions, and computed constraints are aligned.")
        # finally:
        #     self.cursor_toggle_busy(False)
        self.page6_figure2.tight_layout()
        self.page6_figure2.canvas.draw()
        self.page6_figure2.canvas.flush_events()
        self.update()

    def refine_space(self):
        """ Refines (Parameter) Space. Plots the results. """
        print("Refining space ...")
        self.status_set("Space refinement - checking inputs")

        ## Getting values from entry boxes
        self.max_depth = int(self.max_dept_entry.get())
        self.coverage = float(self.coverage_entry.get())
        self.epsilon = float(self.epsilon_entry.get())
        self.delta = float(self.delta_entry.get())

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

        if self.alg.get() == "":
            messagebox.showwarning("Refine space", "Pick algorithm for the refinement before running.")
            return

        if self.constraints == "":
            messagebox.showwarning("Refine space", "Load or calculate constraints before refinement.")
            return

        if not self.validate_space("Refine Space"):
            return

        if int(self.alg.get()) <= 4 and not self.z3_constraints:
            for constraint in self.constraints:
                if is_this_exponential_function(constraint):
                    if not askyesno("Refinement", "Some constraints contain exponential function, we recommend using interval algorithmic (algorithm 5). Do you want to proceed anyway?"):
                        return
                    break

        self.status_set("Space refinement is running ...")
        # print(colored(f"self.space, {self.space.nice_print()}]", "blue"))
        try:
            self.cursor_toggle_busy(True)

            ## TODO - tweak - update this to actually show the progress
            self.new_window = Toplevel(self)
            Label(self.new_window, text="Refinement in progress", anchor=W, justify=LEFT).pack()
            pb_hD = ttk.Progressbar(self.new_window, orient='horizontal', mode='indeterminate')
            pb_hD.pack(expand=True, fill=BOTH, side=TOP)
            pb_hD.start(50)
            self.update()

            ## RETURNS TUPLE -- (SPACE,(NONE, ERROR TEXT)) or (SPACE, )
            ## feeding z3 solver with z3 expressions, python expressions otherwise
            if str(self.solver.get()) == "z3" and int(self.alg.get()) < 5 and self.z3_constraints:
                spam = check_deeper(self.space, self.z3_constraints, self.max_depth, self.epsilon, self.coverage,
                                    silent=self.silent.get(), version=int(self.alg.get()), size_q=False,
                                    debug=self.debug.get(), save=False, where=[self.page6_figure, self.page6_a],
                                    solver=str(self.solver.get()), delta=self.delta, gui=True)
            else:
                spam = check_deeper(self.space, self.constraints, self.max_depth, self.epsilon, self.coverage,
                                    silent=self.silent.get(), version=int(self.alg.get()), size_q=False,
                                    debug=self.debug.get(), save=False, where=[self.page6_figure, self.page6_a],
                                    solver=str(self.solver.get()), delta=self.delta, gui=True)
        finally:
            self.cursor_toggle_busy(False)
            self.new_window.destroy()
        ## If the visualisation of the space did not succeed
        if isinstance(spam, tuple):
            self.space = spam[0]
            messagebox.showinfo("Space refinement", spam[1])
        else:
            # self.show_space(True, True, True, clear=False)
            self.space = spam
            self.page6_figure.tight_layout()  ## By huypn
            self.page6_figure.canvas.draw()
            self.page6_figure.canvas.flush_events()

        self.print_space()

        self.constraints_changed = False
        self.space_changed = False

        ## Autosave
        self.save_space(os.path.join(self.tmp_dir, "space"))

        self.status_set("Space refinement finished.")

    ## VALIDATE VARIABLES (PARAMETERS, constraints, SPACE)
    def validate_parameters(self, where, intervals=True):
        """ Validates (functions, constraints, and space) parameters.

        Args:
            where (struct): a structure pars parameters from (e.g. self.functions)
            intervals (bool): whether to check also parameter intervals
        """
        if not self.parameters:
            print("Parsing parameters ...")
            globals()["parameters"] = set()
            for polynome in where:
                globals()["parameters"].update(find_param_old(polynome, debug=self.debug.get()))
            globals()["parameters"] = sorted(list(globals()["parameters"]))
            self.parameters = globals()["parameters"]
            if not self.silent.get():
                print("parameters", self.parameters)

        if (not self.parameter_domains) and intervals:
            ## TODO Maybe rewrite this as key and pass the argument to load_param_intervals
            self.key = StringVar()
            self.status_set("Choosing ranges of parameters:")
            self.new_window = Toplevel(self)
            label = Label(self.new_window,
                          text="Please choose intervals of the parameters to be used:")
            label.grid(row=0)
            self.key.set(" ")

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
                self.parameter_domains.append([spam_low, spam_high])
                i = i + 1

            ## To be used to wait until the button is pressed
            self.button_pressed.set(False)
            load_param_intervals_button = Button(self.new_window, text="OK",
                                                 command=self.load_param_intervals_from_window)
            load_param_intervals_button.grid(row=i)
            load_param_intervals_button.focus()
            load_param_intervals_button.bind('<Return>', self.load_param_intervals_from_window)

            load_param_intervals_button.wait_variable(self.button_pressed)
            ## print("key pressed")
        elif (len(self.parameter_domains) is not len(self.parameters)) and intervals:
            self.parameter_domains = []
            self.validate_parameters(where=where)

    def validate_constraints(self, position=False):
        """ Validates created properties.

        Args:
            position (string): Name of the place from which is being called e.g. "Refine Space"/"Sample space"
        """
        print("Validating constraints ...")
        ## MAYBE an error here
        if not self.constraints == "":
            print("constraints not empty, not checking them.")
            return True
        if position is False:
            position = "Validating constraints"
        ## If constraints empty create constraints
        if self.functions_changed or self.data_intervals_changed:
            if not self.silent.get():
                print("Functions: ", self.functions)
                print("Intervals: ", self.data_intervals)
            ## If functions empty raise an error (return False)
            if self.functions == "":
                print("No functions loaded nor not computed to create properties")
                messagebox.showwarning(position, "Load or synthesise functions first.")
                return False
            ## If intervals empty raise an error (return False)
            if self.data_intervals == "":
                print("Intervals not computed, properties cannot be computed")
                messagebox.showwarning(position, "Compute intervals first.")
                return False

            ## Check if the number of functions and intervals is equal
            if len(self.functions) != len(self.data_intervals):
                messagebox.showerror(position,
                                     "The number of rational functions and data points (or intervals) is not equal")
                return

            if self.functions_changed:
                self.functions_changed = False

            if self.data_intervals_changed:
                self.data_intervals_changed = False

            ## Create constraints
            self.constraints = ineq_to_constraints(self.functions, self.data_intervals, silent=self.silent.get())
            if self.z3_functions:
                self.z3_constraints = ineq_to_constraints(self.z3_functions, self.data_intervals, silent=self.silent.get())

            self.constraints_changed = True
            self.constraints_file.set("")

            constraints = ""
            for constraint in self.constraints:
                constraints = f"{constraints},\n{constraint}"
            constraints = constraints[2:]
            self.constraints_text.configure(state='normal')
            self.constraints_text.delete('1.0', END)
            self.constraints_text.insert('end', constraints)
            # self.constraints_text.configure(state='disabled')
            if not self.silent.get():
                print("constraints: ", self.constraints)
        return True

    def refresh_space(self):
        """ Unloads space. """
        if self.space:
            if not askyesno("Sample & Refine", "Data of the space, its text representation, and the plot will be lost. Do you want to proceed?"):
                return
        self.space_changed = False
        self.print_space(clear=True)
        self.show_space(None, None, None, clear=True)
        self.space_file.set("")
        self.space = ""
        self.parameters = ""
        self.parameter_domains = []
        self.status_set("Space deleted.")

    def validate_space(self, position=False):
        """ Validates space.

        Args:
            position (string): Name of the place from which is being called e.g. "Refine Space"/"Sample space"
        """
        print("Checking space ...")
        if position is False:
            position = "Validating space"
        ## If the space is empty create a new one
        if self.space == "":
            if not self.silent.get():
                print("Space is empty - creating a new one.")
            ## Parse params and its intervals
            self.validate_parameters(where=self.constraints)
            self.space = space.RefinedSpace(self.parameter_domains, self.parameters)
        else:
            if self.constraints_changed:
                messagebox.showwarning(position,
                                       "Using previously created space with new constraints. Consider using fresh new space.")
                ## Check if the properties and data are valid
                globals()["parameters"] = set()
                for polynome in self.constraints:
                    globals()["parameters"].update(find_param(polynome))
                globals()["parameters"] = sorted(list(globals()["parameters"]))
                self.parameters = globals()["parameters"]

                if not len(self.space.params) == len(self.parameters):
                    messagebox.showerror(position, "Cardinality of the space does not correspond to the constraints. Consider using fresh space.")
                    return False
                elif not sorted(self.space.params) == sorted(self.parameters):
                    messagebox.showerror(position, f"Parameters of the space - {self.space.params} - does not correspond to the one in constraints - {self.parameters}. Consider using fresh space.")
                    return False
        return True

    ## GUI MENU FUNCTIONS
    def edit_config(self):
        """ Opens config file in editor """
        print("Editing config ...")
        if "wind" in platform.system().lower():
            os.startfile(f'{os.path.join(workspace, "../config.ini")}')
        else:
            os.system(f'gedit {os.path.join(workspace, "../config.ini")}')
        self.load_config()  ## Reloading the config file after change
        self.status_set("Config file saved.")

    def show_help(self):
        """ Shows GUI help """
        print("Showing help ...")
        webbrowser.open_new("https://github.com/xhajnal/mpm#mpm")

    def check_updates(self):
        """ Shows latest releases """
        print("Checking for updates ...")
        self.status_set("Checking for updates ...")
        webbrowser.open_new("https://github.com/xhajnal/mpm/releases")

    def print_about(self):
        """ Shows GUI about """
        print("Printing about ...")
        top2 = Toplevel(self)
        top2.title("About")
        top2.resizable(0, 0)
        explanation = f" Mpm version: {self.version} \n More info here: https://github.com/xhajnal/mpm \n Powered by University of Konstanz and Masaryk University"
        Label(top2, justify=LEFT, text=explanation).pack(padx=13, pady=20)
        top2.transient(self)
        top2.grab_set()
        self.wait_window(top2)

        print(explanation)

    ## STATUS BAR FUNCTIONS
    def status_set(self, text, *args):
        """ Inner function to update status bar """
        self.status.config(text=text.format(args))
        self.status.update_idletasks()

    def status_clear(self):
        """ Inner function to update status bar """
        self.status.config(text="")
        self.status.update_idletasks()

    ## INNER TKINTER SETTINGS
    def cursor_toggle_busy(self, busy=True):
        """ Inner function to update cursor """
        if busy:
            ## System dependent cursor setting
            if "wind" in platform.system().lower():
                self.config(cursor='wait')
            else:
                self.config(cursor='clock')
        else:
            self.config(cursor='')
        self.update()

    def report_callback_exception(self, exc, val, tb):
        """ Inner function, Exception handling """
        import traceback
        print("Exception in Tkinter callback", file=sys.stderr)
        sys.last_type = exc
        sys.last_value = val
        sys.last_traceback = tb
        traceback.print_exception(exc, val, tb)
        messagebox.showerror("Error", message=str(val))
        if "maximum recursion depth" in str(val):
            self.python_recursion_depth = self.python_recursion_depth + 1000
            sys.setrecursionlimit(self.python_recursion_depth)

    ## INNER FUNCTIONS
    def create_window_to_load_param_point(self, parameters):
        """ Creates a window a functionality to load values of parameters"""
        self.new_window = Toplevel(self)
        label = Label(self.new_window, text="Please choose values of the parameters to be used:")
        label.grid(row=0)

        i = 1
        ## For each param create an entry
        self.parameter_values = []
        for param in parameters:
            Label(self.new_window, text=param, anchor=W, justify=LEFT).grid(row=i, column=0)
            spam = Entry(self.new_window)
            spam.grid(row=i, column=1)
            spam.insert(END, '0')
            self.parameter_values.append(spam)
            i = i + 1

        ## To be used to wait until the button is pressed
        self.button_pressed.set(False)
        load_true_point_button = Button(self.new_window, text="OK", command=self.load_param_values_from_window)
        load_true_point_button.grid(row=i)
        load_true_point_button.focus()
        load_true_point_button.bind('<Return>', self.load_param_values_from_window)

        load_true_point_button.wait_variable(self.button_pressed)

    def load_param_intervals_from_window(self):
        """ Inner function to parse the param intervals from created window """
        region = []
        for param_index in range(len(self.parameters)):
            ## Getting the values from each entry, low = [0], high = [1]
            region.append([float(self.parameter_domains[param_index][0].get()),
                           float(self.parameter_domains[param_index][1].get())])
        if not self.silent.get():
            print("Region: ", region)
        del self.key
        self.new_window.destroy()
        del self.new_window
        self.parameter_domains = region
        self.button_pressed.set(True)
        if not self.silent.get():
            if self.space:
                print("Space: ", self.space)

    def load_param_values_from_window(self):
        """ Inner function to parse the param values from created window """
        for index, param in enumerate(self.parameter_values):
            self.parameter_values[index] = float(self.parameter_values[index].get())
        self.new_window.destroy()
        del self.new_window
        self.button_pressed.set(True)

    def reinitialise_plot(self, set_onclick=False):
        """ Inner function, reinitialising the page3 plot """
        ## REINITIALISING THE PLOT
        ## This is not in one try catch block because I want all of them to be tried
        try:
            self.page3_plotframe.get_tk_widget().destroy()
        except AttributeError:
            pass
        try:
            self.page3_canvas.get_tk_widget().destroy()
        except AttributeError:
            pass
        try:
            self.page3_toolbar.get_tk_widget().destroy()
        except AttributeError:
            pass
        try:
            self.page3_figure.get_tk_widget().destroy()
        except AttributeError:
            pass
        try:
            self.page3_a.get_tk_widget().destroy()
        except AttributeError:
            pass
        self.page3_figure = pyplt.figure(figsize=(8, 4))
        self.page3_a = self.page3_figure.add_subplot(111)
        if set_onclick:
            def onclick(event):
                self.button_pressed.set(True)
            self.page3_figure.canvas.mpl_connect('button_press_event', onclick)
        self.update()

    def initialise_plot(self, what=False):
        """ Plots the what (figure) into where (Tkinter object - Window/Frame/....) """
        # try:
        #     self.page3_canvas.get_tk_widget().destroy()
        #     self.page3_toolbar.get_tk_widget().destroy()
        #     self.update()
        # except AttributeError:
        #     pass
        self.page3_plotframe = Frame(self.page3)
        self.page3_plotframe.grid(row=5, column=9, columnspan=10, padx=50, pady=4, sticky=N+S+E+W)

        self.page3_canvas = FigureCanvasTkAgg(what, master=self.page3_plotframe)
        self.page3_canvas.draw()
        self.page3_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.page3_toolbar = NavigationToolbar2Tk(self.page3_canvas, self.page3_plotframe)
        self.page3_toolbar.update()
        self.page3_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    def update_progress_bar(self, change):
        """ Updates progress bar

        Args:
            change (number): value to set the progress
        """
        self.progress_bar['value'] = round(100*change)
        self.update()

    def ask_quit(self):
        """ x button handler """
        if askyesno("Quit", "Do you want to quit the application?"):
            self.quit()

    def autoload(self):
        if askyesno("Autoload from tmp folder", "Would you like to load autosaved files from tmp folder?"):
            self.load_model(file=os.path.join(self.tmp_dir, "model.pm"))
            self.load_property(file=os.path.join(self.tmp_dir, "properties.pctl"))
            self.load_parsed_functions(file=os.path.join(self.tmp_dir, "functions.p"))
            # self.load_functions(file=os.path.join(self.tmp_dir, "functions_prism.txt"))
            # self.load_functions(file=os.path.join(self.tmp_dir, "functions_storm.txt"))
            self.load_data_intervals(file=os.path.join(self.tmp_dir, "intervals.p"))
            self.load_data(file=os.path.join(self.tmp_dir, "data.p"))
            self.load_constraints(file=os.path.join(self.tmp_dir, "constraints.p"))
            self.load_space(file=os.path.join(self.tmp_dir, "space.p"))


gui = Gui()
## System dependent fullscreen setting
if "wind" in platform.system().lower():
    gui.state('zoomed')
else:
    gui.attributes('-zoomed', True)

gui.protocol('WM_DELETE_WINDOW', gui.ask_quit)
sys.setrecursionlimit(20000)
gui.mainloop()
