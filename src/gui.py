import copy
import pickle
import os
import re
import sys
import time
import webbrowser
from platform import system
from time import time, localtime, strftime
import tkinter.font as TkFont
from collections.abc import Iterable
from copy import deepcopy
from tkinter import *
from tkinter import scrolledtext, messagebox

from psutil import virtual_memory
from sympy import factor, Interval
from pathlib import Path
from os.path import basename
from tkinter import filedialog, ttk
from tkinter.messagebox import askyesno
from tkinter.ttk import Progressbar
import matplotlib.pyplot as pyplt
import matplotlib
from termcolor import colored

# sys.setrecursionlimit(4000000)

## Importing my code
from common.convert import ineq_to_constraints, parse_numbers
from common.document_wrapper import show_message
from common.files import pickle_dump, pickle_load
from common.my_z3 import is_this_z3_function, translate_z3_function, is_this_exponential_function
from metropolis_hastings import HastingsResults
from mhmh import initialise_mhmh
from refine_space_parallel import check_deeper_parallel

error_occurred = None
matplotlib.use("TKAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import configparser

config = configparser.RawConfigParser()
workspace = os.path.dirname(__file__)
if workspace == "":
    workspace = os.getcwd()
sys.path.append(workspace)


def custom_paste(event):
    """ Fixes 'copy-pasting to Entry doesn't remove selected text' """
    try:
        event.widget.delete("sel.first", "sel.last")
    except:
        pass
    event.widget.insert("insert", event.widget.clipboard_get())
    return "break"


def do_config():
    """ Validates, set up config and creates directories from the config if not existing """
    config.read(os.path.join(workspace, "../config.ini"))

    for it in ["models", "properties", "data", "results", "tmp"]:
        try:
            subdir = config.get("paths", it)
        except configparser.NoSectionError as err:
            config.set("paths", it, f"{os.path.join(workspace, '..', it)}")
            subdir = config.get("paths", it)
            with open(os.path.join(workspace, "..", 'config.ini'), 'w') as configfile:
                config.write(configfile)
        if subdir == "":
            print(colored(f"{'paths', {it}, os.path.join(os.path.join(workspace,'..'), it) }", "blue"))
            config.set("paths", it, f"{os.path.join(os.path.join(workspace,'..'), it) }")
            # return False
        # print("subdir", subdir)
        if not os.path.isabs(subdir):
            main_dir = config.get("mandatory_paths", "cwd")
            if main_dir == "":
                main_dir = os.path.join(workspace, '..')
            subdir = os.path.join(main_dir, subdir)
            # print("new subdir", subdir)
        if not os.path.exists(subdir):
            # print("Making subdirectories", subdir)
            os.makedirs(subdir)
    return True


try:
    if not do_config():
        print("failed at loading folder, user edit")

except Exception as error:
    print(colored(f"An error occurred during loading config file: {error}", "red"))
    error_occurred = error

try:
    from mc_informed import general_create_data_informed_properties
    from load import load_mc_result, load_data, parse_constraints, parse_functions
    from load import parse_weights, parse_data_intervals, parse_data
    from common.model_stuff import parse_params_from_model, find_param, find_param_old
    from common.mathematics import create_proportions_interval
    import space
    from refine_space import check_deeper
    from mc import call_prism_files, call_storm
    from sample_n_visualise import sample_list_funs, eval_and_show, get_param_values, heatmap, bar_err_plot
    from optimize import optimize
except Exception as error:
    print(colored(f"An error occurred during importing module: {error}", "red"))
    error_occurred = error


## class copied from https://stackoverflow.com/questions/20399243/display-message-when-hovering-over-something-with-mouse-cursor-in-python/20399283
class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        """ Display text in tooltip window """

        if self.tipwindow or not text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=text, justify=LEFT,
                      background="#ffffe0", relief=SOLID, borderwidth=1,
                      font=("tahoma", "10", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


## copied from https://stackoverflow.com/questions/20399243/display-message-when-hovering-over-something-with-mouse-cursor-in-python/20399283
def createToolTip(widget, text):
    """ Creates a tooltip for a widget

    Args:
        widget (tk Object): widget to create the tooltip for
        text (string): text of the tooltip
    """
    tool_tip = ToolTip(widget)

    def enter(event):
        tool_tip.showtip(text)

    def leave(event):
        tool_tip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


class Gui(Tk):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        ## Fix copy does not delete selected text problem
        self.bind_class("Entry", "<<Paste>>", custom_paste)

        if error_occurred is not None:
            print(colored(error_occurred, "red"))
            messagebox.showerror("Loading modules", error_occurred)
            raise error_occurred
            # sys.exit()

        ## Trying to configure pyplot
        # pyplt.autoscale()
        pyplt.autoscale(tight=True)

        ## Variables
        ## Directories
        self.cwd = ""  ## Path to the main directory of the project
        self.model_dir = ""  ## Path to model
        self.property_dir = ""  ## Path to temporal properties
        self.data_dir = ""  ## Path to data
        self.data_weights_dir = ""  ## Path to data weights

        self.results_dir = ""  ## Path to results
        self.data_intervals_dir = ""  ## Path to data intervals
        self.constraints_dir = ""  ## Path to constraints
        self.prism_results = ""  ## Path to prism results
        self.storm_results = ""  ## Path to Storm results
        self.refinement_results = ""  ## Path to refinement results
        self.mh_results_dir = ""  ##  Path to mh results
        self.figures_dir = ""  ## Path to saved figures
        self.optimisation_results_dir = ""  ## Path to saved optimisation results
        self.tmp_dir = ""  ## Path for tmp folder

        ## Files
        self.model_file = StringVar()  ## Model file
        self.property_file = StringVar()  ## Property file
        self.data_informed_property_file = StringVar()  ## Data informed property file
        self.data_file = StringVar()  ## Data file
        self.data_weights_file = StringVar()  ## Data weights file
        self.data_intervals_file = StringVar()  ## Data intervals file
        self.mc_result_file = StringVar()  ## Data
        self.functions_file = StringVar()  ## Rational functions file
        self.constraints_file = StringVar()  ## constraints file
        self.space_file = StringVar()  ## Space file
        self.hastings_file = StringVar()  ## Metropolis-Hastings file

        ## Flags for the change
        self.model_changed = False
        self.property_changed = False
        self.functions_changed = False
        self.data_changed = False
        self.data_intervals_changed = False
        self.data_informed_property_changed = False
        self.data_weights_changed = False
        self.constraints_changed = False
        self.space_changed = False
        self.mh_results_changed = False

        ## Flags for modification
        self.model_text_modified = BooleanVar()
        self.properties_text_modified = BooleanVar()
        self.parsed_functions_text_modified = BooleanVar()
        self.data_text_modified = BooleanVar()
        self.data_intervals_text_modified = BooleanVar()
        self.data_weights_text_modified = BooleanVar()
        self.constraints_text_modified = BooleanVar()

        ## True Variables
        # self.model = ""
        # self.property = ""
        self.data = []  ## Experimental estimation of probabilities of functions
        self.data_weights = []  ## weights for the experimental estimation of probabilities of functions
        self.data_informed_property = ""  ## Property containing the interval boundaries from the data  ##TODO rewrite as [], need to go through checks
        self.functions = ""  ## Parameter synthesis results (rational functions)  ##TODO rewrite as [], need to go through checks
        self.z3_functions = ""  ## functions with z3 expressions inside  ##TODO rewrite as [], need to go through checks
        self.data_intervals = []  ## Computed intervals  ##TODO rewrite as [], need to go through checks
        self.parameters = ""  ##  Parsed parameters  ##TODO rewrite as [], need to go through checks
        self.parameter_domains = []  ## Parameters domains as intervals  ##TODO rewrite as [], need to go through checks
        self.constraints = ""  ## Computed or loaded constrains  ##TODO rewrite as [], need to go through checks
        self.z3_constraints = ""  ## Constrains with z3 expressions inside  ##TODO rewrite as [], need to go through checks
        self.space = ""  ## Instance of a RefinedSpace class
        self.mh_results = ""  ## Instance of HastingsResults class
        self.init_mh_with_optimised_point = BooleanVar()
        self.init_mh_with_optimised_point.set(False)

        ## Results
        self.sampled_functions = []  ## List of values of sampled functions True/False
        self.optimised_param_point = ""  ## List of parameter values with least distance
        self.optimised_function_value = ""  ## List of functions values with least distance
        self.optimised_distance = ""  ## The actual distance between functions and data

        ## Heatmap visualisation settings
        self.show_data_in_heatmap = BooleanVar()  ## Chooses between function vs. function - data point
        self.show_data_in_heatmap.set(False)

        ## Space visualisation settings
        self.show_samples = None  ## flag telling whether to show samples
        self.show_refinement = None  ## flag telling whether to show refinement
        self.show_space_true_point = None  ## flag telling whether to show true point
        self.show_mh_true_point = None  ## flag telling whether to show true point
        self.show_quantitative = None  ## flag telling whether to show quantitative sampling
        self.show_red_in_multidim_refinement = BooleanVar()  ## Chooses whether to show unsafe space over safe space in multidimensional plot
        self.show_red_in_multidim_refinement.set(False)
        self.hide_legend_refinement = BooleanVar()  ## Chooses to hide legend in upper plot
        self.hide_legend_refinement.set(False)
        self.hide_title_refinement = BooleanVar()  ## Chooses to hide title in upper plot
        self.hide_title_refinement.set(False)
        ## Metropolis-Hastings visualisation settings
        self.show_mh_as_scatter = BooleanVar()  ## Sets the MH plot to scatter plot (even for 2D)
        self.show_mh_metadata = BooleanVar()  ## Chooses whether to visualise MH metadata plots or not
        self.show_mh_metadata.set(True)

        ## Performance setting
        self.show_progress = True  ## Flag to optimize performance by not showing progress bar

        ## Saving settings
        self.save_as_plain_text = BooleanVar()  ## Flag to save structures as txt instead of pickled
        self.show_opt_point = BooleanVar()  ## Flag whether to show optimised point on figures
        self.save_as_plain_text.set(True)
        ## Save Figures
        self.save = BooleanVar()  ## True if saving on
        self.save.set(True)

        ## General Settings
        self.version = "1.27.4"  ## Version of the gui
        self.silent = BooleanVar()  ## Sets the command line output to minimum
        self.debug = BooleanVar()  ## Sets the command line output to maximum

        ## Default analysis settings
        # self.C = ""  ## Confidence level
        # self.n_samples = ""  ## Number of samples
        ## Load rat. functions
        self.program = StringVar()  ## "prism"/"storm"
        self.factorise = BooleanVar()  ## Flag for factorising functions
        ## Sampling
        self.sample_size = ""  ## Number of samples
        self.store_unsat_samples = True  ## Flag whether to store unsat samples, memory efficient to skipp it
        ## Refinement
        self.max_depth = ""  ## Max recursion depth
        self.coverage = ""  ## Coverage threshold
        self.epsilon = ""  ## Rectangle size threshold
        # self.alg = ""  ## Refinement alg. number
        self.presampled_refinement = BooleanVar()  ## Refinement flag (presample space)
        self.sampling_guided_refinement = BooleanVar()  ## Refinement flag (guide refinement with sampling)
        self.iterative_refinement = BooleanVar()  ## Refinement flag (refine check constraints independently in an iterative way) #Depricated
        # self.solver = ""  ## SMT solver - z3 or dreal
        self.delta = 0.01  ## dreal setting
        self.refinement_timeout = 0  ## timeout for refinement (0 is no timeout)
        self.mh_timeout = 0  ## timeout for Metropolis Hastings (0 is no timeout)

        ## INNER SETTINGS
        self.button_pressed = BooleanVar()  ## Inner variable to close created window
        self.python_recursion_depth = 1000  ## Inner python setting
        self.space_collapsed = True  ## Short / long print of space

        ## Other variables
        self.progress = StringVar()  ## Progress bar - progress value
        self.progress.set("0%")
        self.progress_time = StringVar()  ## Progress bar - time value
        self.progress_time.set("0")

    def gui_init(self):
        """ Initialisation procedure of GUI """
        ## GUI INIT
        self.title('DiPS')
        self.iconphoto(True, PhotoImage(file=os.path.join(workspace, "../icon.png")))

        if self.winfo_screenwidth() < 1600:
            messagebox.showwarning("Screen resolution", "Some features of the tool may not be visible due to low resolution.")
        elif self.winfo_screenwidth() < 1900:
            messagebox.showwarning("Screen resolution", "Some features of the tool may not be fully visible due to low resolution.")

        self.minsize(1000, 300)

        ## Temporal gui features
        self.progress_bar = None
        self.new_window = None

        #################################################### DESIGN ####################################################
        # print("height", self.winfo_height())

        #################################################### STATUS BAR ################################################
        self.status = Label(self, text="", bd=1, relief=SUNKEN, anchor=W)
        self.status.pack(side=BOTTOM, fill=X)

        ################################################ DESIGN - STATUS ###############################################
        frame = Frame(self)  ## Upper frame
        frame.pack(fill=X)

        left_frame = Frame(frame)
        left_frame.grid(row=0, column=0, sticky="nsew")

        Label(left_frame, text=f"Model file:", anchor=W, justify=LEFT).grid(row=0, column=0, sticky=W, padx=4)
        self.model_label = Label(left_frame, textvariable=self.model_file, anchor=W, justify=LEFT)
        self.model_label.grid(row=0, column=1, sticky=W, padx=4)

        Label(left_frame, text=f"Property file:", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, padx=4)
        self.property_label = Label(left_frame, textvariable=self.property_file, anchor=W, justify=LEFT)
        self.property_label.grid(row=1, column=1, sticky=W, padx=4)

        Label(left_frame, text=f"Functions file:", anchor=W, justify=LEFT).grid(row=2, column=0, sticky=W, padx=4)
        self.functions_label = Label(left_frame, textvariable=self.functions_file, anchor=W, justify=LEFT)
        self.functions_label.grid(row=2, column=1, sticky=W, padx=4)

        Label(left_frame, text=f"Data file:", anchor=W, justify=LEFT).grid(row=3, column=0, sticky=W, padx=4)
        self.data_label = Label(left_frame, textvariable=self.data_file, anchor=W, justify=LEFT)
        self.data_label.grid(row=3, column=1, sticky=W, padx=4)

        center_frame = Frame(frame)
        center_frame.grid(row=0, column=1, sticky="nsew")

        Label(center_frame, text=f"Data intervals file:", anchor=W, justify=LEFT).grid(row=0, column=0, sticky=W, padx=4)
        self.data_intervals_label = Label(center_frame, textvariable=self.data_intervals_file, anchor=W, justify=LEFT)
        self.data_intervals_label.grid(row=0, column=1, columnspan=2, sticky=W, padx=4)

        Label(center_frame, text=f"Constraints file:", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, padx=4)
        self.constraints_label = Label(center_frame, textvariable=self.constraints_file, anchor=W, justify=LEFT)
        self.constraints_label.grid(row=1, column=1, columnspan=2, sticky=W, padx=4)

        Label(center_frame, text=f"Data-informed properties file:", anchor=W, justify=LEFT).grid(row=2, column=0, sticky=W, padx=4)
        self.data_informed_properties_label = Label(center_frame, textvariable=self.data_informed_property_file, anchor=W, justify=LEFT)
        self.data_informed_properties_label.grid(row=2, column=1, columnspan=2, sticky=W, padx=4)


        right_frame = Frame(frame)
        right_frame.grid(row=0, column=2, sticky="nsew")

        Label(right_frame, text=f"Space file:", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, padx=4)
        self.space_label = Label(right_frame, textvariable=self.space_file, anchor=W, justify=LEFT)
        self.space_label.grid(row=1, column=1, columnspan=2, sticky=W, padx=4)

        Label(right_frame, text=f"Metropolis-Hastings file:", anchor=W, justify=LEFT).grid(row=2, column=0, sticky=W, padx=4)
        self.hastings_label = Label(right_frame, textvariable=self.hastings_file, anchor=W, justify=LEFT)
        self.hastings_label.grid(row=2, column=1, columnspan=2, sticky=W, padx=4)

        autosave_figures_button = Checkbutton(right_frame, text="Autosave figures", variable=self.save)
        autosave_figures_button.grid(row=3, column=0, sticky=W, padx=4)
        createToolTip(autosave_figures_button, text='Check to autosave results figures in folder results/figures')
        show_print_checkbutton = Checkbutton(right_frame, text="Minimal output", variable=self.silent)
        show_print_checkbutton.grid(row=3, column=1, sticky=W, padx=4)
        debug_checkbutton = Checkbutton(right_frame, text="Extensive output", variable=self.debug)
        debug_checkbutton.grid(row=3, column=2, sticky=W, padx=4)
        mh_metadata_button = Checkbutton(right_frame, text="Show MH metadata plots", variable=self.show_mh_metadata)
        mh_metadata_button.grid(row=4, column=0, sticky=W, padx=4)
        createToolTip(mh_metadata_button, text='Check to plot metadata plots of Metropolis-Hastings')

        save_as_plain_text_button = Checkbutton(right_frame, text="Save structures as plain text", variable=self.save_as_plain_text)
        save_as_plain_text_button.grid(row=4, column=1, sticky=W, padx=4)
        createToolTip(save_as_plain_text_button, text='Check to save functions and constraints as text file instead of compressed pickle files')

        save_as_plain_text_button = Checkbutton(right_frame, text="Show opt point", variable=self.show_opt_point)
        save_as_plain_text_button.grid(row=4, column=2, sticky=W, padx=4)
        createToolTip(save_as_plain_text_button, text='Check to show obtained optimised point on the plots of MH and refinement method.')

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        # print("self.silent", self.silent.get())

        ################################################################################################################
        ################################################ DESIGN - TABS #################################################
        ################################################################################################################
        # Defines and places the notebook widget
        nb = ttk.Notebook(self)  ## Tab part of the GUI
        nb.pack(fill="both", expand=1)

        ################################################### TAB EDIT ###################################################
        page1 = ttk.Frame(nb, name="model_properties")  # Adds tab 1 of the notebook
        nb.add(page1, text='Model & Properties', state="normal", sticky="nsew")

        frame_left = Frame(page1, width=int(self.winfo_width() / 2), height=int(self.winfo_width()/2))  ## Model part
        frame_left.grid_propagate(0)
        frame_left.rowconfigure(2, weight=1)
        frame_left.columnconfigure(6, weight=1)
        frame_left.pack(side=LEFT, fill=X)

        Button(frame_left, text='Open model', command=self.load_model).grid(row=0, column=0, sticky=W, padx=4, pady=4)  # pack(anchor=W)
        Button(frame_left, text='Save model', command=self.save_model).grid(row=0, column=1, sticky=W, padx=4, pady=4)  # pack(anchor=W)
        Label(frame_left, text=f"Loaded model file:", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, padx=4, pady=4)  # pack(anchor=W)

        self.model_text = scrolledtext.ScrolledText(frame_left, width=int(self.winfo_width() / 2), height=int(self.winfo_width()/2))
        # self.model_text.bind("<FocusOut>", self.refresh_model)
        self.model_text.bind("<Key>", lambda x: self.model_text_modified.set(True) if x.char != "" else None)
        # self.model_text.config(state="disabled")
        self.model_text.grid(row=2, column=0, columnspan=16, rowspan=2, sticky=W, padx=4, pady=4)  # pack(anchor=W, fill=X, expand=True)

        frame_right = Frame(page1, width=int(self.winfo_width() / 2), height=int(self.winfo_width()/2))  ## Property part
        frame_right.grid_propagate(0)
        frame_right.rowconfigure(5, weight=1)
        frame_right.columnconfigure(16, weight=1)
        frame_right.pack(side=RIGHT, fill=X)

        Button(frame_right, text='Open properties', command=self.load_property).grid(row=0, column=1, sticky=W, pady=4,
                                                                                   padx=4)  # pack(anchor=W)
        Button(frame_right, text='Save properties', command=self.save_property).grid(row=0, column=2, sticky=W, pady=4)  # pack(anchor=W)
        Label(frame_right, text=f"Loaded property file:", anchor=W, justify=LEFT).grid(row=1, column=1, sticky=W,
                                                                                       pady=4)  # pack(anchor=W)

        self.property_text = scrolledtext.ScrolledText(frame_right, width=int(self.winfo_width() / 2), height=int(self.winfo_width()/2))
        # self.property_text.bind("<FocusOut>", self.refresh_properties)
        self.property_text.bind("<Key>", lambda x: self.properties_text_modified.set(True) if x.char != "" else None)
        # self.property_text.config(state="disabled")
        self.property_text.grid(row=2, column=1, columnspan=16, rowspan=2, sticky=W, pady=4)  # pack(anchor=W, fill=X)

        # print(nb.select(0), type(nb.select(0)))
        # print(page1, type(page1))

        ############################################# TAB SYNTHESISE ###################################################
        page2 = ttk.Frame(nb, name="synthesise")  # Adds tab 2 of the notebook
        nb.add(page2, text='Synthesise functions')
        page2.grid_propagate(0)

        frame_left = Frame(page2, width=int(self.winfo_width() / 2), height=int(self.winfo_width() / 2))
        frame_left.grid_propagate(0)
        frame_left.rowconfigure(5, weight=1)
        frame_left.columnconfigure(6, weight=1)
        frame_left.pack(side=LEFT, fill=X)

        frame_right = Frame(page2, width=int(self.winfo_width() / 2), height=int(self.winfo_width() / 2))
        frame_right.grid_propagate(0)
        frame_right.rowconfigure(5, weight=1)
        frame_right.columnconfigure(3, weight=1)
        frame_right.pack(side=RIGHT, fill=X)

        ## SELECTING THE PROGRAM
        self.program.set("prism")

        ## Left (Model checking) Frame
        Label(frame_left, text=f"Parametric model checking and refinement (Storm and PRISM).", anchor=W, justify=LEFT).grid(row=0, column=0, columnspan=2, sticky=W, padx=4, pady=4)

        Label(frame_left, text="Select the program: ", anchor=W, justify=LEFT).grid(row=1, column=0, sticky=W, padx=4, pady=4)
        Radiobutton(frame_left, text="Prism", variable=self.program, value="prism").grid(row=1, column=1, sticky=W, pady=4)
        radio = Radiobutton(frame_left, text="Storm", variable=self.program, value="storm")
        radio.grid(row=1, column=2, sticky=W, pady=4)
        createToolTip(radio, text='If you did not install Storm this option results in a command that would produce desired output. (You may use docker installation to open command line and insert the command. Then load output file in DiPS.)')

        run_pmc_button = Button(frame_left, text='Run parametric model checking', command=self.synth_params)
        run_pmc_button.grid(row=3, column=0, sticky=W, padx=4, pady=4)
        createToolTip(run_pmc_button, text='Runs parametric model checking of model and properties to obtain rational function using selected model checker - PRISM or Storm')

        open_pmc_file_button = Button(frame_left, text='Open MC output file', command=self.load_functions_file)
        open_pmc_file_button.grid(row=3, column=1, sticky=W, pady=4)
        createToolTip(open_pmc_file_button, text='Loads result of parametric model checking result file (of PRISM/Storm) to parse rational function.')

        Label(frame_left, text=f"Loaded Prism/Storm output file:", anchor=W, justify=LEFT).grid(row=4, column=0, sticky=W, padx=4, pady=4)
        self.mc_result_text = scrolledtext.ScrolledText(frame_left, width=int(self.winfo_width() / 2), height=int(self.winfo_width() / 2), state=DISABLED)
        self.mc_result_text.grid(row=5, column=0, columnspan=16, rowspan=2, sticky=W, padx=4, pady=4)

        ## Right (Parsed functions) Frame
        Label(frame_right, text=f"Parsed rational functions.", anchor=W, justify=LEFT).grid(row=1, column=1, sticky=W, padx=4, pady=4)

        Label(frame_right, text=f"Show function(s):", anchor=W, justify=LEFT).grid(row=2, column=1, sticky=W, padx=4, pady=4)
        Radiobutton(frame_right, text="Original", variable=self.factorise, value=False).grid(row=2, column=2, sticky=W, pady=4)
        Radiobutton(frame_right, text="Factorised", variable=self.factorise, value=True).grid(row=2, column=3, sticky=W, pady=4)

        Button(frame_right, text='Open functions', command=self.load_parsed_functions).grid(row=3, column=1, sticky=W, padx=4, pady=4)
        Button(frame_right, text='Save functions', command=self.save_parsed_functions).grid(row=3, column=2, sticky=W, pady=4)

        Label(frame_right, text=f"Parsed function(s):", anchor=W, justify=LEFT).grid(row=4, column=1, sticky=W, padx=4, pady=4)
        self.functions_parsed_text = scrolledtext.ScrolledText(frame_right, width=int(self.winfo_width() / 2), height=int(self.winfo_width() / 2))
        # self.functions_parsed_text.bind("<FocusOut>", self.refresh_parsed_functions)
        self.functions_parsed_text.bind("<Key>", lambda x: self.parsed_functions_text_modified.set(True) if x.char != "" else None)
        self.functions_parsed_text.grid(row=5, column=1, columnspan=16, rowspan=2, sticky=W, pady=4)

        ######################################### TAB SAMPLE AND VISUALISE #############################################
        page3 = ttk.Frame(nb, name="sampling")
        nb.add(page3, text='Sample functions')
        page3.grid_propagate(0)

        frame_left = Frame(page3, width=int(self.winfo_width() * 0.3), height=int(self.winfo_width() / 2))
        frame_left.grid_propagate(0)
        frame_left.rowconfigure(5, weight=1)
        frame_left.columnconfigure(2, weight=1)
        frame_left.pack(side=LEFT, fill=X)

        self.frame3_right = Frame(page3, width=int(self.winfo_width() * 0.7), height=int(self.winfo_width() / 2))
        self.frame3_right.grid_propagate(0)
        self.frame3_right.rowconfigure(5, weight=1)
        self.frame3_right.columnconfigure(5, weight=1)
        self.frame3_right.pack(side=RIGHT, fill=X)

        Label(frame_left, text="Number of samples per variable (grid size):", anchor=W, justify=LEFT).grid(row=1, column=0, padx=4, pady=4)
        self.fun_sample_size_entry = Entry(frame_left)
        self.fun_sample_size_entry.grid(row=1, column=1)

        sample_function_button = Button(frame_left, text='Sample functions', command=self.sample_fun)
        sample_function_button.grid(row=2, column=0, sticky=W, padx=4, pady=4)
        createToolTip(sample_function_button, text='Samples functions in a regular grid of a given size')
        del sample_function_button

        Label(frame_left, text=f"Values of sampled points:", anchor=W, justify=LEFT).grid(row=3, column=0, sticky=W, padx=4, pady=4)

        self.sampled_functions_text = scrolledtext.ScrolledText(frame_left, width=int(self.winfo_width()/2), height=int(self.winfo_width()/2), state=DISABLED)
        self.sampled_functions_text.grid(row=4, column=0, columnspan=8, rowspan=2, sticky=W, padx=4, pady=4)

        Label(self.frame3_right, text=f"Functions visualisation", anchor=W, justify=CENTER).grid(row=1, column=1, columnspan=3, pady=4)
        plot_functions_in_a_given_point_button = Button(self.frame3_right, text='Plot functions in a given point', command=self.show_funs_in_single_point)
        plot_functions_in_a_given_point_button.grid(row=2, column=1, padx=4, pady=4)
        createToolTip(plot_functions_in_a_given_point_button, "Creates a barplot of function values in the given point, also showing data and interval values (if available).")
        del plot_functions_in_a_given_point_button

        plot_functions_in_all_points_button = Button(self.frame3_right, text='Plot all sampled points', command=self.show_funs_in_all_points)
        plot_functions_in_all_points_button.grid(row=2, column=2, padx=4, pady=4)
        createToolTip(plot_functions_in_all_points_button, "Creates a barplot of function values in each sampled point, also showing data and interval values (if available).")
        del plot_functions_in_all_points_button

        show_heat_map_button = Button(self.frame3_right, text='Heatmap', command=self.show_heatmap)
        show_heat_map_button.grid(row=2, column=3, padx=4, pady=4)
        createToolTip(show_heat_map_button, "Creates a heatmap for each function.")
        del show_heat_map_button

        ## TODO bring on in next update
        # show_data_in_heatmap_button = Checkbutton(self.frame3_right, text="Show distance to data", variable=self.show_data_in_heatmap)
        # show_data_in_heatmap_button.grid(row=2, column=4, padx=4, pady=4)
        # createToolTip(show_data_in_heatmap_button, "Showing distance of functions to data.")
        # del show_data_in_heatmap_button

        self.Next_sample_button = Button(self.frame3_right, text="Next plot", state="disabled", command=lambda: self.button_pressed.set(True))
        self.Next_sample_button.grid(row=3, column=2, padx=4, pady=4)
        createToolTip(self.Next_sample_button, "Iterates through created plots.")

        self.page3_figure = None
        # self.page3_figure = pyplt.figure()
        # self.page3_a = self.page3_figure.add_subplot(111)
        # print("type a", type(self.a))

        # self.page3.rowconfigure(5, weight=1)
        # self.page3.columnconfigure(18, weight=1)

        self.page3_figure_in_use = StringVar()
        self.page3_figure_in_use.set("")

        ################################################### TAB DATA ###################################################
        page4 = ttk.Frame(nb, name="data")
        nb.add(page4, text='Data & Intervals')
        # page4.columnconfigure(0, weight=1)
        # page4.rowconfigure(2, weight=1)
        # page4.rowconfigure(7, weight=1)

        frame_left = Frame(page4, width=int(self.winfo_width() / 2), height=int(self.winfo_height() * 0.8))
        frame_left.grid_propagate(0)
        frame_left.pack(side=LEFT, fill=X)

        frame_right = Frame(page4, width=int(self.winfo_width() / 2), height=int(self.winfo_height() * 0.8))
        frame_right.grid_propagate(0)
        frame_right.rowconfigure(8, weight=1)
        frame_right.columnconfigure(1, weight=1)
        frame_right.pack(side=RIGHT, fill=X)

        label44 = Label(frame_left, text="N_samples, number of samples: ", anchor=W, justify=LEFT)
        label44.grid(row=0, column=0, sticky=W, padx=4, pady=4)
        createToolTip(label44, text='Number of samples')

        self.n_samples_entry = Entry(frame_left)
        self.n_samples_entry.grid(row=0, column=1)

        Button(frame_left, text='Open data file', command=self.load_data).grid(row=1, column=0, sticky=W, padx=4, pady=4)
        Button(frame_left, text='Save data', command=self.save_data).grid(row=1, column=1, sticky=W, padx=4)
        Button(frame_left, text='Plot data', command=self.plot_data).grid(row=1, column=2, sticky=W, padx=4)

        label10 = Label(frame_left, text=f"Loaded data:", anchor=W, justify=LEFT)
        label10.grid(row=2, column=0, sticky=W, padx=4, pady=4)
        createToolTip(label10, text='For each function exactly one data point should be assigned.')

        a = 21
        Button(frame_left, text='Open weights file', command=self.load_data_weights).grid(row=1, column=a, sticky=W, padx=4, pady=4)
        Button(frame_left, text='Save weights', command=self.save_data_weights).grid(row=1, column=a + 1, sticky=W, padx=4)
        Button(frame_left, text='Discard weights', command=self.discard_data_weights).grid(row=1, column=a + 2, sticky=W, padx=4)

        label10 = Label(frame_left, text=f"Loaded weights:", anchor=W, justify=LEFT)
        label10.grid(row=2, column=a, sticky=W, pady=4)
        createToolTip(label10, text='For each data point exactly one weight should be assigned.')
        del a

        small_frame = ttk.Frame(frame_left, width=int(self.winfo_width() / 2), height=int(self.winfo_height() * 0.8))
        small_frame.grid(row=3, column=0, columnspan=50, sticky=W+E, padx=4, pady=4)

        self.data_text = scrolledtext.ScrolledText(small_frame)  # , height=10, width=30
        ## self.data_text.bind("<FocusOut>", self.parse_data)
        # self.data_text = Text(page4, height=12, state=DISABLED)  # , height=10, width=30
        # self.data_text.config(state="disabled")
        # self.data_text.bind("<FocusOut>", self.refresh_data)
        self.data_text.bind("<Key>", lambda x: self.data_text_modified.set(True) if x.char != "" else None)
        self.data_text.pack(side=LEFT, fill=BOTH, expand=True)

        self.data_weights_text = scrolledtext.ScrolledText(small_frame)  # , height=10, width=30
        ## self.data_text.bind("<FocusOut>", self.parse_data)
        # self.data_text = Text(page4, height=12, state=DISABLED)  # , height=10, width=30
        # self.data_text.config(state="disabled")
        # self.data_text.bind("<FocusOut>", self.refresh_data)
        self.data_weights_text.bind("<Key>", lambda x: self.data_weights_text_modified.set(True) if x.char != "" else None)
        self.data_weights_text.pack(side=RIGHT, fill=BOTH, expand=True)

        ## SET THE INTERVAL COMPUTATION SETTINGS
        button41 = Button(frame_left, text='Optimize parameters', command=self.optimize)
        button41.grid(row=4, column=0, sticky=W, padx=4, pady=4)
        createToolTip(button41, text='Using regression')

        label42 = Label(frame_left, text="C, confidence level:", anchor=W, justify=LEFT)
        label42.grid(row=5, column=0, sticky=W, padx=4, pady=4)
        createToolTip(label42, text='Confidence level')

        self.confidence_entry = Entry(frame_left)
        self.confidence_entry.grid(row=5, column=1)

        self.confidence_entry.insert(END, '0.90')
        self.n_samples_entry.insert(END, '60')

        label43 = Label(frame_left, text="Interval method: ", anchor=W, justify=LEFT)
        label43.grid(row=6, column=0, padx=0)
        createToolTip(label43, text='Choose from interval method')

        self.interval_method_entry = ttk.Combobox(frame_left, values=('CLT', 'Rule of three', 'Agresti-Coull', 'Wilson', 'Clopper_Pearson', 'Jeffreys', 'hsb'))
        self.interval_method_entry.grid(row=6, column=1)
        self.interval_method_entry.current(2)

        Button(frame_left, text='Compute intervals', command=self.compute_data_intervals).grid(row=7, column=0, sticky=W, padx=4, pady=4)
        Button(frame_left, text='Open intervals file', command=self.load_data_intervals).grid(row=7, column=1, sticky=W, padx=4, pady=4)
        Button(frame_left, text='Save intervals', command=self.save_data_intervals).grid(row=7, column=2, sticky=W, padx=4, pady=4)

        Label(frame_left, text=f"Loaded/computed intervals:", anchor=W, justify=LEFT).grid(row=8, column=0, sticky=W, padx=4, pady=4)

        self.data_intervals_text = scrolledtext.ScrolledText(frame_left, width=int(self.winfo_width() / 2), height=int(self.winfo_height() * 0.8 / 40), state=DISABLED)  # height=10, width=30
        # self.data_intervals_text.config(state="disabled")
        # self.data_intervals_text.bind("<FocusOut>", self.refresh_data_intervals)
        self.data_intervals_text.bind("<Key>", lambda x: self.data_intervals_text_modified.set(True) if x.char != "" else None)
        self.data_intervals_text.grid(row=9, column=0, rowspan=2, columnspan=50, sticky=W, padx=4, pady=4)
        # ttk.Separator(frame_left, orient=VERTICAL).grid(row=0, column=17, rowspan=10, sticky='ns', padx=50, pady=10)
        for i in range(50):
            frame_left.columnconfigure(i, weight=1)
        ## RIGHT FRAME

        Label(frame_right, text=f"Data informed property section.", anchor=W, justify=LEFT).grid(row=0, column=1, sticky=W, padx=5, pady=4)
        Label(frame_right, text=f"Loaded property file:", anchor=W, justify=LEFT).grid(row=1, column=1, sticky=W, padx=5, pady=4)

        self.property_text2 = scrolledtext.ScrolledText(frame_right, width=int(self.winfo_width() / 2), height=int(self.winfo_height() * 0.8 / 40), state=DISABLED)
        # self.property_text2.config(state="disabled")
        # self.property_text2.bind("<FocusOut>", self.refresh_data)
        self.property_text2.bind("<Key>", lambda x: self.properties_text_modified.set(True) if x.char != "" else None)
        self.property_text2.grid(row=2, column=1, columnspan=16, rowspan=2, sticky=W + E + N + S, padx=5, pady=4)
        Button(frame_right, text='Generate data informed properties', command=self.generate_data_informed_properties).grid(row=4, column=1, sticky=W, padx=5, pady=4)

        Button(frame_right, text='Open data informed properties', command=self.load_data_informed_properties).grid(row=4, column=2, sticky=W, padx=5, pady=4)

        Button(frame_right, text='Save data informed properties', command=self.save_data_informed_properties).grid(row=4, column=3, sticky=W, padx=5, pady=4)

        self.data_informed_property_text = scrolledtext.ScrolledText(frame_right, width=int(self.winfo_width() / 2), height=int(self.winfo_height() * 0.8 / 80), state=DISABLED)
        self.data_informed_property_text.grid(row=5, column=1, columnspan=16, rowspan=4, sticky=W + E + N + S, padx=5, pady=10)
        # Button(frame_right, text='Run parameter-lifting', command=self.external_refine).grid(row=9, column=2, sticky=W, padx=5, pady=4)

        ############################################### TAB CONSTRAINTS ################################################
        page5 = ttk.Frame(nb, width=400, height=200, name="constraints")
        nb.add(page5, text='Constraints')

        for i in range(1, 9):
            page5.columnconfigure(i, weight=1)
        page5.columnconfigure(10, pad=7)
        page5.rowconfigure(3, weight=1)
        page5.rowconfigure(5, pad=7)

        button = Button(page5, text='Calculate constraints', command=self.recalculate_constraints)
        button.grid(sticky=W, padx=12, pady=12)

        self.constraints_text = scrolledtext.ScrolledText(page5)
        # self.constraints_text.bind("<FocusOut>", self.refresh_constraints)
        self.constraints_text.bind("<Key>", lambda x: self.constraints_text_modified.set(True) if x.char != "" else None)
        self.constraints_text.grid(row=1, column=0, columnspan=9, rowspan=4, padx=5, sticky=E+W+S+N)

        label = Label(page5, text=f"Import/Export:", anchor=W, justify=LEFT)
        label.grid(row=5, column=0, padx=5)
        button = Button(page5, text='Open constraints', command=self.load_constraints)
        button.grid(row=5, column=1)
        button = Button(page5, text='Append constraints', command=self.append_constraints)
        button.grid(row=5, column=2)
        button = Button(page5, text='Save constraints', command=self.save_constraints)
        button.grid(row=5, column=3)

        ############################################ TAB SAMPLE AND REFINEMENT #########################################
        page6 = ttk.Frame(nb, name="refine")
        nb.add(page6, text='Analyze space')

        # frame_left = Frame(page6, width=500, height=200)
        # frame_left.pack(side=LEFT, expand=False)
        if self.winfo_screenwidth() < 2500:
            frame_left = Frame(page6, width=int(self.winfo_width() * 0.45), height=int(self.winfo_height()))
        else:
            frame_left = Frame(page6, width=int(self.winfo_width() * 0.4), height=int(self.winfo_height()))  ##4K
        frame_left.pack(side=LEFT)
        frame_left.grid_propagate(0)
        frame_left.rowconfigure(16, weight=1)
        for i in range(0, 9):
            frame_left.columnconfigure(i, weight=1)
        frame_left.columnconfigure(9, pad=7)
        frame_left.rowconfigure(14, weight=1)
        frame_left.rowconfigure(15, pad=7)

        self.frame_center = Frame(page6, width=int(self.winfo_width() / 2), height=int(self.winfo_height()))
        self.frame_center.grid_propagate(0)
        self.frame_center.pack(side=LEFT, fill=BOTH, expand=True)
        # self.frame_center.minsize(int(self.winfo_width() / 2), int(self.winfo_height()))

        ttk.Separator(frame_left, orient=HORIZONTAL).grid(row=1, column=0, columnspan=15, sticky='nwe', padx=10, pady=8)

        label61 = Label(frame_left, text="Grid size: ", anchor=W, justify=LEFT, padx=10)
        label61.grid(row=1, pady=16)
        createToolTip(label61, text='Number of samples per dimension')

        self.sample_size_entry = Entry(frame_left)
        self.sample_size_entry.grid(row=1, column=1)
        self.sample_size_entry.insert(END, '5')

        grid_sampling_button = Button(frame_left, text='Grid sampling', command=self.sample_space)
        grid_sampling_button.grid(row=7, column=0, columnspan=2, padx=10, pady=4)
        createToolTip(grid_sampling_button, text="Creates uniform grid of points with the given number of points in each dimension. Validates constraints in those points. Plots result.")

        grid_quantitative_sampling_button = Button(frame_left, text='Grid quantitative sampling', command=self.sample_space_degree)
        grid_quantitative_sampling_button.grid(row=8, column=0,  columnspan=2, padx=10, pady=4)
        createToolTip(grid_quantitative_sampling_button,
                      text="Creates uniform grid of points with the given number of points in each dimension. Computes sum of distances to disatisfy respective constraint. Plots result.")

        # ttk.Separator(frame_left, orient=VERTICAL).grid(row=1, column=2, rowspan=7, sticky='ns', padx=25, pady=25)

        # label71 = Label(frame_left, text="# of samples: ", anchor=W, justify=LEFT)
        # label71.grid(row=1, column=7)
        # createToolTip(label71, text='Number of samples to be used for sampling - subset of all samples')
        # self.observations_samples_size_entry = Entry(frame_left)
        # self.observations_samples_size_entry.grid(row=1, column=8)
        # self.observations_samples_size_entry.insert(END, '500')

        label71 = Label(frame_left, text="# of iterations: ", anchor=W, justify=LEFT)
        label71.grid(row=1, column=7, padx=(0, 2))
        createToolTip(label71, text='Number of iterations, steps in parameter space')
        self.MH_sampling_iterations_entry = Entry(frame_left)
        self.MH_sampling_iterations_entry.grid(row=1, column=8)
        self.MH_sampling_iterations_entry.insert(END, '500')

        # label72 = Label(frame_left, text="Eps: ", anchor=W, justify=LEFT)
        # label72.grid(row=2, column=7)
        # createToolTip(label72, text='Very small value used as probability of non-feasible values in prior')
        # self.eps_entry = Entry(frame_left)
        # self.eps_entry.grid(row=2, column=8)
        # self.eps_entry.insert(END, '0.0001')

        label72 = Label(frame_left, text="Grid size: ", anchor=W, justify=LEFT)
        label72.grid(row=2, column=7)
        createToolTip(label72, text='Number of segments in the plot')
        self.bins_entry = Entry(frame_left)
        self.bins_entry.grid(row=2, column=8)
        self.bins_entry.insert(END, '20')

        label73 = Label(frame_left, text="Jump size: ", anchor=W, justify=LEFT)
        label73.grid(row=3, column=7)
        createToolTip(label73, text='Standard deviation of walker jump length, the larger value the bigger jump in parameter space are made.')
        self.sd_entry = Entry(frame_left)
        self.sd_entry.grid(row=3, column=8)
        self.sd_entry.insert(END, '0.15')

        label74 = Label(frame_left, text="Burn-in: ", anchor=W, justify=LEFT)
        label74.grid(row=4, column=7)
        createToolTip(label74, text='Trim the fraction of accepted points from beginning')
        self.burn_in_entry = Entry(frame_left)
        self.burn_in_entry.grid(row=4, column=8)
        self.burn_in_entry.insert(END, '0.25')

        label75 = Label(frame_left, text="Timeout: ", anchor=W, justify=LEFT)
        label75.grid(row=5, column=7)
        createToolTip(label75, text='Timeout in seconds')
        self.mh_timeout_entry = Entry(frame_left)
        self.mh_timeout_entry.grid(row=5, column=8)
        self.mh_timeout_entry.insert(END, '3600')

        use_optimised_point_button = Checkbutton(frame_left, text="Use optimised point as initial", variable=self.init_mh_with_optimised_point)
        use_optimised_point_button.grid(row=8, column=7, sticky=W, padx=4, pady=4)
        createToolTip(use_optimised_point_button, text="by ticking this Metropolis-Hastings will start search in optimised point.")

        Button(frame_left, text='Metropolis-Hastings', command=self.hastings).grid(row=9, column=7, columnspan=2, pady=4)
        Button(frame_left, text='MHMH', command=self.mhmh).grid(row=10, column=7, columnspan=2, pady=4)

        # ttk.Separator(frame_left, orient=VERTICAL).grid(row=1, column=5, rowspan=7, sticky='ns', padx=25, pady=25)
        row = 1
        label62 = Label(frame_left, text="Max depth: ", anchor=W, justify=LEFT)
        label62.grid(row=row, column=3, padx=0)
        createToolTip(label62, text='Maximal number of splits, set negative for infinite')
        label63 = Label(frame_left, text="Coverage: ", anchor=W, justify=LEFT)
        row += 1
        label63.grid(row=row, column=3, padx=0)
        createToolTip(label63, text='Proportion of the nonwhite area to be reached')

        # label64 = Label(frame_left, text="Epsilon: ", anchor=W, justify=LEFT)
        # row += 1
        # label64.grid(row=row, column=3, padx=0)
        # createToolTip(label64, text='Minimal size of the rectangle to be checked (if 0 all rectangles are being checked)')

        label65 = Label(frame_left, text="Algorithm: ", anchor=W, justify=LEFT)
        row += 1
        label65.grid(row=row, column=3, padx=0)
        createToolTip(label65, text='Choose from algorithms:\n 1-4 - using SMT solvers \n 1 - DFS search \n 2 - BFS search \n 3 - BFS search with example propagation \n 4 - BFS with example and counterexample propagation \n 5 - interval algorithmic \n 6 - parameter lifting')

        label66 = Label(frame_left, text="SMT solver: ", anchor=W, justify=LEFT)
        row += 1
        label66.grid(row=row, column=3, padx=0)
        createToolTip(label66, text='When using SMT solver (alg 1-4), two solvers are available, z3 or dreal')

        label67 = Label(frame_left, text="Delta for dreal: ", anchor=W, justify=LEFT)
        row += 1
        label67.grid(row=row, column=3, padx=0)
        createToolTip(label67, text='When using dreal solver, delta is used to set solver error boundaries for satisfiability.')

        label68 = Label(frame_left, text="Timeout: ", anchor=W, justify=LEFT)
        row += 1
        label68.grid(row=row, column=3, padx=0)
        createToolTip(label68, text='Timeout in seconds. Set 0 for no timeout.')

        ## TODO uncomment this, version for QEST
        # label69 = Label(frame_left, text="Single Call Timeout: ", anchor=W, justify=LEFT)
        # row += 1
        # label69.grid(row=row, column=3, padx=0)
        # createToolTip(label69, text='Timeout of a single (rectangle) refine call in seconds. Set 0 for no timeout.')

        label610 = Label(frame_left, text="Number of cores to use: ", anchor=W, justify=LEFT)
        row += 1
        label610.grid(row=row, column=3, padx=0)
        createToolTip(label610, text='Number of processes running in parallel, set 1 for sequential.')

        ## TODO uncomment this, version for QEST
        # presampled_refinement_checkbutton = Checkbutton(frame_left, text=" Use presampled refinement", variable=self.presampled_refinement)
        # row += 1
        # presampled_refinement_checkbutton.grid(row=row, column=3, columnspan=2, padx=0)
        # createToolTip(presampled_refinement_checkbutton, text="Uses sampling before first refinement for creating region candidates to refine.")

        sampling_guided_refinement_checkbutton = Checkbutton(frame_left, text=" Use sampling guided refinement", variable=self.sampling_guided_refinement)
        row += 1
        sampling_guided_refinement_checkbutton.grid(row=row, column=3, columnspan=2, padx=0)
        createToolTip(sampling_guided_refinement_checkbutton, text="Before a rectangle is verified, sampling is used to help check satisfiability.")

        # iterative_refinement_checkbutton = Checkbutton(frame_left, text="Use iterative refinement (TBD)", variable=self.iterative_refinement)
        # iterative_refinement_checkbutton.grid(row=8, column=3, padx=0)

        self.max_depth_entry = Entry(frame_left)
        self.coverage_entry = Entry(frame_left)
        # self.epsilon_entry = Entry(frame_left)
        self.alg_entry = ttk.Combobox(frame_left, values=('1', '2', '3', '4', '5'))
        self.solver_entry = ttk.Combobox(frame_left, values=('z3', 'dreal'))
        self.delta_entry = Entry(frame_left)
        self.refinement_timeout_entry = Entry(frame_left)
        self.refinement_single_call_timeout_entry = Entry(frame_left)
        self.refinement_cores_entry = Entry(frame_left)

        row2 = 1
        self.max_depth_entry.grid(row=row2, column=4)
        row2 += 1
        self.coverage_entry.grid(row=row2, column=4)
        # row += 1
        # self.epsilon_entry.grid(row=row2, column=4)
        row2 += 1
        self.alg_entry.grid(row=row2, column=4)
        row2 += 1
        self.solver_entry.grid(row=row2, column=4)
        row2 += 1
        self.delta_entry.grid(row=row2, column=4)
        row2 += 1
        self.refinement_timeout_entry.grid(row=row2, column=4)
        ## TODO uncomment this, version for QEST
        # row += 1
        # self.refinement_single_call_timeout_entry.grid(row=row2, column=4)
        row2 += 1
        self.refinement_cores_entry.grid(row=row2, column=4)

        self.max_depth_entry.insert(END, '5')
        self.coverage_entry.insert(END, '0.95')
        # self.epsilon_entry.insert(END, '0')
        self.alg_entry.current(3)
        self.solver_entry.current(0)
        self.delta_entry.insert(END, '0.01')
        self.refinement_timeout_entry.insert(END, '3600')
        ## TODO uncomment this, version for QEST
        # self.refinement_single_call_timeout_entry.insert(END, '0')
        self.refinement_cores_entry.insert(END, '1')

        exact_refine_button = Button(frame_left, text='DiPS refine', command=self.refine_space)
        row += 1
        exact_refine_button.grid(row=row, column=3, columnspan=2, pady=4, padx=0)
        createToolTip(exact_refine_button, text="Run refinement with SMT solver (z3 / dreal) or interval arithmetic.")

        prism_refine_button = Button(frame_left, text='PRISM refine', command=self.external_refine_PRISM)
        row += 1
        prism_refine_button.grid(row=row, column=3, columnspan=1, pady=4, padx=0)
        createToolTip(prism_refine_button, text="Run approximate refinement using sampling by PRISM.")

        storm_refine_button = Button(frame_left, text='Storm refine', command=self.external_refine_Storm)
        storm_refine_button.grid(row=row, column=4, columnspan=1, pady=4, padx=0)
        createToolTip(storm_refine_button, text="Run parameter lifting, refinement method for models with multi-affine parametrisations, by Storm.")

        row += 1
        ttk.Separator(frame_left, orient=HORIZONTAL).grid(row=row, column=0, columnspan=15, sticky='nwe', padx=10, pady=4)

        row += 1
        Label(frame_left, text="Textual representation of space", anchor=CENTER, justify=CENTER, padx=10).grid(row=row, column=0, columnspan=15, sticky='nwe', padx=10, pady=4)
        self.space_text = scrolledtext.ScrolledText(frame_left, width=int(self.winfo_width() / 2), height=int(self.winfo_height() * 0.8/19), state=DISABLED)
        row += 1
        self.space_text.grid(row=row, column=0, columnspan=9, rowspan=2, sticky=W, padx=10)
        row += 1
        Button(frame_left, text='Extend / Collapse text', command=self.collapse_space_text).grid(row=row, column=3, sticky=S, padx=0, pady=(10, 10))
        Button(frame_left, text='Export text', command=self.export_space_text).grid(row=row, column=4, sticky=S, padx=0, pady=(10, 10))

        frame_right = Frame(page6)
        # frame_right.grid_propagate(0)
        # frame_right.rowconfigure(9, weight=1)
        # frame_right.columnconfigure(1, weight=1)
        frame_right.pack(side=RIGHT, fill=BOTH, anchor=W)

        Button(frame_right, text='Set True point', command=self.set_true_point_space).grid(row=0, column=0, padx=(4, 4), pady=7)
        Button(frame_right, text='Open space', command=self.load_space).grid(row=1, column=0, padx=(4, 4), pady=7)

        open_ref_file_button = Button(frame_right, text='Open PRISM refinement result', command=self.load_prism_refinement_output_file)
        open_ref_file_button.grid(row=2, column=0, padx=(4, 4), pady=7)
        createToolTip(open_ref_file_button, text='Loads result of refinement result file of PRISM to create Refined space.')

        open_ref_file_button = Button(frame_right, text='Open Storm refinement result', command=self.load_storm_refinement_output_file)
        open_ref_file_button.grid(row=3, column=0, padx=(4, 4), pady=7)
        createToolTip(open_ref_file_button, text='Loads result of refinement result file Storm to create Refined space.')

        Button(frame_right, text='Save space', command=self.save_space).grid(row=4, column=0, padx=(4, 4), pady=7)
        Button(frame_right, text='Delete space', command=self.refresh_space).grid(row=5, column=0, padx=(4, 4), pady=7)
        Button(frame_right, text='Customize Plot', command=self.customize_refinement_results).grid(row=6, column=0, padx=(4, 4), pady=7)

        Button(frame_right, text='Set True point', command=self.set_true_point_mh).grid(row=7, column=0, padx=(4, 4), pady=7)
        Button(frame_right, text='Load MH Results', command=self.load_mh_results).grid(row=8, column=0, padx=(4, 4), pady=7)
        Button(frame_right, text='Save MH Results', command=self.save_mh_results).grid(row=9, column=0, padx=(4, 4), pady=7)
        Button(frame_right, text='Delete MH Results', command=self.refresh_mh).grid(row=10, column=0, padx=(4, 4), pady=7)

        Button(frame_right, text='Customize Plots', command=self.customize_mh_results).grid(row=11, column=0, padx=(4, 4), pady=0)
        Button(frame_right, text='Show MH iterations', command=self.show_mh_iterations).grid(row=12, column=0, padx=(4, 4), pady=0)
        Button(frame_right, text='Show Acc points', command=self.show_mh_acc_points).grid(row=13, column=0, padx=(4, 4), pady=0)
        Button(frame_right, text='Export Acc points', command=self.export_acc_points).grid(row=14, column=0, padx=(4, 4), pady=0)

        frame_right.columnconfigure(0, weight=1)
        frame_right.rowconfigure(0, weight=1)
        frame_right.rowconfigure(6, weight=1)
        frame_right.rowconfigure(7, weight=1)
        frame_right.rowconfigure(10, weight=1)

        ##################################################### UPPER PLOT ###############################################
        self.page6_plotframe = Frame(self.frame_center)
        self.page6_plotframe.pack(side=TOP, fill=Y, expand=True, padx=5, pady=5)
        self.page6_figure = pyplt.figure(figsize=(8, 2))
        self.page6_figure.tight_layout()  ## By huypn

        self.page6_canvas = FigureCanvasTkAgg(self.page6_figure, master=self.page6_plotframe)  # A tk.DrawingArea.
        self.page6_canvas.draw()
        self.page6_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.page6_toolbar = NavigationToolbar2Tk(self.page6_canvas, self.page6_plotframe)
        self.page6_toolbar.update()
        self.page6_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.page6_a = self.page6_figure.add_subplot(111)

        self.set_lower_figure()
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
        # load_menu.add_command(label="Load functions", command=self.load_mc_result)
        # load_menu.add_command(label="Load data", command=self.load_data)
        # load_menu.add_command(label="Load space", command=self.load_space)
        # file_menu.add_separator()

        ## MENU-FILE-SAVE
        # save_menu = Menu(file_menu, tearoff=0)
        # file_menu.add_cascade(label="Save", menu=save_menu, underline=0)
        # save_menu.add_command(label="Save model", command=self.save_model)
        # save_menu.add_command(label="Save property", command=self.save_property)
        # # save_menu.add_command(label="Save functions", command=self.save_functions())
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
        # analysis_menu.add_command(label="Compute intervals", command=self.create_intervals_hsb)
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
        self.load_config()  ## Load the config file

    def load_config(self):
        """ Loads variables from the config file """
        ## TODO try catch paths
        ## TODO ADD EVERYWHERE print(f"current config entry {entry name} could be loaded, used the GUI default value")
        from common.config import load_config

        config = load_config()

        self.cwd = config["cwd"]
        # print("self.cwd", self.cwd)

        self.model_dir = config["models"]
        # print("self.model_dir", self.model_dir)

        self.property_dir = config["properties"]
        # print("self.property_dir", self.property_dir)

        self.data_dir = config["data"]
        # print("self.data_dir", self.data_dir)

        self.data_weights_dir = config["data_weights"]
        # print("self.data_weights_dir", self.data_weights_dir)

        ## Results
        self.results_dir = config["results"]
        # print("self.results_dir", self.results_dir)

        self.data_intervals_dir = os.path.join(self.results_dir, "data_intervals")
        # print("self.data_intervals_dir", self.data_intervals_dir)

        self.prism_results = os.path.join(self.results_dir, "prism_results")
        # print("self.prism_results", self.prism_results)

        self.storm_results = os.path.join(self.results_dir, "storm_results")
        # print("self.storm_results", self.storm_results)

        self.refinement_results = os.path.join(self.results_dir, "refinement_results")
        # print("self.refinement_results", self.refinement_results)

        self.constraints_dir = os.path.join(self.results_dir, "constraints")
        # print("self.constraints_dir", self.constraints_dir)

        self.figures_dir = os.path.join(self.results_dir, "figures")
        # print("self.figures_dir", self.figures_dir)

        self.optimisation_results_dir = os.path.join(self.results_dir, "optimisation_results")
        # print("self.optimisation_results_dir", self.optimisation_results_dir)

        self.mh_results_dir = os.path.join(self.results_dir, "mh_results")
        # print("self.mh_results_dir", self.mh_results_dir)

        self.tmp_dir = config["tmp"]
        # print("self.tmp_dir", self.tmp_dir)

        ## Interval settings
        n_samples = config["n_samples"]
        self.n_samples_entry.delete(0, 'end')
        self.n_samples_entry.insert(END, n_samples)

        confidence_level = config["confidence_level"]
        self.confidence_entry.delete(0, 'end')
        self.confidence_entry.insert(END, confidence_level)

        # Space sampling setting
        grid_size = config["grid_size"]
        self.sample_size_entry.delete(0, 'end')
        self.sample_size_entry.insert(END, grid_size)

        self.store_unsat_samples = config["store_unsat_samples"]

        # Space refinement setting
        self.max_depth = config["max_depth"]
        self.max_depth_entry.delete(0, 'end')
        self.max_depth_entry.insert(END, self.max_depth)

        self.coverage = config["coverage"]
        self.coverage_entry.delete(0, 'end')
        self.coverage_entry.insert(END, self.coverage)

        alg = config["alg"]
        self.alg_entry.set(alg)

        solver = config["solver"]
        self.solver_entry.delete(0, 'end')
        self.solver_entry.insert(END, solver)

        self.delta = config["delta"]
        self.delta_entry.delete(0, 'end')
        self.delta_entry.insert(END, self.delta)

        self.refinement_timeout = config["refinement_timeout"]
        self.refinement_timeout_entry.delete(0, 'end')
        self.refinement_timeout_entry.insert(END, self.refinement_timeout)

        # Metropolis-Hastings setting
        mh_iterations = config["mh_iterations"]
        self.MH_sampling_iterations_entry.delete(0, 'end')
        self.MH_sampling_iterations_entry.insert(END, mh_iterations)

        mh_grid_size = config["mh_grid_size"]
        self.bins_entry.delete(0, 'end')
        self.bins_entry.insert(END, mh_grid_size)

        burn_in = config["burn_in"]
        self.burn_in_entry.delete(0, 'end')
        self.burn_in_entry.insert(END, burn_in)

        self.mh_timeout = config["mh_timeout"]
        self.mh_timeout_entry.delete(0, 'end')
        self.mh_timeout_entry.insert(END, self.mh_timeout)

        # Performance setting
        self.show_progress = config["show_progress"]

        # Meta setting
        self.save.set(config["save"])
        self.silent.set(config["silent"])
        self.debug.set(config["debug"])
        self.show_mh_metadata.set(config["show_mh_metadata"])

        pyplt.rcParams["savefig.directory"] = self.figures_dir

        os.chdir(workspace)

        try:
            os.mkdir(os.path.join(self.figures_dir, "tmp"))
        except FileExistsError:
            pass

        try:
            os.mkdir(os.path.join(self.refinement_results, "tmp"))
        except FileExistsError:
            pass

    ## LOGIC
    ## FILE - LOAD, PARSE, SHOW, AND SAVE
    def load_model(self, file=False, ask=True):
        """ Loads model from a text file.

        Args:
            file (path/string): direct path to load the function file
            ask (bool): if False it will not ask questions
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading model ...")
            ## If some model previously loaded
            if len(self.model_text.get('1.0', END)) > 1 and ask:
                if not askyesno("Loading model", "Previously obtained model will be lost. Do you want to proceed?"):
                    return
            self.status_set("Please select the model to be loaded.")

            spam = filedialog.askopenfilename(initialdir=self.model_dir, title="Model loading - Select file",
                                              filetypes=(("pm files", "*.pm"), ("all files", "*.*")))
        ## If no file selected
        if spam == "" or spam == ():
            self.status_set("No file selected.")
            return
        else:
            ## If some model previously loaded
            if len(self.model_text.get('1.0', END)) > 1:
                self.model_changed = True
            self.model_file.set(spam)
            # self.model_text.configure(state='normal')
            self.model_text.delete('1.0', END)
            with open(self.model_file.get(), 'r') as f:
                self.model_text.insert('end', f.read())
            # self.model_text.configure(state='disabled')
            self.status_set("Model loaded.")
            # print("self.model", self.model.get())

            ## Autosave
            if not file:
                self.save_model(os.path.join(self.tmp_dir, "model.pm"))

    def load_property(self, file=False, ask=True):
        """ Loads temporal properties from a text file.

        Args:
            file (path/string): direct path to load the function file
            ask (bool): if False it will not ask questions
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading properties ...")
            ## If some property previously loaded
            if len(self.property_text.get('1.0', END)) > 1 and ask:
                if not askyesno("Loading properties",
                                "Previously obtained properties will be lost. Do you want to proceed?"):
                    return
            self.status_set("Please select the property to be loaded.")

            spam = filedialog.askopenfilename(initialdir=self.property_dir, title="Property loading - Select file",
                                              filetypes=(("property files", "*.pctl"), ("all files", "*.*")))
        ## If no file selected
        if spam == "" or spam == ():
            self.status_set("No file selected.")
            return
        else:
            ## If some property previously loaded
            if len(self.property_text.get('1.0', END)) > 1:
                self.property_changed = True
            self.property_file.set(spam)
            self.property_text.configure(state='normal')
            self.property_text.delete('1.0', END)
            with open(self.property_file.get(), 'r') as f:
                self.property_text.insert('end', f.read())
            # self.property_text.configure(state='disabled')

            self.property_text2.configure(state='normal')
            self.property_text2.delete('1.0', END)
            with open(self.property_file.get(), 'r') as f:
                self.property_text2.insert('end', f.read())
            # self.property_text2.configure(state='disabled')
            self.status_set("Property loaded.")
            # print("self.property", self.property.get())

            ## Autosave
            if not file:
                self.save_property(os.path.join(self.tmp_dir, "properties.pctl"))

    def load_data_informed_properties(self, file=False, ask=True):
        """ Loads Data informed property from temporal properties and data. Prints it.
        Args:
            file (path/string): direct path to load the function file
            ask (bool): if False it will not ask questions
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading data-informed properties ...")
            ## If some data-informed property previously loaded
            if len(self.property_text.get('1.0', END)) > 1 and ask:
                if not askyesno("Loading data-informed properties",
                                "Previously obtained data-informed properties will be lost. Do you want to proceed?"):
                    return
            self.status_set("Please select the data-informed properties to be loaded.")

            spam = filedialog.askopenfilename(initialdir=self.property_dir, title="Data-informed properties loading - Select file",
                                              filetypes=(("property files", "*.pctl"), ("all files", "*.*")))
        ## If no file selected
        if spam == "" or spam == ():
            self.status_set("No file selected.")
            return
        else:
            ## If some property previously loaded
            if len(self.data_informed_property_text.get('1.0', END)) > 1:
                self.data_informed_property_changed = True
            self.data_informed_property_file.set(spam)
            self.data_informed_property_text.configure(state='normal')
            self.data_informed_property_text.delete('1.0', END)
            with open(self.data_informed_property_file.get(), 'r') as f:
                self.data_informed_property_text.insert('end', f.read())
            # self.data_informed_property_text.configure(state='disabled')

            self.status_set("Data-informed property loaded.")

            ## Autosave
            if not file:
                self.save_data_informed_properties(os.path.join(self.tmp_dir, "data_informed_properties.pctl"))

    def load_functions_file(self, file=False, ask=True, program=False, reset_param_and_intervals=True, autosave=False):
        """ Loads parametric model checking output text file, which contain rational functions, symbolic representation
            of satisfaction/rewards of the model checking. Parses and shows the expressions.

        Args:
            file (path/string): direct path to load the function file
            ask (bool): if False it will not ask questions
            program (string): overrides the sel.program setting in ["prism", "storm"]
            reset_param_and_intervals (bool): if True the params will be reset
        """
        if program is False:
            program = self.program.get()

        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading rational functions ...")

            if self.functions_changed and ask:
                if not askyesno("Loading rational functions", "Previously obtained functions will be lost. Do you want to proceed?"):
                    return

            self.status_set("Loading rational functions - checking inputs")

            if not self.silent.get():
                print("Used program: ", program)
            if program == "prism":
                initial_dir = self.prism_results
            elif program == "storm":
                initial_dir = self.storm_results
            else:
                messagebox.showwarning("Load rational functions", "Select a program for which you want to load functions.")
                return

            self.status_set("Please select the prism/storm symbolic results to be loaded.")
            spam = filedialog.askopenfilename(initialdir=initial_dir, title="Rational functions loading - Select file",
                                              filetypes=(("text files", "*.txt"), ("all files", "*.*")))

        ## If no file / not a file selected
        if spam == "" or spam == ():
            self.status_set("No file selected.")
            return

        self.mc_result_file.set(spam)
        self.functions_changed = False

        try:
            self.cursor_toggle_busy(True)
            if self.factorise.get():
                self.status_set("Loading selected file and factorising...")
                if not self.silent.get():
                    print("Loading selected file and factorising...")
            else:
                self.status_set("Loading selected file ...")
                if not self.silent.get():
                    print("Loading selected file ...")

            spam = load_mc_result(os.path.abspath(self.mc_result_file.get()), tool=program,
                                  factorize=self.factorise.get(), rewards_only=False, f_only=False, refinement=False)

            self.functions, rewards = spam[0], spam[1]
            ## Merge functions and rewards
            for expression in rewards:
                self.functions.append(expression)
        finally:
            try:
                self.cursor_toggle_busy(False)
            except TclError:
                return

        ## Print mc result into TextBox
        self.mc_result_text.configure(state='normal')
        self.mc_result_text.delete('1.0', END)
        with open(self.mc_result_file.get(), 'r') as f:
            self.mc_result_text.insert('1.0', f.read())

        self.unfold_functions()

        self.functions_file.set(os.path.abspath(self.mc_result_file.get()))

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
                messagebox.showinfo("Loading rational functions", "Some of the functions contains z3 expressions, these are being stored and used only for z3 refinement, shown functions are translated into python expressions.")
                break

        ## Resetting parsed intervals
        if reset_param_and_intervals:
            self.parameters = []
            self.parameter_domains = []

        ## Check whether loaded
        if not self.functions:
            messagebox.showwarning("Loading rational functions", "No functions loaded. Please check input file.")
        else:
            pass
            ## Autosave
            ## TODO
            # if not file:
            #   self.save_functions(os.path.join(self.tmp_dir, f"functions_{program}"))

        ## AUTOSAVE
        if (not file) or autosave:
            self.save_parsed_functions(os.path.join(self.tmp_dir, f"functions.p"))

    def load_prism_refinement_output_file(self, file=False, reset_param_and_intervals=True, called_directly=True):
        """ Loads model refinement output text file of PRISM/STORM

                Args:
                    file (path/string): direct path to load the output file
                    reset_param_and_intervals (bool): if True the params will be reset
                    called_directly (bool): if True it will say where is the visualisation, otherwise where is text
                """
        self.load_refinement_output_file(file=file, program="prism", reset_param_and_intervals=reset_param_and_intervals,
                                         called_directly=called_directly)

    def load_storm_refinement_output_file(self, file=False, reset_param_and_intervals=True, called_directly=True):
        """ Loads model refinement output text file of PRISM/STORM

                Args:
                    file (path/string): direct path to load the output file
                    reset_param_and_intervals (bool): if True the params will be reset
                    called_directly (bool): if True it will say where is the visualisation, otherwise where is text
                """
        self.load_refinement_output_file(file=file, program="storm",
                                         reset_param_and_intervals=reset_param_and_intervals,
                                         called_directly=called_directly)

    def load_refinement_output_file(self, file=False, program=False, reset_param_and_intervals=True, called_directly=True):
        """ Loads model refinement output text file of PRISM/STORM

        Args:
            file (path/string): direct path to load the output file
            program (string): overrides the self.program setting, in ["prism", "storm"]
            reset_param_and_intervals (bool): if True the params will be reset
            called_directly (bool): if True it will say where is the visualisation, otherwise where is text
        """
        if program is False:
            program = self.program.get()

        if file:
            if not os.path.isfile(file):
                return
        else:
            print(f"Loading {program} output file ...")
            self.status_set(f"Loading {program} output file - checking inputs")

            if not self.silent.get():
                print("Used program: ", program)
            if program == "prism":
                initial_dir = self.prism_results
            elif program == "storm":
                initial_dir = self.storm_results
            else:
                messagebox.showwarning(f"Load partitioning output file", "Select a program for which you want to load refinement results.")
                return

            self.status_set("Please select the prism/storm symbolic results to be loaded.")
            file = filedialog.askopenfilename(initialdir=initial_dir, title=f"Loading {program} results - Select file",
                                              filetypes=(("text files", "*.txt"), ("all files", "*.*")))

        ## If no file / not a file selected
        if file == "" or file == ():
            self.status_set("No file selected.")
            return

        self.mc_result_file.set(file)

        # ## Print mc result into TextBox
        # self.mc_result_text.configure(state='normal')
        # self.mc_result_text.delete('1.0', END)
        # with open(self.mc_result_file.get(), 'r') as f:
        #     self.mc_result_text.insert('1.0', f.read())

        self.status_set("Refinement results loaded.")

        try:
            self.cursor_toggle_busy(True)
            self.status_set(f"Loading selected file: {file}")
            if not self.silent.get():
                print(f"Loading selected file: {file}")

            spam = load_mc_result(file, tool=program, rewards_only=False,  f_only=False, refinement=True, merge_results=True)

            skip_vis = False
            ## Get prams and domains
            self.parameters, self.parameter_domains, time_elapsed = spam[2], spam[3], spam[4]
            ## Get refinements
            spam = spam[0]

            ## Ask whether to show space
            if self.space != "":
                if called_directly:
                    if askyesno("Loading PRISM refinement result", "Space is not clear, do you want to override with current results?"):
                        ## Backup old space
                        self.save_space(os.path.join(self.tmp_dir, "space.p"))
                        self.space = ""
                    else:
                        skip_vis = True
                else:
                    ## Backup old space
                    self.save_space(os.path.join(self.tmp_dir, "space.p"))
                    self.space = ""

            ## Show space
            if not skip_vis:
                if program == "prism":
                    if len(spam) > 1:  ## number of refinements
                        messagebox.showwarning("Loading PRISM refinement result",
                                               "There is more refinements in the result, only the first is used.")
                    ## Get first refinement
                    try:
                        spam = spam[0]
                        self.space = space.RefinedSpace(self.parameter_domains, self.parameters)
                        if len(self.parameters) == 1:
                            for item in spam[0]:
                                self.space.add_green([item])
                            for item in spam[1]:
                                self.space.add_red([item])
                        else:
                            self.space.extend_green(spam[0])
                            self.space.extend_red(spam[1])
                        self.space.time_last_refinement = time_elapsed
                        self.space.time_refinement = self.space.time_refinement + time_elapsed

                        self.clear_space()
                        self.show_refinement = True
                        self.show_refinement = False
                        self.show_space_true_point = False
                        self.show_space(show_refinement=True, show_samples=False, show_true_point=False,
                                        show_opt_point=False, prefer_unsafe=self.show_red_in_multidim_refinement.get(),
                                        title=f"Approximate refinement, achieved_coverage:{round(self.space.get_coverage(), 3)}, solver: {program}")

                        self.space.remove_white(self.parameter_domains)
                        self.print_space()
                    except IndexError:
                        pass

                elif program == "storm":
                    # if len(self.parameters) != 2:
                    #     raise NotImplementedError("The results are loaded, but merging of interval bound-wise results of Storm refinement implemented only for two parameters so far.")
                    self.space = space.RefinedSpace(self.parameter_domains, self.parameters, rectangles_sat=spam[0], rectangles_unsat=spam[1], rectangles_unknown=spam[2])

                    self.space.time_last_refinement = time_elapsed
                    self.space.time_refinement = self.space.time_refinement + time_elapsed
                    self.clear_space()

                    self.show_refinement = True
                    self.show_refinement = False
                    self.show_space_true_point = False
                    self.show_space(show_refinement=True, show_samples=False, show_true_point=False,
                                    show_opt_point=False, prefer_unsafe=self.show_red_in_multidim_refinement.get(),
                                    title=f"Parameter lifting, achieved_coverage:{round(self.space.get_coverage(), 3)}, solver: {program}")
                    self.page6_figure.tight_layout()  ## By huypn
                    self.page6_figure.canvas.draw()
                    self.page6_figure.canvas.flush_events()
                    self.update()
                    self.print_space()
        finally:
            try:
                self.cursor_toggle_busy(False)
            except TclError:
                return

        ## Autosave
        self.save_space(os.path.join(self.tmp_dir, "space.p"))

        self.status_set("Refinement results parsed and loaded.")
        # if called_directly:
        #     if not skip_vis:
        #         messagebox.showinfo(f"Loading {program} refinement results",
        #                             "Visualisation of refinement can be seen in Analyze space tab.")
        # else:
        #     messagebox.showinfo(f"Loading {program} refinement results",
        #                         "Loaded refinement output can be seen in Synthesise functions tab.")

    def store_z3_functions(self):
        """ Stores a copy of functions as a self.z3_functions """
        self.z3_functions = deepcopy(self.functions)
        for index, function in enumerate(self.functions):
            assert isinstance(self.functions, list)
            self.functions[index] = translate_z3_function(function)

    def store_z3_constraints(self):
        """ Stores a copy of constraints as a self.z3_constraints """
        self.z3_constraints = deepcopy(self.constraints)
        assert isinstance(self.constraints, list)
        for index, constraint in enumerate(self.constraints):
            self.constraints[index] = translate_z3_function(constraint)

    def unfold_functions(self):
        """" Unfolds the function dictionary into a single list """

        # from functools import partial
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
            # unfold_functions2 = partial(self.unfold_something_2, [self.functions, "functions", self.functions_window, self.unfold_functions])
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
            self.functions_window.bind('<Return>', self.unfold_functions2)
        else:
            functions = ""
            for function in self.functions:
                functions = f"{functions},\n{function}"
            functions = functions[2:]

            self.functions_parsed_text.configure(state='normal')
            self.functions_parsed_text.delete('1.0', END)
            self.functions_parsed_text.insert('end', functions)
            # self.functions_parsed_text.configure(state='disabled')

    ## NOT WORKING WITH DATA
    # def unfold_something_2(self, something, something_in_text, window_to_destroy, this_function):
    #     """" Dummy method of unfold_something """
    #     try:
    #         something = something[self.key.get()]
    #     except KeyError:
    #         something = something[eval(self.key.get())]
    #
    #     if not self.silent.get():
    #         print(f"Parsed list of {something_in_text}: ", something)
    #     window_to_destroy.destroy()
    #     this_function()

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

    def load_parsed_functions(self, file=False, ask=True):
        """ Loads parsed rational functions from a pickled file.

        Args:
            file (path/string): direct path to load the function file
            ask (bool): if False it will not ask questions
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading parsed functions ...")
            if self.data_changed and ask:
                if not askyesno("Loading parsed functions",
                                "Previously obtained functions will be lost. Do you want to proceed?"):
                    return

            self.status_set("Please select the parsed functions to be loaded.")

            if not self.silent.get():
                print("Model checker: ", self.program.get())
            if self.program.get() == "prism":
                initial_dir = self.prism_results
            elif self.program.get() == "storm":
                initial_dir = self.storm_results
            else:
                messagebox.showwarning("Load functions", "Select a program for which you want to load rational functions.")
                return

            filetypes = (("pickle/text files", "*.p *.txt"), ("all files", "*.*"))
            spam = filedialog.askopenfilename(initialdir=initial_dir, title="Functions saving - Select file",
                                              filetypes=filetypes)

        ## If no file selected
        if spam == "" or spam == ():
            self.status_set("No file selected.")
            return
        else:
            ## Checking the valid type of the loaded file
            ## If loaded PRISM/Storm output instead, redirecting the load
            if Path(spam).suffix == ".txt":
                egg = parse_functions(spam)
                if egg[0].startswith("PRISM"):
                    self.load_functions_file(file=spam, program="prism", autosave=True)
                    return
                elif egg[0].startswith("Storm"):
                    self.load_functions_file(file=spam, program="storm", autosave=True)
                    return

            self.functions = []
            self.functions_changed = True
            self.functions_file.set(spam)
            print("self.functions_file", self.functions_file.get())
            self.z3_functions = ""
            self.mc_result_text.delete('1.0', END)

            if Path(self.functions_file.get()).suffix == ".txt":
                self.functions = parse_functions(self.functions_file.get())
            elif Path(self.functions_file.get()).suffix == ".p":
                self.functions = pickle_load(self.functions_file.get())

            ## Check whether functions not empty
            if not self.functions:
                messagebox.showwarning("Loading functions", "No functions loaded. Please check input file.")
                self.status_set("No functions loaded.")
                return

            print("loaded functions", self.functions)

            ## Factorising the parsed functions
            if self.factorise.get():
                try:
                    self.status_set("Factorising functions ...")
                    self.cursor_toggle_busy(True)

                    ## Progress Bar
                    self.new_window = Toplevel(self)
                    Label(self.new_window, text="Factorising functions progress:", anchor=W, justify=LEFT).pack()
                    Label(self.new_window, textvar=self.progress, anchor=W, justify=LEFT).pack()
                    self.progress_bar = Progressbar(self.new_window, orient=HORIZONTAL, length=100, mode='determinate')
                    self.progress_bar.pack()
                    self.update()
                    for index, function in enumerate(self.functions):
                        ## Factorise
                        self.functions[index] = str(factor(self.functions[index]))
                        self.update_progress_bar(change_to=(index+1)/len(self.functions))
                finally:
                    try:
                        self.cursor_toggle_busy(False)
                        self.new_window.destroy()
                        del self.new_window
                    except TclError:
                        return

            ## Check for z3 expressions
            for index, function in enumerate(self.functions):
                if self.debug.get():
                    print(f"function {index}, {function}")
                if is_this_z3_function(function):
                    self.store_z3_functions()
                    messagebox.showinfo("Loading functions", "Some of the functions contains z3 expressions, these are being stored and used only for z3 refinement, shown functions are translated into python expressions.")
                    break

            ## Print functions into TextBox
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
                self.save_parsed_functions(os.path.join(self.tmp_dir, f"functions.p"))

            self.status_set("Parsed functions loaded.")

    def load_data(self, file=False, ask=True):
        """ Loads data from a file. Either pickled list or comma separated values in one line

        Args:
            file (path/string): direct path to load the data file
            ask (bool): if False it will not ask questions
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading data ...")
            if self.data_changed and ask:
                if not askyesno("Loading data", "Previously obtained data will be lost. Do you want to proceed?"):
                    return

            self.status_set("Please select the data to be loaded.")

            filetypes = (("pickled / text files / comma separated values", "*.p *.txt *.csv"), ("all files", "*.*"))
            spam = filedialog.askopenfilename(initialdir=self.data_dir, title="Data loading - Select file",
                                              filetypes=filetypes)
        ## If no file selected
        if spam == "" or spam == ():
            self.status_set("No file selected.")
            return
        else:
            self.data = False
            self.data_changed = True
            self.data_file.set(spam)

            self.data = load_data(self.data_file.get(), silent=self.silent.get(), debug=not self.silent.get())
            if not self.data:
                messagebox.showerror("Loading data", f"Error, No data loaded.")
                self.status_set("Data not loaded properly.")
                return

            ## Unfolds and shows data
            self.unfold_data()

            if not self.silent.get():
                print("Loaded data: ", self.data)

            ## Clear intervals
            self.data_intervals = []
            self.data_intervals_text.delete('1.0', END)

            ## Autosave
            if not file:
                self.save_data(os.path.join(self.tmp_dir, "data.p"))

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
            print("Unfolded Data", self.data)
        self.new_window.destroy()
        self.unfold_data()

    def load_data_weights(self, file=False, ask=True):
        """ Loads data weights from a given file

        Args:
            file (path/string): direct path to load the data weights file
            ask (bool): if False it will not ask questions
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading data weights ...")

            if self.data_weights and ask:
                if not askyesno("Loading data weights", "Previously obtained weights will be lost. Do you want to proceed?"):
                    return

            self.status_set("Please select the data weights to be loaded.")
            filetypes = (("pickle/text files", "*.p *.txt"), ("all files", "*.*"))
            spam = filedialog.askopenfilename(initialdir=self.data_weights_dir, title="Data weights loading - Select file",
                                              filetypes=filetypes)

        ## If no file selected
        if spam == "" or spam == ():
            self.status_set("No file selected.")
            return
        else:
            self.data_weights_changed = True
            self.data_weights_file.set(spam)

            if Path(self.data_weights_file.get()).suffix == ".txt":
                self.data_weights = parse_weights(self.data_weights_file.get())
            elif Path(self.data_weights_file.get()).suffix == ".p":
                self.data_weights = pickle_load(self.data_weights_file.get())

            weights = ""
            if not self.silent.get():
                print("Loaded data weights", self.data_weights)
            for weight in self.data_weights:
                weights = f"{weights},\n{weight}"
            # print("weights", weights)
            weights = weights[2:]
            self.data_weights_text.configure(state='normal')
            self.data_weights_text.delete('1.0', END)
            self.data_weights_text.insert('end', weights)

            self.data_weights_changed = True

            if not self.data_weights:
                messagebox.showwarning("Loading data weights", "No data weights loaded. Please check input file.")
                self.status_set("No data weights loaded.")
            else:
                ## Autosave
                if not file:
                    self.save_data_weights(os.path.join(self.tmp_dir, "data_weights.p"))
                self.status_set("Data weights loaded.")

    def load_data_intervals(self, file=False, ask=True):
        """ Loads data intervals from a given file

        Args:
            file (path/string): direct path to load the data intervals file
            ask (bool): if False it will not ask questions
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading data intervals ...")

            if self.data_intervals and ask:
                if not askyesno("Loading data intervals", "Previously obtained intervals will be lost. Do you want to proceed?"):
                    return

            self.status_set("Please select the data intervals to be loaded.")
            filetypes = (("pickle/text files", "*.p *.txt"), ("all files", "*.*"))
            spam = filedialog.askopenfilename(initialdir=self.data_intervals_dir, title="Data intervals loading - Select file",
                                              filetypes=filetypes)

        ## If no file selected
        if spam == "" or spam == ():
            self.status_set("No file selected.")
            return
        else:
            self.data_intervals_changed = True
            self.data_intervals_file.set(spam)

            if ".txt" in self.data_intervals_file.get():
                self.data_intervals = parse_data_intervals(self.data_intervals_file.get())
            else:
                self.data_intervals = pickle_load(self.data_intervals_file.get())

            intervals = ""
            if not self.silent.get():
                print("Loaded data intervals", self.data_intervals)
            for interval in self.data_intervals:
                intervals = f"{intervals},\n({interval.inf}, {interval.sup})"
            # print("intervals", intervals)
            intervals = intervals[2:]
            self.data_intervals_text.configure(state='normal')
            self.data_intervals_text.delete('1.0', END)
            self.data_intervals_text.insert('end', intervals)

            self.data_intervals_changed = True

            if not self.data_intervals:
                messagebox.showwarning("Loading data intervals", "No data intervals loaded. Please check input file.")
                self.status_set("No data intervals loaded.")
            else:
                ## Autosave
                if not file:
                    self.save_data_intervals(os.path.join(self.tmp_dir, "data_intervals.p"))
                self.status_set("Data intervals loaded.")

    def recalculate_constraints(self):
        """ Merges functions and intervals into constraints. Shows it afterwards. """
        print("Checking the inputs.")
        self.check_changes("functions")
        self.check_changes("data_intervals")
        self.check_changes("data_weights")

        if self.data_weights:
            raise NotImplementedError("Weighted constraints are not Implemented yet")

        ## If there is some constraints
        if len(self.constraints_text.get('1.0', END)) > 1:
            proceed = messagebox.askyesno("Recalculate constraints",
                                          "Previously obtained constraints will be lost. Do you want to proceed?")
        else:
            proceed = True
        if proceed:
            self.constraints = ""
            self.z3_constraints = ""
            self.validate_constraints(position="constraints", force=True)
            ## Autosave
            self.save_constraints(os.path.join(self.tmp_dir, "constraints.p"))
            self.constraints_file.set(os.path.join(self.tmp_dir, "constraints.p"))
        self.status_set("Constraints recalculated and shown.")

    def load_constraints(self, file=False, append=False, ask=True):
        """ Loads constraints from a pickled file.

        Args:
            file (path/string): direct path to load the constraint file
            append (bool): if True, loaded constraints are appended to previous
            ask (bool): if False it will not ask questions
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading constraints ...")

            if self.constraints_changed and not append and ask:
                if not askyesno("Loading constraints", "Previously obtained constraints will be lost. Do you want to proceed?"):
                    return
            self.status_set("Please select the constraints to be loaded.")
            filetypes = (("pickle/text files", "*.p *.txt"), ("all files", "*.*"))
            spam = filedialog.askopenfilename(initialdir=self.constraints_dir, title="constraints loading - Select file",
                                              filetypes=filetypes)

        if self.debug.get():
            print("old constraints", self.constraints)
            print("old constraints type", type(self.constraints))
            print("loaded constraints file", spam)

        ## If no file selected
        if spam == "" or spam == ():
            self.status_set("No file selected.")
            return
        else:
            self.constraints_changed = True
            self.constraints_file.set(spam)
            self.z3_constraints = ""

            if Path(self.constraints_file.get()).suffix == ".txt":
                if append:
                    self.constraints.extend(parse_constraints(self.constraints_file.get()))
                else:
                    self.constraints = parse_constraints(self.constraints_file.get())
            else:
                if append:
                    if self.constraints == "":
                        self.constraints = []
                    spam = pickle_load(self.constraints_file.get())
                    self.constraints.extend(spam)
                else:
                    try:
                        self.constraints = pickle_load(self.constraints_file.get())
                    except pickle.UnpicklingError:
                        messagebox.showerror("Loading constraints", "Error, no constraints loaded")
                        return
                    # self.constraints = []
                    #
                    # with open(self.constraints_file.get(), 'r') as f:
                    #     for line in f:
                    #         print(line[:-1])
                    #         self.constraints.append(line[:-1])
            if self.debug.get():
                print("self.constraints", self.constraints)

            ## Check for z3 expressions
            for constraint in self.constraints:
                if is_this_z3_function(constraint):
                    self.store_z3_constraints()
                    messagebox.showinfo("Loading constraints",
                                        "Some of the constraints contains z3 expressions, these are being stored and used only for z3 refinement, shown constraints are translated into python expressions.")
                    break

            ## Print constraints into TextBox
            constraints = ""
            for constraint in self.constraints:
                constraints = f"{constraints},\n{constraint}"
            constraints = constraints[2:]
            self.constraints_text.configure(state='normal')
            self.constraints_text.delete('1.0', END)
            self.constraints_text.insert('end', constraints)
            # self.constraints_text.configure(state='disabled')

            ## Resetting parsed intervals
            self.parameters = []
            self.parameter_domains = []

            ## Check whether constraints not empty
            if not self.constraints:
                messagebox.showwarning("Loading constraints", "No constraints loaded. Please check input file.")
                self.status_set("No constraints loaded.")
            else:
                ## Autosave
                if not file:
                    self.save_constraints(os.path.join(self.tmp_dir, "constraints.p"))
                self.status_set("Constraints loaded.")

    def append_constraints(self):
        """ Appends loaded constraints from a pickled file to previously obtained constraints. """
        self.load_constraints(append=True)
        self.status_set("constraints appended.")

    def load_space(self, file=False, ask=True):
        """ Loads space from a pickled file.

        Args:
            file (path/string): direct path to load the space file
            ask (bool): if False it will not ask questions
        """
        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading space ...")

            if self.space and ask:
                if not askyesno("Loading space", "Previously obtained space will be lost. Do you want to proceed?"):
                    return
            ## Delete previous space
            self.refresh_space()

            self.status_set("Please select the space to be loaded.")
            spam = filedialog.askopenfilename(initialdir=self.refinement_results, title="Space loading - Select file",
                                              filetypes=(("pickled files", "*.p"), ("all files", "*.*")))

        ## If no file selected
        if spam == "" or spam == ():
            self.status_set("No file selected.")
            return
        else:
            try:
                self.cursor_toggle_busy(True)
                self.status_set("Space is being loaded.")
                self.space_changed = True
                self.space_file.set(spam)

                try:
                    self.space = pickle_load(self.space_file.get())
                except Exception as err:
                    self.space = ""
                    messagebox.showwarning("Loading space", f"Space could not be load: {err}")

                if self.space == "" or self.space == []:
                    return

                ## Backward compatibility: true_point
                if not hasattr(self.space, "true_point"):
                    self.space.set_true_point(False)

                ## Backward compatibility: opt_point
                if not hasattr(self.space, "opt_point"):
                    if self.optimised_param_point:
                        proceed = askyesno("Loading space",
                                           "Space you are about to load does not have an optimised point store. Do you want to show the one obtained?")
                        if proceed:
                            self.space.opt_point = self.optimised_param_point
                        else:
                            self.space.opt_point = False
                    else:
                        self.space.opt_point = False
                print("simsala bim")

                ## Backward compatibility
                self.space.update()

                ## Ask if you want to visualise the space
                # self.show_samples = messagebox.askyesno("Loaded space", "Do you want to visualise samples?")
                self.show_samples = True
                # self.show_refinement = messagebox.askyesno("Loaded space", "Do you want to visualise refinement (safe & unsafe regions)?")
                self.show_refinement = True
                if self.space.true_point is not None:
                    self.show_space_true_point = True
                    # self.show_true_point = messagebox.askyesno("Loaded space", "Do you want to show the true point?")
                else:
                    self.show_space_true_point = False

                try:
                    self.show_space(show_refinement=self.show_refinement, show_samples=self.show_samples,
                                    show_true_point=self.show_space_true_point, show_opt_point=self.show_opt_point.get(),
                                    show_all=True, prefer_unsafe=self.show_red_in_multidim_refinement.get(),
                                    quantitative=self.show_quantitative)
                except AttributeError as err:
                    self.space = ""
                    print(colored(str(err), "red"))
                    messagebox.showwarning("Loading space", "Space file is corrupted, could not load it.")
                    return
                ## Show the space as niceprint()
                # self.print_space()

                self.space_changed = True

                if not self.space:
                    messagebox.showwarning("Loading space", "No space loaded. Please check input file.")
                    self.status_set("No space loaded.")
                else:
                    ## Autosave
                    if not file:
                        self.save_space(os.path.join(self.tmp_dir, "space.p"))
                    self.status_set("Space loaded.")
            finally:
                try:
                    self.cursor_toggle_busy(False)
                except TclError:
                    return

    def load_mh_results(self, file=False, ask=True):
        """ loads Metropolis-Hastings results (accepted points) and plots them

        Args:
            file (path/string): direct path to load the pickled file
            ask (bool): if False it will not ask questions
        """

        if file:
            if not os.path.isfile(file):
                return
            spam = file
        else:
            print("Loading Metropolis Hasting results ...")

            if self.mh_results_changed and ask:
                if not askyesno("Loading Metropolis Hasting results", "Previously obtained plot will be lost. Do you want to proceed?"):
                    return
            ## Delete previous plot

            self.status_set("Please select the file to be loaded.")
            spam = filedialog.askopenfilename(initialdir=self.mh_results_dir, title="Loading Metropolis Hasting results - Select file",
                                              filetypes=(("pickled files", "*.p"), ("all files", "*.*")))

        ## If no file selected
        if spam == "" or spam == ():
            self.status_set("No file selected.")
            return
        else:
            self.mh_results_changed = True
            self.mh_results: HastingsResults = pickle_load(spam)
            self.hastings_file.set(spam)

            ## Backward compatibility: true point
            if not hasattr(self.mh_results, "true_point"):
                self.mh_results.true_point = False

            ## Backward compatibility: parameter intervals
            if not hasattr(self.mh_results, "parameter_intervals"):
                self.mh_results.fix_missing_parameter_domains()

            ## Backward compatibility: as scatter
            if not hasattr(self.mh_results, "as_scatter"):
                self.mh_results.set_as_scatter(False)

            ## Backward compatibility: true_point
            if not hasattr(self.mh_results, "true_point"):
                self.mh_results.set_true_point(False)

            ## Backward compatibility: opt_point
            if not hasattr(self.mh_results, "opt_point"):
                if self.optimised_param_point:
                    proceed = askyesno("Loading MH result", "MH result you are about to load does not have an optimised point store. Do you want to show the one obtained?")
                    if proceed:
                        self.mh_results.set_opt_point(self.optimised_param_point)
                    else:
                        self.mh_results.set_opt_point(False)
                else:
                    self.mh_results.set_opt_point(False)

            self.refresh_mh_figure(self.mh_results.bins, self.mh_results.burn_in, self.mh_results.as_scatter, True, self.show_opt_point.get())

            ## Autosave
            if not file:
                self.save_mh_results(os.path.join(self.tmp_dir, "mh_results.p"))

            self.status_set("Metropolis Hastings results loaded.")

    def print_space(self, clear=False):
        """ Print the niceprint of the space into space text window.

        Args:
            clear (bool): if True the text is cleared
        """
        if not self.space == "":
            if self.debug.get() and not clear:
                print("space: ", self.space)
                print()
                print("Space nice print:")
                assert isinstance(self.space, space.RefinedSpace)
                print(self.space.nice_print(full_print=not self.space_collapsed))

            self.space_text.configure(state='normal')
            self.space_text.delete('1.0', END)
            if not clear:
                self.space_text.insert('end', self.space.nice_print(full_print=not self.space_collapsed))
            # self.space_text.configure(state='disabled')

    def collapse_space_text(self):
        """ Collapses space text to hide long enumerations of elements"""
        self.space_collapsed = not self.space_collapsed
        self.print_space()

    def clear_space(self, warning=True):
        """ Will clear the space plot """
        self.show_space(False, False, False, False, clear=True, warnings=warning)

    def show_space(self, show_refinement, show_samples, show_true_point, show_opt_point, clear=False, show_all=False,
                   prefer_unsafe=False, quantitative=False, title="", warnings=True, is_parallel_refinement=False,
                   is_presampled=False, is_mhmh=False, is_sampling_guided=False):
        """ Visualises the space in the plot.

        Args:
            show_refinement (bool): if True refinement is shown in the plot
            show_samples (bool): if True samples are shown in the plot
            show_true_point (bool): if True the true point is shown in the plot
            show_opt_point (bool): if True teh optimised point is shown in the plot
            clear (bool): if True the plot is cleared
            show_all (bool):  if True, not only newly added rectangles are shown in the plot
            prefer_unsafe: if True unsafe space is shown in multidimensional space instead of safe
            quantitative (bool): if True show far is the point from satisfying / not satisfying the constraints
            title (string): adding title to the plot
            warnings (bool): if False will not show any warnings
            is_parallel_refinement (int): number of cores used for refinement
            is_presampled (bool): if True it will mark the refinement as presampled
            is_mhmh (bool): if True it will mark the refinement as MHMH, used MH to presampled/precut space
            is_sampling_guided (bool): flag whether refinement was sampling-guided
        """
        try:
            self.cursor_toggle_busy(True)
            self.status_set("Space is being visualised.")
            if not self.space == "":
                if not clear:
                    assert isinstance(self.space, space.RefinedSpace)
                    ## SHOW OPTIMISED POINT
                    if self.show_opt_point:
                        if self.optimised_param_point:
                            self.space.opt_point = self.optimised_param_point

                    self.print_space()
                    figure, axis = self.space.show(green=show_refinement, red=show_refinement, sat_samples=show_samples,
                                                   unsat_samples=show_samples, true_point=show_true_point, save=False,
                                                   opt_point=show_opt_point, where=[self.page6_figure, self.page6_a],
                                                   show_all=show_all, prefer_unsafe=prefer_unsafe, title=title,
                                                   quantitative=quantitative, is_parallel_sampling=True,
                                                   hide_legend=self.hide_legend_refinement.get(),
                                                   hide_title=self.hide_title_refinement.get(),
                                                   is_sampling_guided=is_sampling_guided, is_presampled=is_presampled,
                                                   is_parallel_refinement=is_parallel_refinement, is_mhmh=is_mhmh)
                    ## If no plot provided
                    if figure is None:
                        if warnings:
                            messagebox.showinfo("Show Space", axis)
                        else:
                            return
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
        finally:
            try:
                self.cursor_toggle_busy(False)
            except TclError:
                return

    def set_true_point_mh(self):
        """ Sets the true point for MH """
        try:
            self.new_window.destroy()
        except:
            pass

        if self.mh_results == "" or self.mh_results is None:
            print("No MH results. Cannot set the true point.")
            messagebox.showwarning("Edit True point", "Load/create MH results first.")
            return
        else:
            assert isinstance(self.mh_results, HastingsResults)

            self.parameter_domains = self.mh_results.parameter_intervals
            self.create_window_to_load_param_point(parameters=self.mh_results.params, opt=True)
            # self.mh_results.true_point = self.parameter_point
            self.mh_true_point = self.parameter_point
            self.mh_results.set_true_point(self.mh_true_point)
            print("Setting the mh true point as:", self.parameter_point)
            self.show_mh_true_point = True

            self.refresh_mh_figure(self.mh_results.bins, self.mh_results.burn_in, self.mh_results.as_scatter, self.show_mh_true_point, self.show_opt_point.get())

    def set_true_point_space(self):
        """ Sets the true point of the space """

        try:
            self.new_window.destroy()
        except:
            pass

        if self.space == "":
            print("No space. Cannot set the true point.")
            messagebox.showwarning("Edit True point", "Load/create space first.")
            return
        else:
            # print(self.space.nice_print())
            assert isinstance(self.space, space.RefinedSpace)
            if self.space.true_point:
                try:
                    self.space.true_point_object.remove()
                except ValueError:
                    print(colored("Could not remove true point", "red"))
                    pass
            if self.space.opt_point:
                try:
                    self.space.opt_point_object.remove()
                except ValueError:
                    print(colored("Could not remove true point", "red"))
                    pass
            self.parameter_domains = self.space.region
            self.create_window_to_load_param_point(parameters=self.space.params, opt=True)
            self.space.true_point = self.parameter_point
            self.show_space_true_point = True

            self.print_space()
            figure, axis = self.space.show_points(where=[self.page6_figure, self.page6_a],
                                                  hide_legend=self.hide_legend_refinement.get(),
                                                  show_true_point=self.show_space_true_point,
                                                  show_opt_point=self.show_opt_point.get())

            ## If no plot provided
            if figure is None:
                messagebox.showinfo("Show true point failed", axis)
            else:
                self.page6_figure = figure
                self.page6_a = axis
                self.page6_figure.tight_layout()  ## By huypn
                self.page6_figure.canvas.draw()
                self.page6_figure.canvas.flush_events()

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

    def save_file(self, content, file_name):
        if "." not in basename(file_name):
            if self.save_as_plain_text.get():
                file_name = file_name + ".txt"
                with open(file_name, "w") as file:
                    file.write(str(content))
            else:
                file_name = file_name + ".p"
                pickle_dump(content, file_name)
        else:
            if file_name.split(".")[-1] == "p":
                pickle_dump(content, file_name)
            elif file_name.split(".")[-1] == "txt":
                with open(file_name, "w") as file:
                    file.write(str(content))
            else:
                print(colored(f"Unknown file extension {file_name.split('.')[-1]} when saving file {file_name}. We trim it out.", "red"))
                file_name = file_name.split(".")[0]
                self.save_file(content, file_name)

    def save_model(self, file=False):
        """ Saves obtained model as a file.

        Args:
            file (bool or Path or string): file to save the model
        """
        ## TODO CHECK IF THE MODEL IS NON EMPTY
        # if len(self.model_text.get('1.0', END)) <= 1:
        #    self.status_set("There is no model to be saved.")
        #    return
        self.model_text_modified.set(True)

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

        if "." not in basename(save_model_file):
            save_model_file = save_model_file + ".pm"
        # print("save_model_file", save_model_file)

        with open(save_model_file, "w") as f:
            f.write(self.model_text.get(1.0, END))

        if not file:
            self.model_file.set(save_model_file)
            self.status_set("Model saved.")

    def save_property(self, file=False):
        """ Saves obtained temporal properties as a file.

        Args:
            file (bool or str or Path): file to save the property
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

        if "." not in basename(save_property_file):
            save_property_file = save_property_file + ".pctl"
        # print("save_property_file", save_property_file)

        with open(save_property_file, "w") as f:
            f.write(self.property_text.get(1.0, END))

        if not file:
            self.property_file.set(save_property_file)
            self.status_set("Property saved.")

    def generate_data_informed_properties(self):
        """ Generates Data informed property from temporal properties and data. Prints it. """
        print("Checking the inputs.")
        self.check_changes("properties")

        if self.property_file.get() == "":
            messagebox.showwarning("Data informed property generation", "No property file loaded.")
            return False

        if self.data_intervals == []:
            print("Intervals not computed, properties cannot be generated")
            messagebox.showwarning("Data informed property generation", "Compute intervals first.")
            return False

        # general_create_data_informed_properties(prop_file, intervals, output_file=False)
        self.data_informed_property = general_create_data_informed_properties(self.property_file.get(), self.data_intervals, silent=self.silent.get())
        self.data_informed_property_text.configure(state='normal')
        self.data_informed_property_text.delete('1.0', END)
        spam = ""
        for item in self.data_informed_property:
            spam = spam + str(item) + " &\n"
        spam = spam[:-2]
        self.data_informed_property_text.insert('end', spam)

        ## Autosave
        self.save_data_informed_properties(os.path.join(self.tmp_dir, "data_informed_properties.pctl"))
        self.data_informed_property_file.set(os.path.join(self.tmp_dir, "data_informed_properties.pctl"))
        # self.data_informed_property_text.configure(state='disabled')

    def save_data_informed_properties(self, file=False):
        """ Saves computed data informed property as a text file.

        Args:
            file (bool or str or Path): file to save the data_informed_properties
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

        if "." not in basename(save_data_informed_property_file):
            save_data_informed_property_file = save_data_informed_property_file + ".pctl"
        # print("save_property_file", save_property_file)

        with open(save_data_informed_property_file, "w") as f:
            f.write(self.data_informed_property_text.get('1.0', END))

        if not file:
            self.data_informed_property_file.set(save_data_informed_property_file)
            self.status_set("Data informed property saved.")

    ## TODO MAYBE IN THE FUTURE
    def save_mc_output_file(self, file=False):
        """ Saves parsed functions as a pickled file.

        Args:
            file (bool or str or Path): file to save the functions
        """
        print("Saving the functions ...")

        if self.functions == "":
            self.status_set("There are no functions to be saved.")
            return

        if file:
            save_functions_file = file
        else:
            self.status_set("Please select folder to store the functions in.")
            if self.program == "prism":
                save_functions_file = filedialog.asksaveasfilename(initialdir=self.prism_results,
                                                                   title="Functions saving - Select file",
                                                                   filetypes=(("pickle files", "*.p"), ("text files", "*.txt"), ("all files", "*.*")))
            elif self.program == "storm":
                save_functions_file = filedialog.asksaveasfilename(initialdir=self.storm_results,
                                                                   title="Functions saving - Select file",
                                                                   filetypes=(("pickle files", "*.p"), ("text files", "*.txt"), ("all files", "*.*")))
            else:
                self.status_set("Error - Selected program not recognised.")
                save_functions_file = "Error - Selected program not recognised."
            if not self.silent.get():
                print("Saving functions in file: ", save_functions_file)

            if save_functions_file == "":
                self.status_set("No file selected to store the functions.")
                return

        if "." not in basename(save_functions_file):
            save_functions_file = save_functions_file + ".txt"

        with open(save_functions_file, "w") as f:
            for line in self.functions:
                f.write(line)

        if not file:
            self.functions_file.set(save_functions_file)
            self.status_set("Functions saved.")

    @staticmethod
    def scrap_TextBox(where):
        text = where.get('1.0', END).split("\n")
        if isinstance(text, str):
            text = [text]
        # print("text", text)
        scrap = []
        for line in text:
            if line == "":
                continue
            ## Getting rid of last comma
            scrap.append(re.sub(r',\s*$', '', line))
        return scrap

    def save_parsed_functions(self, file=False):
        """ Saves parsed functions as a pickled file.

        Args:
            file (bool or str or Path): file to save the parsed functions
        """
        functions = self.scrap_TextBox(self.functions_parsed_text)

        if functions == []:
            self.status_set("There are no functions to be saved.")
            messagebox.showwarning("Saving functions", "There are no functions to be saved.")
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
                messagebox.showwarning("Save parsed functions",
                                       "Select a program for which you want to save functions.")
                return
            if self.save_as_plain_text.get():
                filetypes = (("text files", "*.txt"), ("pickle files", "*.p"), ("all files", "*.*"))
            else:
                filetypes = (("pickle files", "*.p"), ("text files", "*.txt"), ("all files", "*.*"))

            save_functions_file = filedialog.asksaveasfilename(initialdir=initial_dir,
                                                               title="Functions saving - Select file",
                                                               filetypes=filetypes)
            if save_functions_file == "":
                self.status_set("No file selected to store the parsed functions.")
                return

        if "." not in basename(save_functions_file):
            if self.save_as_plain_text.get():
                save_functions_file = save_functions_file + ".txt"
            else:
                save_functions_file = save_functions_file + ".p"

        if not self.silent.get() and not file:
            print("Saving parsed functions as a file:", save_functions_file)

        self.save_file(functions, save_functions_file)

        if not file:
            self.functions_file.set(save_functions_file)
            self.status_set("Parsed functions saved.")

    def save_data(self, file=False):
        """ Saves data as a pickled file.

        Args:
            file (bool or str or Path):  file to save the data
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

            if self.save_as_plain_text.get():
                filetypes = (("text files", "*.txt"), ("pickle files", "*.p"), ("all files", "*.*"))
            else:
                filetypes = (("pickle files", "*.p"), ("text files", "*.txt"), ("all files", "*.*"))
            save_data_file = filedialog.asksaveasfilename(initialdir=self.data_dir, title="Data saving - Select file",
                                                          filetypes=filetypes)
            if save_data_file == "":
                self.status_set("No file selected to store the data.")
                return

        if "." not in basename(save_data_file):
            if self.save_as_plain_text.get():
                save_data_file = save_data_file + ".txt"
            else:
                save_data_file = save_data_file + ".p"

        if not self.silent.get():
            print("Saving data as a file:", save_data_file)

        self.save_file(self.data, save_data_file)

        if not file:
            self.data_file.set(save_data_file)
            self.status_set("Data saved.")

    def plot_data(self):
        """ Plots the data. """
        print("Plotting the data ...")

        if not self.data:
            messagebox.showwarning("Saving data", "There is no data to be plotted.")
            self.status_set("There is no data to be plot.")
            return
        if self.data_intervals:
            bar_err_plot(self.data, self.data_intervals, titles=["Data indices", "Data values", f"Summary of {self.n_samples_entry.get()} observations.\n Data intervals visualised as error bars."])
        else:
            bar_err_plot(self.data, self.data_intervals, titles=["Data indices", "Data values", f"Summary of {self.n_samples_entry.get()} observations."])

    def save_data_weights(self, file=False):
        """ Saves data weights as a pickled file.

        Args:
            file (bool or str or Path):  file to save the data weights
        """

        data_weights = self.scrap_TextBox(self.data_weights_text)
        ## Converting strings to weights

        if file:
            save_data_weights_file = file
        else:
            print("Saving the data weights ...")

            if not data_weights:
                messagebox.showwarning("Saving data weights", "There are no data weights to be saved.")
                self.status_set("There are no data weights to be saved.")
                return

            self.status_set("Please select folder to store the data weights in.")
            if self.save_as_plain_text.get():
                filetypes = (("text files", "*.txt"), ("pickle files", "*.p"), ("all files", "*.*"))
            else:
                filetypes = (("pickle files", "*.p"), ("text files", "*.txt"), ("all files", "*.*"))
            save_data_weights_file = filedialog.asksaveasfilename(initialdir=self.data_weights_dir,
                                                                  title="Data weights saving - Select file",
                                                                  filetypes=filetypes)
            if save_data_weights_file == "":
                self.status_set("No file selected to store the data weights.")
                return

        if "." not in basename(save_data_weights_file):
            if self.save_as_plain_text.get():
                save_data_weights_file = save_data_weights_file + ".txt"
            else:
                save_data_weights_file = save_data_weights_file + ".p"

        if not self.silent.get():
            print("Saving data weights as a file:", save_data_weights_file)

        self.save_file(data_weights, save_data_weights_file)

        if not file:
            self.data_weights_file.set(save_data_weights_file)
            self.status_set("Data weights saved.")

    def discard_data_weights(self):
        """ Clears the data weights """
        self.data_weights = []
        self.data_weights_text.delete('1.0', END)
        self.data_weights_file.set("")

    def save_data_intervals(self, file=False):
        """ Saves data intervals as a pickled file.

        Args:
            file (bool or str or Path):  file to save the data intervals
        """

        data_intervals = self.scrap_TextBox(self.data_intervals_text)
        ## Converting strings to intervals
        for index, interval in enumerate(data_intervals):
            # print(data_intervals[index])
            # print(type(data_intervals[index]))
            data_intervals[index] = Interval(*parse_numbers(data_intervals[index]))

        if file:
            save_data_intervals_file = file
        else:
            print("Saving the data intervals ...")

            if not data_intervals:
                messagebox.showwarning("Saving data intervals", "There are no data intervals to be saved.")
                self.status_set("There are no data intervals to be saved.")
                return

            self.status_set("Please select folder to store the data intervals in.")
            if self.save_as_plain_text.get():
                filetypes = (("text files", "*.txt"), ("pickle files", "*.p"), ("all files", "*.*"))
            else:
                filetypes = (("pickle files", "*.p"), ("text files", "*.txt"), ("all files", "*.*"))
            save_data_intervals_file = filedialog.asksaveasfilename(initialdir=self.data_intervals_dir, title="Data intervals saving - Select file",
                                                                    filetypes=filetypes)
            if save_data_intervals_file == "":
                self.status_set("No file selected to store the data intervals.")
                return

        if "." not in basename(save_data_intervals_file):
            if self.save_as_plain_text.get():
                save_data_intervals_file = save_data_intervals_file + ".txt"
            else:
                save_data_intervals_file = save_data_intervals_file + ".p"

        if not self.silent.get():
            print("Saving data intervals as a file:", save_data_intervals_file)

        self.save_file(data_intervals, save_data_intervals_file)

        if not file:
            self.data_intervals_file.set(save_data_intervals_file)
            self.status_set("Data intervals saved.")

    def save_constraints(self, file=False):
        """ Saves constraints as a pickled file.

        Args:
            file (bool or str or Path):  file to save the constraints
        """
        constraints = self.scrap_TextBox(self.constraints_text)

        if file:
            save_constraints_file = file
        else:
            print("Saving the constraints ...")
            if constraints == "":
                self.status_set("There is no constraints to be saved.")
                return

            self.status_set("Please select folder to store the constraints in.")
            if self.save_as_plain_text.get():
                filetypes = (("text files", "*.txt"), ("pickle files", "*.p"), ("all files", "*.*"))
            else:
                filetypes = (("pickle files", "*.p"), ("text files", "*.txt"), ("all files", "*.*"))
            save_constraints_file = filedialog.asksaveasfilename(initialdir=self.constraints_dir, title="constraints saving - Select file",
                                                                 filetypes=filetypes)
            if save_constraints_file == "":
                self.status_set("No file selected to store the constraints.")
                return

        if "." not in basename(save_constraints_file):
            if self.save_as_plain_text.get():
                save_constraints_file = save_constraints_file + ".txt"
            else:
                save_constraints_file = save_constraints_file + ".p"

        if not self.silent.get():
            print("Saving constraints as a file:", save_constraints_file)

        self.save_file(constraints, save_constraints_file)

        if not file:
            self.constraints_file.set(save_constraints_file)
            self.status_set("constraints saved.")

    def save_space(self, file=False):
        """ Saves space as a pickled file.

        Args:
            file (bool or str or Path):  file to save the space
        """
        if file:
            save_space_file = file
        else:
            print("Saving the space ...")
            if self.space == "":
                self.status_set("There is no space to be saved.")
                messagebox.showwarning("Saving Space", "There is no space to be saved.")
                return
            self.status_set("Please select folder to store the space in.")
            save_space_file = filedialog.asksaveasfilename(initialdir=self.refinement_results, title="Space saving - Select file",
                                                           filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
            if save_space_file == "":
                self.status_set("No file selected to store the space in.")
                return

        if "." not in basename(save_space_file):
            save_space_file = save_space_file + ".p"

        if not self.silent.get():
            print("Saving space as a file:", save_space_file)

        assert isinstance(self.space, space.RefinedSpace)
        spacee = copy.copy(self.space)
        spacee.rectangles_unknown = spacee.get_flat_white()
        self.save_file(spacee, save_space_file)
        del spacee

        if not file:
            self.space_file.set(save_space_file)
            self.status_set("Space saved.")

    def save_mh_results(self, file=False):
        """ Saves Metropolis Hastings results a pickled file.

        Args:
            file (bool or str or Path):  file to save Metropolis Hastings results
        """

        if file:
            save_mh_results_file = file
        else:
            print("Saving Metropolis Hastings results ...")
            if self.mh_results == "":
                self.status_set("There is no Metropolis Hastings results to be saved.")
                return
            self.status_set("Please select folder to store Metropolis Hastings results in.")
            save_mh_results_file = filedialog.asksaveasfilename(initialdir=self.mh_results_dir, title="Metropolis Hastings results saving - Select file",
                                                                filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
            if save_mh_results_file == "":
                self.status_set("No file selected to store Metropolis Hastings results in.")
                return

        if "." not in basename(save_mh_results_file):
            save_mh_results_file = save_mh_results_file + ".p"

        if not self.silent.get():
            print("Saving Metropolis Hastings results as a file:", save_mh_results_file)

        self.save_file(self.mh_results, save_mh_results_file)
        # pickle_dump(self.mh_results, os.path.join(self.mh_results_dir, f"mh_results_{strftime('%d-%b-%Y-%H-%M-%S', localtime())}.p"))

        if not file:
            self.hastings_file.set(save_mh_results_file)
            self.status_set("Metropolis Hastings results saved.")

    ## ANALYSIS
    def synth_params(self, refinement=False):
        """ Computes functions from model and temporal properties. Saves output as a text file.

        Args:
            refinement (bool): if True refine space using data-informed properties
        """
        if refinement:
            method = "Space partitioning"
        else:
            method = "Parametric model checking"

        print("Checking the inputs.")
        self.status_set(f"{method} - checking inputs")
        self.check_changes("model")
        self.check_changes("properties")
        proceed = True

        print(f"{method} ...")
        self.status_set(f"{method}.")
        if self.functions_changed:
            proceed = askyesno(f"{method}", f"{method} will overwrite current results. Do you want to proceed?")

        if not proceed:
            return

        if self.model_changed:
            messagebox.showwarning(f"{method}", f"The model for {method} has changed in the mean time, please consider that.")

        if (self.property_changed and not refinement) or (self.data_informed_property_changed and refinement):
            messagebox.showwarning(f"{method}", f"The properties for {method} have changed in the mean time, please consider that.")

        ## If model file not selected load model
        if self.model_file.get() == "":
            self.status_set(f"Load model for {method}")
            self.load_model()

        ## If property file not selected load property
        if refinement:
            if self.data_informed_property_file.get() == "":
                self.status_set(f"Load property for {method}")
                self.load_data_informed_properties()
        else:
            if self.property_file.get() == "":
                self.status_set(f"Load property for {method}")
                self.load_property()

        ## Get model parameters, reset param domains and load new
        self.constants, self.parameters = parse_params_from_model(self.model_file.get(), silent=True)
        self.parameter_domains = []
        self.validate_parameters(where="model", intervals=True, force=True)
        # self.load_param_intervals_from_window()

        print("param domains", self.parameter_domains)
        print()

        if refinement:
            property_file = self.data_informed_property_file.get()
        else:
            property_file = self.property_file.get()

        output_file = filedialog.asksaveasfilename(initialdir=self.prism_results if self.program.get().lower() == "prism" else self.storm_results,
                                                   title=f"{method}- Select file to save output",
                                                   filetypes=(("text files", "*.txt"), ("all files", "*.*")))
        try:
            self.cursor_toggle_busy(True)
            if refinement:
                if self.program.get().lower() == "prism":
                    self.status_set("Space refinement using PRISM is running ...")
                elif self.program.get().lower() == "storm":
                    self.status_set("Parameter-lifting using Storm is running ...")
                self.coverage = round(float(self.coverage_entry.get()), 13)
            else:
                self.status_set(f"Parametric model checking using {self.program.get()} is running ...")
                self.coverage = False

            ## CALL PRISM/STORM
            if self.program.get().lower() == "prism":
                if output_file == ():
                    output_file = str(os.path.join(Path(self.prism_results), str(Path(self.model_file.get()).stem) + "_" + str(Path(property_file).stem) + ".txt"))

                call_prism_files(self.model_file.get(), [], param_intervals=self.parameter_domains, seq=False, no_prob_checks=False,
                                 memory="", model_path="", properties_path=self.property_dir,
                                 property_file=property_file, output_path=output_file,
                                 gui=show_message, silent=self.silent.get(), coverage=self.coverage)
                self.mc_result_file.set(output_file)
                self.status_set(f"{method} using PRISM finished. Output here: {output_file}")
                if refinement:
                    self.load_refinement_output_file(output_file, reset_param_and_intervals=False, called_directly=False)
                else:
                    self.load_functions_file(output_file, reset_param_and_intervals=False)

            elif self.program.get().lower() == "storm":
                if output_file == ():
                    output_file = str(os.path.join(Path(self.storm_results), str(Path(self.model_file.get()).stem) + "_" + str(Path(property_file).stem)))

                try:
                    os.remove(output_file)
                except FileNotFoundError:
                    pass

                if refinement:
                    call_storm(model_file=self.model_file.get(), params=self.parameters, param_intervals=self.parameter_domains,
                               property_file=property_file, storm_output_file=output_file, coverage=self.coverage, time=True,
                               silent=self.silent.get())
                else:
                    call_storm(model_file=self.model_file.get(), params=[], param_intervals=[],
                               property_file=property_file, storm_output_file=output_file, time=True,
                               silent=self.silent.get())
                self.mc_result_file.set(output_file)
                if refinement:
                    self.status_set(f"Parameter lifting using Storm finished. Output here: {output_file}")
                else:
                    self.status_set(f"Parametric model checking using Storm finished. Output here: {output_file}")
                    # self.status_set("Command to run the parameter synthesis saved here: {}", self.functions_file.get())
                if refinement:
                    self.load_refinement_output_file(output_file, reset_param_and_intervals=False, called_directly=False)
                else:
                    self.load_functions_file(output_file, reset_param_and_intervals=False)
            else:
                ## Show window to inform to select the program
                self.status_set(f"Program for {method} not selected")
                messagebox.showwarning(f"{method}", f"Select a program for {method} first.")
                return
        finally:
            try:
                self.cursor_toggle_busy(False)
            except TclError:
                return

        self.model_changed = False
        if refinement:
            self.data_informed_property_changed = False
        else:
            self.property_changed = False
        ## Resetting parsed intervals
        # self.parameters = []
        # self.parameter_domains = []

        # self.save_parsed_functions(os.path.join(self.tmp_dir, "parsed_functions"))
        self.cursor_toggle_busy(False)

    def external_refine_PRISM(self):
        """ Calls PRISM to refine space using data-informed properties. """
        spam = self.program.get()
        self.program.set("prism")
        self.synth_params(refinement=True)
        self.program.set(spam)

    def external_refine_Storm(self):
        """ Calls Storm to refine space using data-informed properties. """
        spam = self.program.get()
        self.program.set("storm")
        self.synth_params(refinement=True)
        self.program.set(spam)

    def sample_fun(self):
        """ Samples functions. Prints the result. """
        print("Checking the inputs.")
        self.check_changes("functions")

        print("Sampling functions ...")
        self.status_set("Sampling functions. - checking inputs")
        if self.fun_sample_size_entry.get() == "":
            messagebox.showwarning("Sampling functions", "Choose grid size, number of samples per dimension.")
            return
        if self.functions == "":
            messagebox.showwarning("Sampling functions", "Load the functions first, please")
            return

        self.status_set("Sampling functions.")
        self.validate_parameters(where=self.functions, force=True)

        try:
            self.cursor_toggle_busy(True)
            self.status_set("Sampling functions ...")
            assert isinstance(self.parameters, list)
            self.sampled_functions = sample_list_funs(self.functions, int(self.fun_sample_size_entry.get()),
                                                      parameters=self.parameters, intervals=self.parameter_domains,
                                                      debug=self.debug.get(), silent=self.silent.get())
        finally:
            try:
                self.cursor_toggle_busy(False)
            except TclError:
                return
        self.sampled_functions_text.configure(state='normal')
        self.sampled_functions_text.delete('1.0', END)
        self.sampled_functions_text.insert('1.0', " function index, [parameter values], function value: \n")
        spam = ""
        assert isinstance(self.sampled_functions, Iterable)
        for item in self.sampled_functions:
            spam = spam + str(item[0]+1) + ", ["
            for index in range(1, len(item)-1):
                spam = spam + str(item[index]) + ", "
            spam = spam[:-2]
            spam = spam + "], " + str(item[-1]) + ",\n"
        self.sampled_functions_text.insert('2.0', spam[:-2])
        # self.sampled_functions_text.configure(state='disabled')
        self.status_set("Sampling functions finished.")

    def show_funs_in_single_point(self):
        """ Plots functions in a given point. """
        print("Checking the inputs.")
        self.check_changes("functions")
        self.check_changes("data")
        self.check_changes("data_intervals")

        print("Plotting functions in a given point ...")
        self.status_set("Plotting functions in a given point.")

        if self.functions == "":
            pass  ## TODO

        if self.functions == "":
            messagebox.showwarning("Plotting functions in a given point.", "Load the functions first, please.")
            return

        ## Disable overwriting the plot by show_funs_in_all_points
        if self.page3_figure_in_use.get():
            if not askyesno("Plotting functions in a given point",
                            "The result plot is currently in use. Do you want override?"):
                return
        self.page3_figure_in_use.set("1")

        self.validate_parameters(where=self.functions, intervals=False)

        self.status_set("Choosing parameters value:")
        assert isinstance(self.parameters, list)
        self.create_window_to_load_param_point(parameters=self.parameters)

        self.reinitialise_plot()

        ## Getting the plot values instead of the plot itself

        #     self.initialise_plot(what=self.page3_figure, where=self.page3_plotframe)
        # else:
        #     pyplt.close()
        #     self.page3_figure = pyplt.figure()
        #     self.page3_a = self.page3_figure.add_subplot(111)
        # print("self.parameter_values", self.parameter_values)
        assert isinstance(self.functions, list)
        assert isinstance(self.data, list)
        assert isinstance(self.data_intervals, list)
        spam, egg = eval_and_show(self.functions, self.parameter_point, parameters=self.parameters,
                                  data=self.data, data_intervals=self.data_intervals,
                                  debug=self.debug.get(), where=[self.page3_figure, self.page3_a])

        if spam is None:
            messagebox.showinfo("Plots functions in a given point.", egg)
        else:
            self.page3_figure = spam
            self.page3_a = egg
            self.initialise_plot3(what=self.page3_figure)
            self.page3_a.autoscale(enable=False)
            self.page3_figure.tight_layout()  ## By huypn
            self.page3_figure.canvas.draw()
            self.page3_figure.canvas.flush_events()

            ## Autosave figure
            if self.save.get():
                self.save_functions_plot("Single_point_functions")

        if not self.silent.get():
            print(f"Using point", self.parameter_point)
        self.status_set("Sampling functions done.")

    def save_functions_plot(self, plot_type):
        """ Saves the plot of visualised functions

        Args:
            plot_type (str): plot type
        """
        time_stamp = str(strftime("%d-%b-%Y-%H-%M-%S", localtime())) + ".png"

        try:
            self.page3_figure.savefig(os.path.join(self.figures_dir, "tmp", f"{plot_type}_{time_stamp}"), bbox_inches='tight')
            print("Figure stored here: ", os.path.join(self.figures_dir, "tmp", f"{plot_type}_{time_stamp}"))
        except:
            print("This figure could not be saved.")
        with open(os.path.join(self.figures_dir, "tmp", "figure_to_title.txt"), "a+") as f:
            f.write(f"{plot_type}_{time_stamp} :\n")
            f.write(f"      functions: {self.functions_file.get()}\n")
            if self.data:
                f.write(f"      data: {self.data_file.get()}\n")

    def show_funs_in_all_points(self):
        """ Shows sampled functions in all sampled points. """
        print("Checking the inputs.")
        self.check_changes("functions")
        self.check_changes("data")
        self.check_changes("data_intervals")

        print("Plotting sampled functions ...")
        self.status_set("Plotting sampled functions.")

        if self.functions == "":
            messagebox.showwarning("Sampling functions", "Load the functions first, please")
            return

        if self.fun_sample_size_entry.get() == "":
            messagebox.showwarning("Sampling functions", "Choose grid size, number of samples per dimension.")
            return

        if self.page3_figure_in_use.get():
            if not askyesno("Show all sampled points", "The result plot is currently in use. Do you want override?"):
                return

        self.page3_figure_in_use.set("2")

        self.validate_parameters(where=self.functions)

        ## To be used to wait until the button is pressed
        self.button_pressed.set(False)
        self.Next_sample_button.config(state="normal")
        self.reinitialise_plot(set_onclick=True)

        assert isinstance(self.parameters, list)

        for parameter_point in get_param_values(self.parameters, int(self.fun_sample_size_entry.get()), intervals=self.parameter_domains, debug=self.debug.get()):
            if self.page3_figure_in_use.get() != "2":
                return

            # print("parameter_point", parameter_point)
            self.page3_a.cla()
            assert isinstance(self.functions, list)
            assert isinstance(self.parameters, list)
            assert isinstance(self.data, list)
            assert isinstance(self.data_intervals, list)
            spam, egg = eval_and_show(self.functions, parameter_point, parameters=self.parameters,
                                      data=self.data, data_intervals=self.data_intervals,
                                      debug=self.debug.get(), where=[self.page3_figure, self.page3_a])

            if spam is None:
                messagebox.showinfo("Plots functions in a given point.", egg)
            else:
                spam.tight_layout()
                self.page3_figure = spam
                self.page3_a = egg

                self.initialise_plot3(what=self.page3_figure)
                # self.page3_a.autoscale(enable=False)
                # self.page3_figure.canvas.draw()
                # self.page3_figure.canvas.flush_events()

            self.Next_sample_button.wait_variable(self.button_pressed)
        # self.Next_sample_button.config(state="disabled")
        self.status_set("Plotting sampled functions finished.")

    def show_heatmap(self):
        """ Shows heatmap - sampling of a function in all sampled points. """
        print("Checking the inputs.")
        self.check_changes("functions")
        if self.show_data_in_heatmap.get():
            self.check_changes("data")

        print("Plotting heatmap of functions ...")
        self.status_set("Plotting heatmap of functions.")

        if self.functions == "":
            messagebox.showwarning("Plot heatmap", "Load the functions first, please")
            return

        if self.fun_sample_size_entry.get() == "":
            messagebox.showwarning("Plot heatmap", "Choose grid size, number of samples per dimension.")
            return

        if self.page3_figure_in_use.get():
            if not askyesno("Plot heatmap", "The result plot is currently in use. Do you want override?"):
                return

        self.validate_parameters(where=self.functions)

        if len(self.parameters) > 2:
            messagebox.showerror("Plot heatmap", f"Could not show this 2D heatmap. Parsed function(s) contain {len(self.parameters)} parameter(s), expected 1 or 2.")
            return

        self.page3_figure_in_use.set("3")
        ## To be used to wait until the button is pressed
        self.button_pressed.set(False)
        self.Next_sample_button.config(state="normal")

        self.reinitialise_plot(set_onclick=True)

        i = 0
        for index, function in enumerate(self.functions):
            if self.page3_figure_in_use.get() != "3":
                return
            i = i + 1
            assert isinstance(self.parameters, list)
            if self.show_data_in_heatmap.get():
                self.page3_figure = heatmap(f"abs({function} - {self.data[index]})", self.parameter_domains,
                                            [int(self.fun_sample_size_entry.get()),
                                             int(self.fun_sample_size_entry.get())],
                                            posttitle=f"|Function - data point|, #{i} : |{function} - {self.data[index]}|", where=True,
                                            parameters=self.parameters, verbose=self.debug.get())
            else:
                self.page3_figure = heatmap(function, self.parameter_domains,
                                            [int(self.fun_sample_size_entry.get()), int(self.fun_sample_size_entry.get())],
                                            posttitle=f"Function number {i}: {function}", where=True,
                                            parameters=self.parameters, verbose=self.debug.get())
            self.initialise_plot3(what=self.page3_figure)

            ## Autosave figure
            ## TODO as this feature creates so far unfixable bugs it will be commented
            # if self.save.get():
            #     self.save_functions_plot(f"Heatmap_function_{i}_of_{len(self.functions)}_")

            self.Next_sample_button.wait_variable(self.button_pressed)
        # self.Next_sample_button.config(state="disabled")
        # self.page3_figure_locked.set(False)
        # self.update()
        self.status_set("Plotting sampled functions finished.")

    def optimize(self):
        """ Search for parameter values minimizing the distance of function to data. """
        print("Checking the inputs.")
        self.check_changes("functions")
        self.check_changes("data")
        self.check_changes("data_weights")

        print("Optimizing the distance between functions and data ...")
        self.status_set("Optimizing the distance between functions and data.")

        if self.functions == "":
            messagebox.showwarning("Optimize functions", "Load the functions first, please")
            return

        if self.data == []:
            messagebox.showwarning("Optimize functions", "Load the data first, please")
            return

        self.validate_parameters(where=self.functions)

        print("Parameters", self.parameters)
        print("Parameter_domains", self.parameter_domains)

        if len(self.functions) != len(self.data):
            messagebox.showwarning("Optimize functions", f"Number of functions ({len(self.functions)}) is not equal to the number of data points ({len(self.data)})")

        try:
            self.cursor_toggle_busy(True)

            ## Progress Bar
            ## TODO - tweak - update this to actually show the progress
            self.new_window = Toplevel(self)
            Label(self.new_window, text="Optimisation progress:", anchor=W, justify=LEFT).pack()
            pb_hd = ttk.Progressbar(self.new_window, orient='horizontal', mode='indeterminate')
            pb_hd.pack(expand=True, fill=BOTH, side=TOP)
            pb_hd.start(50)
            self.update()

            assert isinstance(self.functions, list)
            assert isinstance(self.parameters, list)
            assert isinstance(self.data, list)
            assert isinstance(self.data_weights, list)

            start_time = time()
            result = optimize(self.functions, self.parameters, self.parameter_domains, self.data, weights=self.data_weights, debug=self.debug.get())
            print(colored(f"Optimisation took {time() - start_time} seconds", "yellow"))
        except Exception as error:
            messagebox.showerror("Optimize", f"Error occurred during Optimization: {error}")
            raise error
            # return
        finally:
            try:
                self.cursor_toggle_busy(False)
                self.new_window.destroy()
            except TclError:
                return

        self.optimised_param_point = result[0]
        self.optimised_function_value = result[1]
        self.optimised_distance = result[2]

        window = Toplevel(self)
        window.title('Result of optimisation')
        window.state('normal')
        width = max(len(str(result[0])), len(str(result[1])), len(str(result[2])))

        window.minsize(400, width+20)
        window.resizable(False, False)

        i = 1
        Label(window, text=f"Parameter point: ").grid(row=i); i += 1
        Label(window, text=f"Parameter domains: ").grid(row=i); i += 1
        Label(window, text=f"Function values: ").grid(row=i); i += 1
        Label(window, text=f"Data point: ").grid(row=i); i += 1

        if self.data_weights:
            Label(window, text=f"Weights: ").grid(row=i)
            i = i + 1
        Label(window, text=f"Distance: ").grid(row=i)

        i = 1
        var = StringVar()
        var.set(str(result[0]))
        ent = Entry(window, state='readonly', textvariable=var, width=width, relief='flat', readonlybackground='white',
                    fg='black')
        ent.grid(row=i, column=1); i += 1

        var = StringVar()
        var.set(self.parameter_domains)
        ent = Entry(window, state='readonly', textvariable=var, width=width, relief='flat', readonlybackground='white', fg='black')
        ent.grid(row=i, column=1); i += 1

        var = StringVar()
        var.set(str(result[1]))
        ent = Entry(window, state='readonly', textvariable=var, width=width, relief='flat', readonlybackground='white', fg='black')
        ent.grid(row=i, column=1); i += 1

        var = StringVar()
        var.set(str(self.data))
        ent = Entry(window, state='readonly', textvariable=var, width=width, relief='flat', readonlybackground='white', fg='black')
        ent.grid(row=i, column=1); i += 1

        if self.data_weights:
            var = StringVar()
            var.set(str(self.data_weights))
            ent = Entry(window, state='readonly', textvariable=var, width=width, relief='flat', readonlybackground='white', fg='black')
            ent.grid(row=i, column=1); i += 1

        var = StringVar()
        var.set(str(result[2]))
        ent = Entry(window, state='readonly', textvariable=var, width=width, relief='flat', readonlybackground='white', fg='black')
        ent.grid(row=i, column=1); i += 1

        save_optimisation_button = Button(window, text="Save Result", command=self.save_optimisation_result)
        save_optimisation_button.grid(row=i, column=1)

        ## Autosave
        self.save_optimisation_result(os.path.join(self.tmp_dir, "optimisation_results.txt"))

        print("Parameter point: ", self.optimised_param_point)
        print("Parameter domains: ", self.parameter_domains)
        print("Function values: ", self.optimised_function_value)
        print("Data points: ", self.data)
        if self.data_weights:
            print("Weights: ", self.data_weights)
        print("Distance: ", self.optimised_distance)

    def save_optimisation_result(self, file=False):
        """ Stores optimisation results as a file

        Args:
            file (string or False):  file to store the optimisation results
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

        if "." not in basename(save_opt_result_file):
            save_opt_result_file = save_opt_result_file + ".txt"

        with open(save_opt_result_file, "w") as f:
            f.write(f"Parameters: {self.parameters} \n")
            f.write(f"Parameter domains: {self.parameter_domains} \n")
            f.write(f"Parameter point: {self.optimised_param_point} \n")
            f.write(f"Function values: {self.optimised_function_value} \n")
            f.write(f"Data values: {self.data} \n")
            if self.data_weights:
                f.write(f"Weights: {self.data_weights} \n")
            f.write(f"Distance: {self.optimised_distance} \n")

    ## First, it asks whether it is changed, then selects (text, file, text) accordingly
    def check_changes(self, what):
        """ Checks whether a change occurred and it is necessary to reload

        Args:
            what (string): "model", "properties", "parsed_functions", "data"
            "data_intervals", or "constraints" choosing what to check
        """
        ## SWITCH contains quadruples:  (modifiedflag, file path, text, save_function, load_function)
        switch = {"model": (self.model_text_modified, self.model_file, "model", self.save_model, self.load_model),
                  "properties": (self.properties_text_modified, self.property_file, "properties", self.save_property, self.load_property),
                  "functions": (self.parsed_functions_text_modified, self.functions_file, "functions", self.save_parsed_functions, self.load_parsed_functions),
                  "data": (self.data_text_modified, self.data_file, "data", self.save_data, self.load_data),
                  "data_weights": (self.data_weights_text_modified, self.data_weights_file, "data_weights", self.save_data_weights, self.load_data_weights),
                  "data_intervals": (self.data_intervals_text_modified, self.data_intervals_file, "data_intervals", self.save_data_intervals, self.load_data_intervals),
                  "constraints": (self.constraints_text_modified, self.constraints_file, "constraints", self.save_constraints, self.load_constraints),
                  }
        option = switch[what]
        ## LEGEND
        # modified_flag = option[0].get()
        # file_path = option[1]
        # text = option[2]
        # save_function = option[3]
        # load_function = option[4]

        ## Old check:  len(self.model_text.get('1.0', END)) > 1 and
        ## If modified
        if option[0].get():
            ## If file set
            if option[1].get() != "":
                if not askyesno(f"In the meanwhile the {option[2]} was changed", f"Do you wanna apply these changes?"):
                    option[0].set(False)   ## Set as not changed
                    return
                if askyesno(f"Saving {option[2]}", f"Do you want to overwrite the existing {option[2]} file?"):
                    option[3](option[1].get())  ## Save the thing as a file - overwrite
                else:
                    option[3]()  ## Save the thing as a file - new file
                option[4](option[1].get())  ## Load that thing as a file
                option[0].set(False)  ## Set as not changed

            else:
                if not askyesno(f"In the meanwhile the {option[2]} was changed", f"Do you wanna save these changes as a {option[2]}?"):
                    option[0].set(False)  ## Set as not changed
                    return
                option[3]()           ## Save the thing as a file - new file
                option[4](option[1].get())  ## Load the thing as a file
                option[0].set(False)  ## Set as not changed

    ## Old implementation
    # def refresh_properties(self):
    #     if self.property_text_modified.get():
    #         if self.property_file.get() is not "":
    #             if not askyesno("Editing properties", "Do you wanna apply these changes?"):
    #                 self.property_text_modified.set(False)
    #                 return
    #             self.save_property(self.property_file.get())
    #             self.load_property(self.property_file.get())
    #             self.property_text_modified.set(False)
    #         else:
    #             if not askyesno("Editing properties", "Do you wanna save these changes as a file?"):
    #                 self.property_text_modified.set(False)
    #                 return
    #             self.save_property()
    #             self.load_property(self.property_file.get())
    #             self.property_text_modified.set(False)
    #
    # def refresh_parsed_functions(self, event):
    #     if self.parsed_functions_text_modified.get():
    #         if self.parsed_functions_file.get() is not "":
    #             if not askyesno("Editing functions", "Do you wanna apply these changes?"):
    #                 return
    #             self.save_parsed_functionsy(self.parsed_functions_file.get())
    #             self.load_parsed_functionsy(self.parsed_functions_file.get())
    #         else:
    #             if not askyesno("Editing functions", "Do you wanna save these changes as a file?"):
    #                 return
    #             self.save_parsed_functions()
    #             self.load_parsed_functions(self.parsed_functions_file.get())
    #
    # def refresh_data(self, event):
    #     if self.data_text_modified.get():
    #         if self.data_file.get() is not "":
    #             if not askyesno("Editing data", "Do you wanna apply these changes?"):
    #                 return
    #             self.save_data(self.data_file.get())
    #             self.load_data(self.data_file.get())
    #         else:
    #             if not askyesno("Editing data", "Do you wanna save these changes as a data?"):
    #                 return
    #             self.save_data()
    #             self.load_data(self.data_file.get())
    #
    # def refresh_data_intervals(self, event):
    #     if self.data_intervals_text_modified.get():
    #         if self.data_intervals_file.get() is not "":
    #             if not askyesno("Editing data_intervals", "Do you wanna apply these changes?"):
    #                 return
    #             self.save_data_intervals(self.data_intervals_file.get())
    #             self.load_data_intervals(self.data_intervals_file.get())
    #         else:
    #             if not askyesno("Editing data_intervals", "Do you wanna save these changes as a data_intervals?"):
    #                 return
    #             self.save_data_intervals()
    #             self.load_data_intervals(self.data_intervals_file.get())
    #
    # def refresh_constraints(self, event):
    #     if self.constraints_text_modified.get():
    #         if self.constraints_file.get() is not "":
    #             if not askyesno("Editing constraints", "Do you wanna apply these changes?"):
    #                 return
    #             self.save_constraints(self.constraints_file.get())
    #             self.load_constraints(self.constraints_file.get())
    #         else:
    #             if not askyesno("Editing constraints", "Do you wanna save these changes as a constraints?"):
    #                 return
    #             self.save_constraints()
    #             self.load_constraints(self.constraints_file.get())

    def compute_data_intervals(self):
        """ Creates intervals from data. """
        print("Checking the inputs.")
        self.check_changes("data")

        print("Creating intervals ...")
        self.status_set("Create interval - checking inputs")
        if self.confidence_entry.get() == "":
            messagebox.showwarning("Creating intervals",
                                   "Choose C, confidence level, before creating intervals.")
            return

        if self.n_samples_entry.get() == "":
            messagebox.showwarning("Creating intervals",
                                   "Choose n_samples, number of experimental samples before creating intervals")
            return

        ## If data file not selected load data
        if self.data_file.get() == "":
            self.load_data()
        # print("self.data_file.get()", self.data_file.get())

        ## Refresh the data from the window
        self.parse_data_from_window()

        self.status_set("Intervals are being created ...")
        assert isinstance(self.data, list)

        method = self.interval_method_entry.get()
        ## ('CLT', 'Rule of three', 'Agresti-Coull', 'Wilson', 'Clopper_Pearson', 'Jeffreys')
        self.data_intervals = [create_proportions_interval(float(self.confidence_entry.get()), int(self.n_samples_entry.get()), data_point, method) for data_point in self.data]

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
        self.save_data_intervals(os.path.join(self.tmp_dir, "data_intervals.p"))
        self.data_intervals_file.set(os.path.join(self.tmp_dir, "data_intervals.p"))
        self.status_set("Intervals created.")

    def sample_space(self):
        """ Samples (Parameter) Space. Plots the results. """
        print("Checking the inputs.")
        self.check_changes("constraints")

        print("Sampling space ...")
        self.status_set("Space sampling - checking inputs")
        ## Getting values from entry boxes
        self.sample_size = int(self.sample_size_entry.get())

        ## Checking if all entries filled
        if self.sample_size == "":
            messagebox.showwarning("Sample space", "Choose grid size, number of samples before space sampling.")
            return

        if self.constraints == "":
            messagebox.showwarning("Sample space", "Load or calculate constraints before space sampling.")
            return

        ## Check space
        if not self.validate_space("Sample Space"):
            return

        ## CHECK MEMORY OVERFLOW
        ## 10000000 = 87.4% * 31.3
        ## 1 144 164 is 100% of 31.3GB
        ## 36554 is 100% per 1GB
        mem = virtual_memory()
        if self.sample_size**len(self.parameters) > 36554 * mem.total/1024/1024/1024:
            if not messagebox.askyesno("Sampling functions", "Estimated amount memory needed to sample space is more than this machine possess. Do you want to continue anyway?"):
                return

        self.store_unsat_samples = True
        ## SUGGEST TO NOT TO STORE UNSAT SAMPLES
        if self.sample_size**len(self.parameters) > 36554:
            if messagebox.askyesno("Sampling functions", f"Estimated amount memory needed to sample space is {round(len(self.parameters)**self.sample_size/36554, 1)}GB. Do you want to omit storing unsat points?"):
                self.store_unsat_samples = False

        self.status_set("Space sampling is running ...")
        assert isinstance(self.space, space.RefinedSpace)
        if self.debug.get():
            print("space parameters: ", self.space.params)
            print("constraints: ", self.constraints)
            print("grid size: ", self.sample_size)

        try:
            self.cursor_toggle_busy(True)

            ## Progress Bar
            self.new_window = Toplevel(self)
            Label(self.new_window, text="Sampling progress:", anchor=W, justify=LEFT).pack()
            Label(self.new_window, textvar=self.progress, anchor=W, justify=LEFT).pack()
            self.progress_bar = Progressbar(self.new_window, orient=HORIZONTAL, length=100, mode='determinate')
            self.progress_bar.pack()
            self.update()

            ## This progress is passed as whole to update the thing inside the called function
            assert isinstance(self.constraints, list)
            self.space.grid_sample(self.constraints, self.sample_size, silent=self.silent.get(), save=False,
                                   progress=self.update_progress_bar if self.show_progress else False,
                                   save_memory=not self.store_unsat_samples)
        finally:
            try:
                self.new_window.destroy()
                del self.new_window
                self.cursor_toggle_busy(False)
                self.progress.set("0%")
            except TclError:
                return

        self.print_space()

        if self.show_quantitative:
            self.clear_space()

        self.show_refinement = True
        self.show_samples = True
        self.show_space(show_refinement=True, show_samples=True, show_true_point=self.show_space_true_point,
                        show_opt_point=self.show_opt_point.get(), prefer_unsafe=self.show_red_in_multidim_refinement.get())

        self.show_quantitative = False

        ## Autosave figure
        if self.save.get():
            time_stamp = str(strftime("%d-%b-%Y-%H-%M-%S", localtime())) + ".png"
            self.page6_figure.savefig(os.path.join(self.refinement_results, "tmp", f"Space_sampling_{time_stamp}"), bbox_inches='tight')
            print("Figure stored here: ", os.path.join(self.refinement_results, "tmp", f"Space_sampling_{time_stamp}"))
            with open(os.path.join(self.refinement_results, "tmp", "figure_to_title.txt"), "a+") as f:
                f.write(f"Space_sampling_{time_stamp} :\n")
                f.write(f"      grid_size: {self.sample_size}\n")
                f.write(f"      constraints: {self.constraints_file.get()}\n")

        self.space_changed = False
        self.constraints_changed = False

        ## Autosave
        self.save_space(os.path.join(self.tmp_dir, "space.p"))

        self.status_set("Space sampling finished.")

    def sample_space_degree(self):
        """ Samples (Parameter) Space adn gives quantitative result in state space. Plots the results. """
        print("Checking the inputs.")
        self.check_changes("constraints")

        print("Quantitative space sampling ...")
        self.status_set("Quantitative space sampling  - checking inputs")
        ## Getting values from entry boxes
        self.sample_size = int(self.sample_size_entry.get())

        ## Checking if all entries filled
        if self.sample_size == "":
            messagebox.showwarning("Quantitative space sampling ", "Choose grid size, number of samples before space sampling.")
            return

        if self.constraints == "":
            messagebox.showwarning("Quantitative space sampling ", "Load or calculate constraints before space sampling.")
            return

        ## Check space
        if not self.validate_space("Quantitative space sampling "):
            return

        self.status_set("Quantitative space sampling  is running ...")
        if not self.silent.get():
            assert isinstance(self.space, space.RefinedSpace)
            print("space parameters: ", self.space.params)
            print("constraints: ", self.constraints)
            print("grid size: ", self.sample_size)

        try:
            self.cursor_toggle_busy(True)

            ## Progress Bar
            self.new_window = Toplevel(self)
            Label(self.new_window, text="Quantitative space sampling  progress:", anchor=W, justify=LEFT).pack()
            Label(self.new_window, textvar=self.progress, anchor=W, justify=LEFT).pack()
            self.progress_bar = Progressbar(self.new_window, orient=HORIZONTAL, length=100, mode='determinate')
            self.progress_bar.pack()
            self.update()

            ## This progress is passed as whole to update the thing inside the called function
            assert isinstance(self.constraints, list)
            self.clear_space()
            self.space.grid_sample(self.constraints, self.sample_size, silent=self.silent.get(), save=False,
                                   progress=self.update_progress_bar if self.show_progress else False,
                                   quantitative=True, save_memory=True)
        finally:
            try:
                self.new_window.destroy()
                del self.new_window
                self.cursor_toggle_busy(False)
                self.progress.set("0%")
            except TclError:
                return

        self.print_space()

        self.show_space(show_refinement=False, show_samples=False, show_true_point=self.show_space_true_point,
                        show_opt_point=self.show_opt_point.get(), prefer_unsafe=self.show_red_in_multidim_refinement.get(),
                        quantitative=True)

        ## Autosave figure
        if self.save.get():
            time_stamp = str(strftime("%d-%b-%Y-%H-%M-%S", localtime())) + ".png"
            self.page6_figure.savefig(os.path.join(self.refinement_results, f"Space_sampling_{time_stamp}"), bbox_inches='tight')
            print("Figure stored here: ", os.path.join(self.refinement_results, f"Space_sampling_{time_stamp}"))
            with open(os.path.join(self.refinement_results, "figure_to_title.txt"), "a+") as f:
                f.write(f"Space_sampling_{time_stamp} :\n")
                f.write(f"      grid_size: {self.sample_size}\n")
                f.write(f"      constraints: {self.constraints_file.get()}\n")

        self.space_changed = False
        self.constraints_changed = False
        self.show_quantitative = True
        self.show_samples = False

        ## Autosave
        self.save_space(os.path.join(self.tmp_dir, "space.p"))

        self.status_set("Space sampling finished.")

    def hastings(self):
        """ Samples (Parameter) Space using Metropolis Hastings. """
        print("Checking the inputs.")
        self.check_changes("functions")
        self.check_changes("data")
        self.check_changes("data_weights")

        if self.data_weights:
            raise NotImplementedError("Weighted constraints are not Implemented yet")

        print("Metropolis-Hastings ...")
        self.status_set("Metropolis-Hastings - checking inputs")

        # if self.constraints:
        #     messagebox.showwarning("Metropolis Hastings", "Data and functions are being used to run Metropolis Hasting, make sure they are in accordance with computed constrains.")

        ## TODO transformation back to data and functions from constraints #Hard_task
        if self.functions == "":
            messagebox.showwarning("Metropolis-Hastings", "Load functions before running Metropolis-Hastings.")
            return

        if self.data == []:
            messagebox.showwarning("Metropolis-Hastings", "Load data before running Metropolis-Hastings.")
            return

        if self.constraints_changed:
            messagebox.showwarning("Metropolis-Hastings", "Constraints changed and may not correspond to the functions which are about to be used.")

        ## Check functions / Get function parameters
        self.validate_parameters(where=self.functions)

        self.status_set("Metropolis-Hastings is running ...")
        if not self.silent.get():
            print("functions", self.functions)
            print("function params", self.parameters)
            print("data", self.data)

        # if not self.validate_space("Metropolis-Hastings"):
        #     return

        if self.space:
            assert isinstance(self.space, space.RefinedSpace)
            if self.parameters != self.space.params:
                messagebox.showwarning("Metropolis-Hastings", "Space you have obtained does not correspond to functions which are about to be used.")
        assert isinstance(self.parameters, list)
        self.create_window_to_load_param_point(parameters=self.parameters)

        ## Create a warning
        # if int(self.n_samples_entry.get()) < int(self.observations_samples_size_entry.get()):
        #    messagebox.showwarning("Metropolis Hastings", "Number of samples from observations (data) is higher than number of observation, using all observations as samples.")

        ## Clear figure
        self.set_lower_figure(clear=True)
        # self.page6_figure2.clf()
        # self.page6_b = self.page6_figure2.add_subplot(111)
        # self.page6_figure2.canvas.draw()
        # self.page6_figure2.canvas.flush_events()

        from metropolis_hastings import init_mh

        try:
            self.cursor_toggle_busy(True)

            ## Progress Bar
            if self.show_progress:
                self.new_window = Toplevel(self)
                Label(self.new_window, text="Metropolis Hastings progress:", anchor=W, justify=LEFT).pack()
                Label(self.new_window, textvar=self.progress, anchor=W, justify=LEFT).pack()
                self.progress_bar = Progressbar(self.new_window, orient=HORIZONTAL, length=100, mode='determinate')
                Label(self.new_window, textvar=self.progress_time, anchor=W, justify=LEFT).pack()
                self.progress_bar.pack()
                self.update()

            ## This progress is passed as whole to update the thing inside the called function
            assert isinstance(self.data, list)
            assert isinstance(self.functions, list)

            self.mh_results = init_mh(self.parameters, self.parameter_domains, self.functions, self.data,
                                      int(self.n_samples_entry.get()), int(self.MH_sampling_iterations_entry.get()),
                                      eps=0, # float(self.eps_entry.get()), ## setting eps=0,
                                      sd=float(self.sd_entry.get()), theta_init=self.parameter_point,
                                      where=[self.page6_figure2, self.page6_b],
                                      progress=self.update_progress_bar if self.show_progress else False,
                                      debug=self.debug.get(), bins=int(float(self.bins_entry.get())),
                                      burn_in=float(self.burn_in_entry.get()), is_probability=True,
                                      timeout=int(float(self.mh_timeout_entry.get())), draw_plot=self.draw_plot_window,
                                      metadata=self.show_mh_metadata.get())

            assert isinstance(self.mh_results, HastingsResults)
            if self.show_opt_point.get():
                self.mh_results.set_opt_point(self.optimised_param_point)
            if self.show_mh_true_point:
                self.mh_results.set_true_point(self.mh_true_point)
            spam = self.mh_results.show_mh_heatmap(where=[self.page6_figure2, self.page6_b], show_true_point=self.show_mh_true_point, show_opt_point=self.show_opt_point.get())

            if spam[0] is not False:
                self.page6_figure2, self.page6_b = spam
                self.page6_figure2.tight_layout()
                self.page6_figure2.canvas.draw()
                self.page6_figure2.canvas.flush_events()
                self.update()
            else:
                messagebox.showwarning("Metropolis Hastings", "No accepted point found, not showing the plot")
                ## Clear figure
                self.page6_figure2.clf()
                self.page6_b = self.page6_figure2.add_subplot(111)
                self.page6_figure2.canvas.draw()
                self.page6_figure2.canvas.flush_events()
                self.update()
        finally:
            try:
                if self.show_progress:
                    self.new_window.destroy()
                    del self.new_window
                    self.progress.set("0%")
                self.cursor_toggle_busy(False)
            except TclError:
                return

        ## Autosave
        self.save_mh_results(os.path.join(self.tmp_dir, "mh_results.p"))

        ## Autosave figure
        if self.save.get():
            time_stamp = str(strftime("%d-%b-%Y-%H-%M-%S", localtime())) + ".png"
            self.page6_figure2.savefig(os.path.join(self.mh_results_dir, f"Metropolis-Hastings_{time_stamp}"), bbox_inches='tight')
            print("Figure stored here: ", os.path.join(self.mh_results_dir, f"Metropolis-Hastings_{time_stamp}"))
            with open(os.path.join(self.mh_results_dir, "figure_to_title.txt"), "a+") as f:
                f.write(f"Metropolis-Hastings_{time_stamp} :\n")
                f.write(f"      data: {self.data_file.get()}\n")
                f.write(f"      functions: {self.functions_file.get()}\n")

                # try:
        #     self.cursor_toggle_busy(True)
        #     init_mh(self.space, self.data, self.functions, int(self.n_samples_entry.get()), int(self.N_obs_entry.get()), int(self.MH_samples_entry.get()), float(self.eps_entry.get()), where=[self.page6_figure2, self.page6_b])
        # except:
        #     messagebox.showerror(sys.exc_info()[1], "Try to check whether the data, functions, and computed constraints are aligned.")
        # finally:
        #     self.cursor_toggle_busy(False)

    def mhmh(self):
        """ Runs MHMH, shows plots"""
        print("Checking the inputs.")
        self.check_changes("constraints")
        self.check_changes("data_intervals")
        self.check_changes("functions")
        self.check_changes("data")
        self.check_changes("data_weights")

        ## Internal setting showing that only newly added part should be visualised
        show_all = False
        no_max_depth = False

        ## Getting values from entry boxes
        self.max_depth = int(self.max_depth_entry.get())
        if self.max_depth < 0:
            no_max_depth = True
            self.max_depth = 3
        self.coverage = float(self.coverage_entry.get())
        # self.epsilon = float(self.epsilon_entry.get())
        self.epsilon = 0  ## no minimal size of hyperrectangle
        self.delta = float(self.delta_entry.get())
        if not isinstance(self.space, str):
            self.space_coverage = float(self.space.get_coverage())
        else:
            self.space_coverage = 0
        if self.data_weights:
            raise NotImplementedError("Weighted constraints are not Implemented yet")

        print("MHMH ...")
        self.status_set("MHMH - checking inputs")

        # if self.constraints:
        #     messagebox.showwarning("Metropolis Hastings", "Data and functions are being used to run Metropolis Hasting, make sure they are in accordance with computed constrains.")

        ## TODO transformation back to data and functions from constraints #Hard_task
        if self.functions == "":
            messagebox.showwarning("MHMH", "Load functions before running MHMH.")
            return
        assert isinstance(self.functions, list)

        if self.data == []:
            messagebox.showwarning("MHMH", "Load data before running MHMH.")
            return

        if self.constraints_changed:
            messagebox.showwarning("MHMH", "Constraints changed and may not correspond to the functions which are about to be used.")

        if self.constraints == "":
            messagebox.showwarning("MHMH", "Compute constraints before running MHMH.")
            return
        assert isinstance(self.constraints, list)

        ## Checking if all entries filled
        if self.max_depth == "":
            messagebox.showwarning("MHMH", "Choose max recursion depth before running running MHMH.")
            return

        if self.coverage == "":
            messagebox.showwarning("MHMH", "Choose coverage, nonwhite fraction to reach before running MHMH.")
            return

        if self.epsilon == "":
            messagebox.showwarning("MHMH", "Choose epsilon, min rectangle size before running MHMH.")
            return

        if self.alg_entry.get() == "":
            messagebox.showwarning("MHMH", "Pick algorithm for the refinement before running MHMH.")
            return

        if self.space_coverage >= self.coverage:
            messagebox.showinfo("MHMH", "You already achieved higher coverage than the goal.")
            return

        if not self.validate_space("MHMH"):
            return

        if int(self.alg_entry.get()) <= 4 and not self.z3_constraints:
            for constraint in self.constraints:
                if is_this_exponential_function(constraint):
                    if not askyesno("MHMH",
                                    "Some constraints contain exponential function, we recommend using interval algorithmic (algorithm 5). Do you want to proceed anyway?"):
                        return
                    break

        if int(self.max_depth) > 14:
            if not askyesno("MHMH", "Recursion this deep may cause segmentation fault. Do you want to continue?"):
                return

        if self.space:
            assert isinstance(self.space, space.RefinedSpace)
            if self.parameters != self.space.params:
                messagebox.showwarning("Metropolis-Hastings", "Space you have obtained does not correspond to functions which are about to be used.")
        assert isinstance(self.parameters, list)

        if int(self.refinement_cores_entry.get()) > 1:
            cores = int(self.refinement_cores_entry.get())
        else:
            cores = False

        self.coverage = float(self.coverage_entry.get())
        self.epsilon = 0  ## no minimal size of hyperrectangle
        self.delta = float(self.delta_entry.get())

        if self.mh_results == "":
            rerun_mh = True
        else:
            rerun_mh = askyesno("MHMH", "Do you want to rerun Metropolis-Hasting?")

        if rerun_mh:
            self.hastings()

        assert isinstance(self.mh_results, HastingsResults)
        self.cursor_toggle_busy(True)
        a, b = initialise_mhmh(self.parameters, self.parameter_domains, self.functions, self.constraints, self.data,
                               int(self.n_samples_entry.get()), int(self.MH_sampling_iterations_entry.get()), eps=0,
                               sd=float(self.sd_entry.get()), theta_init=self.parameter_point if rerun_mh else False,
                               is_probability=True, where_mh=[self.page6_figure2, self.page6_b], where_ref=None,
                               progress=False, burn_in=float(self.burn_in_entry.get()), debug=self.debug.get(),
                               bins=int(float(self.bins_entry.get())), mh_timeout=int(float(self.mh_timeout_entry.get())),
                               metadata=self.show_mh_metadata.get(), draw_plot=self.draw_plot_window,
                               save=False, silent=self.silent.get(), recursion_depth=10, epsilon=self.epsilon,
                               delta=self.delta, coverage=self.coverage, version=int(self.alg_entry.get()),
                               solver=str(self.solver_entry.get()), parallel=cores,
                               gui=self.update_progress_bar if self.show_progress else False,
                               ref_timeout=int(float(self.refinement_timeout_entry.get())), mh_result=self.mh_results)

        self.space = a
        self.show_space(show_refinement=True, show_samples=self.show_samples, show_true_point=self.show_space_true_point,
                        show_opt_point=self.show_opt_point.get(), prefer_unsafe=self.show_red_in_multidim_refinement.get(),
                        show_all=show_all, warnings=not (no_max_depth and self.space.get_coverage() < self.coverage),
                        is_sampling_guided=self.sampling_guided_refinement.get(),
                        is_parallel_refinement=int(self.refinement_cores_entry.get()) > 1)
        self.page6_figure.tight_layout()  ## By huypn
        self.page6_figure.canvas.draw()
        self.page6_figure.canvas.flush_events()

        ## Autosave figure
        if self.save.get():
            time_stamp = str(strftime("%d-%b-%Y-%H-%M-%S", localtime())) + ".png"
            self.page6_figure.savefig(os.path.join(self.refinement_results, "tmp", f"Space_refinement_{time_stamp}"),
                                      bbox_inches='tight')
            print("Figure stored here: ",
                  os.path.join(self.refinement_results, "tmp", f"Space_refinement_{time_stamp}"))
            with open(os.path.join(self.refinement_results, "tmp", "figure_to_title.txt"), "a+") as f:
                f.write(f"Space_refinement_{time_stamp} :\n")
                f.write(f"      constraints: {self.constraints_file.get()}\n")

        self.print_space()

        self.constraints_changed = False
        self.space_changed = False

        ## Autosave
        try:
            self.save_space(os.path.join(self.tmp_dir, "space.p"))
        except ValueError as err:
            print(f"Space could not be loaded, {str(err)}")
            messagebox.showwarning("Space could not be saved.", str(err))

        if no_max_depth and self.space.get_coverage() < self.coverage:
            self.refine_space()
        else:
            self.status_set("Space refinement finished.")
        self.cursor_toggle_busy(False)

    def refine_space(self):
        """ Refines (Parameter) Space. Plots the results. """
        ## Internal setting showing that only newly added part should be visualised
        show_all = False

        print("Checking the inputs.")
        self.check_changes("constraints")
        self.check_changes("data_intervals")
        self.check_changes("functions")

        print("Refining space ...")
        self.status_set("Space refinement - checking inputs")

        ## Inner settings
        no_max_depth = False

        ## Getting values from entry boxes
        self.max_depth = int(self.max_depth_entry.get())
        if self.max_depth < 0:
            no_max_depth = True
            self.max_depth = 3
        self.coverage = float(self.coverage_entry.get())
        # self.epsilon = float(self.epsilon_entry.get())
        self.epsilon = 0  ## no minimal size of hyperrectangle
        self.delta = float(self.delta_entry.get())
        if not isinstance(self.space, str):
            self.space_coverage = float(self.space.get_coverage())
        else:
            self.space_coverage = 0

        ## Checking if all entries filled
        if self.max_depth == "":
            messagebox.showwarning("Refine space", "Choose max recursion depth before running refinement.")
            return

        if self.coverage == "":
            messagebox.showwarning("Refine space", "Choose coverage, nonwhite fraction to reach before running refinement.")
            return

        if self.epsilon == "":
            messagebox.showwarning("Refine space", "Choose epsilon, min rectangle size before running refinement.")
            return

        if self.alg_entry.get() == "":
            messagebox.showwarning("Refine space", "Pick algorithm for the refinement before running refinement.")
            return

        # if int(self.alg.get()) == 5:
        #     if self.functions == "":
        #         messagebox.showwarning("Refine space", "Load or synthesise functions before refinement.")
        #         return
        #     if self.data_intervals == []:
        #         messagebox.showwarning("Refine space", "Load or compute data intervals before refinement.")
        #         return
        # else:

        if self.constraints == "":
            messagebox.showwarning("Refine space", "Load or calculate constraints before running refinement.")
            return

        if self.space_coverage >= self.coverage:
            messagebox.showinfo("Refine space", "You already achieved higher coverage than the goal.")
            return

        if not self.validate_space("Refine Space"):
            return

        if int(self.alg_entry.get()) <= 4 and not self.z3_constraints:
            for constraint in self.constraints:
                if is_this_exponential_function(constraint):
                    if not askyesno("Refinement", "Some constraints contain exponential function, we recommend using interval algorithmic (algorithm 5). Do you want to proceed anyway?"):
                        return
                    break

        if self.presampled_refinement.get() and not(self.space.get_sat_samples() + self.space.get_unsat_samples()):
            messagebox.showwarning("Refine space", "No sampling to be used, please run it before running Presampled refinement.")
            return

        if int(self.max_depth) > 14:
            if not askyesno("Space refinement", "Recursion this deep may cause segmentation fault. Do you want to continue?"):
                return

        try:
            single_call_timeout = float(self.refinement_single_call_timeout_entry.get())
        except Exception as err:
            print(str(err))
            single_call_timeout = 0

        self.status_set("Space refinement is running ...")
        # print(colored(f"self.space, {self.space.nice_print()}]", "blue"))

        try:
            self.cursor_toggle_busy(True)

            ## Progress Bar
            if self.show_progress:
                self.new_window = Toplevel(self)
                Label(self.new_window, text="Refinement progress:", anchor=W, justify=LEFT).pack()
                Label(self.new_window, textvar=self.progress, anchor=W, justify=LEFT).pack()

                self.progress_bar = Progressbar(self.new_window, orient=HORIZONTAL, length=100, mode='determinate')
                self.progress_bar.pack(expand=True, fill=BOTH, side=TOP)
                self.update_progress_bar(change_to=0, change_by=False)

                self.update()

            ## Refresh of plot before refinement
            if self.show_quantitative:
                self.clear_space(warning=not(no_max_depth and self.space.get_coverage() < self.coverage))
                self.show_quantitative = False
                show_all = True

            ## RETURNS TUPLE -- (SPACE,(NONE, ERROR TEXT)) or (SPACE, )
            ## feeding z3 solver with z3 expressions, python expressions otherwise
            # if int(self.alg.get()) == 5:
            #     spam = check_deeper(self.space, [self.functions, self.data_intervals], self.max_depth, self.epsilon,
            #                         self.coverage, silent=self.silent.get(), version=int(self.alg.get()), sample_size=False,
            #                         debug=self.debug.get(), save=False, where=[self.page6_figure, self.page6_a],
            #                         solver=str(self.solver.get()), delta=self.delta, gui=self.update_progress_bar)
            if int(self.refinement_cores_entry.get()) > 1:
                if str(self.solver_entry.get()) == "z3" and self.z3_constraints:
                    assert isinstance(self.z3_constraints, list)
                    spam = check_deeper_parallel(self.space, self.z3_constraints, self.max_depth, self.epsilon,
                                                 self.coverage, silent=self.silent.get(), version=int(self.alg_entry.get()),
                                                 sample_size=self.presampled_refinement.get(), debug=self.debug.get(),
                                                 sample_guided=self.sampling_guided_refinement.get(), save=False,
                                                 where=[self.page6_figure, self.page6_a], solver=str(self.solver_entry.get()),
                                                 delta=self.delta, gui=self.update_progress_bar if self.show_progress else False,
                                                 show_space=False, iterative=self.iterative_refinement.get(),
                                                 timeout=int(float(self.refinement_timeout_entry.get())),
                                                 single_call_timeout=single_call_timeout,
                                                 parallel=int(self.refinement_cores_entry.get()))
                else:
                    assert isinstance(self.constraints, list)
                    spam = check_deeper_parallel(self.space, self.constraints, self.max_depth, self.epsilon,
                                                 self.coverage, silent=self.silent.get(), version=int(self.alg_entry.get()),
                                                 sample_size=self.presampled_refinement.get(), debug=self.debug.get(),
                                                 sample_guided=self.sampling_guided_refinement.get(), save=False,
                                                 where=[self.page6_figure, self.page6_a], solver=str(self.solver_entry.get()),
                                                 delta=self.delta, gui=self.update_progress_bar if self.show_progress else False,
                                                 show_space=False, iterative=self.iterative_refinement.get(),
                                                 timeout=int(float(self.refinement_timeout_entry.get())),
                                                 single_call_timeout=single_call_timeout,
                                                 parallel=int(self.refinement_cores_entry.get()))
            else:
                if single_call_timeout > 0:
                    messagebox.showwarning("Refinement settings", "Single call timeout for sequential refinement not implemented")
                    return
                if self.sampling_guided_refinement.get():
                    messagebox.showwarning("Refinement settings", "Sampling guided version is not implemented for sequential version.")
                    return
                if str(self.solver_entry.get()) == "z3" and self.z3_constraints:
                    assert isinstance(self.z3_constraints, list)
                    spam = check_deeper(self.space, self.z3_constraints, self.max_depth, self.epsilon, self.coverage,
                                        silent=self.silent.get(), version=int(self.alg_entry.get()),
                                        sample_size=self.presampled_refinement.get(), debug=self.debug.get(), save=False,
                                        where=[self.page6_figure, self.page6_a], solver=str(self.solver_entry.get()),
                                        delta=self.delta, gui=self.update_progress_bar if self.show_progress else False,
                                        show_space=False, iterative=self.iterative_refinement.get(),
                                        timeout=int(float(self.refinement_timeout_entry.get())),)
                else:
                    assert isinstance(self.constraints, list)
                    spam = check_deeper(self.space, self.constraints, self.max_depth, self.epsilon, self.coverage,
                                        silent=self.silent.get(), version=int(self.alg_entry.get()),
                                        sample_size=self.presampled_refinement.get(), debug=self.debug.get(), save=False,
                                        where=[self.page6_figure, self.page6_a], solver=str(self.solver_entry.get()),
                                        delta=self.delta, gui=self.update_progress_bar if self.show_progress else False,
                                        show_space=False, iterative=self.iterative_refinement.get(),
                                        timeout=int(float(self.refinement_timeout_entry.get())))
        finally:
            try:
                self.cursor_toggle_busy(False)
                if self.show_progress:
                    self.new_window.destroy()
                    self.progress.set("0%")
            except TclError:
                return

        ## If the visualisation of the space did not succeed
        if isinstance(spam, tuple):
            self.space = spam[0]
            if no_max_depth and self.space.get_coverage() < self.coverage:
                pass
            else:
                messagebox.showinfo("Space refinement", spam[1])
        else:
            self.space = spam
            self.show_space(show_refinement=True, show_samples=self.show_samples, show_true_point=self.show_space_true_point,
                            show_opt_point=self.show_opt_point.get(), prefer_unsafe=self.show_red_in_multidim_refinement.get(),
                            show_all=show_all, warnings=not(no_max_depth and self.space.get_coverage() < self.coverage),
                            is_sampling_guided=self.sampling_guided_refinement.get(),
                            is_parallel_refinement=int(self.refinement_cores_entry.get()) > 1)
            self.page6_figure.tight_layout()  ## By huypn
            self.page6_figure.canvas.draw()
            self.page6_figure.canvas.flush_events()

            ## Autosave figure
            if self.save.get():
                time_stamp = str(strftime("%d-%b-%Y-%H-%M-%S", localtime())) + ".png"
                self.page6_figure.savefig(os.path.join(self.refinement_results, "tmp", f"Space_refinement_{time_stamp}"),
                                          bbox_inches='tight')
                print("Figure stored here: ", os.path.join(self.refinement_results, "tmp", f"Space_refinement_{time_stamp}"))
                with open(os.path.join(self.refinement_results, "tmp", "figure_to_title.txt"), "a+") as f:
                    f.write(f"Space_refinement_{time_stamp} :\n")
                    f.write(f"      constraints: {self.constraints_file.get()}\n")

        self.print_space()

        self.constraints_changed = False
        self.space_changed = False

        ## Autosave
        try:
            self.save_space(os.path.join(self.tmp_dir, "space.p"))
        except ValueError as err:
            print(f"Space could not be loaded, {str(err)}")
            messagebox.showwarning("Space could not be saved.", str(err))

        if no_max_depth and self.space.get_coverage() < self.coverage:
            self.refine_space()
        else:
            self.status_set("Space refinement finished.")

    def edit_space(self):
        """ Edits space values. """

        if self.space == "":
            messagebox.showwarning("Edit space", "There is no space to be edit.")
            return

        print("Editing the space.")
        self.validate_parameters(self.space, intervals=True, keep_space=True)
        assert isinstance(self.space, space.RefinedSpace)
        self.space.params = self.parameters
        self.space.set_region(self.parameter_domains)
        self.print_space()
        ## TODO add changing the parameter types

    ## VALIDATE VARIABLES (PARAMETERS, constraints, SPACE)
    def validate_parameters(self, where: Iterable, intervals=True, force=False, keep_space=False):
        """ Validates (functions, constraints, and space) parameters.

        Args:
            where (Iterable): a structure pars parameters from (e.g. self.functions)
            intervals (bool): whether to check also parameter intervals
            force (bool): if True the param_intervals will be redone with possibly of values as default
            keep_space (bool): if Ture it will keep the previous values -- used to edit space
        """
        if keep_space:
            # Apply space values
            assert isinstance(self.space, space.RefinedSpace)
            self.parameters = self.space.params
            self.parameter_domains = self.space.region
            force = True  # Force the values to be changed

        if not self.parameters:
            print("Parsing parameters ...")
            globals()["parameters"] = set()
            if where == "model":
                globals()["parameters"] = parse_params_from_model(self.model_file.get())
            elif isinstance(where, str):
                raise NotImplementedError(f"Validating parameters from {where} is not implemented yet")
            else:
                for polynome in where:
                    globals()["parameters"].update(find_param_old(polynome, debug=self.debug.get()))
            globals()["parameters"] = sorted(list(globals()["parameters"]))
            self.parameters = globals()["parameters"]
            if not self.silent.get():
                print("parameters", self.parameters)

        if (not self.parameter_domains) and intervals or force:
            ## TODO Maybe rewrite this as key and pass the argument to load_param_intervals
            self.key = StringVar()
            self.status_set("Choosing ranges of parameters:")
            self.new_window = Toplevel(self)
            label = Label(self.new_window, text="Please choose intervals of the parameters to be used:")
            label.grid(row=0)
            self.key.set(" ")

            i = 1
            ## For each param create an entry
            self.parameter_domains_entries = []
            for index, param in enumerate(self.parameters):
                Label(self.new_window, text=param, anchor=W, justify=LEFT).grid(row=i, column=0)
                spam_low = Entry(self.new_window)
                spam_high = Entry(self.new_window)
                spam_low.grid(row=i, column=1)
                spam_high.grid(row=i, column=2)
                if self.parameter_domains:
                    spam_low.insert(END, str(self.parameter_domains[index][0]))
                    spam_high.insert(END, str(self.parameter_domains[index][1]))
                else:
                    spam_low.insert(END, '0')
                    spam_high.insert(END, '1')
                self.parameter_domains_entries.append([spam_low, spam_high])
                i = i + 1

            ## To be used to wait until the button is pressed
            self.button_pressed.set(False)
            load_param_intervals_button = Button(self.new_window, text="OK",
                                                 command=self.load_param_intervals_from_window)
            load_param_intervals_button.grid(row=i)
            load_param_intervals_button.focus()
            load_param_intervals_button.bind('<Return>', self.load_param_intervals_from_window)
            # self.new_window.bind('<Return>', self.load_param_intervals_from_window)

            load_param_intervals_button.wait_variable(self.button_pressed)
        elif (len(self.parameter_domains) is not len(self.parameters)) and intervals:
            self.parameter_domains = []
            self.validate_parameters(where=where)

    def validate_constraints(self, position=False, force=False):
        """ Validates created constraints.

        Args:
            position (string): Name of the place from which is being called e.g. "Refine Space"/"Sample space"
            force (bool): force to validate constraints
        """
        print("Validating constraints ...")
        ## MAYBE an error here
        if not self.constraints == "":
            print("constraints not empty, not checking them.")
            return True
        if position is False:
            position = "Validating constraints"
        ## If constraints empty create constraints
        if self.functions_changed or self.data_intervals_changed or force:
            if not self.silent.get():
                print("Functions: ", self.functions)
                print("Intervals: ", self.data_intervals)
            ## If functions empty raise an error (return False)
            if self.functions == "":
                print("No functions loaded nor not computed to create properties")
                messagebox.showwarning(position, "Load or synthesise functions first.")
                return False
            ## If intervals empty raise an error (return False)
            if self.data_intervals == []:
                print("Intervals not computed, properties cannot be computed")
                messagebox.showwarning(position, "Compute intervals first.")
                return False

            ## Check if the number of functions and intervals is equal
            if len(self.functions) != len(self.data_intervals):
                messagebox.showerror(position, "The number of functions and data points (or intervals) is not equal")
                return

            if self.functions_changed:
                self.functions_changed = False

            if self.data_intervals_changed:
                self.data_intervals_changed = False

            ## Create constraints
            assert isinstance(self.functions, list)
            assert isinstance(self.data_intervals, list)
            self.constraints = ineq_to_constraints(self.functions, self.data_intervals, decoupled=False, silent=self.silent.get())
            if self.debug.get():
                print("self.constraints", self.constraints)
            if self.z3_functions:
                assert isinstance(self.z3_functions, list)
                self.z3_constraints = ineq_to_constraints(self.z3_functions, self.data_intervals, decoupled=False, silent=self.silent.get())

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
                print("self.constraints: ", self.constraints)
        return True

    def refresh_space(self):
        """ Refreshes space. """
        if self.space:
            if not askyesno("Analyze space", "Data of the space, its text representation, and the plot will be lost. Do you want to proceed?"):
                return
        self.space_changed = False
        self.print_space(clear=True)
        self.clear_space()
        self.space_file.set("")
        self.space = ""
        # self.parameters = ""
        # self.parameter_domains = []
        self.status_set("Space refreshed.")

    def export_space_text(self, file=False):
        """ Exports textual representation of space into a text file.

        Args:
            file (string or False):  file to export space text
        """
        if file:
            save_space_text_file = file
        else:
            print("Saving the textual representation of space ...")
            if self.space == "":
                self.status_set("There is no space to be saved.")
                messagebox.showwarning("Saving the textual representation of space", "There is no space to be saved.")
                return
            self.status_set("Please select folder to store the space in.")
            save_space_text_file = filedialog.asksaveasfilename(initialdir=self.refinement_results,
                                                                title="Saving the textual representation of space - Select file",
                                                                filetypes=(("text files", "*.txt"), ("all files", "*.*")))
            if save_space_text_file == "":
                self.status_set("No file selected to store Saving the textual representation of space in.")
                return

        if "." not in basename(save_space_text_file):
            save_space_text_file = save_space_text_file + ".txt"

        if not self.silent.get():
            print("Saving the textual representation of space as a file:", save_space_text_file)

        with open(save_space_text_file, "w") as save_space_text_file:
            assert isinstance(self.space, space.RefinedSpace)
            save_space_text_file.write(self.space.nice_print(full_print=True))

        if not file:
            self.status_set("Textual representation of space saved.")

    def customize_refinement_results(self):
        """ Customizes refinement Plot. """
        if self.refinement_results:
            if not askyesno("Analyze space", "Sample & Refinement plot will be lost. Do you want to proceed?"):
                return

        self.new_window = Toplevel(self)
        label = Label(self.new_window, text="Customize Refinement Plot")
        label.grid(row=0)

        show_red_in_multidim_refinement_chekbutton = Checkbutton(self.new_window, text="Show unsafe space instead of safe space in multidimensional plot.", variable=self.show_red_in_multidim_refinement)
        show_red_in_multidim_refinement_chekbutton.grid(row=1, column=0)
        hide_legend_chekbutton = Checkbutton(self.new_window, text="Hide plot legend.", variable=self.hide_legend_refinement)
        hide_legend_chekbutton.grid(row=2, column=0)
        hide_title_chekbutton = Checkbutton(self.new_window, text="Hide plot title.", variable=self.hide_title_refinement)
        hide_title_chekbutton.grid(row=3, column=0)

        ## To be used to wait until the button is pressed
        self.button_pressed.set(False)
        costumize_mh_results_button = Button(self.new_window, text="OK", command=self.change_refinement_plot)
        costumize_mh_results_button.grid(row=4)
        costumize_mh_results_button.focus()
        costumize_mh_results_button.bind('<Return>', self.change_refinement_plot)

        costumize_mh_results_button.wait_variable(self.button_pressed)

    def change_refinement_plot(self, fake_param=False):
        """ Parses window changing for refinement plot. """
        try:
            if self.space != "":
                assert isinstance(self.space, space.RefinedSpace)
                if len(self.space.params) > 2:
                    self.show_space(self.show_refinement, self.show_samples, self.show_space_true_point,
                                    show_opt_point=self.show_opt_point.get(), show_all=True,
                                    prefer_unsafe=self.show_red_in_multidim_refinement.get())
        finally:
            try:
                self.new_window.destroy()
                del self.new_window
                self.cursor_toggle_busy(False)
                self.progress.set("0%")
            except TclError:
                return

    def refresh_mh(self):
        """ Refreshes MH results. """
        if self.mh_results:
            if not askyesno("Analyze space", "Data and the plot of the Metropolis-Hastings will be lost. Do you want to proceed?"):
                return
        self.mh_results_changed = False
        self.mh_results = ""
        self.page6_figure2.clf()
        self.page6_b = self.page6_figure2.add_subplot(111)
        self.page6_figure2.canvas.draw()
        self.page6_figure2.canvas.flush_events()

        self.status_set("MH results refreshed.")

    def customize_mh_results(self):
        """ Customizes MH Plot. """
        if isinstance(self.mh_results, HastingsResults):
            if not askyesno("Metropolis-Hastings", "Metropolis-Hastings plot will be lost. Do you want to proceed?"):
                return
        else:
            messagebox.showinfo("Metropolis-Hastings", "There is no plot to customize!")
            return

        self.new_window = Toplevel(self)
        label = Label(self.new_window, text="Customize MH Plot")
        label.grid(row=0)

        Label(self.new_window, text="Grid size", anchor=W, justify=LEFT).grid(row=1, column=0)
        self.mh_grid_size_entry = Entry(self.new_window)
        self.mh_grid_size_entry.grid(row=1, column=1)
        self.mh_grid_size_entry.insert(END, str(self.mh_results.bins))

        burn_in_label = Label(self.new_window, text="Burn-in", anchor=W, justify=LEFT)
        burn_in_label.grid(row=2, column=0)
        createToolTip(burn_in_label, text='Trim the fraction of accepted points from beginning')
        self.burn_in_entry_2 = Entry(self.new_window)
        self.burn_in_entry_2.grid(row=2, column=1)
        self.burn_in_entry_2.insert(END, str(self.mh_results.get_burn_in()))

        # Label(self.new_window, text="Show 2D MH plot as scatter line plot", anchor=W, justify=LEFT).grid(row=3, column=0)
        show_mh_as_scatter_checkbutton = Checkbutton(self.new_window, text="Show 2D MH plot as scatter line plot", variable=self.show_mh_as_scatter)
        show_mh_as_scatter_checkbutton.grid(row=3, column=0)

        ## To be used to wait until the button is pressed
        self.button_pressed.set(False)
        costumize_mh_results_button = Button(self.new_window, text="OK", command=self.change_MH_Plot)
        costumize_mh_results_button.grid(row=4)
        costumize_mh_results_button.focus()
        costumize_mh_results_button.bind('<Return>', self.change_MH_Plot)

        costumize_mh_results_button.wait_variable(self.button_pressed)

    def change_MH_Plot(self):
        """ Parses window changing MH Plot. """
        try:
            bins = int(self.mh_grid_size_entry.get())
            burn_in = float(self.burn_in_entry_2.get())
            as_scatter = bool(self.show_mh_as_scatter.get())

            self.refresh_mh_figure(bins, burn_in, as_scatter, self.show_mh_true_point, self.show_opt_point.get())
        finally:
            try:
                self.new_window.destroy()
                del self.new_window
                self.cursor_toggle_busy(False)
                self.progress.set("0%")
            except TclError:
                return
            except AttributeError:  ##  for set mh true point
                return

    def refresh_mh_figure(self, bins, burn_in, as_scatter, show_true_point, show_opt_point):
        """ Refreshes MH figure """
        assert isinstance(self.mh_results, HastingsResults)

        ## Clear figure
        self.page6_figure2.clf()
        self.page6_b = self.page6_figure2.add_subplot(111)
        self.page6_figure2.canvas.draw()
        self.page6_figure2.canvas.flush_events()

        self.mh_results.set_burn_in(burn_in)
        self.mh_results.set_bins(bins)
        self.mh_results.set_as_scatter(as_scatter)
        spam = self.mh_results.show_mh_heatmap(where=[self.page6_figure2, self.page6_b], bins=bins, burn_in=burn_in,
                                               as_scatter=as_scatter, show_true_point=show_true_point, show_opt_point=show_opt_point)

        if spam[0] is not False:
            self.page6_figure2, self.page6_b = spam
            self.page6_figure2.tight_layout()
            self.page6_figure2.canvas.draw()
            self.page6_figure2.canvas.flush_events()
            self.update()
        else:
            pass

    def show_mh_iterations(self):
        """ Create Scatter plot showing accepted and rejected points in its given order. """
        if self.mh_results == "":
            return
        else:
            assert isinstance(self.mh_results, HastingsResults)
            self.mh_results.show_iterations(where=self.draw_plot_window)
            self.mh_results.show_iterations_bokeh()
            # self.mh_results.show_iterations(where=self.draw_plot_window)

    def show_mh_acc_points(self):
        """ Shows trace and histogram of accepted points. """
        if self.mh_results == "":
            return
        else:
            assert isinstance(self.mh_results, HastingsResults)
            self.mh_results.show_accepted(where=self.draw_plot_window)
            self.mh_results.show_accepted_bokeh()
            # try:
            #     self.mh_results.show_accepted(where=self.draw_plot_window)
            # except:
            #     self.mh_results.show_accepted_bokeh()

    def export_acc_points(self, file=False):
        """ Exports accepted points of metropolis Hastings.

        Args:
            file (string or False):  file to export accepted points of MH
        """
        if self.mh_results == "":
            return
        else:
            print("Exporting accepted points of MH ...")
            self.status_set("Please select folder to store the export in.")
            acc_mh_export_text_file = filedialog.asksaveasfilename(initialdir=self.mh_results_dir,
                                                                   title="Exporting accepted points of MH - Select file",
                                                                   filetypes=(("text files", "*.txt"), ("all files", "*.*")))
            if acc_mh_export_text_file == "":
                self.status_set("No file selected for the textual representation of accepted points of MH to save in.")
                return

        if "." not in basename(acc_mh_export_text_file):
            acc_mh_export_text_file = acc_mh_export_text_file + ".txt"

        if not self.silent.get():
            print("Saving the textual representation of accepted points of MH as a file:", acc_mh_export_text_file)

        with open(acc_mh_export_text_file, "w") as f:
            assert isinstance(self.mh_results, HastingsResults)
            for item in self.mh_results.get_acc_as_a_list():
                f.write(str(item)+",\n")

        if not file:
            self.status_set("Textual representation of accepted points of MH saved.")

    def validate_space(self, position=False, force=False):
        """ Validates space.

        Args:
            position (string): Name of the place from which is being called e.g. "Refine Space"/"Sample space"
            force (bool): if True the param_intervals will be redone with possibly of values as default
        """
        print("Checking space ...")
        if position is False:
            position = "Validating space"
        ## If the space is empty create a new one
        if self.space == "":
            if not self.silent.get():
                print("Space is empty - creating a new one.")
            ## Parse params and its intervals
            self.validate_parameters(where=self.constraints, force=False)

            ## Check whether param interval loading went good
            if isinstance(self.parameter_domains, list):
                if isinstance(self.parameter_domains[0][0], Entry):
                    self.parameter_domains = []
                    return False

            self.space = space.RefinedSpace(self.parameter_domains, self.parameters)
        else:
            if position == "Space Metropolis-Hastings":
                return True
            if self.constraints_changed:
                messagebox.showwarning(position, "Using previously created space with new constraints. Consider using fresh new space.")
                ## Check if the properties and data are valid
                globals()["parameters"] = set()
                for polynomial in self.constraints:
                    globals()["parameters"].update(find_param(polynomial))
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
        """ Opens config file in editor. """
        print("Editing config ...")
        if "wind" in system().lower():
            os.startfile(f'{os.path.join(workspace, "../config.ini")}')
        else:
            os.system(f'gedit {os.path.join(workspace, "../config.ini")}')
        self.load_config()  ## Reloading the config file after change
        self.status_set("Config file saved.")

    def show_help(self):
        """ Shows GUI help. """
        print("Showing help ...")
        webbrowser.open_new("https://github.com/xhajnal/DiPS#dips-data-informed-parameter-synthesiser")

    def check_updates(self):
        """ Shows latest releases. """
        print("Checking for updates ...")
        self.status_set("Checking for updates ...")
        webbrowser.open_new("https://github.com/xhajnal/DiPS/releases")

    def print_about(self):
        """ Shows GUI about. """
        print("Printing about ...")
        top2 = Toplevel(self)
        top2.title("About")
        top2.resizable(0, 0)
        explanation = f" DiPS version: {self.version} \n More info here: https://github.com/xhajnal/DiPS \n Powered by University of Konstanz, Masaryk University, and Max Planck Institute"
        Label(top2, justify=LEFT, text=explanation).pack(padx=13, pady=20)
        top2.transient(self)
        top2.grab_set()
        self.wait_window(top2)

        print(explanation)

    ## STATUS BAR FUNCTIONS
    def status_set(self, text, *args):
        """ Inner function to update status bar. """
        self.status.config(text=text.format(args))
        self.status.update_idletasks()

    def status_clear(self):
        """ Inner function to update status bar. """
        self.status.config(text="")
        self.status.update_idletasks()

    ## INNER TKINTER SETTINGS
    def cursor_toggle_busy(self, busy=True):
        """ Inner function to update cursor. """
        if busy:
            ## System dependent cursor setting
            if "wind" in system().lower():
                self.config(cursor='wait')
            else:
                self.config(cursor='clock')
        else:
            self.config(cursor='')
        self.update()

    def report_callback_exception(self, exc, val, tb):
        """ Inner function, Exception handling. """
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
    def draw_plot_window(self, figure, axes=False):
        """ Method to create a new window with a figure inside.

        Args:
            figure (figure): a figure to draw into the new window
            axes (axes): axes of the figure
        """
        new_plot_window = Toplevel(self)
        new_plot_frame = Frame(new_plot_window)
        new_plot_frame.pack(fill=BOTH, expand=True)

        new_plot_canvas = FigureCanvasTkAgg(figure, master=new_plot_frame)
        new_plot_toolbar = NavigationToolbar2Tk(new_plot_canvas, new_plot_frame)
        new_plot_toolbar.update()
        new_plot_canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        try:
            new_plot_canvas.draw()
        except OverflowError as err:
            pyplt.rcParams['agg.path.chunksize'] = 10000
            new_plot_canvas.draw()
            show_message(2, "Ploting window", err)

        # canvas.flush_events()
        # self.new_window.update()
        # self.update()

    def create_window_to_load_param_point(self, parameters, opt=False):
        """ Creates a window a functionality to load values of parameters.

        Args:
            parameters (list): list of param names
            opt (bool): whether to use optimised point as default values

        """
        if not self.optimised_param_point:
            opt = False

        self.new_window = Toplevel(self)
        label = Label(self.new_window, text="Please choose values of the parameters to be used:")
        label.grid(row=0)

        i = 1
        ## For each param create an entry
        self.parameter_point_entries = []
        for index, param in enumerate(parameters):
            Label(self.new_window, text=param, anchor=W, justify=LEFT).grid(row=i, column=0)
            spam = Entry(self.new_window)
            spam.grid(row=i, column=1)
            ## Insert the middle of respective domain
            try:
                if self.init_mh_with_optimised_point.get() or opt:
                    spam.insert(END, str(self.optimised_param_point[index]))
                else:
                    spam.insert(END, str((self.parameter_domains[index][0] + self.parameter_domains[index][1])/2))
            except IndexError:
                pass
            self.parameter_point_entries.append(spam)
            i = i + 1

        ## To be used to wait until the button is pressed
        self.button_pressed.set(False)
        load_true_point_button = Button(self.new_window, text="OK", command=self.load_param_point_from_window)
        load_true_point_button.grid(row=i)
        load_true_point_button.focus()
        load_true_point_button.bind('<Return>', self.load_param_point_from_window)

        load_true_point_button.wait_variable(self.button_pressed)

    def load_param_intervals_from_window(self):
        """ Inner function to parse the param intervals from created window. """
        region = []
        assert isinstance(self.parameter_domains, list)

        for param_index in range(len(self.parameters)):
            ## Getting the values from each entry, low = [0], high = [1]
            assert isinstance(self.parameter_domains_entries[param_index][0], Entry)
            assert isinstance(self.parameter_domains_entries[param_index][1], Entry)
            region.append([float(self.parameter_domains_entries[param_index][0].get()),
                           float(self.parameter_domains_entries[param_index][1].get())])
        if not self.silent.get():
            print("Region: ", region)
        del self.key
        self.new_window.destroy()
        del self.new_window
        self.parameter_domains = region
        del self.parameter_domains_entries
        self.button_pressed.set(True)
        if self.debug.get():
            if self.space:
                print("Space: ", self.space)

    def load_param_point_from_window(self):
        """ Inner function to parse the param values from created window. """
        self.parameter_point = []
        assert isinstance(self.parameter_point_entries, list)
        assert all(list(map(lambda x: isinstance(x, Entry), self.parameter_point_entries)))

        for item in self.parameter_point_entries:
            self.parameter_point.append(float(item.get()))
        self.new_window.destroy()
        del self.new_window
        self.button_pressed.set(True)

    def reinitialise_plot(self, set_onclick=False):
        """ Inner function, reinitialising the page3 plot. """
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

    def initialise_plot3(self, what=False):
        """ Plots the what (figure) into where (Tkinter object - Window/Frame/....). """
        ## Old
        # try:
        #     self.page3_canvas.get_tk_widget().destroy()
        #     self.page3_toolbar.get_tk_widget().destroy()
        #     self.update()
        # except AttributeError:
        #     pass
        self.page3_plotframe = Frame(self.frame3_right)
        self.page3_plotframe.grid(row=5, column=1, columnspan=5, padx=5, pady=4, sticky=N+S+E+W)

        self.page3_canvas = FigureCanvasTkAgg(what, master=self.page3_plotframe)
        self.page3_canvas.draw()
        self.page3_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.page3_toolbar = NavigationToolbar2Tk(self.page3_canvas, self.page3_plotframe)
        self.page3_toolbar.update()
        self.page3_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    def update_progress_bar(self, change_to=False, change_by=False, set_time=False, timeout=False):
        """ Updates progress bar.

        Args:
            change_to (number): value to set the progress:  change_to %
            change_by (number): value to add to the progress:  current progress + change_by %
            set_time (str/number): value to set current running time: set_time / timeout s
            timeout (str/number): value to set max running time: set_time / timeout s
        """
        try:
            if change_to is not False:
                self.progress_bar['value'] = 100*change_to
                self.progress.set(f"{round(100*change_to, 2)}%")
            if change_by is not False:
                self.progress_bar['value'] = self.progress_bar['value'] + 100*change_by
                self.progress.set(f"{round(self.progress_bar['value'], 2)}%")
            if set_time is not False:
                self.progress_time.set(f"{set_time}/{timeout} s")
            self.update()
        except TclError:
            return

    def ask_quit(self):
        """ x button handler. """
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.destroy()
            self.quit()

    def autoload(self, yes=False):
        """ Loads tmp files. """
        if yes:
            self.update()
            return
        if askyesno("Autoload from tmp folder", "Would you like to load autosaved files from tmp folder?"):
            self.save.set(False)
            print("Loading tmp files from ", self.tmp_dir)
            self.load_model(file=os.path.join(self.tmp_dir, "model.pm"))
            self.load_property(file=os.path.join(self.tmp_dir, "properties.pctl"))
            self.load_parsed_functions(file=os.path.join(self.tmp_dir, "functions.p"))
            # self.load_mc_result(file=os.path.join(self.tmp_dir, "functions_prism.txt"))
            # self.load_mc_result(file=os.path.join(self.tmp_dir, "functions_storm.txt"))
            self.load_data(file=os.path.join(self.tmp_dir, "data.p"))
            self.load_data_weights(file=os.path.join(self.tmp_dir, "data_weights.p"))
            self.load_data_intervals(file=os.path.join(self.tmp_dir, "data_intervals.p"))
            self.load_constraints(file=os.path.join(self.tmp_dir, "constraints.p"))
            self.load_space(file=os.path.join(self.tmp_dir, "space.p"))
            self.load_mh_results(file=os.path.join(self.tmp_dir, "mh_results.p"))
            self.load_data_informed_properties(file=os.path.join(self.tmp_dir, "data_informed_properties.pctl"))

    def set_lower_figure(self, clear=False):
        """ Configures lower figure on tab 6 (MH results). """
        ##################################################### LOWER PLOT ###############################################
        if clear:
            self.page6_plotframe2.destroy()

        self.page6_plotframe2 = Frame(self.frame_center)
        self.page6_plotframe2.pack(side=TOP, fill=Y, expand=True, padx=5)

        self.page6_figure2 = pyplt.figure(figsize=(8, 2))
        self.page6_figure2.tight_layout()  ## By huypn

        self.page6_canvas2 = FigureCanvasTkAgg(self.page6_figure2, master=self.page6_plotframe2)  # A tk.DrawingArea.
        self.page6_canvas2.draw()
        self.page6_canvas2.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.page6_toolbar2 = NavigationToolbar2Tk(self.page6_canvas2, self.page6_plotframe2)
        self.page6_toolbar2.update()
        self.page6_canvas2.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.page6_b = self.page6_figure2.add_subplot(111)


if __name__ == '__main__':
    sys.setrecursionlimit(4000000)
    # sys.setrecursionlimit(20000)
    info = sys.version_info
    if info[0] < 3:
        sys.exit(f"Python {info[0]} is not supported.")

    # if info[1] == 8:
    #     sys.exit(f"Python 3.8 may cause a visualisation problems, we are sorry. Please use Python 3.7.*")
    #
    # if info[1] == 9:
    #     sys.exit(f"Python 3.9 was not tested and may cause errors. Please use Python 3.7.*")
    #
    # if info[1] != 7:
    #     sys.exit(f"Please python use Python 3.7.*")

    gui = Gui()
    ## System dependent fullscreen setting
    if "wind" in system().lower():
        gui.state('zoomed')
    else:
        gui.attributes('-zoomed', True)
    gui.autoload(True)

    gui.protocol('WM_DELETE_WINDOW', gui.ask_quit)
    gui.gui_init()

    gui.autoload()
    gui.mainloop()
