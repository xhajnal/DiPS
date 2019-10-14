import platform
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

import configparser

config = configparser.ConfigParser()
workspace = os.path.dirname(__file__)
sys.path.append(workspace)

from mc_informed import general_create_data_informed_properties
from load import create_intervals, load_all_functions, find_param, load_data
import space
from synthetise import ineq_to_props, check_deeper
from mc_prism import call_prism_files, call_storm_files
from sample_n_visualise import sample_list_funs, eval_and_show, get_param_values, heatmap

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
        self.load_config()  ## Load the config file

        ## Files
        self.model_file = StringVar()  ## Model file
        self.property_file = StringVar()  ## Property file
        self.data_informed_property_file = StringVar()  ## Data informed property file
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
        # self.model = ""
        # self.property = ""
        self.data = ""
        self.data_informed_property = ""  ## Property containing the interval boundaries from the data
        self.functions = ""  ## Parameter synthesis results (rational functions)
        self.intervals = ""  ## Computed intervals
        self.parameters = ""  ##  Parsed parameters
        self.parameter_intervals = []  ## Parameters intervals
        self.props = ""  ## Derived properties
        self.space = ""  ## Instance of a space Class
        ## Space visualisation settings
        self.show_samples = None
        self.show_refinement = None
        self.show_true_point = None

        ## Settings
        self.version = "1.5.0"  ## Version of the gui
        self.silent = BooleanVar()  ## Sets the command line output to minimum
        self.debug = False  ## Sets the command line output to maximum

        ## Settings/data
        # self.alpha = ""  ## Confidence
        # self.n_samples = ""  ## Number of samples
        self.program = StringVar()  ## "prism"/"storm"
        self.max_depth = ""  ## Max recursion depth
        self.coverage = ""  ## Coverage threshold
        self.epsilon = ""  ## Rectangle size threshold
        self.alg = ""  ## Refinement alg. number

        self.factor = BooleanVar()  ## Flag for factorising rational functions
        self.size_q = ""  ## Number of samples
        self.save = ""  ## True if saving on

        ## OTHER SETTINGS
        self.button_pressed = BooleanVar()  ## Inner variable to close created window

        ## GUI INIT
        self.title('Mpm')
        self.minsize(1000, 300)

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

        Label(frame, text=f"Props file:", anchor=W, justify=LEFT).grid(row=4, column=0, sticky=W, padx=4)
        self.props_label = Label(frame, textvariable=self.props_file, anchor=W, justify=LEFT)
        self.props_label.grid(row=4, column=1, sticky=W, padx=4)

        Label(frame, text=f"Space file:", anchor=W, justify=LEFT).grid(row=5, column=0, sticky=W, padx=4)
        self.space_label = Label(frame, textvariable=self.space_file, anchor=W, justify=LEFT)
        self.space_label.grid(row=5, column=1, sticky=W, padx=4)

        show_print_checkbutton = Checkbutton(frame, text="Hide print in command line", variable=self.silent)
        show_print_checkbutton.grid(row=5, column=9, sticky=E, padx=4)
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
        Button(page2, text='Open Prism/Storm output file', command=self.load_functions).grid(row=3, column=1, sticky=W, pady=4)

        Label(page2, text=f"Loaded Prism/Storm output file:", anchor=W, justify=LEFT).grid(row=4, column=0, sticky=W, padx=4, pady=4)

        self.functions_text = scrolledtext.ScrolledText(page2, height=100, state=DISABLED)
        self.functions_text.grid(row=5, column=0, columnspan=16, rowspan=2, sticky=W, padx=4, pady=4)

        Label(page2, text=f"Rational functions section.", anchor=W, justify=LEFT).grid(row=1, column=17, sticky=W, padx=4, pady=4)
        Button(page2, text='Open functions', command=self.load_parsed_functions).grid(row=3, column=17, sticky=W, padx=4, pady=4)
        Button(page2, text='Save functions', command=self.save_parsed_functions).grid(row=3, column=18, sticky=W, pady=4)

        Label(page2, text=f"Parsed function(s):", anchor=W, justify=LEFT).grid(row=4, column=17, sticky=W, padx=4, pady=4)
        self.functions_parsed_text = scrolledtext.ScrolledText(page2, height=100, state=DISABLED)
        self.functions_parsed_text.grid(row=5, column=17, columnspan=16, rowspan=2, sticky=W, pady=4)


        ## TAB SAMPLE AND VISUALISE
        self.page3 = ttk.Frame(nb, width=400, height=200, name="sampling")
        nb.add(self.page3, text='Sample functions')

        self.page3.rowconfigure(5, weight=1)
        self.page3.columnconfigure(6, weight=1)

        Label(self.page3, text="Set number of samples per variable (grid size):", anchor=W, justify=LEFT).grid(row=1, column=0, padx=4, pady=4)
        self.fun_size_q_entry = Entry(self.page3)
        self.fun_size_q_entry.grid(row=1, column=1)

        Button(self.page3, text='Sample functions', command=self.sample_fun).grid(row=2, column=0, sticky=W, padx=4, pady=4)

        Label(self.page3, text=f"Values of sampled points:", anchor=W, justify=LEFT).grid(row=3, column=0, sticky=W, padx=4, pady=4)

        self.sampled_functions_text = scrolledtext.ScrolledText(self.page3, height=100, state=DISABLED)
        self.sampled_functions_text.grid(row=4, column=0, columnspan=16, rowspan=2, sticky=W, padx=4, pady=4)

        Label(self.page3, text=f"Rational functions visualisation", anchor=W, justify=CENTER).grid(row=1, column=17, columnspan=3, pady=4)
        Button(self.page3, text='Plot functions in a given point', command=self.show_funs_in_single_point).grid(row=2, column=17, sticky=W, padx=4, pady=4)
        Button(self.page3, text='Plot all sampled points', command=self.show_funs_in_all_points).grid(row=2, column=18, sticky=W, padx=4, pady=4)
        Button(self.page3, text='Heatmap', command=self.show_heatmap).grid(row=2, column=19, sticky=W, padx=4, pady=4)
        self.Next_sample_button = Button(self.page3, text="Next plot", state="disabled",
                                         command=lambda: self.button_pressed.set(True))
        self.Next_sample_button.grid(row=3, column=18, sticky=W, padx=4, pady=4)

        self.page3_figure = None
        # self.page3_figure = pyplt.figure()
        # self.page3_a = self.page3_figure.add_subplot(111)
        # print("type a", type(self.a))

        self.page3_figure_in_use = StringVar()
        self.page3_figure_in_use.set("")


        ## TAB DATA
        page4 = ttk.Frame(nb, width=400, height=200, name="data")
        nb.add(page4, text='Data & Intervals')
        # page4.columnconfigure(0, weight=1)
        # page4.rowconfigure(2, weight=1)
        # page4.rowconfigure(7, weight=1)

        Button(page4, text='Open data file', command=self.load_data).grid(row=0, column=0, sticky=W, padx=4, pady=4)

        label10 = Label(page4, text=f"Data:", anchor=W, justify=LEFT)
        label10.grid(row=1, column=0, sticky=W, padx=4, pady=4)
        createToolTip(label10, text='For each rational function exactly one data point should be assigned.')

        self.data_text = Text(page4, height=12)  # , height=10, width=30
        ## self.data_text.bind("<FocusOut>", self.parse_data)
        # self.data_text = Text(page4, height=12, state=DISABLED)  # , height=10, width=30
        # self.data_text.config(state="disabled")
        self.data_text.grid(row=2, column=0, columnspan=2, sticky=W, padx=4, pady=4)

        ## SET THE INTERVAL COMPUTATION SETTINGS
        label42 = Label(page4, text="Set alpha, the confidence:", anchor=W, justify=LEFT)
        label42.grid(row=3)
        createToolTip(label42, text='confidence')
        label43 = Label(page4, text="Set n_samples, number of samples: ", anchor=W, justify=LEFT)
        label43.grid(row=4)
        createToolTip(label43, text='number of samples')

        self.alpha_entry = Entry(page4)
        self.n_samples_entry = Entry(page4)

        self.alpha_entry.grid(row=3, column=1)
        self.n_samples_entry.grid(row=4, column=1)

        self.alpha_entry.insert(END, '0.95')
        self.n_samples_entry.insert(END, '100')

        Button(page4, text='Create intervals', command=self.create_intervals).grid(row=5, column=0, sticky=W, padx=4, pady=4)

        Label(page4, text=f"Intervals:", anchor=W, justify=LEFT).grid(row=6, column=0, sticky=W, padx=4, pady=4)

        self.intervals_text = Text(page4, height=12, state=DISABLED)  # height=10, width=30
        # self.interval_text.config(state="disabled")
        self.intervals_text.grid(row=7, column=0, columnspan=2, sticky=W, padx=4, pady=4)

        ttk.Separator(page4, orient=VERTICAL).grid(row=0, column=11, rowspan=10, sticky='ns', padx=50, pady=10)
        Label(page4, text=f"Data informed property section.", anchor=W, justify=LEFT).grid(row=0, column=12, sticky=W, padx=5, pady=4)
        Label(page4, text=f"Loaded property file:", anchor=W, justify=LEFT).grid(row=1, column=12, sticky=W, padx=5, pady=4)

        self.property_text2 = scrolledtext.ScrolledText(page4, height=4, state=DISABLED)
        # self.property_text2.config(state="disabled")
        self.property_text2.grid(row=2, column=12, columnspan=16, rowspan=2, sticky=W + E + N + S, padx=5, pady=4)
        Button(page4, text='Generate data informed properties', command=self.generate_data_informed_properties).grid(row=5, column=12, sticky=W, padx=5, pady=4)

        self.data_informed_property_text = scrolledtext.ScrolledText(page4, height=4, state=DISABLED)
        self.data_informed_property_text.grid(row=6, column=12, columnspan=16, rowspan=2, sticky=W + E + N + S, padx=5, pady=4)

        Button(page4, text='Save data informed properties', command=self.save_data_informed_properties).grid(row=9, column=12, sticky=W, padx=5, pady=4)


        ## TAB PROPS
        page5 = ttk.Frame(nb, width=400, height=200, name="props")
        nb.add(page5, text='Props')

        page5.rowconfigure(2, weight=1)
        page5.columnconfigure(16, weight=1)

        Button(page5, text='Recalculate props', command=self.recalculate_props).grid(row=0, column=0, sticky=W, padx=4, pady=4)

        self.props_text = scrolledtext.ScrolledText(page5, height=100, state=DISABLED)
        self.props_text.grid(row=1, column=0, columnspan=16, rowspan=2, sticky=W, padx=4,
                             pady=4)  # pack(anchor=W, fill=X)

        Label(page5, text=f"Import/Export:", anchor=W, justify=LEFT).grid(row=3, column=0, sticky=W, padx=4, pady=4)
        Button(page5, text='Open props', command=self.load_props).grid(row=3, column=1, sticky=W, pady=4)
        Button(page5, text='Append props', command=self.append_props).grid(row=3, column=2, sticky=W, pady=4)
        Button(page5, text='Save props', command=self.save_props).grid(row=3, column=3, sticky=W, pady=4)


        ## TAB SAMPLE AND REFINEMENT
        page6 = ttk.Frame(nb, width=400, height=200, name="refine")
        nb.add(page6, text='Sample & Refine space')

        frame_left = Frame(page6, width=200, height=200)
        frame_left.pack(side=LEFT)

        # Button(frame_left, text='Create space', command=self.validate_space).grid(row=0, column=0, sticky=W, padx=4, pady=4)

        ttk.Separator(frame_left, orient=HORIZONTAL).grid(row=1, column=0, columnspan=15, sticky='nwe', padx=10, pady=8)

        label61 = Label(frame_left, text="Set size_q: ", anchor=W, justify=LEFT)
        label61.grid(row=1, pady=16)
        createToolTip(label61, text='number of samples per dimension')

        self.size_q_entry = Entry(frame_left)
        self.size_q_entry.grid(row=1, column=1, columnspan=2)
        self.size_q_entry.insert(END, '5')

        Button(frame_left, text='Sample space', command=self.sample_space).grid(row=7, column=0, sticky=W, padx=10, pady=4)

        ttk.Separator(frame_left, orient=VERTICAL).grid(column=3, row=1, rowspan=8, sticky='ns', padx=10, pady=25)

        label62 = Label(frame_left, text="Set max_dept: ", anchor=W, justify=LEFT)
        label62.grid(row=1, column=4, padx=10)
        createToolTip(label62, text='Maximal number of splits')
        label63 = Label(frame_left, text="Set coverage: ", anchor=W, justify=LEFT)
        label63.grid(row=2, column=4, padx=10)
        createToolTip(label63, text='Proportion of the nonwhite area to be reached')
        label64 = Label(frame_left, text="Set epsilon: ", anchor=W, justify=LEFT)
        label64.grid(row=3, column=4, padx=10)
        createToolTip(label64,
                      text='Minimal size of the rectangle to be checked (if 0 all rectangles are being checked)')
        label65 = Label(frame_left, text="Set algorithm: ", anchor=W, justify=LEFT)
        label65.grid(row=4, column=4, padx=10)
        createToolTip(label65, text='Choose from algorithms:\n 1-4 - using SMT solvers \n 1 - DFS search \n 2 - BFS search \n 3 - BFS search with example propagation \n 4 - BFS with example and counterexample propagation \n 5 - interval algorithmic')

        label66 = Label(frame_left, text="Set solver: ", anchor=W, justify=LEFT)
        label66.grid(row=5, column=4, padx=10)
        createToolTip(label66, text='When using SMT solver (alg 1-4), two options are possible, z3 or dreal (with delta complete decision procedures)')

        label67 = Label(frame_left, text="Set delta: ", anchor=W, justify=LEFT)
        label67.grid(row=6, column=4, padx=10)
        createToolTip(label67, text='When using dreal solver, delta is used to set solver error boundaries for satisfiability.')

        self.max_dept_entry = Entry(frame_left)
        self.coverage_entry = Entry(frame_left)
        self.epsilon_entry = Entry(frame_left)
        self.alg = ttk.Combobox(frame_left, values=('1', '2', '3', '4', '5'))
        self.solver = ttk.Combobox(frame_left, values=('z3', 'dreal'))
        self.delta_entry = Entry(frame_left)

        self.max_dept_entry.grid(row=1, column=5)
        self.coverage_entry.grid(row=2, column=5)
        self.epsilon_entry.grid(row=3, column=5)
        self.alg.grid(row=4, column=5)
        self.solver.grid(row=5, column=5)
        self.delta_entry.grid(row=6, column=5)

        self.max_dept_entry.insert(END, '5')
        self.coverage_entry.insert(END, '0.95')
        self.epsilon_entry.insert(END, '0')
        self.alg.current(3)
        self.solver.current(0)
        self.delta_entry.insert(END, '0.01')

        Button(frame_left, text='Refine space', command=self.refine_space).grid(row=7, column=4, sticky=W, pady=4, padx=10)

        ttk.Separator(frame_left, orient=HORIZONTAL).grid(row=8, column=0, columnspan=15, sticky='nwe', padx=10, pady=4)

        frame_left.rowconfigure(13, weight=1)
        frame_left.columnconfigure(16, weight=1)

        self.space_text = scrolledtext.ScrolledText(frame_left, height=100, state=DISABLED)
        self.space_text.grid(row=12, column=0, columnspan=16, rowspan=2, sticky=W, pady=4)  # pack(anchor=W, fill=X)

        frame_right = Frame(page6, width=200, height=200)
        frame_right.pack(side=TOP)

        Button(frame_right, text='Edit True point', command=self.edit_true_point).pack(pady=10)

        Label(frame_right, text=f"Space Visualisation", anchor=W, justify=CENTER).pack(side=TOP)
        self.page6_plotframe = Frame(frame_right)
        self.page6_plotframe.pack(fill=BOTH, pady=10)
        self.page6_figure = pyplt.figure()
        self.page6_figure.tight_layout()  ## By huypn

        self.page6_canvas = FigureCanvasTkAgg(self.page6_figure, master=self.page6_plotframe)  # A tk.DrawingArea.
        self.page6_canvas.draw()
        self.page6_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.page6_toolbar = NavigationToolbar2Tk(self.page6_canvas, self.page6_plotframe)
        self.page6_toolbar.update()
        self.page6_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.page6_a = self.page6_figure.add_subplot(111)

        Button(frame_left, text='Open space', command=self.load_space).grid(row=14, column=2, sticky=S, padx=4, pady=4)
        Button(frame_left, text='Save space', command=self.save_space).grid(row=14, column=3, sticky=S, padx=4, pady=4)
        Button(frame_left, text='Delete space', command=self.refresh_space).grid(row=14, column=4, sticky=S, padx=4, pady=4)

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
    ## FILE - LOAD, PARSE, SHOW, AND SAVE
    def load_model(self):
        """ Loads model from a text file. """
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

    def load_property(self):
        """ Loads temporal properties from a text file. """
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

    def load_functions(self, file=False):
        """ Loads parameter synthesis output text file

        Args
        -------------
        file (Path/String): direct path to load the function file
        """
        print("Loading functions ...")

        if self.functions_changed:
            if not askyesno("Loading functions", "Previously obtained functions will be lost. Do you want to proceed?"):
                return

        self.status_set("Loading functions - checking inputs")

        if not self.silent.get():
            print("Used program: " , self.program.get())
        if self.program.get() == "prism":
            initial_dir = self.prism_results
        elif self.program.get() == "storm":
            initial_dir = self.storm_results
        else:
            messagebox.showwarning("Load functions", "Select a program for which you want to load data.")
            return

        ## If file to load is NOT preselected
        # print(file)
        if not file:
            self.status_set("Please select the prism/storm symbolic results to be loaded.")
            spam = filedialog.askopenfilename(initialdir=initial_dir, title="Rational functions loading - Select file",
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

        if self.debug:
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

        ## Show loaded functions
        self.functions_text.configure(state='normal')
        self.functions_text.delete('1.0', END)
        self.functions_text.insert('1.0', open(self.functions_file.get(), 'r').read())
        self.functions_text.configure(state='disabled')
        ## Resetting parsed intervals
        self.parameter_intervals = []

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
            self.functions_parsed_text.configure(state='disabled')

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

    def load_parsed_functions(self):
        """ Loads parsed rational functions from a pickled file. """
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
            self.functions = True
            self.functions_file.set(spam)

            if ".p" in self.functions_file.get():
                self.functions = pickle.load(open(self.functions_file.get(), "rb"))

            functions = ""
            for function in self.functions:
                functions = f"({functions},\n{function}"
            functions = functions[2:]

            self.functions_parsed_text.configure(state='normal')
            self.functions_parsed_text.delete('1.0', END)
            self.functions_parsed_text.insert('end', functions)
            self.functions_parsed_text.configure(state='disabled')

            ## Resetting parsed intervals
            self.parameter_intervals = []
            self.status_set("Parsed rational functions loaded.")

    def load_data(self):
        """ Loads data from a file. Either pickled list or comma separated values in one line"""
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
                self.data = load_data(self.data_file.get(), silent=self.silent.get(), debug=not self.silent.get())
                if not self.data:
                    messagebox.showerror("Loading data", f"Error, No data loaded.")
                    self.status_set("Data not loaded properly.")
                    return
                self.unfold_data()
            if not self.silent.get():
                print("Loaded data: ", self.data)
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

    def recalculate_props(self):
        """ Merges rational functions and intervals into props. Shows it afterwards. """
        ## If there is some props
        if len(self.props_text.get('1.0', END)) > 1:
            proceed = messagebox.askyesno("Recalculate props",
                                          "Previously obtained props will be lost. Do you want to proceed?")
        else:
            proceed = True
        if proceed:
            self.props = ""
            self.validate_props(position="Props")
        self.status_set("Props recalculated and shown.")

    def load_props(self, append=False):
        """ Loads props from a pickled file. """
        print("Loading props ...")

        if self.props_changed and not append:
            if not askyesno("Loading props", "Previously obtained props will be lost. Do you want to proceed?"):
                return
        self.status_set("Please select the props to be loaded.")
        spam = filedialog.askopenfilename(initialdir=self.data_dir, title="Props loading - Select file",
                                          filetypes=(("text files", "*.p"), ("all files", "*.*")))

        if not self.silent.get():
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
                    self.props = []
                spam = pickle.load(open(self.props_file.get(), "rb"))
                self.props.extend(spam)
            else:
                self.props = pickle.load(open(self.props_file.get(), "rb"))

                # self.props = []
                #
                # with open(self.props_file.get(), 'r') as file:
                #     for line in file:
                #         print(line[:-1])
                #         self.props.append(line[:-1])
            if not self.silent.get():
                print("self.props", self.props)

            props = ""
            for prop in self.props:
                props = f"{prop},\n{props}"
            props = props[:-2]

            self.props_text.configure(state='normal')
            self.props_text.delete('1.0', END)
            self.props_text.insert('end', props)
            self.props_text.configure(state='disabled')

            ## Resetting parsed intervals
            self.parameter_intervals = []
            self.status_set("Props loaded.")

    def append_props(self):
        """ Appends loaded props from a pickled file to previously obtained props. """
        self.load_props(append=True)
        self.status_set("Props appended.")

    def load_space(self):
        """ Loads space from a pickled file. """
        print("Loading space ...")

        if self.space_changed:
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
            self.show_samples = messagebox.askyesno("Loaded space", "Do you want to visualise samples?")
            self.show_refinement = messagebox.askyesno("Loaded space", "Do you want to visualise refinement (safe & unsafe regions)?")
            if self.space.true_point is not None:
                self.show_true_point = messagebox.askyesno("Loaded space", "Do you want to show the true point?")
            else:
                self.show_true_point = False
            self.show_space(self.show_refinement, self.show_samples, self.show_true_point)

            self.space_changed = True
            self.status_set("Space loaded.")

    def print_space(self, clear=False):
        """ Print the niceprint of the space into space text window. """
        if not self.space == "":
            if not self.silent.get() and not clear:
                print("space", self.space)
                print()
                print("space nice print \n", self.space.nice_print())

            self.space_text.configure(state='normal')
            self.space_text.delete('1.0', END)
            if not clear:
                self.space_text.insert('end', self.space.nice_print())
            self.space_text.configure(state='disabled')

    def show_space(self, show_refinement, show_samples, show_true_point, clear=False):
        """ Visualises the space in the plot. """
        if not self.space == "":
            if not clear:
                figure, axis = self.space.show(green=show_refinement, red=show_refinement, sat_samples=show_samples,
                                               unsat_samples=show_samples, true_point=show_true_point, save=False,
                                               where=[self.page6_figure, self.page6_a])

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
            self.new_window = Toplevel(self)
            label = Label(self.new_window,
                          text="Please choose values of the parameters to be used:")
            label.grid(row=0)

            i = 1
            ## For each param create an entry
            self.parameter_values = []
            for param in self.space.params:
                Label(self.new_window, text=param, anchor=W, justify=LEFT).grid(row=i, column=0)
                spam = Entry(self.new_window)
                spam.grid(row=i, column=1)
                spam.insert(END, '0')
                self.parameter_values.append(spam)
                i = i + 1

            ## To be used to wait until the button is pressed
            self.button_pressed.set(False)
            load_true_point_button = Button(self.new_window, text="OK", command=self.load_true_point_from_window)
            load_true_point_button.grid(row=i)
            load_true_point_button.focus()
            load_true_point_button.bind('<Return>', self.load_true_point_from_window)

            load_true_point_button.wait_variable(self.button_pressed)

            self.print_space()
            self.show_space(self.show_refinement, self.show_samples, True)

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

    def save_model(self):
        """ Saves obtained model as a file. """
        ## TODO CHECK IF THE MODEL IS NON EMPTY
        # if len(self.model_text.get('1.0', END)) <= 1:
        #    self.status_set("There is no model to be saved.")
        #    return

        print("Saving the model ...")
        self.status_set("Please select folder to store the model in.")
        save_model_file = filedialog.asksaveasfilename(initialdir=self.model_dir, title="Model saving - Select file",
                                                       filetypes=(("pm files", "*.pm"), ("all files", "*.*")))
        if save_model_file == "":
            self.status_set("No file selected.")
            return

        if "." not in save_model_file:
            save_model_file = save_model_file + ".pm"
        # print("save_model_file", save_model_file)

        with open(save_model_file, "w") as file:
            file.write(self.model_text.get(1.0, END))

        self.status_set("Model saved.")

    def save_property(self):
        """ Saves obtained temporal properties as a file. """
        print("Saving the property ...")
        ## TODO CHECK IF THE PROPERTY IS NON EMPTY
        # if len(self.property_text.get('1.0', END)) <= 1:
        #    self.status_set("There is no property to be saved.")
        #    return

        self.status_set("Please select folder to store the property in.")
        save_property_file = filedialog.asksaveasfilename(initialdir=self.property_dir,
                                                          title="Property saving - Select file",
                                                          filetypes=(("pctl files", "*.pctl"), ("all files", "*.*")))
        if save_property_file == "":
            self.status_set("No file selected.")
            return

        if "." not in save_property_file:
            save_property_file = save_property_file + ".pctl"
        # print("save_property_file", save_property_file)

        with open(save_property_file, "w") as file:
            file.write(self.property_text.get(1.0, END))

        self.status_set("Property saved.")

    def generate_data_informed_properties(self):
        """ Generates Data informed property from temporal properties and data. Prints it. """
        if self.property_file.get() is "":
            messagebox.showwarning("Data informed property generation", "No property file loaded.")
            return False

        if self.intervals == "":
            print("Intervals not computed, properties cannot be generated")
            messagebox.showwarning("Data informed property generation", "Compute intervals first.")
            return False

        # general_create_data_informed_properties(prop_file, intervals, output_file=False)
        self.data_informed_property = general_create_data_informed_properties(self.property_file.get(), self.intervals, silent=self.silent.get())
        self.data_informed_property_text.configure(state='normal')
        self.data_informed_property_text.delete('1.0', END)
        spam = ""
        for item in self.data_informed_property:
            spam = spam + str(item) + ",\n"
        self.data_informed_property_text.insert('end', spam)
        self.data_informed_property_text.configure(state='disabled')
        # TODO

    def save_data_informed_properties(self):
        """ Saves computed data informed property as a text file. """
        print("Saving data informed property ...")
        ## TODO CHECK IF THE PROPERTY IS NON EMPTY
        # if len(self.property_text.get('1.0', END)) <= 1:
        #    self.status_set("There is no property to be saved.")
        #    return

        self.status_set("Please select folder to store data informed property in.")
        save_data_informed_property_file = filedialog.asksaveasfilename(initialdir=self.property_dir,
                                                                        title="Data informed property saving - Select file",
                                                                        filetypes=(
                                                                        ("pctl files", "*.pctl"), ("all files", "*.*")))
        if save_data_informed_property_file == "":
            self.status_set("No file selected.")
            return

        if "." not in save_data_informed_property_file:
            save_data_informed_property_file = save_data_informed_property_file + ".pctl"
        # print("save_property_file", save_property_file)

        with open(save_data_informed_property_file, "w") as file:
            file.write(self.data_informed_property_text.get('1.0', END))

        self.status_set("Data informed property saved.")

    ## TODO MAYBE IN THE FUTURE
    def save_functions(self):
        """ Saves parsed functions as a pickled file. """
        print("Saving the rational functions ...")

        if self.functions is "":
            self.status_set("There are no rational functions to be saved.")
            return

            ## TODO choose to save rewards or normal functions

        self.status_set("Please select folder to store the rational functions in.")
        if self.program is "prism":
            save_functions_file = filedialog.asksaveasfilename(initialdir=self.prism_results,
                                                               title="Rational functions saving - Select file",
                                                               filetypes=(
                                                               ("pickle files", "*.p"), ("all files", "*.*")))
        elif self.program is "storm":
            save_functions_file = filedialog.asksaveasfilename(initialdir=self.storm_results,
                                                               title="Rational functions saving - Select file",
                                                               filetypes=(
                                                               ("pickle files", "*.p"), ("all files", "*.*")))
        else:
            self.status_set("Error - Selected program not recognised.")
            save_functions_file = "Error - Selected program not recognised."
        if not self.silent.get():
            print("Saving functions in file: ", save_functions_file)

        if save_functions_file == "":
            self.status_set("No file selected.")
            return

        with open(save_functions_file, "w") as file:
            for line in self.props:
                file.write(line)
        self.status_set("Rational functions saved.")

    def save_parsed_functions(self):
        """ Saves parsed rational functions as a pickled file. """
        print("Saving the parsed functions ...")
        if self.functions is "":
            self.status_set("There is no functions to be saved.")
            return

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

        if "." not in save_functions_file:
            save_functions_file = save_functions_file + ".p"

        if not self.silent.get():
            print("Saving parsed functions as a file:", save_functions_file)

        pickle.dump(self.functions, open(save_functions_file, 'wb'))
        self.status_set("Parsed functions saved.")

    def save_data(self):
        """Saves data as a pickled file. """
        print("Saving the data ...")
        if self.data is "":
            self.status_set("There is no data to be saved.")
            return

        self.status_set("Please select folder to store the data in.")
        save_data_file = filedialog.asksaveasfilename(initialdir=self.data_dir, title="Data saving - Select file",
                                                      filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
        if "." not in save_data_file:
            save_data_file = save_data_file + ".p"

        if not self.silent.get():
            print("Saving data as a file:", save_data_file)

        pickle.dump(self.data, open(save_data_file, 'wb'))
        self.status_set("Data saved.")

    def save_props(self):
        """ Saves props as a pickled file. """
        print("Saving the props ...")
        if self.props is "":
            self.status_set("There is no props to be saved.")
            return

        self.status_set("Please select folder to store the props in.")
        save_props_file = filedialog.asksaveasfilename(initialdir=self.data_dir, title="Props saving - Select file",
                                                       filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
        if "." not in save_props_file:
            save_props_file = save_props_file + ".p"

        if not self.silent.get():
            print("Saving props as a file:", save_props_file)

        pickle.dump(self.props, open(save_props_file, 'wb'))
        self.status_set("Props saved.")

    def save_space(self):
        """ Saves space as a pickled file. """
        print("Saving the space ...")
        if self.space is "":
            self.status_set("There is no space to be saved.")
            return
        self.status_set("Please select folder to store the space in.")
        save_space_file = filedialog.asksaveasfilename(initialdir=self.data_dir, title="Space saving - Select file",
                                                       filetypes=(("pickle files", "*.p"), ("all files", "*.*")))
        if "." not in save_space_file:
            save_space_file = save_space_file + ".p"

        if not self.silent.get():
            print("Saving space as a file:", save_space_file)

        pickle.dump(self.space, open(save_space_file, 'wb'))
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
                    self.load_functions(self.functions_file.get())

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
                    self.load_functions(self.functions_file.get())
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
            self.parameter_intervals = []
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

        ## TODO If self.functions got more than one entry
        try:
            self.cursor_toggle_busy(True)
            self.sampled_functions = sample_list_funs(self.functions, int(self.fun_size_q_entry.get()),
                                                intervals=self.parameter_intervals, debug=False, silent=self.silent.get())
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
        self.sampled_functions_text.configure(state='disabled')
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

        ## TODO Maybe rewrite this as key and pass the argument to load_param_intervals
        self.key = StringVar()
        self.status_set("Choosing parameters value:")
        self.new_window = Toplevel(self)
        label = Label(self.new_window,
                      text="Please choose value of respective parameter of the synthesised function(s):")
        label.grid(row=0)
        self.key.set(" ")

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
        load_param_values_button = Button(self.new_window, text="OK", command=self.load_param_values_from_window)
        load_param_values_button.focus()
        load_param_values_button.bind('<Return>', self.load_param_values_from_window)
        load_param_values_button.grid(row=i)

        ## Waiting for the pop-up window closing
        load_param_values_button.wait_variable(self.button_pressed)
        ## print("key pressed")

        self.reinitialise_plot()

        ## TODO If self.functions got more than one entry
        ## Getting the plot values instead of the plot itself

        #     self.initialise_plot(what=self.page3_figure, where=self.page3_plotframe)
        # else:
        #     pyplt.close()
        #     self.page3_figure = pyplt.figure()
        #     self.page3_a = self.page3_figure.add_subplot(111)
        spam, egg = eval_and_show(self.functions, self.parameter_values, give_back=True, debug=False,
                                  where=[self.page3_figure, self.page3_a])

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
        print("Ploting sampled rational functions ...")
        self.status_set("Ploting sampled rational functions.")

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

            spam, egg = eval_and_show(self.functions, parameter_point, give_back=True,
                                      where=[self.page3_figure, self.page3_a])

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
        self.Next_sample_button.config(state="disabled")
        self.status_set("Ploting sampled rational functions finished.")

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
            self.page3_figure = heatmap(function, self.parameter_intervals,
                                        [int(self.fun_size_q_entry.get()), int(self.fun_size_q_entry.get())],
                                        posttitle=f"Function number {i}: {function}", where=True,
                                        parameters=self.parameters)
            self.initialise_plot(what=self.page3_figure)

            self.Next_sample_button.wait_variable(self.button_pressed)
        self.Next_sample_button.config(state="disabled")
        # self.page3_figure_locked.set(False)
        # self.update()
        self.status_set("Ploting sampled rational functions finished.")

    def create_intervals(self):
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
        self.intervals = create_intervals(float(self.alpha_entry.get()), float(self.n_samples_entry.get()), self.data)

        intervals = ""
        if not self.silent.get():
            print("Created intervals", self.intervals)
        for interval in self.intervals:
            intervals = f"{intervals},\n({interval.inf}, {interval.sup})"
        # print("intervals", intervals)
        intervals = intervals[2:]
        self.intervals_text.configure(state='normal')
        self.intervals_text.delete('1.0', END)
        self.intervals_text.insert('end', intervals)
        self.intervals_text.configure(state='disabled')
        self.status_set("Intervals created.")

        self.intervals_changed = True

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

        if self.props == "":
            messagebox.showwarning("Sample space", "Load or calculate props before refinement.")
            return

        ## Check space
        if not self.validate_space("Sample Space"):
            return

        self.status_set("Space sampling is running ...")
        if not self.silent.get():
            print("space.params", self.space.params)
            print("props", self.props)
            print("size_q", self.size_q)

        try:
            self.cursor_toggle_busy(True)
            self.space.sample(self.props, self.size_q, silent=self.silent.get(), save=False)
        finally:
            self.cursor_toggle_busy(False)

        self.print_space()

        self.show_space(show_refinement=False, show_samples=True, show_true_point=self.show_true_point)

        self.space_changed = False
        self.props_changed = False
        self.status_set("Space sampling finished.")

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

        if self.props == "":
            messagebox.showwarning("Refine space", "Load or calculate props before refinement.")
            return

        if not self.validate_space("Refine Space"):
            return

        self.status_set("Space refinement is running ...")
        # print(colored(f"self.space, {self.space.nice_print()}]", "blue"))
        try:
            self.cursor_toggle_busy(True)
            ## RETURNS TUPLE -- (SPACE,(NONE, ERROR TEXT)) or (SPACE, )
            spam = check_deeper(self.space, self.props, self.max_depth, self.epsilon, self.coverage,
                                silent=self.silent.get(),
                                version=int(self.alg.get()), size_q=False, debug=False, save=False,
                                title="", where=[self.page6_figure, self.page6_a], solver=str(self.solver.get()), delta=self.delta)
        finally:
            self.cursor_toggle_busy(False)
        ## If the visualisation of the space did not succeed
        if isinstance(spam, tuple):
            self.space = spam[0]
            messagebox.showinfo("Space refinement", spam[1])
        else:
            self.space = spam
            self.page6_figure.tight_layout()  ## By huypn
            self.page6_figure.canvas.draw()
            self.page6_figure.canvas.flush_events()

        self.print_space()

        self.props_changed = False
        self.space_changed = False
        self.status_set("Space refinement finished.")

    ## VALIDATE VARIABLES (PARAMETERS, PROPS, SPACE)
    def validate_parameters(self, where, intervals=True):
        """ Validates (functions, props, and space) parameters.

        Args
        ------
        where (struct): a structure pars parameters from (e.g. self.functions)
        intervals (Bool): whether to check also parameter intervals
        """
        if not self.parameters:
            globals()["parameters"] = set()
            for polynome in where:
                globals()["parameters"].update(find_param(polynome))
            globals()["parameters"] = sorted(list(globals()["parameters"]))
            self.parameters = globals()["parameters"]
            if not self.silent.get():
                print("parameters", self.parameters)

        if (not self.parameter_intervals) and intervals:
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
                self.parameter_intervals.append([spam_low, spam_high])
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
        elif (len(self.parameter_intervals) is not len(self.parameters)) and intervals:
            self.parameter_intervals = []
            self.validate_parameters(where=where)

    def validate_props(self, position=False):
        """ Validates created properties.

        Args:
        ------
        position: (String) Name of the place from which is being called e.g. "Refine Space"/"Sample space"
        """
        print("Validating props ...")
        ## MAYBE an error here
        if not self.props == "":
            print("Props not empty, not checking them.")
            return True
        if position is False:
            position = "Validating props"
        ## If props empty create props
        if self.functions_changed or self.intervals_changed:
            if not self.silent.get():
                print("Functions: ", self.functions)
                print("Intervals: ", self.intervals)
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

            ## Check if the number of functions and intervals is equal
            if len(self.functions) != len(self.intervals):
                messagebox.showerror(position,
                                     "The number of rational functions and data points (or intervals) is not equal")
                return

            ## Create props
            self.props = ineq_to_props(self.functions, self.intervals, silent=self.silent.get())
            self.props_changed = True
            self.props_file.set("")

            props = ""
            for prop in self.props:
                props = f"{props},\n{prop}"
            props = props[2:]
            self.props_text.configure(state='normal')
            self.props_text.delete('1.0', END)
            self.props_text.insert('end', props)
            self.props_text.configure(state='disabled')
            if not self.silent.get():
                print("Props: ", self.props)
        return True

    def refresh_space(self):
        """ Unloads space. """
        if self.space_changed:
            if not askyesno("Sample & Refine", "Data of the space, its text representation, and the plot will be lost. Do you want to proceed?"):
                return
        self.space_changed = False
        self.print_space(clear=True)
        self.show_space(None, None, None, clear=True)
        self.space_file.set("")
        self.space = ""
        self.status_set("Space deleted.")

    def validate_space(self, position=False):
        """ Validates space.

        Args:
        ------
        position: (String) Name of the place from which is being called e.g. "Refine Space"/"Sample space"
        """
        print("Checking space ...")
        if position is False:
            position = "Validating space"
        ## If the space is empty create a new one
        if self.space == "":
            if not self.silent.get():
                print("Space is empty - creating a new one.")
            ## Parse params and its intervals
            self.validate_parameters(where=self.props)
            self.space = space.RefinedSpace(self.parameter_intervals, self.parameters)
        else:
            if self.props_changed:
                messagebox.showwarning(position,
                                       "Using previously created space with new props. Consider using fresh new space.")
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

    ## GUI MENU FUNCTIONS
    def edit_config(self):
        """ Opens config file in editor """
        print("Editing config ...")
        if "wind" in platform.system().lower():
            ## TODO TEST THIS ON WINDOWS
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

    ## INNER FUNCTIONS
    def load_param_intervals_from_window(self):
        """ Inner function to parse the param intervals from created window """
        region = []
        for param_index in range(len(self.parameters)):
            ## Getting the values from each entry, low = [0], high = [1]
            region.append([float(self.parameter_intervals[param_index][0].get()),
                           float(self.parameter_intervals[param_index][1].get())])
        if not self.silent.get():
            print("Region: ", region)
        del self.key
        self.new_window.destroy()
        del self.new_window
        self.parameter_intervals = region
        self.button_pressed.set(True)
        if not self.silent.get():
            if self.space:
                print("Space: ", self.space)

    def load_true_point_from_window(self):
        """ Inner function to parse the true point from created window """
        true_point = []
        for param_index in range(len(self.space.params)):
            ## Getting the values from each entry
            true_point.append(float(self.parameter_values[param_index].get()))
        if not self.silent.get():
            print("True point set to: ", true_point)
        # del self.key
        self.new_window.destroy()
        del self.new_window
        self.space.true_point = true_point

        print(self.space.nice_print())
        self.button_pressed.set(True)

    def load_param_values_from_window(self):
        """ Inner function to parse the param values from created window"""
        for param_index in range(len(self.parameter_values)):
            ## Getting the values from each entry, low = [0], high = [1]
            self.parameter_values[param_index] = float(self.parameter_values[param_index].get())
        del self.key
        self.new_window.destroy()
        del self.new_window
        self.button_pressed.set(True)
        ## print("self.parameter_values", self.parameter_values)

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
        self.page3_figure = pyplt.figure()
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
        self.page3_plotframe.grid(row=5, column=17, columnspan=3, sticky=W, padx=4, pady=4)

        self.page3_canvas = FigureCanvasTkAgg(what, master=self.page3_plotframe)
        self.page3_canvas.draw()
        self.page3_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.page3_toolbar = NavigationToolbar2Tk(self.page3_canvas, self.page3_plotframe)
        self.page3_toolbar.update()
        self.page3_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)


def quit_gui():
    gui.quit()


gui = Gui()
## System dependent fullscreen setting
if "wind" in platform.system().lower():
    gui.state('zoomed')
else:
    gui.attributes('-zoomed', True)

gui.protocol('WM_DELETE_WINDOW', quit_gui)
gui.mainloop()
