from tkinter import *
import webbrowser
import pickle
import os


class Gui:

    def __init__(self, root):
        root.title('mpm')
        frame = Frame(root, width=400, height=300)
        frame.pack()

        self.model = None
        self.property = None
        self.data = None
        self.space = None

        ## DESIGN

        ## MENU
        main_menu = Menu(root)
        root.config(menu=main_menu)

        ## FILE
        file_menu = Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="File", menu=file_menu)

        ## FILE/LOAD
        load_menu = Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Load", menu=load_menu, underline=0)
        load_menu.add_command(label="Load model", command=self.load_model)
        load_menu.add_command(label="Load property", command=self.load_property)
        load_menu.add_command(label="Load data", command=self.load_data)
        load_menu.add_command(label="Load space", command=self.load_space)
        file_menu.add_separator()

        ## FILE/SAVE
        save_menu = Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Save", menu=save_menu, underline=0)
        save_menu.add_command(label="Save model", command=self.save_model)
        save_menu.add_command(label="Save property", command=self.save_property)
        save_menu.add_command(label="Save data", command=self.save_data)
        save_menu.add_command(label="Save space", command=self.save_space)
        file_menu.add_separator()

        file_menu.add_command(label="Exit", command=frame.quit)

        ## EDIT
        edit_menu = Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Edit", menu=edit_menu)

        ## SHOW
        show_menu = Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Show", menu=show_menu)
        show_menu.add_command(label="Space", command=self.show_space)

        ## ANALYSIS
        analysis_menu = Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Synthesise parameters", command=self.synth_params)
        analysis_menu.add_command(label="Create intervals", command=self.create_intervals)
        analysis_menu.add_command(label="Sample space", command=self.sample_space)
        analysis_menu.add_command(label="Refine space", command=self.refine_space)

        ## SETTINGS
        settings_menu = Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Edit config", command=self.edit_config)

        ## HELP
        help_menu = Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Help", command=self.show_help)
        help_menu.add_command(label="Check for updates", command=self.checkupdates)
        help_menu.add_command(label="About", command=self.printabout)

        ## STATUS BAR
        self.status = Label(root, text="", bd=1, relief=SUNKEN, anchor=W)
        self.status.pack(side=BOTTOM, fill=X)

    ## LOGIC
    ## FILE
    def load_model(self):
        self.status_set("Please select the model to be loaded.")
        print("load model")
        ## TBD
        ## just load the path
        # self.model = filepath
        self.status_set("Model loaded.")

    def load_property(self):
        self.status_set("Please select the property to be loaded.")
        print("load model")
        ## TBD
        ## just load the path
        # self.property = filepath
        self.status_set("Property loaded.")

    def load_data(self):
        self.status_set("Please select the data to be loaded.")
        print("load model")
        ## TBD
        ## just load the path
        # self.data = filepath
        self.status_set("Data loaded.")

    def load_space(self):
        self.status_set("Please select the space to be loaded.")
        print("load space")
        ## TBD
        # self.space = pickle.load(open(filepath, "rb"))
        self.status_set("Space loaded")

    def save_model(self):
        self.status_set("Please select folder to store the model in.")
        print("save model")
        ## TBD
        if isinstance(self.model, os.path):
            print()
            ## os.copy the file
        else:
            with open(filepath) as file:
                for line in self.model:
                    file.write(line)
        self.status_set("Model saved.")

    def save_property(self):
        self.status_set("Please select folder to store the property in.")
        self.status_set("property saved")
        print("save_property")
        ## TBD
        if isinstance(self.model, os.path):
            print()
            ## os.copy the file
        else:
            with open(filepath) as file:
                for line in self.properties:
                    file.write(line)
        self.status_set("Property saved.")

    def save_data(self):
        self.status_set("Please select folder to store the data in.")
        print("save_data")
        ## TBD
        ## get the filename
        pickle.dump(self.data, open(filename, 'wb'))
        self.status_set("Data saved.")

    def save_space(self):
        self.status_set("Please select folder to store the space in.")
        print("save space")
        ## TBD
        ## get the filename
        pickle.dump(self.data, open(filename, 'wb'))
        self.status_set("Space saved.")

    ## EDIT

    ## SHOW
    def show_space(self):
        self.status_set("Please select which parts to be shown.")
        print("show_space")
        ## TBD
        # space.show(self, title="", green=True, red=True, sat_samples=False, unsat_samples=False, save=False)

    ## ANALYSIS
    def synth_params(self):
        self.status_set("Parameter synthesis running ...")
        print("synth params")
        ## TBD takes model, and prism/storm
        # mc_prism.call_prism(args, seq=False, silent=False, model_path=model_path, properties_path=properties_path,
        #                prism_output_path=prism_results, std_output_path=prism_results, std_output_file=False)
        self.status_set("Parameter synthesised. Output here: {}", [os.path.join(prism_results, filename)])

    def create_intervals(self):
        self.status_set("Intervals are being created ...")
        print("create_intervals")
        ## TBD, takes data, alpha, n_samples
        # load.create_intervals(alpha, n_samples, data)
        self.status_set("Intervals created.")

    def sample_space(self):
        self.status_set("Space sampling running ...")
        print("sample_space")
        ## TBD takes size_q (so far only grid sample implemented)
        # space.grid_sample(self, props, size_q, silent=False, save=False)
        self.status_set("Space sampling done.")

    def refine_space(self):
        self.status_set("Space refinement running ...")
        print("refine_space")
        ## TBD takes nothing
        # synthetise.check_deeper(region, props, n, epsilon, coverage, silent, version, size_q=False, debug=False, save=False, title="")
        self.status_set("Space refinement done.")

    ## SETTINGS
    def edit_config(self):
        print("edit config")
        ## TBD edit config
        self.status_set("Config file saved.")

    ## HELP
    def show_help(self):
        print("show_help")
        webbrowser.open_new("https://github.com/xhajnal/mpm#mpm")

    def checkupdates(self):
        print("check updates")
        webbrowser.open_new("https://github.com/xhajnal/mpm/releases")

    def printabout(self):
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
