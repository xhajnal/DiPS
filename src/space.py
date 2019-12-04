import socket
from collections import Iterable
from time import localtime, strftime
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy import prod
import copy
from termcolor import colored  ## Colored output

## ONLY FOR SAVING FILES
import os

from common.math import get_rectangle_volume
from sample_n_visualise import visualise_by_param  ## Multidimensional refinement proxy
from common.document_wrapper import DocumentWrapper  ## Text wrapper for figure tight layout
from common.config import load_config

spam = load_config()
results_dir = spam["results"]
refinement_results = spam["refinement_results"]
del spam


class RefinedSpace:
    """ Class to represent space. The space can be sampled or refined into sat(green), unsat(red), and unknown(white) regions

    Args:
        region (list of intervals): whole space
        params (list of strings):: parameter names
        types (list of string):: parameter types (Real, Int, Bool, ...)
        rectangles_sat (list of intervals): sat (green) space
        rectangles_unsat (list of intervals): unsat (red) space
        rectangles_unknown (list of intervals): unknown (white) space
        sat_samples: (list of points): satisfying points
        unsat_samples: (list of points): unsatisfying points
        true_point (list of numbers):: The true value in the parameter space
        title (string):: text to be added in the end of the Figure titles
    """

    def __init__(self, region, params, types=None, rectangles_sat=False, rectangles_unsat=False, rectangles_unknown=None, sat_samples=None, unsat_samples=None, true_point=False, title=False):
        """
        Args:
            region (list of intervals): whole space
            params (list of strings):: parameter names
            types (list of string):: parameter types (Real, Int, Bool, ...)
            rectangles_sat (list of intervals): sat (green) space
            rectangles_unsat (list of intervals): unsat (red) space
            rectangles_unknown (list of intervals): unknown (white) space
            sat_samples: (list of points): satisfying points
            unsat_samples: (list of points): unsatisfying points
            true_point (list of numbers):: The true value in the parameter space
            title (string):: text to added in the end of the Figure titles, CASE STUDY STANDARD: f"model: {model_type}, population = {population}, sample_size = {sample_size},  \n Dataset = {dataset}, alpha={alpha}, #samples={n_samples}"
        """

        ## REGION
        if not isinstance(region, Iterable):
            raise Exception("Given region is not iterable")
        if isinstance(region, tuple):
            self.region = list(region)
        else:
            self.region = region
        if not isinstance(self.region[0], Iterable):  ## Simple rectangle managing
            self.region = [self.region]
        ### Taking care of unchangeable tuples
        for interval_index in range(len(self.region)):
            if isinstance(region[interval_index], tuple):
                self.region[interval_index] = [self.region[interval_index][0], self.region[interval_index][1]]

        ## PARAMS
        for param in params:
            if param in ["region", "params", "types", "rectangles_sat", "rectangles_unsat", "rectangles_unknown", "sat_samples", "unsat_samples", "true_point"]:
                raise Exception(f"Parameter name {param} is not allowed as it is one of the variables")
            ## TODO add "delta"
            if param in ["funcs", "intervals", "silent", "debug", "constraints", "points", "solver", "recursion_depth",  "epsilon", "coverage", "silent", "version", "sample_size", "debug", "save", "title", "where", "show_space", "solver", "gui"]:
                raise Exception(f"Parameter name {param} is not allowed as it is one of the variables")
        self.params = params

        if not len(self.params) == len(self.region):
            if len(self.params) > len(self.region):
                print(colored(f"Number of parameters ({len(params)}) and dimension of the region ({len(region)}) is not equal", 'red'))
                raise Exception(f"Number of parameters ({len(params)}) and dimension of the region ({len(region)}) is not equal")
            else:
                print(colored(f" Warning: Number of parameters ({len(params)}) and dimension of the region ({len(region)}) is not equal", 'red'))

        if types is None or types is False:
            self.types = []
            ## IF no types are given
            for i in region:
                self.types.append("Real")
        else:
            self.types = types
            if not isinstance(types, Iterable):
                raise Exception("Given types is not iterable")
            if isinstance(types, tuple):
                self.types = [types]
            else:
                self.types = types

            if not len(self.types) == len(self.region):
                print(colored(
                    f"Number of types of parameters ({len(types)}) and dimension of the region ({len(region)}) is not equal",
                    'red'))
                raise Exception(
                    f"Number of types ({len(types)}) and dimension of the region ({len(region)}) is not equal")

        ## SAT RECTANGLES
        # print("rectangles_sat", rectangles_sat)
        if rectangles_sat is False:
            rectangles_sat = []
        if not isinstance(rectangles_sat, Iterable):
            raise Exception("Given rectangles_sat is not iterable")
        if isinstance(rectangles_sat, tuple):
            self.rectangles_sat = [rectangles_sat]
        elif rectangles_sat is False:
            self.rectangles_sat = []
        else:
            self.rectangles_sat = rectangles_sat
        self.rectangles_sat_to_show = copy.copy(self.rectangles_sat)

        ## UNSAT RECTANGLES
        if rectangles_unsat is False:
            rectangles_unsat = []
        # print("rectangles_unsat", rectangles_unsat)
        if not isinstance(rectangles_unsat, Iterable):
            raise Exception("Given rectangles_unsat is not iterable")
        if isinstance(rectangles_unsat, tuple):
            self.rectangles_unsat = [rectangles_unsat]
        elif rectangles_unsat is False:
            self.rectangles_unsat = []
        else:
            self.rectangles_unsat = rectangles_unsat
        self.rectangles_unsat_to_show = copy.copy(self.rectangles_unsat)

        ## UNKNOWN RECTANGLES
        # print("rectangles_unknown", rectangles_unknown)
        if rectangles_unknown is None:
            ## TODO THIS IS NOT CORRECT
            self.rectangles_unknown = [self.region]
        elif not isinstance(rectangles_unknown, Iterable):
            raise Exception("Given rectangles_unknown is not iterable")
        elif isinstance(rectangles_unknown, tuple):
            self.rectangles_unknown = [rectangles_unknown]
        elif rectangles_unknown is False:
            self.rectangles_unknown = []
        else:
            self.rectangles_unknown = rectangles_unknown

        ## SAT SAMPLES
        if sat_samples is None:
            self.sat_samples = []
        elif not isinstance(sat_samples, Iterable):
            raise Exception("Given samples are not iterable")
        else:
            # print("samples", samples)
            self.sat_samples = sat_samples

        ## UNSAT SAMPLES
        if unsat_samples is None:
            self.unsat_samples = []
        elif not isinstance(unsat_samples, Iterable):
            raise Exception("Given samples are not iterable")
        else:
            # print("samples", samples)
            self.unsat_samples = unsat_samples

        if (sat_samples is None) and (unsat_samples is None):
            self.gridsampled = True
        else:
            print("Not sure about the source of the sample points")
            self.gridsampled = False

        ## SET THE TRUE POINT
        if true_point:
            if len(true_point) is len(self.params):
                self.true_point = true_point
            else:
                raise Exception(f"The dimension of the given true point ({len(true_point)}) does not match")
        else:
            self.true_point = None

        ## SET TITLE SUFFIX
        if title:
            self.title = title
        else:
            self.title = ""

        ## INTERNAL VARIABLES
        self.time_last_sampling = 0
        self.time_sampling = 0
        self.time_last_refinement = 0
        self.time_refinement = 0

        ## TEXT WRAPPER
        self.wrapper = DocumentWrapper(width=70)

    def show(self, title="", green=True, red=True, sat_samples=False, unsat_samples=False, true_point=True, save=False, where=False, show_all=True):
        """ Visualises the space

        Args:
            title (string):  title of the figure
            green (bool): if True showing safe space
            red (bool): if True showing unsafe space
            sat_samples (bool): if True showing sat samples
            unsat_samples (bool): if True showing unsat samples
            true_point (bool): if True showing true point
            save (bool): if True, the output is saved
            where (tuple/list): output matplotlib sources to output created figure
            show_all (bool): if True, not only newly added rectangles are shown
        """

        # print("self.true_point", self.true_point)
        if save is True:
            save = str(strftime("%d-%b-%Y-%H:%M:%S", localtime()))+".png"

        if len(self.region) == 1 or len(self.region) == 2:
            # colored(globals()["default_region"], self.region)

            # from matplotlib import rcParams
            # rc('font', **{'family':'serif', 'serif':['Computer Modern Roman']})
            if where:
                fig = where[0]
                pic = where[1]
                plt.autoscale()
                pic.autoscale()
            else:
                fig = plt.figure()
                pic = fig.add_subplot(111, aspect='equal')
            pic.set_xlabel(self.params[0])

            ## Set axis ranges
            region = copy.deepcopy(self.region)
            if len(self.region) == 1:
                if region[0][1] - region[0][0] < 0.1:
                     region[0] = (region[0][0] - 0.2, region[0][1] + 0.2)
                pic.axis([region[0][0], region[0][1], 0, 1])
            max_region_size = region[0][1] - region[0][0]

            if len(self.region) == 2:
                pic.set_ylabel(self.params[1])
                # if region[1][1] - region[1][0] < 0.1:
                #     region[1] = (region[1][0] - 0.2, region[1][1] + 0.2)
                pic.axis([region[0][0], region[0][1], region[1][0], region[1][1]])
                max_region_size = max(max_region_size, region[1][1] - region[1][0])

            pretitle = ""
            if (green or red) and (sat_samples or unsat_samples):
                pretitle = pretitle + "Refinement and Sampling, \n red = unsafe region / unsat points, green = safe region / sat points, white = in between"
            else:
                if green or red:
                    pretitle = pretitle + "Refinement, \n red = unsafe region, green = safe region, white = in between"
                if sat_samples or unsat_samples:
                    pretitle = pretitle + "Samples, \n red = unsat points, green = sat points"
            if (green or red) and (self.rectangles_sat or self.rectangles_unsat) and show_all:
                pretitle = pretitle + f"\n Last refinement took {socket.gethostname()} {round(self.time_last_refinement, 2)} of {round(self.time_refinement, 2)} sec. whole time"
            if (sat_samples or unsat_samples) and (self.sat_samples or self.unsat_samples):
                pretitle = pretitle + f"\n Last sampling took {socket.gethostname()} {round(self.time_last_sampling, 2)} of {round(self.time_sampling, 2)} sec. whole sampling time"
            if pretitle:
                pretitle = pretitle + "\n"
            if green:
                pic.add_collection(self.show_green(show_all=show_all))
            if red:
                pic.add_collection(self.show_red(show_all=show_all))
            if sat_samples and self.sat_samples:
                pic.add_collection(self.show_samples(True))
            if unsat_samples and self.unsat_samples:
                pic.add_collection(self.show_samples(False))
            if self.true_point and true_point:
                # print(self.true_point)
                if (len(self.sat_samples) + len(self.unsat_samples)) == 0 or len(self.region) == 0:
                    size_correction = 0.01 * max_region_size
                else:
                    size_correction = min(1 / (len(self.sat_samples) + len(self.unsat_samples)) ** (1 / len(self.region)), 0.01)
                circle = plt.Circle((self.true_point[0], self.true_point[1]), size_correction*1, color='b', fill=False)
                if where:
                    where[1].add_artist(circle)
                else:
                    plt.gcf().gca().add_artist(circle)

            whole_title = "\n".join(self.wrapper.wrap(f"{pretitle} \n{self.title} \n {title}"))
            pic.set_title(whole_title)
            with open(os.path.join(refinement_results, "figure_to_title.txt"), "a+") as file:
                file.write(f"{save} : {whole_title}\n")

            ## Save the figure
            if save:
                plt.savefig(os.path.join(refinement_results, f"Refinement_{save}"), bbox_inches='tight')
                print("Figure stored here: ", os.path.join(refinement_results, f"Refinement_{save}"))
                with open(os.path.join(refinement_results, "figure_to_title.txt"), "a+") as file:
                    file.write(f"Refinement{save} : {whole_title}\n")
            if where:
                ## TODO probably yield
                # print("returning tuple")
                del region
                return fig, pic
            else:
                # plt.tight_layout()
                plt.show()
            del region

        else:
            print("Multidimensional space")
            ## Show only if sat_samples selected and either there are some unsat samples or the sampling was not grid
            if sat_samples and (not self.gridsampled or self.unsat_samples):
                if self.sat_samples:
                    if where:
                        fig = where[0]
                        ax = where[1]
                        plt.autoscale()
                        ax.autoscale()
                    else:
                        fig, ax = plt.subplots()
                    ## Creates values of the horizontal axis
                    x_axis = []
                    i = 0
                    for dimension in self.sat_samples[0]:
                        i = i + 1
                        x_axis.append(i)
                    if self.true_point and true_point:
                        ax.scatter(x_axis, self.true_point, marker='x', label="true_point")
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.plot(x_axis, self.true_point)
                        plt.legend(loc='upper right', numpoints=1, ncol=3, fontsize=8)

                    ## Get values of the vertical axis for respective line
                    for sample in self.sat_samples:
                        # print("samples", sample)
                        ax.scatter(x_axis, sample)
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.plot(x_axis, sample)
                    ax.set_xlabel("param indices")
                    ax.set_ylabel("parameter value")
                    whole_title = "\n".join(self.wrapper.wrap(f"Sat sample points of the given hyperspace: \nparam names: {self.params},\nparam types: {self.types}, \nboundaries: {self.region}, \n{self.title} \n {title}"))
                    ax.set_title(whole_title)
                    ax.autoscale()
                    ax.margins(0.1)

                    ## Save the figure
                    if save:
                        plt.savefig(os.path.join(refinement_results, f"Samples_sat_{save}"), bbox_inches='tight')
                        print("Figure stored here: ", os.path.join(refinement_results, f"Samples_sat_{save}"))
                        with open(os.path.join(refinement_results, "figure_to_title.txt"), "a+") as file:
                            file.write(f"Samples_sat{save} : {whole_title}\n")

                    if where:
                        ## TODO probably yield
                        print("returning tuple")
                        return fig, ax
                    else:
                        plt.show()
                else:
                    print("No sat samples so far, nothing to show")

            if sat_samples and self.gridsampled and not self.unsat_samples:
                print("Since no unsat samples, the whole grid of points are sat, not visualising this trivial case.")
                if where:
                    return None, "Since no unsat samples, the whole grid of points are sat, not visualising this trivial case."

            ## Show only if unsat_samples selected and either there are some sat samples or the sampling was not grid
            if unsat_samples and (not self.gridsampled or self.sat_samples):
                if self.unsat_samples:
                    fig, ax = plt.subplots()
                    # fig.tight_layout()
                    ## Creates values of the horizontal axis
                    x_axis = []
                    i = 0
                    for dimension in self.unsat_samples[0]:
                        i = i + 1
                        x_axis.append(i)

                    ## Get values of the vertical axis for respective line
                    for sample in self.unsat_samples:
                        # print("samples", sample)
                        ax.scatter(x_axis, sample)
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.plot(x_axis, sample)
                    ax.set_xlabel("param indices")
                    ax.set_ylabel("parameter value")
                    whole_title = "\n".join(self.wrapper.wrap(f"Unsat sample points of the given hyperspace: \nparam names: {self.params},\nparam types: {self.types}, \nboundaries: {self.region}, \n{self.title} \n{title}"))
                    ax.set_title(whole_title)
                    ax.autoscale()
                    ax.margins(0.1)

                    ## Save the figure
                    if save:
                        plt.savefig(os.path.join(refinement_results, f"Samples_unsat_{save}"), bbox_inches='tight')
                        print("Figure stored here: ", os.path.join(refinement_results, f"Samples_unsat_{save}"))
                        with open(os.path.join(refinement_results, "figure_to_title.txt"), "a+") as file:
                            file.write(f"Samples_unsat{save} : {whole_title}\n")
                    if where:
                        ## TODO probably yield
                        print("returning tuple")
                        return fig, ax
                    else:
                        plt.show()
                else:
                    print("No unsat samples so far, nothing to show")

            if unsat_samples and self.gridsampled and not self.sat_samples:
                print("Since no sat samples, the whole grid of points are unsat, not visualising this trivial case.")
                if where:
                    return None, "Since no sat samples, the whole grid of points are unsat, not visualising this trivial case."

            if red or green:
                if where:  ## Return the plot
                    if self.rectangles_sat:  ## If any rectangles to be visualised
                        fig = where[0]
                        ax = where[1]
                        plt.autoscale()
                        ax.autoscale()
                        title = "\n".join(self.wrapper.wrap(f"Refinement,\n Domains of respective parameter of safe subspace.\nLast refinement took {socket.gethostname()} {round(self.time_last_refinement, 2)} of {round(self.time_refinement, 2)} sec. whole time."))
                        fig, ax = visualise_by_param(self.rectangles_sat, title=title, where=[fig, ax])
                        return fig, ax
                    else:
                        return None, "While refining multidimensional space no green area found, no reasonable plot to be shown."
                else:
                    if self.rectangles_sat:
                        ## TODO maybe add the title
                        fig = visualise_by_param(self.rectangles_sat)
                        plt.show()
                    else:
                        print("No sat rectangles so far, nothing to show")

    def get_region(self):
        """ Returns whole domain """
        return self.region

    def get_params(self):
        """ Returns parameters """
        return self.params

    def get_green(self):
        """ Returns green (hyper)rectangles """
        return self.rectangles_sat

    def get_red(self):
        """ Returns red (hyper)rectangles """
        return self.rectangles_unsat

    def get_white(self):
        """ Returns white (hyper)rectangles """
        return self.rectangles_unknown

    def get_nonwhite(self):
        """ Returns nonwhite (hyper)rectangles """
        return self.rectangles_sat + self.rectangles_unsat

    def get_volume(self):
        """ Returns the volume of the space """
        intervals = []
        for interval in self.region:
            intervals.append(interval[1] - interval[0])
        return prod(intervals)

    def get_green_volume(self):
        """ Returns volume of green subspace """
        cumulative_volume = 0

        ## If there is no hyperrectangle in the sat space
        if not self.rectangles_sat:
            return 0.0

        for rectangle in self.rectangles_sat:
            cumulative_volume = cumulative_volume + get_rectangle_volume(rectangle)
        return cumulative_volume

    def get_red_volume(self):
        """ Returns volume of red subspace """
        cumulative_volume = 0

        ## If there is no hyperrectangle in the unsat space
        if not self.rectangles_unsat:
            return 0.0

        for rectangle in self.rectangles_unsat:
            cumulative_volume = cumulative_volume + get_rectangle_volume(rectangle)
        return cumulative_volume

    def get_white_volume(self):
        """ Returns volume of white subspace """
        return self.get_volume() - self.get_nonwhite_volume()

    def get_nonwhite_volume(self):
        """ Returns volume of nonwhite subspace """
        return self.get_green_volume() + self.get_red_volume()

    def get_coverage(self):
        """ Returns proportion of nonwhite subspace (coverage) """
        # print("self.get_nonwhite_volume()", self.get_nonwhite_volume())
        # print("self.get_volume()", self.get_volume())
        if self.get_nonwhite_volume() == 0:
            return 0
        else:
            return self.get_nonwhite_volume() / self.get_volume()

    def get_sat_samples(self):
        """ Returns green (sat) samples """
        return self.sat_samples

    def get_unsat_samples(self):
        """ Returns red (unsat) samples """
        return self.unsat_samples

    def get_all_samples(self):
        """ Returns all (sat and unsat) samples """
        return self.sat_samples + self.unsat_samples

    def get_true_point(self):
        """ Returns the true point """
        return self.true_point

    def add_green(self, green):
        """ Adds green (hyper)rectangle """
        self.rectangles_sat.append(green)
        self.rectangles_sat_to_show.append(green)

    def add_red(self, red):
        """ Adds red (hyper)rectangle """
        self.rectangles_unsat.append(red)
        self.rectangles_unsat_to_show.append(red)

    def add_white(self, white):
        """ Adds white (hyper)rectangle """
        self.rectangles_unknown.append(white)

    def add_sat_samples(self, sat_samples):
        """ Adds sat samples

        Args:
            sat_samples (list): of sat points
        """
        # print("sat_samples", sat_samples)
        self.sat_samples.extend(sat_samples)

    def add_unsat_samples(self, unsat_samples):
        """ Adds unsat samples

        Args:
            unsat_samples (list): of unsat points
        """
        # print("unsat_samples", unsat_samples)
        self.unsat_samples.extend(unsat_samples)

    def remove_green(self, green):
        """ Removes green (hyper)rectangle """
        self.rectangles_sat.remove(green)
        try:
            self.rectangles_sat_to_show.remove(green)
        except:
            pass

    def remove_red(self, red):
        """ Removes red (hyper)rectangle """
        self.rectangles_unsat.remove(red)
        try:
            self.rectangles_unsat_to_show.remove(red)
        except:
            pass

    def remove_white(self, white):
        """ Removes white (hyper)rectangle """
        try:
            self.rectangles_unknown.remove(white)
        except:
            print("Could not remove white area ", white)
            return False
        return True

    ## TODO generalise so that the code is not copied
    def show_green(self, show_all=True):
        """ Adds green (hyper)rectangles to be visualised

        Args:
            show_all (bool): if all, not only newly added rectangles are shown
        """
        rectangles_sat = []
        if len(self.region) > 2:
            print("Error while visualising", len(self.region), "dimensional space")
            return
        elif len(self.region) == 2:
            for rectangle in (self.rectangles_sat_to_show, self.rectangles_sat)[show_all]:
                ## (Rectangle((low_x,low_y), width, height, fc= color)
                rectangles_sat.append(Rectangle((rectangle[0][0], rectangle[1][0]), rectangle[0][1] - rectangle[0][0],
                                                rectangle[1][1] - rectangle[1][0], fc='g'))
        elif len(self.region) == 1:
            for rectangle in (self.rectangles_sat_to_show, self.rectangles_sat)[show_all]:
                ## (Rectangle((low_x,low_y), width, height, fc= color)
                rectangles_sat.append(
                    Rectangle((rectangle[0][0], 0.33), rectangle[0][1] - rectangle[0][0], 0.33, fc='g'))
        self.rectangles_sat_to_show = []
        return PatchCollection(rectangles_sat, facecolor='g', alpha=0.5)

    def show_red(self, show_all=True):
        """ Adds red (hyper)rectangles to be visualised

        Args:
            show_all (bool): if all, not only newly added rectangles are shown
        """
        rectangles_unsat = []
        if len(self.region) > 2:
            print("Error while visualising", len(self.region), "dimensional space")
            return
        elif len(self.region) == 2:
            for rectangle in (self.rectangles_unsat_to_show, self.rectangles_unsat)[show_all]:
                ## (Rectangle((low_x,low_y), width, height, fc= color)
                rectangles_unsat.append(Rectangle((rectangle[0][0], rectangle[1][0]), rectangle[0][1] - rectangle[0][0],
                                                  rectangle[1][1] - rectangle[1][0], fc='r'))
        elif len(self.region) == 1:
            for rectangle in (self.rectangles_unsat_to_show, self.rectangles_unsat)[show_all]:
                ## (Rectangle((low_x,low_y), width, height, fc= color)
                rectangles_unsat.append(
                    Rectangle((rectangle[0][0], 0.33), rectangle[0][1] - rectangle[0][0], 0.33, fc='r'))
        self.rectangles_unsat_to_show = []
        return PatchCollection(rectangles_unsat, facecolor='r', alpha=0.5)

    def grid_sample(self, constraints, sample_size, silent: bool = False, save=False, progress=False):
        """ Executes grid sampling

        Args:
            constraints  (list of strings): array of properties
            sample_size (int): number of samples in dimension
            silent (bool): if silent printed output is set to minimum
            save (bool): if True output is pickled
            progress (Tkinter element): progress bar
        """
        from synthetise import sample
        self.gridsampled = True
        sample(self, constraints, sample_size, compress=True, silent=silent, save=save, progress=progress)

    def show_samples(self, which):
        """ Visualises samples in 2D """
        if not (self.sat_samples or self.unsat_samples):
            return None

        samples = []
        if len(self.region) > 2:
            print("Error while visualising", len(self.region), "dimensional space")
            return None
        elif len(self.region) == 2:
            # print("samples", self.samples)
            try:
                x_size = self.region[0][1] - self.region[0][0]
                y_size = self.region[1][1] - self.region[1][0]
                x_size_correction = min(1 / (len(self.sat_samples) + len(self.unsat_samples)) ** (1 / len(self.region)), 0.01 * x_size)
                y_size_correction = min(1 / (len(self.sat_samples) + len(self.unsat_samples)) ** (1 / len(self.region)), 0.01 * y_size)

            except:
                print("len(self.sat_samples)", len(self.sat_samples))
                print("len(self.unsat_samples)", len(self.unsat_samples))
                print("len(self.region)", len(self.region))
                raise Exception()
            ## CHOOSING SAT OR UNSAT
            if which:
                for rectangle in self.sat_samples:
                    ## (Rectangle((low_x,low_y), width, height, fc= color)
                    # print("rectangle", rectangle)
                    samples.append(Rectangle((rectangle[0]-0.005*x_size, rectangle[1]-0.005*y_size), x_size_correction, y_size_correction, fc='r'))
                return PatchCollection(samples, facecolor='g', alpha=0.5)
            else:
                for rectangle in self.unsat_samples:
                    ## (Rectangle((low_x,low_y), width, height, fc= color)
                    # print("rectangle", rectangle)
                    samples.append(Rectangle((rectangle[0]-0.005*x_size, rectangle[1]-0.005*y_size), x_size_correction, y_size_correction, fc='r'))
                return PatchCollection(samples, facecolor='r', alpha=0.5)

    def nice_print(self, full_print=False):
        """ Returns the class in a human readable format

        Args:
            full_print (bool): if True not truncated print is used
        """
        spam = str(f"params: {self.params}\n")
        spam = spam + str(f"region: {self.region}\n")
        spam = spam + str(f"types: {self.types}\n")
        spam = spam + str(f"rectangles_sat: {(f'{self.rectangles_sat[:5]} ... {len(self.rectangles_sat)-5} more', self.rectangles_sat)[len(self.rectangles_sat) <= 30 or full_print]} \n")
        spam = spam + str(f"rectangles_unsat: {(f'{self.rectangles_unsat[:5]} ... {len(self.rectangles_unsat)-5} more', self.rectangles_unsat)[len(self.rectangles_unsat) <= 30 or full_print]} \n")
        spam = spam + str(f"rectangles_unknown: {(f'{self.rectangles_unknown[:5]} ... {len(self.rectangles_unknown)-5} more', self.rectangles_unknown)[len(self.rectangles_unknown) <= 30 or full_print]} \n")
        spam = spam + str(f"sat_samples: {(f'{self.sat_samples[:5]} ... {len(self.sat_samples)-5} more', self.sat_samples)[len(self.sat_samples) <= 30 or full_print]} \n")
        spam = spam + str(f"unsat_samples: {(f'{self.unsat_samples[:5]} ... {len(self.unsat_samples)-5} more', self.unsat_samples)[len(self.unsat_samples) <= 30 or full_print]} \n")
        spam = spam + str(f"true_point: {self.true_point}\n")
        return spam

    def sampling_took(self, time):
        """ Manages the time the sampling took

        Args:
            time (number): adds the time of the last sampling
        """
        self.time_last_sampling = time
        self.time_sampling = self.time_sampling + self.time_last_sampling

    def refinement_took(self, time):
        """ Manages the time the refinement took

        Args:
            time (number): adds the time of the last refinement
        """
        self.time_last_refinement = time
        self.time_refinement = self.time_refinement + self.time_last_refinement

    def __repr__(self):
        return str([self.region, self.params, self.types, self.rectangles_sat, self.rectangles_unsat,
                    self.rectangles_unknown, self.sat_samples, self.unsat_samples, self.true_point])

    def __str__(self):
        return str([self.region, self.params, self.types, self.rectangles_sat, self.rectangles_unsat,
                    self.rectangles_unknown, self.sat_samples, self.unsat_samples, self.true_point])
