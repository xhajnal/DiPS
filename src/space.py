import os
import socket
import copy
import numpy as np
from matplotlib import colors, patches
from numpy import prod
from collections.abc import Iterable
from time import localtime, strftime
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from termcolor import colored  ## Colored output

## Importing my code
from common.mathematics import get_rectangle_volume
from sample_n_visualise import visualise_by_param  ## Multidimensional refinement proxy
from common.document_wrapper import DocumentWrapper  ## Text wrapper for figure tight layout
from common.config import load_config

spam = load_config()
results_dir = spam["results"]
refinement_results = spam["refinement_results"]
del spam


class RefinedSpace:
    """ Class to represent space. The space can be sampled or refined into sat(green), unsat(red), and unknown(white) regions.
    (hyper)rectangles is a list of intervals, point is a list of numbers

    Args:
        region (list of intervals or tuple of intervals): whole space
        params (list of strings): parameter names
        types (list of string): parameter types (Real, Int, Bool, ...)
        rectangles_sat (list of (hyper)rectangles): sat (green) space
        rectangles_unsat (list of (hyper)rectangles): unsat (red) space
        rectangles_unknown (list of (hyper)rectangles): unknown (white) space
        sat_samples: (list of points): satisfying points
        unsat_samples: (list of points): unsatisfying points
        true_point (point): The true value in the parameter space
        title (string): text to added in the end of the Figure titles, CASE STUDY STANDARD: f"model: {model_type}, population = {population}, sample_size = {sample_size},  \n Dataset = {dataset}, alpha={alpha}, #samples={n_samples}"
    """

    def __init__(self, region, params, types=None, rectangles_sat=False, rectangles_unsat=False, rectangles_unknown=None, sat_samples=None, unsat_samples=None, dist_samples=False, true_point=False, title=False, prefer_unsafe=False):
        """ (hyper)rectangles is a list of intervals, point is a list of numbers
        Args:
            region (list of intervals or tuple of intervals): whole space
            params (list of strings): parameter names
            types (list of string, None, or False): parameter types (Real, Int, Bool, ...), None or False means all Real
            rectangles_sat (list of (hyper)rectangles): sat (green) space
            rectangles_unsat (list of (hyper)rectangles): unsat (red) space
            rectangles_unknown (list of (hyper)rectangles): unknown (white) space
            sat_samples (list of points): satisfying points
            unsat_samples (list of points): unsatisfying points
            dist_samples (dictionary): points in param space to distance of not satisfying the constraints
            true_point (point): The true value in the parameter space
            title (string): text to added in the end of the Figure titles, CASE STUDY STANDARD: f"model: {model_type}, population = {population}, sample_size = {sample_size},  \n Dataset = {dataset}, alpha={alpha}, #samples={n_samples}"
            prefer_unsafe: if True unsafe space is shown in multidimensional space instead of safe

        """

        ## REGION
        if not (isinstance(region, list) or isinstance(region, tuple)):
            raise Exception("Given region is not iterable", region, type(region))
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
            elif isinstance(types, list):
                self.types = types
            else:
                raise Exception("Space - Could not parse types: ", types)

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
        elif isinstance(rectangles_sat, list):
            self.rectangles_sat = rectangles_sat
        else:
            raise Exception("Space - Could not parse rectangles_sat:", rectangles_sat)
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
        elif isinstance(rectangles_unsat, list):
            self.rectangles_unsat = rectangles_unsat
        else:
            raise Exception("Space - Could not parse rectangles_unsat:", rectangles_unsat)
        self.rectangles_unsat_to_show = copy.copy(self.rectangles_unsat)

        ## UNKNOWN RECTANGLES
        # print("rectangles_unknown", rectangles_unknown)
        if rectangles_unknown is None:
            self.rectangles_unknown = {get_rectangle_volume(self.region): [self.region]}
        elif not isinstance(rectangles_unknown, Iterable):
            raise Exception("Given rectangles_unknown is not iterable")
        elif isinstance(rectangles_unknown, dict):
            self.rectangles_unknown = rectangles_unknown
        elif isinstance(rectangles_unknown, tuple):
            if not isinstance(rectangles_unknown[0], Iterable):
                rectangles_unknown = [rectangles_unknown]
            self.rectangles_unknown = dict()
            for rectangle in rectangles_unknown:
                volume = get_rectangle_volume(rectangle)
                if volume in self.rectangles_unknown.keys():
                    self.rectangles_unknown[volume].append(rectangle)
                else:
                    self.rectangles_unknown[volume] = [rectangle]
        elif rectangles_unknown is False:
            self.rectangles_unknown = dict()
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

        ## SET DIST POINTS
        if dist_samples is False:
            self.dist_samples = {}
        else:
            self.dist_samples = dist_samples

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

        self.prefer_unsafe = prefer_unsafe

        self.true_point_object = []

        ## TEXT WRAPPER
        self.wrapper = DocumentWrapper(width=70)

    def show(self, title="", green=True, red=True, sat_samples=False, unsat_samples=False, quantitative=False,
             true_point=True, save=False, where=False, show_all=True, prefer_unsafe=None, hide_legend=False, hide_title=False):
        """ Visualises the space

        Args:
            title (string):  title of the figure
            green (bool): if True showing safe space
            red (bool): if True showing unsafe space
            sat_samples (bool): if True showing sat samples
            unsat_samples (bool): if True showing unsat samples
            quantitative (bool): if True show sampling with how far is the point from satisfying / not satisfying the constraints
            true_point (bool): if True showing true point
            save (bool): if True, the output is saved
            where (tuple/list): output matplotlib sources to output created figure
            show_all (bool): if True, not only newly added rectangles are shown
            prefer_unsafe: if True unsafe space is shown in multidimensional space instead of safe
        """
        if prefer_unsafe is None:
            try:  ## Backward compatibility
                prefer_unsafe = self.prefer_unsafe
            except AttributeError:
                self.prefer_unsafe = False
        else:
            self.prefer_unsafe = prefer_unsafe

        if save is True:
            save = str(strftime("%d-%b-%Y-%H-%M-%S", localtime()))+".png"

        legend_objects, legend_labels = [], []

        if true_point:
            self.show_true_point(where=where, is_inside_of_show=True)
            legend_objects.append(plt.scatter([], [], facecolor='white', edgecolor='blue', label="true_point"))
            legend_labels.append("true point")

        if len(self.region) == 1 or len(self.region) == 2:
            # colored(globals()["default_region"], self.region)

            # from matplotlib import rcParams
            # rc('font', **{'family':'serif', 'serif':['Computer Modern Roman']})
            if where:
                fig = where[0]
                axes = where[1]
                # pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
                plt.autoscale()
                axes.autoscale()
            else:
                fig = plt.figure()
                axes = fig.add_subplot(111, aspect='equal')
                # fig.
            axes.set_xlabel(self.params[0])

            ## Set axis ranges
            region = copy.deepcopy(self.region)
            if len(self.region) == 1:
                if region[0][1] - region[0][0] < 0.1:
                    region[0] = (region[0][0] - 0.2, region[0][1] + 0.2)
                axes.axis([region[0][0], region[0][1], -0.5, 0.5])
            max_region_size = region[0][1] - region[0][0]

            if len(self.region) == 2:
                axes.set_ylabel(self.params[1])
                # if region[1][1] - region[1][0] < 0.1:
                #     region[1] = (region[1][0] - 0.2, region[1][1] + 0.2)
                axes.axis([region[0][0], region[0][1], region[1][0], region[1][1]])
                max_region_size = max(max_region_size, region[1][1] - region[1][0])

            pretitle = ""
            if (green or red) and (sat_samples or unsat_samples):
                pretitle = pretitle + "Refinement and Sampling,"  #\n red = unsafe region / unsat points, green = safe region / sat points, white = in between"
            elif green or red:
                pretitle = pretitle + "Refinement,"  #\n red = unsafe region, green = safe region, white = in between"
            elif sat_samples or unsat_samples:
                pretitle = pretitle + "Samples,"  # \n red = unsat points, green = sat points"
            elif quantitative:
                pretitle = pretitle + "Quantitative samples, \n Sum of L1 distances to dissatisfy constraints. \n The greener the point is the further it is from the threshold \n where it stops to satisfy constraints. \n  Note that green point can be unsat and vice versa."
            if (sat_samples or unsat_samples) and (self.sat_samples or self.unsat_samples):
                pretitle = pretitle + f"\n Last sampling took {socket.gethostname()} {round(self.time_last_sampling, 2)} of {round(self.time_sampling, 2)} sec. whole sampling time"
            if (green or red) and (self.rectangles_sat or self.rectangles_unsat):
                pretitle = pretitle + f"\n Last refinement took {socket.gethostname()} {round(self.time_last_refinement, 2)} of {round(self.time_refinement, 2)} sec. whole refinement time"
            if pretitle:
                pretitle = pretitle + "\n"

            if not quantitative:
                if len(self.region) == 1:
                    ## show 1D space sampling
                    if sat_samples and self.sat_samples:
                        axes.scatter(np.array(list(self.sat_samples)), np.zeros(len(self.sat_samples)), c="green", alpha=0.5, label="sat")
                        legend_objects.append(plt.scatter([], [], c="green", alpha=0.5))
                        legend_labels.append("sat")
                    if unsat_samples and self.unsat_samples:
                        axes.scatter(np.array(list(self.unsat_samples)), np.zeros(len(self.unsat_samples)), c="red", alpha=0.5, label="unsat")
                        legend_objects.append(plt.scatter([], [], c="red", alpha=0.5))
                        legend_labels.append("unsat")
                elif len(self.region) == 2:
                    ## show 2D space sampling
                    if sat_samples and self.sat_samples:
                        axes.scatter(np.array(list(self.sat_samples))[:, 0], np.array(list(self.sat_samples))[:, 1], c="green", alpha=0.5, label="sat")
                        legend_objects.append(plt.scatter([], [], c="green", alpha=0.5))
                        legend_labels.append("sat")
                    if unsat_samples and self.unsat_samples:
                        axes.scatter(np.array(list(self.unsat_samples))[:, 0], np.array(list(self.unsat_samples))[:, 1], c="red", alpha=0.5, label="unsat")
                        legend_objects.append(plt.scatter([], [], c="red", alpha=0.5))
                        legend_labels.append("unsat")
                if green:
                    axes.add_collection(self.show_green(show_all=show_all))
                    legend_objects.append(patches.Patch(color='green', alpha=0.5))
                    legend_labels.append("safe")
                if red:
                    axes.add_collection(self.show_red(show_all=show_all))
                    legend_objects.append(patches.Patch(color='red', alpha=0.5))
                    legend_labels.append("unsafe")
                if red or green:
                    legend_objects.append(patches.Patch(facecolor='white', edgecolor='black'))
                    legend_labels.append("unknown")
                # axes.legend(legend_objects, legend_labels, bbox_to_anchor=(0, 1), loc='lower left', fontsize='small', frameon=False)
                if not hide_legend:
                    axes.legend(legend_objects, legend_labels, loc='upper left', fontsize='small')

            else:
                ## Show quantitative space sampling
                ## Get min, max sat degreeunsaunsa
                min_value = round(min(self.dist_samples.values()), 16)
                max_value = round(max(self.dist_samples.values()), 16)
                ## Setup colour normalisation
                if min_value < 0 < max_value:
                    divnorm = colors.DivergingNorm(vmin=min_value, vcenter=0., vmax=max_value)
                elif min_value > 0:
                    divnorm = colors.DivergingNorm(vmin=-1, vcenter=0., vmax=max_value)
                else:
                    divnorm = colors.DivergingNorm(vmin=min_value, vcenter=0., vmax=1)

                if len(self.region) == 1:
                    ## Show 1D quantitative space sampling
                    if where:
                        spam = axes.scatter(np.array(list(self.dist_samples.keys())), np.zeros(len(self.dist_samples.keys())), c=list(self.dist_samples.values()), cmap='RdYlGn', norm=divnorm)
                        cbar = fig.colorbar(spam, ax=axes)
                    else:
                        plt.scatter(np.array(list(self.dist_samples.keys())), np.zeros(len(self.dist_samples.keys())), c=list(self.dist_samples.values()), cmap='RdYlGn', norm=divnorm)
                        cbar = plt.colorbar()
                elif len(self.region) == 2:
                    ## show 2D quantitative sampling
                    if where:
                        spam = axes.scatter(np.array(list(self.dist_samples.keys()))[:, 0], np.array(list(self.dist_samples.keys()))[:, 1], c=list(self.dist_samples.values()), cmap='RdYlGn', norm=divnorm)
                        cbar = fig.colorbar(spam, ax=axes)
                    else:
                        plt.scatter(np.array(self.dist_samples.keys())[:, 0], np.array(self.dist_samples.keys())[:, 1], c=self.dist_samples.values(), cmap='RdYlGn', norm=divnorm)
                        cbar = plt.colorbar()
                cbar.set_label('Sum of L1 distances to dissatisfy constraints.')

            whole_title = "\n".join(self.wrapper.wrap(f"{pretitle} \n{self.title} \n {title}"))
            if not hide_title:
                axes.set_title(whole_title)

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
                return fig, axes
            else:
                # plt.tight_layout()
                plt.show()
            del region

        else:
            print("Multidimensional space")
            ## Sampling multidim plotting
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
                    x_axis = list(range(1, len(self.sat_samples[0])+1))

                    ## Get values of the vertical axis for respective line
                    for sample in self.sat_samples:
                        # print("samples", sample)
                        ax.scatter(x_axis, sample)
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.plot(x_axis, sample)
                    ax.set_xlabel("param indices")
                    ax.set_ylabel("parameter value")
                    whole_title = "\n".join(self.wrapper.wrap(f"Sat sample points of the given hyperspace: \nparam names: {self.params},\nparam types: {self.types}, \nboundaries: {self.region}. Last sampling took {socket.gethostname()} {round(self.time_last_sampling, 2)} of {round(self.time_sampling, 2)} sec. whole time. \n{self.title} \n {title}"))
                    if not hide_title:
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

            if sat_samples and self.gridsampled and not self.unsat_samples and not (green or red):
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
                    whole_title = "\n".join(self.wrapper.wrap(f"Unsat sample points of the given hyperspace: \nparam names: {self.params},\nparam types: {self.types}, \nboundaries: {self.region}. Last sampling took {socket.gethostname()} {round(self.time_last_sampling, 2)} of {round(self.time_sampling, 2)} sec. whole time. \n{self.title} \n{title}"))
                    if not hide_title:
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

            if unsat_samples and self.gridsampled and not self.sat_samples and not (green or red):
                print("Since no sat samples, the whole grid of points are unsat, not visualising this trivial case.")
                if where:
                    return None, "Since no sat samples, the whole grid of points are unsat, not visualising this trivial case."

            ## Quantitative multidim plotting
            if quantitative:
                if self.dist_samples:
                    if where:
                        fig = where[0]
                        ax = where[1]
                        plt.autoscale()
                        ax.autoscale()
                    else:
                        fig, ax = plt.subplots()
                    ## Creates values of the horizontal axis
                    x_axis = list(range(1, len(self.params)+1))
                    ## Get min, max sat degree
                    min_value = round(min(self.dist_samples.values()), 16)
                    max_value = round(max(self.dist_samples.values()), 16)
                    if min_value == max_value:
                        if max_value < 0:
                            max_value = 0.9 * max_value
                        else:
                            max_value = 1.1 * max_value
                    ## Setup colour normalisation
                    if min_value < 0 < max_value:
                        divnorm = colors.DivergingNorm(vmin=min_value, vcenter=0., vmax=max_value)
                    elif min_value > 0:
                        divnorm = colors.DivergingNorm(vmin=-1, vcenter=0., vmax=max_value)
                    else:
                        divnorm = colors.DivergingNorm(vmin=min_value, vcenter=0., vmax=1)
                    cmap = plt.cm.get_cmap("RdYlGn")

                    ## Get values of the vertical axis for respective line
                    for index, sample in enumerate(list(self.dist_samples.keys())):
                        ## COLOR = cmap(divnorm(0.5)
                        ax.scatter(x_axis, sample, color=cmap(divnorm(list(self.dist_samples.values()))[index]))
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.plot(x_axis, sample, color=cmap(divnorm(list(self.dist_samples.values()))[index]))

                    spam = ax.scatter([], [], c=[], cmap='RdYlGn', norm=divnorm)
                    cbar = fig.colorbar(spam, ax=ax)
                    cbar.set_label('Sum of L1 distances to dissatisfy constraints.')
                    ax.set_xlabel("param indices")
                    ax.set_ylabel("parameter value")
                    whole_title = "\n".join(self.wrapper.wrap(f"Quantitative samples, \n Sum of L1 distances to dissatisfy constraints. \n The greener the point is the further it is from the threshold \n where it stops to satisfy constraints. \n  Note that green point can be unsat and vice versa. \n{self.title} \n {title}"))
                    if not hide_title:
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
                        return fig, ax
                    else:
                        plt.show()
                else:
                    print("No quantitative samples so far, nothing to show")

            ## Refinement multidim plotting
            if red or green:
                if where:  ## Return the plot
                    if not prefer_unsafe:
                        if self.rectangles_sat:  ## If any rectangles to be visualised
                            fig = where[0]
                            ax = where[1]
                            ax.clear()
                            ## TODO check why this is before plotting
                            plt.autoscale()
                            ax.autoscale()
                            title = "\n".join(self.wrapper.wrap(f"Refinement,\n Domains of respective parameter of safe subspace.\nparam names: {self.params}\nparam types: {self.types}\nboundaries: {self.region}\nachieved coverage: {self.get_coverage()}.\nLast refinement took {socket.gethostname()} {round(self.time_last_refinement, 2)} of {round(self.time_refinement, 2)} sec. whole time."))
                            fig, ax = visualise_by_param(self.rectangles_sat, title=title, where=[fig, ax])
                            if true_point:
                                self.show_true_point(where=[fig, ax], is_inside_of_show=True, hide_legend=hide_legend)
                            return fig, ax
                        else:
                            return None, "While refining multidimensional space no green area found, no reasonable plot to be shown."
                    else:
                        if self.rectangles_unsat:  ## If any rectangles to be visualised
                            fig = where[0]
                            ax = where[1]
                            ax.clear()
                            ## TODO check why this is before plotting
                            plt.autoscale()
                            ax.autoscale()
                            title = "\n".join(self.wrapper.wrap(f"Refinement,\n Domains of respective parameter of unsafe subspace.\nparam names: {self.params}\nparam types: {self.types}\nboundaries: {self.region}\nachieved coverage: {self.get_coverage()}.\nLast refinement took {socket.gethostname()} {round(self.time_last_refinement, 2)} of {round(self.time_refinement, 2)} sec. whole time."))
                            fig, ax = visualise_by_param(self.rectangles_unsat, colour='red', title=title, where=[fig, ax])
                            if true_point:
                                self.show_true_point(where=[fig, ax], is_inside_of_show=True, hide_legend=hide_legend)
                            return fig, ax
                        else:
                            return None, "While refining multidimensional space no red area found, no reasonable plot to be shown."
                else:
                    if not prefer_unsafe:
                        if self.rectangles_sat:
                            title = "\n".join(self.wrapper.wrap(f"Refinement,\n Domains of respective parameter of safe subspace.\nparam names: {self.params}\nparam types: {self.types}\nboundaries: {self.region}\nachieved coverage: {self.get_coverage()}.\nLast refinement took {socket.gethostname()} {round(self.time_last_refinement, 2)} of {round(self.time_refinement, 2)} sec. whole time."))
                            fig = visualise_by_param(self.rectangles_sat, title=title)
                            plt.show()
                        else:
                            print("No sat rectangles so far, nothing to show")
                    else:
                        if self.rectangles_unsat:
                            title = "\n".join(self.wrapper.wrap(f"Refinement,\n Domains of respective parameter of unsafe subspace.\nparam names: {self.params}\nparam types: {self.types}\nboundaries: {self.region}\nachieved coverage: {self.get_coverage()}.\nLast refinement took {socket.gethostname()} {round(self.time_last_refinement, 2)} of {round(self.time_refinement, 2)} sec. whole time."))
                            fig = visualise_by_param(self.rectangles_unsat, colour='red', title=title)
                            plt.show()
                        else:
                            print("No unsat rectangles so far, nothing to show")

    def show_true_point(self, where=False, is_inside_of_show=False, hide_legend=False):
        """ Showing true point

        Args:
            where (tuple/list): output matplotlib sources to output created figure
            is_inside_of_show (bool): if True not painting the plot
            """
        if self.true_point:
            if where:
                fig = where[0]
                ax = where[1]
            else:
                fig, ax = plt.subplots()

            legend_objects, legend_labels = [], []

            legend_objects.append(plt.scatter([], [], facecolor='white', edgecolor='blue', label="true_point"))
            legend_labels.append("true point")
            if self.sat_samples or self.unsat_samples:
                legend_objects.append(plt.scatter([], [], c="green", alpha=0.5))
                legend_labels.append("sat")
                legend_objects.append(plt.scatter([], [], c="red", alpha=0.5))
                legend_labels.append("unsat")
            if self.rectangles_sat or self.rectangles_unsat:
                legend_objects.append(patches.Patch(color='green', alpha=0.5))
                legend_labels.append("safe")
                legend_objects.append(patches.Patch(color='red', alpha=0.5))
                legend_labels.append("unsafe")

            max_region_size = self.region[0][1] - self.region[0][0]
            if len(self.params) == 1:
                self.true_point_object = plt.Circle((self.true_point[0], self.true_point[1]), max_region_size/10, color='blue', fill=False, label="true_point")

                if where:
                    ax.add_artist(self.true_point_object)
                else:
                    plt.gcf().gca().add_artist(self.true_point_object)
            elif len(self.params) == 2:
                max_region_size = max(max_region_size, self.region[1][1] - self.region[1][0])
                if (len(self.sat_samples) + len(self.unsat_samples)) == 0 or len(self.region) == 0:
                    size_correction = 0.01 * max_region_size
                else:
                    size_correction = min(1 / (len(self.sat_samples) + len(self.unsat_samples)) ** (1 / len(self.region)), 0.01)
                self.true_point_object = plt.Circle((self.true_point[0], self.true_point[1]), size_correction * 1, color='blue', fill=False, label="true_point")
                if where:
                    ax.add_artist(self.true_point_object)
                else:
                    plt.gcf().gca().add_artist(self.true_point_object)
            else:
                ## Multidim true point
                ## TODO maybe not working without GUI
                x_axis = list(range(1, len(self.params) + 1))
                self.true_point_object = ax.scatter(x_axis, self.true_point, marker='$o$', label="true_point", color="blue")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.plot(x_axis, self.true_point, color="blue")
            if not hide_legend:
                ax.legend(legend_objects, legend_labels, loc='upper left', fontsize='small')
            if not is_inside_of_show:
                if where:
                    return fig, ax
                else:
                    plt.show()

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
        """ Returns white space as dictionary """
        return self.rectangles_unknown

    def get_flat_white(self):
        """ Returns white space as a flat list"""
        rectangles_unknown = []
        for key in self.rectangles_unknown.keys():
            rectangles_unknown.extend(self.rectangles_unknown[key])
        return rectangles_unknown

        ## Old implementation (slower)
        # ## https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
        # return [item for sublist in self.rectangles_unknown.values() for item in sublist]

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
        volume = get_rectangle_volume(white)
        if volume in self.rectangles_unknown.keys():
            self.rectangles_unknown[volume].append(white)
        else:
            self.rectangles_unknown[volume] = [white]

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

    def add_degree_samples(self, samples):
        """ Adds samples and their sat degree (distance from not satisfying the constraints)

        Args:
            samples (dict): of samples to sat degree
        """
        self.dist_samples.update(samples)

    def remove_green(self, green):
        """ Removes green (hyper)rectangle """
        self.rectangles_sat.remove(green)
        try:
            self.rectangles_sat_to_show.remove(green)
        except ValueError:
            pass

    def remove_red(self, red):
        """ Removes red (hyper)rectangle """
        self.rectangles_unsat.remove(red)
        try:
            self.rectangles_unsat_to_show.remove(red)
        except ValueError:
            pass

    def remove_white(self, white):
        """ Removes white (hyper)rectangle """
        try:
            volume = get_rectangle_volume(white)
            self.rectangles_unknown[volume].remove(white)
        except Exception as ex:
            print(ex)
            print("Could not remove white area ", white)
            return False
        return True

    def count_green_rectangles(self):
        """ Returns number of green hyper rectangles"""
        return len(self.rectangles_sat)

    def count_red_rectangles(self):
        """ Returns number of red hyper rectangles"""
        return len(self.rectangles_unsat)

    def count_white_rectangles(self):
        """ Returns number of white hyper rectangles"""
        count = 0
        for rectangles in self.rectangles_unknown.keys():
            count = count + len(self.rectangles_unknown[rectangles])
        return count

    def count_sat_samples(self):
        """ Returns number of sat samples"""
        return len(self.sat_samples)

    def count_unsat_samples(self):
        """ Returns number of unsat samples"""
        return len(self.unsat_samples)

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
                    Rectangle((rectangle[0][0], -0.33), rectangle[0][1] - rectangle[0][0], 0.66, fc='g'))
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
                    Rectangle((rectangle[0][0], -0.33), rectangle[0][1] - rectangle[0][0], 0.66, fc='r'))
        self.rectangles_unsat_to_show = []
        return PatchCollection(rectangles_unsat, facecolor='r', alpha=0.5)

    def grid_sample(self, constraints, sample_size, silent: bool = False, save=False, debug=False, progress=False, quantitative=False):
        """ Executes grid sampling

        Args:
            constraints  (list of strings): array of properties
            sample_size (int): number of samples in dimension
            silent (bool): if silent printed output is set to minimum
            save (bool): if True output is pickled
            debug (bool): if True extensive print will be used
            progress (Tkinter element): progress bar
            quantitative (bool): if True return how far is the point from satisfying / not satisfying the constraints
        """
        from sample_space import sample
        self.gridsampled = True
        sample(self, constraints, sample_size, compress=True, silent=silent, save=save, debug=debug, progress=progress, quantitative=quantitative)

    ## TODO DEPRICATED NOT USED NOW, plot.scatter used instead
    def show_samples(self, which):
        """ Visualises samples in 2D"""
        if not (self.sat_samples or self.unsat_samples):
            return None

        samples = []
        if len(self.region) > 2:
            print("Error while visualising", len(self.region), "dimensional space")
            return None
        elif len(self.region) == 1:
            return PatchCollection([], facecolor='r', alpha=0.5)  ## TODO
        elif len(self.region) == 2:
            # print("samples", self.samples)
            try:
                x_size = self.region[0][1] - self.region[0][0]
                y_size = self.region[1][1] - self.region[1][0]
                x_size_correction = min(1 / (len(self.sat_samples) + len(self.unsat_samples)) ** (1 / len(self.region)), 0.01) * x_size
                y_size_correction = min(1 / (len(self.sat_samples) + len(self.unsat_samples)) ** (1 / len(self.region)), 0.01) * y_size
            except Exception as err:
                print("len(self.sat_samples)", len(self.sat_samples))
                print("len(self.unsat_samples)", len(self.unsat_samples))
                print("len(self.region)", len(self.region))
                raise err
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

        rectangles_unknown = self.get_flat_white()

        text = str(f"params: {self.params}\n")
        text = text + str(f"region: {self.region}\n")
        text = text + str(f"types: {self.types}\n")
        text = text + str(f"coverage: {self.get_coverage()} \n")
        text = text + str(f"rectangles_sat: {(f'{self.rectangles_sat[:5]} ... {len(self.rectangles_sat)-5} more', self.rectangles_sat)[len(self.rectangles_sat) <= 30 or full_print]} \n")
        text = text + str(f"rectangles_unsat: {(f'{self.rectangles_unsat[:5]} ... {len(self.rectangles_unsat)-5} more', self.rectangles_unsat)[len(self.rectangles_unsat) <= 30 or full_print]} \n")
        text = text + str(f"rectangles_unknown: {(f'{rectangles_unknown[:5]} ... {len(rectangles_unknown)-5} more', rectangles_unknown)[len(rectangles_unknown) <= 30 or full_print]} \n")
        text = text + str(f"sat_samples: {(f'{self.sat_samples[:5]} ... {len(self.sat_samples)-5} more', self.sat_samples)[len(self.sat_samples) <= 30 or full_print]} \n")
        text = text + str(f"unsat_samples: {(f'{self.unsat_samples[:5]} ... {len(self.unsat_samples)-5} more', self.unsat_samples)[len(self.unsat_samples) <= 30 or full_print]} \n")
        try:
            text = text + str(f"quantitative_samples: {(f'{list(self.dist_samples.items())[:5]} ... {len(self.dist_samples)-5} more', self.dist_samples)[len(self.dist_samples) <= 30 or full_print]} \n")
        except Exception as err:
            print("DIST SAMPLES NOT FOUND PROBABLY OLD VERSION OF REFINED SPACE")
            print(str(err))
            pass
        text = text + str(f"true_point: {self.true_point}\n")
        return text

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

    def update(self):
        """ Make backwards compatible """
        if isinstance(self.rectangles_unknown, list):
            rectangles = {}
            for rectangle in self.rectangles_unknown:
                volume = get_rectangle_volume(rectangle)
                if volume not in rectangles.keys():
                    rectangles[volume] = [rectangle]
                else:
                    rectangles[volume].append(rectangle)
            self.rectangles_unknown = rectangles

    def __repr__(self):
        return str([self.region, self.params, self.types, self.rectangles_sat, self.rectangles_unsat,
                    self.rectangles_unknown, self.sat_samples, self.unsat_samples, self.true_point])

    def __str__(self):
        return str([self.region, self.params, self.types, self.rectangles_sat, self.rectangles_unsat,
                    self.rectangles_unknown, self.sat_samples, self.unsat_samples, self.true_point])
