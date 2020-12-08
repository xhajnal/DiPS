import multiprocessing
import os
from collections import Callable
from platform import system
from time import time
from socket import gethostname
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure

from common.document_wrapper import DocumentWrapper, niceprint

## Importing my code
from termcolor import colored

from common.files import pickle_dump
from common.mathematics import nCr
from common.config import load_config
from common.document_wrapper import DocumentWrapper

import warnings

spam = load_config()
# results_dir = spam["results"]
tmp_dir = spam["tmp"]
del spam

wrapper = DocumentWrapper(width=75)


def maximize_plot():
    try:
        mng = plt.get_current_fig_manager()
        if "wind" in system().lower():
            mng.window.state('zoomed')
        else:
            mng.resize(*mng.window.maxsize())
    except:
        pass


class HastingsResults:
    """ Class to represent Metropolis Hastings results"""
    def __init__(self, params, theta_init, accepted, rejected, observations_count: int, observations_samples_count: int,
                 mh_sampling_iterations: int, eps=0, sd=0.15, burn_in=0.25, pretitle="", title="", bins=20, last_iter=0,
                 timeout=0, time_it_took=0):
        """
        Args:
            params (list of strings): parameter names
            accepted (np.array): accepted points with iteration index
            rejected (np.array): rejected points with iteration index
            observations_count (int): total number of observations
            observations_samples_count (int): sample size from the observations
            mh_sampling_iterations (int): number of iterations/steps of walker in param space
            eps (number): very small value used as probability of non-feasible values in prior
            sd (float): variation of walker in parameter space
            burn_in (number): fraction or count of how many samples will be trimmed from beginning
            pretitle (string): title to be put in front of title
            title (string): title of the plot
            bins (int): number of segments in the heatmap plot (used only for 2D param space)
        """
        ## Inside variables
        self.params = params
        self.theta_init = theta_init

        ## Results
        self.accepted = accepted
        self.rejected = rejected

        ## Results setting
        self.observations_count = observations_count
        self.observations_samples_count = observations_samples_count
        self.mh_sampling_iterations = mh_sampling_iterations
        self.eps = eps
        self.sd = sd

        self.burn_in = burn_in
        # try:  ## backward compatibility
        #     self.not_burn_in = not_burn_in
        # except AttributeError as exx:
        #     if "'HastingsResults' object has no attribute 'not_burn_in'" in exx:
        #         self.not_burn_in = show
        self.title = title
        self.pretitle = pretitle

        self.bins = bins

        self.last_iter = last_iter
        self.timeout = timeout
        self.time_it_took = time_it_took

    def get_burn_in(self):
        """ Returns fraction of the burned-in part"""
        if 0 < self.burn_in < 1:
            return self.burn_in
        elif len(self.accepted):
            return min(1, self.burn_in / len(self.accepted))
        else:
            return None

    def merge_acc_and_rej(self):
        """ Returns both, accepted and rejected samples, in a single list"""
        spam = np.empty([len(self.accepted) + len(self.rejected), len(self.params) + 1])
        acc_index = 0
        rej_index = 0
        for i in range(len(spam)):
            if acc_index < len(self.accepted):
                if int(self.accepted[acc_index][-1]) == i:
                    spam[i] = np.append(self.accepted[acc_index][:-1], True)
                    acc_index = acc_index + 1
                else:
                    spam[i] = np.append(self.rejected[rej_index][:-1], False)
                    rej_index = rej_index + 1
            else:
                spam[i] = np.append(self.rejected[rej_index][:-1], False)
                rej_index = rej_index + 1
        return spam

        # spam = np.empty([len(self.accepted) + len(self.rejected), len(self.params) + 1])
        # acc_index = 0
        # rej_index = 0
        # for i in range(len(self.accepted)):
        #     if int(self.accepted[acc_index][-1]) == i:
        #         spam[i] = np.append(self.accepted[acc_index][:-1], True)
        #         acc_index = acc_index + 1
        #     else:
        #         spam[i] = np.append(self.rejected[rej_index][:-1], False)
        #         rej_index = rej_index + 1
        # for i in range(acc_index + rej_index, len(spam)):
        #     spam[i] = np.append(self.rejected[rej_index][:-1], False)
        #     rej_index = rej_index + 1
        # return spam

        # spam = []
        # acc_index = 0
        # rej_index = 0
        # for i in range(len(self.accepted)):
        #     if int(self.accepted[acc_index][-1]) == i:
        #         spam.append(np.append(self.accepted[acc_index][:-1], True))
        #         acc_index = acc_index + 1
        #     else:
        #         spam.append(np.append(self.rejected[rej_index][:-1], False))
        #         rej_index = rej_index + 1
        # for i in range(rej_index, len(self.rejected)):
        #     spam.append(np.append(self.rejected[rej_index][:-1], False))
        # return spam

    def keep_index(self, burn_in=False):
        """ Translates burn-in into index which should be kept"""
        if not burn_in:
            burn_in = self.burn_in

        if 0 < burn_in < 1:
            keep_index = int(burn_in * self.accepted.shape[0]) + 1
        else:
            keep_index = int(burn_in) + 1
            burn_in = round(burn_in / self.accepted.shape[0], 2)

        return keep_index, burn_in

    def get_not_burn_in(self):
        """ Returns fraction of not burned-in part"""
        if self.get_burn_in() is not None:
            return 1 - self.get_burn_in()
        else:
            return None

    def get_all_accepted(self):
        """ Returns the list of ALL accepted point"""
        return self.accepted

    def get_accepted(self):
        """ Return the list of TRIMMED accepted points"""
        keep_index, burn_in = self.keep_index(self.burn_in)
        return self.accepted[keep_index:]

    def set_accepted(self, accepted):
        """ Sets the accepted points"""
        self.accepted = accepted

    def set_rejected(self, rejected):
        """ Sets rejected points"""
        self.rejected = rejected

    def set_burn_in(self, burn_in):
        """ Sets burn-in period"""
        self.burn_in = burn_in

    def set_bins(self, bins):
        """ Sets bins, used in the plots"""
        self.bins = bins

    def get_acc_as_a_list(self):
        """ Returns accepted points in a list"""
        return self.accepted.tolist()

    def get_rej_as_a_list(self):
        """ Returns rejected points in a list"""
        return self.rejected.tolist()

    def show_mh_heatmap(self, where=False, bins=False, burn_in=None, as_scatter=False, debug=False):
        """ Visualises the result of Metropolis Hastings as a heatmap

        Args:
            where (tuple/list): output matplotlib sources to output created figure
            bins (int): number of segments in the plot (used only for heatmap - 2D param space)
            burn_in (number or None): discards the fraction/number of accepted points, None - use class value (to trim burn-in period)
            as_scatter (bool): Sets the plot to scatter plot even for 2D output
            debug (bool): if True extensive print will be used

        @author: xhajnal
        @edit: denis
        """
        # import matplotlib as mpl
        # mpl.rcParams.update(mpl.rcParamsDefault)
        # plt.style.use('default')

        if self.accepted.size == 0:
            raise Exception("Set of accepted points is empty!")

        ## Backwards compatibility
        if len(self.params) == len(self.accepted[0]):
            print("old data")
            indices = np.linspace(1, len(self.accepted), num=len(self.accepted))
            indices = np.array(list(map(lambda x: [x], indices)))
            self.accepted = np.hstack((self.accepted, indices))
            # for index, item in enumerate(self.accepted):
            #     self.accepted[index] = self.accepted[index]

        if debug:
            print("burn-in", burn_in)
            print("self.accepted", self.accepted)

        if burn_in is None:
            try:  ## Backward compatibility
                burn_in = self.burn_in
            except AttributeError as exx:
                if "'HastingsResults' object has no attribute 'burn_in'" in str(exx):
                    try:
                        burn_in = (100 - self.not_burn_in)/100  ## Backward compatibility
                        self.burn_in = burn_in
                    except:
                        pass
                    try:
                        if 0 <= self.show <= 1:  ## Backward compatibility
                            burn_in = 1 - self.show  ## Backward compatibility
                        elif 0 <= self.show <= 100:  ## Backward compatibility
                            burn_in = 1 - self.show/100  ## Backward compatibility
                        elif self.show < 0:  ## Backward compatibility
                            burn_in = len(self.accepted) + self.show  ## Backward compatibility
                        else:
                            burn_in = len(self.accepted) - self.show  ## Backward compatibility
                        self.burn_in = burn_in
                    except:
                        pass
        try:  ## backward compatibility
            if self.mh_sampling_iterations < 0:
                pass
        except AttributeError as exx:
            self.mh_sampling_iterations = self.MH_sampling_iterations  ## Backward compatibility

        if burn_in < 0:
            raise Exception("MH - wrong burn-in setting.")
        if burn_in > len(self.accepted):
            raise Exception("MH - Burn-in values set higher than accepted point. Nothing to show.")

        ## Convert fraction to show into exact number
        keep_index, burn_in = self.keep_index(burn_in)

        if self.last_iter > 0:
            self.title = f'Estimate of MH algorithm, {niceprint(self.last_iter)} iterations, sample size = {self.observations_count}, \n trimming first {burn_in * 100}% of {niceprint(self.accepted.shape[0])} acc points, init point: {self.theta_init}, \n It took {gethostname()} {round(self.time_it_took, 2)} second(s)'
        else:
            self.title = f'Estimate of MH algorithm, {niceprint(self.mh_sampling_iterations)} iterations, sample size = {self.observations_count}, \n trimming first {burn_in * 100}% of {niceprint(self.accepted.shape[0])} acc points, init point: {self.theta_init}, \n It took {gethostname()} {round(self.time_it_took, 2)} second(s)'

        if debug:
            print("self.accepted[keep_index:, 0]", self.accepted[keep_index:, 0])

        if bins is not False:
            self.bins = bins

        ## Multidimensional case
        if len(self.params) > 3 or as_scatter:
            if where:
                fig = where[0]
                ax = where[1]
                plt.autoscale()
                ax.autoscale()
            else:
                fig, ax = plt.subplots()

            ## Creates values of the horizontal axis
            # x_axis = list(range(1, len(self.accepted[0]) + 1))

            ## Time check
            # from time import time
            # import socket
            # start_time = time()

            ## Get values of the vertical axis for respective line
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ## Thanks to Den for optimisation
            egg = self.accepted[keep_index:].T
            egg = egg[:-1]
            try:
                ax.plot(egg, '.-', markersize=15)  ## TODO Check 10 bees default -> ZeroDivisionError: integer division or modulo by zero
            except ZeroDivisionError as err:
                print("accepted points", self.accepted)
                print("accepted points transposed", self.accepted.T)
                print("accepted points transposed", self.egg)
                raise err

            # for sample in self.accepted[not_burn_in:]:
            #    ax.scatter(x_axis, sample)
            #    ax.plot(x_axis, sample)

            ax.set_xlabel("param indices")
            ax.set_ylabel("parameter value")
            ax.set_title(wrapper.wrap(self.title))
            ax.autoscale()
            ax.margins(0.1)
            # print(colored(f"  It took {socket.gethostname()}, {time() - start_time} seconds to run", "blue"))

            if where:
                return fig, ax
            else:
                maximize_plot()
                plt.show()
        elif len(self.params) == 2:
            if where:
                plt.hist2d(self.accepted[keep_index:, 0], self.accepted[keep_index:, 1], bins=self.bins)
                plt.xlabel(self.params[0])
                plt.ylabel(self.params[1])
                plt.title(wrapper.wrap(self.title))
                where[1] = plt.colorbar()
                where[1].set_label('# of accepted points per bin', rotation=270, labelpad=20)
                return where[0], where[1]
            else:
                plt.figure(figsize=(12, 6))
                plt.hist2d(self.accepted[keep_index:, 0], self.accepted[keep_index:, 1], bins=self.bins)
                figure = plt.colorbar()
                plt.xlabel(self.params[0])
                plt.ylabel(self.params[1])
                plt.title(self.title)
                figure.ax.set_ylabel('# of accepted points per bin', rotation=270, labelpad=20)
                maximize_plot()
                plt.show()
        else:
            spam = np.ones(len(self.accepted[keep_index:, 0]))
            if where:
                plt.hist2d(self.accepted[keep_index:, 0], spam, bins=self.bins)
                plt.xlabel(self.params[0])
                plt.ylabel("")
                plt.title(wrapper.wrap(self.title))
                where[1] = plt.colorbar()
                where[1].set_label('# of accepted points per bin', rotation=270, labelpad=20)
                return where[0], where[1]
            else:
                plt.figure(figsize=(12, 6))
                plt.hist2d(self.accepted[keep_index:, 0], spam, bins=self.bins)
                figure = plt.colorbar()
                plt.xlabel(self.params[0])
                plt.ylabel("")
                plt.title(self.title)
                figure.ax.set_ylabel('# of accepted points per bin', rotation=270, labelpad=20)
                maximize_plot()
                plt.show()

    def show_iterations(self, where=False):
        """ Create Scatter plot showing accepted and rejected points in its given order

        Args:
           where (bool or callable): method to forward the figure
        """
        if self.accepted.size == 0:
            raise Exception("Set of accepted points is empty")

        if where:
            fig = Figure(figsize=(10, 10))
        else:
            fig = plt.figure()
        if len(self.params) == 2:
            plots = 3
        else:
            plots = len(self.params)

        borderline_index = self.keep_index(self.get_burn_in())[0]
        for index, param in enumerate(self.params):
            ax = fig.add_subplot(plots, 1, index + 1)
            if borderline_index > 1:
                ax.axvline(x=borderline_index + 0.5, color='black', linestyle='-', label="burn-in threshold")

            ax.scatter(self.rejected[:, -1], self.rejected[:, index], marker='x', c="r", label='Rejected', alpha=0.5)
            ax.scatter(self.accepted[:, -1], self.accepted[:, index], marker='.', c="b", label='Accepted', alpha=0.5)

            ## TODO probably can be optimised
            # ax.axvline(x=self.get_burn_in() * self.mh_sampling_iterations, color='black', linestyle='-',
            #            label="burn-in threshold")
            ## New code adding information how the accepted and rejected points are connected
            # X_accept, X_reject, Y_accept, Y_reject = [], [], [], []
            # for point_index, point in enumerate(self.both):
            #     if point[1] is False:
            #         X_reject.append(point_index)
            #         Y_reject.append(point[0][index])
            #     else:
            #         X_accept.append(point_index)
            #         Y_accept.append(point[0][index])
            # ax.scatter(X_reject, Y_reject, marker='x', c="r", label='Rejected', alpha=0.5)
            # ax.scatter(X_accept, Y_accept, marker='.', c="b", label='Accepted', alpha=0.5)

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("MH Iteration")
            ## Previous code before that information
            # ax.plot(rejected[:, index], 'rx', label='Rejected', alpha=0.5)
            # ax.plot(accepted[:, index], 'b.', label='Accepted', alpha=0.5)
            # ax.set_xlabel("Index")
            ax.set_ylabel(f"${param}$")
            ax.set_title(f"Accepted and Rejected values of ${param}$.")
            ax.grid()
            ax.legend()
        ## Plane plot, phase space
        if len(self.params) == 2:
            ax = fig.add_subplot(plots, 1, 3)
            both = self.merge_acc_and_rej()
            ax.plot(both[:, 0], both[:, 1], label="Path", c="grey", alpha=0.1)
            ax.plot(both[:, 0][0], both[:, 1][0], 'g+', label="Start")
            del both
            ax.plot(self.accepted[:, 0], self.accepted[:, 1], 'b.', label='Accepted', alpha=0.3)
            ax.plot(self.rejected[:, 0], self.rejected[:, 1], 'rx', label='Rejected', alpha=0.3)
            ax.set_xlabel(self.params[0])
            ax.set_ylabel(self.params[1])
            ax.legend()
            ax.set_title("Trace of Accepted and Rejected points in a plane.")
        if not where:
            maximize_plot()
            plt.show()
        else:
            where(fig)

    def show_iterations_bokeh(self, where=False):
        """ Create Scatter plot showing accepted and rejected points in its given order using bokeh

        Args:
           where (bool or callable): method to forward the figure
        """
        if self.accepted.size == 0:
            raise Exception("Set of accepted points is empty")

        plots = []
        borderline_index = self.keep_index(self.get_burn_in())[0]

        for index, param in enumerate(self.params):
            ## Scatter plot
            tools = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
            p = bokeh_figure(title=f"Trace of accepted points for {param}", tools=tools)
            # p.scatter(range(len(self.accepted[:, index])), self.accepted[:, index], fill_alpha=0.6, line_color=None)
            if borderline_index > 1:
                vline = Span(location=borderline_index + 0.5, dimension='height', line_color='black', line_width=1)
                p.renderers.extend([vline])
            p.scatter(self.rejected[:, -1], self.rejected[:, index], color="red", fill_alpha=0.6, legend_label='Rejected')
            p.scatter(self.accepted[:, -1], self.accepted[:, index], color="blue", fill_alpha=0.6, legend_label='Accepted')
            p.xaxis.axis_label = 'MH Iteration'
            p.yaxis.axis_label = f'{param}'
            p.legend.location = "top_left"
            plots.append([p])

        if len(self.params) == 2:
            both = self.merge_acc_and_rej()
            p = bokeh_figure(title=f"Trace of Accepted and Rejected points in a plane", tools=tools)
            p.line(both[:, 0], both[:, 1], color="grey", alpha=0.1, legend_label='Path')
            p.scatter(both[:, 0][0], both[:, 1][0], legend_label="Start")
            if both[0][-1]:  ## if the first point is accepted
                p.scatter(self.accepted[:, 0][1:], self.accepted[:, 1][1:], alpha=0.3, legend_label='Accepted', color="blue")
                p.scatter(self.rejected[:, 0], self.rejected[:, 1], alpha=0.3, legend_label='Rejected', color="red")
            else:
                p.scatter(self.accepted[:, 0], self.accepted[:, 1], alpha=0.3, legend_label='Accepted', color="blue")
                p.scatter(self.rejected[:, 0][1:], self.rejected[:, 1][1:], alpha=0.3, legend_label='Rejected', color="red")
            del both
            p.xaxis.axis_label = self.params[0]
            p.yaxis.axis_label = self.params[1]
            p.legend.location = "top_left"
            plots.append([p])

        output_file(os.path.join(tmp_dir, "Trace_of_Accepted_and_Rejected_points.html"), title=f"Trace of Accepted and Rejected points in a plane.")
        show(gridplot(plots))  # open a browser

    def show_accepted(self, where=False):
        """ Trace and histogram of accepted points

        Args:
           where (bool or callable): method to forward the figure
        """

        if self.accepted.size == 0:
            raise Exception("Set of accepted points is empty")

        try:
            bins = self.bins
        except:
            bins = 20

        if where:
            fig = Figure(figsize=(20, 10))
        else:
            fig = plt.figure()
        if len(self.params) == 2:
            gs = gridspec.GridSpec(3, 2, figure=fig)
        else:
            gs = gridspec.GridSpec(len(self.params), 2, figure=fig)

        borderline_index = self.keep_index(self.get_burn_in())[0]

        for index, param in enumerate(self.params):
            ## Trace of accepted points for respective parameter
            ax = fig.add_subplot(gs[index, 0])
            ax.plot(self.accepted[:, index])
            if borderline_index > 1:
                ax.axvline(x=borderline_index - 0.5, color='black', linestyle='-', label="burn-in threshold")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(f"Trace of accepted points for ${param}$")
            ax.set_xlabel(f"Index")
            ax.set_ylabel(f"${param}$")

            ## Histogram of accepted points for respective parameter
            ax = fig.add_subplot(gs[index, 1])
            # X = sorted(list(set(accepted[:, index])))
            # Y = []
            # for i in X:
            #     Y.append(list(accepted[:, index]).count(i))
            # ax.bar(X, Y)
            # plt.xticks(range(len(functions)), range(len(functions) + 1))
            ax.hist(self.accepted[:, index], bins=bins, density=True)
            ax.set_ylabel("Occurrence")
            ax.set_xlabel(f"${param}$")
            ax.set_title(f"Histogram of accepted points for ${param}$, {bins} bins.")
            fig.tight_layout()
        ## Plane plot, phase space
        if len(self.params) == 2:
            ax = fig.add_subplot(gs[2, :])
            ax.plot(self.accepted[:, 0], self.accepted[:, 1],  label="Path", c="grey", alpha=0.1)
            ax.plot(self.accepted[:, 0], self.accepted[:, 1], 'b.', label='Accepted', alpha=0.3)
            ax.plot(self.accepted[:, 0][0], self.accepted[:, 1][0], 'g+', label="Start")
            ax.set_xlabel(self.params[0])
            ax.set_ylabel(self.params[1])
            ax.legend()
            ax.set_title("Trace of Accepted points in a plane.")

        if not where:
            maximize_plot()
            plt.show()
        else:
            where(fig)

    def show_accepted_bokeh(self, where=False):
        """ Trace and histogram of accepted points using bokeh
        
        Args:
           where (bool or callable): unused in this method
        """
        if self.accepted.size == 0:
            raise Exception("Set of accepted points is empty")

        try:
            bins = self.bins
        except:
            bins = 20

        plots = []
        borderline_index = self.keep_index(self.get_burn_in())[0]

        for index, param in enumerate(self.params):
            triplets = []

            ## Scatter plot of accepted points
            tools = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
            p = bokeh_figure(title=f"Trace of accepted points for {param}", tools=tools)
            # p.scatter(range(len(self.accepted[:, index])), self.accepted[:, index], fill_alpha=0.6, line_color=None)
            if borderline_index > 1:
                vline = Span(location=borderline_index - 0.5, dimension='height', line_color='black', line_width=1)
                p.renderers.extend([vline])
            p.line(range(len(self.accepted[:, index])), self.accepted[:, index])
            p.xaxis.axis_label = 'Index'
            p.yaxis.axis_label = f'{param}'
            p.legend.location = "top_left"
            triplets.append(p)

            ## Histogram of accepted points
            hist, edges = np.histogram(self.accepted[:, index], density=True, bins=bins)
            # p = bokeh_figure(title=f"Histogram of accepted points for {param}, {bins} bins", tools='', background_fill_color="#fafafa")
            p = bokeh_figure(title=f"Histogram of accepted points for {param}, {bins} bins", tools='')
            # p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.5)
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")
            p.y_range.start = 0
            p.legend.location = "center_right"
            # p.legend.background_fill_color = "#fefefe"
            p.xaxis.axis_label = f'{param}'
            p.yaxis.axis_label = 'Occurrence'
            p.legend.location = "top_left"
            p.grid.grid_line_color = "white"

            triplets.append(p)

            ## Histogram of trimmed accepted points
            if borderline_index > 1:
                hist, edges = np.histogram(self.accepted[borderline_index:, index], density=True, bins=bins)
                # p = bokeh_figure(title=f"Histogram of accepted points for {param}, {bins} bins", tools='', background_fill_color="#fafafa")
                p = bokeh_figure(title=f"Histogram of not-trimmed accepted points for {param}, {bins} bins", tools='')
                # p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.5)
                p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")
                p.y_range.start = 0
                p.legend.location = "center_right"
                # p.legend.background_fill_color = "#fefefe"
                p.xaxis.axis_label = f'{param}'
                p.yaxis.axis_label = 'Occurrence'
                p.legend.location = "top_left"
                p.grid.grid_line_color = "white"

                triplets.append(p)

            plots.append(triplets)

        if len(self.params) == 2:
            p = bokeh_figure(title=f"Trace of Accepted points in a plane", tools=tools)
            p.line(self.accepted[:, 0], self.accepted[:, 1], color="grey", alpha=0.1, legend_label='Path')
            p.scatter(self.accepted[:, 0], self.accepted[:, 1], alpha=0.3, legend_label='Accepted')
            p.scatter(self.accepted[:, 0][0], self.accepted[:, 1][0], color="green", legend_label='First accepted point')
            p.xaxis.axis_label = self.params[0]
            p.yaxis.axis_label = self.params[1]
            p.legend.location = "top_left"
            plots.append([p])

        output_file(os.path.join(tmp_dir, f"Trace_of_accepted_points.html"), title=f"Trace_of_accepted_points")
        show(gridplot(plots))  # open a browser


## Now unused
def sample_functions(functions, data_means):
    """ Will sample functions according to the pdf as given by the polynomials

    Args:
        functions TODO @Tanja
        data_means TODO @Tanja

    Returns:
         ## TODO @Tanja

    @author: tpetrov
    """
    x = np.random.uniform(0, 1)
    i = 0
    while x > sum(data_means[0:i]) and (i < len(functions)):
        i = i + 1
    return i - 1


def get_truncated_normal(mean=0.0, sd=1.0, low=0.0, upp=10.0):
    """ Returns truncated normal distribution

    Args:
        mean (float): mean
        sd (float): standard deviation
        low (float): lower bound
        upp (float): upper bound

    """
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs()


def transition_model_a(theta, parameter_intervals, sd=0.15):
    """" Defines how to walk around the parameter space, set a new point,
        using normal distribution around the old point

    Args:
        theta (list): old parameter point
        parameter_intervals (list of tuples): domains of parameters
        sd (float): standard deviation of normal dist. of walker

    Returns:
        theta_new (list): new parameter point within the domains

    @author: tpetrov
    @edit: xhajnal, denis
    """
    # if sort:
    #     sd = 0.15  ## Standard deviation of the normal distribution
    # else:
    #     sd = 0.15  ## TODO FIND OPTIMAL VALUE
    theta_new = np.zeros(len(theta))  ## New point initialisation

    ## For each parameter
    ## TODO why we change all params and not just one in random?
    for index, param in enumerate(theta):
        temp = parameter_intervals[index][0] - 1  ## Lower bound of first parameter - 1, forcing to find a new value
        #### THIS WAS NOT WORKING
        # if sort:
        #     temp = get_truncated_normal(theta[index], sd, low=max(temp <= parameter_intervals[index][0], theta_new[max(0, index - 1)]), upp=parameter_intervals[index][1])
        # else:
        #     while (temp <= parameter_intervals[index][0]) or (temp >= parameter_intervals[index][1]):
        #         temp = np.random.normal(theta[index], sd)
        # max_param = theta_new[max(0, index - 1)]
        while (temp < parameter_intervals[index][0]) or (temp > parameter_intervals[index][1]) or (sort and temp < max_param):
            ## Generate new parameter value from normal distribution
            if sort and max_param > theta[index]:
                temp = get_truncated_normal(mean=max_param, sd=sd, low=max_param, upp=parameter_intervals[index][1])
                if temp < max_param:
                    temp = temp + abs(max_param-temp)
            else:
                temp = np.random.normal(theta[index], sd)
                # For some reason slower
                # temp = get_truncated_normal(mean=theta[index], sd=sd, low=parameter_intervals[index][0], upp=parameter_intervals[index][1])
        ## Store only if the param value inside the domains
        theta_new[index] = temp

    return theta_new


## Now unused
def prior(theta, parameter_intervals):
    """ Very simple prior estimator only for MH as it checks if the parametrisation is inside of respective domain or not
        This simulates uniform distribution
    Args:
        theta: (tuple): parameter point
        parameter_intervals (list of pairs): parameter domains, (min, max)

    Returns:
        1 for all parametrisations inside the parameter space
        0 otherwise

    @author: xhajnal
    """

    for index, value in enumerate(theta):
        ## If inside of param domains
        if (theta[index] < parameter_intervals[index][0]) or (theta[index] > parameter_intervals[index][1]):
            return 0
    return 1


def acceptance(x_likelihood, x_new_likelihood):
    """ Decides whether to accept new point, x_new, or not, based on its likelihood

    Args:
        x_likelihood: likelihood of the old parameter point
        x_new_likelihood: likelihood of the new parameter point

    Returns:
         True if the new points is accepted

    @author: tpetrov
    """
    ## If likelihood of new point is higher (than likelihood of current point) accept the new point
    if x_new_likelihood > x_likelihood:
        return True
    else:
        ## Chance to accept even if the likelihood of the new point is lower (than likelihood of current point)
        accept = np.random.uniform(0, 1)
        return accept < (np.exp(x_new_likelihood - x_likelihood))


## Now unused
def acceptance_default(x_likelihood, x_new_likelihood):
    """ Decides whether to accept new point, x_new, or not, based on its likelihood

    Args:
        x_likelihood: likelihood of the old parameter point
        x_new_likelihood: likelihood of the new parameter point

    Returns:
         True if the new points is accepted

    @author: xhajnal
    """
    ## If likelihood of new point is higher (than likelihood of current point) accept the new point
    if x_new_likelihood > x_likelihood:
        return True


global glob_param_names
global glob_theta


def eval_function(function):
    """ Evaluates function in with global glob_param_names and glob_theta"""
    for index, param in enumerate(glob_theta):
        locals()[glob_param_names[index]] = glob_theta[index]
    return eval(function)


def manual_log_like_normal(params, theta, functions, data, sample_size, eps=0, parallel=False, debug=False):
    """ Log likelihood of functions in parameter point theta drawing the data, P(functions(theta)| data)

    Args:
        params (list of string): parameter names
        theta (list): parameter point to evaluate functions in
        functions (list of strings): functions to be evaluated in theta
        data (list): data that we wish to model, measurement values
        sample_size (int): number of samples in data
        eps (number): very small value used as probability of non-feasible values in prior - deprecated now
        parallel (Bool): flag to run this in parallel mode
        debug (bool): if True extensive print will be used

    Returns:
         likelihood (float): P(data | functions(theta))

    @author: tpetrov
    @edit: xhajnal
    """
    warnings.filterwarnings("error")
    # print("data", data)
    # print("functions", functions)

    ## Assignment of parameter values
    for index, param in enumerate(theta):
        locals()[params[index]] = theta[index]

    ## OLD CODE
    # ## Dictionary optimising performance - not evaluating the same functions again
    # evaled_functions = {}

    ## TODO represent observations as data
    res = 0

    if parallel:
        if isinstance(parallel, int):
            pool_size = parallel
        else:
            pool_size = multiprocessing.cpu_count() - 1

        # global glob_param_names
        # global glob_theta
        globals()["glob_theta"] = theta
        globals()["glob_param_names"] = params
        with multiprocessing.Pool(pool_size) as p:
            evaled_functions = list(p.map(eval_function, functions))
        if debug:
            print("evaled_functions", evaled_functions)

    ## Via wiki https://en.wikipedia.org/wiki/Maximum_likelihood_estimation Discrete distribution, continuous parameter space
    for index, data_point in enumerate(data):
        if parallel:
            point = evaled_functions[index]
        else:
            point = eval(functions[index])

        # lik = C(n,k) * p**k * (1-p)**(n-k)  ## formula
        # lik = nCr(sample_size, data_point* sample_size) * point**(data_point*sample_size) * (1-point)**(sample_size-data_point*sample_size)  ## Our representation
        ## log_lik = np.log(nCr(sample_size, data_point * sample_size)) + (data_point * sample_size * np.log(point) + (1 - data_point) * sample_size * np.log(1 - point))  ## Original log likelihood, but the C(n,k) does not change that the one loglik is greater and it strikes out in subtraction part
        try:
            pseudo_log_lik = (data_point * sample_size * np.log(point) + (1 - data_point) * sample_size * np.log(1 - point))
        except RuntimeWarning as warn:
            if debug:
                print(warn)
                print("function/data index:", index)
                print("theta:", theta)
                print(f"functions[{index}]:", point)
                print(f"data[{index}]:", data_point)
            # if "divide by zero encountered in log" in str(warn):
            #     pseudo_log_lik = float("-inf")
            if point <= 0 or point >= 1:
                ## When the point is exactly 1 or 0 and data is respective value as well
                if (data_point == 0 or data_point == 1) and (point == 0 or point == 1):
                    raise warn
                else:
                    pseudo_log_lik = float("-inf")
            else:
                raise warn
            # show_message(2, "MH", f"function value {point} is invalid for log")
        res = res + pseudo_log_lik
        if debug:
            print(f"param point {theta}")
            print(f"data_point {data_point}")
            print(f"function {eval(functions[index])}")
            likelihood = nCr(sample_size, data_point*sample_size) * point**(data_point*sample_size) * (1-point)**(sample_size-data_point*sample_size)
            print(colored(f"likelihood {likelihood}", "blue"))
            ## Default form
            # print(colored(f"pseudo log-likelihood {np.log(point ** (data_point * sample_size) * (1 - point) ** (sample_size - data_point * sample_size))}", "blue"))
            print(colored(f"pseudo log-likelihood {pseudo_log_lik}", "blue"))
            ## Default form
            print(colored(f"log likelihood {np.log(likelihood)}", "blue"))
            print(colored(f"log-likelihood {np.log(nCr(sample_size, data_point * sample_size)) + np.log(point)*(data_point*sample_size) + np.log(1-point)*(sample_size-data_point*sample_size)}", "blue"))

            print()
        if str(res) == "-inf":
            warnings.filterwarnings("default")  ## normal state
            return res

    # for index, data_point in enumerate(data):
    #     sigma = np.sqrt((data_point - eval(functions[index])) ** 2 / sample_size)
    #     res = res - sample_size * np.log(sigma * np.sqrt(2 * np.pi)) - ((data_point - eval(functions[index])) ** 2 / (2 * sigma ** 2))

    ## OLD CODE - using observation - problem how to encode overlapping observation of events
    # for data_point in observations:
    #     # print("data_point", data_point)
    #     # print("functions[data_point]", functions[data_point])
    #
    #     if data_point in evaled_functions.keys():
    #         temp = evaled_functions[data_point]
    #     else:
    #         evaled_functions[data_point] = eval(functions[data_point])
    #         temp = evaled_functions[data_point]
    #
    #     # print(temp)
    #     # print(np.log(temp))
    #
    #     if temp < eps:
    #         temp = eps
    #     if temp > 1. - eps:
    #         temp = 1. - eps
    #
    #     # print(res)
    #     res = res + np.log(temp)  # +np.log(prior(x))

    # print(res)
    warnings.filterwarnings("default")  ## normal state
    return res


                        debug=False):
def metropolis_hastings(likelihood_computer, prior_rule, transition_model, acceptance_rule, params, parameter_intervals,
                        param_init, functions, data, sample_size, iterations, eps, sd, progress=False, timeout=0,
                        debug=False):
    """ The core method of the Metropolis Hasting

        likelihood_computer (function(space, theta, functions, observation/data, eps)): function returning the likelihood that functions in theta point generated the data
        prior_rule (function(theta, eps)): prior function
        transition_model (function(theta)): a function that draws a sample from a symmetric distribution and returns it
        acceptance_rule (function(theta, theta_new)): decides whether to accept or reject the new sample
        params (list of strings): parameter names
        parameter_intervals (list of pairs): parameter domains
        param_init  (pair of numbers): starting parameter point
        functions (list of strings): expressions to be evaluated and compared with data
        data (list of numbers): data that we wish to model, measurement values
        sample_size (int): number of observations in data
        iterations (int): number of steps of walker
        eps (number): very small value used as probability of non-feasible values in prior - not used now
        sd (float): variation of walker in parameter space
        progress (Tkinter element or False): function processing progress
        timeout (int): timeout in seconds (0 for no timeout)
        debug (bool): if True extensive print will be used

    Returns:
        accepted, rejected (tuple of np.arrays): tuple of accepted and rejected parameter points

    @author: tpetrov
    @edit: xhajnal
    """
    try:
        start_time = globals()["start_time"]
    except KeyError:
        start_time = time()

    theta = param_init
    accepted = []
    ## Setting the initial point as rejected so it will be shown in plots
    ## Even though it is never compared and hence should not be acc nor rej
    rejected = [np.append(theta, 0)]
    ## For each MCMC iteration do
    has_moved = True
    theta_lik = 0
    for iteration in range(1, iterations + 1):
        ## Walk in parameter space - Get new parameter point from the current one
        theta_new = transition_model(theta, parameter_intervals, sd=sd, sort=sort)
        # print("theta_new", theta_new)
        # if sort:
        #     if sorted(list(theta_new)) != list(theta_new):
        #         print(colored(f"{theta_new} is decreasing", "red"))

        ## Estimate likelihood of current point
        ## (space, theta, functions, data, eps)

        ## Not recalculating the likelihood if we did not move
        if has_moved:
            theta_lik = likelihood_computer(params, theta, functions, data, sample_size, eps, debug=debug)
        # print("theta_lik", theta_lik)
        ## Estimate likelihood of new point
        theta_new_lik = likelihood_computer(params, theta_new, functions, data, sample_size, eps, debug=debug)
        # print("theta_new_lik", theta_new_lik)
        if debug:
            print("iteration:", iteration)
        # print("theta_lik + np.log(prior(theta, parameter_intervals))", theta_lik + np.log(prior(theta, parameter_intervals)))
        # print("theta_new_lik + np.log(prior(theta, parameter_intervals))", theta_new_lik + np.log(prior(theta_new, parameter_intervals)))

        ## If new point accepted
        ## old acceptance rule checking using uniform distribution ## this is done by the walker
        # if acceptance_rule(theta_lik + np.log(prior_rule(theta, parameter_intervals)), theta_new_lik + np.log(prior_rule(theta_new, parameter_intervals))):
        if acceptance_rule(theta_lik, theta_new_lik):
            ## Go to the new point
            has_moved = True
            theta = theta_new
            accepted.append(np.append(theta_new, iteration))
            if debug:
                print(f"new point: {theta_new} accepted")
        else:
            has_moved = False
            rejected.append(np.append(theta_new, iteration))
            if debug:
                print(f"new point: {theta_new} rejected")
        if progress:
            assert isinstance(progress, Callable)
            progress(iteration/iterations, False, int(time() - globals()["start_time"]), timeout)

        ## Finish iterations after timeout
        if (time() - start_time) > timeout > 0:
            try:
                globals()["mh_results"].last_iter = iteration
                globals()["mh_results"].time_it_took = time() - globals()["start_time"]
            except KeyError:
                pass
            break

    try:
        globals()["mh_results"].time_it_took = time() - globals()["start_time"]
    except KeyError:
        pass

    return np.array(accepted), np.array(rejected)


def initialise_sampling(params, parameter_intervals, functions, data, sample_size: int,  mh_sampling_iterations: int, eps=0,
                        sd=0.15, theta_init=False, where=False, progress=False, burn_in=False, bins=20, timeout=False,
                        debug=False, metadata=True, draw_plot=False):
    """ Initialisation method for Metropolis Hastings

    Args:
        params (list of strings): parameter names
        parameter_intervals (list of tuples): domains of parameters
        theta_init (list of floats): initial parameter point
        functions (list of strings): expressions to be evaluated and compared with data
        data (list of floats): measurement values
        sample_size (int): total number of observations in data
        mh_sampling_iterations (int): number of iterations/steps in searching in space
        eps (number): very small value used as probability of non-feasible values in prior - not used now
        sd (float): variation of walker in parameter space
        where (tuple/list or False): output matplotlib sources to output created figure, if False a new will be created
        progress (Tkinter element or False): function processing progress
        burn_in (number): fraction or count of how many samples will be trimmed from beginning
        bins (int): number of segments per dimension in the output plot
        timeout (int): timeout in seconds (0 for no timeout)
        debug (bool): if True extensive print will be used
        metadata (bool): if True metadata will be plotted
        draw_plot (Callable): function showing intermediate plots

    @author: tpetrov
    @edit: xhajnal
    """
    ## Internal settings
    start_time = time()
    globals()["start_time"] = start_time

    ##                        HastingsResults(params, theta_init, accepted, rej observations_count, observations_samples_count, MH_sampling_iterations, eps, burn_in, pretitle, title, bins, last_iter,  timeout, time_it_took, rescale):
    globals()["mh_results"] = HastingsResults(params, theta_init, [], [], sample_size, sample_size, mh_sampling_iterations, eps, burn_in=burn_in, title="", bins=bins, last_iter=0, timeout=timeout)

    # print("Parameter point", theta_true)

    ## If no starting point given
    if not theta_init:
        theta_init = []
        ## Select point which is center of each interval
        for index, param in enumerate(parameter_intervals):
            theta_init.append((parameter_intervals[index][0] + parameter_intervals[index][1])/2)
        # theta_init = [(parameter_intervals[0][0] + parameter_intervals[0][1])/2, (parameter_intervals[1][0] + parameter_intervals[1][1])/2]  ## Middle of the intervals # np.ones(10)*0.1

    ## TODO do we need this?
    for index, param in enumerate(params):
        globals()[param] = theta_init[index]
        print(f"{param} = {theta_init[index]}")

    # ## Maintaining observations
    # # If given observations or data
    # if data:
    #     ## Checking the type of observations (experiment/data)
    #     if len(data) > len(functions):
    #         ## Already given observations
    #         observations_count = len(data)
    #     else:
    #         ## Changing the data into observations
    #         spam = []
    #         index = 0
    #         for observation in data:
    #             for times in range(int(observation * float(sample_size))):
    #                 spam.append(index)
    #             index = index + 1
    #         data = spam
    # else:
    #     data_means = [eval(fi) for fi in functions]
    #     print("data means", data_means)
    #     samples = []
    #     for i in range(sample_size):
    #         samples.append(sample_functions(functions, data_means))
    #     print("samples", samples)
    #
    #     data = np.array(samples)[np.random.randint(0, sample_size, sample_size)]
    print("data", data)
    print("Initial parameter point: ", theta_init)

    print(colored(f"Initialisation of Metropolis-Hastings took {round(time() - start_time, 4)} seconds", "yellow"))
    ## MAIN LOOP
    #                    metropolis_hastings(likelihood_computer, prior_rule, transition_model, acceptance_rule, params, parameter_intervals, param_init, functions, data, sample_size, iterations,        eps,     sd,      progress=False,      timeout=0,  debug=False, sort=False):
    accepted, rejected = metropolis_hastings(manual_log_like_normal, prior, transition_model_a, acceptance, params, parameter_intervals, theta_init, functions, data, sample_size, mh_sampling_iterations, eps=eps, sd=sd, progress=progress, timeout=timeout, debug=debug, sort=sort)

    print(colored(f"Metropolis-Hastings took {round(time()-start_time, 4)} seconds", "yellow"))

    globals()["mh_results"].set_accepted(accepted)
    globals()["mh_results"].set_rejected(rejected)

    print("accepted.shape", accepted.shape)
    if len(accepted) == 0:
        print("Metropolis-Hastings, no accepted point found")
        return False

    ## Dumping results
    print(f"Set of accepted points is stored here: {tmp_dir}/accepted.p")
    pickle_dump(accepted, os.path.join(tmp_dir, f"accepted.p"))
    print(f"Set of rejected points is stored here: {tmp_dir}/rejected.p")
    pickle_dump(rejected, os.path.join(tmp_dir, f"rejected.p"))
    print(f"Whole class is stored here: {tmp_dir}/mh_class.p")
    pickle_dump(globals()["mh_results"], os.path.join(tmp_dir, f"mh_class.p"))

    ## Showing metadata visualisations
    if metadata:
        if not where:
            globals()["mh_results"].show_iterations()
            globals()["mh_results"].show_accepted()
        else:
            globals()["mh_results"].show_iterations(where=draw_plot)
            globals()["mh_results"].show_accepted(where=draw_plot)

    ## TODO make a option to set to see the unzoomed space - freaking hard
    ## "Currently hist2d calculates it's own axis limits, and any limits previously set are ignored." (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.hist2d.html)
    ## Option 2 - https://stackoverflow.com/questions/29175093/creating-a-log-linear-plot-in-matplotlib-using-hist2d
    ## No scale

    if where:
        return globals()["mh_results"]
    else:
        globals()["mh_results"].show_mh_heatmap(where=where)
