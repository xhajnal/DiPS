import configparser
import os
from pathlib import Path


def load_config(path=False):
    """ Loads config file

    Args
        path (str or Path): path to the config file, set False for default path
    """
    config = configparser.ConfigParser()
    workspace = os.path.dirname(__file__)
    current_directory = os.getcwd()
    os.chdir(workspace)

    if path is False:
        config.read(os.path.join(workspace, "../../config.ini"))
    else:
        if os.path.isabs(Path(path)):
            config.read(path)
        else:
            config.read(os.path.join(workspace, path))

    # config.sections()

    if not config.sections():
        raise Exception("Config file", "Config file, config.ini, not properly loaded.")

    prism_path = config.get("mandatory_paths", "prism_path")

    cwd = config.get("mandatory_paths", "cwd")
    if cwd == "":
        cwd = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
    if not os.path.isabs(cwd):
        cwd = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
    # print("os.path.dirname(__file__)", os.path.dirname(__file__))
    # print("cwd", cwd)

    model_dir = config.get("paths", "models")
    if model_dir == "":
        model_dir = "models"
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(cwd, model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    property_dir = config.get("paths", "properties")
    if property_dir == "":
        property_dir = "properties"
    if not os.path.isabs(property_dir):
        property_dir = os.path.join(cwd, property_dir)
    if not os.path.exists(property_dir):
        os.makedirs(property_dir)

    data_dir = config.get("paths", "data")
    if data_dir == "":
        data_dir = "data"
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(cwd, data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_weights_dir = config.get("paths", "data_weights")
    if data_weights_dir == "":
        data_weights_dir = "data_weights"
    if not os.path.isabs(data_weights_dir):
        data_weights_dir = os.path.join(cwd, data_weights_dir)
    if not os.path.exists(data_weights_dir):
        os.makedirs(data_weights_dir)

    ## Results
    results_dir = config.get("paths", "results")
    if results_dir == "":
        results_dir = "results"
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(cwd, results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # print("results_dir", results_dir)

    prism_results = os.path.join(results_dir, "prism_results")
    if not os.path.exists(prism_results):
        os.makedirs(prism_results)

    storm_results = os.path.join(results_dir, "storm_results")
    if not os.path.exists(storm_results):
        os.makedirs(storm_results)

    data_intervals_dir = os.path.join(results_dir, "data_intervals")
    if not os.path.exists(data_intervals_dir):
        os.makedirs(data_intervals_dir)

    data_weights_dir = os.path.join(results_dir, "data_weights")
    if not os.path.exists(data_weights_dir):
        os.makedirs(data_weights_dir)

    constraints_dir = os.path.join(results_dir, "constraints")
    if not os.path.exists(constraints_dir):
        os.makedirs(constraints_dir)

    refinement_results = os.path.join(results_dir, "refinement_results")
    if not os.path.exists(refinement_results):
        os.makedirs(refinement_results)

    figures_dir = os.path.join(results_dir, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    optimisation_results_dir = os.path.join(results_dir, "optimisation_results")
    if not os.path.exists(optimisation_results_dir):
        os.makedirs(optimisation_results_dir)

    mh_results_dir = os.path.join(results_dir, "mh_results")
    if not os.path.exists(mh_results_dir):
        os.makedirs(mh_results_dir)

    tmp_dir = config.get("paths", "tmp")
    if not os.path.isabs(tmp_dir):
        tmp_dir = os.path.join(cwd, tmp_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    # print("tmp_dir", tmp_dir)

    z3_path = config.get("paths", "z3_path")

    try:
        refine_timeout = int(config.get("settings", "refine_timeout"))
    except configparser.NoOptionError:
        refine_timeout = 3600

    my_config = {"prism_path": prism_path, "models": model_dir, "properties": property_dir, "data": data_dir,
                 "results": results_dir, "prism_results": prism_results, "storm_results": storm_results,
                 "data_intervals": data_intervals_dir, "data_weights": data_weights_dir,
                 "constraints_dir": constraints_dir, "refinement_results": refinement_results, "figures": figures_dir,
                 "optimisation_results": optimisation_results_dir, "figures_dir": figures_dir,
                 "mh_results": mh_results_dir, "tmp": tmp_dir, "z3_path": z3_path, "refine_timeout": refine_timeout,
                 "cwd": cwd}

    ## Interval settings
    try:
        n_samples = config.get("settings", "number_of_samples")
        my_config["n_samples"] = int(n_samples)
    except configparser.NoOptionError:
        my_config["n_samples"] = 100
    try:
        confidence_level = config.get("settings", "confidence_level")
        my_config["confidence_level"] = float(confidence_level)
    except configparser.NoOptionError:
        my_config["confidence_level"] = 0.95

    # Space sampling setting
    try:
        grid_size = config.get("settings", "grid_size")
        my_config["grid_size"] = int(grid_size)
    except configparser.NoOptionError:
        my_config["grid_size"] = 10

    try:
        my_config["store_unsat_samples"] = config.get("settings", "store_unsat_samples").lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']
    except configparser.NoOptionError:
        my_config["store_unsat_samples"] = True

    # Space refinement setting
    try:
        max_depth = config.get("settings", "max_depth")
        my_config["max_depth"] = int(max_depth)
    except configparser.NoOptionError:
        my_config["max_depth"] = 15
    try:
        coverage = config.get("settings", "coverage")
        my_config["coverage"] = float(coverage)
    except configparser.NoOptionError:
        my_config["coverage"] = 0.9
    try:
        alg = config.get("settings", "algorithm")
        my_config["alg"] = int(alg)
    except configparser.NoOptionError:
        my_config["alg"] = 4
    try:
        solver = str(config.get("settings", "solver"))
        my_config["solver"] = solver
    except configparser.NoOptionError:
        my_config["solver"] = "z3"
    try:
        delta = config.get("settings", "delta")
        my_config["delta"] = float(delta)
    except configparser.NoOptionError:
        my_config["delta"] = 0.01
    try:
        refinement_timeout = config.get("settings", "refine_timeout")
        my_config["refinement_timeout"] = float(refinement_timeout)
    except configparser.NoOptionError:
        my_config["refinement_timeout"] = 7200

    # Metropolis-Hastings setting
    try:
        mh_iterations = config.get("settings", "iterations")
        my_config["mh_iterations"] = int(mh_iterations)
    except configparser.NoOptionError:
        my_config["mh_iterations"] = 50000

    try:
        mh_grid_size = config.get("settings", "mh_grid_size")
        my_config["mh_grid_size"] = int(mh_grid_size)
    except configparser.NoOptionError:
        my_config["mh_grid_size"] = 20

    try:
        burn_in = config.get("settings", "burn_in")
        my_config["burn_in"] = float(burn_in)
    except configparser.NoOptionError:
        my_config["burn_in"] = 0.25

    try:
        mh_timeout = config.get("settings", "mh_timeout")
        my_config["mh_timeout"] = int(mh_timeout)
    except configparser.NoOptionError:
        my_config["mh_timeout"] = 3600

    # Meta setting
    try:
        save = config.get("settings", "show_progress").lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']
        my_config["show_progress"] = save
    except configparser.NoOptionError:
        my_config["show_progress"] = True

    try:
        save = config.get("settings", "autosave_figures").lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']
        my_config["save"] = save
    except configparser.NoOptionError:
        my_config["save"] = True

    try:
        silent = config.get("settings", "minimal_output").lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']
        my_config["silent"] = silent
    except configparser.NoOptionError:
        my_config["silent"] = False

    try:
        debug = config.get("settings", "extensive_output").lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']
        my_config["debug"] = debug
    except configparser.NoOptionError:
        my_config["debug"] = False

    try:
        show_mh_metadata = config.get("settings", "show_mh_metadata").lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']
        my_config["show_mh_metadata"] = show_mh_metadata
    except configparser.NoOptionError:
        my_config["show_mh_metadata"] = True

    os.chdir(current_directory)

    return my_config
