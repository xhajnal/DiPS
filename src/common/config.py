import configparser
import os


def load_config():
    config = configparser.ConfigParser()
    workspace = os.path.dirname(__file__)
    current_directory = os.getcwd()
    os.chdir(workspace)

    config.read(os.path.join(workspace, "../../config.ini"))
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

    refine_timeout = config.get("settings", "refine_timeout")

    os.chdir(current_directory)

    return {"prism_path": prism_path, "models": model_dir, "properties": property_dir, "data": data_dir,
            "results": results_dir, "prism_results": prism_results, "storm_results": storm_results,
            "data_intervals": data_intervals_dir, "constraints_dir": constraints_dir,
            "refinement_results": refinement_results, "figures": figures_dir,
            "optimisation_results": optimisation_results_dir, "figures_dir": figures_dir,
            "mh_results": mh_results_dir, "tmp": tmp_dir, "z3_path": z3_path, "refine_timeout": refine_timeout}
