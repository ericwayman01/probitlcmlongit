from probitlcmlongit import _core

# for func_ptr_table
from probitlcmlongit.funcs_across_draws import geweke
from probitlcmlongit.funcs_across_draws import integrated_autocorr2
from probitlcmlongit.results_class import Results

import numpy as np

import argparse, itertools, json, pathlib, pickle, tomllib

def geweke_curried(x):
    result = geweke(x, 0.1, 0.5, 2)
    zscores = result[:, 1]
    return zscores[0]

def integrated_autocorr2_processed(x):
    return integrated_autocorr2(x).item()

def process_cube(func_ptr, results_obj, burnin,
                 replic_path, param_name, replic_num):
    subtract_dict = {
        "my_lambda": "lambda"
    }
    fname = None
    if param_name == "my_lambda":
        fname = "draws_" + subtract_dict[param_name] + ".abin"
    else:
        fname = "draws_" + param_name + ".abin"
    fpath = replic_path.joinpath(fname)
    my_data = _core.load_arma_cube_np(str(fpath))
    print("my_data.shape", my_data.shape)
    x_max = my_data.shape[0]
    y_max = my_data.shape[1]
    for i in range(0, x_max):
        for j in range(0, y_max):
            results_obj[i, j, replic_num - 1] = \
                func_ptr(my_data[i, j, burnin:])

# this could be almost called "process mat" but it's not that general:
# the first two and last columns are deleted, and the function also
# accounts for the different filename conventions that were used
def process_threshold(func_ptr, results_obj, burnin,
                      replic_path, param_name, replic_num, dim_num):
    str_num_part = None
    if param_name == "kappas":
        str_num_part = f"{dim_num:03}"
    else:
        str_num_part = str(dim_num)
    subtract_dict = {
        "gammas": "gamma",
        "kappas": "kappa"
    }
    fname = "draws_" + subtract_dict[param_name] + "_" + str_num_part + ".abin"
    fpath = replic_path.joinpath(fname)
    my_data = _core.load_arma_mat_np(str(fpath))
    my_data = np.delete(my_data, [0, 1, -1], 1)
    for i in range(0, len(my_data[1])):
        results_obj[replic_num - 1, i] = \
            func_ptr(my_data[burnin:, i])

def process_vec(func_ptr, results_obj, burnin,
                replic_path, param_name, replic_num):
    fname = "draws_" + param_name + ".abin"
    fpath = replic_path.joinpath(fname)
    my_data = _core.load_arma_vec_np(str(fpath)).ravel()
    results_obj[replic_num - 1] = func_ptr(my_data[burnin:])


def load_json_data(dirname, type_of_run, process_dir_path, results_dir_name):
    jsonfilename = dirname + ".json"
    jsonfile_src_path  = process_dir_path.joinpath(jsonfilename)
    json_data = json.loads(jsonfile_src_path.read_bytes())
    ## load fixed_vals
    run_path = pathlib.Path(results_dir_name)
    jsonfile_src_path = None
    if type_of_run == "simulation":
        jsonfile_src_path = run_path.joinpath("json_files",
                                              "01_fixed_vals.json")
    elif type_of_run == "data_analysis":
        jsonfile_src_path = run_path.joinpath("01_fixed_vals.json")        
    json_data2 = json.loads(jsonfile_src_path.read_bytes())
    json_data.update(json_data2)
    # do more additions
    if type_of_run == "simulation":
        q = json_data["q"]
        per_effect_M_j_s = json_data["per_effect_M_j_s"]
        M_j_s = list(itertools.chain.from_iterable(
            itertools.repeat(M_j, q) for M_j in per_effect_M_j_s))
        json_data["M_j_s"] = M_j_s
    elif type_of_run == "data_analysis":
        jsonfile_src_path = run_path.joinpath("M_j_s.json")
        M_j_s = json.loads(jsonfile_src_path.read_bytes())
        json_data["M_j_s"] = M_j_s
    json_data["L"] = json_data["L_k_s"][0]
    ## load H and H_otr
    effects_dict = dict()
    fpath = process_dir_path.joinpath("effects_table.txt")
    effects_table = _core.load_arma_umat_np(str(fpath))
    effects_dict["effects_list"] = [str(xx) for xx in effects_table]
    json_data["H"] = len(effects_dict["effects_list"])
    fpath = process_dir_path.joinpath("effects_table_trans.txt")
    effects_table = _core.load_arma_umat_np(str(fpath))
    effects_dict["effects_list_trans"] = [str(xx) for xx in effects_table]
    json_data["H_otr"] = len(effects_dict["effects_list_trans"])
    return json_data


def initialize_results_obj(json_data, params_list, type_of_run, n_replics):
    # create local vars for readability
    chain_length_after_burnin = json_data["chain_length_after_burnin"]
    burnin = json_data["burnin"]
    J = json_data["J"]
    K = json_data["K"]
    L = json_data["L_k_s"][0]
    D = None
    if type_of_run == "simulation":
        if json_data["covariates"] == "age_assignedsex":
            D = 2 + 1
    elif type_of_run == "data_analysis":
        D = json_data["covariates"] + 1
    # J = q * len(per_effect_M_j_s)
    M_j_s = json_data["M_j_s"]
    H = json_data["H"]
    H_otr = json_data["H_otr"]
    # begin initialization
    results = Results()
    # do an initialization step with one replic
    for x in params_list:
        if x == "beta":
            results.beta = np.empty((H, J, n_replics))
        elif x == "gammas":
            for k in range(1, K + 1):
                results.gammas[k] = np.empty((n_replics, L + 1 - 3))
        elif x == "kappas":
            for j in range(1, J + 1):
                M_j = M_j_s[j - 1]
                results.kappas[j] = np.empty((n_replics, M_j + 1 - 3))
        elif x == "my_lambda":
            results.my_lambda = np.empty((D, K, n_replics))
        elif x == "Rmat":
            results.Rmat = np.empty((K, K, n_replics))
        elif x == "xi":
            results.xi = np.empty((H_otr, K, n_replics))
        elif x == "omega":
            results.omega = np.empty((n_replics))
    return results

def process_replics(results, dtypes_dict, func_ptr,
                    json_data, params_list, type_of_run, n_replics,
                    process_dir_path):
    for replic_num in range(1, n_replics + 1):
        dir_name = None
        if type_of_run == "simulation":
            dir_name = f"replic_{replic_num:03}"
        elif type_of_run == "data_analysis":
            dir_name = f"chain_{chain_num:03}"
        replic_path = process_dir_path.joinpath(dir_name)
        for param_name in params_list:
            if dtypes_dict[param_name] == "cube":
                process_cube(func_ptr, getattr(results, param_name),
                             json_data["burnin"],
                             replic_path, param_name, replic_num)
            elif dtypes_dict[param_name] == "mat": # these are dicts
                dim_max = None
                match param_name:
                    case "gammas":
                        dim_max = json_data["K"]
                    case "kappas":
                        dim_max = json_data["J"]
                my_dict = getattr(results, param_name)
                for i in range(1, dim_max + 1):
                    if param_name == "kappas" and \
                            json_data["M_j_s"][i - 1] == 2:
                        pass
                    else:
                        process_threshold(func_ptr,
                                          my_dict[i],
                                          json_data["burnin"],
                                          replic_path, param_name,
                                          replic_num, i)
            elif dtypes_dict[param_name] == "vec":
                process_vec(func_ptr, getattr(results, param_name),
                            json_data["burnin"],
                            replic_path, param_name, replic_num)


def run_stuff(func_name, results_dir_name,
              environ, type_of_run, scenario_or_setup_num, chain_num):
    if environ == "cluster":
        print(("This functionality is not supported for "
               "the cluster environment at this time."))
        return None
    func_ptr_table = {
        "geweke": geweke_curried,
        "integrated_autocorr2": integrated_autocorr2_processed
    }
    func_ptr = func_ptr_table[func_name]
    # set up data
    n_replics = 1
    if type_of_run == "data_analysis":
        n_replics = 1
    # set up path and load scenario data
    config_fname = None
    if type_of_run == "simulation":
        fname = "config_simulation.toml"
    elif type_of_run == "data_analysis":
        fname = "config_data_analysis.toml"
    with open(fname, "rb") as fileObj:
        config = tomllib.load(fileObj)
    process_dir = None
    if environ == "laptop":
        process_dir = config["laptop_process_dir"]
    elif environ == "cluster":
        process_dir = config["cluster_process_dir"]
    process_dir_path = pathlib.Path(process_dir)
    process_dir_path = process_dir_path.joinpath(results_dir_name)
    dirname = None
    if type_of_run == "simulation":
        dirname = f"scenario_{scenario_or_setup_num:04}"
    elif type_of_run == "data_analysis":
        dirname = f"setup_{scenario_or_setup_num:04}"
    process_dir_path = process_dir_path.joinpath(dirname)
    json_data = load_json_data(dirname, type_of_run,
                               process_dir_path, results_dir_name)
    # load local values
    # process params
    params_list = ["beta", "gammas", "kappas", "my_lambda", "Rmat",
                   "xi", "omega"]
    if json_data["L"] <= 2:
        params_list.remove("gammas")
    results = initialize_results_obj(json_data, params_list, type_of_run,
                                     n_replics)
    dtypes_dict = {
        "beta": "cube",
        "gammas": "mat",
        "kappas": "mat",
        "my_lambda": "cube",
        "omega": "vec",
        "Rmat": "cube",
        "Sigma": "cube",
        "xi": "cube"
    }
    process_replics(results, dtypes_dict, func_ptr,
                    json_data, params_list, type_of_run, n_replics,
                    process_dir_path)
    # amalgamate results
    fname_parts = [dirname, func_name, "raw"]
    fname = "_".join(fname_parts)
    fname = fname + ".pickle"
    fpath = process_dir_path.joinpath(fname)
    with open(fpath, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for param_name in params_list:
        if dtypes_dict[param_name] == "cube":
            setattr(results, param_name,
                    np.average(getattr(results, param_name), axis=2))
        elif dtypes_dict[param_name] == "mat": # these are dicts
            my_dict = getattr(results, param_name)
            match param_name:
                case "gammas":
                    dim_max = json_data["K"]
                case "kappas":
                    dim_max = json_data["J"]
            for i in range(1, dim_max + 1):
                if param_name == "kappas" and json_data["M_j_s"][i - 1] == 2:
                    pass
                else:
                    my_dict[i] = np.average(my_dict[i], axis=0)
        elif dtypes_dict[param_name] == "vec":
            setattr(results, param_name,
                    np.average(
                        getattr(results, param_name)).item())
    # save results
    fname_parts = [dirname, func_name, "post_averaging"]
    fname = "_".join(fname_parts)
    fname = fname + ".pickle"
    fpath = process_dir_path.joinpath(fname)
    with open(fpath, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--func_name",
                        choices = ["geweke", "integrated_autocorr2"],
                        required=True)
    parser.add_argument("--results_dir_name", required=True)
    parser.add_argument("--environ",
                        choices = ["laptop", "cluster"],
                        required=True)
    parser.add_argument("--type_of_run",
                        choices = ["simulation", "data_analysis"],
                        required=True)
    parser.add_argument("--scenario_or_setup_num", required=True)
    parser.add_argument("--chain_num") # only used for data_analysis runs
    args = parser.parse_args()
    chain_num = None
    if args.type_of_run == "data_analysis" and args.chain_num is not None:
        chain_num = int(args.chain_num)
    run_stuff(args.func_name, args.results_dir_name, args.environ,
              args.type_of_run, int(args.scenario_or_setup_num),
              chain_num)
