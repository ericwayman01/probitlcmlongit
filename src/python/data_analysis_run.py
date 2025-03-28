# This file is part of "probitlcmlongit" which is released under GPL v3.
#
# Copyright (c) 2022-2025 Eric Alan Wayman <ericwaymanpublications@mailworks.org>.
#
# This program is FLO (free/libre/open) software: you can redistribute
# it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

## standard library
import pathlib
import argparse
import subprocess
import shutil
import time # for data_analysis_run_cluster in environ cluster
import json
import tomllib

## other modules
import numpy as np

## my modules
from probitlcmlongit import data_analysis_run_single_chain
from probitlcmlongit import run_helpers
from probitlcmlongit import report_helpers

from probitlcmlongit import _core

def data_analysis_setup(environ, dataset, setup_num):
    # same format for config file    
    run_dir = pathlib.Path.cwd()
    # load config file
    with open("config_data_analysis.toml", "rb") as fileObj:
        config = tomllib.load(fileObj)
    if environ == "laptop":
        process_dir = config["laptop_process_dir"]
    elif environ == "cluster":
        process_dir = config["cluster_process_dir"]
    process_dir = pathlib.Path(process_dir)
    ## run all chains for this data analysis
    dataset = dataset
    setup_num = int(setup_num)
    jsonfilename_stem = f"setup_{setup_num:04}"
    # set up path to dataset_dir
    dataset_dir = run_dir.joinpath(dataset)
    # set up data_analysis_path (like the process_dir of scenario_launch)
    process_dir = process_dir.joinpath(dataset)
    data_analysis_path = process_dir.joinpath(jsonfilename_stem)
    data_analysis_path.mkdir(parents=True, exist_ok=True)
    # copy json file to data_analysis_path (a directory)
    jsonfilename = jsonfilename_stem + ".json"
    jsonfile_src_path = run_dir.joinpath(dataset, jsonfilename)
    jsonfile_dest_path = data_analysis_path.joinpath(jsonfilename)
    shutil.copy2(jsonfile_src_path, jsonfile_dest_path)    
    setup_dict = json.loads(jsonfile_dest_path.read_bytes())
    # load fixed vals dict
    fpath = run_dir.joinpath(dataset, "01_fixed_vals.json")
    fixed_other_vals_dict = json.loads(fpath.read_text())
    setup_dict.update(fixed_other_vals_dict)
    # continue
    alphas_table = run_helpers.create_alphas_table(setup_dict)
    # get mydict to run functions
    mydict = dict() # a dict consisting of keys of max levels and values
                    #     that are lists of dimensions with that max level
    L_k_s = setup_dict["L_k_s"]
    for k, x in enumerate(L_k_s):
        if x not in mydict:
            mydict[x] = list()
        mydict[x].append(k)
    # run functions
    pos_to_remove_and_effects_tables = \
        run_helpers.find_pos_to_remove_and_effects_tables(
            setup_dict, fixed_other_vals_dict, mydict, alphas_table)
    save_arma_umat_given_fname(alphas_table,
                               data_analysis_path, "alphas_table.txt")
    if "pos_to_remove" in pos_to_remove_and_effects_tables:
        save_arma_umat_given_fname(
            pos_to_remove_and_effects_tables["pos_to_remove"],
            data_analysis_path,
            "pos_to_remove.txt")
    save_arma_umat_given_fname(
        pos_to_remove_and_effects_tables["effects_table"],
        data_analysis_path,
        "effects_table.txt")
    # deal with transition model: pos_to_remove_trans and effects_table_trans
    if "pos_to_remove_trans" in pos_to_remove_and_effects_tables:
        save_arma_umat_given_fname(
            pos_to_remove_and_effects_tables["pos_to_remove_trans"],
            data_analysis_path,
            "pos_to_remove_trans.txt")
    save_arma_umat_given_fname(
        pos_to_remove_and_effects_tables["effects_table_trans"],
        data_analysis_path,
        "effects_table_trans.txt")
    # continue
    number_of_chains = config["number_of_chains"]
    return(setup_num, dataset_dir,
           process_dir, data_analysis_path, number_of_chains)

def save_arma_umat_given_fname(my_umat, data_analysis_path, fname):
    fpath = data_analysis_path.joinpath(fname)
    _core.save_arma_umat_np(my_umat, str(fpath))

def data_analysis_run_laptop(setup_num, dataset_dir,
                             data_analysis_path, number_of_chains,
                             hyperparam_tuning, tuning_path, custom_delta):
    for chainnum in range(1, number_of_chains + 1):
        data_analysis_run_single_chain.main(setup_num,
                                            dataset_dir,
                                            data_analysis_path,
                                            chainnum,
                                            hyperparam_tuning, tuning_path,
                                            custom_delta)

def data_analysis_run_cluster(setup_num, dataset_dir,
                              data_analysis_path, number_of_chains):
    for chainnum in range(1, number_of_chains + 1):
        args_list = list()
        args_list.append(f"SETUPNUM={setup_num}")
        args_list.append(f"DATASETDIR={dataset_dir}")
        args_list.append(f"DATAANALYSISPATH={data_analysis_path}")
        args_list.append(f"CHAINNUM={chainnum}")
        part_string = ",".join(args_list)
        input_string = "--export" + "=" + part_string
        subprocess.run(["sbatch", input_string, "data_analysis.slurm"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environ", choices = ["laptop", "cluster"],
                        required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--setup_num", required=True)
    parser.add_argument("--custom_delta", choices = ["1", "0"],)
    # parse args
    args = parser.parse_args()
    environ = args.environ
    dataset = args.dataset
    setup_num = int(args.setup_num)
    custom_delta = int(args.custom_delta)
    # run data_analysis_setup
    setup_num, dataset_dir, \
        process_dir, data_analysis_path, \
        number_of_chains = data_analysis_setup(
            environ, dataset, setup_num)
    # run chains
    hyperparam_tuning = False
    tuning_path = process_dir.joinpath("hyperparam_tuning")
    if args.environ == "laptop":
        data_analysis_run_laptop(setup_num, dataset_dir,
                                 data_analysis_path, number_of_chains,
                                 hyperparam_tuning, tuning_path,
                                 custom_delta)
    elif args.environ == "cluster":
        data_analysis_run_cluster(setup_num, dataset_dir,
                                  data_analysis_path, number_of_chains)
