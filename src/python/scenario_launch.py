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
import shutil
import argparse
import subprocess
import json
import itertools
import tomllib

## other modules
import numpy as np

## my modules
from probitlcmlongit import scenario_run_single_replic
from probitlcmlongit import report_helpers
from probitlcmlongit import run_helpers

from probitlcmlongit import _core

def scenario_launch_setup(environ, sim_info_dir_name,
                          scenarionumber):
    # set up paths
    run_dir = pathlib.Path.cwd()
    # load config file
    with open("config_simulation.toml", "rb") as fileObj:
        config = tomllib.load(fileObj)
    if environ == "laptop":
        process_dir = config["laptop_process_dir"]
    elif environ == "cluster":
        process_dir = config["cluster_process_dir"]
    process_dir = pathlib.Path(process_dir)
    sim_info_dir_name = pathlib.Path(sim_info_dir_name)
    process_dir = process_dir.joinpath(sim_info_dir_name)
    process_dir.mkdir(parents=True, exist_ok=True)
    ## run all replications for this particular scenario
    scenarionumber = int(scenarionumber)
    # zero-based scenario number for generating seeds
    scenarionumber_zb = scenarionumber - 1
    jsonfilename_stem = f"scenario_{scenarionumber:04}"
    number_of_replics = config["number_of_replics"]
    seed_value_scenario = number_of_replics * scenarionumber_zb
    # copy all files over to proper process_dir
    param_files_path = run_dir.joinpath(sim_info_dir_name, "scenario_files",
                                        jsonfilename_stem)
    scenario_path = process_dir.joinpath(jsonfilename_stem)
    shutil.copytree(param_files_path, scenario_path)
    # get other json files path
    other_json_files_path = run_dir.joinpath(sim_info_dir_name, "json_files")
    return(scenario_path, jsonfilename_stem, process_dir,
           other_json_files_path, number_of_replics, scenarionumber_zb)

def scenario_launch_laptop(jsonfilename_stem,
                           process_dir, other_json_files_path,
                           number_of_replics,
                           scenario_number_zb,
                           hyperparam_tuning = False,
                           tuning_path = "",
                           missing_data = 0):
    scenario_path = process_dir.joinpath(jsonfilename_stem)
    scenario_datagen_params_path = scenario_path.joinpath("datagen_params")
    for replicnum in range(1, number_of_replics + 1):
        scenario_run_single_replic.main(jsonfilename_stem,
                                        other_json_files_path,
                                        scenario_path,
                                        scenario_datagen_params_path,
                                        scenario_number_zb,
                                        replicnum,
                                        number_of_replics,
                                        hyperparam_tuning,
                                        tuning_path,
                                        missing_data)

def scenario_launch_cluster(jsonfilename_stem,
                            process_dir, other_json_files_path,
                            number_of_replics,
                            scenario_number_zb,
                            missing_data):
    # note that for environ cluster, must check that all runs are complete
    #     before generating final report
    scenario_path = process_dir.joinpath(jsonfilename_stem) 
    scenario_datagen_params_path = scenario_path.joinpath("datagen_params")
    for replicnum in range(1, number_of_replics + 1):
        args_list = list()
        args_list.append(f"JSONFILENAMESTEM={jsonfilename_stem}")
        args_list.append(f"OTHERJSONFILESPATH={other_json_files_path}")
        args_list.append(f"SCENARIOPATH={scenario_path}")
        args_list.append(
            f"SCENARIODATAGENPARAMSPATH={scenario_datagen_params_path}")
        args_list.append(f"SCENARIONUMBERZB={scenario_number_zb}")
        args_list.append(f"REPLICNUM={replicnum}")
        args_list.append(f"NUMBEROFREPLICS={number_of_replics}")
        args_list.append(f"MISSINGDATA={missing_data}")
        part_string = ",".join(args_list)
        input_string = "--export" + "=" + part_string
        subprocess.run(["sbatch", input_string, "simulation.slurm"])

## main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environ", choices = ["laptop", "cluster"],
                        required=True)
    parser.add_argument("--sim_info_dir_name", required=True)
    parser.add_argument("--scenarionumber", required=True)
    parser.add_argument("--missing_data", choices = ["1", "0"])
    # parse args
    args = parser.parse_args()
    environ = args.environ
    sim_info_dir_name = args.sim_info_dir_name
    scenarionumber = args.scenarionumber
    missing_data = int(args.missing_data)
    # run scenario_launch_setup
    scenario_path, jsonfilename_stem, process_dir, \
        other_json_files_path, number_of_replics, \
        scenarionumber_zb = scenario_launch_setup(environ,
                                                  sim_info_dir_name,
                                                  scenarionumber)
    # run replications
    hyperparam_tuning = False
    tuning_path = process_dir.joinpath("hyperparam_tuning")
    if environ == "laptop":
        scenario_launch_laptop(jsonfilename_stem, process_dir,
                               other_json_files_path,
                               number_of_replics, scenarionumber_zb,
                               hyperparam_tuning,
                               tuning_path,
                               missing_data)
    elif environ == "cluster":
        scenario_launch_cluster(jsonfilename_stem, process_dir,
                                other_json_files_path,
                                number_of_replics, scenarionumber_zb,
                                missing_data)
