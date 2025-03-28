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

import argparse
import pathlib
import tomllib
from probitlcmlongit import _core

def calculate_waic(setup_num, dataset_dir, data_analysis_path,
                         chain_results_path, chain_number):
    _core.calculate_waic(setup_num,
                         str(dataset_dir),
                         str(data_analysis_path),
                         str(chain_results_path),
                         chain_number)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environ", choices = ["laptop", "cluster"],
                        required=True)
    parser.add_argument("--dataset", required=True) # e.g. tang
    parser.add_argument("--setup_num", required=True) # e.g. setup_0001.json
    parser.add_argument("--chain_number", required=True) # e.g. 1
    args = parser.parse_args()
    # parse config file
    run_dir = pathlib.Path.cwd()
    # load config file
    with open("config_data_analysis.toml", "rb") as fileObj:
        config = tomllib.load(fileObj)
    if args.environ == "laptop":
        process_dir = config["laptop_process_dir"]
    elif args.environ == "cluster":
        process_dir = config["cluster_process_dir"]
    process_dir = pathlib.Path(process_dir)
    # set up paths
    dataset = args.dataset
    setup_num = int(args.setup_num)
    chain_number = args.chain_number
    chain_number = int(chain_number)
    # set up path to dataset_dir
    dataset_dir = run_dir.joinpath(dataset)
    data_analysis_path = process_dir.joinpath(dataset, f"setup_{setup_num:04}")
    chain_number_dirname = f"chain_{chain_number:03}"
    chain_results_path = data_analysis_path.joinpath(chain_number_dirname)
    calculate_waic(setup_num, dataset_dir, data_analysis_path,
                   chain_results_path, chain_number)
