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

from probitlcmlongit import _core

import pathlib

import numpy as np


# my functions

def check_results_list(results_list, results_string):
    for idx, result in enumerate(results_list):
        zscores = None
        zscores = result[:, 1]
        zscore = zscores[0]
        if (zscore < -2.0) or (zscore > 2.0):
            str_to_add = f"failed: {idx}. zscore: "
            results_string += str_to_add
        results_string += str(zscore)
        results_string += "\n"
    return results_string

def check_results_dict(results_dict, results_string):
    for key, result in results_dict.items():
        zscores = None
        zscores = result[:, 1]
        zscore = zscores[0]
        if (zscore < -2.0) or (zscore > 2.0):
            str_to_add = f"failed: {key}. zscore: "
            results_string += str_to_add
        results_string += str(zscore)
        results_string += "\n"
    return results_string

def check_beta(replic_path, results_string, burnin):
    draws_beta = _core.load_arma_cube_np(
        str(replic_path.joinpath("draws_beta.abin")))
    draws_delta = _core.load_arma_ucube_np(
        str(replic_path.joinpath("draws_delta.abin")))
    n_rows = draws_beta.shape[0]
    n_cols = draws_beta.shape[1]
    results_dict = dict()
    for x in range(0, n_rows):
        for y in range(0, n_cols):
            my_key = (x, y)
            results_dict[my_key] = geweke(
                draws_beta[x, y, burnin:], 0.1, 0.5, 2)
    results_string += "checking beta results:\n"
    results_string = check_results_dict(results_dict, results_string)
    results_string += "\n"
    return results_string

def check_gammas(replic_path, results_string, data, burnin):
    K = data["K"]
    results_list = list()
    for k in range(1, K + 1):
        if data["L_k_s"][k - 1] > 2:
            fname = f"draws_gamma_{k}.abin"
            gamma_fullmat = _core.load_arma_mat_np(
                str(replic_path.joinpath(fname)))
            gamma_submat = np.delete(gamma_fullmat, [0, 1, -1], 1)
            n_cols = gamma_submat.shape[1]
            for i in range(n_cols):
                results_list.append(geweke(gamma_submat[burnin:, i],
                                           0.1, 0.5, 2))
    results_string += "checking gamma results:\n"
    results_string = check_results_list(results_list, results_string)
    results_string += "\n"
    return results_string

def check_omega(replic_path, results_string, burnin):
    fname = "draws_omega.abin"
    draws_omega = _core.load_arma_mat_np(
        str(replic_path.joinpath(fname))).ravel()
    draws_omega = draws_omega[burnin:]
    results_list = list()
    results_list.append(geweke(draws_omega, 0.1, 0.5, 2))
    results_string += "checking omega results:\n"
    results_string = check_results_list(results_list, results_string)
    results_string += "\n"
    return results_string

def check_simple_matrix_param(replic_path, results_string, param_name,
                              burnin):
    fname = f"draws_{param_name}.abin"
    draws_param = _core.load_arma_cube_np(str(replic_path.joinpath(fname)))
    n_rows = draws_param[:, :, 0].shape[0]
    n_cols = draws_param[:, :, 0].shape[1]
    results_dict = dict()
    for x in range(0, n_rows):
        for y in range(0, n_cols):
            my_key = (x, y)
            results_dict[my_key] = geweke(draws_param[x, y, burnin:],
                                          0.1, 0.5, 2)
    results_string += f"checking {param_name} results:\n"
    results_string = check_results_dict(results_dict, results_string)
    results_string += "\n"
    return results_string
    
def run_geweke_test(replic_path, data):
    burnin = data["burnin"]
    results_string = ""
    results_string = check_beta(replic_path, results_string,
                                burnin)
    results_string = check_gammas(replic_path, results_string,
                                  data, burnin)
    results_string = check_omega(replic_path, results_string, burnin)
    results_string = check_simple_matrix_param(replic_path,
                                               results_string,
                                               "Rmat",
                                               burnin)
    results_string = check_simple_matrix_param(replic_path,
                                               results_string,
                                               "lambda",
                                               burnin)
    if data["T"] > 1:
        results_string = check_simple_matrix_param(
            replic_path, results_string, "xi", burnin)
    fpath = replic_path.joinpath("convergence_test_results.txt")
    with open(fpath, "w") as f:
        f.writelines(results_string)
