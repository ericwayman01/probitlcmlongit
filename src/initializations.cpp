/*
 * This file is part of "probitlcmlongit" which is released under GPL v3.
 *
 * Copyright (c) 2022-2025 Eric Alan Wayman <ericwaymanpublications@mailworks.org>.
 *
 * This program is FLO (free/libre/open) software: you can redistribute
 * it and/or modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "struct_mcmcdraws.hpp"
#include "struct_othervals.hpp"

#include "initializations.hpp"
#include "latent_state_related.hpp"
#include "sampling_generic.hpp"

#include <armadillo>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // type conversion

#include <string>
#include <cmath> // for std::ceil
#include <algorithm> // for std::sort

namespace py = pybind11;

void find_initial_alphas(std::string process_dir, std::string path_to_Ymat,
                         const int T, const int N, const int K,
                         arma::uvec L_k_s, const int seed_value) {
    std::vector<int> L_k_s_vec = arma::conv_to< std::vector<int> >::from(L_k_s);
    py::function find_initial_alphas_py =
        py::reinterpret_borrow<py::function>(
            py::module::import(
                "probitlcmlongit.exec_numpy").attr("find_initial_alphas")
            );
    find_initial_alphas_py(process_dir, path_to_Ymat,
                           T, N, K, L_k_s_vec, seed_value);
}


void initialize_star_variable(arma::mat & Ymat_star_current,
                              const arma::field<arma::mat> & thresholds_field,
                              const arma::uvec & levels_vec,
                              const arma::uword & num_of_dims,
                              const arma::umat & Ymat,
                              const arma::uword & T, const arma::uword & N) {
    // nomenclature used here is for the Ymat and Ymat_star case
    arma::uword y_nj_t;
    arma::uword idx;
    arma::uword M_j;
    arma::rowvec kappa_j;
    for (arma::uword j = 1; j <= num_of_dims; ++j) {
        M_j = levels_vec(j - 1);
        kappa_j = thresholds_field(j - 1).row(0);
        for (arma::uword t = 1; t <= T; ++t) {
            for (arma::uword n = 1; n <= N; ++n) {
                idx = N * (t - 1) + n - 1;
                y_nj_t = Ymat(idx, j - 1);
                if (y_nj_t == 0) {
                    Ymat_star_current(idx, j - 1) = -0.3;
                } else if (y_nj_t == M_j - 1) {
                    Ymat_star_current(idx, j - 1) = kappa_j(M_j - 1) + 0.3;
                } else {
                    Ymat_star_current(idx, j - 1) =
                        (kappa_j(y_nj_t + 1) - kappa_j(y_nj_t)) / 2.0;
                }
            }
        }
    }
}


void initialize_mcmc_variables(OtherVals & othervals,
                               MCMCDraws & draws,
                               arma::umat & Ymat,
                               const arma::mat & Xmat,
                               int seed_value,
                               const DatagenVals & datagenvals,
                               bool is_simulation,
                               int missing_data) {
    const arma::uword & T = othervals.dimensions.at("T");
    const arma::uword & N = othervals.dimensions.at("N");
    const arma::uword & J = othervals.dimensions.at("J");
    const arma::uword & K = othervals.dimensions.at("K");
    const arma::uword & H = othervals.dimensions.at("H");
    const arma::uword & D = othervals.dimensions.at("D");
    const arma::uword & H_K = othervals.dimensions.at("H_K");
    const arma::uword & total_chain_length = othervals.total_chain_length;
    // initializations for original model
    std::string path_to_Ymat;
    if (is_simulation) {
        path_to_Ymat = othervals.replic_path + "/" + "datagen_Ymat.txt";
    } else { // is a data analysis
        path_to_Ymat = othervals.dataset_dir + "/" + "responses.txt";
    }
    // set up Ymat_pred_chunk (draws from posterior predictive distn),
    // and alpha_chunk (the latter is for the WAIC calculation)
    if (!is_simulation) {
        draws.Ymat_pred_chunk = arma::ucube(T * N, J, othervals.stream_max_val);
        draws.alpha_chunk = arma::ucube(T * N, K, othervals.stream_max_val);
    }
    arma::uword ctr_rows = 0;
    arma::uword ctr_md = 0;
    if (missing_data == 1 && is_simulation) { // operates on fanned data
        // (recall that simulation datagen is done in fanned form)
        arma::uword my_NT = Ymat.n_rows;
        // find missing_pct position
        // k is 1-based
        arma::uword k = std::ceil(my_NT * othervals.missing_data_pct);
        // ob means 1-based
        arma::uvec all_row_ids = arma::regspace<arma::uvec>(0, my_NT - 1);
        arma::uvec all_row_ids_shuffled = arma::shuffle(all_row_ids);
        arma::uvec missing_row_ids = all_row_ids_shuffled.subvec(0, k - 1);
        // build map_of_vecs1 (eventually convert to a field of uvecs)
        // and map_of_vecs2
        std::map<arma::uword, std::vector<arma::uword>> map_of_vecs1;
        std::map<arma::uword, std::vector<arma::uword>> map_of_vecs2;
        for (arma::uword i = 0; i < missing_row_ids.n_elem; ++i) {
            int row_id = missing_row_ids(i);
            int t = row_id / N + 1;
            int n = row_id - N * (t - 1) + 1;
            if (map_of_vecs1.find(n) == map_of_vecs1.end()) {
                map_of_vecs1[n] = std::vector<arma::uword>();
            }
            map_of_vecs1[n].push_back(row_id);
            if (map_of_vecs2.find(n) == map_of_vecs2.end()) {
                map_of_vecs2[n] = std::vector<arma::uword>();
            }
            map_of_vecs2[n].push_back(t);
        }
        // take the keys of the map and put them in a uvec
        // (first put them in a vector)
        std::vector<arma::uword> my_keys;
        arma::uword ctr;
        for (
                std::map<arma::uword,
                         std::vector<arma::uword>>::iterator it =
                    map_of_vecs1.begin();
                    it != map_of_vecs1.end(); ++it) {
            my_keys.push_back(it->first);
        }
        std::sort(my_keys.begin(), my_keys.end());
        othervals.md_respondent_ids = arma::conv_to<arma::uvec>::from(my_keys);
        // build respondent_counts
        othervals.respondent_counts = arma::uvec(N);
        for (arma::uword n = 1; n <= N; ++n) {
            if (std::find(my_keys.begin(), my_keys.end(), n) == my_keys.end()) {
                othervals.respondent_counts(n - 1) = T;
            } else {
                othervals.respondent_counts(n - 1) = T - map_of_vecs2[n].size();
            }
        }
        // initialize field
        othervals.md_missing_row_nums =
            arma::field<arma::uvec>(othervals.md_respondent_ids.n_elem);
        othervals.md_missing_row_nums_nonfanned =
            arma::field<arma::uvec>(othervals.md_respondent_ids.n_elem);
        arma::uword idx = 0;
        for (const auto & x : my_keys) {
            std::vector<arma::uword> tmpvec = map_of_vecs1[x];
            std::sort(tmpvec.begin(), tmpvec.end());
            arma::uvec my_row_ids = arma::conv_to<arma::uvec>::from(tmpvec);
            othervals.md_missing_row_nums(idx) = my_row_ids;
            idx += 1;
        }
        // note: the above replaces the use of the find_missing_row_nums
        //     function
        // we still need the find_missing_row_nums_nonfanned information
        //     so we call that function now
        // first, we build md_pos_missing
        // then, we run find_missing_row_nums_nonfanned
        idx = 0;
        arma::uword my_keys_len = my_keys.size();
        othervals.md_pos_missing = arma::field<arma::uvec>(my_keys_len);
        for (const auto & x : my_keys) {
            std::vector<arma::uword> tmpvec = map_of_vecs2[x];
            std::sort(tmpvec.begin(), tmpvec.end());
            arma::uvec my_ts = arma::conv_to<arma::uvec>::from(tmpvec);
            othervals.md_pos_missing(idx) = my_ts;
            idx += 1;
        }
        find_missing_row_nums_nonfanned(othervals);
        Ymat = rearrange_data_to_nonfanned_umat(Ymat, N, T);
    }
    // do initialization
    if (missing_data == 1) { // operates on nonfanned data
        arma::uword mod_number;
        if (is_simulation) {
            mod_number = 30;
        } else {
            mod_number = 6;
        }
        arma::uword md_num = othervals.md_respondent_ids.n_elem;
        for (arma::uword i = 0; i < md_num; ++i) {
            // in this loop, pos_vec1 is the uvec of missing positions for
            //     missing data respondent i
            // pos_vec2 is the uvec missing row nums of missing data
            //     respondent i in non-fanned form
            arma::uvec pos_vec1 = othervals.md_pos_missing(i);
            arma::uword pos_vec1_len = pos_vec1.n_elem;
            arma::uvec pos_vec2 = othervals.md_missing_row_nums_nonfanned(i);
            arma::uword pos_vec2_len = pos_vec2.n_elem;
            for (arma::uword this_pos = 0; this_pos < pos_vec1_len;
                 ++this_pos) {
                arma::uword pos_vec1_this = pos_vec1(this_pos);
                if (pos_vec1_this % mod_number == 1) {
                    arma::uword j = 0;
                    bool tmp_flag = true;
                    while (tmp_flag) {
                        if (this_pos + j + 1 < pos_vec1_len) {
                            if (    pos_vec1(this_pos + j + 1) -
                                    pos_vec1(this_pos + j) > 1) {
                                tmp_flag = false;
                            } else {
                                j += 1;
                            }
                        } else {
                            tmp_flag = false;
                        }
                    }
                    Ymat.row(pos_vec2(this_pos)) = Ymat.row(
                        pos_vec2(this_pos) + j + 1);
                } else {
                    Ymat.row(pos_vec2(this_pos)) = Ymat.row(
                        pos_vec2(this_pos) - 1);
                }
            }
        }
        Ymat = rearrange_data_umat(Ymat, N, T);
    }
    //
    if (missing_data == 1) {
        std::string fpath = othervals.dataset_dir + "/" + "responses.txt";
        Ymat.save(fpath, arma::arma_ascii);
        fpath = othervals.dataset_dir + "/" + "covariates.txt";
        Xmat.save(fpath, arma::arma_ascii);
    }
    find_initial_alphas(othervals.replic_path, path_to_Ymat,
                        T, N, K,
                        othervals.L_k_s, seed_value);
    draws.alpha_current = arma::umat(T * N, K);
    draws.alpha_previous = arma::umat(T * N, K);
    draws.alpha_current.load(
        othervals.replic_path + "/" + "initial_alphas.txt");
    arma::mat Ymat_doubles = arma::conv_to<arma::mat>::from(Ymat);
    arma::mat initial_design_vectors = get_design_vectors_from_alpha(
        draws.alpha_current, othervals.design_matrix, othervals.basis_vector);
    draws.beta = arma::cube(H, J, total_chain_length);
    // estimate beta
    draws.beta.slice(0) = arma::solve(initial_design_vectors, Ymat_doubles);
    // make beta non-negative
    draws.beta.slice(0).transform( [](double val) {
        return (val < 0.0) ? 0.0 : val; } );
    draws.delta = arma::Cube<arma::uword>(H, J, total_chain_length);
    draws.delta.slice(0).ones(); // initialize delta to all ones    
    draws.kappa = arma::field<arma::mat>(J);
    prepare_thresholds(draws.kappa,
                       othervals.M_j_s,
                       J,
                       total_chain_length,
                       1.6, 2.4);
    draws.Ymat_star_current = arma::mat(T * N, J);
    initialize_star_variable(draws.Ymat_star_current,
                             draws.kappa,
                             othervals.M_j_s,
                             J,
                             Ymat,
                             T, N);
    // draws.Ymat_star_current = datagenvals.Ymat_star;
    draws.Ymat_star_previous = arma::mat(T * N, J);
    arma::mat alpha_doubles = arma::conv_to<arma::mat>::from(
        draws.alpha_current);
    draws.lambda = arma::cube(D, K, total_chain_length);
    draws.xi = arma::cube(othervals.dimensions.at("H_otr"),
                          K, total_chain_length);
    arma::mat Wmat = prepare_w(Xmat, othervals, draws.alpha_current);
    arma::mat lambda_xi_concat = arma::solve(Wmat, alpha_doubles);
    draws.lambda.slice(0) = lambda_xi_concat.rows(0, D - 1);
    draws.xi.slice(0) = lambda_xi_concat.rows(D,
                                              D + othervals.dimensions.at(
                                                  "H_otr") - 1);
    // zeta_expa
    draws.lambda_expa = arma::cube(D, K, total_chain_length);
    draws.xi_expa = arma::cube(othervals.dimensions.at("H_otr"),
                               K, total_chain_length);
    draws.lambda_expa.slice(0) = draws.lambda.slice(0);
    draws.xi_expa.slice(0) = draws.xi.slice(0);
    // initialize gamma_k's
    draws.gamma = arma::field<arma::mat>(K);
    draws.gamma_expa = arma::field<arma::mat>(K);
    prepare_thresholds(draws.gamma,
                       othervals.L_k_s,
                       K,
                       total_chain_length,
                       1.6, 2.4);
    for (arma::uword k = 1; k <= K; ++k) {
        arma::uword L_k = othervals.L_k_s(k - 1);
        draws.gamma_expa(k - 1) = arma::mat(total_chain_length, L_k + 1);
        draws.gamma_expa(k - 1).row(0) = draws.gamma(k - 1).row(0);
    }
    // alpha and alpha_star
    draws.alpha_star_current = arma::mat(T * N, K);
    initialize_star_variable(draws.alpha_star_current,
                             draws.gamma,
                             othervals.L_k_s,
                             K,
                             draws.alpha_current,
                             T, N);
    draws.alpha_star_previous = arma::mat(T * N, K);
    draws.alpha_star_expa_current = arma::mat(T * N, K);
    draws.alpha_star_expa_current = draws.alpha_star_current;
    draws.alpha_star_expa_previous = arma::mat(T * N, K);
    // Rmat
    draws.Rmat = arma::cube(K, K, total_chain_length);
    arma::mat I_K = arma::eye(K, K);
    draws.Rmat.slice(0) = I_K;
    // Sigma
    draws.Sigma = arma::cube(K, K, total_chain_length);
    draws.Sigma.slice(0) = draws.Rmat.slice(0);
    // omega
    draws.omega = arma::vec(total_chain_length);
    draws.omega(0) = 0.5;
    // initialize derived parameters
    draws.theta_j_mats_sums = arma::field<arma::mat>(J);
    for (arma::uword j = 1; j <= J; ++j) {
        draws.theta_j_mats_sums(j - 1) = arma::mat(
            H_K, othervals.M_j_s(j - 1));
        draws.theta_j_mats_sums(j - 1).zeros();
    }
    // note that this statistic is not calculated for each
    //     time point separately
    draws.class_counts = arma::umat(N * T, H_K);
    // initialize total_class_counts_per_draw
    draws.total_class_counts_per_draw = arma::umat(total_chain_length, H_K);
    draws.total_class_counts_per_draw.zeros();
    arma::uvec current_r_alpha_class_numbers = convert_alpha_to_class_numbers(
            draws.alpha_current, othervals.basis_vector);
    // set total class counts to all zeros at the start
    draws.total_class_counts_per_draw.row(
        0) = calc_total_class_counts_for_single_draw(
            othervals, current_r_alpha_class_numbers, 0);
}
