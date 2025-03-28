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

#include <string>
#include <cmath> // for std::pow

#include <armadillo>
#include <nlohmann/json.hpp>

#include "latent_state_related.hpp"
#include "struct_othervals.hpp"
#include "file_io.hpp"
#include "run_mcmc.hpp"

void calculate_waic(int setup_num,
                    std::string dataset_dir,
                    std::string data_analysis_path,
                    std::string chain_results_path,
                    int chain_number) {
    // set up OtherVals
    OtherVals othervals;
    othervals.stream_number = 1;
    othervals.stream_slice_ctr = 1;
    othervals.stream_max_val = 100;
    othervals.thinning_interval = 10;
    // continuing
    arma::umat Ymat;
    arma::field<std::string> empty_header(0);
    std::string fpath = dataset_dir + "/" + "responses.csv";
    Ymat.load(arma::csv_name(fpath, empty_header,
                             arma::csv_opts::strict));
    arma::mat Xmat;
    fpath = dataset_dir + "/" + "covariates.csv";
    Xmat.load(arma::csv_name(fpath, empty_header,
                             arma::csv_opts::strict));
    // read in the json file
    // find fname
    std::string fnamestem = "setup_";
    int padded_length = 4;
    std::string s = std::to_string(setup_num);
    s.insert(s.begin(), padded_length - s.size(), '0');
    std::string fname = fnamestem + s + ".json";
    // bring fname into path
    std::string fulljsonfilename = data_analysis_path + "/" + fname;
    std::ifstream fstream_for_json(fulljsonfilename);
    nlohmann::json json_object = nlohmann::json::parse(fstream_for_json);

    // merge in other json files
    fulljsonfilename = dataset_dir + "/" + "01_fixed_vals.json";
    nlohmann::json json_object_two = read_json_file(fulljsonfilename);
    json_object.update(json_object_two);
    
    //// load dimension-related quantities
    const arma::uword T = json_object.at("T");
    const arma::uword K = json_object.at("K");
    const arma::uword N = Ymat.n_rows / T;
    const arma::uword J = Ymat.n_cols;
    std::vector<arma::uword> tmpvec;
    tmpvec = json_object.at("L_k_s").get<std::vector<arma::uword>>();
    arma::uvec L_k_s(tmpvec);
    // find M_j_s and set it in othervals
    arma::uvec M_j_s(J);
    for (arma::uword j = 1; j <= J; ++j) {
        // in the following line, we must add 1 since the responses are 0-based
        M_j_s(j - 1) = arma::max(Ymat.col(j - 1)) + 1;
    }
    const arma::uword burnin = json_object.at("burnin");
    const arma::uword chain_length_after_burnin = json_object.at(
        "chain_length_after_burnin");
    const arma::uword total_chain_length = burnin + chain_length_after_burnin;
    const arma::uword order = json_object.at("order");
    arma::uvec pos_to_remove; // may be empty
    if (order < K) {
        std::string filename = data_analysis_path + "/" + "pos_to_remove.txt";
        pos_to_remove.load(filename);
    }
    // set random seed equal to zero-based chain number
    int seed_value = chain_number - 1;
    arma::arma_rng::set_seed(seed_value);
    // prepare other required dimension quantities
    const arma::uword H_K = arma::prod(L_k_s);
    arma::uword n_param_draws = chain_length_after_burnin /
        othervals.thinning_interval;
    // in C++, rounds down by default
    // load draws data
    arma::cube draws_beta;
    arma::field<arma::mat> kappa(J);
    fpath = chain_results_path + "/" + "draws_beta.txt";
    draws_beta.load(fpath);
    for (arma::uword j = 1; j <= J; ++j) { 
        fname = pad_string_with_zeros(3, "draws_kappa_",
                                                  std::to_string(j));
        fpath = chain_results_path + "/" + fname;
        kappa(j - 1).load(fpath);
    }
    // begin logging
    std::ofstream log_file(chain_results_path + "/" + "log_waic.txt");
    arma::mat lik(N, n_param_draws);
    arma::mat log_lik(N, n_param_draws);
    arma::mat log_lik_sq(N, n_param_draws);
    // alpha_chunk_variable
    arma::ucube alpha_chunk;
    fname = pad_string_with_zeros(3, "alpha_chunk_",
                                  std::to_string(othervals.stream_number));
    fpath = chain_results_path + "/" + fname;
    alpha_chunk.load(fpath);
    // generate design matrix
    arma::umat alphas;
    fpath = data_analysis_path + "/" + "alphas_table.txt";
    alphas.load(fpath);
    othervals.alphas_table = alphas;
    othervals.design_matrix = generate_design_matrix(alphas,
                                                     K,
                                                     L_k_s,
                                                     order,
                                                     pos_to_remove);
    othervals.basis_vector = calculate_basis_vector(L_k_s, K);
    double final_value = 0;
    arma::uword my_idx = 1;
    for (arma::uword s = burnin;
                     s < total_chain_length;
                     // ++s) {
                     s += othervals.thinning_interval) {
        double log_current_step_value = 0;
        double y_nj_t;
        const arma::mat & beta = draws_beta.slice(s);
        const arma::umat alpha_chunk_slice = alpha_chunk.slice(
            othervals.stream_slice_ctr - 1);
        arma::mat dMat = get_design_vectors_from_alpha(alpha_chunk_slice,
                                                       othervals.design_matrix,
                                                       othervals.basis_vector);
        arma::mat dMat_beta = dMat * beta;
        // #pragma omp parallel for
        for (arma::uword n = 1; n <= N; ++n) {
            double current_n_val = 0;
            for (arma::uword t = 1; t <= T; ++t) {
                arma::urowvec y_n_t = Ymat.row((t - 1) * N + n - 1);
                for (arma::uword j = 1; j <= J; ++j) {
                    const arma::rowvec & kappa_j = kappa(j - 1).row(
                        s - 1);
                    double y_nj_t = y_n_t(j - 1);
                    double d_n_t_beta_j = dMat_beta((t - 1) * N + n - 1, j - 1);
                    current_n_val += std::log(
                        arma::normcdf(
                            kappa_j(y_nj_t + 1) - d_n_t_beta_j) -
                        arma::normcdf(kappa_j(y_nj_t) - d_n_t_beta_j));
                }
            }
            lik(n - 1, my_idx - 1) = std::exp(current_n_val);
            log_lik(n - 1, my_idx - 1) = current_n_val;
            log_lik_sq(n - 1, my_idx - 1) = std::pow(current_n_val, 2);
        }
        // load appropriate alpha
        // counters
        if (othervals.stream_slice_ctr == othervals.stream_max_val) {
            othervals.stream_slice_ctr = 1;
            othervals.stream_number += 1;
            if (othervals.stream_number <= (
                    chain_length_after_burnin /
                    (othervals.stream_max_val * othervals.thinning_interval)
                    )
                ) {
                fname = pad_string_with_zeros(3, "alpha_chunk_",
                                              std::to_string(
                                                  othervals.stream_number));
                fpath = chain_results_path + "/" + fname;
                alpha_chunk.load(fpath);
            }
        } else {
            othervals.stream_slice_ctr += 1;
        }
        my_idx += 1;
    }
    // save log liks
    fpath = chain_results_path + "/" + "likelihood_mat.txt";
    lik.save(fpath, arma::arma_ascii);
    fpath = chain_results_path + "/" + "log_likelihood_mat.txt";
    log_lik.save(fpath, arma::arma_ascii);
    fpath = chain_results_path + "/" + "log_likelihood_sq_mat.txt";
    log_lik_sq.save(fpath, arma::arma_ascii);
    // manuscript did not use the following code, which was based on derivation
    //     by the first author, but rather used the waic function from the
    //     R package entitled loo.
    // now calculate waic based on those values
    //// first calculate p_waic_hat
    // arma::vec vec1 = arma::sum(log_lik_sq, 1);
    // arma::vec vec2 = arma::sum(log_lik, 1);
    // arma::vec result1 = (1.0 / n_param_draws) * vec1 - \
    //     arma::pow((1.0 / n_param_draws) * vec2, 2);
    // arma::vec result2 = (n_param_draws / (n_param_draws - 1)) * result1;
    // double p_waic_hat = arma::sum(result2);
    // //// now calculate lpd_hat
    // arma::vec vec3 = (1.0 / n_param_draws) * arma::sum(lik, 1);
    // double lpd_hat = arma::sum(arma::log(vec3));
    // double elpd_waic_hat = lpd_hat - p_waic_hat;
    // arma::vec waic_results(3);
    // waic_results(0) = lpd_hat;
    // waic_results(1) = p_waic_hat;
    // waic_results(2) = elpd_waic_hat;
    // fpath = chain_results_path + "/" + "waic_results.txt";
    // waic_results.save(fpath, arma::arma_ascii);
}
