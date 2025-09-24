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

// missing data-related
arma::uvec create_giant_uvec_of_missing_row_nums(
        arma::uword num_md_resp_ids,
        arma::field<arma::uvec> & field_of_uvecs) {
    arma::uvec all_missing_row_nums; // only used for missing data
    // convert md_missing_row_nums into one giant sorted uvec
    std::vector<arma::uword> all_missing_row_nums_stdvec;
    for (arma::uword i = 0; i < num_md_resp_ids; ++i) {
        arma::uvec tmpvec = field_of_uvecs(i);
        for (arma::uword j = 0; j < tmpvec.n_elem; ++j) {
            all_missing_row_nums_stdvec.push_back(tmpvec(j));
        }
    }
    std::sort(all_missing_row_nums_stdvec.begin(),
              all_missing_row_nums_stdvec.end());
    all_missing_row_nums = arma::conv_to<arma::uvec>::from(
        all_missing_row_nums_stdvec);
    return all_missing_row_nums;
}

void calculate_waic(int setup_num,
                    std::string dataset_dir,
                    std::string data_analysis_path,
                    std::string chain_results_path,
                    int chain_number,
                    int missing_data) {
    // set up OtherVals
    OtherVals othervals;
    othervals.stream_number = 1;
    othervals.stream_slice_ctr = 1;
    othervals.stream_max_val = 100;
    othervals.thinning_interval = 10;
    std::string fname;
    std::string fpath;
    if (missing_data == 1) {
        // load everything except md_pos_present
        //     (that was only used for build Ymat from form with empty rows,
        //     which we don't need here)
        fname = "respondent_counts.txt";
        fpath = dataset_dir + "/" + fname;
        othervals.respondent_counts.load(fpath);
        othervals.dimensions["N"] = othervals.respondent_counts.n_elem;
    }
    // read in the json file
    // find fname
    std::string fnamestem = "setup_";
    int padded_length = 4;
    std::string s = std::to_string(setup_num);
    s.insert(s.begin(), padded_length - s.size(), '0');
    fname = fnamestem + s + ".json";
    // bring fname into path
    std::string fulljsonfilename = dataset_dir + "/" + fname;
    std::ifstream fstream_for_json(fulljsonfilename);
    nlohmann::json json_object = nlohmann::json::parse(fstream_for_json);
    // merge in other json files
    fulljsonfilename = dataset_dir + "/" + "01_fixed_vals.json";
    nlohmann::json json_object_two = read_json_file(fulljsonfilename);
    json_object.update(json_object_two);
    //// load dimension-related quantities
    othervals.dimensions["T"] = json_object.at("T");
    othervals.dimensions["K"] = json_object.at("K");
    const arma::uword T = json_object.at("T");
    const arma::uword K = json_object.at("K");

    if (missing_data == 1) {
        othervals.md_pos_missing = load_missing_data_positions(
            dataset_dir, "md_pos_missing.json");
        fname = "md_respondent_ids.txt";
        fpath = dataset_dir + "/" + fname;
        othervals.md_respondent_ids.load(fpath);
        // find missing_row_nums in fanned and nonfanned forms
        othervals.md_missing_row_nums = arma::field<arma::uvec>(
            othervals.md_respondent_ids.n_elem);
        othervals.md_missing_row_nums_nonfanned = arma::field<arma::uvec>(
            othervals.md_respondent_ids.n_elem);
        find_missing_row_nums(othervals);
        othervals.md_missing_row_nums.print("md_missing_row_nums");
        find_missing_row_nums_nonfanned(othervals);
        // othervals.md_missing_row_nums_nonfanned.print("md_missing_row_nums");
        // end missing data stuff
    }
    // load data
    // notes for the case of missing data:
    //     Ymat is already unfanned, but alpha is fanned. must unfan it 
    //         after loading it
    //     Xmat is unfanned as well
    arma::umat Ymat;
    arma::field<std::string> empty_header(0);
    fpath = dataset_dir + "/" + "responses.csv";
    Ymat.load(arma::csv_name(fpath, empty_header,
                             arma::csv_opts::strict));
    arma::mat Xmat;
    fpath = dataset_dir + "/" + "covariates.csv";
    Xmat.load(arma::csv_name(fpath, empty_header,
                             arma::csv_opts::strict));
    arma::uvec all_missing_row_nums_nonfanned; // only used if missing_data == 1
    if (missing_data == 1) {
        // convert md_missing_row_nums into one giant sorted uvec
        arma::uvec all_missing_row_nums = create_giant_uvec_of_missing_row_nums(
            othervals.md_respondent_ids.n_elem,
            othervals.md_missing_row_nums);
        all_missing_row_nums.print("all_missing_row_nums:");
        all_missing_row_nums_nonfanned = create_giant_uvec_of_missing_row_nums(
                othervals.md_respondent_ids.n_elem,
                othervals.md_missing_row_nums_nonfanned);
        // drop relevant Xmat rows
        std::cout << "Xmat rows before: " << std::to_string(Xmat.n_rows)
                  << std::endl;
        Xmat.shed_rows(all_missing_row_nums);
        std::cout << "Xmat rows after: " << std::to_string(Xmat.n_rows)
                  << std::endl;
    }
    // if missing_data == 0, set othervals.dimensions["N"] since it has
    //     not yet been set
    if (missing_data == 0) {
        othervals.dimensions["N"] = Ymat.n_rows / T;            
    }
    const arma::uword N = othervals.dimensions["N"];

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
    M_j_s.print("M_j_s:");
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
    fpath = chain_results_path + "/" + "draws_beta.abin";
    draws_beta.load(fpath);
    for (arma::uword j = 1; j <= J; ++j) {
        fname = "draws_kappa_"
            + pad_string_with_zeros(std::to_string(j), 3)
            + ".abin";
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
    fname = "alpha_chunk_"
        + pad_string_with_zeros(std::to_string(othervals.stream_number), 3)
        + ".txt";
    fpath = chain_results_path + "/" + fname;
    alpha_chunk.load(fpath);
    // deal with missing data situation
    if (missing_data == 1) {
        arma::ucube alpha_reduced = arma::ucube(Xmat.n_rows,
                                                alpha_chunk.n_cols,
                                                alpha_chunk.n_slices);
        for (arma::uword i = 0; i < alpha_chunk.n_slices; ++i) {
            // rearrange from fanned to nonfanned
            arma::umat tmpmat = rearrange_data_to_nonfanned_umat(
                alpha_chunk.slice(i), N, T);
            // shed missing data rows
            tmpmat.shed_rows(all_missing_row_nums_nonfanned);
            alpha_reduced.slice(i) = tmpmat;
        }
        // overwrite alpha_chunk
        alpha_chunk = alpha_reduced;
    }
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
    arma::uword column_idx = 1;
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
        arma::uword suppl_idx = 0; // only used if missing_data == 1
        for (arma::uword n = 1; n <= N; ++n) {
            double current_n_val = 0;
            if (missing_data == 1) {
                arma::uword my_counts = othervals.respondent_counts(n - 1);
                for (arma::uword ctr = 0; ctr < my_counts; ++ctr) {
                    arma::urowvec y_n_t = Ymat.row(suppl_idx + ctr);
                    for (arma::uword j = 1; j <= J; ++j) {
                        const arma::rowvec & kappa_j = kappa(j - 1).row(
                            s - 1);
                        double y_nj_t = y_n_t(j - 1);
                        double d_n_t_beta_j = dMat_beta(suppl_idx + ctr, j - 1);
                        // std::cout << "current_n_val: " << current_n_val
                        //           << std::endl;
                        current_n_val += std::log(
                            arma::normcdf(
                                kappa_j(y_nj_t + 1) - d_n_t_beta_j) -
                            arma::normcdf(kappa_j(y_nj_t) - d_n_t_beta_j));
                    }
                    // if (ctr == 0) {
                    //     std::cout << "dude's first time point = "
                    //               << current_n_val << std::endl;
                    // }
                }
                // update overall ctr
                suppl_idx += my_counts;
            } else {
                for (arma::uword t = 1; t <= T; ++t) {
                    arma::urowvec y_n_t = Ymat.row((t - 1) * N + n - 1);
                    for (arma::uword j = 1; j <= J; ++j) {
                        const arma::rowvec & kappa_j = kappa(j - 1).row(
                            s - 1);
                        double y_nj_t = y_n_t(j - 1);
                        double d_n_t_beta_j = dMat_beta((t - 1) * N + n - 1,
                                                        j - 1);
                        current_n_val += std::log(
                            arma::normcdf(
                                kappa_j(y_nj_t + 1) - d_n_t_beta_j) -
                            arma::normcdf(kappa_j(y_nj_t) - d_n_t_beta_j));
                    }
                }
            }
            // calc liks for this respondent
            // std::cout << "n = " << n << ", "
            //           << "current_n_val = " << current_n_val << std::endl;
            lik(n - 1, column_idx - 1) = std::exp(current_n_val);
            log_lik(n - 1, column_idx - 1) = current_n_val;
            log_lik_sq(n - 1, column_idx - 1) = std::pow(current_n_val, 2);
        }
        // load latest chunk if necessary
        // counters
        if (othervals.stream_slice_ctr == othervals.stream_max_val) {
            othervals.stream_slice_ctr = 1;
            othervals.stream_number += 1;
            if (othervals.stream_number <= (
                    chain_length_after_burnin /
                    (othervals.stream_max_val * othervals.thinning_interval)
                    )
                ) {
                fname = "alpha_chunk_"
                    +  pad_string_with_zeros(
                           std::to_string(othervals.stream_number), 3)
                    + ".txt";
                fpath = chain_results_path + "/" + fname;
                alpha_chunk.load(fpath);
                if (missing_data == 1) {
                    arma::ucube alpha_reduced = arma::ucube(
                        Xmat.n_rows, alpha_chunk.n_cols, alpha_chunk.n_slices);
                    for (arma::uword i = 0; i < alpha_chunk.n_slices; ++i) {
                        // rearrange from fanned to nonfanned
                        arma::umat tmpmat = rearrange_data_to_nonfanned_umat(
                            alpha_chunk.slice(i), N, T);
                        // shed missing data rows
                        tmpmat.shed_rows(all_missing_row_nums_nonfanned);
                        alpha_reduced.slice(i) = tmpmat;
                    }
                    // overwrite alpha_chunk
                    alpha_chunk = alpha_reduced;
                }
            }
        } else {
            othervals.stream_slice_ctr += 1;
        }
        column_idx += 1;
    }
    // save log liks
    fpath = chain_results_path + "/" + "likelihood_mat.txt";
    lik.save(fpath, arma::arma_ascii);
    fpath = chain_results_path + "/" + "log_likelihood_mat.txt";
    log_lik.save(fpath, arma::arma_ascii);
    fpath = chain_results_path + "/" + "log_likelihood_sq_mat.txt";
    log_lik_sq.save(fpath, arma::arma_ascii);
}
