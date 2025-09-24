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
        
#include "struct_datagenvals.hpp"
#include "struct_mcmcdraws.hpp"
#include "struct_othervals.hpp"

#include "latent_state_related.hpp"
#include "log_of_norm_cdf.hpp"
#include "mvnorm.hpp"
#include "run_mcmc.hpp"
#include "sampling.hpp"
#include "sampling_generic.hpp"

#include <armadillo>

#include <vector>
#include <algorithm> // for std::min, std::max, std::max_element
#include <cmath> // for std::log, std::sqrt, std::exp

// functions for sampling block 2 for this model

std::vector<double> calc_part_of_conds(
            const arma::umat & design_matrix,
            arma::vec beta_j, arma::uword h, arma::uword H) {
    std::vector<double> results;
    // for the outer loop
    arma::urowvec d_u(H);
    arma::vec beta_j_less_h(H - 1);
    // for the inner loop
    arma::urowvec d_v(H);
    double dotprod;
    beta_j_less_h = beta_j;
    beta_j_less_h(0) = 0;
    beta_j_less_h(h - 1) = 0;
    // 0-based
    arma::urowvec d_u_copy;
    for (arma::uword i = 1; i < H; ++i) {
        d_u = design_matrix.row(i);
        d_u(0) = 0;
        d_u(h - 1) = 0;
        arma::uvec ids = arma::find(d_u == 1);
        for (auto const &j: ids) {
            d_v = design_matrix.row(j);
            d_v(0) = 0;
            if (d_v(h - 1) == 0) {
                dotprod = arma::dot(d_u - d_v, beta_j_less_h);
                results.push_back(-1.0 * dotprod);                    
            }
        }
    }
    return results;    
}

double calc_L(std::vector<double> & part_of_conds) {
    double overall_max = *std::max_element(part_of_conds.begin(),
                                           part_of_conds.end());
    return overall_max;
}

bool check_cond1(std::vector<double> & part_of_conds) {
    double overall_max = *std::max_element(part_of_conds.begin(),
                                           part_of_conds.end());
    if (overall_max <= 0) {
        return true;
    } else {
        return false;
    }
}

// b is the larger value. takes in numbers a and b,
// calculates log(Phi(b)) and log(Phi(a)),
// and reutrns Phi(b) - Phi(a)
double calc_cdf_diff(double b, double a) {
    double y = log_of_norm_cdf(b);
    double x = log_of_norm_cdf(a);
    double result = y + std::log(-std::expm1(x - y));
    return std::exp(result);
}

// missing data-related
arma::uword sample_from_cdf(const arma::vec & my_cdf) {
    arma::uword len_my_cdf = my_cdf.n_elem;
    double u = arma::randu();
    arma::uword i = 0;
    while (i < len_my_cdf && u > my_cdf[i]) {
        i += 1;
    }
    // deal with edge case
    if (i == len_my_cdf) {
        i -= 1;
    }
    return i;
}

// missing data-related
void sample_Ymat_missing(OtherVals & othervals,
                         MCMCDraws & draws,
                         arma::umat & Ymat,
                         const arma::uword & draw_number) {
    arma::uword md_respondent_ids_num = othervals.md_respondent_ids.n_elem;
    arma::mat dMat = get_design_vectors_from_alpha(draws.alpha_current,
                                                   othervals.design_matrix,
                                                   othervals.basis_vector);
    arma::mat dMat_beta = dMat * draws.beta.slice(draw_number);
   #pragma omp parallel for
    for (arma::uword i = 0; i < md_respondent_ids_num; ++i) {
        arma::uvec md_missing_rows = othervals.md_missing_row_nums(i);
        arma::uword md_missing_rows_num = md_missing_rows.n_elem;
        for (arma::uword z = 0; z < md_missing_rows_num; ++z) {
            arma::uword my_row_num = md_missing_rows(z);
            const arma::uword & J = othervals.dimensions.at("J");
            for (arma::uword j = 1; j <= J; ++j) {
                arma::uword M_j = othervals.M_j_s(j - 1);
                double dntbj = dMat_beta(my_row_num, j - 1);
                arma::rowvec kappa_j = draws.kappa(j - 1).row(draw_number);
                arma::vec the_probs(M_j);
                for (arma::uword m = 0; m < M_j; ++m) {
                    // do the calc
                    double b = kappa_j(m + 1) - dntbj;
                    double a = kappa_j(m) - dntbj;
                    the_probs(m) = calc_cdf_diff(b, a);
                }
                // create cdf
                arma::vec my_cdf = arma::cumsum(the_probs);
                Ymat(my_row_num, j - 1) = sample_from_cdf(my_cdf);
            }
        }
    }
}

void sample_beta_and_delta(DatagenVals & datagenvals,
                           const OtherVals & othervals,
                           MCMCDraws & draws,
                           arma::uword draw_number) {
    const arma::uword & J = othervals.dimensions.at("J");
    const arma::uword & H = othervals.dimensions.at("H");
    double omega = draws.omega(draw_number - 1);
    arma::mat dMat = get_design_vectors_from_alpha(draws.alpha_previous,
                                                   othervals.design_matrix,
                                                   othervals.basis_vector);
    arma::mat dMat_prime_dMat = arma::trans(dMat) * dMat;
    arma::mat dMat_prime_y_star = arma::trans(dMat) * draws.Ymat_star_current;
    double sigma_beta_sq = othervals.fixed_constants.at("sigma_beta_sq");
    double sigma_beta = std::sqrt(sigma_beta_sq);
    // mat to write over, which we assign to draws struct at the end
    arma::umat delta_draw = draws.delta.slice(draw_number - 1);
    // enter loops
    // note: the vector we use for beta is really the previous block beta,
    //       whose elements are, one-by-one, either replaced or left unchanged.
    //       so we set our (new) beta_j to previous_beta_j, and work through
    //       element-by-element.
    // Note: Parallelized with OpenMP
    #pragma omp parallel for
    for (arma::uword j = 1; j <= J; ++j) {
        arma::vec beta_j = draws.beta.slice(draw_number - 1).col(j - 1);
        arma::ucolvec delta_j = draws.delta.slice(
            draw_number - 1).col(j - 1);
        for (arma::uword h = 1; h <= H; ++h) {
            arma::uword delta_hj_prime;
            double beta_hj_prime;
            std::vector<double> part_of_conds = calc_part_of_conds(
                othervals.design_matrix, beta_j, h, H);
            double L = calc_L(part_of_conds);
            if (h == 1) {
                L = -1 * arma::datum::inf;
            }
            // Calculate c_2 and then c_1
            double c_2_sq_denom = dMat_prime_dMat(h - 1,
                                                  h - 1) + (1 / sigma_beta_sq);
            double c_2_sq = 1 / c_2_sq_denom;
            double c_2 = std::sqrt(c_2_sq);
            arma::vec beta_j_tmp = beta_j;
            beta_j_tmp(h - 1) = 0;
            arma::vec c_1_tmp = dMat_prime_y_star.col(j - 1) -
                (dMat_prime_dMat * beta_j_tmp);
            double c_1 = c_2_sq * c_1_tmp(h - 1);
            double c_1_sq = c_1 * c_1;
            if (check_cond1(part_of_conds)) { // it is not a point mass
                double log_numer_part_1 = std::log(omega) -
                    std::log(arma::normcdf(-1.0 * L / sigma_beta, 0.0, 1.0)) +
                    std::log(c_2) - std::log(sigma_beta);                
                double log_numer_part_2 = (c_1_sq / (2 * c_2_sq)) +
                    log_of_norm_cdf(-(L - c_1) / c_2);
                double log_numer = log_numer_part_1 + log_numer_part_2;
                double s_array[2];
                s_array[0] = log_numer;
                s_array[1] = std::log(1 - omega);
                double log_denom = log_sum_exp(s_array, 2);
                double omega_tilde = std::exp(log_numer - log_denom);
                delta_hj_prime = sample_bernoulli_distn(omega_tilde);
            } else { // cond1 not satisfied, so it's a point-mass
                     // distribution at 1
                delta_hj_prime = 1;
            }
            if (h == 1) {
                delta_hj_prime = 1;
            }

            // choose correct delta if custom_delta is enabled
            if (othervals.custom_delta == true) {
                delta_hj_prime = othervals.custom_delta_matrix(h - 1, j - 1);
            }
            // now that delta_hj has been sampled, sample beta_hj            
            if (delta_hj_prime == 0) {
                beta_hj_prime = 0;
            } else {
                if (h == 1) {
                    beta_hj_prime = arma::randn(arma::distr_param(c_1, c_2));
                    if (std::isnan(beta_hj_prime)) {
                        dMat_prime_y_star.print("dMat_prime_y_star:");
                    }
                } else {
                    beta_hj_prime = sample_normal_truncated(c_1, c_2, L,
                                                            arma::datum::inf);
                    if (std::isnan(beta_hj_prime)) {
                        dMat_prime_y_star.print("dMat_prime_y_star:");
                    }
                }
            }
            // assign the sampled values
            beta_j.at(h - 1) = beta_hj_prime;
            delta_j.at(h - 1) = delta_hj_prime;
        }
        draws.beta.slice(draw_number).col(j - 1) = beta_j;
        draws.delta.slice(draw_number).col(j - 1) = delta_j;
    }
}

// helper functions for sampling block 3 for this model

SigmaRelatVals perform_sigma_calculations(const OtherVals & othervals,
                                          const arma::mat & Sigma) {
    const arma::uword & K = othervals.dimensions.at("K");
    SigmaRelatVals sigmarelatvals;
    sigmarelatvals.Sigma_neg_k_neg_k_s = arma::field<arma::mat>(K);
    sigmarelatvals.Sigma_k_neg_k_s = arma::field<arma::mat>(K);
    sigmarelatvals.Sigma_neg_k_k_s = arma::field<arma::mat>(K);
    sigmarelatvals.Sigma_k_k_s = arma::vec(K);
    // do sigma calculations
    // initialize Sigma temporary variables
    arma::mat Sigma_neg_k_neg_k(K - 1, K - 1);
    arma::mat Sigma_k_neg_k(1, K - 1);
    arma::mat Sigma_neg_k_k(K - 1, 1);
    double Sigma_k_k;
    // do the sigma_k calculations, one for each k
    arma::uvec vec_of_indices;
    for (arma::uword k = 1; k <= K; ++k) { // recall always K >= 2
        arma::uvec uvec_k = {k - 1};
        vec_of_indices = calc_vec_of_indices(k, K);
        Sigma_neg_k_neg_k = Sigma.submat(vec_of_indices,
                                         vec_of_indices);
        Sigma_k_neg_k = Sigma.submat(uvec_k, vec_of_indices);
        Sigma_neg_k_k = Sigma.submat(vec_of_indices, uvec_k);
        Sigma_k_k = Sigma(k - 1, k - 1);
        // assign results to components of fields
        sigmarelatvals.Sigma_neg_k_neg_k_s(k - 1) = Sigma_neg_k_neg_k;
        sigmarelatvals.Sigma_k_neg_k_s(k - 1) = Sigma_k_neg_k;
        sigmarelatvals.Sigma_neg_k_k_s(k - 1) = Sigma_neg_k_k;
        sigmarelatvals.Sigma_k_k_s(k - 1) = arma::as_scalar(Sigma_k_k);
    }
    return sigmarelatvals;
}

// longit
arma::uword draw_alpha_nk_t(const OtherVals & othervals,
                            MCMCDraws & draws,
                            std::vector<double> cond_mean_and_sqrt,
                            const arma::mat & my_Xmat,
                            const arma::umat & my_alpha,
                            const arma::mat & my_alpha_star_expa,
                            const arma::mat & lambda_expa,
                            const arma::mat & xi_expa,
                            arma::uword n, arma::uword k,
                            arma::uword N, arma::uword t, arma::uword T,
                            const arma::uword draw_number) {
    const arma::uword & J = othervals.dimensions.at("J");
    arma::uword L_k = othervals.L_k_s(k - 1);
    arma::urowvec alpha_n_t = my_alpha.row(t - 1);
    arma::urowvec art_alpha_n_t = alpha_n_t;
    arma::vec prob_art_alpha_nk_t_values(L_k); // to store the values
    arma::vec log_multinoulli_cdf_values(L_k);
    double s_array[L_k];
    const arma::mat & Sigma = draws.Sigma.slice(draw_number - 1);
    // aliases
    const arma::mat & beta = draws.beta.slice(draw_number);
    const arma::rowvec & Ymat_star_n_t = draws.Ymat_star_current.row(
        N * (t - 1) + n - 1);
    const arma::rowvec & gamma_expa_k = draws.gamma_expa(
        k - 1).row(draw_number - 1);
    // do work
    for (arma::uword i = 0; i <= L_k - 1; ++i) { // 0-based (inherently)
        art_alpha_n_t(k - 1) = i;
        arma::mat art_d_n_t = get_design_vectors_from_alpha(
            art_alpha_n_t, othervals.design_matrix, othervals.basis_vector);
        arma::vec art_d_n_t_beta_trans = arma::trans(art_d_n_t * beta);
        arma::vec y_n_star_t_trans = arma::trans(Ymat_star_n_t);
        arma::vec single_log_norm_pdf = arma::log_normpdf(
            y_n_star_t_trans, art_d_n_t_beta_trans, arma::ones(J));
        double calc_part_1 = arma::sum(single_log_norm_pdf);
        double log_cdf1 = log_of_norm_cdf((gamma_expa_k(i + 1) - 
                                           cond_mean_and_sqrt[0]) /
                                          cond_mean_and_sqrt[1]);
        double log_cdf2 = log_of_norm_cdf((gamma_expa_k(i) -
                                           cond_mean_and_sqrt[0]) /
                                          cond_mean_and_sqrt[1]);
        double log_of_cdf_diff = calc_log_of_normal_cdf_difference(log_cdf1,
                                                                   log_cdf2);
        if (t < T) {
            arma::rowvec alpha_n_star_t_plus_one_expa = my_alpha_star_expa.row(
                t);
            // calculate my_mean
            arma::rowvec d_t = get_design_vectors_from_alpha(
                my_alpha.row(t - 1), othervals.design_matrix_trans,
                othervals.basis_vector);
            // note in the following line that "t" means something different
            //     when used as an index (indexing uses 0-base in C++)
            //     and in math notation (our t's range from 1 through T)
            arma::rowvec my_mean = my_Xmat.row(t) * lambda_expa + d_t * xi_expa;
            double calc_part_2 = log_mvnorm_pdf(alpha_n_star_t_plus_one_expa,
                                                my_mean, Sigma);
            s_array[i] = calc_part_1 + calc_part_2 + log_of_cdf_diff;
        } else {
            s_array[i] = calc_part_1 + log_of_cdf_diff;
        }
    }
    double log_c = -1.0 * log_sum_exp(s_array, L_k);    
    for (arma::uword i = 0; i < L_k; ++i) {
        log_multinoulli_cdf_values(i) = log_c + log_sum_exp(s_array, i + 1);
    }
    double log_u = std::log(arma::randu());
    arma::uword i = 0;
    while (log_multinoulli_cdf_values(i) < log_u) {
        ++i;
    }
    // deal with the rare numerical issue that occurs even on the log scale
    //     (i.e. all cdf values are 0)
    if (i > L_k - 1) {
        i = L_k - 1;
    }
    arma::uword alpha_nk_t = i;
    return alpha_nk_t;
}

std::vector<double> calc_cond_mean_and_sqrt_crosssec(
            SigmaRelatVals sigmarelatvals,
            const arma::rowvec & data_n,
            const arma::mat & alpha_star_expa_draw,
            const arma::mat & alpha_star_slope_expa,
            const arma::uvec & inner_vec_of_indices,
            arma::uword n, arma::uword k, arma::uword N) {
    // initialize variable
    std::vector<double> cond_mean_and_sqrt;
    // do calculations
    arma::vec alpha_star_slope_expa_k = alpha_star_slope_expa.col(
        k - 1);
    arma::mat alpha_star_slope_expa_neg_k = alpha_star_slope_expa.cols(
        inner_vec_of_indices);
    arma::vec b = arma::inv(sigmarelatvals.Sigma_neg_k_neg_k_s(k - 1)) *
        sigmarelatvals.Sigma_neg_k_k_s(k - 1);
    double a_part_one = arma::as_scalar(data_n * alpha_star_slope_expa_k);
    double a_part_two = arma::as_scalar(data_n *
                                        alpha_star_slope_expa_neg_k * b);
    double a = a_part_one - a_part_two;
    double Sigma_kknegk_part_two = arma::as_scalar(
        sigmarelatvals.Sigma_k_neg_k_s(k - 1) *
        arma::inv(sigmarelatvals.Sigma_neg_k_neg_k_s(k - 1)) *
        sigmarelatvals.Sigma_neg_k_k_s(k - 1));
    double Sigma_kknegk = sigmarelatvals.Sigma_k_k_s(k - 1) -
        Sigma_kknegk_part_two;
    arma::rowvec myrow = alpha_star_expa_draw.row(n - 1);
    arma::rowvec myrow_two = myrow.cols(inner_vec_of_indices);
    double my_mean = a + arma::as_scalar(myrow_two * b);
    // store results in variable
    cond_mean_and_sqrt.push_back(my_mean);
    cond_mean_and_sqrt.push_back(sqrt(Sigma_kknegk));
    return cond_mean_and_sqrt;
}

std::vector<double> calc_cond_mean_and_sqrt_longit(
            SigmaRelatVals sigmarelatvals,
            const arma::rowvec & X_n_t,
            const arma::rowvec & d_n_t_minus_one,
            const arma::rowvec & alpha_star_expa_row,
            const arma::mat & lambda_expa,
            const arma::mat & xi_expa,
            const arma::uvec & inner_vec_of_indices,
            arma::uword k) {
    // initialize variable
    std::vector<double> cond_mean_and_sqrt;
    // do calculations
    arma::vec lambda_expa_k = lambda_expa.col(k - 1);
    arma::mat lambda_expa_neg_k = lambda_expa.cols(inner_vec_of_indices);
    arma::vec xi_expa_k = xi_expa.col(k - 1);
    arma::mat xi_expa_neg_k = xi_expa.cols(inner_vec_of_indices);
    arma::vec b = arma::inv(sigmarelatvals.Sigma_neg_k_neg_k_s(k - 1)) *
        sigmarelatvals.Sigma_neg_k_k_s(k - 1);
    double a_part_one = arma::as_scalar(X_n_t * lambda_expa_k +
        d_n_t_minus_one * xi_expa_k);
    double a_part_two = arma::as_scalar(
        (X_n_t * lambda_expa_neg_k + d_n_t_minus_one * xi_expa_neg_k) * b);
    double a = a_part_one - a_part_two;
    double Sigma_kknegk_part_two = arma::as_scalar(
        sigmarelatvals.Sigma_k_neg_k_s(k - 1) *
        arma::inv(sigmarelatvals.Sigma_neg_k_neg_k_s(k - 1)) *
        sigmarelatvals.Sigma_neg_k_k_s(k - 1));
    double Sigma_kknegk = sigmarelatvals.Sigma_k_k_s(k - 1) -
        Sigma_kknegk_part_two;
    arma::rowvec myrow_two = alpha_star_expa_row.cols(inner_vec_of_indices);
    double my_mean = a + arma::as_scalar(myrow_two * b);
    // store results in variable
    cond_mean_and_sqrt.push_back(my_mean);
    cond_mean_and_sqrt.push_back(sqrt(Sigma_kknegk));
    return cond_mean_and_sqrt;
}

// block 3 for this model
void sample_alpha_and_alpha_star_expa_longit(
    DatagenVals & datagenvals,
            const OtherVals & othervals, MCMCDraws & draws,
            const arma::mat & Xmat, const arma::uword & draw_number) {
    const arma::uword & T = othervals.dimensions.at("T");
    const arma::uword & N = othervals.dimensions.at("N");
    const arma::uword & K = othervals.dimensions.at("K");
    const arma::uword & D = othervals.dimensions.at("D");
    const arma::uword & H_otr = othervals.dimensions.at("H_otr");
    arma::mat Sigma = draws.Sigma.slice(draw_number - 1);
    arma::mat lambda_expa = draws.lambda_expa.slice(draw_number - 1);
    arma::mat xi_expa = draws.xi_expa.slice(draw_number - 1);
    arma::mat alpha_star_expa_draw = draws.alpha_star_expa_previous;
    arma::umat alpha_draw = draws.alpha_previous;
    SigmaRelatVals sigmarelatvals = perform_sigma_calculations(
        othervals, Sigma);
    #pragma omp parallel for
    for (arma::uword n = 1; n <= N; ++n) {
        arma::uvec my_indices = arma::regspace<arma::uvec>(n - 1, N, N*T - 1);
        arma::umat my_alpha(T, K);
        my_alpha = alpha_draw.rows(my_indices);
        arma::mat my_alpha_star_expa(T, K);
        my_alpha_star_expa = alpha_star_expa_draw.rows(my_indices);
        arma::mat my_Xmat(T, D);
        my_Xmat = Xmat.rows(my_indices);
        for (arma::uword t = 1; t <= T; ++t) {
            for (arma::uword k = 1; k <= K; ++k) {
                arma::rowvec d_n_t_minus_one(H_otr);
                if (t == 1) {
                    d_n_t_minus_one.zeros();
                } else {
                    d_n_t_minus_one = get_design_vectors_from_alpha(
                        my_alpha.row((t - 1) - 1),
                        othervals.design_matrix_trans,
                        othervals.basis_vector);
                }
                const arma::rowvec & gamma_expa_k = draws.gamma_expa(
                    k - 1).row(draw_number - 1);
                arma::uvec inner_vec_of_indices = calc_vec_of_indices(k, K);
                std::vector<double> cond_mean_and_sqrt = 
                    calc_cond_mean_and_sqrt_longit(
                        sigmarelatvals,
                        my_Xmat.row(t - 1),
                        d_n_t_minus_one,
                        my_alpha_star_expa.row(t - 1),
                        lambda_expa, xi_expa, inner_vec_of_indices, k);
                arma::uword alpha_nk_t = draw_alpha_nk_t(
                    othervals,
                    draws,
                    cond_mean_and_sqrt,
                    my_Xmat,
                    my_alpha,
                    my_alpha_star_expa,
                    lambda_expa,
                    xi_expa,
                    n, k,
                    N, t, T,
                    draw_number);
                my_alpha(t - 1, k - 1) = alpha_nk_t;
                double alpha_star_expa_nk_t = sample_normal_truncated(
                    cond_mean_and_sqrt[0], cond_mean_and_sqrt[1],
                    gamma_expa_k(alpha_nk_t), gamma_expa_k(alpha_nk_t + 1));
                my_alpha_star_expa(t - 1, k - 1) = alpha_star_expa_nk_t;
            }
        }
        alpha_draw.rows(my_indices) = my_alpha;
        alpha_star_expa_draw.rows(my_indices) = my_alpha_star_expa;
    }
    draws.alpha_current = alpha_draw;
    draws.alpha_star_expa_current = alpha_star_expa_draw;
}

// block 4 for this model
void sample_gamma_expa(const OtherVals & othervals,
                       MCMCDraws & draws,
                       const arma::uword & draw_number) {
    const arma::uword & K = othervals.dimensions.at("K");
    for (arma::uword k = 0; k < K; ++k) { // 0-based
        double gamma_expa_kl_lb, gamma_expa_kl_ub;
        double max_alpha_star_expas, min_alpha_star_expas;
        arma::uword L_k = othervals.L_k_s(k);
        arma::mat gamma_expa_k_draws_mat = draws.gamma_expa(k);
        arma::rowvec gamma_expa_k = gamma_expa_k_draws_mat.row(draw_number - 1);
        const arma::uvec & alpha_k = draws.alpha_current.col(k);
        const arma::vec & alpha_star_expa_k = draws.alpha_star_expa_current.col(
            k);
        arma::uvec ids;
        if (L_k > 2) {
            for (arma::uword l = 2; l < L_k; ++l) { // 0-based
                // calculate lower bound of truncated exponential
                ids = arma::find(alpha_k == l - 1);
                if (ids.n_elem == 0) {
                    gamma_expa_kl_lb = gamma_expa_k(l - 1);
                } else {
                    max_alpha_star_expas = arma::max(
                        alpha_star_expa_k.elem(ids));
                    gamma_expa_kl_lb = std::max(max_alpha_star_expas,
                                                gamma_expa_k(l - 1));
                }
                // calculate upper bound of truncated exponential
                ids = arma::find(alpha_k == l);
                if (ids.n_elem == 0) {
                    gamma_expa_kl_ub = gamma_expa_k(l+1);
                } else {
                    min_alpha_star_expas = arma::min(
                        alpha_star_expa_k.elem(ids));
                    gamma_expa_kl_ub = std::min(min_alpha_star_expas,
                                                gamma_expa_k(l+1));
                }
                // now sample gamma_{kl}
                if (l < L_k - 1) {
                    gamma_expa_k(l) = arma::randu(
                        arma::distr_param(gamma_expa_kl_lb, gamma_expa_kl_ub));
                } else {
                    gamma_expa_k(l) = sample_exponential_truncated(
                        0.001, gamma_expa_kl_lb, gamma_expa_kl_ub);
                }
            }
            draws.gamma_expa(k).row(draw_number) = gamma_expa_k;
        } else {
            draws.gamma_expa(k).row(draw_number) = draws.gamma_expa(k).row(
                draw_number - 1); // will just be (-inf, constant, inf)
        }
    }
}

// block 5, longit
void sample_Sigma_and_zeta_expa(const OtherVals & othervals,
                                MCMCDraws & draws,
                                const arma::mat & Xmat,
                                const arma::mat & d_one_through_T_minus_one,
                                const arma::uword & draw_number) {
    // prepare variables
    const arma::uword & T = othervals.dimensions.at("T");
    const arma::uword & N = othervals.dimensions.at("N");
    const arma::uword & K = othervals.dimensions.at("K");
    const arma::uword & D = othervals.dimensions.at("D");
    const arma::uword & H_otr = othervals.dimensions.at("H_otr");
    const arma::mat & alpha_star_expa = draws.alpha_star_expa_current;
    // do work
    arma::mat Wmat = prepare_w(Xmat, othervals, draws.alpha_current);
    //// sample Sigma
    arma::mat Imat(D + H_otr, D + H_otr);
    Imat.eye();
    arma::mat W_trans_W = arma::trans(Wmat) * Wmat;
    // our formula is M = M_1^{-1} * M_2, so we use arma::solve
    arma::mat L_2_hat = arma::solve(W_trans_W + Imat,
                                    arma::trans(Wmat) * alpha_star_expa);
    if (draw_number == 1 || draw_number == 2) {
        Wmat.save(othervals.replic_path + "/" + "Wmat_number_" +
                  std::to_string(draw_number) + ".txt",
                  arma::arma_ascii);
    }
    if (draw_number >= 1 && draw_number <= 5) {
        L_2_hat.save(othervals.replic_path + "/" + "L_2_hat_number_" +
                  std::to_string(draw_number) + ".txt",
                  arma::arma_ascii);
    }
    arma::mat my_mat_1 = alpha_star_expa - Wmat * L_2_hat;
    arma::mat S = arma::trans(my_mat_1) * my_mat_1 +
        arma::trans(L_2_hat) * L_2_hat;
    arma::mat V_0(K, K);
    V_0.eye();
    arma::mat my_mean = arma::symmatu(S + V_0);
    arma::uword v_0 = K + 1;
    //// sample from inverse Wishart after symmetrizing
    arma::mat Sigma = arma::iwishrnd(my_mean, N*T + v_0);
    draws.Sigma.slice(draw_number) = Sigma;
    //// sample zeta
    arma::mat zeta_expa(D + H_otr, K);
    arma::mat Psi = arma::inv(W_trans_W + Imat);
    zeta_expa = sample_matrix_variate_normal(L_2_hat, Psi, Sigma, D + H_otr, K);
    draws.lambda_expa.slice(draw_number) = zeta_expa.rows(0, D - 1);
    draws.xi_expa.slice(draw_number) = zeta_expa.rows(D, D + H_otr - 1);
}

// block 6 for this model
void sample_omega(const OtherVals & othervals, MCMCDraws & draws,
                  const arma::uword & draw_number) {
    const arma::uword & H = othervals.dimensions.at("H");
    const arma::uword & J = othervals.dimensions.at("J");
    const double & omega_0 = othervals.fixed_constants.at("omega_0");
    const double & omega_1 = othervals.fixed_constants.at("omega_1");
    // do real work
    double a, b;
    double sum_delta = arma::accu(draws.delta.slice(draw_number));
    a = sum_delta + omega_0;
    b = H * J - sum_delta + omega_1;
    draws.omega(draw_number) = sample_beta_distn(a, b);
}
