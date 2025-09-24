/*
 * This file is part of "probitlcm" which is released under GPL v3.
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

#ifndef HEADERS_CALCULATE_WAIC_HPP_
#define HEADERS_CALCULATE_WAIC_HPP_

// missing data-related
arma::uvec create_giant_uvec_of_missing_row_nums(
        arma::uword,
        arma::field<arma::uvec> &);

void calculate_waic(int,
                    std::string,
                    std::string,
                    std::string,
                    int,
                    int);

#endif
