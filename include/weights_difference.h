/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef FAIR_TOPK_WEIGHTSDIFFERENCE_H
#define FAIR_TOPK_WEIGHTSDIFFERENCE_H

#include <array>
#include <utility>
#include <type_traits>

#include <soplex.h>

namespace FairTopK {

static constexpr double soplexEpsilon = 1e-8;

template <int... index>
consteval std::array<int, sizeof...(index)> getIndexSequence(std::integer_sequence<int, index...>&& seqs) noexcept {
    return { index... };
}

template <std::size_t N>
consteval std::array<double, N> getConstArray(double val) noexcept {
    std::array<double, N> arr;
    for (std::size_t i = 0; i < N; i++)
        arr[i] = val;
    return arr;
}

template <int dimension>
std::array<double, 3 * dimension> setupWeightsDiffLPAuxVector(const Eigen::VectorXd& weights, double margin) {
    constexpr int count = 3 * dimension;
    std::array<double, count> weightsDiffLPAuxVector;

    int idx = 0;
    for (int i = 0; i < dimension - 1; i++) {
        double lb = std::max(0.0, weights(i) - margin);
        double ub = std::min(1.0, weights(i) + margin);
        weightsDiffLPAuxVector[idx++] = lb;
        weightsDiffLPAuxVector[idx++] = ub;
    }
    {
        double lb = std::max(0.0, weights(dimension - 1) - margin);
        double ub = std::min(1.0, weights(dimension - 1) + margin);

        weightsDiffLPAuxVector[idx++] = 1.0 - ub;
        weightsDiffLPAuxVector[idx++] = 1.0 - lb;
    }

    for (int i = 0; i < dimension - 1; i++) {
        weightsDiffLPAuxVector[idx++] = weights(i);
    }
    weightsDiffLPAuxVector[idx++] = 1.0 - weights(dimension - 1);

    return weightsDiffLPAuxVector;
}

template <int dimension>
using WeightsDiffLPAuxVector = std::invoke_result_t<decltype(setupWeightsDiffLPAuxVector<dimension>), const Eigen::VectorXd&, double>;

template <int dimension>
inline void setUpWeightsDiffLPWeightVars(const WeightsDiffLPAuxVector<dimension>& auxiliaryVector, 
    const soplex::DSVector &dummyCol, soplex::SoPlex& soplexSolver) {
    for (int i = 0; i < dimension - 1; i++) {
        soplexSolver.addColReal(soplex::LPCol(0.0, dummyCol, auxiliaryVector[2 * i + 1], auxiliaryVector[2 * i]));
    }
}

template <int totalVarsCount, int dimension> requires (totalVarsCount - dimension >= dimension - 1)
void setUpWeightsDiffLPConstrs(const WeightsDiffLPAuxVector<dimension>& weightsDiffLPAuxVector, soplex::DSVector& row,
    soplex::SoPlex& soplexSolver) {
    constexpr std::array<double, dimension - 1> ones = FairTopK::getConstArray<dimension - 1>(1.0);

    constexpr std::array<int, dimension - 1> indexArray =
        FairTopK::getIndexSequence(std::make_integer_sequence<int, dimension - 1>{});

    constexpr int offset = 2 * dimension;

    //On the last weight
    row.add(dimension - 1, indexArray.data(), ones.data());
    soplexSolver.addRowReal(soplex::LPRow(weightsDiffLPAuxVector[offset - 2], row, weightsDiffLPAuxVector[offset - 1]));
    row.clear();

    constexpr int boundConstrVarsCount = 2;
    
    std::array<int, boundConstrVarsCount> boundConstrVars;
    constexpr std::array<double, boundConstrVarsCount> coeffsPos = { 1.0, -1.0 };
    constexpr std::array<double, boundConstrVarsCount> coeffsNeg = { 1.0, 1.0 };

    constexpr int boundConstrVarsOffset = totalVarsCount - dimension;

    for (int i = 0; i < dimension - 1; i++) {
        boundConstrVars[0] = i;
        boundConstrVars[1] = i + boundConstrVarsOffset;

        double constant = weightsDiffLPAuxVector[offset + i];

        row.add(boundConstrVarsCount, boundConstrVars.data(), coeffsPos.data());
        soplexSolver.addRowReal(soplex::LPRow(row, soplex::LPRow::LESS_EQUAL, constant));
        row.clear();

        row.add(boundConstrVarsCount, boundConstrVars.data(), coeffsNeg.data());
        soplexSolver.addRowReal(soplex::LPRow(row, soplex::LPRow::GREATER_EQUAL, constant));
        row.clear();
    }

    {
        double constant = weightsDiffLPAuxVector[offset + dimension - 1];

        row.add(dimension - 1, indexArray.data(), ones.data());
        row.add(boundConstrVarsOffset + dimension - 1, 1.0);
        soplexSolver.addRowReal(soplex::LPRow(row, soplex::LPRow::GREATER_EQUAL, constant));
        row.clear();

        row.add(dimension - 1, indexArray.data(), ones.data());
        row.add(boundConstrVarsOffset + dimension - 1, -1.0);
        soplexSolver.addRowReal(soplex::LPRow(row, soplex::LPRow::LESS_EQUAL, constant));
        row.clear();
    }
}

}

#endif