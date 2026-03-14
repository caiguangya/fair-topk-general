/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef FAIR_TOPK_STABILIZATION_H
#define FAIR_TOPK_STABILIZATION_H

#include <vector>
#include <array>
#include <algorithm>
#include <tuple>
#include <utility>
#include <iostream>

#include <Eigen/Dense>
#include <sdlp/sdlp.hpp>

#include "utility.h"
#include "tie_breaking.h"

namespace FairTopK {

namespace Detail {

template <int dimension>
void stabilizeFairWeightVectorStcDim(const std::vector<Eigen::VectorXd> &points, const std::vector<Groups>& groups, 
    int k, GroupsMask pGroups, const std::vector<std::pair<int, int> >& pGroupsBounds, 
    const Eigen::VectorXd& refWeights, double weightSpaceMargin, double epsilon, Eigen::VectorXd& weights);

void stabilizeFairWeightVectorDynDim(const std::vector<Eigen::VectorXd> &points, const std::vector<Groups>& groups, 
    int k, GroupsMask pGroups, const std::vector<std::pair<int, int> >& pGroupsBounds, 
    const Eigen::VectorXd& refWeights, double weightSpaceMargin, double epsilon, Eigen::VectorXd& weights);

}

template <int dimension = -1>
void stabilizeFairWeightVector(const std::vector<Eigen::VectorXd> &points, const std::vector<Groups>& groups, 
    int k, GroupsMask pGroups, const std::vector<std::pair<int, int> >& pGroupsBounds, 
    const Eigen::VectorXd& refWeights, double weightSpaceMargin, double epsilon, Eigen::VectorXd& weights) {
    if constexpr (dimension > 0) {
        Detail::stabilizeFairWeightVectorStcDim<dimension>(points, groups, k, pGroups, pGroupsBounds, 
            refWeights, weightSpaceMargin, epsilon, weights);
    }
    else {
        Detail::stabilizeFairWeightVectorDynDim(points, groups, k, pGroups, pGroupsBounds, 
            refWeights, weightSpaceMargin, epsilon, weights);
    }
}

namespace Detail {

template <int dimension> requires (dimension >= 2)
double findStableWeightVector(const std::vector<Eigen::VectorXd> &points, const std::vector<int>& topKIndices, int kthIdx,
    const std::array<double, dimension>& lbs, const std::array<double, dimension>& ubs,
    Eigen::Matrix<double, dimension, 1>& vector);

template <int dimension>
void stabilizeFairWeightVectorStcDim(const std::vector<Eigen::VectorXd> &points, const std::vector<Groups>& groups, 
    int k, GroupsMask pGroups, const std::vector<std::pair<int, int> >& pGroupsBounds, 
    const Eigen::VectorXd& refWeights, double weightSpaceMargin, double epsilon, Eigen::VectorXd& weights) {
    std::vector<std::tuple<double, Groups, int> > pts;
    int count = points.size();

    std::array<double, dimension> lbs, ubs;
    for (int i = 0; i < dimension; i++) {
        lbs[i] = std::max(0.0, refWeights(i) - weightSpaceMargin);
        ubs[i] = std::min(1.0, refWeights(i) + weightSpaceMargin);
    }

    pts.reserve(count);
    for (int i = 0; i < count; i++) {
        pts.emplace_back(weights.dot(points[i]), groups[i], i);
    }

    std::nth_element(pts.begin(), pts.begin() + (k - 1), pts.end(),
        [](const auto& p0, const auto& p1) { return std::get<0>(p0) > std::get<0>(p1); });

    double kthScore = std::get<0>(pts[k - 1]);

    int numPGroups = pGroupsBounds.size();
    auto pGroupsVec = getPGroupsVec(pGroups);

    std::vector<int> topKIndices;
    topKIndices.reserve(k);

    std::array<int, maxNumGroups> pGroupsBaseCounts{};
    
    std::vector<Groups> tiesGroups;
    std::vector<double> tiesScore;
    std::vector<int> tiesIndices;
    std::vector<int> tiesSelectedIndices;

    int vacant = 0;
    
    for (int i = 0; i < k; i++) {
        const auto [score, groups, idx] = pts[i];

        if (score - kthScore > epsilon) {
            if ((groups & pGroups) != 0) {
                updatePGroupsCounts<1>(numPGroups, groups, pGroupsVec, pGroupsBaseCounts);
            }
            topKIndices.push_back(idx);
        }
        else {
            tiesGroups.push_back(groups & pGroups);
            tiesScore.push_back(refWeights.dot(points[idx]));
            tiesIndices.push_back(idx);
            vacant += 1;
        }
    }

    for (int i = k; i < count; i++) {
        const auto [score, groups, idx] = pts[i];

        if (kthScore - score <= epsilon) {
            tiesGroups.push_back(groups & pGroups);
            tiesScore.push_back(refWeights.dot(points[idx]));
            tiesIndices.push_back(idx);
        }
    }

    tiesSelectedIndices.reserve(vacant);

    bool fair = obtainFairSelectionWithLargestUtility(vacant, pGroupsVec, pGroupsBounds,
        tiesGroups, tiesScore, tiesIndices, pGroupsBaseCounts, tiesSelectedIndices);

    if (!fair) {
        std::cerr << "Error: Failed to stabilize a fair weight vector" << std::endl;
        return;
    }

    for (auto tiesSelectedIdx : tiesSelectedIndices) topKIndices.push_back(tiesSelectedIdx);

    std::sort(topKIndices.begin(), topKIndices.end());

    std::sort(tiesSelectedIndices.begin(), tiesSelectedIndices.end(), 
        [&points](int t0, int t1) {
            const auto& p0 = points[t0];
            const auto& p1 = points[t1];

            return std::lexicographical_compare(p0.data(), p0.data() + dimension, p1.data(), p1.data() + dimension);
        });

    double maxGap = 0.0;
    Eigen::Matrix<double, dimension, 1> maxGapVector;
    Eigen::Matrix<double, dimension, 1> trialVector;

    {
        double gap = Detail::findStableWeightVector<dimension>(points, topKIndices, tiesSelectedIndices[0], lbs, ubs, trialVector);
        if (gap > maxGap) {
            maxGap = gap;
            maxGapVector = trialVector;
        }
    }

    Eigen::VectorXd prePt = points[tiesSelectedIndices[0]];
    for (int i = 1; i < vacant; i++) {
        int tiesSelectedIdx = tiesSelectedIndices[i];
        if (points[tiesSelectedIdx] == prePt) continue;

        prePt = points[tiesSelectedIdx];

        double gap = Detail::findStableWeightVector<dimension>(points, topKIndices, tiesSelectedIdx, lbs, ubs, trialVector);
        if (gap > maxGap) {
            maxGap = gap;
            maxGapVector = trialVector;
        }
    }

    if (maxGap > 0.0) {
        weights = Eigen::VectorXd::Map(maxGapVector.data(), dimension);
    }
}

template <>
double findStableWeightVector<2>(const std::vector<Eigen::VectorXd> &points, const std::vector<int>& topKIndices, int kthIdx,
    const std::array<double, 2>& lbs, const std::array<double, 2>& ubs,
    Eigen::Vector2d& results);

template <int dimension> requires (dimension >= 2)
double findStableWeightVector(const std::vector<Eigen::VectorXd> &points, const std::vector<int>& topKIndices, int kthIdx,
    const std::array<double, dimension>& lbs, const std::array<double, dimension>& ubs,
    Eigen::Matrix<double, dimension, 1>& results) {
    constexpr int projDimension = dimension - 1;
    using ProjPlane = FairTopK::Plane<projDimension>;
    using ProjPlaneNormalVector = ProjPlane::NormalVector;

    int count = points.size();
    int k = topKIndices.size();

    std::vector<std::pair<ProjPlane, bool> > halfSpaces;
    halfSpaces.reserve(count - 1);

    Eigen::VectorXd kthPoint = points[kthIdx];

    int topKEleIdx = 0;
    for (int i = 0; i < count; i++) {
        bool isTopK = false;
        if (topKEleIdx < k && topKIndices[topKEleIdx] == i) {
            isTopK = true;
            topKEleIdx += 1;
        }

        Eigen::VectorXd diff = points[i] - kthPoint;
        if ((diff.array() > 0.0).all() || (diff.array() < 0.0).all() || (diff.array() == 0.0).all()) continue;

        ProjPlane projectedPlane;
        projectedPlane.normal = ProjPlaneNormalVector::Map(diff.data());
        projectedPlane.normal -= diff(projDimension) * ProjPlaneNormalVector::Ones();
        projectedPlane.constant = -diff(projDimension);

        double norm = projectedPlane.normal.template lpNorm<Eigen::Infinity>();
        projectedPlane.normal /= norm;
        projectedPlane.constant /= norm;
        
        halfSpaces.emplace_back(std::move(projectedPlane), isTopK);
    }

    using LPConstrsMat = Eigen::Matrix<double, projDimension + 1, -1>;
    using LPVector = Eigen::Matrix<double, projDimension + 1, 1>;
    using ColVector = Eigen::Matrix<double, projDimension, 1>;

    int halfSpaceCount = halfSpaces.size();
    constexpr int addConstrsCount = 2 * (projDimension + 1);

    LPConstrsMat mat = LPConstrsMat::Zero(projDimension + 1, halfSpaceCount + addConstrsCount);
    Eigen::VectorXd rhs(halfSpaceCount + addConstrsCount);

    LPVector objCoeffs = LPVector::Zero();
    objCoeffs(projDimension) = -1.0;

    for (int i = 0; i < projDimension; i++) {
        double lb = lbs[i];
        double ub = ubs[i];
            
        mat(i, 2 * i) = 1.0;
        rhs(2 * i) = ub;
        mat(i, 2 * i + 1) = -1.0;
        rhs(2 * i + 1) = -lb;
    }

    {
        int lastTwoOffset = 2 * projDimension;

        double lb = lbs[projDimension];
        double ub = ubs[projDimension];

        mat.col(lastTwoOffset).template head<projDimension>() = -ColVector::Ones();
        rhs(lastTwoOffset) = ub - 1.0;
        mat.col(lastTwoOffset + 1).template head<projDimension>() = ColVector::Ones();
        rhs(lastTwoOffset + 1) = 1.0 - lb;
    }

    int offset = addConstrsCount;
    for (int i = 0; i < halfSpaceCount; i++) {
        const auto& [halfSpacePlane, isTopK] = halfSpaces[i];
        int colIdx = offset + i;
        if (isTopK) {
            mat.col(colIdx).template head<projDimension>() = -ColVector::Map(halfSpacePlane.normal.data());
            rhs(colIdx) = -halfSpacePlane.constant;
        }
        else {
            mat.col(colIdx).template head<projDimension>() = ColVector::Map(halfSpacePlane.normal.data());
            rhs(colIdx) = halfSpacePlane.constant;
        }
        mat(projDimension, colIdx) = 1.0;
    }

    double negGap = sdlp::linprog<projDimension + 1>(objCoeffs, mat, rhs, results);

    if (negGap < 0.0) {
        results(projDimension) = 1.0 - results.template head<projDimension>().sum();
    }

    return -negGap;
}

}

}

#endif