/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "stabilization.h"
#include <limits>
#include <algorithm>
#include <boost/mp11/algorithm.hpp>

namespace FairTopK {

namespace Detail {

void stabilizeFairWeightVectorDynDim(const std::vector<Eigen::VectorXd> &points, const std::vector<Groups>& groups, 
    int k, GroupsMask pGroups, const std::vector<std::pair<int, int> >& pGroupsBounds, 
    const Eigen::VectorXd& refWeights, double weightSpaceMargin, double epsilon, Eigen::VectorXd& weights) {
    
    constexpr int minDimension = 2;
    constexpr int maxDimension = 10;

    int dimension = refWeights.size();

    if (dimension < minDimension || dimension > maxDimension) {
        std::cerr << "Error: Data dimensionality is unsupported for stabilizing fair weight vectors." << std::endl;
        return;
    }

    constexpr int dimCount = maxDimension - minDimension + 1;
    int dimDiff = dimension - minDimension;

    boost::mp11::mp_with_index<dimCount>(dimDiff,
        [&points, &groups, k, pGroups, &pGroupsBounds, &refWeights, weightSpaceMargin, epsilon, &weights](auto dimDiff) { 
            stabilizeFairWeightVectorStcDim<dimDiff() + minDimension>(points, groups, k, pGroups, pGroupsBounds, 
                refWeights, weightSpaceMargin, epsilon, weights); 
        });
}

template <>
double findStableWeightVector<2>(const std::vector<Eigen::VectorXd> &points, const std::vector<int>& topKIndices, int kthIdx,
    const std::array<double, 2>& lbs, const std::array<double, 2>& ubs,
    Eigen::Vector2d& results) {
    int count = points.size();
    int k = topKIndices.size();

    Eigen::VectorXd kthPoint = points[kthIdx];

    double lb = std::max(lbs[0], 1.0 - ubs[1]);
    double ub = std::min(ubs[0], 1.0 - lbs[1]);

    double leftEndPt = std::numeric_limits<double>::lowest();
    double rightEndPt = std::numeric_limits<double>::max();

    int topKEleIdx = 0;
    for (int i = 0; i < count; i++) {
        bool isTopK = false;
        if (topKEleIdx < k && topKIndices[topKEleIdx] == i) {
            isTopK = true;
            topKEleIdx += 1;
        }

        Eigen::VectorXd diff = points[i] - kthPoint;
        if ((diff.array() > 0.0).all() || (diff.array() < 0.0).all() || (diff.array() == 0.0).all()) continue;

        double factor = diff(0) - diff(1);
        if (factor == 0.0) continue;

        double crossPt = -diff(1) / factor;

        if (factor > 0.0) {
            if (isTopK) leftEndPt = std::max(leftEndPt, crossPt);
            else rightEndPt = std::min(rightEndPt, crossPt);
        }
        else {
            if (isTopK) rightEndPt = std::min(rightEndPt, crossPt);
            else leftEndPt = std::max(leftEndPt, crossPt);
        }

        if (rightEndPt <= leftEndPt) break;
    }

    double gap = 0.0;
    if (rightEndPt > leftEndPt) {
        double midPt = std::clamp((rightEndPt + leftEndPt) / 2.0, lb, ub);
        
        gap = std::min(midPt - leftEndPt, rightEndPt - midPt);
        if (gap > 0.0) {
            results(0) = midPt;
            results(1) = 1.0 - midPt;
        }
    }

    return gap;
}

}

}