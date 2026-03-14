/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#include <iostream>
#include <algorithm>
#include <thread>
#include <type_traits>
#include <array>
#include <cstddef>
#include <boost/functional/hash.hpp>
#include <boost/mp11/algorithm.hpp>
#include <Eigen/Dense>
#include <utility>

#include <cds/gc/nogc.h>
#include <cds/container/michael_list_nogc.h>
#include <cds/container/split_list_set_nogc.h>
#include <xenium/kirsch_kfifo_queue.hpp>
#include <xenium/reclamation/generic_epoch_based.hpp>
#include <sdlp/sdlp.hpp>
#include <soplex.h>

#include "data_loader.h"
#include "memory.h"
#include "utility.h"
#include "weights_difference.h"
#include "stabilization.h"
#include "experiments.h"

struct KSetComparator {
    int operator ()(const int *left, const int *right) const {
        for (int i = 0; i < num; i++) {
            if (left[i] < right[i]) return -1;
            else if (left[i] > right[i]) return 1;
        }
        return 0;
    }
    static void Configurate(int k) noexcept { num = k; }
private:
    static int num;
};
int KSetComparator::num = 0;

struct KSetHash {
    std::size_t operator ()(const int* const kset) const {
        return boost::hash_range(kset, kset + num);
    }
    static void Configurate(int k) noexcept { num = k; }
private:
    static int num;
};
int KSetHash::num = 0;

void insertionSort(int *indices, int k, int substIdx) {
    int val = indices[substIdx];
    for (int i = substIdx; i < k - 1; i++) {
        indices[i] = indices[i + 1];
    }

    int idx = k - 1;
    while (idx > 0 && indices[idx - 1] > val) {
        indices[idx] = indices[idx - 1];
        idx -= 1;
    }
    indices[idx] = val;
}

template<int dimension>
std::pair<Eigen::Matrix<double, dimension, -1>, Eigen::VectorXd> initLPConstraints(
    const std::vector<Eigen::VectorXd> &points, const Eigen::VectorXd& weights, double margin) {
    int count = points.size();
    constexpr int addConstrsCount = 2 * dimension;

    using LPConstrsMat = Eigen::Matrix<double, dimension, -1>;
    using ColVec = Eigen::Matrix<double, dimension - 1, 1>;

    LPConstrsMat mat = LPConstrsMat::Zero(dimension, count + addConstrsCount);
    Eigen::VectorXd rhs(count + addConstrsCount);

    for (int i = 0; i < count; i++) {
        const auto &pt = points[i];
        mat.col(i).template head<dimension - 1>() = 
            ColVec::Map(pt.data()) - (pt(dimension - 1) * ColVec::Ones());
        mat(dimension - 1, i) = -1.0;
        rhs(i) = -pt(dimension - 1);
    }

    int offset = count;

    for (int i = 0; i < dimension - 1; i++) {
        double lb = std::max(0.0, weights(i) - margin);
        double ub = std::min(1.0, weights(i) + margin);
            
        mat(i, offset + 2 * i) = 1.0;
        rhs(offset + 2 * i) = ub;
        mat(i, offset + 2 * i + 1) = -1.0;
        rhs(offset + 2 * i + 1) = -lb;
    }

    {
        int lastTwoOffset =  offset + 2 * (dimension - 1);

        double lb = std::max(0.0, weights(dimension - 1) - margin);
        double ub = std::min(1.0, weights(dimension - 1) + margin);

        mat.col(lastTwoOffset).template head<dimension - 1>() = -ColVec::Ones();
        rhs(lastTwoOffset) = ub - 1.0;
        mat.col(lastTwoOffset + 1).template head<dimension - 1>() = ColVec::Ones();
        rhs(lastTwoOffset + 1) = 1.0 - lb;
    }

    return { std::move(mat), std::move(rhs) };
}

template <int dimension>
void setupWeightsDiffLPSolver(int k, const int *kSet,
    const FairTopK::WeightsDiffLPAuxVector<dimension>& weightsDiffLPAuxVector,
    const Eigen::Matrix<double, dimension, -1>& mat, const Eigen::VectorXd& rhs, soplex::SoPlex& soplexSolver) {

    soplexSolver.setIntParam(soplex::SoPlex::OBJSENSE, soplex::SoPlex::OBJSENSE_MINIMIZE);
    soplexSolver.setRealParam(soplex::SoPlex::RealParam::FEASTOL, FairTopK::soplexEpsilon);
    soplexSolver.setRealParam(soplex::SoPlex::RealParam::FPFEASTOL, FairTopK::soplexEpsilon);

    soplexSolver.spxout.setVerbosity(soplex::SPxOut::ERROR);

    soplex::DSVector dummyCol(0);

    FairTopK::setUpWeightsDiffLPWeightVars<dimension>(weightsDiffLPAuxVector, dummyCol, soplexSolver);

    soplexSolver.addColReal(soplex::LPCol(0.0, dummyCol, soplex::infinity, 0.0));
    for (int i = 0; i < dimension; i++) {
        soplexSolver.addColReal(soplex::LPCol(1.0, dummyCol, soplex::infinity, 0.0));
    }

    constexpr std::array<int, dimension> indexArray = FairTopK::getIndexSequence(std::make_integer_sequence<int, dimension>{});

    soplex::DSVector row(dimension + dimension);

    FairTopK::setUpWeightsDiffLPConstrs<dimension + dimension, dimension>(weightsDiffLPAuxVector, row, soplexSolver);

    int count = rhs.size();
    int kSetEleIdx = 0;
    for (int i = 0; i < count; i++) {
        const auto& colVec = mat.col(i);
        row.add(dimension, indexArray.data(), colVec.data());

        if (kSetEleIdx < k && kSet[kSetEleIdx] == i) {
            soplexSolver.addRowReal(soplex::LPRow(row, soplex::LPRow::GREATER_EQUAL, rhs(i)));
            kSetEleIdx += 1;
        }
        else {
            soplexSolver.addRowReal(soplex::LPRow(row, soplex::LPRow::LESS_EQUAL, rhs(i)));
        }

        row.clear();
    }
}

void getFirstKSet(int k, const std::vector<double>& scores, int *firstKSet) {
    int count = scores.size();
    std::vector<std::pair<double, int> > scorePairs(count);

    for (int i = 0; i < count; i++) {
        scorePairs[i] = { scores[i], i };
    }

    std::nth_element(scorePairs.begin(), scorePairs.begin() + (k - 1), scorePairs.end(),
        [](const auto& p0, const auto& p1) { return p0.first > p1.first; });

    for (int i = 0; i < k; i++) {
        firstKSet[i] = scorePairs[i].second;
    }
}

template <int projDimension>
inline bool fastPruning(int prevEle, int newEle,
    const std::vector<Eigen::VectorXd>& points, const std::vector<Eigen::Matrix<double, projDimension, 1> >& extremePoints,
    double epsilon) {
    using PlaneNormalVector = FairTopK::Plane<projDimension>::NormalVector;
    
    Eigen::VectorXd diff = points[prevEle] - points[newEle];
    
    if ((diff.array() == 0.0).all()) return false;

    FairTopK::Plane<projDimension> plane;
    plane.normal = PlaneNormalVector::Map(diff.data());
    plane.normal -= diff(projDimension) * PlaneNormalVector::Ones();
    plane.constant = -diff(projDimension);

    return FairTopK::testIntersection(plane, extremePoints, epsilon) == false;
}

template <int dimension>
inline bool solveLP(const int *potKSet, int k, 
    const Eigen::Matrix<double, dimension, 1>& objCoeffs, 
    Eigen::Matrix<double, dimension, -1>& mat,
    Eigen::VectorXd& rhs, 
    Eigen::Matrix<double, dimension, 1>& vars) {
    for (int i = 0; i < k; i++) {
        mat.col(potKSet[i]) *= -1.0;
        rhs(potKSet[i]) = -rhs(potKSet[i]);
    }

    double val = sdlp::linprog<dimension>(objCoeffs, mat, rhs, vars);

    for (int i = 0; i < k; i++) {
        mat.col(potKSet[i]) *= -1.0;
        rhs(potKSet[i]) = -rhs(potKSet[i]);
    }

    return val != INFINITY;
}

template <int dimension>
void findVectorWithMinimumWeightsDifference(int k, const int *kSet,
    const FairTopK::WeightsDiffLPAuxVector<dimension>& weightDiffLPAuxVector,
    const Eigen::Matrix<double, dimension, -1>& mat, const Eigen::VectorXd& rhs,
    double& optScore, Eigen::Matrix<double, dimension - 1, 1>& weights) {
    
    soplex::SoPlex soplexSolver;
    setupWeightsDiffLPSolver<dimension>(k, kSet, weightDiffLPAuxVector, mat, rhs, soplexSolver);

    soplex::SPxSolver::Status stat = soplexSolver.solve();
    if (stat == soplex::SPxSolver::OPTIMAL) {
        double negWeightsDifference = -soplexSolver.objValueReal();

        if (negWeightsDifference > optScore) {
            optScore = negWeightsDifference;

            soplex::DVector vars(dimension + dimension);
            soplexSolver.getPrimal(vars);

            weights = Eigen::Matrix<double, dimension - 1, 1>::Map(vars.get_const_ptr());
        }
    }
}

template <std::size_t numGroups>
inline void setKSetPGroupsCounts(FairTopK::Groups preGroups, FairTopK::Groups newGroups, 
    FairTopK::GroupsMask pGroups, const boost::container::small_vector<int, numGroups>& pGroupsVec,
    int numPGroups, int *pGroupsCounts) {
    if ((preGroups & pGroups) != 0) {
        for (int i = 0; i < numPGroups; i++) {
            auto mask = FairTopK::getGroupsMask(pGroupsVec[i]);
            if ((preGroups & mask) != 0)
                pGroupsCounts[i] -= 1;
        }
    }

    if ((newGroups & pGroups) != 0) {
        for (int i = 0; i < numPGroups; i++) {
            auto mask = FairTopK::getGroupsMask(pGroupsVec[i]);
            if ((newGroups & mask) != 0)
                pGroupsCounts[i] += 1;
        }
    }
}

inline bool isFair(int numPGroups, const int *pGroupsCounts, const std::vector<std::pair<int , int> >& pGroupsBounds) {
    for (int i = 0; i < numPGroups; i++) {
         auto [lowerBound, upperBound] = pGroupsBounds[i];
         int pGroupCount = pGroupsCounts[i];
         if (pGroupCount < lowerBound || pGroupCount > upperBound)
            return false;
    }

    return true;
}

struct PotKSet {
    const int *kSet = nullptr;
    std::int32_t substIdx = -1;
    std::int32_t newEle = -1;
};

using KSetArena = FairTopK::MemoryArena<int, FairTopK::CacheLineAlign, FairTopK::CacheLineAlign>;
using PotKSetPool = FairTopK::MemoryPool<PotKSet>;

using reclaimer = xenium::policy::reclaimer<xenium::reclamation::debra<> >;
using padding_bytes = xenium::policy::padding_bytes<0>;
using KSetQueue = xenium::kirsch_kfifo_queue<PotKSet*, padding_bytes, reclaimer>;

template <class TermChecker = std::nullptr_t>
void enumeratePotKSet(int count, int k, const std::vector<Eigen::VectorXd> &points, const int* kSet, 
    KSetQueue& kSetQueue, PotKSetPool *pool, TermChecker&& termChecker = nullptr) {
    int kSetEleIdx = 0;
    for (int i = 0; i < count; i++) {
        if (kSetEleIdx < k && kSet[kSetEleIdx] == i) {
            kSetEleIdx += 1;
        }
        else {
            const auto& newPt = points[i].array();
            for (int j = 0; j < k; j++) {
                if constexpr (!std::is_same<TermChecker, std::nullptr_t>()) {
                    if (std::forward<TermChecker>(termChecker)()) return;
                }

                const auto& prePt = points[kSet[j]].array();
                if ((newPt >= prePt).any())
                     kSetQueue.push(pool->Alloc(kSet, j, i));
            }
        }
    }
}

template <std::size_t numGroups>
void processFirstKSet(const std::vector<Eigen::VectorXd> &points, const std::vector<FairTopK::Groups>& groups,
    int k, const std::vector<double> &scores, FairTopK::GroupsMask pGroups, int numPGroups,
    const boost::container::small_vector<int, numGroups>& pGroupsVec, int* firstKSet) {
    getFirstKSet(k, scores, firstKSet);
    std::sort(firstKSet, firstKSet + k);

    std::array<int, FairTopK::maxNumGroups> pGroupsCounts{};
    for (int i = 0; i < k; i++) {
        FairTopK::Groups candidateGroups = groups[firstKSet[i]];

        if ((candidateGroups & pGroups) == 0) continue;

        FairTopK::updatePGroupsCounts<1>(numPGroups, candidateGroups, pGroupsVec, pGroupsCounts);
    }
    std::copy(pGroupsCounts.data(), pGroupsCounts.data() + numPGroups, firstKSet + k);
}

template <int dimension, FairTopK::Optimization opt>
bool solve(int totalThreadCount,
    const std::vector<Eigen::VectorXd> &points, const std::vector<FairTopK::Groups>& groups,
    int k, FairTopK::GroupsMask pGroups, const std::vector<std::pair<int , int> >& pGroupsBounds, double margin,
    Eigen::VectorXd& weights) {
    constexpr double epsilon = 1e-8;

    KSetComparator::Configurate(k);
    KSetHash::Configurate(k);

    struct split_list_traits : public cds::container::split_list::traits {
        typedef KSetHash hash;
        struct ordered_list_traits : public cds::container::michael_list::traits {
            typedef KSetComparator compare;
        };
    };

    using KSetsHashSet = cds::container::SplitListSet<cds::gc::nogc, int *, split_list_traits>;

    KSetsHashSet kSets(std::max(points.size(), (std::size_t)16384));
    KSetQueue kSetQueue(std::clamp(totalThreadCount * 4, 32, 512));

    //Keeps everything relevant in memory until completion of all spawned threads
    KSetArena **arenas = new KSetArena* [totalThreadCount];
    PotKSetPool **pools = new PotKSetPool* [totalThreadCount];

    std::atomic<int> workingThreadCount;
    std::atomic_flag found = ATOMIC_FLAG_INIT;

    std::vector<double> optScores;
    if constexpr (opt != FairTopK::Optimization::None) {
        optScores = std::vector<double>(totalThreadCount, std::numeric_limits<double>::lowest());
    }

    double *optWeightVecsPtr = nullptr;
    constexpr std::size_t optWeightAlign = FairTopK::CacheLineAlign / 2;
    constexpr std::size_t optWeightAlignComplement = optWeightAlign - 1;
    constexpr std::size_t optWeightAlignedSize =
        (((dimension - 1) * sizeof(double) + optWeightAlignComplement) & (~optWeightAlignComplement)) / sizeof(double);
    if constexpr (opt != FairTopK::Optimization::None) {
        optWeightVecsPtr = FairTopK::allocAligned<double, optWeightAlign>(optWeightAlignedSize * totalThreadCount);
    }

    workingThreadCount.store(totalThreadCount, std::memory_order_relaxed);

    auto refWeights = weights;

    auto extremePoints = FairTopK::computeExtremePoints<dimension>(refWeights, margin, epsilon);

    std::vector<double> scores(points.size(), 0.0);
    std::transform(points.cbegin(), points.cend(), scores.begin(), 
        [&weights](const auto& point) -> double { return point.dot(weights); } );

    auto func = [&points, margin, &extremePoints, k, &kSetQueue, &kSets, &workingThreadCount,
                 &groups, pGroups, &pGroupsBounds, &scores, &optScores, optWeightVecsPtr,
                 arenas, pools, &refWeights, &weights, &found]<bool init>(int threadIdx, std::bool_constant<init>) {
        bool decremented = false;
        int count = points.size();
        
        bool localFair = false;
        double localOptScore = std::numeric_limits<double>::lowest();
        Eigen::Matrix<double, dimension - 1, 1> localOptWeightVec;

        auto pGroupsVec = FairTopK::getPGroupsVec(pGroups);
        int numPGroups = pGroupsBounds.size();

        int nodeSize = k + numPGroups;
        constexpr std::size_t nodeAlignComplement = FairTopK::CacheLineAlign - 1;
        std::size_t nodeAlignedSize = ((nodeSize * sizeof(int) + nodeAlignComplement) & (~nodeAlignComplement)) / sizeof(int);
        KSetArena *arena = new KSetArena(128 * nodeAlignedSize);

        PotKSetPool *pool = new PotKSetPool(std::min(count * k, 4194304));
        arenas[threadIdx] = arena;
        pools[threadIdx] = pool;

        if constexpr (init) {
            int *firstKSet = arena->Alloc(nodeSize, std::false_type{});

            processFirstKSet(points, groups, k, scores, pGroups, numPGroups, pGroupsVec, firstKSet);
            
            kSets.insert(firstKSet);

            enumeratePotKSet(count, k, points, firstKSet, kSetQueue, pool);

            if constexpr (opt == FairTopK::Optimization::None) {
                if (found.test(std::memory_order_relaxed)) return;
            }
        }

        auto [mat, rhs] = initLPConstraints<dimension>(points, refWeights, margin);

        FairTopK::WeightsDiffLPAuxVector<dimension> weightsDiffLPAuxVector;
        if constexpr (opt == FairTopK::Optimization::WeightsDifference) {
            weightsDiffLPAuxVector = FairTopK::setupWeightsDiffLPAuxVector<dimension>(refWeights, margin);
        }

        using LPVector = Eigen::Matrix<double, dimension, 1>;

        LPVector vars = LPVector::Zero();
        LPVector objCoeffs = LPVector::Zero();
        objCoeffs(dimension - 1) = -1.0;

        int *potKSetEles = nullptr;

        //Increment the workingThreadCount between two linearization points and right before the linearization point of pop()
        auto foundBeforeFunc = [&decremented, &workingThreadCount]() noexcept {
            if (decremented) {
                workingThreadCount.fetch_add(1, std::memory_order_relaxed);
                decremented = false;
            }
        };

        auto foundChecker = [&found]() noexcept { return found.test(std::memory_order_relaxed); };

        while (true) {
            if constexpr (opt == FairTopK::Optimization::None) {
                if (found.test(std::memory_order_relaxed)) break;
            }

            auto potKSetPtOpt = kSetQueue.pop(foundBeforeFunc);
            if (potKSetPtOpt) {
                PotKSet potKSet = **potKSetPtOpt;
                pool->Dealloc(*potKSetPtOpt);
                const int *kSet = potKSet.kSet;

                if (potKSetEles == nullptr) potKSetEles = arena->Alloc(nodeSize, std::false_type{});
                
                std::copy(kSet, kSet + nodeSize, potKSetEles);
                int prevEle = potKSetEles[potKSet.substIdx];
                int newEle = potKSet.newEle;

                potKSetEles[potKSet.substIdx] = newEle;
                insertionSort(potKSetEles, k, potKSet.substIdx);
                    
                if (kSets.contains(potKSetEles) != kSets.end()) continue;

                if (fastPruning(prevEle, newEle, points, extremePoints, epsilon)) continue;

                bool valid = solveLP<dimension>(potKSetEles, k, objCoeffs, mat, rhs, vars);

                if (!valid) continue;
                
                setKSetPGroupsCounts(groups[kSet[potKSet.substIdx]], groups[potKSet.newEle], pGroups, pGroupsVec,
                    numPGroups, potKSetEles + k);
                    
                if (isFair(numPGroups, potKSetEles + k, pGroupsBounds)) {
                    if constexpr (opt == FairTopK::Optimization::Utility || opt == FairTopK::Optimization::StableUtility) {
                        localFair = true;

                        double utility = 0.0;
                        for (int i = 0; i < k; i++) 
                            utility += scores[potKSetEles[i]];

                        if (utility > localOptScore) {
                            localOptScore = utility;
                            localOptWeightVec = vars.template head<dimension - 1>();
                        }
                    }
                    else if constexpr (opt == FairTopK::Optimization::WeightsDifference) {
                        localFair = true;

                        findVectorWithMinimumWeightsDifference(k, potKSetEles, weightsDiffLPAuxVector, mat, rhs, 
                            localOptScore, localOptWeightVec);
                    }
                    else {
                        if (!found.test_and_set(std::memory_order_acquire)) {
                            vars(dimension - 1) = 1.0 - vars.template head<dimension - 1>().sum();
                            weights = Eigen::VectorXd::Map(vars.data(), dimension);
                        }

                        return;
                    }
                }

                if constexpr (opt == FairTopK::Optimization::None) {
                    if (found.test(std::memory_order_relaxed)) break;
                }
                    
                auto iter = kSets.insert(potKSetEles);

                if (iter == kSets.end()) continue;
                
                if constexpr (opt == FairTopK::Optimization::None) {
                    enumeratePotKSet(count, k, points, potKSetEles, kSetQueue, pool, foundChecker);
                }
                else {
                    enumeratePotKSet(count, k, points, potKSetEles, kSetQueue, pool);
                }

                potKSetEles = nullptr;
            }
            else {
                if (!decremented) {
                    workingThreadCount.fetch_sub(1, std::memory_order_relaxed);
                    decremented = true;
                }
                if (workingThreadCount.load(std::memory_order_relaxed) <= 0)
                    break;
            }
        }

        if constexpr (opt != FairTopK::Optimization::None) {
            if (localFair) {
                found.test_and_set(std::memory_order_relaxed);
                optScores[threadIdx] = localOptScore;

                double *optWeightVec = optWeightVecsPtr + threadIdx * optWeightAlignedSize;
                std::copy(localOptWeightVec.data(), localOptWeightVec.data() + dimension - 1, optWeightVec);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(totalThreadCount);
    threads.push_back(std::thread(func, 0, std::true_type{}));
    for (int i = 1; i < totalThreadCount; i++)
        threads.push_back(std::thread(func, i, std::false_type{}));

    for (auto& thread : threads) thread.join();

    for (int i = 0; i < totalThreadCount; i++) {
        delete arenas[i];
        delete pools[i];
    }
    delete[] arenas;
    delete[] pools;

    bool fair = found.test(std::memory_order_relaxed);

    if constexpr (opt != FairTopK::Optimization::None) {
        if (fair) {
            double optScore = std::numeric_limits<double>::lowest();
            int optIdx = -1;
            for (int i = 0; i < totalThreadCount; i++) {
                double localOptScore = optScores[i];
                if (localOptScore > optScore) {
                    optScore = localOptScore;
                    optIdx = i;
                }
            }

            const double *optWeightVec = optWeightVecsPtr + optIdx * optWeightAlignedSize;
            weights.head(dimension - 1) = Eigen::VectorXd::Map(optWeightVec, dimension - 1);
            weights(dimension - 1) = 1.0 - weights.head(dimension - 1).sum();
        }
    }

    if (optWeightVecsPtr) FairTopK::freeAligned<optWeightAlign>(optWeightVecsPtr);

    if constexpr (opt == FairTopK::Optimization::StableUtility) {
        if (fair) {
            FairTopK::stabilizeFairWeightVector<dimension>(points, groups, k, pGroups, pGroupsBounds, 
                refWeights, margin, epsilon, weights);
        }
    }
    
    return fair;
}

int main(int argc, char* argv[]) {
    std::vector<Eigen::VectorXd> points;
    std::vector<FairTopK::Groups> groups;
    FairTopK::GroupsMask protectedGroups = 0;

    auto [fileName, params] = FairTopK::parseCommandLine(argc, argv);

    bool success = FairTopK::DataLoader::readPreprocessedDataset(fileName, points, groups, protectedGroups);
    if (!success) return -1;

    int dimension = points[0].rows();
    constexpr int minDimension = 3;
    constexpr int maxDimension = 6;

    if (dimension < minDimension || dimension > maxDimension) {
        std::cerr << "Error: Only support datasets with 3 <= dimensions <= 6" << std::endl;
        return -1;
    }
    
    int threadCount = params.threadCount > 0 ? params.threadCount : std::thread::hardware_concurrency();

    constexpr int dimCount = maxDimension - minDimension + 1;
    int dimDiff = dimension - minDimension;

    auto solveFunc = boost::mp11::mp_with_index<(std::size_t)FairTopK::Optimization::NumOptions>((std::size_t)params.opt,
        [dimDiff](auto opt) {
            return boost::mp11::mp_with_index<dimCount>(dimDiff,
                [opt](auto dimDiff) { return solve<dimDiff() + minDimension, FairTopK::Optimization(opt())>; });
        });

    FairTopK::fairTopkExperiments(points, groups, protectedGroups, params, 
        [threadCount, solveFunc]<class... Args>(Args&&... params) { 
            return solveFunc(threadCount, std::forward<Args>(params)...);
    });

    return 0;
}