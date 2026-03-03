/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#ifndef FAIR_TOPK_BSPTREE_H
#define FAIR_TOPK_BSPTREE_H

#include <stack>
#include <concepts>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <Eigen/Dense>
#include <sdlp/sdlp.hpp>

#include "utility.h"
#include "memory.h"

namespace FairTopK {

template <class Func, int dimension>
concept ValidFairnessChecker = requires(Func func, const std::vector<std::pair<Plane<dimension>, bool> >& halfSpaces) {
    { func(halfSpaces) } -> std::convertible_to<bool>;
};

template <class Func, int dimension>
concept LegitFairnessChecker = ValidFairnessChecker<Func, dimension> || std::same_as<Func, std::nullptr_t>;

template <int dimension>
class BSPTree {
public:
    BSPTree() = default;
    template <class Func = std::nullptr_t> requires LegitFairnessChecker<Func, dimension>
    bool insert(const Plane<dimension>& plane, Func&& fairnessChecker = nullptr);

    ~BSPTree() = default;
    BSPTree(const BSPTree&) = delete;
    BSPTree(BSPTree&&) = delete;
    BSPTree& operator=(const BSPTree&) = delete;
    BSPTree& operator=(BSPTree&&) = delete;
private:
    using BSPPlane = Plane<dimension>;
    struct Node {
        BSPPlane plane;
        Node *left = nullptr; //positive
        Node *right = nullptr; //negative
    };

public:
    class BSPTreeLeafConstIterator {
    public:
        using value_type = std::vector<std::pair<BSPPlane, bool> >;
        using difference_type = std::ptrdiff_t;
        using size_type = std::size_t;
        using reference = const std::vector<std::pair<BSPPlane, bool> >&;
        using pointer = const std::vector<std::pair<BSPPlane, bool> > *;

        BSPTreeLeafConstIterator(const Node *root);
        BSPTreeLeafConstIterator() = default;
        BSPTreeLeafConstIterator(const BSPTreeLeafConstIterator&) = default;
        BSPTreeLeafConstIterator(BSPTreeLeafConstIterator&&) = default;
        BSPTreeLeafConstIterator& operator=(const BSPTreeLeafConstIterator&) = default;
        BSPTreeLeafConstIterator& operator=(BSPTreeLeafConstIterator&&) = default;
        ~BSPTreeLeafConstIterator() = default;

        bool isEnd() const { return halfSpaces.empty(); }
        bool operator==(std::nullptr_t p) const { return isEnd(); }
        bool operator!=(std::nullptr_t p) const { return !isEnd(); }

        reference operator*() const { return halfSpaces; }
        pointer operator->() const { return &halfSpaces; }

        inline BSPTreeLeafConstIterator& operator++();

    private:
        void findNext();

        std::vector<std::pair<BSPPlane, bool> > halfSpaces;
        std::stack<const Node *> nodeStack;
	};

    using const_iterator = BSPTreeLeafConstIterator;
    const_iterator cbegin() const { return const_iterator(root); }

private:
    bool testIntersection(const BSPPlane& plane, const std::vector<std::pair<BSPPlane, bool> >& halfSpaces);

    MemoryArena<Node, CacheLineAlign, std::max(alignof(Node), (std::size_t)4)> nodePool;
    Node *root = nullptr;

    constexpr static double bspEpsilon = 1e-10;
};

template <int dimension>
template <class Func> requires LegitFairnessChecker<Func, dimension>
bool BSPTree<dimension>::insert(const Plane<dimension>& plane, Func&& fairnessChecker) {
    std::vector<std::pair<BSPPlane, bool> > halfSpaces;
    if (root == nullptr) {
        root = nodePool.Alloc();
        root->plane = plane;
        halfSpaces.emplace_back(root->plane, true);
        if constexpr (!std::is_same<Func, std::nullptr_t>()) {
            if (std::forward<Func>(fairnessChecker)(halfSpaces)) {
                return true;
            }
        }
        halfSpaces[0].second = false;
        if constexpr (!std::is_same<Func, std::nullptr_t>()) {
            if (std::forward<Func>(fairnessChecker)(halfSpaces)) {
                return true;
            }
        }
        return false;
    }

    constexpr std::uintptr_t mask = 0x3;
    std::stack<std::uintptr_t> nodeStack;

    nodeStack.push((std::uintptr_t)root);
    while (!nodeStack.empty()) {
        std::uintptr_t markedNode = nodeStack.top();
        Node *node = (Node *)(markedNode & (~mask));
        unsigned int visitCounter = markedNode & mask;

        if (visitCounter >= 2) {
            nodeStack.pop();
            halfSpaces.pop_back();
            continue;
        }
        nodeStack.top() = ((std::uintptr_t)node | (visitCounter + 1));

        bool isPositive = (visitCounter == 0);
        if (isPositive) {
            halfSpaces.emplace_back(node->plane, true);
        }
        else {
            halfSpaces.back().second = false;
        }

        if (testIntersection(plane, halfSpaces)) {
            Node *nextNode = isPositive ? node->left : node->right;
            if (nextNode) {
                nodeStack.push((std::uintptr_t)nextNode);
            }
            else {
                if constexpr (!std::is_same<Func, std::nullptr_t>()) {
                    if (std::forward<Func>(fairnessChecker)(halfSpaces)) {
                        return true;
                    }
                }

                Node *newNode = nodePool.Alloc();
                newNode->plane = plane;
                if (isPositive) node->left = newNode;
                else node->right = newNode;
            }
        }
    }

    return false;
}

template <int dimension>
bool BSPTree<dimension>::testIntersection(const BSPPlane& plane, const std::vector<std::pair<BSPPlane, bool> >& halfSpaces) {
    if (halfSpaces.empty()) return true;
    if (halfSpaces.size() == 1) {
        const auto& [halfSpacePlane, isPositive] = halfSpaces[0];
        double dotProd = plane.normal.normalized().dot(halfSpacePlane.normal.normalized());
        if (std::abs(1.0 - dotProd) <= bspEpsilon) {
            return isPositive ? plane.constant >= halfSpacePlane.constant :
                                plane.constant <= halfSpacePlane.constant;
        }
        return true;
    }
    
    using LPConstrsMat = Eigen::Matrix<double, dimension + 1, -1>;
    using LPVector = Eigen::Matrix<double, dimension + 1, 1>;
    using ColVector = Eigen::Matrix<double, dimension, 1>;

    int count = halfSpaces.size();
    constexpr int addConstrsCount = 2;

    LPConstrsMat mat = LPConstrsMat::Ones(dimension + 1, count + addConstrsCount);
    Eigen::VectorXd rhs(count + addConstrsCount);

    LPVector objCoeffs = LPVector::Zero();
    objCoeffs(dimension) = -1.0;
    LPVector results;

    for (int i = 0; i < count; i++) {
        const auto& [halfSpacePlane, isPositive] = halfSpaces[i];
        if (isPositive) {
            mat.col(i).template head<dimension>() = 
                -ColVector::Map(halfSpacePlane.normal.data());
            rhs(i) = -halfSpacePlane.constant;
        }
        else {
            mat.col(i).template head<dimension>() = 
                ColVector::Map(halfSpacePlane.normal.data());
            rhs(i) = halfSpacePlane.constant;
        }
    }

    mat.col(count).template head<dimension>() = -ColVector::Map(plane.normal.data());
    mat(dimension, count) = 0.0;
    rhs(count) = -plane.constant;
    mat.col(count + 1).template head<dimension>() = ColVector::Map(plane.normal.data());
    mat(dimension, count + 1) = 0.0;
    rhs(count + 1) = plane.constant;

    double val = sdlp::linprog<dimension + 1>(objCoeffs, mat, rhs, results);

    return val < 0.0;
}

template <int dimension>
BSPTree<dimension>::BSPTreeLeafConstIterator::BSPTreeLeafConstIterator(const Node *root) {
    if (root == nullptr) return;

    const Node *node = root;
    while (node) {
        halfSpaces.emplace_back(node->plane, true);
        nodeStack.push(node);
        node = node->left;
    }
}

template <int dimension>
inline BSPTree<dimension>::BSPTreeLeafConstIterator& BSPTree<dimension>::BSPTreeLeafConstIterator::operator++() {
    findNext();
    return *this;
}

template <int dimension>
void BSPTree<dimension>::BSPTreeLeafConstIterator::findNext() {
    while (!nodeStack.empty()) {
        const Node *node = nodeStack.top();
        auto& lastHalfSpace = halfSpaces.back();

        if (lastHalfSpace.second) {
            lastHalfSpace.second = false;
            node = node->right;
            while (node) {
                nodeStack.push(node);
                halfSpaces.emplace_back(node->plane, true);
                node = node->left;
            }
            break;
        }
        else {
            nodeStack.pop();
            halfSpaces.pop_back();
        }
    }
}

}

#endif