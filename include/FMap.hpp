#ifndef FMAP_HPP_
#define FMAP_HPP_

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>

/**
 * This file implement a fmap like operator, namely %,
 * in order to allow aplying a T(T) function on a sparse
 * vector or sparse matrix. The function SHOULDN'T be
 * modifying the argument.
 *
 * It's not a true fmap since it can't apply a
 * U(T) function.
 *
 * The operator f %= A apply the function f to each
 * value of A, replacing them by the result.
 * It's returning a reference to A.
 * When f is without side efect, we have
 * f %= A ::: A = f % A, although no copy are made
 * when using %=.
 */

namespace ffnn
{
    using namespace boost::numeric::ublas;

    template<typename T, typename U>
    vector<T> &operator% (U f, vector<T> &&v)
    {
        for (int i = 0; i < v.size(); i++)
            v(i) = f(v(i));
        return v;
    }
    template<typename T, typename U>
    vector<T> operator% (U f, const vector<T> &v)
    {
        auto w(v);
        for (int i = 0; i < w.size(); i++)
            w(i) = f(w(i));
        return w;
    }
    template<typename T, typename U>
    vector<T> &operator%= (U f, vector<T> &v)
    {
        for (int i = 0; i < v.size(); i++)
            v(i) = f(v(i));
        return v;
    }
    template<typename T, typename U>
    matrix<T> &operator% (U f, matrix<T> &&m)
    {
        for (int i = 0; i < m.size1(); i++)
            for (int j = 0; j < m.size2(); j++)
                m(i, j) = f(m(i, j));
        return m;
    }
    template<typename T, typename U>
    matrix<T> operator% (U f, const matrix<T> &m)
    {
        auto n(m);
        for (int i = 0; i < n.size1(); i++)
            for (int j = 0; j < n.size2(); j++)
                n(i, j) = f(n(i, j));
        return n;
    }
    template<typename T, typename U>
    matrix<T> &operator%= (U f, matrix<T> &m)
    {
        for (int i = 0; i < m.size1(); i++)
            for (int j = 0; j < m.size2(); j++)
                m(i, j) = f(m(i, j));
        return m;
    }
}

#endif /* !FMAP_HPP_ */
