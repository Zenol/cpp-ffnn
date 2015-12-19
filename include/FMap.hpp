#ifndef FMAP_HPP_
#define FMAP_HPP_

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>

/**
 * This file implement a fmap like operator, namely %,
 * in order to allow aplying a T(T) function on a sparse
 * vector or sparse matrix.
 *
 * It's not a true fmap since it can't apply a
 * U(T) function.
 */

namespace ffnn
{
    using namespace boost::numeric::ublas;

    template<typename T, typename U>
    mapped_vector<T> &operator% (U f, mapped_vector<T> &&v)
    {
        for (int i = 0; i < v.size(); i++)
            v(i) = f(v(i));
        return v;
    }
    template<typename T, typename U>
    mapped_vector<T> operator% (U f, const mapped_vector<T> &v)
    {
        auto w(v);
        for (int i = 0; i < v.size(); i++)
            w(i) = f(w(i));
        return w;
    }
    template<typename T, typename U>
    mapped_matrix<T> &operator% (U f, mapped_matrix<T> &&m)
    {
        for (int i = 0; i < m.size1(); i++)
            for (int j = 0; j < m.size2(); j++)
                m(i, j) = f(m(i, j));
        return m;
    }
    template<typename T, typename U>
    mapped_matrix<T> operator% (U f, const mapped_matrix<T> &m)
    {
        auto n(m);
        for (int i = 0; i < n.size1(); i++)
            for (int j = 0; j < n.size2(); j++)
                n(i, j) = f(n(i, j));
        return n;
    }
}

#endif /* !FMAP_HPP_ */
