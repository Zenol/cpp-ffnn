#ifndef LAYER_HPP_
#define LAYER_HPP_

#include <functional>
#include <algorithm>
#include <random>
#include <cmath>

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>


#include "FMap.hpp"

namespace ffnn
{
    using namespace boost::numeric::ublas;
    using namespace boost;
    using namespace std;

    template<typename T>
    class Layer
    {
    public:
        /**
         * \param input_size The length of the input vector
         * \param output_size The number of neurons inside the layer
         */
        Layer(unsigned int input_size, unsigned int output_size, T(*threshold)(T))
            :weights(output_size, input_size), biases(output_size),
             threshold_function(threshold)
        {};
        Layer(unsigned int input_size, unsigned int output_size,
              std::function<T(T)> threshold)
            :weights(output_size, input_size), biases(output_size),
             threshold_function(threshold)
        {};

        mapped_vector<T> operator<< (const mapped_vector<T> &input) const
        {
            return threshold_function % mapped_vector<T>(biases + prod(weights, input));
        }

        //! Randomize weights and biases with values in [0, 1].
        void randomize(void)
        {
            // Notice this function doesn't work for non-fractional type T.
            auto f = [this](T x) -> T
                {
                    return T(engine() - engine.min()) / T(engine.max());
                };
            f %= weights;
            f %= biases;
        }

        //! Randomize weights and biases with values in
        //! [0, minstd_rand::max() - minstd_rand::min()].
        void randomize_int(void)
        {
            // Notice this function doesn't work for non-fractional type T.
            auto f = [this](T x) -> T {return T(engine() - engine.min());};
            f %= weights;
            f %= biases;
        }

        //! Display function
        friend
        std::ostream &operator<< (std::ostream &oss, const Layer<T> &l)
        {
            oss << "Layer<> :" << std::endl;
            oss << "  Weigths : " << l.weights << std::endl;
            oss << "  Biases : " << l.biases;
            return oss;
        }

        //! Random generator engine. Used by randomize() and
        //! randomize_int().
        std::minstd_rand engine;

    private:
        //! Weights of the neural network applied to the inputs.
        mapped_matrix<T> weights;
        //! Biases aplied befor computing the threshold function.
        mapped_vector<T> biases;
        //! The threshold function applied to the weighted sum of inputs.
        std::function<T(T)> threshold_function;
    };


    template<typename T>
    T sigmoid(const T x)
    {
        return static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
    };

    template<typename T>
    mapped_vector<T> operator >> (const mapped_vector<T> &input, const Layer<T> layer)
    {
        return layer << input;
    }

};

#endif /* !LAYER */
