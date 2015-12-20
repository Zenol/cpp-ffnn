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

    template<typename T>
    class Network;

    template<typename T>
    class Layer
    {
    public:
        /**
         * \param input_size The length of the input vector
         * \param output_size The number of neurons inside the layer
         */
        Layer(unsigned int input_size, unsigned int output_size,
              T(*threshold)(T), T(*derivative)(T))
            :weights(output_size, input_size), biases(output_size),
             threshold_function(threshold), derivative_function(derivative)
        {};
        Layer(unsigned int input_size, unsigned int output_size,
              std::function<T(T)> threshold, std::function<T(T)> derivative)
            :weights(output_size, input_size), biases(output_size),
             threshold_function(threshold), derivative_function(derivative)
        {};

        unsigned int get_input_size() const {return weights.size2();};
        unsigned int get_output_size() const {return weights.size1();};

        vector<T> operator<< (const vector<T> &input) const
        {
            return threshold_function % vector<T>(biases + prod(weights, input));
        }

        //! Randomize weights and biases with values in [-1, 1].
        void randomize(void)
        {
            // Notice this function doesn't work for non-fractional type T.
            std::uniform_real_distribution<> dis(-1, 1);
            auto f = [this, &dis](T x) -> T {return dis(eng);};
            f %= weights;
            f %= biases;
        }

        //! Randomize weights and biases with values in
        //! [0, minstd_rand::max() - minstd_rand::min()].
        void randomize_int(void)
        {
            // Notice this function doesn't work for non-fractional type T.
            auto f = [this](T x) -> T {return T(eng() - eng.min());};
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
        std::minstd_rand eng;

    private:
        //! Weights of the neural network applied to the inputs.
        matrix<T> weights;
        //! Biases aplied befor computing the threshold function.
        vector<T> biases;
        //! The threshold function applied to the weighted sum of inputs.
        std::function<T(T)> threshold_function;
        //! The function used to compute the derivate.
        //! It takes a = threshold_function(z) as input.
        std::function<T(T)> derivative_function;

        friend Network<T>;
    };


    template<typename T>
    T sigmoid(const T x)
    {
        return static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
    };

    template<typename T>
    T sigmoid_prime(const T a)
    {
        return a * (static_cast<T>(1) - a);
    };

    template<typename T>
    vector<T> operator >> (const vector<T> &input, const Layer<T> layer)
    {
        return layer << input;
    }
};

#endif /* !LAYER */
