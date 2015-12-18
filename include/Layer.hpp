#include <functional>
#include <cmath>

#include <ftl/prelude.h>

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>

namespace ffnn
{

    using namespace boost::numeric::ublas;
    using namespace boost;
    using namespace std;

    template<typename T>
    class Layer
    {
        /**
         * \param input_size The length of the input vector
         * \param output_size The number of neurons inside the layer
         */
        Layer(unsigned int input_size, unsigned int output_size,
              function<T(T)> threshold)
            :
            weights(input_size, output_size), threshold_function(threshold)
        {};

        mapped_vector<T> operator >> (const mapped_vector<T> &input)
        {
            return threshold_function % (biases + weights * input);
        }

    private:
        //! Weights of the neural network applied to the inputs.
        mapped_matrix<T> weights;
        //! Biases aplied befor computing the threshold function.
        mapped_vector<T> biases;
        //! The threshold function applied to the weighted sum of inputs.
        function<T(T)> threshold_function;
    };

    template<typename T>
    T sigmoid(const T x)
    {
        return static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
    };

};
