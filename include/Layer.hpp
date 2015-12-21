#ifndef LAYER_HPP_
#define LAYER_HPP_

#include <functional>
#include <algorithm>
#include <random>
#include <cmath>

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

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
        Layer()
        {};
        Layer(const boost::property_tree::ptree &tree)
        {
            load(tree);
        };

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

        boost::property_tree::ptree serialize()
        {
            namespace pt = boost::property_tree;

            pt::ptree root, layer, ts_fct, w, b;

            layer.put("threshold_function", "sigmoid");
            layer.put("input_size", weights.size2());
            layer.put("output_size", weights.size1());

            for (int x = 0; x < weights.size1(); x++)
            {
                pt::ptree row;
                for (int y = 0; y < weights.size2(); y++)
                {
                    pt::ptree value;
                    value.put("", weights(x, y));
                    row.push_back(std::make_pair("", value));
                }
                w.push_back(std::make_pair("", row));
            }
            layer.put_child("weights", w);

            for (int i = 0; i < weights.size1(); i++)
            {
                pt::ptree value;
                value.put("", biases[i]);
                b.push_back(std::make_pair("", value));
            }
            layer.put_child("biases", b);

            return layer;
        }

        //! Display function
        friend
        std::ostream &operator<< (std::ostream &oss, const Layer<T> &l)
        {
            boost::property_tree::write_json(oss, l.serialize());
            return oss;
        }

        bool load(boost::property_tree::ptree tree) throw(boost::property_tree::ptree_bad_path)
        {
            unsigned int input_size = tree.get("input_size", 0);
            unsigned int output_size = tree.get("output_size", 0);
            //TODO : select the right function
            std::string threshold_fct = tree.get("threshold_function", "");

            biases.resize(output_size);
            int i = 0;
            for (auto &b : tree.get_child("biases"))
            {
                if (i >= output_size)
                {
                    weights.resize(0, 0);
                    biases.resize(0, 0);
                    return false;
                }
                biases[i] = b.second.get("", T(0));
                i++;
            }

            weights.resize(output_size, input_size);
            int x = 0, y = 0;
            for (auto &row : tree.get_child("weights"))
            {
                y = 0;
                for (auto &cell : row.second)
                {
                    if(x >= output_size || y >= input_size)
                    {
                        weights.resize(0, 0);
                        biases.resize(0, 0);
                        return false;
                    }
                    weights(x, y) = cell.second.get("", T(0));
                    y++;
                }
                x++;
            }

            return true;
        }

        bool empty() const
        {
            if (weights.size1() == 0
                || weights.size2() == 0
                || biases.size() == 0)
                return true;
            return false;
        };

        bool valid() const
        {
            if(weights.size1() != biases.size())
                return false;
            if(empty())
                return false;
            return true;
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
    vector<T> operator>> (const vector<T> &input, const Layer<T> layer)
    {
        return layer << input;
    }
};

#endif /* !LAYER */
