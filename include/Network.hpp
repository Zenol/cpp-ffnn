#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include <Layer.hpp>

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>

namespace ffnn
{
    using namespace boost::numeric::ublas;

    template<typename T>
    class Network
    {
    public:
        typedef std::vector<Layer<T>> layer_list;

        //! Push back a layer on the layer list
        //! if the input size match the previous layer
        //! output size (number of neurons).
        bool connect_layer(const Layer<T> &layer)
        {
            if (!layers.empty())
                if (layers.back().get_output_size() != layer.get_input_size())
                    return false;

            layers.push_back(layer);
            return true;
        }

        //! Remove the last layer on the layer list.
        void disconnect_layer()
        {layers.pop_back();};

        //! Return the list of layers
        const layer_list& get_layers()
        {return layers;};

        //! Compute forward pass of the network
        //! \return The list of neuron outputs. The input is saw as the first layer.
        std::vector<mapped_vector<T>> forward(const mapped_vector<T> &input)
        {
            std::vector<mapped_vector<T>> out_list;

            out_list.push_back(input);
            for (auto layer : layers)
                out_list.push_back(layer << out_list.back());

            return out_list;
        }

        //! Evaluate a network
        mapped_vector<T> eval(const mapped_vector<T> &input)
        {
            mapped_vector<T> output(input);
            for (auto layer : layers)
                output = output >> layer;
            return output;
        }

        void forward(const mapped_vector<T> &input, const mapped_vector<T> &output)
        {
            //////////////////////////////////////////
            // Compute the forward pass from the input
            //
            auto a_vec = forward(input);

            /////////////
            // Compute the delta_list, wich is the list of all derivative
            // dC_over_dz where z_l is the weightenen sum of an input incoming
            // into the layer l.

            // L means the last layer, and l a layer beetween 1(input) and L.
            // dC_over_da means the gradient of C on the direction a.
            auto dC_over_da = a_vec.back() - output;

            // The delta list is the list of all gradients
            std::vector<mapped_vector<T>> delta_list;

            // Reverse order browsing of outputs of neurons
            auto a_vec_it = a_vec.rbegin();
            //Delta L :
            auto delta_L = element_prod(dC_over_da, a_vec_it->threshold_function % *a_vec_it);
            delta_list.push_front(delta_L);
            for (auto l : layers)
            {
                auto exp1 = prod(trans(l.weights), delta_list.front());
                auto exp2 = l.derivative_function % *a_vec_it;
                auto r = element_prod(exp1, exp2);

                a_vec_it++;
            };

            ///////////////////////////////////////////////////////////////
            // Compute the derivative of the wieghts and the biases, namely
            // dC_over_dw and dC_over_db.
            
        }
    private:
        layer_list layers;
    };
}

#endif /* !NETWORK_HPP_ */
