#include "Layer.hpp"
#include "Network.hpp"
#include <boost/numeric/ublas/io.hpp>

using namespace ffnn;

int main () {
    mapped_vector<float> v(10);
    mapped_vector<float> w(2);
    v[1] = 42;
    w[1] = 1;

    Layer<float> layer1(10, 4, ffnn::sigmoid<float>);
    Layer<float> layer2(4, 2, ffnn::sigmoid<float>);

    layer1.randomize();
    layer2.randomize();

    Network<float> net;

    std::cout << net.connect_layer(layer1) << std::endl;
    std::cout << net.connect_layer(layer2) << std::endl;

    std::cout << "Foward: " << std::endl;
    for (auto output : net.forward(v))
        std::cout << output << std::endl;

    std::cout << std::endl;

    std::cout << "Eval : " << std::endl;
    std::cout << net.eval(v) << std::endl;

    for (int i = 0; i < 300; i++)
    {
        std::cout << "Training :" << std::endl;
        net.train(3, v, w);
        std::cout << net.eval(v) << std::endl;
    }
    return 0;
}
