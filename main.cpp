#include "Layer.hpp"
#include "Network.hpp"
#include <boost/numeric/ublas/io.hpp>

using namespace ffnn;

int main ()
{
    //Create network

    Layer<float> layer1(784, 15, ffnn::sigmoid<float>);
    Layer<float> layer2(15, 10, ffnn::sigmoid<float>);

    layer1.randomize();
    layer2.randomize();

    Network<float> net;

    if (!net.connect_layer(layer1) || !net.connect_layer(layer2))
    {
        std::cout << "Can't connect layers" << std::endl;
    }

    //Load MNIST dataset
    ImageSet imgset;

    std::cout << "Foward: " << std::endl;
    for (auto output : net.forward(v))
        std::cout << output << std::endl;

    for (int i = 0; i < 10000; i++)
    {
        //Training here
    }

    return 0;
}
