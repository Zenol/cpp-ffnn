#include "Layer.hpp"
#include "Network.hpp"
#include "MNIST.hpp"
#include <boost/numeric/ublas/io.hpp>

using namespace ffnn;

int main ()
{
    //Create network

    Layer<double> layer1(784, 15, ffnn::sigmoid<double>, ffnn::sigmoid_prime<double>);
    Layer<double> layer2(15, 10, ffnn::sigmoid<double>, ffnn::sigmoid_prime<double>);

    layer1.randomize();
    layer2.randomize();

    Network<double> net;

    if (!net.connect_layer(layer1) || !net.connect_layer(layer2))
    {
        std::cout << "Can't connect layers" << std::endl;
    }

    //Convert it to a list of matrices
    std::vector<vector<double>> img_list;
    std::vector<mapped_vector<double>> label_list;
    {
        //Load MNIST dataset
        std::cout << "MNIST loading..." << std::endl;
        MNIST::ImageSet imgset;
        imgset.load("train-images-idx3-ubyte");
        MNIST::LabelSet labelset;
        labelset.load("train-labels-idx1-ubyte");

        for (int i = 0; i < imgset.count; i++)
        {
            vector<double> img(imgset.images[i].size());
            for (int j = 0; j < imgset.images[i].size(); j++)
                img[j] = imgset.images[i][j];

            if (i % 1000 == 0)
                std::cout << "Loaded: " << i << "\r" << std::flush;

            img_list.push_back(std::move(img));
            label_list.push_back(unit_vector<double>(10, labelset.labels[i]));
        }
    }
    std::cout << "MNIST Loaded     " << std::endl;

    //Training network
    for (int i = 0; i < 5000 /*img_list.size()*/ ; i++)
    {
        net.train(3, img_list[i], label_list[i]);

        if (i % 10 == 0)
            std::cout << "Trained: " << i << "\r" << std::flush;
    }


    //Checking efficiency
    int count = 0;
    for (int i = 0; i < 20; i++) // img_list.size()
    {
//        count += net.eval(unit_vector<double>(10, img_list[i]) == label_list[i]);

        if (i % 1000 == 0)
            std::cout << "Checked: " << i << "\r" << std::flush;

        std::cout << net.eval(img_list[i]) << "\n" << label_list[i] << std::endl;
    }
    std::cout << "Efficiency\t\t\t" << std::endl;
    std::cout << (float)count / (float)img_list.size() << std::endl;

    return 0;
}
