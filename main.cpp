#include "Layer.hpp"
#include <boost/numeric/ublas/io.hpp>

using namespace ffnn;

int main () {
    mapped_vector<float> v(10);

    Layer<float> layer(10, 4, ffnn::sigmoid<float>);
    layer.engine.seed(42);
    layer.randomize();

    std::cout << layer << std::endl;
    std::cout << (layer << v) << std::endl;
    return 0;
}
