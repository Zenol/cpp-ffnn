#include "Layer.hpp"
#include <boost/numeric/ublas/io.hpp>

using namespace ffnn;

int main () {
    mapped_vector<float> v(10, 1);

    Layer<float> layer(10, 4, std::function<float(float)>(ffnn::sigmoid));

    std::cout << (v >> layer);
    return 0;
}
