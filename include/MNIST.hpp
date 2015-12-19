#ifndef MNIST_HPP_
#define MNIST_HPP_

#include <boost/numeric/ublas/matrix.hpp>
#include <list>
#include <string>

namespace MNIST
{
    using namespace boost::numeric::ublas;

    typedef unsigned char word8;

    class ImageSet
    {
    public:
        void load(std::string filename);

        unsigned int count;
        unsigned int magic;
        unsigned int w;
        unsigned int h;

        std::vector<std::vector<word8>> images;
    };

    class LabelSet
    {
    public:
        void load(std::string filename);

        unsigned int magic;
        unsigned int count;
        std::vector<word8> labels;
    };
};

#endif /* !MNIST_HPP_ */
