#include "MNIST.hpp"

#include <fstream>

namespace MNIST
{

    void ImageSet::load(std::string filename)
    {
        std::ifstream ifs(filename, std::ios::binary);

        ifs.read(reinterpret_cast<char*>(&magic), 4);
        ifs.read(reinterpret_cast<char*>(&count), 4);
        ifs.read(reinterpret_cast<char*>(&h), 4);
        ifs.read(reinterpret_cast<char*>(&w), 4);
        magic = be32toh(magic);
        count = be32toh(count);
        h = be32toh(h);
        w = be32toh(w);

        for (int i = 0; i < count; i++)
        {
            std::vector<word8> image(count);
            ifs.read(reinterpret_cast<char*>(image.data()), w * h);
            images.push_back(std::move(image));
        }
    }

    void LabelSet::load(std::string filename)
    {
        std::ifstream ifs(filename, std::ios::binary);

        ifs.read(reinterpret_cast<char*>(&magic), 4);
        ifs.read(reinterpret_cast<char*>(&count), 4);
        magic = be32toh(magic);
        count = be32toh(count);

        labels.resize(count);
        ifs.read(reinterpret_cast<char*>(labels.data()), count);
    }
}
