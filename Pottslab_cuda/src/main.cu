#include <iostream>
#include <opencv2/opencv.hpp>
#include "helper.h"

using namespace std;

int main(int argc, char **argv) {

    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;

    string image_path = "";
    bool ret = getParam("i", image_path, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> -gamma <gamma_value>" << endl; return 1; }

    string gamma = "";
    ret = getParam("gamma", gamma, argc, argv);
    if (!ret) cerr << "ERROR: no gamma value given" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> -gamma <gamma_value>" << endl; return 1; }

    cv::Mat mIn = cv::imread(image_path.c_str(), -1);

    showImage("imageIn", mIn, 100, 100);


    cv::waitKey(0);
    cvDestroyAllWindows();

    return 0;
}