#include <iostream>
#include <opencv2/opencv.hpp>
#include "Image.h"

#include "GPUPottsSolver.cu"

using namespace std;

int main(int argc, char **argv) {

    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;

    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t numberChannels = 0;
    double gamma = 0;
    Timer timer;

    string image_path = "";
    bool ret = getParam("i", image_path, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> -gamma <gamma_value>" << endl; return 1; }

    string gamma_str = "";
    ret = getParam("gamma", gamma_str, argc, argv);
    if (!ret) cerr << "ERROR: no gamma value given" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> -gamma <gamma_value>" << endl; return 1; }
    gamma = ::atof(gamma_str.c_str());

    ImageRGB inputImage(image_path, true);
    height = inputImage.GetHeight();
    width = inputImage.GetWidth();
    numberChannels = inputImage.GetChannels();
    ImageRGB outputImage(width, height);


    GPUPottsSolver gpuPottsSolver(inputImage.GetRawDataPtr(), gamma, width, height, numberChannels);

    timer.start();
    gpuPottsSolver.solvePottsProblem();
    timer.end();

    gpuPottsSolver.downloadOuputImage(outputImage);


    cout << "Duration: " << timer.get() * 1000 << "ms" << endl;

    inputImage.Show("Input Image", 100, 100);
    outputImage.Show("Output Image", 500, 100);
    cv::waitKey(0);
    cvDestroyAllWindows();

    return 0;
}