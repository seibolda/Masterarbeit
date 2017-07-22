#include <iostream>
#include <opencv2/opencv.hpp>
#include "Image.h"

#include "GPUPottsSolver.h"
#include "CPUPottsSolver.h"


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
    double muStep = 2;
    uint32_t chunkSize = 0;
    float stopTol = 1e-10;
    uint8_t chunkOffsetChangeType = 0;
    uint32_t maxIterations = 25;
    bool verbose = false;
    bool quadraticError = true;
    bool isotropic = true;
    bool isGPU = false;

    Timer timer;

    string usageString = " -i <image> -gamma <float_value> (-chunksize <uint_value> -stoptol <float_value>";
    usageString += " -chunkoffsetchangetype [0|1|2] -maxiterations <uint_value> -v [true|false]";
    usageString += " -nonquadraticerror [true|false] -unisotropic [true|false] -gpu [true|false])";

    string image_path = "";
    bool ret = getParam("i", image_path, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc < 5) { cout << "Usage: " << argv[0] << usageString << endl; return 1; }

    string gamma_str = "";
    ret = getParam("gamma", gamma_str, argc, argv);
    if (!ret) cerr << "ERROR: no gamma value given" << endl;
    if (argc < 5) { cout << "Usage: " << argv[0] << usageString << endl; return 1; }
    gamma = ::atof(gamma_str.c_str());

    string chunksize_str = "";
    ret = getParam("chunksize", chunksize_str, argc, argv);
    if(ret) {
        chunkSize = ::atoi(chunksize_str.c_str());
    }

    string stopTol_str = "";
    ret = getParam("stoptol", stopTol_str, argc, argv);
    if(ret) {
        stopTol = ::atof(stopTol_str.c_str());
    }

    string chunkOffsetChangeType_str = "";
    ret = getParam("chunkoffsetchangetype", chunkOffsetChangeType_str, argc, argv);
    if(ret) {
        chunkOffsetChangeType = ::atoi(chunkOffsetChangeType_str.c_str());
    }

    string maxIterations_str = "";
    ret = getParam("maxiterations", maxIterations_str, argc, argv);
    if(ret) {
        maxIterations = ::atoi(maxIterations_str.c_str());
    }

    string verbose_str = "";
    ret = getParam("v", verbose_str, argc, argv);
    verbose = ret;

    string quadraticError_str = "";
    ret = getParam("nonquadraticerror", quadraticError_str, argc, argv);
    quadraticError = !ret;

    string isotropic_str = "";
    ret = getParam("unisotropic", isotropic_str, argc, argv);
    isotropic = !ret;

    string isgpu_str = "";
    ret = getParam("gpu", isgpu_str, argc, argv);
    isGPU = ret;

    ImageRGB inputImage(image_path, true);
    height = inputImage.GetHeight();
    width = inputImage.GetWidth();
    numberChannels = inputImage.GetChannels();
    ImageRGB outputImage(width, height);




    if(isGPU) {
        GPUPottsSolver gpuPottsSolver(inputImage.GetRawDataPtr(), gamma, muStep, width, height, numberChannels, chunkSize,
                                      stopTol, chunkOffsetChangeType, maxIterations, verbose, quadraticError);

        timer.start();
        if(isotropic) {
            gpuPottsSolver.solvePottsProblem8ADMM();
        } else {
            gpuPottsSolver.solvePottsProblem4ADMM();
        }
        timer.end();

        outputImage.SetRawData(gpuPottsSolver.getResultPtr());
        inputImage.Show("Input Image", 100, 100);
        outputImage.Show("Output Image", 100+width, 100);
        cv::waitKey(0);
    } else {
        CPUPottsSolver cpuPottsSolver(inputImage.GetRawDataPtr(), gamma, muStep, width, height, numberChannels, chunkSize,
                                      stopTol, chunkOffsetChangeType, maxIterations, verbose, quadraticError);

        timer.start();
        if(isotropic) {
            cpuPottsSolver.solvePottsProblem8ADMM();
        } else {
            cpuPottsSolver.solvePottsProblem4ADMM();
        }
        timer.end();

        outputImage.SetRawData(cpuPottsSolver.getResultPtr());

        inputImage.Show("Input Image", 100, 100);
        outputImage.Show("Output Image", 100+width, 100);
        cv::waitKey(0);
    }





    if(verbose) {
        cout << "Duration: " << timer.get() * 1000 << "ms" << endl;
    }

    cvDestroyAllWindows();

    return 0;
}