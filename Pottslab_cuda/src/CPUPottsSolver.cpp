#include "CPUPottsSolver.h"
#include "potts/CudaPotts.cu"

CPUPottsSolver::CPUPottsSolver(float *inputImage, float newGamma, float newMuStep, size_t newW, size_t newH,
                               size_t newNc, uint32_t newChunkSize, float newStopTol, uint8_t newChunkOffsetChangeType,
                               uint32_t newMaxIterations, bool isVerbose, bool isQuadraticError) :
        PottsSolver(inputImage, newGamma, newMuStep, newW, newH, newNc, newChunkSize, newStopTol, newChunkOffsetChangeType,
                    newMaxIterations, isVerbose, isQuadraticError) {

    in = (float*)malloc(h*w*nc* sizeof(float));
    memcpy(in, inputImage, h*w*nc*sizeof(float));
    u = (float*)malloc(h*w*nc*sizeof(float));
    memset(u, 0, h*w*nc*sizeof(float));
    v = (float*)malloc(h*w*nc*sizeof(float));
    memcpy(v, in, h*w*nc*sizeof(float));
    w_ = (float*)malloc(h*w*nc*sizeof(float));
    memcpy(w_, in, h*w*nc*sizeof(float));
    z = (float*)malloc(h*w*nc*sizeof(float));
    memcpy(z, in, h*w*nc*sizeof(float));
    lam1 = (float*)malloc(h*w*nc*sizeof(float));
    memset(lam1, 0, h*w*nc*sizeof(float));
    lam2 = (float*)malloc(h*w*nc*sizeof(float));
    memset(lam2, 0, h*w*nc*sizeof(float));
    lam3 = (float*)malloc(h*w*nc*sizeof(float));
    memset(lam3, 0, h*w*nc*sizeof(float));
    lam4 = (float*)malloc(h*w*nc*sizeof(float));
    memset(lam4, 0, h*w*nc*sizeof(float));
    lam5 = (float*)malloc(h*w*nc*sizeof(float));
    memset(lam5, 0, h*w*nc*sizeof(float));
    lam6 = (float*)malloc(h*w*nc*sizeof(float));
    memset(lam6, 0, h*w*nc*sizeof(float));
    temp = (float*)malloc(h*w*nc*sizeof(float));
    memset(temp, 0, h*w*nc*sizeof(float));
    weights = (float*)malloc(h*w*sizeof(float));
    memset(weights, 0, h*w*sizeof(float));
    weightsPrime = (float*)malloc(h*w*sizeof(float));
    memset(weightsPrime, 0, h*w*sizeof(float));

    arrJ = (uint32_t*)malloc((dimension*2+1) * sizeof(uint32_t));
    memset(arrJ, 0, (dimension*2+1) * sizeof(uint32_t));
    arrP = (float*)malloc(dimension * sizeof(float));
    memset(arrP, 0, dimension * sizeof(float));
    m = (float*)malloc(dimension * nc * sizeof(float));
    memset(m, 0, dimension * nc * sizeof(float));
    s = (float*)malloc(dimension * sizeof(float));
    memset(s, 0, dimension * sizeof(float));
    wPotts = (float*)malloc(dimension * sizeof(float));
    memset(wPotts, 0, dimension * sizeof(float));
}

CPUPottsSolver::~CPUPottsSolver() {
    free((void*)in);
    free((void*)u);
    free((void*)v);
    free((void*)w_);
    free((void*)z);
    free((void*)lam1);
    free((void*)lam2);
    free((void*)lam3);
    free((void*)lam4);
    free((void*)lam5);
    free((void*)lam6);
    free((void*)temp);
    free((void*)weights);
    free((void*)weightsPrime);

    free((void*)arrJ);
    free((void*)arrP);
    free((void*)m);
    free((void*)s);
    free((void*)wPotts);
}

float CPUPottsSolver::updateError() {
    float error = 0;
    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            for(uint32_t c = 0; c < nc; ++c) {
                if(quadraticError) {
                    error += abs(temp[col + row*w + h*w*c]) * abs(temp[col + row*w + h*w*c]);
                } else {
                    error += abs(temp[col + row*w + h*w*c]);
                }

            }
        }
    }
    return error;
}

void CPUPottsSolver::clearHelperMemory() {
    memset(arrJ, 0, (dimension*2+1) * sizeof(uint32_t));
    memset(arrP, 0, dimension * sizeof(float));
    memset(m, 0, dimension * nc * sizeof(float));
    memset(s, 0, dimension * sizeof(float));
    memset(wPotts, 0, dimension * sizeof(float));
}

void CPUPottsSolver::horizontalPotts4ADMM(uint32_t nHor, uint32_t colorOffset) {
    uint32_t weightsIndex = 0;
    uint32_t index = 0;
    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            weightsIndex = col + row*w;
            for(uint32_t c = 0; c < nc; ++c) {
                index = col + row*w + h*w*c;
                u[index] = (weights[weightsIndex] * in[index] + v[index] * mu - lam1[index]) / weightsPrime[weightsIndex];
            }
        }
    }

    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < ceil((double)w/(double)chunkSize) + 1; ++col) {
            applyHorizontalPottsSolver(u, weightsPrime, arrJ, arrP, m, s, wPotts, gammaPrime, w, h, nc, nHor, colorOffset, chunkSize, chunkSizeOffset, row, col);
        }
    }

    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            for(uint32_t c = 0; c < nc; ++c) {
                copyDataBackHorizontally(u, arrJ, m, wPotts, row, col, c, w, h, colorOffset);
            }
        }
    }

    clearHelperMemory();
}

void CPUPottsSolver::verticalPotts4ADMM(uint32_t nVer, uint32_t colorOffset) {
    uint32_t weightsIndex = 0;
    uint32_t index = 0;
    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            weightsIndex = col + row*w;
            for(uint32_t c = 0; c < nc; ++c) {
                index = col + row*w + h*w*c;
                v[index] = (weights[weightsIndex] * in[index] + u[index] * mu + lam1[index]) / weightsPrime[weightsIndex];
            }
        }
    }

    for(uint32_t col = 0; col < w; ++col) {
        for(uint32_t row = 0; row < ceil((double)h/(double)chunkSize) + 1; ++row) {
            applyVerticalPottsSolver(v, weightsPrime, arrJ, arrP, m, s, wPotts, gammaPrime, w, h, nc, nVer, colorOffset, chunkSize, chunkSizeOffset, row, col);
        }
    }

    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            for(uint32_t c = 0; c < nc; ++c) {
                copyDataBackVertically(v, arrJ, m, wPotts, row, col, c, w, h, colorOffset);
            }
        }
    }

    clearHelperMemory();
}

void CPUPottsSolver::solvePottsProblem4ADMM() {
    if (0 == fNorm) {
        return;
    }
    uint32_t iteration = 0;

    uint32_t nHor = w;
    uint32_t nVer = h;
    uint32_t colorOffset = (w+1)*(h+1);

    float stopThreshold = stopTol * fNorm;

//    ImageRGB testImage(w, h);

    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            weights[col + row*w] = 1.0f;
        }
    }

    gammaPrime = 2 * gamma;

    while (error >= stopThreshold) {

        for(uint32_t row = 0; row < h; ++row) {
            for(uint32_t col = 0; col < w; ++col) {
                weightsPrime[col + row*w] = weights[col + row*w] + mu;
            }
        }

        horizontalPotts4ADMM(nHor, colorOffset);

        verticalPotts4ADMM(nVer, colorOffset);

        for(uint32_t row = 0; row < h; ++row) {
            for(uint32_t col = 0; col < w; ++col) {
                for(uint32_t c = 0; c < nc; ++c) {
                    uint32_t index = col + w * row + w * h * c;
                    temp[index] = u[index] - v[index];
                    lam1[index] = lam1[index] + temp[index] * mu;
                }
            }
        }

//        testImage.SetRawData(u);
//        testImage.Show("Test Image", 100+w, 100);
//        cv::waitKey(0);



        error = updateError();

//        if (verbose) {
//            printf("Iteration: %d error: %f\n", iteration, error);
//        }

        iteration++;

        mu = mu * muStep;

        updateChunkSizeOffset();

        if(iteration > maxIterations)
            break;
    }

    if (verbose) {
        printf("Iterations: %d\n", iteration);
    }
}

void CPUPottsSolver::horizontalPotts8ADMM(uint32_t nHor, uint32_t colorOffsetHorVer) {
    uint32_t weightsIndex = 0;
    uint32_t index = 0;
    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            weightsIndex = col + row*w;
            for(uint32_t c = 0; c < nc; ++c) {
                index = col + row*w + h*w*c;
                u[index] = (weights[weightsIndex] * in[index] + 2 * mu * (v[index] + w_[index] + z[index])
                            + 2 * (-lam1[index] - lam2[index] - lam3[index])) / weightsPrime[weightsIndex];
            }
        }
    }

    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < ceil((double)w/(double)chunkSize) + 1; ++col) {
            applyHorizontalPottsSolver(u, weightsPrime, arrJ, arrP, m, s, wPotts, gammaPrimeC, w, h, nc, nHor, colorOffsetHorVer, chunkSize, chunkSizeOffset, row, col);
        }
    }

    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            for(uint32_t c = 0; c < nc; ++c) {
                copyDataBackHorizontally(u, arrJ, m, wPotts, row, col, c, w, h, colorOffsetHorVer);
            }
        }
    }

    clearHelperMemory();

}

void CPUPottsSolver::verticalPotts8ADMM(uint32_t nVer, uint32_t colorOffsetHorVer) {
    uint32_t weightsIndex = 0;
    uint32_t index = 0;
    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            weightsIndex = col + row*w;
            for(uint32_t c = 0; c < nc; ++c) {
                index = col + row*w + h*w*c;
                v[index] = (weights[weightsIndex] * in[index] + 2 * mu * (u[index] + w_[index] + z[index])
                            + 2 * (lam1[index] - lam4[index] - lam5[index])) / weightsPrime[weightsIndex];
            }
        }
    }

    for(uint32_t col = 0; col < w; ++col) {
        for(uint32_t row = 0; row < ceil((double)h/(double)chunkSize) + 1; ++row) {
            applyVerticalPottsSolver(v, weightsPrime, arrJ, arrP, m, s, wPotts, gammaPrimeC, w, h, nc, nVer, colorOffsetHorVer, chunkSize, chunkSizeOffset, row, col);
        }
    }

    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            for(uint32_t c = 0; c < nc; ++c) {
                copyDataBackVertically(v, arrJ, m, wPotts, row, col, c, w, h, colorOffsetHorVer);
            }
        }
    }

    clearHelperMemory();
}

void CPUPottsSolver::diagonalPotts8ADMM(uint32_t nDiags, uint32_t colorOffsetDiags) {
    uint32_t weightsIndex = 0;
    uint32_t index = 0;
    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            weightsIndex = col + row*w;
            for(uint32_t c = 0; c < nc; ++c) {
                index = col + row*w + h*w*c;
                w_[index] = (weights[weightsIndex] * in[index] + 2 * mu * (u[index] + v[index] + z[index])
                             + 2 * (lam2[index] + lam4[index] - lam6[index])) / weightsPrime[weightsIndex];
            }
        }
    }



    for(uint32_t col = 0; col < h+w; ++col) {
        for(uint32_t row = 0; row < ceil((double)w/(double)chunkSize) + 1; ++row) {
            if(col < w) {
                applyDiagonalUpperPottsSolver(w_, weightsPrime, arrJ, arrP, m, s, wPotts, gammaPrimeD, w, h, nc, nDiags, colorOffsetDiags, chunkSize, chunkSizeOffset, row, col);
            } else if (col > w) {
                applyDiagonalLowerPottsSolver(w_, weightsPrime, arrJ, arrP, m, s, wPotts, gammaPrimeD, w, h, nc, nDiags, colorOffsetDiags, chunkSize, chunkSizeOffset, row, col);
            }
        }
    }

    uint32_t length = 0;
    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            for(uint32_t c = 0; c < nc; ++c) {
                length = std::min(h, w - col);
                if(row < length) {
                    copyDataBackDiagonallyUpper(w_, arrJ, m, wPotts, row, col, c, w, h, colorOffsetDiags, nDiags);
                }
            }
        }
    }
    for(uint32_t row = 1; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            for(uint32_t c = 0; c < nc; ++c) {
                length = std::min(h - row, w);
                if(col < length) {
                    copyDataBackDiagonallyLower(w_, arrJ, m, wPotts, row, col, c, w, h, colorOffsetDiags, nDiags);
                }
            }
        }
    }

    clearHelperMemory();
}

void CPUPottsSolver::antidiagonalPotts8ADMM(uint32_t nDiags, uint32_t colorOffsetDiags) {
    uint32_t weightsIndex = 0;
    uint32_t index = 0;
    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            weightsIndex = col + row*w;
            for(uint32_t c = 0; c < nc; ++c) {
                index = col + row*w + h*w*c;
                z[index] = (weights[weightsIndex] * in[index] + 2 * mu * (u[index] + v[index] + w_[index])
                            + 2 * (lam3[index] + lam5[index] + lam6[index])) / weightsPrime[weightsIndex];
            }
        }
    }

    for(uint32_t col = 0; col < h+w; ++col) {
        for(uint32_t row = 0; row < ceil((double)w/(double)chunkSize) + 1; ++row) {
            if(col < w) {
                applyAntiDiagonalUpperPottsSolver(z, weightsPrime, arrJ, arrP, m, s, wPotts, gammaPrimeD, w, h, nc, nDiags, colorOffsetDiags, chunkSize, chunkSizeOffset, row, col);
            } else if (col > w) {
                applyAntiDiagonalLowerPottsSolver(z, weightsPrime, arrJ, arrP, m, s, wPotts, gammaPrimeD, w, h, nc, nDiags, colorOffsetDiags, chunkSize, chunkSizeOffset, row, col);
            }
        }
    }

    uint32_t length = 0;
    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            for(uint32_t c = 0; c < nc; ++c) {
                length = std::min(h, w - col);
                if(row < length) {
                    copyDataBackAntiDiagonallyUpper(z, arrJ, m, wPotts, row, col, c, w, h, colorOffsetDiags, nDiags);
                }
            }
        }
    }
    for(uint32_t row = 1; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            for(uint32_t c = 0; c < nc; ++c) {
                length = std::min(h - row, w);
                if(col < length) {
                    copyDataBackAntiDiagonallyLower(z, arrJ, m, wPotts, row, col, c, w, h, colorOffsetDiags, nDiags);
                }
            }
        }
    }

    clearHelperMemory();
}

void CPUPottsSolver::solvePottsProblem8ADMM() {
    if (0 == fNorm) {
        return;
    }
    uint32_t iteration = 0;

    float stopThreshold = stopTol * fNorm;

    uint32_t nHor = w;
    uint32_t nVer = h;
    uint32_t colorOffsetHorVer = (w+1)*(h+1);

    uint32_t nDiags = std::min(h, w);
    uint32_t colorOffsetDiags = (std::min(h, w)+1)*(w+h-1);

//    ImageRGB testImage(w, h);

    float omegaC = sqrt(2.0) - 1.0;
    float omegaD = 1.0 - sqrt(2.0)/2.0;
    gammaPrimeC = 4.0 * omegaC * gamma;
    gammaPrimeD = 4.0 * omegaD * gamma;

    for(uint32_t row = 0; row < h; ++row) {
        for(uint32_t col = 0; col < w; ++col) {
            weights[col + row*w] = 1;
        }
    }

    while (error >= stopThreshold) {
        for(uint32_t row = 0; row < h; ++row) {
            for(uint32_t col = 0; col < w; ++col) {
                weightsPrime[col + row*w] = weights[col + row*w] + (6.0 * mu);
            }
        }

        horizontalPotts8ADMM(nHor, colorOffsetHorVer);

        diagonalPotts8ADMM(nDiags, colorOffsetDiags);

        verticalPotts8ADMM(nVer, colorOffsetHorVer);

        antidiagonalPotts8ADMM(nDiags, colorOffsetDiags);

//        testImage.SetRawData(z);
//        testImage.Show("Test Image", 100+w, 100);
//        testImage.SaveImage("antidiagonal.png");
//        cv::waitKey(0);

        for(uint32_t row = 0; row < h; ++row) {
            for (uint32_t col = 0; col < w; ++col) {
                for (uint32_t c = 0; c < nc; ++c) {
                    uint32_t index = col + w * row + w * h * c;
                    temp[index] = u[index] - v[index];
                    lam1[index] = lam1[index] + mu * (u[index] - v[index]);
                    lam2[index] = lam2[index] + mu * (u[index] - w_[index]);
                    lam3[index] = lam3[index] + mu * (u[index] - z[index]);
                    lam4[index] = lam4[index] + mu * (v[index] - w_[index]);
                    lam5[index] = lam5[index] + mu * (v[index] - z[index]);
                    lam6[index] = lam6[index] + mu * (w_[index] - z[index]);
                }
            }
        }

        error = updateError();

//        if (verbose) {
//            printf("Iteration: %d error: %f\n", iteration, error);
//        }

        iteration++;

        mu = mu * muStep;

        updateChunkSizeOffset();

        if(iteration > maxIterations)
            break;
    }

    if (verbose) {
        printf("Iterations: %d\n", iteration);
    }
}

/*void CPUPottsSolver::downloadOutputImage(ImageRGB outputImage) {
    outputImage.SetRawData(u);
}*/

float* CPUPottsSolver::getResultPtr() {
    return u;
}

void CPUPottsSolver::downloadOutputMatlab(float *outputImage) {
    memcpy(outputImage, u, h*w*nc*sizeof(float));
}