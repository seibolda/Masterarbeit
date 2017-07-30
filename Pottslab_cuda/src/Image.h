#ifndef _IMAGE_H_
#define _IMAGE_H_

#include "util/helper.h"
#include <cstdint>
#include <string>
#include <iostream>

using namespace cv;

typedef cv::Mat CVMat;
typedef cv::Mat* CVMatPtr;
typedef float* RawBufferPtr;

class ImageGS;

class Image
{
public:
	Image(std::string& filename, bool normalize);
	//Image(uint32_t width, uint32_t height, uint32_t nChannels);
	Image();
	virtual ~Image();
	
	virtual void Load(std::string filename, bool normalize) = 0;
	void SetRawData(float* newData);
	void SetMatrix(CVMat& newMat);
	void UpdateRawData();
	void UpdateMatrix();
	
	void Show(const std::string& windowName, float factor, uint32_t posX, uint32_t posY);
	void Show(const std::string& windowName, uint32_t posX, uint32_t posY);
	virtual void ConvertAndScale(CVMat& output, float scale) {}
	
	uint32_t GetWidth();
	uint32_t GetHeight();
	uint32_t GetChannels();
	std::string GetFilename();
	
	size_t GetSizeInBytes();
	size_t GetChannelSizeInBytes();
	
	size_t GetSizeInElements();
	size_t GetChannelSizeInElements();
	
	bool IsNormalized();
	
	CVMatPtr GetMatrixCV();
    void SaveImage(std::string name);
	RawBufferPtr GetRawDataPtr();

private:
	void convert_layered_to_interleaved(float *aOut, const float *aIn, int w, int h, int nc);
	void convert_interleaved_to_layered(float *aOut, const float *aIn, int w, int h, int nc);
	
	
protected:
	void convert_mat_to_layered();
	void convert_layered_to_mat();	
	
	std::string m_filename;
	bool m_normalized;
	
	RawBufferPtr m_rawData;
	CVMat m_mat;
};

class ImageRGB : public Image
{
public:
	ImageRGB(std::string filename, bool normalize);
	ImageRGB(uint32_t width, uint32_t height);
	ImageRGB();
	~ImageRGB();
	
	void Load(std::string filename, bool normalize) override;
	//void ScaleChannelValues(float scale) override;
	ImageGS ToGS();
};

class ImageGS : public Image
{
public:
	ImageGS(std::string filename, bool normalize);
	ImageGS(uint32_t width, uint32_t height);
	ImageGS();
	~ImageGS();
	
	void Load(std::string filename, bool normalize) override;
	void ConvertAndScale(CVMat& output, float scale) override;
	void SetTo(float valueToSet, CVMat otherImage, int refValue)
	{
		m_mat.setTo(valueToSet, otherImage == refValue);
		convert_mat_to_layered();
	}
	ImageRGB ToRGB();
};

class ImageDepth : public Image
{
public:
	ImageDepth(std::string filename, bool normalize);
	//ImageGeneric(uint32_t width, uint32_t height);
	ImageDepth();
	~ImageDepth();
	
	void Load(std::string filename, bool normalize) override;
	void ConvertAndScale(CVMat& output, float scale) override;
	ImageGS ToGS();
};

#endif
