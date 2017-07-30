#include "Image.h"

//
// Image
//
Image::Image(std::string& filename, bool normalize)
	: m_filename(filename), m_normalized(normalize), m_rawData(nullptr)
{
}

Image::Image()
	: m_rawData(nullptr), m_normalized(false), m_filename("")
{
}

Image::~Image()
{
	delete[] m_rawData;
}

uint32_t Image::GetWidth()
{
	return m_mat.cols;
}

uint32_t Image::GetHeight()
{
	return m_mat.rows;
}

uint32_t Image::GetChannels()
{
	return m_mat.channels();
}

CVMatPtr Image::GetMatrixCV()
{
	return &m_mat;
}

void Image::SaveImage(std::string name)
{
    convert_layered_to_mat();
	cv::imwrite(name, m_mat*255.0f);
}

size_t Image::GetSizeInBytes()
{
	return (this->GetHeight()*this->GetWidth()*this->GetChannels()*sizeof(float));
}

size_t Image::GetSizeInElements()
{
	return (this->GetHeight()*this->GetWidth()*this->GetChannels());
}

size_t Image::GetChannelSizeInBytes()
{
	return (this->GetWidth()*this->GetHeight()*sizeof(float));
}

size_t Image::GetChannelSizeInElements()
{
	return (this->GetWidth()*this->GetHeight());
}

RawBufferPtr Image::GetRawDataPtr()
{
	return m_rawData;
}

void Image::UpdateRawData()
{
	convert_mat_to_layered();
}

void Image::SetRawData(float* newData)
{
	m_rawData = newData;
	this->UpdateMatrix();
}

void Image::UpdateMatrix()
{
	convert_layered_to_mat();
}

void Image::SetMatrix(CVMat& newMat)
{
	m_mat = newMat;
	this->UpdateRawData();
}

void Image::Show(const std::string& windowName, float factor, uint32_t posX, uint32_t posY)
{
	showImage(windowName, m_mat*factor, (int)posX, (int)posY);
}

void Image::Show(const std::string& windowName, uint32_t posX, uint32_t posY)
{
	showImage(windowName, m_mat, (int)posX, (int)posY);
}

void Image::convert_layered_to_interleaved(float *aOut, const float *aIn, int w, int h, int nc)
{
	if (nc==1) { memcpy(aOut, aIn, w*h*sizeof(float)); return; }
    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[(nc-1-c) + nc*(x + (size_t)w*y)] = aIn[x + (size_t)w*y + nOmega*c];
            }
        }
    }
}

void Image::convert_interleaved_to_layered(float *aOut, const float *aIn, int w, int h, int nc)
{
	if (nc==1) { memcpy(aOut, aIn, w*h*sizeof(float)); return; }
    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[x + (size_t)w*y + nOmega*c] = aIn[(nc-1-c) + nc*(x + (size_t)w*y)];
            }
        }
    }
}

void Image::convert_mat_to_layered()
{
	convert_interleaved_to_layered(m_rawData, (float*)m_mat.data, m_mat.cols, m_mat.rows, m_mat.channels());
}

void Image::convert_layered_to_mat()
{
	convert_layered_to_interleaved((float*)m_mat.data, m_rawData, m_mat.cols, m_mat.rows, m_mat.channels());
}

//
// RGB
//
ImageRGB::ImageRGB(std::string filename, bool normalize)
	: Image()
{
	m_filename = filename;
	m_normalized = normalize;
	Load(filename, normalize);
}

ImageRGB::ImageRGB(uint32_t width, uint32_t height)
	: Image()
{
	m_rawData = new float[width*height*3];
	CVMat m(height, width, CV_32FC3);
	m_mat = m;
	m_mat.convertTo(m_mat, CV_32FC3);
}

ImageRGB::ImageRGB()
	: Image()
{
}

ImageRGB::~ImageRGB()
{
}

void ImageRGB::Load(std::string filename, bool normalize)
{
	m_filename = filename;
	//std::cout << filename << std::endl;
	m_mat = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	//std::cout << "RGB matrix loaded" << std::endl;
	m_mat.convertTo(m_mat,CV_32FC3);
	//std::cout << "RGB matrix converted" << std::endl;
    if(normalize)
    	m_mat /= 255.0f;
    
    m_rawData = new float[(size_t)(m_mat.rows * m_mat.cols * m_mat.channels())];
    //std::cout << "Raw data allocation" << std::endl;
    
    convert_mat_to_layered();
}

ImageGS ImageRGB::ToGS()
{
	return ImageGS(m_filename, m_normalized);
}

//
// GS
//
ImageGS::ImageGS(std::string filename, bool normalize)
	: Image()
{
	m_filename = filename;
	m_normalized = normalize;
	Load(filename, normalize);
}

ImageGS::ImageGS(uint32_t width, uint32_t height)
	: Image()
{
	m_rawData = new float[width*height];
	CVMat m(height, width, CV_32FC1);
	m_mat = m;
	m_mat.convertTo(m_mat, CV_32FC1);
}

ImageGS::ImageGS()
	: Image()
{
}

ImageGS::~ImageGS()
{
}

void ImageGS::Load(std::string filename, bool normalize)
{
	m_mat = cv::imread(filename.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	m_mat.convertTo(m_mat,CV_32FC1);
    if(normalize)
    	m_mat /= 255.0f;
    
    m_rawData = new float[(size_t)(m_mat.rows * m_mat.cols)];
    
    convert_mat_to_layered();
}

void ImageGS::ConvertAndScale(CVMat& output, float scale)
{
	m_mat.convertTo(m_mat, CV_32FC1, scale);
	
	convert_mat_to_layered();
}

ImageRGB ImageGS::ToRGB()
{
	return ImageRGB(m_filename, m_normalized);
}

//
// Generic
//
ImageDepth::ImageDepth(std::string filename, bool normalize)
	: Image()
{
	m_filename = filename;
	m_normalized = normalize;
	Load(filename, normalize);
}

ImageDepth::ImageDepth()
	: Image()
{
}

ImageDepth::~ImageDepth()
{
}

void ImageDepth::Load(std::string filename, bool normalize)
{
	m_mat = cv::imread(filename.c_str(), -1);
	if(normalize)
		m_mat /= 255.0f;
	
	m_rawData = new float[(size_t)(m_mat.rows * m_mat.cols)];
	
	convert_mat_to_layered();
}

void ImageDepth::ConvertAndScale(CVMat& output, float scale)
{
	m_mat.convertTo(output, CV_32FC1, scale);
	
	convert_mat_to_layered();
}

ImageGS ImageDepth::ToGS()
{
	return ImageGS(m_filename, m_normalized);
}
