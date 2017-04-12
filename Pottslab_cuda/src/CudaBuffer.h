#ifndef _CUDABUFFER_H_
#define _CUDABUFFER_H_

#include "helper.h"

template <class T>
class CudaBuffer
{
public:
	CudaBuffer(T* data, size_t bufferSizeElements);
	CudaBuffer();
	~CudaBuffer();
	
	void CreateBuffer(size_t bufferSizeElements);
	void UploadData(T* data);
	T* DownloadData();
	void SetBytewiseValue(int32_t value);
	void CopyDeviceToDevice(T* d_src);
	void DestroyBuffer();
	
	T* GetDevicePtr();
	size_t GetBufferSizeBytes();
	
private:
	T* m_devicePtr;
	size_t m_bufferSizeBytes;
};

#endif
