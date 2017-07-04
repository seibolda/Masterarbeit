#include "CudaBuffer.h"

template <class T>
CudaBuffer<T>::CudaBuffer(T* data, size_t bufferSizeElements)
	: m_devicePtr(nullptr), m_bufferSizeBytes(bufferSizeElements*sizeof(T))
{
	this->CreateBuffer(bufferSizeElements);
	this->UploadData(data);
}

template <class T>
CudaBuffer<T>::CudaBuffer()
	: m_devicePtr(nullptr), m_bufferSizeBytes(0)
{
}

template <class T>
CudaBuffer<T>::~CudaBuffer()
{
}

template <class T>
void CudaBuffer<T>::CreateBuffer(size_t bufferSizeElements)
{
	m_bufferSizeBytes = bufferSizeElements*sizeof(T);
	cudaMalloc(&m_devicePtr, m_bufferSizeBytes*sizeof(T)); CUDA_CHECK;
}

template <class T>
void CudaBuffer<T>::UploadData(T* data)
{
	cudaMemcpy(m_devicePtr, data, m_bufferSizeBytes, cudaMemcpyHostToDevice); CUDA_CHECK;
}

template <class T>
T* CudaBuffer<T>::DownloadData()
{
	T* retVal = new T[m_bufferSizeBytes];
	cudaMemcpy(retVal, m_devicePtr, m_bufferSizeBytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
	return retVal;
}

template <class T>
void CudaBuffer<T>::SetBytewiseValue(int32_t value)
{
	cudaMemset(m_devicePtr, value, m_bufferSizeBytes); CUDA_CHECK;
}

template <class T>
void CudaBuffer<T>::CopyDeviceToDevice(T* d_src)
{
    cudaMemcpy(m_devicePtr, d_src, m_bufferSizeBytes, cudaMemcpyDeviceToDevice); CUDA_CHECK;
}

template <class T>
void CudaBuffer<T>::DestroyBuffer()
{
	cudaFree(m_devicePtr); CUDA_CHECK;
}

template <class T>
T* CudaBuffer<T>::GetDevicePtr()
{
	return m_devicePtr;
}

template <class T>
size_t CudaBuffer<T>::GetBufferSizeBytes()
{
	return m_bufferSizeBytes;
}

template class CudaBuffer<float>;
template class CudaBuffer<int16_t>;
template class CudaBuffer<uint32_t>;
template class CudaBuffer<int32_t>;
