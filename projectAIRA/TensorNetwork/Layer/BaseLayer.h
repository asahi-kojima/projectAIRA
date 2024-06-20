#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <map>
#include <memory>
#include <vector>
#include <iostream>
#include <cassert>
#include "debug-setting.h"
#include "typeinfo.h"
#include "Tensor/Tensor.h"

namespace aoba
{
	namespace nn
	{
		namespace optimizer
		{
			class Optimizer;
		}

		namespace layer
		{

			class BaseLayer;
			class Layer;

			//PrimitiveLayer
			class AffineCore;
			class ReLUCore;
			class AddCore;
			class SplitCore;
			class AddAsInnerCore;
			class SequentialCore;
			class Convolution;

			//LossFunction
			class CrossEntropyWithSMCore;
		}

		namespace tensor
		{
			class IOTensor;
			class Tensor;
			class TensorCore;
		}
	}
}


//class aoba::nn::tensor::IOTensor
//{
//public:
//	template<typename ... Args>
//	IOTensor(Args ... args)
//		:mTensorTbl(0)
//	{
//		tensor::Tensor inputTbl[] = {args...};
//		mInputNum = sizeof(inputTbl) / sizeof(inputTbl[0]);
//
//		for (u32 i = 0; i < mInputNum; i++)
//		{
//			mTensorTbl.push_back(inputTbl[i]);
//		}
//	}
//
//	u32 mInputNum;
//	std::vector<tensor::Tensor> mTensorTbl;
//
//	//Tensor operator[](u32 index)
//	//{
//	//	if (index >= mInputNum)
//	//	{
//	//		assert(0);
//	//	}
//
//	//	return mTensorTbl[index];
//	//}
//};


//�R���X�g���N�^�Ŏq�e���\����shared_ptr������this��o�^�������Čp���B
//��肪�N�����炱�����^���B
class aoba::nn::layer::BaseLayer : public std::enable_shared_from_this<BaseLayer>
{
public:

	friend class aoba::nn::tensor::TensorCore;
	using iotype = std::vector<tensor::Tensor>;

	BaseLayer(u32 = 1, u32 = 1);
	BaseLayer(u32, u32, u32);
	BaseLayer(u32, u32, u32, u32);
	virtual ~BaseLayer() {}

	iotype callForward(const iotype&);
	void callBackward(u32 downstream_index);
	//void regist_this_to_output_tensor();

	u32 get_input_tensor_num() const { return m_input_tensor_num; }
	u32 get_output_tensor_num() const { return m_output_tensor_num; }


	bool isOnCuda() const
	{
		return m_on_cuda;
	}

	const std::vector<std::shared_ptr<aoba::nn::tensor::TensorCore> >& getTrainableParamTbl() const
	{
		return mTrainableParameterTbl;
	}

	const std::map<std::string, Layer >& getInternalLayerTbl() const
	{
		return m_internal_layer_tbl;
	}

protected:
	using TensorCore = aoba::nn::tensor::TensorCore;
	using Tensor = aoba::nn::tensor::Tensor;

	bool having_unique_implimention = true;
	bool m_init_finish = false;
	bool m_on_cuda = false;



	std::vector<std::shared_ptr<TensorCore> > mTrainableParameterTbl;
	std::map<std::string, Layer > m_internal_layer_tbl;
	std::map<std::string, Layer>& mlayer;//��L�̃G�C���A�X:���̂܂܂��ƒ�������

	/// <summary>
	/// ���̑w�����������e���\��
	///�i�e�w�̓e���\���p�̃������𒼐ڌ��Ă���C���[�W�j
	/// </summary>
	std::vector<std::shared_ptr<TensorCore>> m_output_tensorcore_tbl;

	/// <summary>
	/// ���`���ŃC���v�b�g���ꂽ�e���\�������o���Ă����p
	/// ���ꂪ�Ȃ��Ƌt�`���������ōs���Ȃ��Ȃ�B
	/// </summary>
	std::vector<std::weak_ptr<TensorCore> > mInputTensorCoreTbl;


	//���͂Əo�͂̃e���\���̐����L�^
	const u32 m_input_tensor_num;
	const u32 m_output_tensor_num;
	const u32 m_trainable_parameter_num;

	//void init_childtensor_with(std::shared_ptr<LayerSkeleton>&& )



	const std::shared_ptr<TensorCore>& getTensorCoreFrom(const Tensor& tensor);

	void initialize();
	void genDownStreamTensor(u32 childNo);


	//TensorCore�ɃA�N�Z�X���邽�߂̊֐��Q

private:

	std::vector<bool> m_downstream_backward_checkTbl;

	void disconnect_upstream_tensorcore(s32 location)
	{
		mInputTensorCoreTbl[location].reset();
	}

	/// <summary>
	/// �e�w���Ǝ��ɍs��forward�����͂��̉��z�֐��Ɏ�������B
	/// </summary>
	/// <param name="input_tensors"></param>
	/// <returns></returns>
	virtual iotype forward(const iotype& input_tensors) = 0;

	/// <summary>
	/// �e�w���Ǝ��ɍs��forward�����͂��̉��z�֐��Ɏ�������B
	/// </summary>
	/// <param name="input_tensors"></param>
	/// <returns></returns>
	virtual void backward()
	{
		having_unique_implimention = false;
	}
};


using nnModule = aoba::nn::layer::BaseLayer;


namespace aoba
{
	namespace nn
	{
		namespace layer
		{

			BaseLayer::iotype operator+(const BaseLayer::iotype& input0, const BaseLayer::iotype& input1);
			BaseLayer::iotype operator+(const tensor::Tensor& input0, const tensor::Tensor& input1);
		}
	}
}

//template <typename T, typename ... Args>
//aoba::nn::layer::nnLayer aoba::nn::layer::gen(Args ... args)
//{
//	LayerCore::Layer layer{};
//	layer.mLayerCore = std::make_shared<T>(args...);
//	return layer;
//}
//
//template <typename T, typename ... Args>
//aoba::nn::layer::nnLayer aoba::nn::layer::gen(const char* layerName, Args ... args)
//{
//	LayerCore::Layer layer{};
//	layer.mLayerCore = std::make_shared<T>(args...);
//	layer.mLayerName = layerName;
//	return layer;
//}

