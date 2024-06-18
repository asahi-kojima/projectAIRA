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

/*
Layer.h
TensorCore.h
必ずこの順番でインクルードすること。
*/

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

			class Layer;

			class LayerBase;

			class AffineCore;
			class ReLUCore;
			class AddCore;
			class SplitCore;
			class AddAsInnerCore;
			class SequentialCore;

			class CrossEntropyWithSMCore;
		}

		namespace tensor
		{
			class Tensor;
			class TensorCore;
		}
	}
}



//コンストラクタで子テンソルにshared_ptr化したthisを登録したくて継承。
//問題が起きたらここを疑う。
class aoba::nn::layer::LayerBase : public std::enable_shared_from_this<LayerBase>
{
public:

	friend class aoba::nn::tensor::TensorCore;
	friend class aoba::nn::optimizer::Optimizer;
	using iotype = std::vector<tensor::Tensor>;

	LayerBase(u32 = 1, u32 = 1);
	LayerBase(u32, u32, u32);
	LayerBase(u32, u32, u32, u32);
	virtual ~LayerBase() {}

	iotype callForward(const iotype&);
	void callBackward(u32 downstream_index);
	//void regist_this_to_output_tensor();

	u32 get_input_tensor_num() const { return m_input_tensor_num; }
	u32 get_output_tensor_num() const { return m_output_tensor_num; }





protected:
	using TensorCore = aoba::nn::tensor::TensorCore;
	using Tensor = aoba::nn::tensor::Tensor;

	bool having_unique_implimention = true;
	bool m_init_finish = false;
	bool m_on_cuda = false;



	std::vector<std::shared_ptr<TensorCore> > mTrainableParameterTbl;
	std::map<std::string, std::shared_ptr<LayerBase> > m_internal_layer_tbl;
	std::map<std::string, std::shared_ptr<LayerBase> >& mlayer;//上記のエイリアス:そのままだと長いから

	/// <summary>
	/// この層が生成したテンソル
	///（各層はテンソル用のメモリを直接見ているイメージ）
	/// </summary>
	std::vector<std::shared_ptr<TensorCore>> m_output_tensorcore_tbl;

	/// <summary>
	/// 順伝搬でインプットされたテンソル情報を覚えておく用
	/// これがないと逆伝搬を自動で行えなくなる。
	/// </summary>
	std::vector<std::weak_ptr<TensorCore> > mInputTensorCoreTbl;


	//入力と出力のテンソルの数を記録
	const u32 m_input_tensor_num;
	const u32 m_output_tensor_num;
	const u32 m_trainable_parameter_num;

	//void init_childtensor_with(std::shared_ptr<LayerSkeleton>&& )



	const std::shared_ptr<TensorCore>& getTensorCoreFrom(const Tensor& tensor);

	void initialize();
	void genDownStreamTensor(u32 childNo);


	//TensorCoreにアクセスするための関数群

private:

	std::vector<bool> m_downstream_backward_checkTbl;

	void disconnect_upstream_tensorcore(s32 location)
	{
		mInputTensorCoreTbl[location].reset();
	}

	/// <summary>
	/// 各層が独自に行うforward処理はこの仮想関数に実装する。
	/// </summary>
	/// <param name="input_tensors"></param>
	/// <returns></returns>
	virtual iotype forward(const iotype& input_tensors) = 0;

	/// <summary>
	/// 各層が独自に行うforward処理はこの仮想関数に実装する。
	/// </summary>
	/// <param name="input_tensors"></param>
	/// <returns></returns>
	virtual void backward()
	{
		having_unique_implimention = false;
	}


	struct inputDatainfo
	{
		u32 dim;
		u32 batch_size;
		u32 channel;
		u32 height;
		u32 width;
	};
};


using nnModule = aoba::nn::layer::LayerBase;


namespace aoba
{
	namespace nn
	{
		namespace layer
		{

			LayerBase::iotype operator+(const LayerBase::iotype& input0, const LayerBase::iotype& input1);
			LayerBase::iotype operator+(const tensor::Tensor& input0, const tensor::Tensor& input1);
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

