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
		}

		namespace tensor
		{
			class Tensor;
			class TensorCore;
		}
	}
}

class aoba::nn::layer::Layer
{
public:
	class nnLayer;

	class LayerSkeleton;

	class AffineCore;
	class ReLUCore;
	class AddCore;
	class SplitCore;
	class AddAsInnerCore;
	class SequentialCore;

	class CrossEntropyWithSMCore;
};


//コンストラクタで子テンソルにshared_ptr化したthisを登録したくて継承。
//問題が起きたらここを疑う。
class aoba::nn::layer::Layer::LayerSkeleton : public std::enable_shared_from_this<LayerSkeleton>
{
public:

	friend class aoba::nn::tensor::TensorCore;
	friend class aoba::nn::optimizer::Optimizer;
	using iotype = std::vector<tensor::Tensor>;

	LayerSkeleton(u32 = 1, u32 = 1);
	LayerSkeleton(u32, u32, u32);
	LayerSkeleton(u32, u32, u32, u32);
	virtual ~LayerSkeleton() {}

	iotype callForward(const iotype&);
	void callBackward(u32 downstream_index);
	void regist_this_to_output_tensor();

	u32 get_input_tensor_num() const { return m_input_tensor_num; }
	u32 get_output_tensor_num() const { return m_output_tensor_num; }





protected:
	using TensorCore = aoba::nn::tensor::TensorCore;
	using Tensor = aoba::nn::tensor::Tensor;

	bool unique_implimention_layer = true;
	bool m_init_finish = false;
	bool m_on_cuda = false;



	std::vector<std::shared_ptr<TensorCore> > m_parameter_tbl;
	std::map<std::string, std::shared_ptr<Layer::LayerSkeleton> > m_internal_layer_tbl;
	std::map<std::string, std::shared_ptr<Layer::LayerSkeleton> >& mlayer;//上記のエイリアス:そのままだと長いから

	/// <summary>
	/// この層が生成したテンソル
	///（各層はテンソル用のメモリを直接見ているイメージ）
	/// </summary>
	std::vector<std::shared_ptr<TensorCore>> m_child_tensorcore_tbl;

	/// <summary>
	/// 順伝搬でインプットされたテンソル情報を覚えておく用
	/// これがないと逆伝搬を自動で行えなくなる。
	/// </summary>
	std::vector<std::weak_ptr<TensorCore> > mInputTensorCoreTbl;


	//入力と出力のテンソルの数を記録
	const u32 m_input_tensor_num;
	const u32 m_output_tensor_num;

	//void init_childtensor_with(std::shared_ptr<Layer::LayerSkeleton>&& )



	const std::shared_ptr<TensorCore>& getTensorCoreFrom(const Tensor& tensor);


	void genTensor(u32 childNo, std::shared_ptr<TensorCore>&& tensorcore)
	{
		if (childNo >= m_output_tensor_num)
		{
			assert(0);
		}

		m_child_tensorcore_tbl[childNo] = tensorcore;
		m_child_tensorcore_tbl[childNo]->_m_location_in_upstream_layer = childNo;
		m_child_tensorcore_tbl[childNo]->regist_parent_layercore(shared_from_this());
	}

private:

	std::vector<bool> m_downstream_backward_checkTbl;

	void common_check_before_forward(const iotype& input_tensors);
	//{
	//	//まず引き数に与えられた入力テンソル数が層が決めた値に一致しているか確認。
	//	if (input_tensors.size() != m_input_tensor_num)
	//	{
	//		std::cout << "The number of input tensor must be " << m_input_tensor_num
	//			<< ". \nBut current input num is " << input_tensors.size() << "."
	//			<< std::endl;
	//		exit(1);
	//	}
	//
	//	//入力テンソル間のGPU利用設定に矛盾がないかチェックする。
	//	bool on_cuda = input_tensors[0].pTensorCore->_m_on_cuda;
	//	for (u32 i = 1; i < m_input_tensor_num; i++)
	//	{
	//		if (input_tensors[i].pTensorCore->_m_on_cuda != on_cuda)
	//		{
	//			std::cout << "Between input tensor's, CPU/GPU setting contradict!" << std::endl;
	//			exit(1);
	//		}
	//	}
	//
	//	if (m_init_finish && (on_cuda != m_on_cuda))
	//	{
	//		std::cout << "Between input and layer, CPU/GPU setting contradict!" << std::endl;
	//		exit(1);
	//	}
	//
	//	//入力されたテンソルをこの層の入力テンソルテーブルに登録する。
	//	for (u32 i = 0; i < m_input_tensor_num; i++)
	//	{
	//		//過去に入力があった場合、i番目の入力スロットに過去の入力テンソルが登録されている。
	//		//それをここで一度解除する。
	//		if (std::shared_ptr<TensorCore> p = mInputTensorCoreTbl[i].lock())
	//		{
	//			//上流テンソルに依頼して、双方向にリンクを切ってもらう。
	//			p->disconnect_bidirection();
	//		}
	//
	//		auto& tensorcore = input_tensors[i].pTensorCore;
	//
	//		//過去にどこかの層に入力されていた場合、下流の層情報が登録されている。
	//		//ここでそれを解除する。
	//		if (tensorcore->_m_downstream_layer)
	//		{
	//			tensorcore->disconnect_bidirection();
	//		}
	//		mInputTensorCoreTbl[i] = tensorcore;
	//		tensorcore->connect(shared_from_this(), i);
	//	}
	//}
	void disconnect_bidirection(s32 location)
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
		unique_implimention_layer = false;
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


using nnModule = aoba::nn::layer::Layer::LayerSkeleton;


namespace aoba
{
	namespace nn
	{
		namespace layer
		{

			Layer::LayerSkeleton::iotype operator+(const Layer::LayerSkeleton::iotype& input0, const Layer::LayerSkeleton::iotype& input1);
			Layer::LayerSkeleton::iotype operator+(const tensor::Tensor& input0, const tensor::Tensor& input1);
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

