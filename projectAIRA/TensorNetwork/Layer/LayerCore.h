#pragma once
#include <cuda_runtime.h>
#include <map>
#include <device_launch_parameters.h>
#include "Tensor/Tensor.h"



//コンストラクタで子テンソルにshared_ptr化したthisを登録したくて継承。
//問題が起きたらここを疑う。
class LayerCore : public std::enable_shared_from_this<LayerCore>
{
public:
	friend class TensorCore;
	using iotype = std::vector<Tensor>;

	LayerCore(u32 = 1, u32 = 1);
	LayerCore(u32, u32, u32);
	LayerCore(u32, u32, u32, u32);
	virtual ~LayerCore() {}

	iotype callForward(const iotype&);
	void callBackward();
	void regist_this_to_output_tensor();

	u32 get_input_tensor_num() const { return m_input_tensor_num; }
	u32 get_output_tensor_num() const { return m_output_tensor_num; }


	/// <summary>
	/// 各層が独自に行うforward処理はこの仮想関数に実装する。
	/// </summary>
	/// <param name="input_tensors"></param>
	/// <returns></returns>
	virtual iotype forward(const iotype& input_tensors) = 0;


	class Layer
	{
	public:
		template <typename T, typename ... Args>
		friend Layer gen(Args ... args);
		template <typename T, typename ... Args>
		friend Layer gen(const char* layerName, Args ... args);

		Layer() {}
		Layer(const Layer&);
		Layer(const std::shared_ptr<LayerCore>&, std::string);

		LayerCore::iotype operator()(const LayerCore::iotype& input) const;

		template <typename ... Args>
		LayerCore::iotype operator()(Args ... args) const
		{
			Tensor tensor_tbl[] = { args... };
			const u32 input_tensor_num = (sizeof(tensor_tbl) / sizeof(tensor_tbl[0]));

			LayerCore::iotype input_tensor_as_vector(input_tensor_num);

			for (u32 i = 0, end = input_tensor_num; i < end; i++)
			{
				input_tensor_as_vector[i] = tensor_tbl[i];
			}

			return mLayerCore->callForward(input_tensor_as_vector);
		}
		u32 get_input_tensor_num() const
		{
			return mLayerCore->get_input_tensor_num();
		}

		u32 get_output_tensor_num() const
		{
			return mLayerCore->get_output_tensor_num();
		}

		const std::shared_ptr<LayerCore>& getLayerCore() const
		{
			return mLayerCore;
		}
	private:
		std::shared_ptr<LayerCore> mLayerCore;
		std::string mLayerName;
	};

protected:
	bool unique_implimention_layer = true;
	bool m_init_finish = false;
	bool m_on_cuda = false;

	//std::vector<bool> m_downstream_tensor_backward_finish;
	/// <summary>
	/// 各層が独自に行うforward処理はこの仮想関数に実装する。
	/// </summary>
	/// <param name="input_tensors"></param>
	/// <returns></returns>
	virtual void backward()
	{
		unique_implimention_layer = false;
		std::cout << "no implement" << std::endl;
	}

	std::vector<std::shared_ptr<TensorCore> > m_parameter_tbl;
	std::map<std::string, Layer> m_internal_layer_tbl;
	std::map<std::string, Layer>& mlayer;//上記のエイリアス:そのままだと長いから

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


	void common_check_before_forward(const iotype& input_tensors)
	{
		//まず引き数に与えられた入力テンソル数が層が決めた値に一致しているか確認。
		if (input_tensors.size() != m_input_tensor_num)
		{
			std::cout << "The number of input tensor must be " << m_input_tensor_num
				<< ". \nBut current input num is " << input_tensors.size() << "."
				<< std::endl;
			exit(1);
		}

		//入力テンソル間のGPU利用設定に矛盾がないかチェックする。
		bool on_cuda = input_tensors[0].pTensorCore->_m_on_cuda;
		for (u32 i = 1; i < m_input_tensor_num; i++)
		{
			if (input_tensors[i].pTensorCore->_m_on_cuda != on_cuda)
			{
				std::cout << "Between input tensor's, CPU/GPU setting contradict!" << std::endl;
				exit(1);
			}
		}

		if (m_init_finish && (on_cuda != m_on_cuda))
		{
			std::cout << "Between input and layer, CPU/GPU setting contradict!" << std::endl;
			exit(1);
		}

		//入力されたテンソルをこの層の入力テンソルテーブルに登録する。
		for (u32 i = 0; i < m_input_tensor_num; i++)
		{
			//過去に入力があった場合、i番目の入力スロットに過去の入力テンソルが登録されている。
			//それをここで一度解除する。
			if (std::shared_ptr<TensorCore> p = mInputTensorCoreTbl[i].lock())
			{
				//上流テンソルに依頼して、双方向にリンクを切ってもらう。
				p->disconnect_bidirection();
			}

			auto& tensorcore = input_tensors[i].pTensorCore;

			//過去にどこかの層に入力されていた場合、下流の層情報が登録されている。
			//ここでそれを解除する。
			if (tensorcore->_m_downstream_layer)
			{
				tensorcore->disconnect_bidirection();
			}
			mInputTensorCoreTbl[i] = tensorcore;
			tensorcore->connect(shared_from_this(), i);
		}
	}

	void disconnect_bidirection(s32 location)
	{
		mInputTensorCoreTbl[location].reset();
	}


	const std::shared_ptr<TensorCore>& getTensorCoreFrom(const Tensor& tensor)
	{
		return tensor.pTensorCore;
	}


};

class Accessor2TensorCore
{
public:
	inline static DataType* getAddressOnCpuFrom(Tensor tensor)
	{
		return tensor.pTensorCore->_m_cpu_data_address;
	}
	inline static DataType* getAddressOnCpuFrom(const std::shared_ptr<TensorCore>& tensor_ptr)
	{
		return tensor_ptr->_m_cpu_data_address;
	}
	inline static DataType* getGradAddressOnCpuFrom(const std::shared_ptr<TensorCore>& tensor_ptr)
	{
		return tensor_ptr->_m_cpu_grad_data_address;
	}
	inline static DataType* getAddressOnGpuFrom(Tensor tensor)
	{
		return tensor.pTensorCore->_m_gpu_data_address;
	}
	inline static DataType* getAddressOnGpuFrom(const std::shared_ptr<TensorCore>& tensor_ptr)
	{
		return tensor_ptr->_m_gpu_data_address;
	}
	inline static DataType* getGradAddressOnGpuFrom(const std::shared_ptr<TensorCore>& tensor_ptr)
	{
		return tensor_ptr->_m_gpu_grad_data_address;
	}

	inline static u32 getDataSize(Tensor tensor)
	{
		return tensor.pTensorCore->mDataSize;
	}

	inline static u32 getDataSize(const std::shared_ptr<TensorCore>& tensor_ptr)
	{
		return tensor_ptr->mDataSize;
	}

	inline static bool on_cuda(const Tensor& tensor)
	{
		return tensor.pTensorCore->_m_on_cuda;
	}
	inline static bool get_need_grad(const std::shared_ptr<TensorCore>& tensor_ptr)
	{
		return tensor_ptr->_m_need_grad;
	}
	inline static void set_value(const Tensor& t, u32 index, DataType value)
	{
		t.pTensorCore->_m_cpu_data_address[index] = value;
	}

};


LayerCore::iotype operator+(const LayerCore::iotype& input0, const LayerCore::iotype& input1);
LayerCore::iotype operator+(const Tensor& input0, const Tensor& input1);

using Layer = LayerCore::Layer;

template <typename T, typename ... Args>
Layer gen(Args ... args)
{
	Layer layer{};
	layer.mLayerCore = std::make_shared<T>(args...);
	return layer;
}

template <typename T, typename ... Args>
Layer gen(const char* layerName, Args ... args)
{
	Layer layer{};
	layer.mLayerCore = std::make_shared<T>(args...);
	layer.mLayerName = layerName;
	return layer;
}
