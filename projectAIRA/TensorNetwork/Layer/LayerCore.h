#pragma once
#include "Tensor/Tensor.h"


//コンストラクタで子テンソルにshared_ptr化したthisを登録したくて継承。
//問題が起きたらここを疑う。
class LayerCore : public std::enable_shared_from_this<LayerCore>
{
public:
	friend class TensorCore;
	using iotype = std::vector<Tensor>;

	LayerCore(u32=1, u32=1);
	LayerCore(u32, u32, u32);
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

	virtual iotype operator()(const iotype& input)
	{
		return forward(input);
	}

protected:
	bool m_init_finish = false;
	bool m_use_gpu = false;
	/// <summary>
	/// 各層が独自に行うforward処理はこの仮想関数に実装する。
	/// </summary>
	/// <param name="input_tensors"></param>
	/// <returns></returns>
	virtual void backward()
	{
		std::cout << "no implement" << std::endl;
	}

	std::vector<std::shared_ptr<TensorCore> > m_parameter_tbl;

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

private:
	void disconnect_bidirection(s32 location)
	{
		mInputTensorCoreTbl[location].reset();
	}
};

class Accessor2TensorCore
{
public:
	inline static TensorCore::DataType* getAddressOnCpuFrom(Tensor tensor)
	{
		return tensor.pTensorCore->_m_cpu_data_address;
	}
	inline static TensorCore::DataType* getAddressOnCpuFrom(const std::shared_ptr<TensorCore>& tensor_ptr)
	{
		return tensor_ptr->_m_cpu_data_address;
	}

	inline static TensorCore::DataType* getAddressOnGpuFrom(Tensor tensor)
	{
		return tensor.pTensorCore->_m_gpu_data_address;
	}
	inline static TensorCore::DataType* getAddressOnGpuFrom(const std::shared_ptr<TensorCore>& tensor_ptr)
	{
		return tensor_ptr->_m_gpu_data_address;
	}

	inline static u32 getDataSize(Tensor tensor)
	{
		return tensor.pTensorCore->mDataSize;
	}

	inline static u32 getDataSize(const std::shared_ptr<TensorCore>& tensor_ptr)
	{
		return tensor_ptr->mDataSize;
	}

	inline static std::vector<u32> getTensorShape(const Tensor& tensor)
	{
		return  tensor.getShape();
	}
};