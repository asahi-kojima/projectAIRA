#pragma once
#include <memory>
#include <vector>
#include <iostream>
#include "typeinfo.h"

class LayerCore;

class TensorCore
{
public:
	friend class LayerCore;

	using DataType = f32;
	using DataAddress = DataType*;

	//逆伝搬関数
	void backward() const;


	DataType getData(u32 index) const { return 1; }
	u32 getDataSize() const{ return mDataSize; }
	void setName(const std::string& name) { _m_debug_name = name; }//debug

	void regist_parent_layercore(const std::shared_ptr<LayerCore>& parent_layercore)
	{
		_m_parent_layer = parent_layercore;
	}

private:
	//データが置いてあるアドレス
	DataAddress _m_cpu_date_address;
	DataAddress _m_gpu_data_address;

	//勾配情報。以下のフラグが立っている場合のみ確保する
	DataAddress _m_cpu_grad_date_address;
	DataAddress _m_gpu_grad_data_address;
	//勾配が必要か。末端などでは必要ない。
	bool _m_need_grad;

	u32 mDataSize;
	

	//GPUを利用するか否か
	bool _m_use_gpu;

	//親と子供を把握しておく
	//forwardとbackwardの処理で必要。
	std::weak_ptr<LayerCore> _m_parent_layer;
	std::vector<std::shared_ptr<LayerCore> > _m_child_layer_tbl;

	std::string _m_debug_name;//debug


};