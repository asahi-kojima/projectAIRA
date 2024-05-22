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

	void forward() const;
	void backward() const;
	void setName(const std::string& name) { _m_debug_name = name; }//debug
private:
	//データが置いてあるアドレス
	DataAddress _m_cpu_date_address;
	DataAddress _m_gpu_data_address;

	DataAddress _m_cpu_grad_date_address;
	DataAddress _m_gpu_grad_data_address;

	//勾配が必要か。末端などでは必要ない。
	bool _m_need_grad;

	//GPUを利用するか否か
	bool _m_use_gpu;

	//親と子供を把握しておく
	//forwardとbackwardの処理で必要。
	std::shared_ptr<LayerCore> _m_parent_module;
	std::vector<std::shared_ptr<LayerCore> > _m_child_module_tbl;

	std::string _m_debug_name;//debug


	void regist_parent_module(std::shared_ptr<LayerCore> module_address)
	{
		_m_parent_module = module_address;
	}
	void regist_child_module(std::shared_ptr<LayerCore> module_address)
	{
		_m_child_module_tbl.push_back(module_address);
	}
};