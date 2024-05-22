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
	//�f�[�^���u���Ă���A�h���X
	DataAddress _m_cpu_date_address;
	DataAddress _m_gpu_data_address;

	DataAddress _m_cpu_grad_date_address;
	DataAddress _m_gpu_grad_data_address;

	//���z���K�v���B���[�Ȃǂł͕K�v�Ȃ��B
	bool _m_need_grad;

	//GPU�𗘗p���邩�ۂ�
	bool _m_use_gpu;

	//�e�Ǝq����c�����Ă���
	//forward��backward�̏����ŕK�v�B
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