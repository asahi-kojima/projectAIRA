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

	//�t�`���֐�
	void backward() const;


	DataType getData(u32 index) const { return 1; }
	u32 getDataSize() const{ return mDataSize; }
	void setName(const std::string& name) { _m_debug_name = name; }//debug

	void regist_parent_layercore(const std::shared_ptr<LayerCore>& parent_layercore)
	{
		_m_parent_layer = parent_layercore;
	}

private:
	//�f�[�^���u���Ă���A�h���X
	DataAddress _m_cpu_date_address;
	DataAddress _m_gpu_data_address;

	//���z���B�ȉ��̃t���O�������Ă���ꍇ�̂݊m�ۂ���
	DataAddress _m_cpu_grad_date_address;
	DataAddress _m_gpu_grad_data_address;
	//���z���K�v���B���[�Ȃǂł͕K�v�Ȃ��B
	bool _m_need_grad;

	u32 mDataSize;
	

	//GPU�𗘗p���邩�ۂ�
	bool _m_use_gpu;

	//�e�Ǝq����c�����Ă���
	//forward��backward�̏����ŕK�v�B
	std::weak_ptr<LayerCore> _m_parent_layer;
	std::vector<std::shared_ptr<LayerCore> > _m_child_layer_tbl;

	std::string _m_debug_name;//debug


};