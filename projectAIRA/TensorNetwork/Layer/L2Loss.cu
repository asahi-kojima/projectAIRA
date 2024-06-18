//#include "L2Loss.h"
//
//
//Layer L2Loss()
//{
//	Layer l2loss = gen<L2LossCore>("L2Loss");
//	return l2loss;
//}
//
//
//L2LossCore::L2LossCore()
//	: LayerCore(2, 1, 1)
//{
//}
//
//
//
//LayerCore::iotype L2LossCore::forward(const LayerCore::iotype& input_tensors)
//{
//	const auto& input_tensorcore0 = *getTensorCoreFrom(input_tensors[0]);
//	const auto& input_tensorcore1 = *getTensorCoreFrom(input_tensors[1]);
//
//
//	const auto batchSize_lhs = input_tensorcore0.mBatchSize;
//	const auto dataSize_lhs  = input_tensorcore0.mCHW;
//	const auto batchSize_rhs = input_tensorcore1.mBatchSize;
//	const auto dataSize_rhs  = input_tensorcore1.mCHW;
//
//	//形状は任意でいいが、要素数が一致していないと演算が出来ない。
//	if (dataSize_lhs != dataSize_rhs)
//	{
//		std::cout << "Input tensor size between LHS & RHS is not equal@AddCore::forward" << std::endl;
//		exit(1);
//	}
//	if (batchSize_lhs != batchSize_rhs)
//	{
//
//	}
//
//	//初期化が終わっていない場合、ここでインプットされたテンソルに合わせ動的に確保/初期化を行う。
//	if (!m_init_finish)
//	{
//		auto& child_tensorcore = m_child_tensorcore_tbl[0];
//		child_tensorcore = std::make_shared<TensorCore>(input_tensorcore0, true);
//		child_tensorcore->regist_parent_layercore(shared_from_this());
//
//		if (input_tensorcore0.m_on_cuda)
//		{
//			m_on_cuda = true;
//			child_tensorcore->to_cuda("");
//		}
//		m_init_finish = true;
//	}
//
//	const auto& child_tensorcore = *m_child_tensorcore_tbl[0];
//
//	auto dataSize = child_tensorcore.mDataSize;
//	if (dataSize_lhs != dataSize_rhs || dataSize != dataSize_lhs)
//	{
//		std::cout << "Input tensor size between LHS & RHS & Output is not match." << std::endl;
//		exit(1);
//	}
//
//
//
//	std::cout << "Add forward " << (m_on_cuda ? "On GPU" : "on CPU") << std::endl;
//	if (m_on_cuda)
//	{
//		auto input_address0 = input_tensorcore0._m_gpu_data_address;
//		auto input_address1 = input_tensorcore1._m_gpu_data_address;
//		auto output_address = child_tensorcore._m_gpu_data_address;
//
//		dim3 block(256);
//		dim3 grid((dataSize + block.x - 1) / block.x);
//		relu_forward_gpu_impl << <grid, block >> > (input_address0, input_address1, output_address, dataSize);
//		CUDA_SYNCHRONIZE_DEBUG;
//	}
//	else
//	{
//		l2loss_forward_cpu_impl(input_tensors);
//	}
//
//	return iotype{ Tensor(m_child_tensorcore_tbl[0]) };
//}
//
//
//void L2LossCore::l2loss_forward_cpu_impl(const LayerCore::iotype&)
//{
//
//}