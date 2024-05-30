#include "root_TensorNetwork.h"
#include <tuple>
#include <cassert>
#include "Layer/Add.h"
#include "Layer/AddAsInner.h"
#include "Layer/ReLU.h"
#include "Layer/Sequential.h"
std::tuple<Tensor, Tensor> convert(const LayerCore::iotype& tensor_vec)
{
	if (tensor_vec.size() != 2)
	{
		assert(0);
	}

	return std::tuple<Tensor, Tensor>(tensor_vec[0], tensor_vec[1]);
}

//000000000000000000000000000000000000000000000000000000000000000000000000000000
//開発用
void init(Tensor& t, DataType value)
{
	for (u32 i = 0; i < t.getTensorDataSize(); i++)
	{
		t[i] = value;
	}
}

void init_linear(Tensor& t, DataType value)
{
	for (u32 i = 0; i < t.getTensorDataSize(); i++)
	{
		t[i] = value * i;
	}
}

void confirm(const Tensor& tensor)
{
	const auto dataSize = Accessor2TensorCore::getDataSize(tensor);
	const auto pointer = Accessor2TensorCore::getAddressOnCpuFrom(tensor);

	std::vector<TensorCore::DataType> v(dataSize);
	for (u32 i = 0; i < dataSize; i++)
	{
		v[i] = pointer[i];
	}
}
//111111111111111111111111111111111111111111111111111111111111111111111111111111


int main()
{
	//テスト１
	std::cout << "===============================" << std::endl;
	std::cout << "Test1" << std::endl;
	std::cout << "===============================" << std::endl;
	for (u32 i = 0; i < 100; i++)
	{
		std::cout << "-------------------------------" << std::endl;
		std::cout << "Loop : " << i << std::endl;
		std::cout << "-------------------------------" << std::endl;
		const u32 N = 1000;
		auto add0 = AddAsInner(); add0.mLayerName = "add0";
		auto add1 = Add(); add1.mLayerName = "add1";
		auto relu = ReLU(); relu.mLayerName = "relu";
		auto seq = Sequential(ReLU(), ReLU(), ReLU(), ReLU());

		Tensor t0(N, 3, 28, 28); init(t0, 1); t0.to_cuda();
		Tensor s0(false, N, 3, 28, 28); init_linear(s0, 2); s0.to_cuda();
		for (u32 i = 0; i < 2; i++)
		{
			auto t1 = add0(t0, s0);
			Tensor s0(N, 3, 28, 28); init(s0, 1); confirm(t1[0]); s0.to_cuda();
			auto t2 = add1(t1[0], s0);



			auto t3 = relu(t2[0]);
			auto t4 = seq(t3);
			t4[0].backward();
		}
	}


	//テスト２
	std::cout << "===============================" << std::endl;
	std::cout << "Test2" << std::endl;
	std::cout << "===============================" << std::endl;
	{
		Tensor t0(10, 3, 28, 28); init(t0, 1); t0.to_cuda();
		Tensor t1(10, 3, 28, 28); init(t1, 2); t1.to_cuda();
		auto t2 = t0 + t1;
		t2[0].backward();
	}


	//テスト３
	std::cout << "===============================" << std::endl;
	std::cout << "Test3" << std::endl;
	std::cout << "===============================" << std::endl;
	{
		Tensor t0(10, 3, 28, 28); init(t0, 1);
		t0.to_cuda();
		auto seq = Sequential(ReLU(), ReLU(), ReLU(), ReLU());
		auto t1 = seq(t0);
		t1[0].backward();
	}

	std::cout << "free check" << std::endl;
}


//例えばこんなことをやりたい。
//class Affine : public LayerCore
//{
//public:
//	Affine() : LayerCore(param = 2)
//	{
//		mWeight = mParameterTbl[0];
//		mBias = mParameterTbl[1];
//
//		mInnerLayerCoreTbl.
//	}
//
//	Otype forward(auto input)
//	{
//
//	}
//
//
//private:
//	std::shared_ptr<TensorCore> mWeight;
//	std::shared_ptr<TensorCore> mBias;
//};
//
//class DNet : public LayerCore
//{
//public:
//	DNet() : LayerCore(param = 2)
//	{
//		mMainLayer = Sequential(Affine(), ReLU(), Affine(), ReLU());
//	}
//
//	iotype forward(const iotype& input) override
//	{
//		auto output = mMainLayer(input);
//		return output;
//	}
//
//private:
//	std::shared_ptr<LayerCore> mMainLayer;
//};
//
//class G : public LayerCore {};
//
//int exMain()
//{
//	auto d = DNet();
//	auto g = GNet();
//	auto lossNet = L1Loss();
//	auto optimizer = Adam();
//	for (u32 i = 0; i < 1000; i++)
//	{
//		auto z = Tensor{};
//		auto fake_img = g(z);
//		auto out = d(fake_img);
//		auto loss = lossNet(out, torch.ones());
//
//		d.zero_grad();
//		g.zero_grad();
//
//		loss.backward();
//		optimizer(d, g);
//
//
//	}
//}

