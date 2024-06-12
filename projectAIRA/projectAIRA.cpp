#include "root_TensorNetwork.h"
#include <tuple>
#include <cassert>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ostream>
#include "Layer/Add.h"
#include "Layer/AddAsInner.h"
#include "Layer/ReLU.h"
#include "Layer/Sequential.h"
#include "Layer/CrossEntropyWithSM.h"
#include "gpu-manager.h"
#include "Optimizer/SGD.h"

using namespace aoba::nn;
using namespace layer;
using namespace optimizer;
using namespace tensor;

std::tuple<Tensor, Tensor> convert(const Layer::LayerSkeleton::iotype& tensor_vec)
{
	if (tensor_vec.size() != 2)
	{
		assert(0);
	}

	return std::tuple<Tensor, Tensor>(tensor_vec[0], tensor_vec[1]);
}

//0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
void loadMnistFromBin(std::string filePath, std::vector<f32>& data, u32 loadByteSize)
{
	std::cout << "load start [" << filePath << "]";

	std::ifstream fin(filePath, std::ios::in | std::ios::binary);
	if (!fin)
	{
		std::cout << "\nthis program can't open the file : " << filePath << "\n" << std::endl;
		return;
	}
	fin.read(reinterpret_cast<char*>(data.data()), loadByteSize);

	std::cout << "-----> load finish" << std::endl;
}

void mnistNormalizer(std::vector<f32>& v, u32 size)
{
	for (u32 i = 0; i < size; i++)
	{
		const u32 offset = i * 784;

		f32 mu = 0;
		for (u32 j = 0; j < 784; j++)
		{
			mu += v[offset + j] / 784.0f;
		}

		f32 sigma2 = 0.0f;
		for (u32 j = 0; j < 784; j++)
		{
			sigma2 += (v[offset + j] - mu) * (v[offset + j] - mu) / 784.0f;
		}

		f32 sigma = std::sqrtf(sigma2);
		for (u32 j = 0; j < 784; j++)
		{
			v[offset + j] = (v[offset + j] - mu) / sigma;
		}
	}
}

void setupMnistData(std::vector<f32>& trainingData, std::vector<f32>& trainingLabel, std::vector<f32>& testData, std::vector<f32>& testLabel)
{
	constexpr u32 dataSize = 784;
	constexpr u32 trainingDataNum = 60000;
	constexpr u32 testDataNum = 10000;
	trainingData.resize(trainingDataNum * dataSize);
	trainingLabel.resize(trainingDataNum);
	testData.resize(testDataNum * dataSize);
	testLabel.resize(testDataNum);

	loadMnistFromBin("C:\\Users\\asahi\\Downloads\\mnist_data_train.bin", trainingData, sizeof(f32) * trainingData.size());
	loadMnistFromBin("C:\\Users\\asahi\\Downloads\\mnist_label_train.bin", trainingLabel, sizeof(f32) * trainingLabel.size());
	loadMnistFromBin("C:\\Users\\asahi\\Downloads\\mnist_data_test.bin", testData, sizeof(f32) * testData.size());
	loadMnistFromBin("C:\\Users\\asahi\\Downloads\\mnist_label_test.bin", testLabel, sizeof(f32) * testLabel.size());

	mnistNormalizer(trainingData, trainingDataNum);
	mnistNormalizer(testData, testDataNum);
}
//1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111

//000000000000000000000000000000000000000000000000000000000000000000000000000000
//開発用
void init(Tensor& t, DataType value)
{
	for (u32 i = 0; i < t.getDataSize(); i++)
	{
		t(i) = value;
	}
}

void init_linear(Tensor& t, DataType value)
{
	for (u32 i = 0; i < t.getDataSize(); i++)
	{
		t(i) = value * i;
	}
}

void init_normal(Tensor& t)
{
	std::random_device seed_gen;
	std::mt19937 engine(seed_gen());
	std::normal_distribution<float> dist(0.0f, 1.0);
	for (u32 i = 0; i < t.getDataSize(); i++)
	{
		t(i) = dist(engine);
	}
}


void confirm(const Tensor& tensor)
{
	const auto dataSize = tensor.getDataSize();
	//const auto pointer = Accessor2TensorCore::getAddressOnCpuFrom(tensor);

	std::vector<DataType> v(dataSize);
	for (u32 i = 0; i < dataSize; i++)
	{
		v[i] = tensor(i);
	}
}
//111111111111111111111111111111111111111111111111111111111111111111111111111111

class MyLayer : public nnModule
{
public:
	MyLayer()
	{
		seq = aoba::nn::layer::Sequential(ReLU(), ReLU());
	}

	iotype forward(const iotype& input) override
	{
		return seq(input);
	}

	aoba::nn::layer::Layer::nnLayer seq;
};



int main()
{
	bool gpu_is_available = gpu_manager::gpu_is_available();

	constexpr u32 dataSize = 784;
	constexpr u32 trainingDataNum = 60000;
	constexpr u32 testDataNum = 10000;
	std::vector<f32> trainingData, trainingLabel, testData, testLabel;
	setupMnistData(trainingData, trainingLabel, testData, testLabel);


	constexpr u32 batch_size = 100;
	constexpr u32 batched_data_num = trainingDataNum / batch_size;
	std::vector<Tensor> input_tensor_tbl(batched_data_num);
	for (auto& v : input_tensor_tbl)
	{
		v = Tensor(batch_size, dataSize);
	}
	std::vector<Tensor> correct_tensor_tbl(batched_data_num);
	for (auto& v : correct_tensor_tbl)
	{
		v = Tensor(batch_size, 1);
	}

	for (u32 Bn = 0; Bn < batched_data_num; Bn++)
	{
		auto& tensor = input_tensor_tbl[Bn];
		for (u32 N = 0; N < batch_size; N++)
		{
			for (u32 i = 0; i < dataSize; i++)
			{
				const u32 batched_index = Bn * (dataSize * batch_size) + (N * dataSize + i);
				const u32 index = N * dataSize + i;
				tensor(index) = trainingData[batched_index];
			}
		}
	}

	for (u32 Bn = 0; Bn < batched_data_num; Bn++)
	{
		auto& tensor = correct_tensor_tbl[Bn];
		for (u32 N = 0; N < batch_size; N++)
		{
			const u32 batched_index = Bn * batch_size + N;
			tensor(N) = trainingLabel[batched_index];
		}
	}

	int sss = 1 + 1;
	//////テスト１
	//{
	//	std::cout << "===============================" << std::endl;
	//	std::cout << "Test1" << std::endl;
	//	std::cout << "===============================" << std::endl;
	//	for (u32 i = 0; i < 100; i++)
	//	{
	//		std::cout << "-------------------------------" << std::endl;
	//		std::cout << "Loop : " << i << std::endl;
	//		std::cout << "-------------------------------" << std::endl;
	//		const u32 N = 10;
	//		auto add0 = AddAsInner();
	//		auto add1 = Add();
	//		auto relu = ReLU();
	//		auto seq = Sequential(ReLU(), ReLU(), ReLU(), ReLU());

	//		Tensor t0(N, 3, 28, 28); init(t0, 1); t0.to_cuda(gpu_is_available);
	//		Tensor s0(N, 3, 28, 28, false); init_linear(s0, 2); s0.to_cuda(gpu_is_available);
	//		for (u32 i = 0; i < 2; i++)
	//		{
	//			auto t1 = add0(t0, s0);
	//			Tensor s0(N, 3, 28, 28); init(s0, 1); s0.to_cuda(gpu_is_available);
	//			auto t2 = add1(t1[0], s0);



	//			auto t3 = relu(t2[0]);
	//			auto t4 = seq(t3);
	//			t4[0].backward();
	//		}
	//	}
	//}

	////テスト２
	//{
	//	std::cout << "===============================" << std::endl;
	//	std::cout << "Test2" << std::endl;
	//	std::cout << "===============================" << std::endl;
	//	{
	//		Tensor t0(10, 3, 28, 28); init(t0, 1); t0.to_cuda(gpu_is_available);
	//		Tensor t1(10, 3, 28, 28); init(t1, 2); t1.to_cuda(gpu_is_available);
	//		auto t2 = t0 + t1;
	//		t2[0].backward();
	//	}
	//}


	////テスト３
	//{
	//	std::cout << "===============================" << std::endl;
	//	std::cout << "Test3" << std::endl;
	//	std::cout << "===============================" << std::endl;
	//	{
	//		Tensor t0(10, 3, 28, 28); init(t0, 1);
	//		t0.to_cuda(gpu_is_available);
	//		auto seq = Sequential(ReLU(), ReLU(), ReLU(), ReLU());
	//		auto t1 = seq(t0);
	//		t1[0].backward();
	//	}
	//}

	////テスト4
	//{
	//	std::cout << "===============================" << std::endl;
	//	std::cout << "Test4" << std::endl;
	//	std::cout << "===============================" << std::endl;
	//	{
	//		Tensor t0(10, 3, 28, 28); init(t0, 1);
	//		t0.to_cuda(gpu_is_available);
	//		auto mylayer = gen<MyLayer>();
	//		auto t1 = mylayer(t0);
	//	}
	//}


	////テスト5
	//{
	//	std::cout << "===============================" << std::endl;
	//	std::cout << "Test5" << std::endl;
	//	std::cout << "===============================" << std::endl;
	//	{
	//		Tensor t0(10, 3, 28, 28); init_normal(t0);
	//		t0.to_cuda(gpu_is_available);
	//		auto relu = ReLU();
	//		auto t1 = relu(t0);
	//		t1[0].synchronize_from_GPU_to_CPU();
	//		confirm(t1[0]);
	//	}
	//}

	////テスト6
	//{
	//	std::cout << "===============================" << std::endl;
	//	std::cout << "Test6" << std::endl;
	//	std::cout << "===============================" << std::endl;
	//	{
	//		Tensor t0(10, 3, 28, 28); init_normal(t0);
	//		t0.to_cuda(gpu_is_available);
	//		auto split = Split();
	//		auto add = Add();
	//		auto t1 = split(t0);
	//		auto t2 = add(t1);
	//		t2[0].backward();
	//		//t2[0].synchronize_from_GPU_to_CPU();
	//		//confirm(t1[0]);
	//	}
	//}

	//////テスト7
	//{
	//	std::cout << "===============================" << std::endl;
	//	std::cout << "Test7" << std::endl;
	//	std::cout << "===============================" << std::endl;
	//	for (u32 i = 0; i < 100; i++)
	//	{
	//		std::cout << "-------------------------------" << std::endl;
	//		std::cout << "Loop : " << i << std::endl;
	//		std::cout << "-------------------------------" << std::endl;
	//		const u32 N = 1;
	//		auto add0 = AddAsInner();
	//		auto add1 = AddAsInner();
	//		auto relu = ReLU();
	//		auto seq = Sequential(ReLU(), ReLU(), ReLU(), ReLU());

	//		Tensor t0(N, 1); init(t0, 1);// t0.to_cuda(gpu_is_available);
	//		Tensor t1(N, 1); init(t1, 1);// t1.to_cuda(gpu_is_available);
	//		auto t = t0 + t1;
	//		t[0].backward();
	//	}
	//}

	////テスト8
	{
		//auto seq = Sequential(Affine(300), ReLU(), Affine(100), ReLU(), Affine(10));
		auto seq = Sequential(Affine(100), ReLU(), Affine(10));
		auto lossFunc = CrossEntropyWithSM();
		auto optim = Optimizer::SGD(0.1f * 10);
		optim(seq);
		std::cout << "===============================" << std::endl;
		std::cout << "Test8" << std::endl;
		std::cout << "===============================" << std::endl;

		for (u32 i = 0; i < 1000; i++)
		{
			std::cout << "-------------------------------" << std::endl;
			std::cout << "Loop : " << i << std::endl;
			std::cout << "-------------------------------" << std::endl;
			for (u32 Bn = 0; Bn < batched_data_num; Bn++)
			{
				Tensor& training_tensor = input_tensor_tbl[Bn];  //training_tensor.to_cuda("");
				Tensor& correct_tensor = correct_tensor_tbl[Bn]; //correct_tensor.to_cuda("");
				auto t = seq(training_tensor);
				auto loss = lossFunc(t[0], correct_tensor);
				std::cout << Tensor::getLoss(loss[0]) << std::endl;
				loss[0].backward();
				optim.optimize();
			}
		}

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

