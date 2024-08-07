﻿#include "root_TensorNetwork.h"
#include <tuple>
#include <cassert>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ostream>

#include "gpu-manager.h"
#include "Optimizer/SGD.h"
#include "helper.h"

#include "Layer/MaxPooling.h"

using namespace aoba::nn;
using namespace layer;
using namespace optimizer;
using namespace tensor;

std::tuple<Tensor, Tensor> convert(const BaseLayer::iotype& tensor_vec)
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

void init_grad(Tensor& t, DataType value)
{
	for (u32 i = 0; i < t.getDataSize(); i++)
	{
		t.d(i) = value;
	}
}


void init_linear(Tensor& t, DataType value)
{
	for (u32 i = 0; i < t.getDataSize(); i++)
	{
		t(i) = value * i;
	}
}

void init_offset_linear(Tensor& t, DataType value, s32 offset)
{
	for (s32 i = 0; i < t.getDataSize(); i++)
	{
		t(i) = value * (i - offset);
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

void check_Affine()
{
	std::cout << "=======================" << std::endl;
	std::cout << "Affine Debug" << std::endl;
	std::cout << "=======================" << std::endl;
	auto affine4GPU = Affine(100);
	auto affine4CPU = Affine(100);

	Tensor testTensor4GPU = Tensor(10, 3, 28, 28, true);
	init(testTensor4GPU, 1);
	testTensor4GPU.to_cuda(true);
	Tensor testTensor4CPU = Tensor(10, 3, 28, 28, true);
	init(testTensor4CPU, 1);

	auto outGPU = affine4GPU(testTensor4GPU);
	auto outCPU = affine4CPU(testTensor4CPU);
	outGPU[0].synchronize_from_GPU_to_CPU();

	for (u32 i = 0; i < outCPU[0].getDataSize(); i++)
	{
		auto cpuValue = outCPU[0](i);
		auto gpuValue = outGPU[0](i);
		auto error = abs(cpuValue - gpuValue) / abs(cpuValue);

		if (error * 100 > 5)
		{
			std::cout << error << std::endl;
			break;
		}
	}

	for (u32 i = 0; i < outCPU[0].getDataSize(); i++)
	{
		outCPU[0].d(i) = 1;
		outGPU[0].d(i) = 1;
	}
	outGPU[0].synchronize_from_CPU_to_GPU();

	outCPU[0].backward();
	outGPU[0].backward();

	testTensor4GPU.synchronize_from_GPU_to_CPU();
	for (u32 i = 0; i < testTensor4GPU.getDataSize(); i++)
	{
		auto cpuValue = testTensor4CPU.d(i);
		auto gpuValue = testTensor4GPU.d(i);
		auto error = abs(cpuValue - gpuValue) / abs(cpuValue);

		if (error * 100 > 5)
		{
			std::cout << error << std::endl;
			break;
		}
	}
}

void check_Conv()
{
	std::cout << "=======================" << std::endl;
	std::cout << "Convolution Debug" << std::endl;
	std::cout << "=======================" << std::endl;
	auto conv4GPU = Convolution(3, 4, 2, 2);
	auto conv4CPU = Convolution(3, 4, 2, 2);

	Tensor testTensor4GPU = Tensor(10, 3, 28, 28, true);
	init_linear(testTensor4GPU, 1);
	testTensor4GPU.to_cuda(true);
	Tensor testTensor4CPU = Tensor(10, 3, 28, 28, true);
	init_linear(testTensor4CPU, 1);

	auto outGPU = conv4GPU(testTensor4GPU);
	auto outCPU = conv4CPU(testTensor4CPU);
	outGPU[0].synchronize_from_GPU_to_CPU();

	for (u32 i = 0; i < outCPU[0].getDataSize(); i++)
	{
		auto cpuValue = outCPU[0](i);
		auto gpuValue = outGPU[0](i);
		auto error = abs(cpuValue - gpuValue) / abs(cpuValue);

		if (error * 100 > 5)
		{
			std::cout << error << std::endl;
			break;
		}
	}

	for (u32 i = 0; i < outCPU[0].getDataSize(); i++)
	{
		outCPU[0].d(i) = i;
		outGPU[0].d(i) = i;
	}
	outGPU[0].synchronize_from_CPU_to_GPU();

	outCPU[0].backward();
	outGPU[0].backward();

	testTensor4GPU.synchronize_from_GPU_to_CPU();
	for (u32 i = 0; i < testTensor4GPU.getDataSize(); i++)
	{
		auto cpuValue = testTensor4CPU.d(i);
		auto gpuValue = testTensor4GPU.d(i);
		auto error = abs(cpuValue - gpuValue) / abs(cpuValue);

		if (error * 100 > 5)
		{
			std::cout << error << std::endl;
			break;
		}
	}
}

void check_TransposeConv()
{
	std::cout << "=======================" << std::endl;
	std::cout << "TransposeConv Debug" << std::endl;
	std::cout << "=======================" << std::endl;
	auto transposeConv4GPU = TransposeConv(3, 4, 2, 2);
	auto transposeConv4CPU = TransposeConv(3, 4, 2, 2);

	Tensor testTensor4GPU = Tensor(10, 3, 28, 28, true);
	init_linear(testTensor4GPU, 1);
	testTensor4GPU.to_cuda(true);
	Tensor testTensor4CPU = Tensor(10, 3, 28, 28, true);
	init_linear(testTensor4CPU, 1);

	auto outGPU = transposeConv4GPU(testTensor4GPU);
	auto outCPU = transposeConv4CPU(testTensor4CPU);
	outGPU[0].synchronize_from_GPU_to_CPU();

	for (u32 i = 0; i < outCPU[0].getDataSize(); i++)
	{
		auto cpuValue = outCPU[0](i);
		auto gpuValue = outGPU[0](i);
		auto error = abs(cpuValue - gpuValue) / abs(cpuValue);

		if (error * 100 > 5)
		{
			std::cout << error << std::endl;
			break;
		}
	}

	for (u32 i = 0; i < outCPU[0].getDataSize(); i++)
	{
		outCPU[0].d(i) = i;
		outGPU[0].d(i) = i;
	}
	outGPU[0].synchronize_from_CPU_to_GPU();

	outCPU[0].backward();
	outGPU[0].backward();

	testTensor4GPU.synchronize_from_GPU_to_CPU();
	for (u32 i = 0; i < testTensor4GPU.getDataSize(); i++)
	{
		auto cpuValue = testTensor4CPU.d(i);
		auto gpuValue = testTensor4GPU.d(i);
		auto error = abs(cpuValue - gpuValue) / abs(cpuValue);

		if (error * 100 > 5)
		{
			std::cout << error << std::endl;
			break;
		}
	}
}

void check_MaxPooling()
{
	std::cout << "=======================" << std::endl;
	std::cout << "MaxPooling Debug" << std::endl;
	std::cout << "=======================" << std::endl;
	auto maxpooling4GPU = MaxPooling(4, 2, 2);
	auto maxpooling4CPU = MaxPooling(4, 2, 2);

	Tensor testTensor4GPU = Tensor(10, 3, 28, 28, true);
	init_offset_linear(testTensor4GPU, 1, 5 * 3 * 28 * 28);
	testTensor4GPU.to_cuda(true);
	Tensor testTensor4CPU = Tensor(10, 3, 28, 28, true);
	init_offset_linear(testTensor4CPU, 1, 5 * 3 * 28 * 28);

	auto outGPU = maxpooling4GPU(testTensor4GPU);
	auto outCPU = maxpooling4CPU(testTensor4CPU);
	outGPU[0].synchronize_from_GPU_to_CPU();

	for (u32 i = 0; i < outCPU[0].getDataSize(); i++)
	{
		auto cpuValue = outCPU[0](i);
		auto gpuValue = outGPU[0](i);
		auto error = abs(cpuValue - gpuValue) / abs(cpuValue);

		if (error * 100 > 5)
		{
			std::cout << error << std::endl;
			break;
		}
	}

	for (u32 i = 0; i < outCPU[0].getDataSize(); i++)
	{
		outCPU[0].d(i) = i;
		outGPU[0].d(i) = i;
	}
	outGPU[0].synchronize_from_CPU_to_GPU();

	outCPU[0].backward();
	outGPU[0].backward();

	testTensor4GPU.synchronize_from_GPU_to_CPU();
	for (u32 i = 0; i < testTensor4GPU.getDataSize(); i++)
	{
		auto cpuValue = testTensor4CPU.d(i);
		auto gpuValue = testTensor4GPU.d(i);
		auto error = abs(cpuValue - gpuValue) / abs(cpuValue);

		if (error * 100 > 5)
		{
			std::cout << error << std::endl;
			break;
		}
	}
}

void check_ReLU()
{
	std::cout << "=======================" << std::endl;
	std::cout << "ReLU Debug" << std::endl;
	std::cout << "=======================" << std::endl;
	auto relu4GPU = ReLU();
	auto relu4CPU = ReLU();

	Tensor testTensor4GPU = Tensor(10, 3, 28, 28, true);
	init(testTensor4GPU, 1);
	testTensor4GPU.to_cuda(true);
	Tensor testTensor4CPU = Tensor(10, 3, 28, 28, true);
	init(testTensor4CPU, 1);

	auto outGPU = relu4GPU(testTensor4GPU);
	auto outCPU = relu4CPU(testTensor4CPU);
	outGPU[0].synchronize_from_GPU_to_CPU();

	for (u32 i = 0; i < outCPU[0].getDataSize(); i++)
	{
		auto cpuValue = outCPU[0](i);
		auto gpuValue = outGPU[0](i);
		auto error = abs(cpuValue - gpuValue) / abs(cpuValue);

		if (error * 100 > 5)
		{
			std::cout << error << std::endl;
			break;
		}

	}

	for (u32 i = 0; i < outCPU[0].getDataSize(); i++)
	{
		outCPU[0].d(i) = 1;
		outGPU[0].d(i) = 1;
	}
	outGPU[0].synchronize_from_CPU_to_GPU();

	outCPU[0].backward();
	outGPU[0].backward();

	testTensor4GPU.synchronize_from_GPU_to_CPU();
	for (u32 i = 0; i < testTensor4GPU.getDataSize(); i++)
	{
		auto cpuValue = testTensor4CPU.d(i);
		auto gpuValue = testTensor4GPU.d(i);
		auto error = abs(cpuValue - gpuValue) / abs(cpuValue);

		if (error * 100 > 5)
		{
			std::cout << error << std::endl;
			break;
		}
	}
}

void check_Tanh()
{
	std::cout << "=======================" << std::endl;
	std::cout << "Tanh Debug" << std::endl;
	std::cout << "=======================" << std::endl;
	auto tanh4GPU = Tanh();
	auto tanh4CPU = Tanh();

	Tensor testTensor4GPU = Tensor(10, 3, 28, 28, true);
	init(testTensor4GPU, 1);
	testTensor4GPU.to_cuda(true);
	Tensor testTensor4CPU = Tensor(10, 3, 28, 28, true);
	init(testTensor4CPU, 1);

	auto outGPU = tanh4GPU(testTensor4GPU);
	auto outCPU = tanh4CPU(testTensor4CPU);
	outGPU[0].synchronize_from_GPU_to_CPU();

	for (u32 i = 0; i < outCPU[0].getDataSize(); i++)
	{
		auto cpuValue = outCPU[0](i);
		auto gpuValue = outGPU[0](i);
		auto error = abs(cpuValue - gpuValue) / abs(cpuValue);

		if (error * 100 > 5)
		{
			std::cout << error << std::endl;
			break;
		}

	}

	for (u32 i = 0; i < outCPU[0].getDataSize(); i++)
	{
		outCPU[0].d(i) = 1;
		outGPU[0].d(i) = 1;
	}
	outGPU[0].synchronize_from_CPU_to_GPU();

	outCPU[0].backward();
	outGPU[0].backward();

	testTensor4GPU.synchronize_from_GPU_to_CPU();
	for (u32 i = 0; i < testTensor4GPU.getDataSize(); i++)
	{
		auto cpuValue = testTensor4CPU.d(i);
		auto gpuValue = testTensor4GPU.d(i);
		auto error = abs(cpuValue - gpuValue) / abs(cpuValue);

		if (error * 100 > 5)
		{
			std::cout << error << std::endl;
			break;
		}
	}
}

void check_BatchNorm()
{
	std::cout << "=======================" << std::endl;
	std::cout << "BatchNorm Debug" << std::endl;
	std::cout << "=======================" << std::endl;
	auto BatchNorm4GPU = BatchNorm();
	auto BatchNorm4CPU = BatchNorm();

	Tensor testTensor4GPU = Tensor(10, 3, 28, 28, true);
	init(testTensor4GPU, 1);
	testTensor4GPU.to_cuda(true);
	Tensor testTensor4CPU = Tensor(10, 3, 28, 28, true);
	init(testTensor4CPU, 1);

	auto outGPU = BatchNorm4GPU(testTensor4GPU);
	auto outCPU = BatchNorm4CPU(testTensor4CPU);
	outGPU[0].synchronize_from_GPU_to_CPU();

	for (u32 i = 0; i < outCPU[0].getDataSize(); i++)
	{
		auto cpuValue = outCPU[0](i);
		auto gpuValue = outGPU[0](i);
		auto error = abs(cpuValue - gpuValue) / abs(cpuValue);

		if (error * 100 > 5)
		{
			std::cout << error << std::endl;
			break;
		}

	}

	for (u32 i = 0; i < outCPU[0].getDataSize(); i++)
	{
		outCPU[0].d(i) = 1;
		outGPU[0].d(i) = 1;
	}
	outGPU[0].synchronize_from_CPU_to_GPU();

	outCPU[0].backward();
	outGPU[0].backward();

	testTensor4GPU.synchronize_from_GPU_to_CPU();
	for (u32 i = 0; i < testTensor4GPU.getDataSize(); i++)
	{
		auto cpuValue = testTensor4CPU.d(i);
		auto gpuValue = testTensor4GPU.d(i);
		auto error = abs(cpuValue - gpuValue) / abs(cpuValue);

		if (error * 100 > 5)
		{
			std::cout << error << std::endl;
			break;
		}
	}
}


//111111111111111111111111111111111111111111111111111111111111111111111111111111

class MyLayer : public nnModule
{
public:
	MyLayer()
	{
		mlayer["Split"] = Split();
		mlayer["seq0"] = Sequential(Affine(100), ReLU(), Affine(10));
		mlayer["seq1"] = Sequential(Affine(10));
		mlayer["Add"] = Add();
	}

	iotype forward(const iotype& input) override
	{
		auto s = mlayer["Split"](input);
		auto t0 = mlayer["seq0"](s[0]);
		auto t1 = mlayer["seq1"](s[1]);
		auto a = mlayer["Add"](t0[0], t1[0]);
		return a;
	}

};

constexpr u32 nz = 100;
constexpr u32 ngf = 8;

class GNet : public nnModule
{
public:
	GNet()
	{
		mlayer["seq"] = Sequential(
			TransposeConv(ngf * 4, 5, 1, 1),//1->3
			BatchNorm(),
			ReLU(),
			TransposeConv(ngf * 2, 5, 2, 1),//3->7
			BatchNorm(),
			ReLU(),
			TransposeConv(ngf * 1, 4, 2, 1),//7->14
			BatchNorm(),
			ReLU(),
			TransposeConv(1, 4, 2, 1)//14->28
			//Tanh()
		);
	}

	iotype forward(const iotype& input) override
	{
		return mlayer["seq"](input);
	}
};

constexpr u32 ndf = ngf;
class DNet : public nnModule
{
public:
	DNet()
	{
		mlayer["seq"] = Sequential(
			Convolution(ndf, 4, 2, 1),//28->14
			BatchNorm(),
			ReLU(),
			Convolution(ndf * 2, 4, 2, 1),//14->7
			BatchNorm(),
			ReLU(),
			Convolution(ndf * 4, 4, 2, 1),//7->3
			BatchNorm(),
			ReLU(),
			Convolution(ndf * 8, 5, 1, 1)//1->1
		);
	}

	iotype forward(const iotype& input) override
	{
		return mlayer["seq"](input);
	}
};


class TestModule : public nnModule
{
public:
	TestModule()
	{
		mlayer["seq"] = Sequential(
			Convolution(ndf, 4, 2, 1),//28->14
			BatchNorm(),
			ReLU(),
			Convolution(ndf * 2, 4, 2, 1),//14->7
			BatchNorm(),
			ReLU(),
			Convolution(ndf * 4, 4, 2, 1),//7->3
			BatchNorm(),
			ReLU(),
			Convolution(ndf * 8, 5, 1, 1)//1->1
		);
		mlayer["affine"] = Affine(100);
		mlayer["seq1"] = Sequential(
			Convolution(ndf, 4, 2, 1),//28->14
			BatchNorm(),
			Sequential(
				Convolution(ndf, 4, 2, 1),//28->14
				BatchNorm(), Sequential(
					Convolution(ndf, 4, 2, 1),//28->14
					BatchNorm(),
					ReLU(),
					Convolution(ndf * 2, 4, 2, 1),//14->7
					BatchNorm(),
					ReLU(),
					Convolution(ndf * 4, 4, 2, 1),//7->3
					BatchNorm(),
					ReLU(),
					Convolution(ndf * 8, 5, 1, 1)//1->1
				)
			)
		);
	}

	iotype forward(const iotype& input) override
	{
		return mlayer["seq"](input);
	}
};


int main()
{
	/*auto t = gen<TestModule>();
	t.save(".");
	return 1;*/
	//check_Affine();
	//check_ReLU();
	//check_Conv();
	//check_MaxPooling();
	//check_BatchNorm();
	//check_TransposeConv();
	//check_Tanh();
	//return 1;

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
		v = Tensor(batch_size, 1, 28, 28);
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
	////テスト１
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

	//

	//return 1;
	//return 1;
	////テスト8
	//{
	//	auto seq = Sequential(Affine(300), ReLU(), Affine(300), ReLU(), Affine(300), ReLU(), Affine(100), ReLU(), Affine(10));
	//	//auto seq = Sequential(Affine(50), ReLU(), Affine(10));
	//	//auto seq = Sequential(Affine(10));
	//	//auto affine0 = Affine(100);
	//	//auto relu = ReLU();
	//	//auto affine1 = Affine(10);

	//	auto lossFunc = CrossEntropyWithSM();
	//	auto optim = Optimizer::SGD(0.001f);
	//	optim(seq);
	//	std::cout << "===============================" << std::endl;
	//	std::cout << "Test8" << std::endl;
	//	std::cout << "===============================" << std::endl;
	//	bool on_cuda = true;
	//	for (u32 i = 0; i < 1000; i++)
	//	{
	//		std::cout << "-------------------------------" << std::endl;
	//		std::cout << "Loop : " << i << std::endl;
	//		std::cout << "-------------------------------" << std::endl;

	//		DataType average_prob = 0.0f;
	//		for (u32 Bn = 0; Bn < batched_data_num; Bn++)
	//		{
	//			Tensor& training_tensor = input_tensor_tbl[Bn];  training_tensor.to_cuda(on_cuda);
	//			Tensor& correct_tensor = correct_tensor_tbl[Bn]; correct_tensor.to_cuda(on_cuda);
	//			auto t = seq(training_tensor);
	//			auto loss = lossFunc(t[0], correct_tensor);
	//			loss[0].synchronize_from_GPU_to_CPU();
	//			auto prob = Optimizer::convert_loss_to_prob(loss[0](0));
	//			average_prob = (average_prob * Bn + prob) / (Bn + 1);
	//			//std::cout << prob * 100 << std::endl;
	//			loss[0].backward();
	//			optim.optimize();

	//			progressBar(Bn, batched_data_num, average_prob * 100);
	//		}
	//		std::cout << std::endl;
	//	}
	//}

	////テスト9
	//{
	//	//auto seq = Sequential(Affine(300), ReLU(), Affine(300), ReLU(), Affine(300), ReLU(), Affine(100), ReLU(), Affine(10));
	//	//auto seq = gen<MyLayer>();

	//	//auto seq = Sequential(Convolution(3,4,2, 1), BatchNorm(), ReLU(), Affine(10));
	//	auto seq = Sequential(Convolution(3, 4, 2, 1), BatchNorm(), ReLU(), Convolution(3, 4, 2, 1), BatchNorm(), ReLU(), Affine(10));
	//	//auto seq = Sequential(Affine(10));
	//	//auto affine0 = Affine(100);
	//	//auto relu = ReLU();
	//	//auto affine1 = Affine(10);

	//	auto lossFunc = CrossEntropyWithSM();
	//	auto optim = Adam(0.001f);
	//	optim(seq);
	//	std::cout << "===============================" << std::endl;
	//	std::cout << "Test9" << std::endl;
	//	std::cout << "===============================" << std::endl;
	//	bool on_cuda = true;
	//	for (u32 i = 0; i < 1000; i++)
	//	{
	//		std::cout << "-------------------------------" << std::endl;
	//		std::cout << "Loop : " << i << std::endl;
	//		std::cout << "-------------------------------" << std::endl;

	//		DataType average_prob = 0.0f;
	//		for (u32 Bn = 0; Bn < batched_data_num; Bn++)
	//		{
	//			Tensor& training_tensor = input_tensor_tbl[Bn];  training_tensor.to_cuda(on_cuda);
	//			Tensor& correct_tensor = correct_tensor_tbl[Bn]; correct_tensor.to_cuda(on_cuda);
	//			auto t = seq(training_tensor);
	//			auto loss = lossFunc(t[0], correct_tensor);
	//			loss[0].synchronize_from_GPU_to_CPU();
	//			auto prob = BaseOptimizer::convert_loss_to_prob(loss[0](0));
	//			average_prob = (average_prob * Bn + prob) / (Bn + 1);
	//			//std::cout << prob * 100 << std::endl;
	//			loss[0].backward();
	//			optim.optimize();

	//			progressBar(Bn, batched_data_num, average_prob * 100);
	//		}
	//		std::cout << std::endl;
	//	}
	//}


	////テスト10
	//{
	//	//auto seq = Sequential(Affine(300), ReLU(), Affine(300), ReLU(), Affine(300), ReLU(), Affine(100), ReLU(), Affine(10));
	//	//auto seq = gen<MyLayer>();

	//	auto seq = Sequential(Convolution(1, 3, 1, 1, 0.0001f), ReLU(), Tanh());// , MaxPooling(3, 1, 1));
	//	auto split = Split();
	//	auto tanh_1 = Tanh();
	//	//auto seq = Sequential(Affine(10));
	//	//auto affine0 = Affine(100);
	//	//auto relu = ReLU();
	//	//auto affine1 = Affine(10);

	//	auto lossFunc = L2Loss();
	//	auto optim = SGD(0.00001f);
	//	optim(seq);
	//	std::cout << "===============================" << std::endl;
	//	std::cout << "Test10" << std::endl;
	//	std::cout << "===============================" << std::endl;
	//	bool on_cuda = true;
	//	for (u32 i = 0; i < 1000; i++)
	//	{
	//		std::cout << "-------------------------------" << std::endl;
	//		std::cout << "Loop : " << i << std::endl;
	//		std::cout << "-------------------------------" << std::endl;

	//		DataType average_prob = 0.0f;
	//		Tensor comparison = Tensor(batch_size, 1, 28, 28); comparison.to_cuda(on_cuda);
	//		for (u32 Bn = 0; Bn < batched_data_num; Bn++)
	//		{
	//			Tensor& training_tensor = input_tensor_tbl[Bn];  training_tensor.to_cuda(on_cuda);
	//			auto splited_training_data = split(training_tensor);
	//			auto t = seq(splited_training_data[0]);
	//			auto loss = lossFunc(t[0], tanh_1(splited_training_data[1])[0]);
	//			loss[0].synchronize_from_GPU_to_CPU();
	//			loss[0].backward();
	//			optim.optimize();

	//			progressBar(Bn, batched_data_num, loss[0](0));
	//		}
	//		std::cout << std::endl;
	//	}
	//}


	//テスト11
	{
		//auto seq = Sequential(Affine(300), ReLU(), Affine(300), ReLU(), Affine(300), ReLU(), Affine(100), ReLU(), Affine(10));
		auto d = gen<DNet>();
		auto g = gen<GNet>();
		auto split = Split();
		/*d.save(".\\");
		g.save(".\\");
		split.save(".\\");*/
		//auto seq = Sequential(Affine(10));
		//auto affine0 = Affine(100);
		//auto relu = ReLU();
		//auto affine1 = Affine(10);

		auto lossFunc = L2Loss();
		auto optim = Adam(0.01f, 0.5f, 0.999f);
		optim(d);
		optim(g);
		std::cout << "===============================" << std::endl;
		std::cout << "Test11" << std::endl;
		std::cout << "===============================" << std::endl;
		bool on_cuda = true;

		
		d.load(".");
		g.load(".");
		for (u32 i = 0; i < 1000; i++)
		{
			std::cout << "-------------------------------" << std::endl;
			std::cout << "Loop : " << i << std::endl;
			std::cout << "-------------------------------" << std::endl;


			DataType average_loss = 0.0f;
			for (u32 Bn = 0; Bn < batched_data_num; Bn++)
			{
				Tensor& training_tensor = input_tensor_tbl[Bn];  training_tensor.to_cuda(on_cuda);
				auto splited_training_data = split(training_tensor);
				auto t = d(splited_training_data[0]);
				auto s = g(t);
				auto loss = lossFunc(s[0], splited_training_data[1]);
				loss[0].synchronize_from_GPU_to_CPU();
				loss[0].backward();
				optim.optimize();
				average_loss = (average_loss * Bn + loss[0](0)) / (Bn + 1);
				progressBar(Bn, batched_data_num, average_loss);
#ifdef _DEBUG
				aoba::nn::layer::debugTimers.size();
#endif
				//d.load(".");
			}
			//d.save(".");
			//g.save(".");
			std::cout << std::endl;
		}
	}
	std::cout << "free check" << std::endl;
}

