#include "Layer/Layer.h"
#include "Adam.h"

namespace
{
	__global__ void optimize_gpu_impl()
	{

	}
}



namespace aoba
{
	namespace nn
	{
		namespace optimizer
		{
			Adam::Adam(DataType learningRate)
				:BaseOptimizer(learningRate)
			{
			}

			Adam::~Adam()
			{
			}


			void Adam::initialize()
			{
#ifdef _DEBUG
				std::cout << "Adam::initialize()" << std::endl;
#endif
			}
		}
	}
}