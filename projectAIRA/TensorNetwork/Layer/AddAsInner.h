//#pragma once
//#include "Layer.h"
//#include "Add.h"
//
//namespace aoba {
//	namespace nn {
//		namespace layer {
//			class Layer::AddAsInnerCore : public Layer::LayerSkeleton
//			{
//			public:
//				AddAsInnerCore() : LayerSkeleton(2, 1)
//				{
//					mlayer["add"] = Add();
//					mAdd = Add();
//				}
//
//				virtual iotype forward(const iotype& input_tensors) override
//				{
//					return mlayer["add"](input_tensors);
//				}
//
//			private:
//				Layer mAdd;
//			};
//
//
//
//			inline Layer::nnLayer AddAsInner()
//			{
//				return gen<Layer::AddAsInnerCore>("AddAsInner");
//			}
//		}
//	}
//}
//
//
//
//
