//#pragma once
//#include "Layer.h"
//#include "Add.h"
//
//namespace aoba {
//	namespace nn {
//		namespace layer {
//			class AddAsInnerCore : public LayerSkeleton
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
//			inline nnLayer AddAsInner()
//			{
//				return gen<AddAsInnerCore>("AddAsInner");
//			}
//		}
//	}
//}
//
//
//
//
