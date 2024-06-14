#pragma once
#include "typeinfo.h"


inline void progressBar(u32 currentLoop, u32 totalLoop, f32 loss, u32 length = 80)
{
	const u32 grid = totalLoop / length;
	std::string s = "\r";
	for (u32 i = 0; i < static_cast<u32>((static_cast<f32>(length) * currentLoop) / totalLoop); i++)
	{
		s += "=";
	}
	s += ">";
	const s32 spaceLength = static_cast<s32>(length - s.length() + 2);
	for (s32 i = 0; i < spaceLength; i++)
	{
		s += " ";
	}
	s += " " + std::to_string(static_cast<u32>(static_cast<f32>(currentLoop * 100) / totalLoop)) + "/100  :  " + std::to_string(loss);
	printf(s.c_str());
};