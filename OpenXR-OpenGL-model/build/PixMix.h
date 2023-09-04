#pragma once
// code implmentation based on the https://github.com/Mugichoko445/PixMix-Inpainting
#include <vector>
#include "OneLvPixMix.h"

namespace dr
{
	using namespace det;

	class PixMix
	{
	public:
		PixMix();
		~PixMix();

		void Execute(cv::InputArray sourceColor, cv::InputArray sourceMask, cv::OutputArray resultInpaint, const PixMixParams& settings, bool enableDebugVisualization = false);

	private:
		std::vector<OneLvPixMix> pm;
		cv::Mat3b colorData;
		cv::Mat1b alphaData;

		void Initialize(cv::InputArray color, cv::InputArray mask, const int blurSize);
		int  DeterminePyramidLevel(int width, int height);
		void PopulateInferiorLevel(OneLvPixMix& pmUpper, OneLvPixMix& pmLower);
		void BlendBorder(cv::OutputArray output);
	};
}