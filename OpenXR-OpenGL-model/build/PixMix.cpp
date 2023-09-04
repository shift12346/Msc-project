#include "PixMix.h"
// code implmentation based on the https://github.com/Mugichoko445/PixMix-Inpainting
namespace dr
{
	PixMix::PixMix() { }
	PixMix::~PixMix() { }

	void PixMix::Execute(cv::InputArray sourceColor, cv::InputArray sourceMask, cv::OutputArray resultInpaint, const PixMixParams& settings, bool enableDebugVisualization)
	{
		assert(color.size() == mask.size());
		assert(color.type() == CV_8UC3);
		assert(mask.type() == CV_8U);

		Initialize(sourceColor, sourceMask, settings.blurSize);

		for (int lv = int(pm.size()) - 1; lv >= 0; --lv)
		{
			pm[lv].execute(settings);
			if (lv > 0) PopulateInferiorLevel(pm[lv], pm[lv - 1]);

#pragma region DEBUG_VIZ
			if (enableDebugVisualization)
			{
				cv::UMat vizColor, vizPosMap;
				cv::imshow("debug - inpainted color", vizColor);
				cv::imshow("debug - colord position map", vizPosMap);
				cv::waitKey(1);
			}
#pragma endregion

		}

		BlendBorder(resultInpaint);
	}

	void PixMix::Initialize(cv::InputArray color, cv::InputArray mask, const int blurSize)
	{

		// Construct the image pyramid
		pm.resize(DeterminePyramidLevel(color.cols(), color.rows()));

		// Initialize the base of the pyramid
		pm[0].Initilization(color.getMat(), mask.getMat());

		for (int level = 1; level < pm.size(); ++level)
		{
			auto halfSize = pm[level - 1].RetrieveColorPointer()->size() / 2;

			// Downsample the color from the previous level
			cv::Mat3b downsampledColor;
			cv::resize(*(pm[level - 1].RetrieveColorPointer()), downsampledColor, halfSize, 0.0, 0.0, cv::INTER_LINEAR);

			// Downsample and threshold the mask simultaneously
			cv::Mat1b downsampledMask(halfSize);
			auto prevMask = pm[level - 1].RetrieveMaskPointer();
			for (int row = 0; row < halfSize.height; ++row)
			{
				for (int col = 0; col < halfSize.width; ++col)
				{
					uchar value = cv::saturate_cast<uchar>(((*prevMask)(2 * row, 2 * col) +
						(*prevMask)(2 * row + 1, 2 * col) +
						(*prevMask)(2 * row, 2 * col + 1) +
						(*prevMask)(2 * row + 1, 2 * col + 1)) / 4.0);
					downsampledMask(row, col) = value < 255 ? 0 : 255;
				}
			}

			// Initialize the current pyramid level with the downsampled data
			pm[level].Initilization(downsampledColor, downsampledMask);
		}

		// Prepare for the final composition
		colorData = color.getMat().clone();
		cv::blur(mask, alphaData, cv::Size(blurSize, blurSize));
	}

	int PixMix::DeterminePyramidLevel(int width, int height)
	{
		// Initialize the pyramid level to 1
		int pyramidLevel = 1;

		// Start with the smallest dimension of the image
		int minSize = std::min(width, height);

		// Continually halve the size until it's less than 5
		// or until pyramid level reaches 6
		while (minSize >= 10 && pyramidLevel < 6)
		{
			minSize /= 2;
			pyramidLevel++;
		}

		return pyramidLevel;
	}

	void PixMix::PopulateInferiorLevel(OneLvPixMix& pmUpper, OneLvPixMix& pmLower)
	{

		// Upsample color from the upper pyramid level to the size of the lower level
		cv::Mat3b upsampledColor;
		cv::resize(*(pmUpper.RetrieveColorPointer()), upsampledColor, pmLower.RetrieveColorPointer()->size(), 0.0, 0.0, cv::INTER_LINEAR);

		// Upsample position map from the upper pyramid level and adjust values based on position
		cv::Mat2i upsampledPosMap;
		cv::resize(*(pmUpper.RetrievePositionMapPointer()), upsampledPosMap, pmLower.RetrievePositionMapPointer()->size(), 0.0, 0.0, cv::INTER_NEAREST);

		for (int row = 0; row < upsampledPosMap.rows; ++row)
		{
			auto posMapRow = upsampledPosMap.ptr<cv::Vec2i>(row);
			for (int col = 0; col < upsampledPosMap.cols; ++col)
			{
				posMapRow[col] = posMapRow[col] * 2 + cv::Vec2i(row % 2, col % 2);
			}
		}

		// Get the lower pyramid level's data
		cv::Mat3b& colorLower = *(pmLower.RetrieveColorPointer());
		cv::Mat1b& maskLower = *(pmLower.RetrieveMaskPointer());
		cv::Mat2i& posMapLower = *(pmLower.RetrievePositionMapPointer());

		const int widthLower = colorLower.cols;
		const int heightLower = colorLower.rows;

		// Update the color and position map in the lower pyramid level based on the mask
		for (int row = 0; row < heightLower; ++row)
		{
			auto colorLowerRow = colorLower.ptr<cv::Vec3b>(row);
			auto colorUpsampledRow = upsampledColor.ptr<cv::Vec3b>(row);
			auto maskLowerRow = maskLower.ptr<uchar>(row);
			auto posMapLowerRow = posMapLower.ptr<cv::Vec2i>(row);
			auto posMapUpsampledRow = upsampledPosMap.ptr<cv::Vec2i>(row);

			for (int col = 0; col < widthLower; ++col)
			{
				if (maskLowerRow[col] == 0)
				{
					colorLowerRow[col] = colorUpsampledRow[col];
					posMapLowerRow[col] = posMapUpsampledRow[col];
				}
			}
		}
	}

	void PixMix::BlendBorder(cv::OutputArray output)
	{
		// Convert color data to floating point for accurate blending calculations
		cv::Mat3f originalColorFloat;
		cv::Mat3f pyramidColorFloat;
		cv::Mat1f alphaFloat;

		colorData.convertTo(originalColorFloat, CV_32FC3, 1.0 / 255.0);
		pm[0].RetrieveColorPointer()->convertTo(pyramidColorFloat, CV_32FC3, 1.0 / 255.0);
		alphaData.convertTo(alphaFloat, CV_32F, 1.0 / 255.0);

		// Prepare the output matrix to receive the blended data
		cv::Mat3f blendedOutput(originalColorFloat.size());

		const int numRows = colorData.rows;
		const int numCols = colorData.cols;

		// Blend using the alpha values
		for (int row = 0; row < numRows; ++row)
		{
			auto originalColorRow = originalColorFloat.ptr<cv::Vec3f>(row);
			auto pyramidColorRow = pyramidColorFloat.ptr<cv::Vec3f>(row);
			auto blendedOutputRow = blendedOutput.ptr<cv::Vec3f>(row);
			auto alphaRow = alphaFloat.ptr<float>(row);

			for (int col = 0; col < numCols; ++col)
			{
				float alphaValue = alphaRow[col];
				blendedOutputRow[col] = alphaValue * originalColorRow[col] + (1.0f - alphaValue) * pyramidColorRow[col];
			}
		}

		// Convert the blended output back to byte format
		blendedOutput.convertTo(output, CV_8UC3, 255.0);
	}
}