#ifndef DEBUG_HELPERS_HPP
#define DEBUG_HELPERS_HPP

#include <string>
#include <sstream>

template <typename T>
std::string ToString(const T& value)
{
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

namespace cv
{
    inline void showAndSave(std::string name, const cv::Mat& m)
    {
        cv::imshow(name, m);
        cv::imwrite(name + ".png", m);
		//cv::waitKey(25);
    }

	inline cv::Mat getMatchesImage(cv::Mat query, cv::Mat pattern, const std::vector<cv::KeyPoint>& queryKp, const std::vector<cv::KeyPoint>& trainKp, std::vector<cv::DMatch> matches, int maxMatchesDrawn)
	{
		cv::Mat outImg;

		if (matches.size() > maxMatchesDrawn)
		{
			matches.resize(maxMatchesDrawn);
		}

		cv::drawMatches
			(
			query, 
			queryKp, 
			pattern, 
			trainKp,
			matches, 
			outImg, 
			cv::Scalar(0,200,0,255), 
			cv::Scalar::all(-1),
			std::vector<char>(), 
			cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
			);

		return outImg;
	}
}

#endif