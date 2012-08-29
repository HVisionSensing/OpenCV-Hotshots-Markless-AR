////////////////////////////////////////////////////////////////////
// File includes:
#include "ARDrawingContext.hpp"
#include "ARPipeline.hpp"

////////////////////////////////////////////////////////////////////
// Standard includes:
#include <opencv2/opencv.hpp>
#include <gl/gl.h>
#include <gl/glu.h>

void drawAR(void* param);
void processVideo(const cv::Mat& patternImage, CameraCalibration& calibration, cv::VideoCapture& capture);
void processSingleImage(const cv::Mat& patternImage, CameraCalibration& calibration, const cv::Mat& image);

static const char * ARWindowName = "Markerless AR";

int main(int argc, const char * argv[])
{
    // Change this calibration to yours:
    CameraCalibration calibration(526.58037684199849f, 524.65577209994706f, 318.41744018680112f, 202.96659047014398f);
    
    if (argc < 2)
    {
        std::cout << "Input image not specified" << std::endl;
        std::cout << "Usage: markerless_ar_demo <pattern image> [filepath to recorded video or image]" << std::endl;
        return 1;
    }

    // Try to read the pattern:
    cv::Mat patternImage = cv::imread(argv[1]);
    if (patternImage.empty())
    {
        std::cout << "Input image cannot be read" << std::endl;
        return 2;
    }

    //cv::resize(patternImage, patternImage, cv::Size(patternImage.cols/2, patternImage.rows/2));

    if (argc == 2)
    {
        processVideo(patternImage, calibration, cv::VideoCapture());
    }
    else if (argc == 3)
    {
        std::string input = argv[2];
        cv::Mat testImage = cv::imread(input);
        if (!testImage.empty())
        {
            processSingleImage(patternImage, calibration, testImage);
        }
        else 
        {
            cv::VideoCapture cap;
            if (cap.open(input))
            {
                processVideo(patternImage, calibration, cap);
            }
        }
    }
    else
    {
        std::cerr << "Invalid number of arguments passed" << std::endl;
        return 1;
    }

    return 0;
}

void processVideo(const cv::Mat& patternImage, CameraCalibration& calibration, cv::VideoCapture& capture)
{
	// Grab first frame to get the frame dimensions
	cv::Mat currentFrame;  
	capture >> currentFrame;
	cv::Size frameSize(currentFrame.cols, currentFrame.rows);

    ARPipeline pipeline(patternImage, calibration);
    ARDrawingContext drawingCtx(ARWindowName, frameSize, calibration);

    while (capture.grab() && capture.retrieve(currentFrame))
    {
        drawingCtx.updateBackground(currentFrame);

        // Process frame and update it's location
        drawingCtx.isPatternPresent = pipeline.processFrame(currentFrame);
        drawingCtx.patternPose = pipeline.getPatternLocation();

        // Invoke redraw of the OpenGL window
        drawingCtx.updateBackground(currentFrame);

        // Wait for keyboard input for 1 ms
        int keyCode = cv::waitKey();
        
        // Quit if user pressed 'q' or ESC button
        if (keyCode == 'q' || keyCode == 27) 
            break;
    }
}

void processSingleImage(const cv::Mat& patternImage, CameraCalibration& calibration, const cv::Mat& image)
{
    ARPipeline pipeline(patternImage, calibration);
	bool patternPresent = pipeline.processFrame(image);
	cv::waitKey(-1);

    ARDrawingContext drawingCtx(ARWindowName, cv::Size(image.cols, image.rows), calibration);
    drawingCtx.updateBackground(image);
    drawingCtx.isPatternPresent = patternPresent;
    drawingCtx.patternPose = pipeline.getPatternLocation();

	while (cv::waitKey() != 27)
	{
		drawingCtx.updateWindow();
	}
}

