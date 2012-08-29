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
    ARPipeline pipeline(patternImage, calibration);
    ARDrawingContext drawingCtx(calibration);

    // Grab first frame to get the frame dimensions
    cv::Mat currentFrame;  
    capture >> currentFrame;

    // Create window with OpenGL support
    cv::namedWindow(ARWindowName, cv::WINDOW_OPENGL);
    
    // Resize it exactly to video size
    cv::resizeWindow(ARWindowName, currentFrame.cols, currentFrame.rows);

    // Initialize OpenGL draw callback:
    cv::setOpenGlContext(ARWindowName);
    cv::setOpenGlDrawCallback(ARWindowName, drawAR, &drawingCtx);

    while (capture.grab() && capture.retrieve(currentFrame))
    {
        drawingCtx.updateBackground(currentFrame);

        // Process frame and update it's location
        drawingCtx.isPatternPresent = pipeline.processFrame(currentFrame);
        drawingCtx.patternPose = pipeline.getPatternLocation();

        // Invoke redraw of the OpenGL window
        cv::updateWindow(ARWindowName);

        // Wait for keyboard input for 1 ms
        int keyCode = cv::waitKey(1);
        
        // Quit if user pressed 'q' or ESC button
        if (keyCode == 'q' || keyCode == 27) 
            break;
    }
}

void processSingleImage(const cv::Mat& patternImage, CameraCalibration& calibration, const cv::Mat& image)
{
    ARPipeline pipeline(patternImage, calibration);
    ARDrawingContext drawingCtx(calibration);

    cv::namedWindow(ARWindowName, cv::WINDOW_OPENGL);
    cv::resizeWindow(ARWindowName, 640, 480);
    cv::setOpenGlContext(ARWindowName);
    cv::setOpenGlDrawCallback(ARWindowName, drawAR, &drawingCtx);

    drawingCtx.updateBackground(image);
    bool found = pipeline.processFrame(image);
    drawingCtx.isPatternPresent = found;
    if (found)
    {
        drawingCtx.patternPose = pipeline.getPatternLocation();
    }

    cv::updateWindow(ARWindowName);
    cv::waitKey(-1);
}

void drawAR(void* param)
{
    ARDrawingContext * ctx = static_cast<ARDrawingContext*>(param);
    if (ctx)
    {
        ctx->draw();
    }
}