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
  CameraCalibration calibration(526.58037684199849, 524.65577209994706, 318.41744018680112, 202.96659047014398);

  if (argc < 2)
  {
    std::cerr << "Input image not specified" << std::endl;
    return 1;
  }

  cv::Mat patternImage = cv::imread(argv[1]);
  cv::resize(patternImage, patternImage, cv::Size(patternImage.cols/2, patternImage.rows/2));
  if (patternImage.empty())
  {
    std::cerr << "Input image cannot be read" << std::endl;
    return 3;
  }

  if (argc == 2)
  {
    processVideo(patternImage, calibration, cv::VideoCapture());
  }
  else if (argc == 3)
  {
    std::string input = argv[2];
    cv::Mat image = cv::imread(input);
    if (!image.empty())
    {
      processSingleImage(patternImage, calibration, image);
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
  int frameToProcess = 50;

  ARPipeline pipeline(patternImage, calibration);
  ARDrawingContext drawingCtx(calibration);

  cv::namedWindow(ARWindowName, cv::WINDOW_OPENGL);
  cv::resizeWindow(ARWindowName, 640, 480);
  cv::setOpenGlContext(ARWindowName);
  cv::setOpenGlDrawCallback(ARWindowName, drawAR, &drawingCtx);

  cv::Mat currentFrame;  
  while (capture.grab() && capture.retrieve(currentFrame))
  {
    //frameToProcess--;
    //cv::imshow("Live",currentFrame);

    drawingCtx.updateBackground(currentFrame);
    bool found = pipeline.processFrame(currentFrame);

    drawingCtx.patternPresent = found;
    if (found)
    {
      drawingCtx.patternPose = pipeline.getPatternLocation();
    }

    cv::updateWindow(ARWindowName);
    int code = cv::waitKey(1);
    if (code == 'q')
      break;

    if (frameToProcess == 0)
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
  drawingCtx.patternPresent = found;
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