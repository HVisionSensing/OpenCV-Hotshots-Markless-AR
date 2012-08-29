#ifndef ARDrawingContext_HPP
#define ARDrawingContext_HPP

////////////////////////////////////////////////////////////////////
// File includes:
#include "GeometryTypes.hpp"
#include "CameraCalibration.hpp"

////////////////////////////////////////////////////////////////////
// Standard includes:
#include <opencv2/opencv.hpp>

class ARDrawingContext
{
public:
  ARDrawingContext(const CameraCalibration& c);

  bool                isPatternPresent;
  Transformation      patternPose;

  //! Request the redraw of the OpenGl window
  void draw();

  //! Set the new frame for the background
  void updateBackground(const cv::Mat& frame);

private:
  //! Draws the background with video
  void drawCameraFrame();

  //! Draws the AR
  void drawAugmentedScene();

  //! Builds the right projection matrix from the camera calibration for AR
  void buildProjectionMatrix(const CameraCalibration& calibration, int w, int h, Matrix44& result);
  
  //! Draws the coordinate axis 
  void drawCoordinateAxis();
  
  //! Draw the cube model
  void drawCubeModel();

private:
  bool               m_isTextureInitialized;
  unsigned int       m_backgroundTextureId;
  CameraCalibration  m_calibration;
  cv::Mat            m_backgroundImage;
};

#endif