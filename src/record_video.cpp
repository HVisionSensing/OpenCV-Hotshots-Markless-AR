//////////////////////////////////////////////////////////////////////////
// Standard includes:
#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>
#include <list>

int main(int argc, char * argv[])
{
  std::list<cv::Mat> frames;

  cv::VideoCapture capture(1); // open the default camera

  if( !capture.isOpened() ) 
  {
    printf("Camera failed to open!\n");
    return -1;
  }

  cv::Mat frame;
  capture >> frame; // get first frame for size

  cv::namedWindow("video",1);

  for(;;)
  {
    // get a new frame from camera
    capture >> frame; 

    frames.push_back(frame.clone());

    // show frame on screen
    cv::imshow("video", frame); 

    if (cv::waitKey(10) >= 0)
      break;
  }

  // record video
  cv::VideoWriter record("RobotVideo.avi", 0, 30, frame.size(), true);
  if( !record.isOpened() )
  {
    printf("VideoWriter failed to open!\n");
    return -1;
  }

  for (std::list<cv::Mat>::iterator i = frames.begin(); i != frames.end(); ++i)
  {
    record << *i;
  }

  // the camera will be deinitialized automatically in VideoCapture destructor
  // the recorded video will be closed automatically in the VideoWriter destructor
  return 0;
}