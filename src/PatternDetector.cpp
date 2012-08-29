////////////////////////////////////////////////////////////////////
// File includes:
#include "PatternDetector.hpp"

////////////////////////////////////////////////////////////////////
// Standard includes:
#include <cmath>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <cassert>

PatternDetector::PatternDetector(cv::Ptr<cv::FeatureDetector> detector, 
                                 cv::Ptr<cv::DescriptorExtractor> extractor, 
                                 cv::Ptr<cv::DescriptorMatcher> matcher, 
                                 bool buildPyramidLayers, 
                                 bool ratioTest)
  : m_detector(detector)
  , m_extractor(extractor)
  , m_matcher(matcher)
  , m_buildPyramid(buildPyramidLayers)
  , enableRatioTest(ratioTest)
  , enableHomographyRefinement(true)
{
}

void PatternDetector::train(const Pattern& pattern)
{
  m_pattern = pattern;

  std::vector<cv::Mat> descriptors(1);
  descriptors[0] = pattern.descriptors.clone();

  m_matcher->clear();
  m_matcher->add(descriptors);
  m_matcher->train();
}

void PatternDetector::computePatternFromImage(const cv::Mat& image, Pattern& pattern)
{
  int numImages = 4;
  float step = sqrtf(2.0f);

  // Store original image in pattern structure
  pattern.size = cv::Size(image.cols, image.rows);
  pattern.data = image.clone();

  // Build 2d and 3d contours (3d contour lie in XY plane since it's planar)
  pattern.points2d.resize(4);
  pattern.points3d.resize(4);

  // Image dimensions
  const float w = image.cols;
  const float h = image.rows;

  // Normalized dimensions:
  const float maxSize = std::max(w,h);
  const float unitW = w / maxSize;
  const float unitH = h / maxSize;

  pattern.points2d[0] = cv::Point2f(0,0);
  pattern.points2d[1] = cv::Point2f(w,0);
  pattern.points2d[2] = cv::Point2f(w,h);
  pattern.points2d[3] = cv::Point2f(0,h);

  pattern.points3d[0] = cv::Point3f(-unitW, -unitH, 0);
  pattern.points3d[1] = cv::Point3f( unitW, -unitH, 0);
  pattern.points3d[2] = cv::Point3f( unitW,  unitH, 0);
  pattern.points3d[3] = cv::Point3f(-unitW,  unitH, 0);

  cv::Mat gray;
  getGray(image, gray);
  extractFeatures(gray, pattern.keypoints, pattern.descriptors);

  // Detect features on pyramid levels only if the feature detector is not scale-invariant (GFTT, ORB, FAST etc)
  // The m_buildPyramid indicates this case and we will simulate scale-invariant by extracting a lot of features from numImages downsampled images
  if (m_buildPyramid)
  {
    std::vector<cv::KeyPoint> kpts;
    cv::Mat descriptors;
    
    for (int i = 0; i < numImages; i++)
    {
      float scale = step * (i+1);
      cv::Mat downsampled;

      // Resize image down to required size
      cv::Size dstSize(image.cols / scale, image.rows / scale);
      cv::resize(gray, downsampled, dstSize, 0, 0, cv::INTER_AREA);

      if (!extractFeatures(downsampled, kpts, descriptors))
        break;

      // Scale features to dimensions of original image
      for (size_t k = 0; k < kpts.size(); k++)
      {
        kpts[k].pt *= scale;
        kpts[i].size *= scale;
      }

      // Add keypoints
      std::copy(kpts.begin(), kpts.end(), std::back_inserter(pattern.keypoints));

      // Add descriptors
      pattern.descriptors = appendDescriptors(pattern.descriptors, descriptors);
    }
  }
}

cv::Mat getMatchesImage(cv::Mat query, cv::Mat pattern, const std::vector<cv::KeyPoint>& queryKp, const std::vector<cv::KeyPoint>& trainKp, const std::vector<cv::DMatch>&matches)
{
   cv::Mat outImg;
   cv::drawMatches(query, queryKp, pattern, trainKp, matches, outImg);
   return outImg;
}

bool PatternDetector::findPattern(const cv::Mat& image, PatternTrackingInfo& info)
{
  getGray(image, gray);
  extractFeatures(gray, queryKeypoints, queryDescriptors);
  getMatches(queryDescriptors, matches);

#if _DEBUG
  cv::imwrite("RoughMatches.png", getMatchesImage(image, m_pattern.data, queryKeypoints, m_pattern.keypoints, matches));
#endif

#if _DEBUG
  cv::Mat outImg;
  cv::Mat tmp = image.clone();
#endif

  bool homographyFound = refineMatchesWithHomography(queryKeypoints, m_pattern.keypoints, matches, roughHomography);

  if (homographyFound)
  {
    //cv::imwrite("RefinedUsingRansacMatches1.png", getMatchesImage(image, m_pattern.data, queryKeypoints, m_pattern.keypoints, matches));

    if (enableHomographyRefinement)
    {
      cv::warpPerspective(gray, warped, roughHomography, m_pattern.size, cv::WARP_INVERSE_MAP | cv::INTER_CUBIC);
#if _DEBUG
      cv::imshow("Warped image",warped);
#endif

      // Get refined matches:
      std::vector<cv::KeyPoint> warpedKeypoints;
      std::vector<cv::DMatch> refinedMatches;

      extractFeatures(warped, warpedKeypoints, queryDescriptors);
      getMatches(queryDescriptors, refinedMatches);

      homographyFound = refineMatchesWithHomography(warpedKeypoints, m_pattern.keypoints, refinedMatches, refinedHomography);
#if _DEBUG
      cv::imwrite("MatchesWithRefinedPose.png", getMatchesImage(warped, m_pattern.data, warpedKeypoints, m_pattern.keypoints, refinedMatches));
#endif
      info.homography = roughHomography * refinedHomography;

      // Transform contour with rough homography
#if _DEBUG
      cv::perspectiveTransform(m_pattern.points2d, info.points2d, roughHomography);
      info.draw2dContour(tmp, CV_RGB(0,200,0));
#endif

      // Transform contour with precise homography
      cv::perspectiveTransform(m_pattern.points2d, info.points2d, info.homography);
#if _DEBUG
      info.draw2dContour(tmp, CV_RGB(200,0,0));
#endif
    }
    else
    {
      info.homography = roughHomography;

      // Transform contour with rough homography
      cv::perspectiveTransform(m_pattern.points2d, info.points2d, roughHomography);
#if _DEBUG
      info.draw2dContour(tmp, CV_RGB(0,200,0));
#endif
    }
  }

#if _DEBUG
  if (1)
  {
    cv::drawMatches( tmp, queryKeypoints, m_pattern.data, m_pattern.keypoints, matches, outImg);
    cv::imshow("Matches", outImg);
  }
  std::cout << "Features:" << std::setw(4) << queryKeypoints.size() << " Matches: " << std::setw(4) << matches.size() << std::endl;
#endif
  
  return homographyFound;
}

void PatternDetector::getGray(const cv::Mat& image, cv::Mat& gray)
{
  if (image.channels()  == 3)
    cv::cvtColor(image, gray, CV_BGR2GRAY);
  else if (image.channels() == 4)
    cv::cvtColor(image, gray, CV_BGRA2GRAY);
  else if (image.channels() == 1)
    gray = image;
}

cv::Mat PatternDetector::appendDescriptors(cv::Mat& a, cv::Mat& b)
{
  assert(a.cols == b.cols);
  assert(a.type() == b.type());

  cv::Mat result(a.rows + b.rows, a.cols, a.type());

  if (a.type() == CV_32F)
  {
    for (int r = 0; r < a.rows; r++)
      for (int c = 0; c < a.cols; c++)
        result.at<float>(r,c) = a.at<float>(r,c);

    for (int r = 0; r < b.rows; r++)
      for (int c = 0; c < b.cols; c++)
        result.at<float>(r+a.rows,c) = b.at<float>(r,c);
  }
  else if (a.type() == CV_8UC1)
  {
    for (int r = 0; r < a.rows; r++)
      for (int c = 0; c < a.cols; c++)
        result.at<unsigned char>(r,c) = a.at<unsigned char>(r,c);

    for (int r = 0; r < b.rows; r++)
      for (int c = 0; c < b.cols; c++)
        result.at<unsigned char>(r + a.rows,c) = b.at<unsigned char>(r,c);
  }

  return result;
}

bool PatternDetector::extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
  assert(!image.empty());
  assert(image.channels() == 1);

  m_detector->detect(image, keypoints);
  if (keypoints.empty())
    return false;

  m_extractor->compute(image, keypoints, descriptors);
  if (keypoints.empty())
    return false;

  return true;
}

void PatternDetector::getMatches(const cv::Mat& queryDescriptors, std::vector<cv::DMatch>& matches)
{
  matches.clear();

  if (enableRatioTest)
  {
    m_matcher->knnMatch(queryDescriptors, knnMatches, 2);

    for (size_t i=0; i<knnMatches.size(); i++)
    {
      cv::DMatch& bestMatch   = knnMatches[i][0];
      cv::DMatch& betterMatch = knnMatches[i][1];

      float distanceRatio = bestMatch.distance / betterMatch.distance;
      if (distanceRatio < 0.75)
      {
        matches.push_back(bestMatch);
      }
    }
  }
  else
  {
    m_matcher->match(queryDescriptors, matches);
  }
}

bool PatternDetector::refineMatchesWithHomography(const std::vector<cv::KeyPoint>& queryKeypoints,const std::vector<cv::KeyPoint>& trainKeypoints, std::vector<cv::DMatch>& matches, cv::Mat& homography)
{
  const int MinMatchesAllowed = 8;

  if (matches.size() < MinMatchesAllowed)
    return false;

  std::vector<cv::Point2f> srcKp(matches.size()), dstKp(matches.size());
  for (size_t i=0; i<matches.size(); i++)
  {
    srcKp[i] = trainKeypoints[matches[i].trainIdx].pt;
    dstKp[i] = queryKeypoints[matches[i].queryIdx].pt;
  }

  std::vector<unsigned char> mask(srcKp.size());
  homography = cv::findHomography(srcKp, dstKp, CV_FM_RANSAC, 3, mask);

  size_t numberOfInliers = std::count(mask.begin(), mask.end(), 1);

  std::vector<cv::DMatch> inliers;
  for (size_t i=0; i<mask.size(); i++)
  {
    if (mask[i])
      inliers.push_back(matches[i]);
  }

  matches.swap(inliers);
  return matches.size() > MinMatchesAllowed;
}
