/**
 * \file CameraInfoLight.h
 * \author ROS CameraInfo.h message, modified by Céline Craye
 * \version 0.1
 * \date 20 july. 2016
 * \brief CameraInfoLight struct
 *        Object containing camera calibration parameters. Default values are factory values from
 *        the Kinect and Xtion pro live sensors
 */


#ifndef CAMERAINFOLIGHT_H
#define CAMERAINFOLIGHT_H


#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <cv.h>

struct CameraInfoLight
{

  CameraInfoLight()
    : height(0)
    , width(0)
    , D()
    , K()
    , R()
    , P()
    , binning_x(0)
    , binning_y(0){
      D = (cv::Mat_<double>(1,5) << 0.0, 0.0, 0.0, 0.0, 0.0);
      K = (cv::Mat_<double>(3,3) << 575.8157348632812, 0.0, 314.5, 0.0, 575.8157348632812, 235.5, 0.0, 0.0, 1.0);
      R = (cv::Mat_<double>(3,3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
      P = (cv::Mat_<double>(3,4) << 575.8157348632812, 0.0, 314.5, 0.0, 0.0, 575.8157348632812, 235.5, 0.0, 0.0, 0.0, 1.0, 0.0);
      height = 480;
      width = 640;
  }
  CameraInfoLight(const char* type)
    : height(0)
    , width(0)
    , D()
    , K()
    , R()
    , P()
    , binning_x(0)
    , binning_y(0){
      if(strcmp(type, "depth"))
      {
          D = (cv::Mat_<double>(1,5) << 0.0, 0.0, 0.0, 0.0, 0.0);
          K = (cv::Mat_<double>(3,3) << 575.8157348632812, 0.0, 314.5, 0.0, 575.8157348632812, 235.5, 0.0, 0.0, 1.0);
          R = (cv::Mat_<double>(3,3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
          P = (cv::Mat_<double>(3,4) << 575.8157348632812, 0.0, 314.5, 0.0, 0.0, 575.8157348632812, 235.5, 0.0, 0.0, 0.0, 1.0, 0.0);
      }
      else if(strcmp(type, "RGB"))
      {
          D = (cv::Mat_<double>(1,5) << 0.0, 0.0, 0.0, 0.0, 0.0);
          K = (cv::Mat_<double>(3,3) << 525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0);
          R = (cv::Mat_<double>(3,3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
          P = (cv::Mat_<double>(3,4) << 525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0);
      }
      height = 480;
      width = 640;
  }
  CameraInfoLight( uint32_t height, uint32_t width, cv::Mat D, cv::Mat K,cv::Mat R, cv::Mat P, uint32_t binning_x, uint32_t binning_y
)
    : height(height)
    , width(width)
    , D(D)
    , K(K)
    , R(R)
    , P(P)
    , binning_x(binning_x)
    , binning_y(binning_y){
  }


  uint32_t height;

  uint32_t width;

   cv::Mat D;

   cv::Mat K;

   cv::Mat R;

   cv::Mat P;

   typedef uint32_t _binning_x_type;
  _binning_x_type binning_x;

   typedef uint32_t _binning_y_type;
  _binning_y_type binning_y;



}; // struct CameraInfo

#endif // SENSOR_MSGS_MESSAGE_CAMERAINFO_H
