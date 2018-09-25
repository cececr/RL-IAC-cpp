/**
 * \file dataformater.h
 * \brief DataFormater
 * \author Céline Craye
 * \version 0.1
 * \date 11 / 19 / 2015
 *
 * This class makes a bridge between kinect processed images providing either
 * feature maps or segmentation maps, and the machine learning tools taking as input
 * matrices of features and labels.
 *
 */

#ifndef DATAFORMATER_H
#define DATAFORMATER_H
#include <cv.h>

class DataFormater
{
public:
    DataFormater(int max_size = 0,
                 int resampling_factor = 20,
                 bool ignore_unknown = true);
    bool maps_reformat(std::vector<cv::Mat> feature_maps, cv::Mat segmentation_map);
    bool maps_reformat(std::vector<cv::Mat> feature_maps, std::vector<int> labels, std::vector<cv::Rect> bounding_boxes);
    bool clear_buffers();
    bool get_buffers(cv::Mat& data, cv::Mat &classes);
    int get_buffers_length();

private:
    int max_size;
    int resampling_factor;
    bool ignore_unknown;
    cv::Mat trainingData;
    cv::Mat trainingClasses;
};

#endif // DATAFORMATER_H
