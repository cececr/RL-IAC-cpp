/**
 * \file deepfeatureextractor.h
 * \brief DeepFeatureExtractor
 * \author Céline Craye
 * \version 0.1
 * \date 4 / 14 / 2016
 *
 * Custom DOxygen-style template. Provide here a description of the file
 *
 */

#ifndef DEEPFEATUREEXTRACTOR_H
#define DEEPFEATUREEXTRACTOR_H

//#define CPU_ONLY
#include "FeatureExtractor.h"

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)

enum EXTRACTOR_MODE {
    MODE_VPS,
    MODE_KPS,
    MODE_KPS_VPS
};

class DeepFeatureExtractor : public FeatureExtractor
{
public:
    // Abstract functions
    DeepFeatureExtractor();
    DeepFeatureExtractor(int downsampling_ratio, int nb_superpixels);
    std::vector<cv::Mat> getFeatureMap(cv::Mat input);
    int getNbFeatures();
    int getFeatureType();

    // Other
    DeepFeatureExtractor(
              const std::string& model_file,
              const std::string& trained_file,
              const std::string& mean_file,
              const std::string& extract_layer,
              const int nscales, const int nb_superpixels);
    void ExtractFeatures(cv::Mat& img, std::vector<cv::Rect>& bboxes, int classLabel, std::vector<cv::Mat>& res);
    cv::Mat ExtractDeepFeatures(cv::Mat& img, std::string layer_name = "");

private:
    std::vector<cv::Mat> pad_images(cv::Mat input, int padding);
    cv::Mat reformat_feature_maps(std::vector<cv::Mat> padded_features);

    cv::Mat SetMean(caffe::shared_ptr<Net<float> > net);
    void WrapInputLayer(caffe::shared_ptr<Net<float> > net, std::vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, cv::Mat& meanImg, std::vector<cv::Mat>* input_channels);
    void SetNetworkParams(caffe::shared_ptr<Net<float> > net);

    cv::Mat RunNetwork(caffe::shared_ptr<Net<float> > net, cv::Mat& meanImg, cv::Mat& img,std::string layer_name);

private:
    std::string layer_name;
    boost::shared_ptr<Net<float> > net;
    float meanRGB[3];
    cv::Mat _mean;
    std::string extract_layer;
    int nscales;
    bool use_mean;

};
#endif
