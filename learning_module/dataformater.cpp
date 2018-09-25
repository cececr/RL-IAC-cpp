/**
 * \file dataformater.cpp
 * \brief DataFormater
 * \author Céline Craye
 * \version 0.1
 * \date 11 / 19 / 2015
 *
 * Please see header for details
 *
 */

#include "dataformater.h"

/**
 * @brief DataFormater::DataFormater
 * @param max_size max number of sample allowed in the buffer
 * @param resampling_factor downsampling input images by this factor
 * @param ignore_unknown whether we process unknown data as not salient (true) or not (false)
 */
DataFormater::DataFormater(int max_size, int resampling_factor, bool ignore_unknown)
{
    this->max_size = max_size;
    this->resampling_factor = resampling_factor;
    this->ignore_unknown = ignore_unknown;
}

/**
 * @brief DataFormater::maps_reformat
 *        turns extracted feature maps and segmentation maps into samples and labels formated
 *        for classifiers. If dataset exceeds a limit size, samples are randomly removed
 * @param feature_maps the vector of RGB feature maps
 * @param segmentation_map the associated depth segmentation map
 * @return true anyways
 */
bool DataFormater::maps_reformat(std::vector<cv::Mat> feature_maps, cv::Mat segmentation_map)
{
    /* Format feature maps */
    cv::Mat feature_maps_mat;
    cv::merge(feature_maps,feature_maps_mat);

    /* Downsample data */
    cv::Size training_size(resampling_factor,resampling_factor);
    resize(segmentation_map, segmentation_map, training_size,cv::INTER_NEAREST );
    resize(feature_maps_mat, feature_maps_mat, training_size,cv::INTER_NEAREST );

    /* Reshape matrices */
    feature_maps_mat = feature_maps_mat.reshape(1,training_size.height*training_size.width);
    segmentation_map = segmentation_map.reshape(1,training_size.height*training_size.width);

    /* Find data with missing labels */
    cv::Mat missing_data;
    cv::inRange(segmentation_map,1,254,missing_data);
    if(!ignore_unknown)
    {
        /// If unknown are considered, all unknowns are set to "not salient" value
        segmentation_map.setTo(0, missing_data);
        missing_data = cv::Mat::zeros(segmentation_map.size(), CV_8U);
    }
    /* Format labels */
    if(!segmentation_map.empty())
    {
        segmentation_map.convertTo(segmentation_map,CV_32F);
        segmentation_map = segmentation_map/255;
    }

    /* Add data to the buffers */
    for(int i = 0 ; i < segmentation_map.rows ; i++)
    {
        if(missing_data.at<uchar>(i)> 0 )
            continue;
        trainingData.push_back(feature_maps_mat.row(i));
        trainingClasses.push_back(segmentation_map.row(i));
    }

    /* Remove data if buffers are full */
    if(max_size > 0 && trainingClasses.rows > max_size)
    {
        cv::Mat trainingDataCopy = trainingData.clone();
        cv::Mat trainingClassesCopy = trainingClasses.clone();
        std::vector<int> rand_idx;
        for(int i = 0 ; i < trainingClasses.rows ; i++)
            rand_idx.push_back(i);
        std::random_shuffle(rand_idx.begin(),rand_idx.end());
        for(int i = 0 ; i < trainingClasses.rows ; i++)
        {
            trainingData.row(i) = trainingDataCopy.row(rand_idx[i]);
            trainingClasses.row(i) = trainingClassesCopy.row(rand_idx[i]);
        }
        int nsamplesRemoved = trainingClasses.rows - max_size;
        trainingData(cv::Rect(0,
                              nsamplesRemoved,
                              trainingData.cols,
                              trainingData.rows - nsamplesRemoved)).copyTo(trainingData);
        trainingClasses(cv::Rect(0,
                                 nsamplesRemoved,
                                 trainingClasses.cols,
                                 trainingClasses.rows - nsamplesRemoved)).copyTo(trainingClasses);
    }

    return true;
}

bool DataFormater::maps_reformat(std::vector<cv::Mat> feature_maps, std::vector<int> labels, std::vector<cv::Rect> bounding_boxes)
{
    assert(labels.size() == bounding_boxes.size());
    /* Calculate integral image for each feature map */
    std::vector<cv::Mat> integral_feature_maps;
    for(size_t i = 0 ;  i < feature_maps.size() ; i++)
    {
        cv::Mat integral;
        cv::integral(feature_maps[i], integral);
        integral_feature_maps.push_back(integral);
    }
    cv::Mat integral_features_mat;
    merge(integral_feature_maps, integral_features_mat);

    /* For each bounding box, calculate average feature value */
    for(size_t i = 0 ; i < labels.size() ; i++)
    {
        cv::Mat class_mat = labels[i] * cv::Mat::ones(1,1,CV_32F);
        cv::Rect r = bounding_boxes[i];
        cv::Mat integral_topleft = integral_features_mat.row(r.y).col(r.x);
        integral_topleft.reshape(1,1);

        cv::Mat integral_bottomleft = integral_features_mat.row(r.y+r.height).col(r.x);
        integral_bottomleft.reshape(1,1);

        cv::Mat integral_topright = integral_features_mat.row(r.y).col(r.x+r.width);
        integral_topright.reshape(1,1);

        cv::Mat integral_bottomright = integral_features_mat.row(r.y+r.height).col(r.x+r.width);
        integral_bottomright.reshape(1,1);
        cv::Mat avg_square = 1/((float)r.height*r.width)*( integral_topleft
                                                          +integral_bottomright
                                                          -integral_bottomleft
                                                          -integral_topright);
        trainingData.push_back(avg_square);
        trainingClasses.push_back(class_mat);
    }

    /* Remove data if buffers are full */
    if(max_size > 0 && trainingClasses.rows > max_size)
    {
        cv::Mat trainingDataCopy = trainingData.clone();
        cv::Mat trainingClassesCopy = trainingClasses.clone();
        std::vector<int> rand_idx;
        for(int i = 0 ; i < trainingClasses.rows ; i++)
            rand_idx.push_back(i);
        std::random_shuffle(rand_idx.begin(),rand_idx.end());
        for(int i = 0 ; i < trainingClasses.rows ; i++)
        {
            trainingData.row(i) = trainingDataCopy.row(rand_idx[i]);
            trainingClasses.row(i) = trainingClassesCopy.row(rand_idx[i]);
        }
        int nsamplesRemoved = trainingClasses.rows - max_size;
        trainingData(cv::Rect(0,
                              nsamplesRemoved,
                              trainingData.cols,
                              trainingData.rows - nsamplesRemoved)).copyTo(trainingData);
        trainingClasses(cv::Rect(0,
                                 nsamplesRemoved,
                                 trainingClasses.cols,
                                 trainingClasses.rows - nsamplesRemoved)).copyTo(trainingClasses);
    }

    return true;
}

/**
 * @brief DataFormater::clear_buffers empty samples and labels buffers
 * @return true anyways
 */
bool DataFormater::clear_buffers()
{
    trainingData = cv::Mat();
    trainingClasses = cv::Mat();
    return true;
}

/**
 * @brief DataFormater::get_buffers returns sample and labels buffers copies
 * @param data a reference to the samples buffer copy
 * @param classes a reference to the labels buffer copy
 * @return true anyways ...
 */
bool DataFormater::get_buffers(cv::Mat &data, cv::Mat &classes)
{
    data = trainingData.clone();
    classes = trainingClasses.clone();
    return true;
}

/**
 * @brief DataFormater::get_buffers_length
 * @return the length of the buffers !
 */
int DataFormater::get_buffers_length()
{
    return trainingClasses.rows;
}
