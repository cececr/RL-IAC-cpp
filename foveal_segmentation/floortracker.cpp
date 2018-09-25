/**
 * \file floortracker.cpp
 * \brief
 * \author Céline Craye
 * \version 0.1
 * \date 10 / 29 / 2015
 *
 * For a detailed description of the class, see floortracker.h
 *
 */

#include "floortracker.h"
#include<highgui.h>

template <typename T>
/**
 * @brief FloorTracker<T>::FloorTracker constructor
 * @param ransac_dist_thresh min distance for inliers to belong to the plane (RANSAC parameter)
 * @param color_thresh min pixel color difference between two frames to be considered as same
 * @param assume_floor_down force floor tracking to be only at the bottom part of the image
 */
FloorTracker<T>::FloorTracker(float ransac_dist_thresh, int color_thresh, bool assume_floor_down)
{
    INIT = false;
    this->ransac_dist_thresh = ransac_dist_thresh;
    this->color_thresh = color_thresh;
    this->assume_floor_down = assume_floor_down;

    /* Initialize Floor estimator */
    ransac_segmenter = boost::make_shared<pcl::SACSegmentation<T> >();
    ransac_segmenter->setOptimizeCoefficients(true);
    ransac_segmenter->setModelType(pcl::SACMODEL_PLANE);
    ransac_segmenter->setMethodType(pcl::SAC_RANSAC);
    ransac_segmenter->setDistanceThreshold(ransac_dist_thresh);
    /* Initialize floor coefficients */
    floor_coeffs = boost::make_shared<pcl::ModelCoefficients>();

}

template <typename T>
/**
 * @brief FloorTracker<T>::update_frame update tracker with new input data
 * @param rgb_frame the OpenCV RGB new frame from Kinect data
 * @param cloud the PCL new point cloud from Kinect data
 */
void FloorTracker<T>::update_frame(cv::Mat rgb_frame, const typename PtCld::ConstPtr& cloud)
{
    this->rgb_frame = rgb_frame;
    this->cloud = cloud;
}

template <typename T>
/**
 * @brief FloorTracker<T>::update_frame update tracker with new input data
 * @param rgb_frame the OpenCV RGB new frame from Kinect data
 * @param depth_map the OpenCV depth map new frame from Kinect data
 */
void FloorTracker<T>::update_frame(cv::Mat rgb_frame, cv::Mat depth_map)
{
    this->rgb_frame = rgb_frame;

    /* transform depth map to pcl point cloud */
    PtCld cloud;

    cv::Mat depth_map_int;
    depth_map.convertTo( depth_map_int, CV_32SC1); // convert the image data to float type

    if(!depth_map_int.data){
        std::cerr << "No depth data!!!" << std::endl;
        return;
//        exit(EXIT_FAILURE);
    }

    cloud.width = depth_map_int.cols; //Dimensions must be initialized to use 2-D indexing
    cloud.height = depth_map_int.rows;
    cloud.points.resize(cloud.width*cloud.height);

    register float constant = 1.0f / 525;
    register int centerX = (cloud.width >> 1);
    int centerY = (cloud.height >> 1);
    register int depth_idx = 0;
    for (int v = -centerY; v < centerY; ++v)
    {
        for (register int u = -centerX; u < centerX; ++u, ++depth_idx)
        {
            T& pt = cloud.points[depth_idx];
            pt.z = depth_map_int.at<int>(depth_idx) * 0.001f;
            pt.x = static_cast<float> (u) * pt.z * constant;
            pt.y = static_cast<float> (v) * pt.z * constant;
        }
    }

    cloud.sensor_origin_.setZero ();
    cloud.sensor_orientation_.w () = 0.0f;
    cloud.sensor_orientation_.x () = 1.0f;
    cloud.sensor_orientation_.y () = 0.0f;
    cloud.sensor_orientation_.z () = 0.0f;

    typename PtCld::Ptr ptrCloud = boost::make_shared<PtCld >(cloud);
    this->cloud = ptrCloud;
}

template <typename T>
void FloorTracker<T>::add_prior(cv::Rect prior_square)
{
    prior = prior_square;
}

template <typename T>
/**
 * @brief FloorTracker<T>::track re-estimates main plane coefficients based on previous data
 * @return true if tracking was successful, false otherwise
 */
bool FloorTracker<T>::track()
{
    cv::Mat mask;
    /* If tracker is initialized, get mask for next coefficient estimation */
    if(INIT)
        mask = make_floor_mask() & make_rgb_diff_mask();
    if(prior.area()>0)
    {
        mask = mask = cv::Mat::zeros(cloud->height,cloud->width, CV_8U);
        mask(prior).setTo(255);
        prior = cv::Rect();
    }

    /* estimate new coefficients */
    INIT = estimate_floor_coeff(mask);

    /* update values */
    old_rgb_frame = rgb_frame;
    old_floor_mask = floor_mask;
    return INIT;
}

template <typename T>
/**
 * @brief FloorTracker<T>::reset reset tracker and other parameters
 * @return true anyway
 */
bool FloorTracker<T>::reset()
{
    INIT = false;
    old_rgb_frame = cv::Mat();
    rgb_frame = cv::Mat();
    cloud.reset();
    return true;
}


template <typename T>
/**
 * @brief FloorTracker<T>::get_floor_coeff
 * @return the current floor coefficients.
 */
pcl::ModelCoefficients FloorTracker<T>::get_floor_coeff()
{
    return  *floor_coeffs;
}

template <typename T>
/**
 * @brief FloorTracker<T>::get_ptr_floor_coeff
 * @return the current ptr of floor coefficients.
 */
pcl::ModelCoefficients::Ptr FloorTracker<T>::get_ptr_floor_coeff()
{
    return floor_coeffs;
}

template <typename T>
/**
 * @brief FloorTracker<T>::make_floor_mask find all pixels representing the plane
 * @return a Mat where bright pixels represent the main plane
 */
cv::Mat FloorTracker<T>::make_floor_mask()
{
    /* Initialize variables */
    floor_mask = cv::Mat::zeros(cloud->height,cloud->width, CV_8U);
    Eigen::Vector3f floor_normal(floor_coeffs->values[0],
                                 floor_coeffs->values[1],
                                 floor_coeffs->values[2]);

    /* For each point of the cloud, check distance to plane */
    for (std::size_t k = 0; k < cloud->points.size(); k++)
    {
        Eigen::Vector3f pointPos(cloud->points[k].x,
                                 cloud->points[k].y,
                                 cloud->points[k].z);
        double distPoint = floor_normal.dot(pointPos);
        double distPlane = floor_coeffs->values[3];
        /// add corresponding pixel to mask if distance is small enough
        if (fabs(distPlane + distPoint) < ransac_dist_thresh)
        {
            floor_mask.at<uchar>(k) = 255;
        }
    }
    return floor_mask;
}

template <typename T>
/**
 * @brief make_rgb_diff_mask compares previous and current RGB frames and highlights pixels with similar values
 * @return the mask of pixels with similar values
 */
cv::Mat FloorTracker<T>::make_rgb_diff_mask()
{
    /* Sanity check */
    assert(!rgb_frame.empty());
    assert(!old_rgb_frame.empty());

    /* convert to int */
    rgb_frame.convertTo(rgb_frame,CV_32S);
    old_rgb_frame.convertTo(old_rgb_frame,CV_32S);

    /* threshold frame difference */
    cv::Mat mask_diff = abs(rgb_frame - old_rgb_frame);
    cv::inRange(mask_diff,0,color_thresh,mask_diff);

    return mask_diff;
}

template <typename T>
/**
 * @brief estimate_floor_coeff Apply ransac on a subset of the point cloud and extract the main plane coefficients
 * @param mask the mask of pixels to consider for plane estimation
 * @return true if plane estimation was successful, false otherwise
 */
bool FloorTracker<T>::estimate_floor_coeff(cv::Mat mask)
{
    /* Sanity check */
    assert(cloud != NULL);
    if(mask.empty())
    {
        mask = cv::Mat::ones(cloud->height,cloud->width, CV_8U);
    }
    if(assume_floor_down)
    {
        mask(cv::Rect(0,0,cloud->width, cloud->height/2)).setTo(0);
    }

    /* Initialize objects */
    PtInd::Ptr inliers = boost::make_shared<PtInd>();//(new pcl::PointIndices);
    PtInd::Ptr segIndices = boost::make_shared<PtInd>();

    /* Select indices to filter */
    for(int k = 0 ; k < cloud->size() ; k++)
    {
        if( mask.at<uchar>(k)> 0)
        {
            segIndices->indices.push_back(k);
        }
    }

    /* Segment ! */
    ransac_segmenter->setInputCloud(cloud);
    ransac_segmenter->setIndices(segIndices);
    ransac_segmenter->segment(*inliers, *floor_coeffs);

    /* Make sure plane normal points towards the kinect */
    float sign;
    (floor_coeffs->values[1] > 0) ? sign = -1 : sign = 1;
    floor_coeffs->values[0] = sign*floor_coeffs->values[0];
    floor_coeffs->values[1] = sign*floor_coeffs->values[1];
    floor_coeffs->values[2] = sign*floor_coeffs->values[2];
    floor_coeffs->values[3] = sign*floor_coeffs->values[3];

    /* Create floor mask */
    if (inliers->indices.size() == 0)
    {
        PCL_ERROR ("Could not estimate a planar model");
        std::cerr << "Model coefficients: "
                  << floor_coeffs->values[0] << " "
                  << floor_coeffs->values[1] << " "
                  << floor_coeffs->values[2] << " "
                  << floor_coeffs->values[3] << std::endl;
        return false;
    }
    return true;
}

template <typename T>
/**
 * @brief FloorTracker<T>::~FloorTracker DESTRUCTOOOR!!
 */
FloorTracker<T>::~FloorTracker()
{
}


template class FloorTracker<pcl::PointXYZRGB>;
template class FloorTracker<pcl::PointXYZ>;

