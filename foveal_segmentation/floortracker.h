/**
 * \file floortracker.h
 * \brief
 * \author Céline Craye
 * \version 0.1
 * \date 10 / 29 / 2015
 *
 * This class is used for tracking the main plane of a scene, typically the floor or a table
 * It requires pcl point clouds as well as RGB images.
 * The core idea behing the tracking is that between two consecutive frames, the floor location
 * and floor color is roughly the same. Thus, instead of re-estimating the plane equation based
 * on the entire point cloud, the tracker selects indices that are both the same color as
 * in previous frame and part of the main plane in the previous frame. Based on these asumptions,
 * the number of pixels to process is significantly decreased, making floor estimation much
 * faster. In addition, it happens sometimes that the main plane differs between two consecutive
 * frames (switching between floor and wall for example). The tracker ensures a better stability
 * in that regard, as long as the successive points of view are close enough.
 * The algorithm relies on the PCL implementation of RANSAC for main plane estimation.
 *
 */

#ifndef FLOORTRACKER_H
#define FLOORTRACKER_H

#include <fstream>

#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/geometry.h>

#include <cv.h>

template <typename T>
class FloorTracker
{
    typedef typename pcl::PointCloud<T> PtCld;
    typedef          pcl::PointIndices  PtInd;

public:
    FloorTracker(float ransac_dist_thresh = 0.05, int color_thresh = 10, bool assume_floor_down = true);
    void update_frame(cv::Mat rgb_frame, const typename PtCld::ConstPtr& cldPtrIn);
    void update_frame(cv::Mat rgb_frame, cv::Mat depth_map);
    void add_prior(cv::Rect prior_square);
    bool track();
    bool reset();
    cv::Mat make_floor_mask();
    cv::Mat make_rgb_diff_mask();
    pcl::ModelCoefficients get_floor_coeff();
    pcl::ModelCoefficients::Ptr get_ptr_floor_coeff();

    ~FloorTracker();
private:
    bool estimate_floor_coeff(cv::Mat mask = cv::Mat());

    typename PtCld::ConstPtr cloud;
    cv::Mat rgb_frame;
    cv::Mat floor_mask;
    cv::Mat old_rgb_frame;
    cv::Mat old_floor_mask;
    bool INIT;
    bool assume_floor_down;
    cv::Rect prior;

    // PCL processing objects
    boost::shared_ptr<pcl::SACSegmentation<T> > ransac_segmenter;
    pcl::ModelCoefficients::Ptr floor_coeffs;
    float ransac_dist_thresh;
    int color_thresh;

};

#endif // FLOORTRACKER_H
