/**
 * \file pt_cld_segmentation.hpp
 * \author Louis-Charles Caron, modified by Céline Craye
 * \version 0.2
 * \date 2 nov. 2015
 * \brief PtCldSegmentation class.
 *         Finds and segments object in point clouds.
 *
 * Please check pt_cld_segmentation.h file for usage and description details.
 */

#include "pt_cld_segmentation.h"
#include <iostream>
#include <time.h>
#include <highgui.h>
#include <map>
#include <boost/iterator/counting_iterator.hpp>

template <typename T>
/**
 * @brief PtCldSegmentation<T>::PtCldSegmentation default constructor. See header for parameters description
 * @param floor_equation_filename  the name of the file containing the floor equation.
 *        If the file does not exist or is unreadable, an estimate of the floor plane is done automatically and written on disk.
 *        Alternatively, if you provide a floor equation with SetFloorEquation(), this file won't be used.
 * @param ransac_floor_dist_thresh the tolerance threshold (in meters) for the RANSAC floor estimation.
 *        Useful only if function EstimateMainPlaneCoefs() is called
 * @param normal_estim_max_depth distance threshold (in meters) between a point and its neighborhood to provide a normal estimation
 * @param normal_estim_smooth_size radius (in pixels) of the point neighborhood for normal estimation
 * @param max_depth_visibility max distance (in meters) between a point and the kinect to be processed
 * @param max_obj_dist_to_floor max distance (in meters) between a point and the floor to be processed
 * @param min_obj_dist_to_floor min distance (in meters) between a point and the theoretical floor equation to be processed
 * @param floor_angular_tolerance min value of the (normalized, between 0 and 1) dot product for a point to be part of the floor
 * @param ransac_wall_dist_thresh the tolerance threshold (in meters) for the RANSAC wall estimation.
 * @param min_wall_diag_size min diagonal size (in meters) for a plane to be detected as a wall
 * @param wall_angular_tolerance max value of the (normalized, between 0 and 1) dot product for a point to be part of the wall
 * @param min_dist_btw_obj max value of the (normalized, between 0 and 1) dot product for a point to be part of the wall
 *        This parameter is useful only if merge_clusters_option = true
 * @param obj_min_diag_size minimum diagonal size (in meters) for an object to be detected
 * @param obj_max_diag_size maximum diagonal size (in meters) for an object to be detected
 * @param obj_at_border_pix_tolerance number of pixels considered at the border of the FoV to detect objects at the border of it
 * @param voxel_leaf_size size (in meters) of the voxel leafs. Smaller values provide more accurate but slower processing.
 * @param max_obj_bottom_dist_to_floor max distance (in meters) between the bottom of a blob and the floor to be consider as an object.
 * @param min_obj_pixel_size min number of pixels in the image for a blob to be considered as an object
 * @param use_tracking_option whether we want to use blob tracking (true) or not (false)
 * @param merge_clusters_option whether we want to merge blobs by projecting them on the floor plane (true) or not (false)
 * @param wrong_floor_thresh between 0 and 1, tolerance to detect wrong floor estimation. 0 = always ok, 1 = always wrong.
 */
PtCldSegmentation<T>::PtCldSegmentation(std::string floor_equation_filename = "floor.dat",
                                        float ransac_floor_dist_thresh = 0.05,
                                        float normal_estim_max_depth = 0.02,
                                        float normal_estim_smooth_size = 10.0,
                                        float max_depth_visibility = 4.0,
                                        float max_obj_dist_to_floor = 2.0,
                                        float min_obj_dist_to_floor = 0.1,
                                        float floor_angular_tolerance = 0.9,
                                        float ransac_wall_dist_thresh = 0.1,
                                        float min_wall_diag_size = 1.5,
                                        float wall_angular_tolerance = 0.2,
                                        float min_dist_btw_obj = 0.06,
                                        float obj_min_diag_size = 0.2,
                                        float obj_max_diag_size = 1.3,
                                        int obj_at_border_pix_tolerance = 30,
                                        float voxel_leaf_size = 0.03,
                                        float max_obj_bottom_dist_to_floor = 1.3,
                                        int min_obj_pixel_size = 200,
                                        bool use_tracking_option = false,
                                        bool merge_clusters_option = false,
                                        float wrong_floor_thresh = 0.3)
    : debug_(0),
      floor_equation_filename_(floor_equation_filename),
      ransac_floor_dist_thresh_(ransac_floor_dist_thresh),
      normal_estim_max_depth_(normal_estim_max_depth),
      normal_estim_smooth_size_(normal_estim_smooth_size),
      max_depth_visibility_(max_depth_visibility),
      max_obj_dist_to_floor_(max_obj_dist_to_floor),
      min_obj_dist_to_floor_(min_obj_dist_to_floor),
      floor_angular_tolerance_(floor_angular_tolerance),
      ransac_wall_dist_thresh_(ransac_wall_dist_thresh),
      min_wall_diag_size_(min_wall_diag_size),
      wall_angular_tolerance_(wall_angular_tolerance),
      min_dist_btw_obj_(min_dist_btw_obj),
      obj_min_diag_size_(obj_min_diag_size),
      obj_max_diag_size_(obj_max_diag_size),
      obj_at_border_pix_tolerance_(obj_at_border_pix_tolerance),
      voxel_leaf_size_(voxel_leaf_size),
      max_obj_bottom_dist_to_floor_(max_obj_bottom_dist_to_floor),
      min_obj_pixel_size_(min_obj_pixel_size),
      use_tracking_option_(use_tracking_option),
      merge_clusters_option_(merge_clusters_option),
      wrong_floor_thresh_(wrong_floor_thresh),
      floor_equation_(boost::make_shared<pcl::ModelCoefficients>()),
      has_floor_equation_(false)
{
    // Set parameters of PCL processing objects
    SetParams();
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::~PtCldSegmentation default destructor
 */
PtCldSegmentation<T>::~PtCldSegmentation()
{
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::SetParams Setting main parameters
 */
void PtCldSegmentation<T>::SetParams()
{
    // Load cam models
    CameraInfoLight msg("RGB");
    rgb_cam_model.fromCameraInfo(msg);
    CameraInfoLight msg_depth("depth");
    depth_cam_model.fromCameraInfo(msg_depth);

    // Load stored floor plane coefs
    floor_equation_->values.resize(4);
    std::ifstream coefs_file(this->floor_equation_filename_.c_str(), std::ios::in | std::ios::binary);

    if (coefs_file.is_open())
    {
        coefs_file.read( (char*) &this->floor_equation_->values[0],
                           sizeof(this->floor_equation_->values[0]) );
        coefs_file.read( (char*) &this->floor_equation_->values[1],
                           sizeof(this->floor_equation_->values[1]) );
        coefs_file.read( (char*) &this->floor_equation_->values[2],
                           sizeof(this->floor_equation_->values[2]) );
        coefs_file.read( (char*) &this->floor_equation_->values[3],
                           sizeof(this->floor_equation_->values[3]) );
        coefs_file.close();
        this->floor_equation_eigen_[0] = this->floor_equation_->values[0];
        this->floor_equation_eigen_[1] = this->floor_equation_->values[1];
        this->floor_equation_eigen_[2] = this->floor_equation_->values[2];
        this->floor_equation_eigen_[3] = this->floor_equation_->values[3];
        has_floor_equation_ = true;
    }

    // Ransac floor detector parameters
    floor_ransac_segmenter_.reset(new pcl::SACSegmentation<T>);
    floor_ransac_segmenter_->setOptimizeCoefficients(true);
    floor_ransac_segmenter_->setModelType(pcl::SACMODEL_PLANE);
    floor_ransac_segmenter_->setMethodType(pcl::SAC_RANSAC);
    floor_ransac_segmenter_->setDistanceThreshold(this->ransac_floor_dist_thresh_);

    // Maximum distance cropbox filter
    visibility_crop_box_ = boost::make_shared<pcl::CropBox<T> >();
    Eigen::Vector4f crop_box_max_bounds;
    crop_box_max_bounds[0] = max_depth_visibility_;
    crop_box_max_bounds[1] = max_depth_visibility_;
    crop_box_max_bounds[2] = max_depth_visibility_;
    crop_box_max_bounds[3] = 1.;
    visibility_crop_box_->setMax(crop_box_max_bounds);
    Eigen::Vector4f crop_box_min_bounds;
    crop_box_min_bounds[0] = -max_depth_visibility_;
    crop_box_min_bounds[1] = -max_depth_visibility_;
    crop_box_min_bounds[2] = -max_depth_visibility_;
    crop_box_min_bounds[3] = 1.;
    visibility_crop_box_->setMin(crop_box_min_bounds);

    // Voxel grid
    voxel_grid_ = boost::make_shared<pcl::VoxelGrid<T> >();
    voxel_grid_->setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);

    // Normal estimation based on integral image (not used anymore)
    normal_estimator_= boost::make_shared<pcl::IntegralImageNormalEstimation<T, pcl::Normal> >();
    normal_estimator_->setNormalEstimationMethod(normal_estimator_->AVERAGE_3D_GRADIENT);
    normal_estimator_->setMaxDepthChangeFactor(normal_estim_max_depth_);
    normal_estimator_->setNormalSmoothingSize(normal_estim_smooth_size_);

    // Cropbox filters for plane estimation (not used anymore)
    floor_distance_crop_box_= boost::make_shared<pcl::CropBox<T> >();
    Eigen::Vector4f floor_crop_box_max_bounds;
    floor_crop_box_max_bounds[0] = std::numeric_limits<float>::max();
    floor_crop_box_max_bounds[1] = max_obj_dist_to_floor_;
    floor_crop_box_max_bounds[2] = std::numeric_limits<float>::max();
    floor_crop_box_max_bounds[3] = 1.;
    floor_distance_crop_box_->setMax(floor_crop_box_max_bounds);
    Eigen::Vector4f floor_crop_box_min_bounds;
    floor_crop_box_min_bounds[0] = -std::numeric_limits<float>::max();
    floor_crop_box_min_bounds[1] = min_obj_dist_to_floor_;
    floor_crop_box_min_bounds[2] = -std::numeric_limits<float>::max();
    floor_crop_box_min_bounds[3] = 1.;
    floor_distance_crop_box_->setMin(floor_crop_box_min_bounds);

    // Walls RANSAC
    wall_ransac_segmenter_= boost::make_shared<pcl::SACSegmentation<T> >();
    wall_ransac_segmenter_->setModelType(pcl::SACMODEL_PLANE);
    wall_ransac_segmenter_->setMethodType(pcl::SAC_RANSAC);
    wall_ransac_segmenter_->setOptimizeCoefficients(true);
    wall_ransac_segmenter_->setDistanceThreshold(ransac_wall_dist_thresh_);
    wall_ransac_segmenter_->setMaxIterations(100);

    // Projection
    floor_projector_= boost::make_shared<pcl::ProjectInliers<T> >();
    floor_projector_->setModelType(pcl::SACMODEL_PLANE);
    floor_projector_->setCopyAllData(true);

    // Euclidean clustering
    clustering_algo_= boost::make_shared<pcl::EuclideanClusterExtraction<T> >();
    typename pcl::search::KdTree<T>::Ptr treePtr = boost::make_shared<pcl::search::KdTree<T> >();
    clustering_algo_->setSearchMethod(treePtr);
    clustering_algo_->setClusterTolerance(min_dist_btw_obj_);
    intECMinSize_ = 0.5*(obj_min_diag_size_/voxel_leaf_size_)
                       *(obj_min_diag_size_/voxel_leaf_size_);
    intECMaxSize_ =     (obj_max_diag_size_/voxel_leaf_size_)
                       *(obj_max_diag_size_/voxel_leaf_size_);
    clustering_algo_->setMinClusterSize(intECMinSize_);
    clustering_algo_->setMaxClusterSize(intECMaxSize_);

}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::SetDebugCloud
 *        set debug cloud value. 0: no debug cloud, 1: full debug cloud
 * @param debug_cloud
 */
void PtCldSegmentation<T>::SetDebugCloud(int debug_cloud)
{
    this->debug_ = debug_cloud;
}


template <typename T>
/**
 * @brief PtCldSegmentation<T>::SetFloorEquation
 *        sets floor equation and makes it compatible with processing steps
 * @param floor_equation the four coefficients of the floor equation
 */
void PtCldSegmentation<T>::SetFloorEquation(pcl::ModelCoefficients::Ptr floor_equation)
{
    floor_equation_ = floor_equation;
    // Make sur plane normal is correctly oriented
    if(floor_equation_->values[2]>0)
    {
        floor_equation_->values[0] = - floor_equation_->values[0];
        floor_equation_->values[1] = - floor_equation_->values[1];
        floor_equation_->values[2] = - floor_equation_->values[2];
        floor_equation_->values[3] = - floor_equation_->values[3];
    }
    floor_equation_eigen_ = Eigen::Vector4f(floor_equation_->values.data());
    has_floor_equation_ = true;
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::SetInputCloud sets new input point cloud
 * @param input_cloud the input point cloud
 */
void PtCldSegmentation<T>::SetInputCloud(const typename PtCld::ConstPtr &input_cloud)
{
    kinect_cloud_ = input_cloud;
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::setInputCloud sets new input point cloud from depth map
 * @param depth_map the depth map to transform
 */
void PtCldSegmentation<T>::setInputCloudFromDepth(cv::Mat & depth_map)
{
    pcl::PointCloud<T> cloud;

    cv::Mat depth_map_int;// = cv::Mat(depth_map.size(), CV_32SC1);
    depth_map.convertTo( depth_map_int, CV_32SC1); // convert the image data to int type

    if(!depth_map_int.data){
        std::cerr << "No depth data!!!" << std::endl;
        typename PtCld::Ptr ptrCloud = boost::make_shared<PtCld>();
        copyPointCloud(cloud, *ptrCloud);
        kinect_cloud_ = ptrCloud;
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
    typename PtCld::Ptr ptrCloud = boost::make_shared<PtCld>();
    copyPointCloud(cloud, *ptrCloud);
    kinect_cloud_ = ptrCloud;
}


template <typename T>
/**
 * @brief PtCldSegmentation<T>::EstimateMainPlaneCoefs
 *          in the case where no floor equation is provided, estimate one based
 *          on the main point cloud's plane.
 * @return true if estimation was successful, false otherwise
 */
bool PtCldSegmentation<T>::EstimateMainPlaneCoefs()
{
    if (this->kinect_cloud_ != NULL)
    {
        PtInd::Ptr tmp = boost::make_shared<PtInd>();

        floor_ransac_segmenter_->setInputCloud(this->kinect_cloud_);
        floor_ransac_segmenter_->segment(*tmp, *this->floor_equation_);
        // Make sure plane normal points towards the kinect
        if (this->floor_equation_->values[1] > 0)
        {
            this->floor_equation_eigen_[0] = -this->floor_equation_->values[0];
            this->floor_equation_eigen_[1] = -this->floor_equation_->values[1];
            this->floor_equation_eigen_[2] = -this->floor_equation_->values[2];
            this->floor_equation_eigen_[3] = -this->floor_equation_->values[3];
        }
        else
        {
            this->floor_equation_eigen_[0] = this->floor_equation_->values[0];
            this->floor_equation_eigen_[1] = this->floor_equation_->values[1];
            this->floor_equation_eigen_[2] = this->floor_equation_->values[2];
            this->floor_equation_eigen_[3] = this->floor_equation_->values[3];
        }

        std::cout << "Main plane coefficients: "
                  << this->floor_equation_eigen_[0] << ", "
                  << this->floor_equation_eigen_[1] << ", "
                  << this->floor_equation_eigen_[2] << ", "
                  << this->floor_equation_eigen_[3] << "." << std::endl;
        // Store floor plane coefs
        std::ofstream coefs_file(this->floor_equation_filename_.c_str(),
                                 std::ios::out | std::ios::binary );

        std::cout << floor_equation_filename_ << " opened ? "<< coefs_file.is_open() << std::endl;
        if (coefs_file.is_open())
        {
            coefs_file.write( (const char*) &this->floor_equation_->values[0],
                                sizeof(this->floor_equation_->values[0]) );
            coefs_file.write( (const char*) &this->floor_equation_->values[1],
                                sizeof(this->floor_equation_->values[1]) );
            coefs_file.write( (const char*) &this->floor_equation_->values[2],
                                sizeof(this->floor_equation_->values[2]) );
            coefs_file.write( (const char*) &this->floor_equation_->values[3],
                                sizeof(this->floor_equation_->values[3]) );
            coefs_file.close();
            std::cout << "Stored in file: " << this->floor_equation_filename_.c_str() << std::endl;
        }

        has_floor_equation_ = true;
        return true;
    }
    else
        return false;
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::Segment
 *        Core segmentation method. Determines objects in the input point cloud and produce
 *        output objects such as mask, point clouds, normals and bounding boxes
 */
bool PtCldSegmentation<T>::Segment()
{
    if(kinect_cloud_->empty())
    {
        segmentationMask = cv::Mat();
    }
    /* Initialize variables */
    if (debug_ == 1)
        debug_cloud_ = boost::make_shared<PtCld>(*kinect_cloud_);

    segmentationMask = cv::Mat::zeros(kinect_cloud_->height,kinect_cloud_->width ,CV_8UC3);
    segmentationMask.setTo(GetSegColor(FAR));

    /* Reset buffers */
    blobs_2d_bboxes_.clear();
    obj_2d_bboxes_.clear();
    walls_list_.clear();
    obj_voxels_.clear();
    obj_indices_.clear();
    bad_segmentation = false;

    // compute normals on the whole cloud
    NormCld::Ptr normals = boost::make_shared<NormCld>();
    ComputeNormals(kinect_cloud_, normals);

    // Remove points lying too far from the kinect
    PtInd::Ptr close_pts_indices = boost::make_shared<PtInd>();
    RemoveMaxDistPoints(kinect_cloud_, close_pts_indices);
    if (debug_ == 2)
        debug_cloud_ = boost::make_shared<PtCld>(*kinect_cloud_, close_pts_indices->indices);
    SetMaskIndices(close_pts_indices,GetSegColor(UNAVAILABLE));

    // Remove unavailable data
    PtInd::Ptr available_pts_indices = boost::make_shared<PtInd>();
    RemoveUnavailable(kinect_cloud_, close_pts_indices, available_pts_indices);
    SetMaskIndices(available_pts_indices,GetSegColor(FLOOR));

    // Remove floor plane
    PtInd::Ptr no_floor_pts_indices = boost::make_shared<PtInd>();
    RemoveFloor(kinect_cloud_, available_pts_indices, no_floor_pts_indices, normals);
    if (debug_ == 3)
        debug_cloud_ = boost::make_shared<PtCld>(*kinect_cloud_, no_floor_pts_indices->indices);
    SetMaskIndices(no_floor_pts_indices,GetSegColor(WALL));

    // Remove walls
    PtInd::Ptr no_wall_pts_indices = boost::make_shared<PtInd>();
    RemoveWalls(kinect_cloud_, no_floor_pts_indices, no_wall_pts_indices, normals);
    if (debug_ == 4)
        debug_cloud_ = boost::make_shared<PtCld>(*kinect_cloud_, no_wall_pts_indices->indices);
    SetMaskIndices(no_wall_pts_indices,GetSegColor(NO_CLUSTER));

    // image-based blob finder
    std::vector<PtInd::Ptr> img_blobs_indices;
    ImageBasedClustering(kinect_cloud_, no_wall_pts_indices, normals, img_blobs_indices);

    PtInd::Ptr all_blobs_indices = boost::make_shared<PtInd>();
    ConcatenateIndices(img_blobs_indices,all_blobs_indices);
    if (debug_ == 5)
        debug_cloud_ = boost::make_shared<PtCld>(*kinect_cloud_, all_blobs_indices->indices);
    SetMaskIndices(all_blobs_indices,GetSegColor(FLOOR_ARTIFACTS));
    std::vector<int> kept_idx(boost::counting_iterator<int>(0),
                              boost::counting_iterator<int>(img_blobs_indices.size()));

    // Voxelize
    typename PtCld::Ptr img_blobs_cloud_vox = boost::make_shared<PtCld>();
    std::vector<PtInd::Ptr> img_blobs_indices_vox ;
    Voxelize(kinect_cloud_, img_blobs_indices, img_blobs_cloud_vox, img_blobs_indices_vox);

    // remove floor artifacts
    std::vector<PtInd::Ptr> no_floor_artifact_indices ;
    std::vector<int> no_floor_artifact_kept_idx;
    RemoveFloorArtifacts(img_blobs_cloud_vox,
                         img_blobs_indices_vox,
                         no_floor_artifact_indices,
                         no_floor_artifact_kept_idx);
    UpdateKeptIdx(kept_idx,no_floor_artifact_kept_idx);
    SetMaskIndices(img_blobs_indices, kept_idx, GetSegColor(TOUCH_BORDER));

    // Remove blobs too close to a border of the image, or the max dist box
    std::vector<PtInd::Ptr> no_border_blobs_indices;
    std::vector<int> no_border_kept_idx;
    RemoveClustersAtBorder(img_blobs_cloud_vox,
                           no_floor_artifact_indices,// was img_blobs_indices_vox
                           no_border_blobs_indices,
                           no_border_kept_idx);
    UpdateKeptIdx(kept_idx,no_border_kept_idx);
    SetMaskIndices(img_blobs_indices, kept_idx, GetSegColor(LARGE_CLUSTER));


    // Remove too large and too high blobs
    std::vector<pcl::PointIndices::Ptr> small_enough_blobs_indices;
    std::vector<int> small_enough_kept_idx;
    RemoveLargeClusters(img_blobs_cloud_vox,
                        no_border_blobs_indices,
                        small_enough_blobs_indices,
                        small_enough_kept_idx);
    UpdateKeptIdx(kept_idx,small_enough_kept_idx);
    SetMaskIndices(img_blobs_indices,kept_idx,GetSegColor(WALL_ARTIFACTS));
    if (debug_ == 6)
    {
        PtInd::Ptr concat_blobs_indices = boost::make_shared<PtInd>();
        ConcatenateIndices(small_enough_blobs_indices,concat_blobs_indices);
        debug_cloud_ = boost::make_shared<PtCld>(*img_blobs_cloud_vox, concat_blobs_indices->indices);
    }

    // Remove wall artifacts
    std::vector<pcl::PointIndices::Ptr> no_artifacts_blobs_indices;
    std::vector<int> no_artifacts_kept_idx;
    RemoveWallArtifacts(img_blobs_cloud_vox,
                        small_enough_blobs_indices,
                        no_artifacts_blobs_indices,
                        no_artifacts_kept_idx);
    UpdateKeptIdx(kept_idx,no_artifacts_kept_idx);
    SetMaskIndices(img_blobs_indices,kept_idx,GetSegColor(OBJECT));
    if (debug_ == 7)
    {
        PtInd::Ptr concat_blobs_indices = boost::make_shared<PtInd>();
        ConcatenateIndices(no_artifacts_blobs_indices,concat_blobs_indices);
        debug_cloud_ = boost::make_shared<PtCld>(*img_blobs_cloud_vox, concat_blobs_indices->indices);
    }

    // "Unvoxelize" (just get back to the original indices)
    obj_indices_.clear();
    obj_indices_.reserve(kept_idx.size());
    for(int i = 0 ; i < kept_idx.size() ; i++)
        obj_indices_.push_back(img_blobs_indices[kept_idx[i]]);

    // If projection clustering is enabled, Do So !
    if(merge_clusters_option_ && !obj_indices_.empty())
    {
        PtInd::Ptr concat_objects_indices = boost::make_shared<PtInd>();
        ConcatenateIndices(obj_indices_,concat_objects_indices);
        typename PtCld::Ptr obj_cloud_vox = boost::make_shared<PtCld>();
        typename PtCld::Ptr proj_objects_cloud_vox = boost::make_shared<PtCld>();
        Voxelize(kinect_cloud_,concat_objects_indices, obj_cloud_vox);
        std::vector<PtInd::Ptr> proj_objects_indices_vox;
        ProjectionBasedClustering(obj_cloud_vox,
                                  proj_objects_cloud_vox,
                                  proj_objects_indices_vox);
        obj_indices_.clear();
        obj_indices_.reserve(proj_objects_indices_vox.size());
        UnVoxelize(proj_objects_indices_vox, obj_indices_);
    }

    // Make output objects
    MakeObjectsPtCldAndNorm(kinect_cloud_, normals , obj_indices_);
    MakeObjectsBBsAndPos(kinect_cloud_, obj_indices_, img_blobs_indices);
    if(debug_ == 1)
        MakeColorDebugCloud();

    // Updates for tracking
    if(use_tracking_option_)
    {
        oldSegmentationMask = GetSegmentationMask();
    }
    if(bad_segmentation)
    {
        oldSegmentationMask = cv::Mat();
    }

    return !bad_segmentation ;
}


template <typename T>
/**
 * @brief PtCldSegmentation<T>::RemoveMaxDistPoints
 *        Find points of the cloud that are further than a certain distance
 *        and remove them from the indices list
 * @param cloud_in the input point cloud
 * @param indices_out the list of indices that are close enough to the kinect
 */
void PtCldSegmentation<T>::RemoveMaxDistPoints(const typename PtCld::ConstPtr &cloud_in,
                                               PtInd::Ptr& indices_out)
{
    visibility_crop_box_->setInputCloud(cloud_in);
    visibility_crop_box_->filter(indices_out->indices);
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::RemoveUnavailable
 *        Find points of the cloud that are numerically irrelevent
 *        and remove them from the indices list.
 * @param cloud_in the input point cloud
 * @param indices_in the list of indices to process
 * @param indices_out the list of indices that are not irrelevent
 */
void PtCldSegmentation<T>::RemoveUnavailable(const typename PtCld::ConstPtr &cloud_in,
                                             const PtInd::ConstPtr &indices_in,
                                             PtInd::Ptr &indices_out)
{
    for(int j = 0 ; j < indices_in->indices.size() ; j++)
    {
        int idx = indices_in->indices[j];
        if(cloud_in->points[idx].z > 0)
        {
            indices_out->indices.push_back(idx);
        }
    }
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::ComputeNormals Compute point cloud normals
 * @param cloud_in the input point cloud
 * @param normals the corresponding normal point cloud
 */
void PtCldSegmentation<T>::ComputeNormals(const typename PtCld::ConstPtr& cloud_in,
                                          NormCld::Ptr& normals)
{
    normal_estimator_->setInputCloud(cloud_in);
    normal_estimator_->compute(*normals);
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::RemoveFloor
 *        based on main plane estimation, filter out elements of the cloud that belong to this plane.
 * @param cloud_in the input point cloud
 * @param indices_in the list of indices to process
 * @param indices_out the list of indices that do not belong to the plane.
 * @param normals the cloud of normal vectors
 */
void PtCldSegmentation<T>::RemoveFloor(const typename PtCld::ConstPtr cloud_in,
                                       const PtInd::ConstPtr& indices_in,
                                       PtInd::Ptr& indices_out,
                                       NormCld::Ptr& normals)
{

    // Find points whose position is close to floor plane estimate
    Eigen::Vector3f floor_normal_eigen(floor_equation_->values[0],
                                         floor_equation_->values[1],
                                         floor_equation_->values[2]);
    PtInd::Ptr floor_indices = boost::make_shared<PtInd>();
    floor_indices->indices.resize(indices_in->indices.size());
    std::size_t nb_floor_pt = 0, nb_consistent  = 0;
    for (std::size_t i = 0; i < indices_in->indices.size(); i++)
    {
        Eigen::Vector3f pointPos(cloud_in->points[indices_in->indices[i]].x,
                                 cloud_in->points[indices_in->indices[i]].y,
                                 cloud_in->points[indices_in->indices[i]].z);
        // Distance from kinect to the point, along the floor plane's normal
        double distPoint = floor_normal_eigen.dot(pointPos);
        // Distance from kinect to floor, along the floor plane's normal
        double distPlane = floor_equation_->values[3];
        if (fabs(distPlane + distPoint) < min_obj_dist_to_floor_)// floor_distance_tolerance
        {
            nb_floor_pt++;
            /// check if normal is relevent with floor plane.
            Eigen::Vector3f pointNormal(normals->points[indices_in->indices[i]].normal_x,
                                        normals->points[indices_in->indices[i]].normal_y,
                                        normals->points[indices_in->indices[i]].normal_z);
            double dotFloorPoint = fabs(floor_normal_eigen.dot(pointNormal));
            if (dotFloorPoint > floor_angular_tolerance_ )
            {
                floor_indices->indices[nb_floor_pt] = indices_in->indices[i];
                nb_consistent ++;
            }
        }
    }

    // if floor plane is not consistent with the geometry, set segmentation as unreliable.
    if((float)nb_consistent / (float)nb_floor_pt < wrong_floor_thresh_)
    {
        bad_segmentation = true;
    }

    floor_indices->indices.resize(nb_floor_pt);

    // Set CropBox filter with estimated plane coefficients
    Eigen::Vector3f rotation_axis;
    rotation_axis[0] = atan2(static_cast<double>(floor_equation_eigen_[2]),
                      sqrt(pow(static_cast<double>(floor_equation_eigen_[0]),2)
                          +pow(static_cast<double>(floor_equation_eigen_[1]),2)));
    rotation_axis[1] = 0;
    rotation_axis[2] = atan2(static_cast<double>(-floor_equation_eigen_[0]),
                      static_cast<double>(floor_equation_eigen_[1]));
    floor_distance_crop_box_->setRotation(rotation_axis);
    Eigen::Vector3f translation_vect;
    translation_vect[0] = -floor_equation_eigen_[0]*floor_equation_eigen_[3];
    translation_vect[1] = -floor_equation_eigen_[1]*floor_equation_eigen_[3];
    translation_vect[2] = -floor_equation_eigen_[2]*floor_equation_eigen_[3];
    floor_distance_crop_box_->setTranslation(translation_vect);

    // Remove plane
    floor_distance_crop_box_->setInputCloud(cloud_in);
    floor_distance_crop_box_->setIndices(indices_in);
    floor_distance_crop_box_->filter(indices_out->indices);
}


template <typename T>
/**
 * @brief PtCldSegmentation<T>::RemoveWalls
 *        based on main plane estimation, find elements that are perpendicular and big enough.
 *        These should be walls and are filtered out.
 * @param cloud_in the input point cloud
 * @param indices_in the list of indices to process
 * @param indices_out the list of indices that do not belong to walls.
 * @param normals the cloud of normal vectors
 */
void PtCldSegmentation<T>::RemoveWalls(const typename PtCld::ConstPtr cloud_in,
                                       const PtInd::ConstPtr& indices_in,
                                       pcl::PointIndices::Ptr &indices_out,
                                       NormCld::Ptr& normals)
{
    /// Celine's version of removeWalls
    /// This method is designed for fast computation and uses a few "weird" tricks ...
    ///
    /// Instead of running ransac on the whole frame, we run several successive times on
    /// small portions of it and extend inliers to the whole frame. The insight about doing so
    /// is that walls are usually compact and very consistent portions of the frame.
    /// More precisely, we split the frame into 16 equally spaced sub-frames. One by one, we
    /// apply ransac on each sub-frame, based on the indices provided by indices_in.
    /// If any, the main plane is compared with the floor equation and kept only if orthogonal
    /// to this one. If so, we find the correspond inliers in the whole frame. If the inlier set
    /// matches the wall expected dimensions and has a contact with a border, it is considered
    /// as wall.
    ///
    /// To further speed up computation, we do not process the half above part of the frame that
    /// is less likely to containt walls, and we iteratively remove points that have been found
    /// to be walls at a previous iteration.

    // Initialize variables
    wall_ransac_segmenter_->setInputCloud(cloud_in);
    int x_step = cloud_in->width/4;
    int y_step = cloud_in->height/4;
    cv::Mat remove_indices = cv::Mat::zeros(1,kinect_cloud_->width*kinect_cloud_->height, CV_8U);
    PtInd::Ptr wall_inliers = boost::make_shared<PtInd>();

    // divide frame into sub frames
    for(int i =  0 ; i < 4 ; i++)
        for(int j = 0 ; j < 2 ; j++)
        {
            // get indices in image cluster
            PtInd::Ptr ransac_input_indices = boost::make_shared<PtInd>();
            for(int k = 0 ; k < indices_in->indices.size() ; k++)
            {
                int idx = indices_in->indices[k];
                int y = floor(idx/kinect_cloud_->width);
                int x = idx % kinect_cloud_->width;
                if(   x >= x_step*i  && x < x_step*(i+1)
                   && y >= y_step*j  && y < y_step*(j+1))
                {
                    if(remove_indices.at<uchar>(indices_in->indices[i]) == 0)
                    {
                        ransac_input_indices->indices.push_back(idx);
                    }
                }
            }
            if(ransac_input_indices->indices.size() < x_step*y_step/4)
                continue;

            // do ransac
            pcl::ModelCoefficients::Ptr plane_coeff = boost::make_shared<pcl::ModelCoefficients>();
            PtInd::Ptr plane_inliers = boost::make_shared<PtInd>();
            wall_ransac_segmenter_->setIndices(ransac_input_indices);
            wall_ransac_segmenter_->segment(*plane_inliers, *plane_coeff);

            if (plane_coeff->values.size() == 0)
                continue;

            // check if plane is perpendicular
            Eigen::Vector3f plane_normal(plane_coeff->values[0],
                                           plane_coeff->values[1],
                                           plane_coeff->values[2]);
            Eigen::Vector3f floor_normal(floor_equation_eigen_[0],
                                           floor_equation_eigen_[1],
                                           floor_equation_eigen_[2]);
            float dotFloorPlane = fabs(floor_normal.dot(plane_normal));
            if (dotFloorPlane < wall_angular_tolerance_)
            {
                // find and remove inliers in the whole image
                wall_inliers->indices.clear();
                for (std::size_t k = 0; k < indices_in->indices.size(); k++)
                {
                    Eigen::Vector3f pointPos(cloud_in->points[indices_in->indices[k]].x,
                                             cloud_in->points[indices_in->indices[k]].y,
                                             cloud_in->points[indices_in->indices[k]].z);
                    double distPoint = plane_normal.dot(pointPos);
                    double distPlane = plane_coeff->values[3];
                    Eigen::Vector3f pointNormal(normals->points[indices_in->indices[k]].normal_x,
                                                normals->points[indices_in->indices[k]].normal_y,
                                                normals->points[indices_in->indices[k]].normal_z);
                        double dotFloorPoint = fabs(plane_normal.dot(pointNormal));
                    if (fabs(distPlane + distPoint) < ransac_wall_dist_thresh_
                            && (dotFloorPoint > floor_angular_tolerance_ ) )
                    {

                        wall_inliers->indices.push_back(indices_in->indices[k]);
                    }
                }

                // backproject inliers and get biggest blob
                cv::Mat mask = cv::Mat::zeros(cloud_in->height,cloud_in->width,CV_8U);
                for(int m  = 0 ; m < wall_inliers->indices.size() ; m++)
                {
                    mask.at<uchar>(wall_inliers->indices[m]) = 255;
                }

                std::vector<std::vector<cv::Point> > contours;
                cv::findContours(mask,contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
                double biggest_contour = 0;
                double biggest_index = 0;
                for(int m  = 0 ; m < contours.size() ; m++)
                {
                    double area =cv::contourArea(contours[m], false);
                    if(area> biggest_contour)
                    {
                        biggest_contour = area;
                        biggest_index = m;
                    }
                }
                mask.setTo(0);
                cv::drawContours(mask, contours, biggest_index, 255, CV_FILLED);
                wall_inliers->indices.clear();
                for(int m = 0 ; m < mask.rows*mask.cols ; m++)
                {
                    if(mask.at<uchar>(m) > 0)
                        wall_inliers->indices.push_back(m);
                }

                // check that bounding box of the wall candidate is greater than expected diagonal
                Eigen::Vector4f posMin, posMax, diag;
                pcl::getMinMax3D(*cloud_in, *wall_inliers, posMin, posMax);
                diag = posMax-posMin;
                if(sqrt(diag[0]*diag[0] + diag[1]*diag[1] + diag[2]*diag[2] ) > min_wall_diag_size_)
                {
                    // a wall must touch at least one border
                    pcl::PointIndices borderIndices;
                    ComputeBorderIndices(cloud_in, wall_inliers, borderIndices);
                    if(borderIndices.indices.size()>0)
                    {
                        for(size_t k = 0 ; k < wall_inliers->indices.size() ; k++)
                        {
                            remove_indices.at<uchar>(wall_inliers->indices[k]) = 1;
                        }
                        // store equation
                        walls_list_.push_back(plane_coeff);
                    }
                }
            }
        }

    // fill indices_out vector
    PtInd::Ptr non_wall_indices = boost::make_shared<PtInd>();
    non_wall_indices->indices.resize(
                static_cast<std::size_t>(remove_indices.cols - countNonZero(remove_indices)));
    int ct = 0;
    for(int i = 0 ; i < indices_in->indices.size() ; i++)
    {
        if(remove_indices.at<uchar>(indices_in->indices[i]) == 0)
        {
            non_wall_indices->indices[ct] = indices_in->indices[i];
            ct++;
        }
    }
    indices_out = non_wall_indices;
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::Voxelize Turns raw point cloud into voxel point cloud
 * @param cloud_in the input point cloud
 * @param indices_in the list of indices to process
 * @param cloud_out the voxelized point cloud
 */
void PtCldSegmentation<T>::Voxelize(const typename PtCld::ConstPtr cloud_in,
                                    const PtInd::ConstPtr& indices_in,
                                    typename PtCld::Ptr& cloud_out)
{
    voxel_grid_->setInputCloud(cloud_in);
    voxel_grid_->setIndices(indices_in);
    voxel_grid_->filter(*cloud_out);
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::Voxelize Turns raw point cloud into voxel point cloud
 *        Voxelization is done based on a list of indices and returns the corresponding
 *        voxelized list of indices.
 * @param cloud_in the input raw point cloud
 * @param v_indices_in a vector of indices to process
 * @param cloud_out the voxelized point cloud
 * @param v_indices_out the corresponding (re-indexed) list of indices
 */
void PtCldSegmentation<T>::Voxelize(const typename PtCld::ConstPtr cloud_in,
                                    std::vector< PtInd::Ptr> & v_indices_in,
                                    typename PtCld::Ptr& cloud_out,
                                    std::vector< PtInd::Ptr> & v_indices_out)
{
    int ind = 0;
    for(int i = 0 ; i < v_indices_in.size() ; i++)
    {
        // Get voxels for the blob
        voxel_grid_->setInputCloud(cloud_in);
        voxel_grid_->setIndices(v_indices_in[i]);

        typename PtCld::Ptr v_cloud = boost::make_shared<PtCld>();
        v_cloud->points.clear();
        PtInd::Ptr v_indices = boost::make_shared<PtInd>();
        voxel_grid_->filter(*v_cloud);
        // add voxels to the list
        cloud_out->points.insert(cloud_out->points.end(),
                                 v_cloud->points.begin(),
                                 v_cloud->points.end());

        // get indices
        for(int j = 0 ; j < v_cloud->points.size() ; j++)
        {
            v_indices->indices.push_back(ind);
            ind ++;
        }
        v_indices_out.push_back(v_indices);
    }
}


template <typename T>
/**
 * @brief PtCldSegmentation<T>::ProjectionBasedClustering
 *        Creates clusters of points that are likely to be objects
 * @param cloud_in the input point cloud
 * @param cloud_out the point cloud of clusterized elements
 * @param v_indices_out a vector of list of indices.
 *        Each list is the set of indices (of cloud_out) from a cluster
 */
void PtCldSegmentation<T>::ProjectionBasedClustering(const typename PtCld::ConstPtr cloud_in,
                                                     typename PtCld::Ptr& cloud_out,
                                                     std::vector<PtInd::Ptr>& v_indices_out)
{
    // Project cloud on floor
    floor_projector_->setModelCoefficients(floor_equation_);
    floor_projector_->setInputCloud(cloud_in);
    floor_projector_->filter(*cloud_out);

    // Cluster point cloud to find objects indices
    std::vector<PtInd> clusters_indices;
    clustering_algo_->setInputCloud(cloud_out);
    clustering_algo_->extract(clusters_indices);

    // Make vector of PointIndices::Ptr
    std::size_t nbClusters = clusters_indices.size();
    v_indices_out.resize(nbClusters);
    for (std::size_t i = 0; i < nbClusters; ++i)
    {
        PtInd::Ptr cluster_indices_ptr = boost::make_shared<PtInd>(clusters_indices[i]);
        v_indices_out[i] = cluster_indices_ptr;
    }
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::ImageBasedClustering
 *        Clusterizes the input point cloud based on image geometry.
 * @param cloud_in the input point cloud
 * @param indices_in the indices to be processed
 * @param normals the input normal vectors
 * @param v_indices_out a vector containing a list of indices for each extracted cluster
 */
void PtCldSegmentation<T>::ImageBasedClustering(const typename PtCld::ConstPtr cloud_in,
                                                const PtInd::ConstPtr& indices_in,
                                                NormCld::Ptr& normals,
                                                std::vector<PtInd::Ptr>& v_indices_out
                                                )
{
    /* make mask with input indices */
    cv::Mat blobs_mask = cv::Mat::zeros(cloud_in->height,cloud_in->width, CV_8U);
    for(int j = 0 ; j < indices_in->indices.size() ; j++)
    {
        blobs_mask.at<uchar>(indices_in->indices[j]) = 255;
    }
    /* Remove sharp contours (NaN normals) */
    for(int j = 0 ; j < normals->points.size() ; j++)
    {
        if(isnan(normals->points[j].data_n[0]))
            blobs_mask.at<uchar>(j) = 0;
    }
    /* find contours */
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(blobs_mask.clone(),contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    /* make blobs and filter if too small */
    for(int i  = 0 ; i < contours.size() ; i++)
    {
        double area = cv::contourArea(contours[i], false);
        if(area < min_obj_pixel_size_)
        {
            continue;
        }
        PtInd::Ptr blob_indices = boost::make_shared<PtInd>();
        cv::Mat mask = cv::Mat::zeros(blobs_mask.size(), blobs_mask.type());
        cv::drawContours(mask, contours, i, 255, CV_FILLED);
        cv::erode(mask,mask,cv::Mat());
        mask = mask & blobs_mask;

        for(int m = 0 ; m < mask.rows*mask.cols ; m++)
        {
            if(mask.at<uchar>(m) > 0)
                blob_indices->indices.push_back(m);
        }
        v_indices_out.push_back(blob_indices);
    }

    /* TODO: Recover points where normal is absent */
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::RemoveClustersAtBorder
 *        Check if each input point is at a certain distance of the border.
 *        Remove clusters that contains points too close to the border.
 * @param cloud_in the input point cloud
 * @param v_indices_in The list of clustered indices to process
 * @param v_indices_out The list of filtered clustered indices
 */

/**
 * @brief PtCldSegmentation<T>::RemoveClustersAtBorder
 *        Check if each input point is at a certain distance of the border.
 *        Remove clusters that contains points too close to the border.
 * @param cloud_in the input point cloud
 * @param v_indices_in The list of clustered indices to process
 * @param v_indices_out The list of filtered clustered indices
 * @param kept_idx the vector elements that have been kept in the process
 */
void PtCldSegmentation<T>::RemoveClustersAtBorder(const typename PtCld::ConstPtr cloud_in,
                                                  std::vector<PtInd::Ptr>& v_indices_in,
                                                  std::vector<PtInd::Ptr>& v_indices_out,
                                                  std::vector<int> & kept_idx)
{
    /// Celine's version
    /// Here, I used the intrinsic parameters of the kinect to determine the borders
    /// of the depth map. This is more accurate than using a static frame border in
    /// the RGB frame reference.

    // Initialize variables
    std::size_t nbClusters = v_indices_in.size();
    v_indices_out.reserve(nbClusters);
    pcl::PointIndices borderIndices;


    // for each cluster, determine indices that are close to border
    for (std::size_t i = 0; i < nbClusters; ++i)
    {
        // get border of the kinect depth frame
        borderIndices.indices.clear();
        ComputeBorderIndices(cloud_in, v_indices_in[i], borderIndices);
        // get border of the maximum distance to the kinect
        Eigen::Vector4f posMin, posMax;
        pcl::getMinMax3D(*cloud_in, *v_indices_in[i], posMin, posMax);
        if(borderIndices.indices.size()==0
        )
        {
            kept_idx.push_back(i);
            v_indices_out.push_back(v_indices_in[i]);
        }
        else
        {
            if(!use_tracking_option_) continue;
            // save this cluster if tracking is ok
            if(oldSegmentationMask.empty()) continue;

            cv::Mat voxelProjMask = oldSegmentationMask.clone();
            for(std::size_t j = 0 ; j < v_indices_in[i]->indices.size() ; j++)
            {
                //back project points on mask;
                cv::Point3d pt_cv(cloud_in->points[v_indices_in[i]->indices[j]].x,
                                  cloud_in->points[v_indices_in[i]->indices[j]].y,
                                  cloud_in->points[v_indices_in[i]->indices[j]].z);
                cv::Point2d uv;
                if(rgb_cam_model.initialized())
                    uv = rgb_cam_model.project3dToPixel(pt_cv);
                float u = uv.x;
                float v = uv.y;
                if(u >= 0 && u < voxelProjMask.cols
                    && v >= 0 && v < voxelProjMask.rows)
                    voxelProjMask.at<uchar>(v,u) = 255;
            }
            // if 20% of the voxels are on salient stuff, track !
            if(cv::countNonZero(voxelProjMask-oldSegmentationMask) < 0.8*v_indices_in[i]->indices.size())
            {
                kept_idx.push_back(i);
                v_indices_out.push_back(v_indices_in[i]);
            }
        }
    }
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::RemoveLargeClusters
 *        Check that cluster bounding box is less than max object size and filter.
 *        The method also filters out objects whose bounding box bottom is too far from the floor.
 * @param cloud_in the input point cloud
 * @param v_indices_in The list of clustered indices to process
 * @param v_indices_out The list of filtered clustered indices
 * @param kept_idx the vector elements that have been kept in the process
 */
void PtCldSegmentation<T>::RemoveLargeClusters(const typename PtCld::ConstPtr cloud_in,
                                               std::vector<PtInd::Ptr>& v_indices_in,
                                               std::vector<PtInd::Ptr>& v_indices_out,
                                               std::vector<int>&  kept_idx)
{
    for(int i = 0 ; i < v_indices_in.size() ; i++)
    {
        Eigen::Vector4f ptmin, ptmax;
        pcl::getMinMax3D(*cloud_in, *v_indices_in[i], ptmin, ptmax);

        if(!(ptmax[0]-ptmin[0] > obj_max_diag_size_
             || ptmax[1]-ptmin[1] > obj_max_diag_size_
             || ptmax[2]-ptmin[2] > obj_max_diag_size_))
        {
            // calculate distance from the bottom of the object to the floor
            double dot = floor_equation_->values[0]*ptmin[0]
                    + floor_equation_->values[1]*ptmin[1]
                    + floor_equation_->values[2]*ptmin[2]
                    + floor_equation_->values[3];
            double norm = std::sqrt(floor_equation_->values[0]*floor_equation_->values[0]
                    + floor_equation_->values[1]*floor_equation_->values[1]
                    +floor_equation_->values[2]*floor_equation_->values[2]);
            // Keep indices only if the 2 conditions are met (bottom low & small enough)
            if(fabs(dot)/norm < max_obj_bottom_dist_to_floor_)
            {
                kept_idx.push_back(i);
                v_indices_out.push_back(v_indices_in[i]);
            }

        }
    }
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::RemoveWallArtifacts
 *        Check that cluster bounding box is not to close to a wall
 * @param cloud_in the input point cloud
 * @param v_indices_in The list of clustered indices to process
 * @param v_indices_out The list of filtered clustered indices
 * @param kept_idx the vector elements that have been kept in the process
 */
void PtCldSegmentation<T>::RemoveWallArtifacts(const typename PtCld::ConstPtr cloud_in,
                                               std::vector<PtInd::Ptr>& v_indices_in,
                                               std::vector<PtInd::Ptr>& v_indices_out,
                                               std::vector<int>&  kept_idx)
{
    for(int i = 0 ; i < v_indices_in.size() ; i++)
    {
        bool no_close_wall = true;
        for(int j = 0 ; j < walls_list_.size() ; j++)
        {
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cloud_in, *v_indices_in[i], centroid);
            /// get min pt distance to the wall
            pcl::ModelCoefficients wall_eq = *walls_list_[j];
            double dot =  wall_eq.values[0]*centroid[0]
                        + wall_eq.values[1]*centroid[1]
                        + wall_eq.values[2]*centroid[2]
                        + wall_eq.values[3];
            double norm = std::sqrt(  wall_eq.values[0]*wall_eq.values[0]
                                    + wall_eq.values[1]*wall_eq.values[1]
                                    + wall_eq.values[2]*wall_eq.values[2]);
            if(fabs(dot)/norm < 0.1)
            {
                no_close_wall = false;
                break;
            }
        }

        if(no_close_wall)
        {
            kept_idx.push_back(i);
            v_indices_out.push_back(v_indices_in[i]);
        }
    }
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::RemoveFloorArtifacts
 *        Check that cluster bounding box is not to close to the floor
 * @param cloud_in the input point cloud
 * @param v_indices_in The list of clustered indices to process
 * @param v_indices_out The list of filtered clustered indices
 * @param kept_idx the vector elements that have been kept in the process
 */
void PtCldSegmentation<T>::RemoveFloorArtifacts(const typename PtCld::ConstPtr cloud_in,
                                                std::vector<PtInd::Ptr>& v_indices_in,
                                                std::vector<PtInd::Ptr>& v_indices_out,
                                                std::vector<int>&  kept_idx)
{
    for(int i = 0 ; i < v_indices_in.size() ; i++)
    {
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud_in, *v_indices_in[i], centroid);
        double dot = floor_equation_->values[0]*centroid[0]
                   + floor_equation_->values[1]*centroid[1]
                   + floor_equation_->values[2]*centroid[2]
                   + floor_equation_->values[3];
        double norm = std::sqrt( floor_equation_->values[0]*floor_equation_->values[0]
                               + floor_equation_->values[1]*floor_equation_->values[1]
                               + floor_equation_->values[2]*floor_equation_->values[2]);
        if(fabs(dot)/norm > 1.5*min_obj_dist_to_floor_)
        {
            kept_idx.push_back(i);
            v_indices_out.push_back(v_indices_in[i]);
        }
    }
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::UnVoxelize
 *        Retrieve original points based on the current voxelized point cloud.
 *        Result is a subset of the original point cloud
 * @param v_vox_indices_in The list of clustered indices of the voxel grid
 * @param v_indices_out The corresponding list of cluster indices of the original point cloud
 */
void PtCldSegmentation<T>::UnVoxelize(std::vector<PtInd::Ptr>& v_vox_indices_in,
                                      std::vector<PtInd::Ptr>& v_indices_out)
{
    std::size_t nbClusters = v_vox_indices_in.size();
    v_indices_out.resize(nbClusters);

    for (std::size_t i = 0; i < nbClusters; ++i)
    {
        // find the number of points in the original point cloud that belong to this cluster
        std::size_t nbPoints = 0;
        std::size_t nbPointsVox = v_vox_indices_in[i]->indices.size();
        for (std::size_t j = 0; j < nbPointsVox; ++j)
        {
            nbPoints += voxel_grid_->index_start_[v_vox_indices_in[i]->indices[j]+1]
                      - voxel_grid_->index_start_[v_vox_indices_in[i]->indices[j]];
        }
        // populate the output list of indices for this cluster
        PtInd::Ptr full_cld_indices = boost::make_shared<PtInd>();
        full_cld_indices->indices.reserve(nbPoints);
        for (std::size_t j = 0; j < nbPointsVox; ++j)
        {
            std::size_t start = voxel_grid_->index_start_[v_vox_indices_in[i]->indices[j]];
            std::size_t end = voxel_grid_->index_start_[v_vox_indices_in[i]->indices[j]+1];
            for (std::size_t k = start; k < end; ++k)
            {
                full_cld_indices->indices.push_back(voxel_grid_->index_vector_[k].cloud_point_index);
            }
        }
        v_indices_out[i] = full_cld_indices;
    }
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::MakeObjectsBBsAndPos
 *        Finds bounding boxes of image blobs, objects on the RGB frame,
 *        bounding boxes in the 3D point cloud referece
 *        centroids in the 3D point cloud reference
 * @param cloud_in the input point cloud
 * @param v_obj_indices_in the input vector containing the list of indices for each object
 * @param v_blobs_indices_in the input vector containing the list of indices for each image blob
 */
void PtCldSegmentation<T>::MakeObjectsBBsAndPos(const typename PtCld::ConstPtr cloud_in,
                                                const std::vector<PtInd::Ptr>& v_obj_indices_in,
                                                const std::vector<PtInd::Ptr>& v_blobs_indices_in)
{

    // 2D bounding boxes
    for(int m = 0 ; m < v_blobs_indices_in.size() ; m++)
    {
        blobs_2d_bboxes_.push_back(Make2Dbboxes(v_blobs_indices_in[m]));
    }
    for(int m = 0 ; m < v_obj_indices_in.size() ; m++)
    {
        obj_2d_bboxes_.push_back(Make2Dbboxes(v_obj_indices_in[m]));
    }

    // 3D bounding boxes and centroid
    std::size_t nbObjects = v_obj_indices_in.size();
    obj_3d_bboxes_.resize(nbObjects);
    for (std::size_t i = 0; i < nbObjects; ++i)
    {
        // 3d bounding box
        pcl::getMinMax3D(*cloud_in, *v_obj_indices_in[i],
                         obj_3d_bboxes_[i].min,
                         obj_3d_bboxes_[i].max);
        // Compute object 3d position
        pcl::compute3DCentroid(*cloud_in, *v_obj_indices_in[i],
                               obj_3d_bboxes_[i].pos);
    }
}


template <typename T>
/**
 * @brief PtCldSegmentation<T>::MakeObjectsPtCldAndNorm
 *        Create a list of point clouds for each segmented object
 * @param cloud_in the input full point cloud
 * @param v_indices_in the input vector containing the list of indices for each cluster
 */
/**
 * @brief PtCldSegmentation<T>::MakeObjectsPtCldAndNorm
 *        Create a list of point clouds and a list of normals for each object
 * @param cloud_in the input full point cloud
 * @param normals_in the input full normals cloud
 * @param v_indices_in the input vector containing the list of indices for each cluster
 */
void PtCldSegmentation<T>::MakeObjectsPtCldAndNorm(const typename PtCld::ConstPtr cloud_in,
                                                  const typename NormCld::ConstPtr normals_in,
                                                  const std::vector<PtInd::Ptr>& v_indices_in)
{
    std::size_t nbObjects = v_indices_in.size();
    obj_pt_clouds_.resize(nbObjects);
    obj_normals_.resize(nbObjects);

    for (std::size_t i = 0; i < nbObjects; ++i)
    {
        // Make object point cloud from indices
        typename PtCld::Ptr cloud = boost::make_shared<PtCld>(*cloud_in,v_indices_in[i]->indices);
        obj_pt_clouds_[i] = cloud;
        // Make object normals from indices
        NormCld::Ptr norm = boost::make_shared<NormCld>(*normals_in,v_indices_in[i]->indices);
        obj_normals_[i] = norm;
    }
}

template <typename T>
/**
 * @brief PtCldSegmentation::MakeColorDebugCloud use segmentation mask to give colors to the debug cloud
 */
void PtCldSegmentation<T>::MakeColorDebugCloud()
{
    if(!debug_cloud_ ||segmentationMask.empty())
        return;
    assert(debug_cloud_->height == segmentationMask.rows);
    assert(debug_cloud_->width == segmentationMask.cols);
    for(int i = 0 ; i < segmentationMask.cols*segmentationMask.rows ; i++)
    {
        cv::Vec3b color = segmentationMask.at<cv::Vec3b>(i);
        SetDebugCloudColor(i,color);
    }
}


template <typename T>
/**
 * @brief PtCldSegmentation<T>::ComputeBorderIndices
 *        Based on intrinsic kinect parameters, projects 3D points back to depth frame and
 *        determine input points that are close to the depth frame border
 * @param cloud_in the input point cloud
 * @param indices_in the indices to process
 * @param indices_out the indices foud to be close to border
 */
void PtCldSegmentation<T>::ComputeBorderIndices(const typename PtCld::ConstPtr cloud_in,
                                                PtInd::Ptr &indices_in,
                                                PtInd &indices_out)
{

    for (std::size_t i = 0; i < indices_in->indices.size(); i++)
    {
        int k = indices_in->indices[i];
        if(!(cloud_in->points[k].z > 0))
            continue;

        cv::Point3d pt_cv(cloud_in->points[k].x,
                          cloud_in->points[k].y,
                          cloud_in->points[k].z);
        cv::Point2d uv;
        if(depth_cam_model.initialized())
            uv = depth_cam_model.project3dToPixel(pt_cv);
        float u = uv.x;
        float v = uv.y;

        /* If depth pixel is out of bound, keep it */
        if(   u < obj_at_border_pix_tolerance_
           || u > kinect_cloud_->width-obj_at_border_pix_tolerance_
           || v < obj_at_border_pix_tolerance_+15
           || v > kinect_cloud_->height-obj_at_border_pix_tolerance_)
        {
            indices_out.indices.push_back(k);
            segmentationMask.at<cv::Vec3b>(k) = GetSegColor(BORDER);
        }
    }
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::ComputeBorderIndices
 *        Find border indices on the entire point cloud. This is used for debug
 * @param cloud_in the cloud
 * @param indices_out the indices foud to be close to border
 */
void PtCldSegmentation<T>::ComputeBorderIndices(const typename PtCld::ConstPtr cloud_in,
                                                PtInd& indices_out)
{
    PtInd::Ptr indices_in = boost::make_shared<PtInd>();
    for(int i = 0 ; i < cloud_in->points.size() ; i++)
    {
        indices_in->indices.push_back(i);
    }
    // Calling other method to find borders
    ComputeBorderIndices(cloud_in,indices_in,indices_out);
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::SetMaskIndices
 *        Color a list of indices of the map with a given color
 * @param indices_in the list of indices to set color
 * @param color the color !
 */
void PtCldSegmentation<T>::SetMaskIndices(const PtInd::ConstPtr& indices_in,
                                          cv::Vec3b color)
{
    assert(!segmentationMask.empty());
    for(int j = 0 ; j < indices_in->indices.size() ; j++)
    {
        segmentationMask.at<cv::Vec3b>(indices_in->indices[j]) = color;
    }
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::SetMaskIndices
 *        Color a list of indices of the map with a given color
 * @param v_indices_in a vector containing lists of indices
 * @param kept_indices the vector elements to consider for painting
 * @param color the color !
 */
void PtCldSegmentation<T>::SetMaskIndices(std::vector< PtInd::Ptr>& v_indices_in,
                                          std::vector<int> & kept_indices,
                                          cv::Vec3b color)
{
    assert(!segmentationMask.empty());
    for(int i = 0 ; i < kept_indices.size() ; i++)
        for(int j = 0 ; j < v_indices_in[kept_indices[i]]->indices.size() ; j++)
        {
            segmentationMask.at<cv::Vec3b>(v_indices_in[kept_indices[i]]->indices[j]) = color;
        }
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::SetMaskIndices
 *        Color the map with a given color, except for listed indices
 * @param indices_in the list of indices to keep unchanged
 * @param color the color !
 */
void PtCldSegmentation<T>::SetMaskInv_indices(const PtInd::ConstPtr& indices_in,
                                             cv::Vec3b color)
{
    assert(!segmentationMask.empty());
    int j = 0;
    for(int i = 0 ; i < kinect_cloud_->points.size() ; i++)
    {
        if(indices_in->indices[j] == i)
        {
            j++;
        }
        else
        {
            segmentationMask.at<cv::Vec3b>(i) = color;
        }
    }
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::SetDebugCloudColor
 * @param indices_in the list of indices to set color
 * @param color the color!
 */
void PtCldSegmentation<T>::SetDebugCloudColor(PtInd::Ptr &indices_in,
                                              cv::Vec3b color)
{
    assert(debug_cloud_->points.size()>0);
    for (std::size_t i = 0; i < indices_in->indices.size(); i++)
    {
        int32_t idx = indices_in->indices[i];
        debug_cloud_->points[idx].r=color[2];
        debug_cloud_->points[idx].g=color[1];
        debug_cloud_->points[idx].b=color[0];
    }

}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::SetDebugCloudColor
 * @param ind the index !
 * @param color the color !
 */
void PtCldSegmentation<T>::SetDebugCloudColor(int ind, cv::Vec3b color)
{
    debug_cloud_->points[ind].r=color[2];
    debug_cloud_->points[ind].g=color[1];
    debug_cloud_->points[ind].b=color[0];
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::Make2Dbboxes
 * @param indices_in the list of indices used to create the bounding box
 * @return the corresponding bounding box
 */
cv::Rect PtCldSegmentation<T>::Make2Dbboxes(const PtInd::ConstPtr& indices_in)
{
    cv::Mat mask = cv::Mat::zeros(segmentationMask.size(), CV_8UC1);
    for(int j = 0 ; j < indices_in->indices.size() ; j++)
    {
        mask.at<uchar>(indices_in->indices[j]) = 255;
    }

    cv::Mat foreground_loc;
    cv::findNonZero(mask, foreground_loc);
    foreground_loc = foreground_loc.reshape(1);
    double xmin, xmax;
    cv::minMaxLoc(foreground_loc.col(0),&xmin,&xmax);
    double ymin, ymax;
    cv::minMaxLoc(foreground_loc.col(1),&ymin,&ymax);
    return cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin);
}

template<typename T>
/**
 * @brief PtCldSegmentation<T>::ConcatenateIndices
 *        useful function that turns a vector of PtInd to a single PtInd
 * @param v_indices_in the vector of indices to concatenate
 * @param indices_out the result !
 */
void PtCldSegmentation<T>::ConcatenateIndices(std::vector<PtInd::Ptr> & v_indices_in ,
                                              PtInd::Ptr & indices_out
                                             )
{
    for(int m = 0 ; m < v_indices_in.size() ; m++)
    {
        indices_out->indices.insert(indices_out->indices.end(),
                                    v_indices_in[m]->indices.begin(),
                                    v_indices_in[m]->indices.end());
    }
}

template<typename T>
/**
 * @brief PtCldSegmentation<T>::UpdateKeptIdx
 *        useful function that keeps only elements specified by a vector of indices
 * @param processed_vector the vector to be purged
 * @param indices_to_keep indices to keep
 */
void PtCldSegmentation<T>::UpdateKeptIdx(std::vector<int> & processed_vector,
                                         std::vector<int> &indices_to_keep)
{
    std::vector<int> old_idx;
    old_idx.swap(processed_vector);
    processed_vector.clear();
    for(std::size_t i = 0 ; i < indices_to_keep.size() ; i++)
        processed_vector.push_back(old_idx[indices_to_keep[i]]);
}

template<typename T>
/**
 * @brief PtCldSegmentation<T>::SetRgbCamModel
 *        Specify an RGB cam model
 * @param cam_model the camera model
 */
void PtCldSegmentation<T>::SetRgbCamModel(CameraInfoLight cam_model)
{
    this->rgb_cam_model = cam_model;
}
template<typename T>
/**
 * @brief PtCldSegmentation<T>::SetDepthCamModel
 *        Specify an RGB cam model
 * @param cam_model the camera model
 */
void PtCldSegmentation<T>::SetDepthCamModel(CameraInfoLight cam_model)
{
    this->depth_cam_model = cam_model;
}

template<typename T>
/**
 * @brief PtCldSegmentation<T>::GetDebugCloud
 * @return the debug cloud
 */
typename PtCldSegmentation<T>::PtCld::Ptr PtCldSegmentation<T>::GetDebugCloud()
{
    return debug_cloud_;
}

/**
 * @brief The compare_colors struct
 *        used for the mapping in GetSegmentationMask
 */
struct compare_colors{
bool operator()(cv::Vec3b x, cv::Vec3b y)
{
    return x[0] < y[0]
            || (x[0] == y[0] && x[1] < y[1])
            || (x[0] == y[0] && x[1] == y[1] && x[2] < y[2]);
}
};

template <typename T>
/**
 * @brief PtCldSegmentation<T>::GetSegmentationMask
 * @return the grayscale segmentation mask
 */
cv::Mat PtCldSegmentation<T>::GetSegmentationMask()
{
    assert(!segmentationMask.empty());

    // Initialize segmentation map
    cv::Mat graySeg = cv::Mat::zeros(kinect_cloud_->height,kinect_cloud_->width,CV_8U);
    // create a color map
    std::map< cv::Vec3b ,SegColor , compare_colors> colorMap;
    for(size_t i = SEGCOLOR_BEGIN ; i  != SEGCOLOR_END ; i++)
    {
        SegColor state = static_cast< SegColor >(i);
        cv::Vec3b cvColor = GetSegColor(state);
        colorMap[cvColor] = state;
    }

    // Fill grayscale segmentation mask
    for(int i = 0 ; i < segmentationMask.cols*segmentationMask.rows ; i++)
    {
        cv::Vec3b cvColor = segmentationMask.at<cv::Vec3b>(i);
        SegColor color = colorMap[cvColor];
        graySeg.at<uchar>(i) = GetSegGrayscale(color);
    }
    return graySeg;
}

template <typename T>
/**
 * @brief PtCldSegmentation::GetColorSegmentationMask
 * @return The colored segmentation mask
 */
cv::Mat PtCldSegmentation<T>::GetColorSegmentationMask()
{
    return segmentationMask;
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::get2DBBox
 * @return the 2d bounding box for each blob
 */
std::vector<cv::Rect> PtCldSegmentation<T>::Get2DBBox()
{
    return blobs_2d_bboxes_;
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::get2DObjectsBBox
 * @return  the 2d bounding box for each object
 */
std::vector<cv::Rect> PtCldSegmentation<T>::Get2DObjectsBBox()
{
    return obj_2d_bboxes_;
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::get3dBBoxes
 * @return the 3d bouding boxes and centroids for each object
 */
void PtCldSegmentation<T>::Get3dBBoxes(std::vector<PtCldSegmentation::PtCldPosAndBBox> &bboxes)
{
    bboxes =  obj_3d_bboxes_;
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::getObjectPtClds
 * @return a vector of point cloud for each object
 */
void PtCldSegmentation<T>::GetObjectPtClds(std::vector<typename PtCld::Ptr> & obj_clouds)
{
    obj_clouds =  obj_pt_clouds_;
}
template <typename T>
/**
 * @brief PtCldSegmentation<T>::getObjectNormals
 * @return a vector of normals for each object
 */
void PtCldSegmentation<T>::GetObjectNormals(std::vector<NormCld::Ptr> & obj_normals)
{
   obj_normals = obj_normals_;
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::hasFloorEquation
 * @return whether plane estimation was done or not
 */
bool PtCldSegmentation<T>::hasFloorEquation()
{
    return has_floor_equation_;

}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::GetSegColor
 *        converts semantic name to color
 * @param color the semantic color name
 * @return the corresponding color
 */
cv::Vec3b PtCldSegmentation<T>::GetSegColor(SegColor color)
{
    cv::Vec3b cvColor(0,0,0);
    switch(color)
    {
        case SEGCOLOR_BEGIN:
            break;
        case SEGCOLOR_END:
            break;
        case FAR : // light gray
            cvColor = cv::Vec3b(125,125,125);
            break;
        case UNAVAILABLE : // dark gray
            cvColor = cv::Vec3b(75,75,75);
            break;
        case FLOOR : // blue
            cvColor = cv::Vec3b(255,0,0);
            break;
        case WALL : // green
            cvColor = cv::Vec3b(0,255,0);
            break;
        case NO_CLUSTER :
            cvColor = cv::Vec3b(0,0,255);
            break;
        case TOUCH_BORDER : //yellow
            cvColor = cv::Vec3b(0,255,255);
            break;
        case LARGE_CLUSTER : // cyan
            cvColor = cv::Vec3b(255,255,0);
            break;
        case WALL_ARTIFACTS : // purple
            cvColor = cv::Vec3b(255,0,255);
            break;
        case FLOOR_ARTIFACTS:
            cvColor = cv::Vec3b(125,0,0);
            break;
        case OBJECT : //white
            cvColor = cv::Vec3b(255,255,255);
            break;
        case BORDER : //black
            cvColor = cv::Vec3b(0,0,0);
            break;
    }
    return cvColor;
}

template <typename T>
/**
 * @brief PtCldSegmentation<T>::GetSegColor
 *        converts semantic name to grayscale
 * @param color the semantic color name
 * @return the corresponding color
 */
int PtCldSegmentation<T>::GetSegGrayscale(SegColor color)
{
    int cvColor = 0;
    switch(color)
    {
        case SEGCOLOR_BEGIN:
            break;
        case SEGCOLOR_END:
            break;
        case FAR :
            cvColor = 75;
            break;
        case UNAVAILABLE :
            cvColor = 75;
            break;
        case FLOOR :
            cvColor = 0;
            break;
        case WALL :
            cvColor = 0;
            break;
        case NO_CLUSTER :
            cvColor = 125;
            break;
        case TOUCH_BORDER :
            cvColor = 125;
            break;
        case LARGE_CLUSTER :
            cvColor = 125;
            break;
        case WALL_ARTIFACTS :
            cvColor = 125;
            break;
        case FLOOR_ARTIFACTS:
            cvColor = 125;
            break;
        case OBJECT :
            cvColor = 255;
            break;
        case BORDER :
            cvColor = 75;
            break;
    }
    return cvColor;
}
