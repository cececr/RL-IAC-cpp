/**
 * \file pt_cld_segmentation.h
 * \author Louis-Charles Caron, modified by Céline Craye
 * \version 0.2
 * \date 2 nov. 2015
 * \brief PtCldSegmentation class.
 *         Finds and segments object in point clouds.
 *
 *
 *
 *
 *  \section Inputs
 *      To create a PtCldSegmentation object, you will need to provide the following parameters
 *      - \b std::string \e floor_equation_filename : the name of the file containing the floor equation.
 *        If the file does not exist or is unreadable, an estimate of the floor plane is done automatically and written on disk.
 *        Alternatively, if you provide a floor equation with SetFloorEquation(), this file won't be used.
 *      - \b float \e ransac_floor_dist_thresh : the tolerance threshold (in meters) for the RANSAC floor estimation.
 *        Useful only if function EstimateMainPlaneCoefs() is called
 *      - \b float \e normal_estim_max_depth : distance threshold (in meters) between a point and its neighborhood to provide a normal estimation
 *      - \b float \e normal_estim_smooth_size : radius (in pixels) of the point neighborhood for normal estimation
 *      - \b float \e max_depth_visibility : max distance (in meters) between a point and the kinect to be processed
 *      - \b float \e max_obj_dist_to_floor : max distance (in meters) between a point and the floor to be processed
 *      - \b float \e min_obj_dist_to_floor: min distance (in meters) between a point and the theoretical floor equation to be processed
 *      - \b float \e floor_angular_tolerance : min value of the (normalized, between 0 and 1) dot product for a point to be part of the floor
 *      - \b float \e ransac_wall_dist_thresh : the tolerance threshold (in meters) for the RANSAC wall estimation.
 *      - \b float \e min_wall_diag_size : min diagonal size (in meters) for a plane to be detected as a wall
 *      - \b float \e wall_angular_tolerance : max value of the (normalized, between 0 and 1) dot product for a point to be part of the wall
 *      - \b float \e min_dist_btw_obj : min distance (in meters) between two objects to be considered as two different objects.
 *        This parameter is useful only if merge_clusters_option = true
 *      - \b float \e obj_min_diag_size : minimum diagonal size (in meters) for an object to be detected
 *      - \b float \e obj_max_diag_size : maximum diagonal size (in meters) for an object to be detected
 *      - \b int \e obj_at_border_pix_tolerance : number of pixels considered at the border of the FoV to detect objects at the border of it
 *      - \b float \e voxel_leaf_size : size (in meters) of the voxel leafs. Smaller values provide more accurate but slower processing.
 *      - \b float \e max_obj_bottom_dist_to_floor : max distance (in meters) between the bottom of a blob and the floor to be consider as an object.
 *      - \b int \e min_obj_pixel_size :  min number of pixels in the image for a blob to be considered as an object
 *      - \b bool \e use_tracking_option : whether we want to use blob tracking (true) or not (false)
 *      - \b bool \e merge_clusters_option : whether we want to merge blobs by projecting them on the floor plane (true) or not (false)
 *      - \b float \e wrong_floor_thresh : between 0 and 1, tolerance to detect wrong floor estimation. 0 = always ok, 1 = always wrong.
 *
 *      additionnaly, if you want a debug cloud, call method PtCldSegmentation::SetDebugCloud(1).
 **
 *  \section Outputs
 *      The algorithm produces a set of objects that can be useful for further processing. The following functions produce
 *          - PtCldSegmentation::GetDebugCloud() a colored debug cloud (the same as color segmentation mask, but in 3D !)
 *          - PtCldSegmentation::GetSegmentationMask() a grayscale segmentation mask
 *          - PtCldSegmentation::GetColorSegmentationMask() a colored segmentation mask
 *          - PtCldSegmentation::Get2DBBox() 2D bounding boxes of blobs (whether objects or not)
 *          - PtCldSegmentation::Get2DObjectsBBox() 2D bounding boxes of segmented objects
 *          - PtCldSegmentation::Get3dBBoxes() 3D bounding boxes and centroids of segmented objects
 *          - PtCldSegmentation::GetObjectPtClds() a vector of clouds of segmented objects
 *          - PtCldSegmentation::GetObjectNormals() a vector of normals of segmented objects
 *
 *      The grayscale segmentation mask provides four states of segmentation depending on the way elements were filtered. Those states are :
 *          - \b salient : The pixel belongs to an object that has been clearly and reliably segmented
 *          - <b>not salient</b> : The pixel belongs to the floor or a wall that has been clearly identified
 *          - \b unknown : The pixel belongs to a blob that could not be clearly identified
 *          - \b unavailable : The pixel does not have a 3d representation, or is too far from the kinect
 *      The colored segmentation mask and point cloud can be interpreted as follows
 *          - <b>light gray</b> (FAR): pixel is too far from the Kinect
 *          - <b> dark gray</b> (UNAVAILABLE): pixel has not 3D point values
 *          - <b> light blue</b>(FLOOR) : pixel belongs to the floor
 *          - \b green (WALL) : pixel belongs to a wall
 *          - \b red (NO_CLUSTER) : pixel does not belong to any blob (most likely noise)
 *          - \b yellow (TOUCH_BORDER) : pixel belong to a blob that touches the border of the field of view
 *          - \b cyan (LARGE_CLUSTER) : pixel belongs to a blob that is too large, or whose bottom is too high
 *          - \b dark blue (FLOOR_ARTIFACTS) : pixel belongs to a blob whose centroid is too close to the floor
 *          - \b purple (WALL_ARTIFACTS) : pixel belongs to a blob whose centroid is too close to a wall
 *          - \b white (OBJECT) : pixel belongs to a blob identified as an object
 *          - \b black (BORDER) : the pixel is at the border of the FoV
 *
 *   \section Implementation
The class PtCldSegmentation relies on the PCL 1.7 and OpenCV 2.4 libraries. Not compatible yet with OpenCV3
The algorithm takes as input a point cloud (typically the depth registered point cloud from a Kinect or an Xtion pro live)
and an estimate of the floor plane equation. The point cloud is first taken a a raw set of points and filtered by the steps described below.

\subsection Definitions
In this text we define as frame (rgb or depth) the images produced by the kinect, either from the rgb sensor, or from the depth.
When reasoning in frames, points are pixels, neighbors are pixel neighbors, and back projecting a 3D point in a frame corresponds to setting to "true"
the corresponding pixel. Back projecting a set of pixel then produces a binary mask of the projection frame.

We also define the field of view (or Fov) as being the cone starting from the depth frame optical center, stopping at the max visibility distance of the kinect,
and containing all 3D points visible by the kinect given this point of view. An object at the border of the FoV is then very likely to be only partially
seen by the kinect.

\subsection ComputeNormals
When possible, local normals are calculated for each point of the cloud.
Pixels with too strong variations in their neighborhood are not assigned a normal.
Parameters for defining the normal estimation are \b normal_estim_max_depth_ and \b normal_estim_smooth_size_
to define the neighborhood for the local estimation and the variation threshold above which estimation should not be done.

\subsection RemoveMaxDistPoints
Points lying too far from the Kinect are filtered out. Parameters \b max_depth_visibility is used to define the max authorized distance to the kinect.

\subsection RemoveUnavailable
Points with non-numerical values are filtered out. Non numerical values are due to reflective surfaces, shadows, or projection of the depth map on the rgb frame.

\subsection RemoveFloor
Points belonging to the floor are filtered out. We use for that the (normalized) floor equation  \f$ (a,b,c,d) \f$ and identify the points of the cloud that verify the conditions

(1)  \f$ \left| ax+by+cz+d \right| < \epsilon \f$ , \f$\epsilon \approx 0\f$

and

(2)  \f$ \left|n_x  a + n_y  b+n_z c \right| > \tau  \f$ , \f$\tau < 1\f$,  \f$\tau \approx 1\f$

(1) forces for the point \f$(x,y,z)\f$ to be close enough to the theoretical plane,
and (2) constaints the point to have a (unit) normal \f$(n_x,n_y,n_z)\f$ close enough to the plane normal (dot product close to 1).
Parameters \b max_obj_dist_to_floor, \b min_obj_dist_to_floor (\f$ = \epsilon \f$ ) and \b floor_angular_tolerance (\f$ = \tau \f$ )
are used in this step to filter out points too far from the floor, too close to the floor or with an irrelevent normal.
A sanity check is done to make sure that the plane estimation is consistent with the floor of the point cloud:
if less than a certain ratio (provided by param \b wrong_floor_thresh) of the points satisfying (1) also satisfy (2),
the plane estimation is considered unreliable, and the segmentation process returns an error signal.

\subsection RemoveWalls
Walls are identified and points belonging to them are filtered out.
We define walls as being big planes orthogonal to the floor plane, and having a contact with the border of the FoV of the sensor.
To detect walls we run the RANSAC algorithm on several portions of the point cloud, with a sensitivity provided by parameter \b ransac_wall_dist_thresh.
Portions are defined by dividing the frame (image) according to a 4x4 equally spaced grid.
When the main plane in a given portion is estimated, we apply conditions (1) and (2) on the whole point cloud, using the new plane equation instead of the floor one.
This time we use parameter \b ransac_wall_dist_thresh to define \f$ \epsilon \f$ and \b wall_angular_tolerance for \f$ \tau \f$.
Points verifying the conditions are projected back on the RGB frame, thus producing a binary mask.
The biggest connected component of the mask is determined, and the corresponding points of the cloud are retrieved.
They are considered as a wall if some points are parts of the border of the FoV, and if the diagonal of the isolated cloud exceeds a certain size
(defined by parameter \b min_wall_diag_size). The list of all wall floor equation is kept for later processing.
This method is used as it was found to be faster than applying the RANSAC several times on the whole cloud.
To further speed up the computation, we only apply RANSAC on the top half of the frame (i.e 8 times).

\subsection ImageBasedClustering
The remaining point cloud is turned into a set of blobs based on image processing.
For this step, we project points that have not been filtered out yet on the RGB frame, and we remove the one without normal estimation.
When a point does not have a normal because of a sharp depth variation in its neighborhood. T
he absence of normal then makes a natual delimitation between objects. The image-based clustering only consist in detected connected components
in the binary mask produced by back projected points that do have a normal estimation.
The set of 3D points associated with a connected component is called a blob (or a cluster), and is used in the next processing steps.
To remove noise, we filter out blobs that are less than a certain number of pixels (defined by parameter \b min_obj_pixel_size)

\subsection Voxelize
Points of the blobs are converted to voxels in order to downsample the data and speed up computation. The voxel leaf size is given by parameter \b voxel_leaf_size.

\subsection RemoveFloorArtifacts
Blobs that are too close to the floor are filtered out. For that, the centroid of each blob is determined,
and centroids closer than a certain threshold ( = 1.5* \b min_obj_dist_to_floor) are filtered out.

\subsection RemoveClustersAtBorder
Blobs that have a contact with the FoV are filtered out. We use for that the CameraInfoLight object, where default parameters are the one of the Kinect.
The borders of the depth frame are also the border of the FoV, so that blobs are back projected on the depth frame.
A border threshold is set by parameter \b obj_at_border_pix_tolerance, and blobs having common pixels with the border are filtered out.
A tracking system, enabled by parameter \b use_tracking_option "saves" blobs having a contact with a border,
if they have at least 50% of their points in common with objects detected in the previous frame. This option is however not really recommended.

\subsection RemoveLargeClusters
Blobs having a too large diagonal, or having a bottom too far from the floor are removed. Based on the bounding box of the blob, we remove the ones with a diagonal bigger than
\b obj_max_diag_size parameter, and we check if the bottom of the object is lower than \b max_obj_bottom_dist_to_floor. If not, the blob is also filtered out.


\subsection RemoveWallArtifacts
Blobs that are too close to a wall are filtered out. For that, the centroid of each blob is determined, and centroids closer than a certain threshold ( = 1.5*\b ransac_wall_dist_thresh)
of at least one wall are filtered out.

\subsection Unvoxelization
If \b merge_clusters_option is disabled, the remaining blobs are considered as objects. The last processing step consits in
turning voxels back to the corresponding original points, so that objects are upsampled to their original resolution.

\subsection ProjectionBasedClustering
If option \b merge_clusters_option is enabled, a final step is applied to group blobs belonging to the same objects and that would have been divided by the image-based segmentation.
For this step, we turn again the remaining point cloud into voxels, we project all of them on the floor, and we use a KD-Tree-based clustering to group voxels that are close enough from
each other. The minimum distance between two objects is defined by parameter \b min_dist_btw_obj. Clusters are then unvoxelized.

 *
 * \mainpage
 *      The segmentation method uses geometrical properties of the indoor scenes to produce the segmentation map.
 *      The idea is to consider that objects to segment are lying on a major planar surface that is either the floor or a table,
 *      and that large planes perpendicular to the major planar surface are walls.
 *      To be segmented, objects also have to meet geometrical, intrinsic and extrinsic criterion.
 *
 *      Intrinsic criterion are
 *          - having a diaginal size large enough
 *          - having a diaginal size small enough
 *          - having enough points in the input point cloud
 *
 *      Extrinsic criterion are
 *          - not being too far from the kinect
 *          - having a centroid not too far from the floor
 *          - having a centroid not too close from the floor
 *          - having a centroid not too close from the walls
 *          - not being in contact with the border of the field of view of the sensor
 *
 *  \section Usage
 *      The algorithm has been tested on Microsoft Kinect or Asus Xtion pro live point clouds. Please check PtCldSegmentation
 *      documentation for more informations about inputs and outputs.
 *      There is no warranty on other types of sensors.
 *      The algorithm has been tested on tabletops and floor surfaces. Although other surfaces could work,
 *      the main constraints is that they need to be flat enough to be identified as "floor"
 *      1. Construct a PtCldSegmentation object with desired parameters (see documentation for details).
 *         Defaults values are available, but it is recommended to adapt them for a given situation
 *      2. Provide a point cloud bu calling PtCldSegmentation::SetInputCloud()
 *      3. Provide an equation of the major plane (floor or table) using PtCldSegmentation::SetFloorEquation(). It is recommended to do so
 *         by using the FloorTracker. If the floor equation remain constant in the whole sequence, (for example, on a robot).
 *         A floor filename can be provided in the constructor (step 1).
 *         If no equation is provided, the main plane is determined by applying a RANSAC algorithm on the input point cloud.
 *         Hopefully, the main plane obtained is the floor plane. The equation then remains the same until you call PtCldSegmentation::EstimateMainPlaneCoefs().
 *      4. Optional : Set CameraInfoLight if you wish to use calibration data different from the Kinect/Xtion default values.
 *         Do so by calling PtCldSegmentation::SetDepthCamModel() for depth frame and PtCldSegmentation::SetRgbCamModel() for rgb frame.
 *      5. Call PtCldSegmentation::Segment() to run the segmentation.
 *      6. Get the outputs by calling
 *          - PtCldSegmentation::GetDebugCloud() for the colored debug cloud
 *          - PtCldSegmentation::GetSegmentationMask() for the grayscale segmentation mask
 *          - PtCldSegmentation::GetColorSegmentationMask() for the color segmentation mask
 *          - PtCldSegmentation::Get2DBBox() for the blobs 2D bounding boxes
 *          - PtCldSegmentation::Get2DObjectsBBox() for the objects 2D bounding boxes
 *          - PtCldSegmentation::Get3dBBoxes() for the objects 3D bounding boxes and centroids
 *          - PtCldSegmentation::GetObjectPtClds() for the objects point clouds
 *          - PtCldSegmentation::GetObjectNormals() for the objects normals
 */

#ifndef PT_CLD_SEGMENTATION_H
#define PT_CLD_SEGMENTATION_H

#include <fstream>
#include <cv.h>

#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/geometry.h>

#include "image_geometry/pinhole_camera_model.h"
#include <pcl/filters/crop_box.h>
#include "voxel_grid_fix.hpp"
#include "crop_box_fix.hpp"



//using namespace pcl;
template <typename T>
class PtCldSegmentation
{
    typedef typename pcl::PointCloud<T>           PtCld;
    typedef          pcl::PointIndices            PtInd;
    typedef          pcl::PointCloud<pcl::Normal> NormCld;


  struct PtCldPosAndBBox
  {
    Eigen::Vector4f pos, min, max;
  };

enum SegColor {SEGCOLOR_BEGIN,
                FAR,
                UNAVAILABLE,
                FLOOR,
                WALL,
                NO_CLUSTER,
                TOUCH_BORDER,
                LARGE_CLUSTER,
                WALL_ARTIFACTS,
                OBJECT,
                BORDER,
                FLOOR_ARTIFACTS,
                SEGCOLOR_END};

cv::Vec3b GetSegColor(SegColor color);
int GetSegGrayscale(SegColor color);

public:
  PtCldSegmentation(std::string floor_equation_filename,
                    float ransac_floor_dist_thresh,
                    float normal_estim_max_depth,
                    float normal_estim_smooth_size,
                    float max_depth_visibility,
                    float max_obj_dist_to_floor,
                    float min_obj_dist_to_floor,
                    float floor_angular_tolerance,
                    float ransac_wall_dist_thresh,
                    float min_wall_diag_size,
                    float wall_angular_tolerance,
                    float min_dist_btw_obj,
                    float obj_min_diag_size,
                    float obj_max_diag_size,
                    int obj_at_border_pix_tolerance,
                    float voxel_leaf_size,
                    float max_obj_bottom_dist_to_floor,
                    int min_obj_pixel_size,
                    bool use_tracking_option,
                    bool merge_clusters_option,
                    float wrong_floor_thresh);

  virtual ~PtCldSegmentation();


  void SetFloorEquation (pcl::ModelCoefficients::Ptr floor_equation);
  void SetDebugCloud    (int debug_cloud);
  void SetInputCloud    (const typename PtCld::ConstPtr & input_cloud);
  void setInputCloudFromDepth(cv::Mat & depth_map);
  void SetDepthCamModel (CameraInfoLight cam_model);
  void SetRgbCamModel   (CameraInfoLight cam_model);

  bool Segment();
  bool EstimateMainPlaneCoefs();
  bool hasFloorEquation();

  /* Output functions */
  typename PtCld::Ptr GetDebugCloud();
  cv::Mat GetSegmentationMask();
  cv::Mat GetColorSegmentationMask();
  std::vector<cv::Rect> Get2DBBox();
  std::vector<cv::Rect> Get2DObjectsBBox();
  void Get3dBBoxes(std::vector<PtCldSegmentation::PtCldPosAndBBox> & bboxes);
  void GetObjectPtClds(std::vector<typename PtCld::Ptr> &obj_clouds);
  void GetObjectNormals(std::vector<NormCld::Ptr> & obj_normals);


protected:
  // Debug
  int debug_;
  // Parameters
  std::string floor_equation_filename_;
  double      ransac_floor_dist_thresh_;
  double      normal_estim_max_depth_;
  double      normal_estim_smooth_size_;
  double      max_depth_visibility_;
  int         intFloorMinNbrPoints_;
  double      max_obj_dist_to_floor_;
  double      min_obj_dist_to_floor_;
  double      floor_angular_tolerance_;
  double      ransac_wall_dist_thresh_;
  int         intMinNbrPointInWall_;
  float       min_wall_diag_size_;
  double      wall_angular_tolerance_;
  double      min_dist_btw_obj_;
  int         intECMinSize_;
  int         intECMaxSize_;
  float       obj_min_diag_size_;
  float       obj_max_diag_size_;
  int         obj_at_border_pix_tolerance_;
  double      voxel_leaf_size_;
  float       max_obj_bottom_dist_to_floor_;
  int         min_obj_pixel_size_;
  bool        use_tracking_option_;
  bool        merge_clusters_option_;
  float       wrong_floor_thresh_;

  // Vectors of objects information
  std::vector<struct PtCldPosAndBBox>   obj_3d_bboxes_;
  std::vector<typename PtCld::Ptr>      obj_pt_clouds_;
  std::vector<NormCld::Ptr>             obj_normals_;
  std::vector<cv::Rect>                 blobs_2d_bboxes_;
  std::vector<cv::Rect>                 obj_2d_bboxes_;
  std::vector<typename PtCld::Ptr >     obj_voxels_;
  std::vector<PtInd::Ptr >              obj_indices_;

private:
  // Functions
  void RemoveMaxDistPoints     (const typename PtCld::ConstPtr& cloud_in,
                                PtInd::Ptr& indices_out
                               );
  void RemoveUnavailable       (const typename PtCld::ConstPtr& cloud_in,
                                const PtInd::ConstPtr& indices_in,
                                PtInd::Ptr& indices_out
                               );
  void ComputeNormals          (const typename PtCld::ConstPtr& cloud_in,
                                NormCld::Ptr& normals
                               );
  void RemoveFloor             (const typename PtCld::ConstPtr cloud_in,
                                const PtInd::ConstPtr& indices_in,
                                PtInd::Ptr& indices_out,
                                NormCld::Ptr& normals
                               );
  void RemoveWalls             (const typename PtCld::ConstPtr cloud_in,
                                const PtInd::ConstPtr& indices_in,
                                PtInd::Ptr& indices_out,
                                NormCld::Ptr& normals
                               );
  void RemoveFloorArtifacts    (const typename PtCld::ConstPtr cloud_in,
                                std::vector<PtInd::Ptr>& v_indices_in,
                                std::vector<PtInd::Ptr>& v_indices_out,
                                std::vector<int> &kept_idx);
  void Voxelize                (const typename PtCld::ConstPtr cloud_in,
                                const PtInd::ConstPtr& indices_in,
                                typename PtCld::Ptr& cloud_out
                               );
  void Voxelize                (const typename PtCld::ConstPtr cloud_in,
                                std::vector< PtInd::Ptr> & v_indices_in,
                                typename PtCld::Ptr& cloud_out,
                                std::vector< PtInd::Ptr> & v_indices_out
                               );
  void UnVoxelize              (std::vector<PtInd::Ptr>& v_vox_indices_in,
                                std::vector<PtInd::Ptr>& v_indices_out
                               );
  void ProjectionBasedClustering(const typename PtCld::ConstPtr cloud_in,
                                typename PtCld::Ptr& cloud_out,
                                std::vector<PtInd::Ptr>& v_indices_out
                               );
  void ImageBasedClustering    (const typename PtCld::ConstPtr cloud_in,
                                const PtInd::ConstPtr& indices_in,
                                NormCld::Ptr& normals,
                                std::vector<PtInd::Ptr>& v_indices_out
                               );
  void RemoveClustersAtBorder  (const typename PtCld::ConstPtr cloud_in,
                                std::vector<PtInd::Ptr>& v_indices_in,
                                std::vector<PtInd::Ptr>& v_indices_out,
                                std::vector<int>&  kept_idx
                               );
  void RemoveLargeClusters     (const typename PtCld::ConstPtr cloud_in,
                                std::vector<PtInd::Ptr>& v_indices_in,
                                std::vector<PtInd::Ptr>& v_indices_out,
                                std::vector<int>&  kept_idx
                               );
  void RemoveWallArtifacts     (const typename PtCld::ConstPtr cloud_in,
                                std::vector<PtInd::Ptr>& v_indices_in,
                                std::vector<PtInd::Ptr>& v_indices_out,
                                std::vector<int>&  kept_idx
                               );
  void MakeObjectsBBsAndPos    (const typename PtCld::ConstPtr cloud_in,
                                const std::vector<PtInd::Ptr>& v_obj_indices_in,
                                const std::vector<PtInd::Ptr>& v_blobs_indices_in
                               );

  void MakeObjectsPtCldAndNorm (const typename PtCld::ConstPtr cloud_in,
                                const NormCld::ConstPtr normals_in,
                                const std::vector<PtInd::Ptr>& v_indices_in
                               );
  void MakeColorDebugCloud     (
                               );
  void ComputeBorderIndices    (const typename PtCld::ConstPtr cloud_in,
                                PtInd::Ptr &indices_in,
                                PtInd &indices_out
                               );
  void ComputeBorderIndices    (const typename PtCld::ConstPtr cloud_in,
                                PtInd& indices_out
                               );
  void SetMaskIndices          (const PtInd::ConstPtr& indices_in,
                                cv::Vec3b color
                               );
  void SetMaskIndices          (std::vector<PtInd::Ptr>& v_indices_in,
                                std::vector<int> &kept_indices,
                                cv::Vec3b color
                               );
  void SetMaskInv_indices       (const PtInd::ConstPtr& indices_in,
                                cv::Vec3b color
                               );
  void SetDebugCloudColor      (PtInd::Ptr &indices_in,
                                cv::Vec3b color
                               );
  void SetDebugCloudColor      (int ind,
                                cv::Vec3b color
                               );
  cv::Rect Make2Dbboxes        (const PtInd::ConstPtr& indices_in);

  void ConcatenateIndices      (std::vector<PtInd::Ptr> & v_indices_in ,
                                PtInd::Ptr & indices_out
                               );

  void UpdateKeptIdx           (std::vector<int> & processed_vector,
                                std::vector<int> &indices_to_keep);
  void SetParams();



  // Debug
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr debug_cloud_;

  // PCL processing objects
  boost::shared_ptr<pcl::SACSegmentation<T> > floor_ransac_segmenter_;
  boost::shared_ptr<pcl::SACSegmentation<T> > wall_ransac_segmenter_;
  typename pcl::CropBox<T>::Ptr               visibility_crop_box_;
  typename pcl::CropBox<T>::Ptr               floor_distance_crop_box_;
  typename pcl::VoxelGrid<T>::Ptr             voxel_grid_;
  typename pcl::IntegralImageNormalEstimation
                        <T, pcl::Normal>::Ptr normal_estimator_;

  typename pcl::ProjectInliers<T>::Ptr        floor_projector_;
  boost::shared_ptr
        <pcl::EuclideanClusterExtraction<T> > clustering_algo_;
  image_geometry::PinholeCameraModel          depth_cam_model;
  image_geometry::PinholeCameraModel          rgb_cam_model;


  // Variables
  pcl::ModelCoefficients::Ptr   floor_equation_;
  bool                          has_floor_equation_;
  bool                          bad_segmentation;
  typename PtCld::ConstPtr      kinect_cloud_;
  Eigen::Vector4f               floor_equation_eigen_;
  std::vector<pcl::ModelCoefficients::Ptr> walls_list_;
  cv::Mat                       segmentationMask;
  cv::Mat                       oldSegmentationMask;

};

#include "pt_cld_segmentation.hpp"

#endif // PT_CLD_SEGMENTATION_H
