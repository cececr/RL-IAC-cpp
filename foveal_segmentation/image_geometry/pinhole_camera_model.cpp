#include "pinhole_camera_model.h"
#include <boost/make_shared.hpp>
#include <iostream>

namespace image_geometry {

enum DistortionState { NONE, CALIBRATED, UNKNOWN };

struct PinholeCameraModel::Cache
{
  DistortionState distortion_state;

  cv::Mat_<double> K_binned, P_binned; // Binning applied, but not cropping
  
  mutable bool full_maps_dirty;
  mutable cv::Mat full_map1, full_map2;

  mutable bool reduced_maps_dirty;
  mutable cv::Mat reduced_map1, reduced_map2;
  
  mutable bool rectified_roi_dirty;
  mutable cv::Rect rectified_roi;

  Cache()
    : full_maps_dirty(true),
      reduced_maps_dirty(true),
      rectified_roi_dirty(true)
  {
  }
};

PinholeCameraModel::PinholeCameraModel()
{
}

PinholeCameraModel::PinholeCameraModel(const PinholeCameraModel& other)
{
  if (other.initialized())
    fromCameraInfo(other.cam_info_);
}

// For uint32_t, string, bool...
template<typename T>
bool update(const T& new_val, T& my_val)
{
  if (my_val == new_val)
    return false;
  my_val = new_val;
  return true;
}

// For boost::array, std::vector
template<typename MatT>
bool updateMat(const MatT& new_mat, MatT& my_mat, cv::Mat_<double>& cv_mat, int rows, int cols)
{
    assert(rows == my_mat.rows);
    assert(cols == my_mat.cols);
    cv_mat = my_mat.clone();
  return true;
}

bool PinholeCameraModel::fromCameraInfo(const CameraInfoLight &msg)
{
  // Create our repository of cached data (rectification maps, etc.)
  if (!cache_)
    cache_ = boost::make_shared<Cache>();
  
  // Binning = 0 is considered the same as binning = 1 (no binning).
  uint32_t binning_x = msg.binning_x ? msg.binning_x : 1;
  uint32_t binning_y = msg.binning_y ? msg.binning_y : 1;
  
  // Update any parameters that have changed. The full rectification maps are
  // invalidated by any change in the calibration parameters OR binning.
  bool &full_dirty = cache_->full_maps_dirty;
  full_dirty |= update(msg.height, cam_info_.height);
  full_dirty |= update(msg.width,  cam_info_.width);
  full_dirty |= updateMat(msg.D, cam_info_.D, D_, 1, 5);
  full_dirty |= updateMat(msg.K, cam_info_.K, K_full_, 3, 3);
  full_dirty |= updateMat(msg.R, cam_info_.R, R_, 3, 3);
  full_dirty |= updateMat(msg.P, cam_info_.P, P_full_, 3, 4);
  full_dirty |= update(binning_x, cam_info_.binning_x);
  full_dirty |= update(binning_y, cam_info_.binning_y);

  // The reduced rectification maps are invalidated by any of the above or a
  // change in ROI.
  bool reduced_dirty  = full_dirty;

  // Figure out how to handle the distortion
  cache_->distortion_state = (cam_info_.D.at<double>(0,0) == 0.0) ? NONE : CALIBRATED;

  // If necessary, create new K_ and P_ adjusted for binning and ROI
  /// @todo Calculate and use rectified ROI
  bool adjust_binning = (binning_x > 1) || (binning_y > 1);
  bool adjust_roi = false;

  if (!adjust_binning && !adjust_roi) {
    K_ = K_full_;
    P_ = P_full_;
  }
  else {
    K_full_.copyTo(K_);
    P_full_.copyTo(P_);


    if (binning_x > 1) {
      double scale_x = 1.0 / binning_x;
      K_(0,0) *= scale_x;
      K_(0,2) *= scale_x;
      P_(0,0) *= scale_x;
      P_(0,2) *= scale_x;
      P_(0,3) *= scale_x;
    }
    if (binning_y > 1) {
      double scale_y = 1.0 / binning_y;
      K_(1,1) *= scale_y;
      K_(1,2) *= scale_y;
      P_(1,1) *= scale_y;
      P_(1,2) *= scale_y;
      P_(1,3) *= scale_y;
    }
  }

  return reduced_dirty;
}

void PinholeCameraModel::project3dToPixel(const cv::Point3d& xyz, cv::Point2d& uv_rect)
{
  uv_rect = project3dToPixel(xyz);
}

cv::Size PinholeCameraModel::fullResolution() const
{
  assert( initialized() );
  return cv::Size(cam_info_.width, cam_info_.height);
}

cv::Point2d PinholeCameraModel::project3dToPixel(const cv::Point3d& xyz) const
{
  assert( initialized() );
  assert(P_(2, 3) == 0.0); // Calibrated stereo cameras should be in the same plane

  // [U V W]^T = P * [X Y Z 1]^T
  // u = U/W
  // v = V/W
  cv::Point2d uv_rect;
  uv_rect.x = (fx()*xyz.x + Tx()) / xyz.z + cx();
  uv_rect.y = (fy()*xyz.y + Ty()) / xyz.z + cy();
  return uv_rect;
}

void PinholeCameraModel::projectPixelTo3dRay(const cv::Point2d& uv_rect, cv::Point3d& ray)
{
  ray = projectPixelTo3dRay(uv_rect);
}

cv::Point3d PinholeCameraModel::projectPixelTo3dRay(const cv::Point2d& uv_rect) const
{
  assert( initialized() );

  cv::Point3d ray;
  ray.x = (uv_rect.x - cx() - Tx()) / fx();
  ray.y = (uv_rect.y - cy() - Ty()) / fy();
  ray.z = 1.0;
  return ray;
}

void PinholeCameraModel::rectifyImage(const cv::Mat& raw, cv::Mat& rectified, int interpolation) const
{
  assert( initialized() );

  switch (cache_->distortion_state) {
    case NONE:
      raw.copyTo(rectified);
      break;
    case CALIBRATED:
      initRectificationMaps();
      cv::remap(raw, rectified, cache_->reduced_map1, cache_->reduced_map2, interpolation);
      break;
    default:
      assert(cache_->distortion_state == UNKNOWN);
      throw Exception("Cannot call rectifyImage when distortion is unknown.");
  }
}


void PinholeCameraModel::rectifyPoint(const cv::Point2d& uv_raw, cv::Point2d& uv_rect)
{
  uv_rect = rectifyPoint(uv_raw);
}

cv::Point2d PinholeCameraModel::rectifyPoint(const cv::Point2d& uv_raw) const
{
  assert( initialized() );

  if (cache_->distortion_state == NONE)
    return uv_raw;
  if (cache_->distortion_state == UNKNOWN)
    throw Exception("Cannot call rectifyPoint when distortion is unknown.");
  assert(cache_->distortion_state == CALIBRATED);

  /// @todo cv::undistortPoints requires the point data to be float, should allow double
  cv::Point2f raw32 = uv_raw, rect32;
  const cv::Mat src_pt(1, 1, CV_32FC2, &raw32.x);
  cv::Mat dst_pt(1, 1, CV_32FC2, &rect32.x);
  cv::undistortPoints(src_pt, dst_pt, K_, D_, R_, P_);
  return rect32;
}

void PinholeCameraModel::unrectifyPoint(const cv::Point2d& uv_rect, cv::Point2d& uv_raw)
{
  uv_raw = unrectifyPoint(uv_rect);
}

cv::Point2d PinholeCameraModel::unrectifyPoint(const cv::Point2d& uv_rect) const
{
  assert( initialized() );

  if (cache_->distortion_state == NONE)
    return uv_rect;
  if (cache_->distortion_state == UNKNOWN)
    throw Exception("Cannot call unrectifyPoint when distortion is unknown.");
  assert(cache_->distortion_state == CALIBRATED);

  /// @todo Make this just call projectPixelTo3dRay followed by cv::projectPoints. But
  /// cv::projectPoints requires 32-bit float, which is annoying.

  // Formulae from docs for cv::initUndistortRectifyMap,
  // http://opencv.willowgarage.com/documentation/cpp/camera_calibration_and_3d_reconstruction.html

  // x <- (u - c'x) / f'x
  // y <- (v - c'y) / f'y
  // c'x, f'x, etc. (primed) come from "new camera matrix" P[0:3, 0:3].
  double x = (uv_rect.x - cx() - Tx()) / fx();
  double y = (uv_rect.y - cy() - Ty()) / fy();
  // [X Y W]^T <- R^-1 * [x y 1]^T
  double X = R_(0,0)*x + R_(1,0)*y + R_(2,0);
  double Y = R_(0,1)*x + R_(1,1)*y + R_(2,1);
  double W = R_(0,2)*x + R_(1,2)*y + R_(2,2);
  // x' <- X/W, y' <- Y/W
  double xp = X / W;
  double yp = Y / W;
  // x'' <- x'(1+k1*r^2+k2*r^4+k3*r^6) + 2p1*x'*y' + p2(r^2+2x'^2)
  // y'' <- y'(1+k1*r^2+k2*r^4+k3*r^6) + p1(r^2+2y'^2) + 2p2*x'*y'
  // where r^2 = x'^2 + y'^2
  double r2 = xp*xp + yp*yp;
  double r4 = r2*r2;
  double r6 = r4*r2;
  double a1 = 2*xp*yp;
  double k1 = D_(0,0), k2 = D_(0,1), p1 = D_(0,2), p2 = D_(0,3), k3 = D_(0,4);
  double barrel_correction = 1 + k1*r2 + k2*r4 + k3*r6;
  if (D_.cols == 8)
    barrel_correction /= (1.0 + D_(0,5)*r2 + D_(0,6)*r4 + D_(0,7)*r6);
  double xpp = xp*barrel_correction + p1*a1 + p2*(r2+2*(xp*xp));
  double ypp = yp*barrel_correction + p1*(r2+2*(yp*yp)) + p2*a1;
  // map_x(u,v) <- x''fx + cx
  // map_y(u,v) <- y''fy + cy
  // cx, fx, etc. come from original camera matrix K.
  return cv::Point2d(xpp*K_(0,0) + K_(0,2), ypp*K_(1,1) + K_(1,2));
}

cv::Rect PinholeCameraModel::rectifyRoi(const cv::Rect& roi_raw) const
{
  assert( initialized() );

  /// @todo Actually implement "best fit" as described by REP 104.
  
  // For now, just unrectify the four corners and take the bounding box.
  cv::Point2d rect_tl = rectifyPoint(cv::Point2d(roi_raw.x, roi_raw.y));
  cv::Point2d rect_tr = rectifyPoint(cv::Point2d(roi_raw.x + roi_raw.width, roi_raw.y));
  cv::Point2d rect_br = rectifyPoint(cv::Point2d(roi_raw.x + roi_raw.width,
                                                 roi_raw.y + roi_raw.height));
  cv::Point2d rect_bl = rectifyPoint(cv::Point2d(roi_raw.x, roi_raw.y + roi_raw.height));

  cv::Point roi_tl(std::ceil (std::min(rect_tl.x, rect_bl.x)),
                   std::ceil (std::min(rect_tl.y, rect_tr.y)));
  cv::Point roi_br(std::floor(std::max(rect_tr.x, rect_br.x)),
                   std::floor(std::max(rect_bl.y, rect_br.y)));

  return cv::Rect(roi_tl.x, roi_tl.y, roi_br.x - roi_tl.x, roi_br.y - roi_tl.y);
}

cv::Rect PinholeCameraModel::unrectifyRoi(const cv::Rect& roi_rect) const
{
  assert( initialized() );

  /// @todo Actually implement "best fit" as described by REP 104.
  
  // For now, just unrectify the four corners and take the bounding box.
  cv::Point2d raw_tl = unrectifyPoint(cv::Point2d(roi_rect.x, roi_rect.y));
  cv::Point2d raw_tr = unrectifyPoint(cv::Point2d(roi_rect.x + roi_rect.width, roi_rect.y));
  cv::Point2d raw_br = unrectifyPoint(cv::Point2d(roi_rect.x + roi_rect.width,
                                                  roi_rect.y + roi_rect.height));
  cv::Point2d raw_bl = unrectifyPoint(cv::Point2d(roi_rect.x, roi_rect.y + roi_rect.height));

  cv::Point roi_tl(std::floor(std::min(raw_tl.x, raw_bl.x)),
                   std::floor(std::min(raw_tl.y, raw_tr.y)));
  cv::Point roi_br(std::ceil (std::max(raw_tr.x, raw_br.x)),
                   std::ceil (std::max(raw_bl.y, raw_br.y)));

  return cv::Rect(roi_tl.x, roi_tl.y, roi_br.x - roi_tl.x, roi_br.y - roi_tl.y);
}

void PinholeCameraModel::initRectificationMaps() const
{
  /// @todo For large binning settings, can drop extra rows/cols at bottom/right boundary.
  /// Make sure we're handling that 100% correctly.
  
  if (cache_->full_maps_dirty) {
    // Create the full-size map at the binned resolution
    /// @todo Should binned resolution, K, P be part of public API?
    cv::Size binned_resolution = fullResolution();
    binned_resolution.width  /= binningX();
    binned_resolution.height /= binningY();

    cv::Mat_<double> K_binned, P_binned;
    if (binningX() == 1 && binningY() == 1) {
      K_binned = K_full_;
      P_binned = P_full_;
    }
    else {
      K_full_.copyTo(K_binned);
      P_full_.copyTo(P_binned);
      if (binningX() > 1) {
        double scale_x = 1.0 / binningX();
        K_binned(0,0) *= scale_x;
        K_binned(0,2) *= scale_x;
        P_binned(0,0) *= scale_x;
        P_binned(0,2) *= scale_x;
        P_binned(0,3) *= scale_x;
      }
      if (binningY() > 1) {
        double scale_y = 1.0 / binningY();
        K_binned(1,1) *= scale_y;
        K_binned(1,2) *= scale_y;
        P_binned(1,1) *= scale_y;
        P_binned(1,2) *= scale_y;
        P_binned(1,3) *= scale_y;
      }
    }
    
    // Note: m1type=CV_16SC2 to use fast fixed-point maps (see cv::remap)
    cv::initUndistortRectifyMap(K_binned, D_, R_, P_binned, binned_resolution,
                                CV_16SC2, cache_->full_map1, cache_->full_map2);
    cache_->full_maps_dirty = false;
  }

  if (cache_->reduced_maps_dirty) {

  }
}

} //namespace image_geometry

