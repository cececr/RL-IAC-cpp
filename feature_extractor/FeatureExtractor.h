/**
 * \file FeatureExtractor.h
 * \brief tools for feature extraction
 * \author Céline Craye
 * \version 0.1
 * \date Nov 17, 2014
 *
 * Available feature extraction modes are
 *
 * - Itti&Koch features. Based on
 * "A Model of Saliency-Based Visual Attention for Rapid Scene Analysis"
 *  by Laurent Itti, Christof Koch and Ernst Niebur (PAMI 1998).
 * Extracts 42 feature maps based on center-surround oppositions for color, intensity
 * and orientation.
 *
 * - Make3D inspired features. Inspired from
 * "3-D Depth Reconstruction from a Single Still Image",
 * Ashutosh Saxena, Sung H. Chung, Andrew Y. Ng.
 * International Journal of Computer Vision (IJCV), Aug 2007.
 * Fully described at
 * "Exploration Strategies for Incremental Learning of Object-Based Visual Saliency"
 * Craye, C., Filliat, D., & Goudou, J. F. (2015, August). In ICDL-EPIROB.
 * Computes SEEDs superpixels, replaces by mean superpixel values and constructs a
 * vector of features based on pixel neighborhood at three different scales.
 * These features were found to be more efficient than Itti&Koch ones.
 *
 * When creating a feature extractor object, a downsampling ratio is applied to speed up
 * computation. The output feature maps have a size that depends on this downsampling
 * ratio. Be careful when manipulating them.
 *
 */

#ifndef FEATUREEXTRACTOR_H_
#define FEATUREEXTRACTOR_H_

#include <cv.h>
#include "seeds2.h"

//using namespace cv;
using namespace std;

class FeatureExtractor {
public:
    static const int ITTI = 0;
    static const int MAKE3D = 1;
    static const int DEEP = 2;
    FeatureExtractor();
    FeatureExtractor(int downsampling_ratio, int nb_superpixels);
	virtual ~FeatureExtractor();
    virtual vector<cv::Mat> getFeatureMap(cv::Mat input) = 0;
    virtual int getNbFeatures() = 0;
    virtual int getFeatureType() = 0;
	bool isInit();
    void reset();
    vector<cv::Mat> getAllMaps();
    cv::Size getInputSize();
    int getDownsamplingRatio();
    void updateFeatureMap(cv::Mat input);
    cv::Mat getSuperpixelsMap();
    cv::Mat getSuperpixelsMap(cv::Mat input);
    cv::Mat  getFeatureAt(cv::Point target);
    cv::Mat  getFeatureAt(cv::Point target, vector<cv::Mat> featureMap);
    bool computeSeeds(cv::Mat input, bool smooth = false);
    cv::Mat getMeanSeedsMap();
    cv::Mat getIdxSeedsMap();


protected:
	bool INIT;
    cv::Mat superpixelsMap;
    int downsampling_ratio;
    int nb_superpixels;
    SeedsWrapper sw;

private:
    std::vector<cv::Mat> allMaps;

};

class IttiFeatureExtractor: public FeatureExtractor {

public:
	IttiFeatureExtractor();
    IttiFeatureExtractor(int downsampling_ratio, int nb_superpixels);
    std::vector<cv::Mat> getFeatureMap(cv::Mat input);
    int getNbFeatures();
    int getFeatureType();
};

class Make3DFeatureExtractor: public FeatureExtractor {

public:
    Make3DFeatureExtractor();
    Make3DFeatureExtractor(int downsampling_ratio, int nb_superpixels, int make3d_scale);
    std::vector<cv::Mat> getFeatureMap(cv::Mat input);
    int getNbFeatures();
    int getFeatureType();

private:
    int make3d_scale;
};

//class DeepFeatureExtractor: public FeatureExtractor {

//public :
//    DeepFeatureExtractor();
//    DeepFeatureExtractor(int downsampling_ratio);
//    std::vector<cv::Mat> getFeatureMap(cv::Mat input);
//    int getNbFeatures();
//    int getFeatureType();

//private:
//    bool load_dl_network();
//};


#endif /* FEATUREEXTRACTOR_H_ */
