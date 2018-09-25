/**
 * \file FeatureExtractor.cpp
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


#include "FeatureExtractor.h"
#include "saliencyFeatureItti.h"

using namespace cv;


FeatureExtractor::FeatureExtractor() {
    INIT = false;
    downsampling_ratio = 2;
    nb_superpixels = 350;
}

FeatureExtractor::FeatureExtractor(int downsampling_ratio, int nb_superpixels) {
    INIT = false;
    this->downsampling_ratio = downsampling_ratio;
    this->nb_superpixels = nb_superpixels;
}

FeatureExtractor::~FeatureExtractor() {
}


vector<Mat> FeatureExtractor::getAllMaps()
{
    return allMaps;
}

bool FeatureExtractor::isInit()
{
    return INIT;
}

Size FeatureExtractor::getInputSize()
{
    Size s;
    if(!allMaps.empty())
    {
        s = allMaps[0].size();
    }
    return s;
}

int FeatureExtractor::getDownsamplingRatio()
{
    return downsampling_ratio;
}

void FeatureExtractor::updateFeatureMap(Mat input)
{
    allMaps = getFeatureMap(input);
}

Mat FeatureExtractor::getSuperpixelsMap()
{
    return superpixelsMap;
}

Mat FeatureExtractor::getSuperpixelsMap(Mat input)
{
    /* Get superpixels */
    bool smooth = false;
    computeSeeds(input, smooth);
    return getIdxSeedsMap();
}

Mat FeatureExtractor::getFeatureAt(Point target)
{
    return getFeatureAt(target,allMaps);
}

void FeatureExtractor::reset()
{

    for(size_t i = 0 ; i < allMaps.size() ; i++)
        allMaps[i].release();
    INIT = false;
}


/**
 * @brief get feature vector of a specific target
 * @param ttarget The point you want the feature. Should be within appropriate range
 * @param featureMap the feature maps. Cannot be empty and should be floats Mat
 * @return A Mat containing the feature vector of target location
 */
Mat FeatureExtractor::getFeatureAt(Point target, vector<Mat> featureMap)
{
    assert(featureMap.size() > 0);
    assert(target.x >= 0
            && target.x < featureMap[0].cols
            && target.y >= 0
            && target.y < featureMap[0].rows);

    /* Get sample */
    Mat sample(1,featureMap.size(),CV_32F);
    for(size_t k = 0 ; k < featureMap.size() ; k++)
    {
        sample.at<float>(k) = featureMap[k].at<float>(target);
    }
    return sample;
}

/**
 * @brief Create SEEDS object and compute superpixels
 * @param input the input image
 * @param smooth use smoothing or not
 * @return whether superpixel calculation was successful or not
 */
bool FeatureExtractor::computeSeeds(Mat input, bool smooth)
{
    sw = SeedsWrapper(input,nb_superpixels,3, smooth);
    sw.computeSuperPixels();
    return true;
}

/**
 * @brief get a map with mean color value of each superpixel
 * @return the map
 */
Mat FeatureExtractor::getMeanSeedsMap()
{
    Mat meanMap,stdMap;
    sw.getMeanSTDMaps(meanMap,stdMap);
    return meanMap;
}

/**
 * @brief get a map with each superpixel indices
 * @return the map
 */
Mat FeatureExtractor::getIdxSeedsMap()
{
    return sw.getLabelsMap();
}

IttiFeatureExtractor::IttiFeatureExtractor():FeatureExtractor(){}
IttiFeatureExtractor::IttiFeatureExtractor(int downsampling_ratio, int nb_superpixels)
                     :FeatureExtractor(downsampling_ratio,nb_superpixels){}



/**
 * @brief calculate Itti&Koch features and average with seeds superpixels
 * @param input
 * @return the vector of feature maps
 */
vector<Mat> IttiFeatureExtractor::getFeatureMap(Mat input)
{
    /* downsample data */
    resize(input, input,
           Size(input.cols/downsampling_ratio, input.rows/downsampling_ratio));

    /* Get superpixels */
    bool smooth = false;
    computeSeeds(input, smooth);

    /* Compute and get Itti saliency maps */
    saliencyFeatureItti itti;
    vector<Mat> featureMaps = itti.getAllMaps(input);

    /* Resize and average feature maps */
    for(size_t i = 0 ; i < featureMaps.size() ; i++)
    {
        Mat map = featureMaps[i];
        /// Make sure maps are float Mat
        map.convertTo(map, CV_32F);
        /// To make computation easier, resize maps
        resize(map,map,input.size());
        /// Apply mean superpixel on maps (too slow for now ...)
        //map = sw.getSuperpixelAverage(map);
        /// fill featureMaps
        featureMaps[i] = map;
    }
    INIT = true;
    return featureMaps;
}

int IttiFeatureExtractor::getNbFeatures()
{
    return 42;
}

int IttiFeatureExtractor::getFeatureType()
{
    return ITTI;
}


Make3DFeatureExtractor::Make3DFeatureExtractor():FeatureExtractor(){this->make3d_scale = 50;}
Make3DFeatureExtractor::Make3DFeatureExtractor(int downsampling_ratio, int nb_superpixels, int make3d_scale)
                       :FeatureExtractor(downsampling_ratio, nb_superpixels)
{
    this->make3d_scale = make3d_scale;
}


/**
 * @brief calculate Make3D features
 * @param input
 * @return the vector of feature maps
 */
std::vector<Mat> Make3DFeatureExtractor::getFeatureMap(Mat input)
{
    cout << "MAKE3D" << endl;
    if(input.empty())
    {
        return std::vector<Mat>();
    }
    /* downsample data */
    resize(input, input,
           Size(input.cols/downsampling_ratio, input.rows/downsampling_ratio));

    /* Get superpixels */
    bool smooth = false;
    computeSeeds(input, smooth);

    /* Get mean superpixels values */
    Mat meanMat;
    vector<Mat>meanMap_vect;
    meanMat = getMeanSeedsMap();
    meanMat.convertTo(meanMat, CV_32F);
    split(meanMat,meanMap_vect);

    /* Initialize feature map */
    vector<Mat> featureMaps;
    featureMaps.insert(featureMaps.begin(),meanMap_vect.begin(), meanMap_vect.end());

    /* Get neighbors feature maps*/
    cout << "make3d_scale " << make3d_scale << endl;
    cout << "downsampling_ratio " << downsampling_ratio << endl;
    cout << "nb_superpixels " << nb_superpixels << endl;
    int scalezero = make3d_scale/downsampling_ratio;
    int width = input.cols;
    int height = input.rows;
    for(int scale = 0 ; scale < 3 ; scale ++)
    {
        int scaleDist = pow(2,scale)*scalezero;

        /// create padding
        Mat paddedmeanMat;
        cv::copyMakeBorder( meanMat, paddedmeanMat,
                            scaleDist, scaleDist, scaleDist, scaleDist,
                            cv::BORDER_REPLICATE );

        /// create shift rectangles
        vector<Rect> rect_vect;
        rect_vect.push_back(Rect(0,scaleDist,width, height));
        rect_vect.push_back(Rect(2*scaleDist,scaleDist,width, height));
        rect_vect.push_back(Rect(scaleDist,0,width, height));
        rect_vect.push_back(Rect(scaleDist,2*scaleDist,width, height));

        /// for each rectangle, copy subimage and add to featureMaps vector
        for(size_t k = 0 ; k < rect_vect.size() ; k++)
        {
            Mat shiftedMeanMat;
            vector<Mat> shiftedMeanMap_vect;
            paddedmeanMat(rect_vect[k]).copyTo(shiftedMeanMat);
            split(shiftedMeanMat,shiftedMeanMap_vect);
            featureMaps.insert(featureMaps.end(),shiftedMeanMap_vect.begin(), shiftedMeanMap_vect.end());
        }
    }
    INIT = true;
    return featureMaps;
}

int Make3DFeatureExtractor::getNbFeatures()
{
    return 39;
}

int Make3DFeatureExtractor::getFeatureType()
{
    return MAKE3D;
}



