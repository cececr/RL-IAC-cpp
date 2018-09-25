/**
 * \file saliencyDetectionItti.h
 * \brief Akman implementation of Itti&Koch. Just removed the ROS-related functions.
 * \author Céline Craye
 * \version 0.1
 * \date 2014
 */

//===================================================================================
// Name        : saliencyDetectionItti.h
// Author      : Oytun Akman, oytunakman@gmail.com
// Version     : 1.0
// Copyright   : Copyright (c) 2010 LGPL
// Description : C++ implementation of "A Model of Saliency-Based Visual Attention
//				 for Rapid Scene Analysis" by Laurent Itti, Christof Koch and Ernst
//				 Niebur (PAMI 1998).												  
//===================================================================================


#ifndef _saliencyFeatureItti_H_INCLUDED_
#define _saliencyFeatureItti_H_INCLUDED_


// OpenCV
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"

//using namespace cv;
using namespace std;

class saliencyFeatureItti
{
protected:

public:
        saliencyFeatureItti()
        {
    	}


        ~saliencyFeatureItti()
        {
    	}

        void calculateSaliencyMap(const cv::Mat* src, cv::Mat* dst, int scaleBase);
    	void combineFeatureMaps(int scale);
        vector<cv::Mat> getAllMaps(cv::Mat input); // get all feature maps

        cv::Mat conspicuityMap_I;
        cv::Mat conspicuityMap_C;
        cv::Mat conspicuityMap_O;
        cv::Mat S;

private:
        cv::Mat r,g,b,R,G,B,Y,I;
        vector<cv::Mat> gaussianPyramid_I;
        vector<cv::Mat> gaussianPyramid_R;
        vector<cv::Mat> gaussianPyramid_G;
        vector<cv::Mat> gaussianPyramid_B;
        vector<cv::Mat> gaussianPyramid_Y;

        void createChannels(const cv::Mat* src);
        void createScaleSpace(const cv::Mat* src, vector<cv::Mat>* dst, int scale);

    	void normalize_rgb();
    	void create_RGBY();
    	void createIntensityFeatureMaps();
    	void createColorFeatureMaps();
    	void createOrientationFeatureMaps(int orientation);
        void mapNormalization(cv::Mat* src);
    	void clearBuffers();

        vector<cv::Mat> featureMaps_I;
        vector<cv::Mat> featureMaps_RG;
        vector<cv::Mat> featureMaps_BY;
        vector<cv::Mat> featureMaps_0;
        vector<cv::Mat> featureMaps_45;
        vector<cv::Mat> featureMaps_90;
        vector<cv::Mat> featureMaps_135;
};
#endif

