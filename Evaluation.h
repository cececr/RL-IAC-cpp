/**
 * \file Evaluation.h
 * \brief
 * \author Céline Craye
 * \version 0.1
 * \date 4 / 21 / 2015
 *
 * Custom DOxygen-style template. Provide here a description of the file
 *
 */


#ifndef EVALUATION_H
#define EVALUATION_H

#include <cv.h>
#include "printdebug.h"

//using namespace cv;
using namespace std;

namespace Eval
{
    static const int F1_SCORE = 0;
    static const int ACCURACY = 1;
    static const int HARMONIC_MEAN = 2;
    /*
     * \brief evaluate accuracy of our estimated saliency map compared to a ground truth
     * \param GT the ground truth mask
     * \param estimate the estimated saliency map
     */
    inline cv::Mat evaluate(cv::Mat GT, cv::Mat estimate)
    {
        if(GT.channels() >1)
            cvtColor(GT,GT,CV_BGR2GRAY);
        if(estimate.channels() >1)
            cvtColor(estimate,estimate,CV_BGR2GRAY);
        cv::Mat unavailable;
        cv::inRange(GT, 1,254, unavailable);
        cv::inRange(GT,254,255,GT);
        cv::inRange(estimate,255/2,255,estimate);

        unavailable = 255-unavailable;

        double tp = cv::countNonZero(GT & estimate & unavailable);
        double tn = cv::countNonZero((255-GT)& (255-estimate) & unavailable);
        double fp = cv::countNonZero((255-GT) & estimate & unavailable);
        double fn = cv::countNonZero(GT & (255-estimate) & unavailable);

        cout << "sensitivity/recall = " << tp/(tp+fn) << endl;
        cout << "specificity = " << tn/(fp+tn) << endl;
        cout << "precision = " << tp/(tp+fp) << endl;
        cout << "accuracy = " << (tp+tn)/(tp+tn+fp+fn) << endl;
        cout << "F1 score = " << 2*tp/(2*tp+fp+fn) << endl;

        return cv::Mat();
    }


    inline vector<double> getConfMatValues(cv::Mat GT, cv::Mat estimate)
    {
        /* Normalize GT and estimates */
        if(GT.channels() == 3)
            cv::cvtColor(GT,GT,CV_RGB2GRAY);
        if(estimate.channels() == 3)
            cv::cvtColor(estimate,estimate,CV_RGB2GRAY);
        GT.convertTo(GT,CV_32F);
        estimate.convertTo(estimate,CV_32F);
        double min,max;
        minMaxLoc(GT,&min,&max);
        min = 0;
        if (max > 1)
            max = 255;
        else
            max = 1;
        GT = GT/max;

        cv::minMaxLoc(estimate,&min,&max);
        min = 0;
        if (max > 1)
            max = 255;
        else
            max = 1;
        estimate = estimate/max;

        /* Get available data */
        cv::Mat available;
        cv::Mat negatives, positives;
        cv::inRange(GT,0,0,negatives);
        cv::inRange(GT,1,1,positives);
        available = (positives + negatives)/255;


        /* Truncate data */
        GT = GT > 0;
        GT = GT/255;
        estimate = estimate > 0.5;
        estimate = estimate/255;

        /* Compute confusion matrix */
        double tp = cv::countNonZero(GT & estimate & available);
        double tn = cv::countNonZero((1-GT)& (1-estimate ) & available);
        double fp = cv::countNonZero((1-GT) & estimate & available);
        double fn = cv::countNonZero(GT & (1-estimate) & available);
        vector<double> confmat;
        confmat.push_back(tp);
        confmat.push_back(tn);
        confmat.push_back(fp);
        confmat.push_back(fn);

        return confmat;
    }

    inline double getHarmonicMean(cv::Mat GT, cv::Mat estimate)
    {

        vector<double> confmat = getConfMatValues(GT, estimate);

        /* Compute F1 measure */
        double tp = confmat[0];
        double tn = confmat[1];
        double fp = confmat[2];
        double fn = confmat[3];
        if(tp+fn == 0)
        {
            return NAN;
        }
        if(tn+fp == 0)
        {
            return NAN;
        }

        return 0.5*(tp/(tp+fn)+tn/(tn+fp));
    }

    inline double getF1measure(cv::Mat GT, cv::Mat estimate)
    {

        vector<double> confmat = getConfMatValues(GT, estimate);

        /* Compute F1 measure */
        double tp = confmat[0];
        //double tn = confmat[1];
        double fp = confmat[2];
        double fn = confmat[3];
        if(tp+fp+fn == 0)
        {
            return NAN;
        }
        if(tp+fn == 0)
        {
            return NAN;
        }

        return 2*tp/(2*tp+fp+fn);
    }

    inline float getAccu(cv::Mat GT, cv::Mat estimate)
    {
        vector<double> confmat = getConfMatValues(GT, estimate);

        /* Compute F1 measure */
        double tp = confmat[0];
        double tn = confmat[1];
        double fp = confmat[2];
        double fn = confmat[3];

        return (tp+tn)/(tp+tn+fp+fn);
    }

    inline double getTemporalCompacity(cv::Mat oldClusters, cv::Mat clusterMap)
    {
        oldClusters.convertTo(oldClusters,CV_32S);
        clusterMap.convertTo(clusterMap,CV_32S);
        int diff = cv::countNonZero(clusterMap - oldClusters);
        if (diff == 0) return 0;
        int same = oldClusters.cols*oldClusters.rows-diff;
        return (double)same/(double)diff;
    }

    inline double getSpatialCompacity(cv::Mat clusterMap)
    {
        cv::Mat contours, contoursx, contoursy;
        clusterMap.convertTo(contours,CV_16U);
        double min, max;
        cv::minMaxLoc(clusterMap, &min,&max);
        double res = 0;
        for(int i = min ; i <= max ; i++)
        {
            cv::Mat mask;
            cv::inRange(clusterMap, i,i,mask);
            cv::Mat kernelx = (cv::Mat_<float>(3,3) << 0, 0, 0, 1, 0, -1, 0, 0, 0);
            cv::Mat kernely = (cv::Mat_<float>(3,3) << 0, 1, 0, 0, 0, 0, 0, -1, 0);
            filter2D(mask,contoursx,CV_32F, kernelx);
            filter2D(mask,contoursy,CV_32F, kernely);

            contours = abs(contoursx) + abs(contoursy);
            contours.convertTo(contours,CV_8U);
            threshold(contours,contours,0,255,CV_THRESH_BINARY);
            int diff = cv::countNonZero(contours);
            int same = clusterMap.cols*clusterMap.rows-diff;
            if (diff == 0)
            {
                diff = 1;same = 0;
            }
            res += (double)same/(double)diff;
        }
        return res;
    }

    inline double getUncertainty(cv::Mat estimate)
    {
        double min, max;
        cv::minMaxLoc(estimate, &min,&max);
        if (min == max)
            return 0;

        estimate.convertTo(estimate, CV_32F);
        estimate = (estimate-min)/(max-min);

        cv::Mat uncertainty = 1-2*cv::abs(estimate-0.5);
        return cv::mean(uncertainty).val[0];
    }

    inline double evaluate(cv::Mat GT, cv::Mat estimate, int metrics)
    {
        switch(metrics)
        {
            case F1_SCORE :
                return getF1measure(GT,estimate);
            case ACCURACY:
                return getAccu(GT,estimate);
            case HARMONIC_MEAN:
                return getHarmonicMean(GT,estimate);
            default:
                return getAccu(GT,estimate);
        }
    }
}

#endif // EVALUATION_H
