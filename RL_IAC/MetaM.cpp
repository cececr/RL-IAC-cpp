/*
 * MetaM.cpp
 *
 *  Created on: Nov 10, 2014
 *      Author: celine
 */

#include "MetaM.h"
#include "printdebug.h"
#include "Evaluation.h"

#include <iostream>
#include <fstream>

using namespace cv;

MetaM::MetaM(int evaluation_metrics, bool use_per_frame_eval, int window, int smooth)
{
    this->evaluation_metrics = evaluation_metrics;
    this->use_per_frame_eval = use_per_frame_eval;
    if(use_per_frame_eval)
    {
        this->window = window; // number of samples for linear regression
        this->smooth = smooth; // number of samples for data smoothing
    }
    else
    {
        this->window = window* INPUT_DATA_SIZE;
        this->smooth = smooth* INPUT_DATA_SIZE;
    }
    errorHistory.push_back((float)1);
    smoothedHistory = errorHistory;
    uncertaintyHistory.push_back((float)1);
}

MetaM::~MetaM()
{
}

bool MetaM::updateErrors(Mat estimates, Mat observations)
{
    /* Sanity check */
    assert(estimates.size() == observations.size());
    assert(estimates.type() == observations.type());
    /* Normalize data between 0 and 1 */
//    estimates = estimates > 0.5;
//    estimates.convertTo(estimates,observations.type());
//    estimates = estimates/255;

    if(!use_per_frame_eval)
    {
        /* Force estimates to have a fixed size */
        resize_input_data(estimates,observations);
        PrintDebug::print(observations, "observations");
        PrintDebug::print(estimates, "estimates");
        /* update histories and progress estimation parameters */
        trueLabelsHistory.push_back(observations);
        estimatesHistory.push_back(estimates);
        int nsamples = observations.rows;
        Mat est, obs;
        for(int i = 0 ; i < nsamples ; i++)
        {
            int windowsize = min(window, trueLabelsHistory.rows);
            int sliding_origin = trueLabelsHistory.rows - windowsize - (nsamples-1) + i;
            if(sliding_origin < 0)
                continue;

            Rect r(0,sliding_origin,1,windowsize);
            estimatesHistory(r).copyTo(est);
            trueLabelsHistory(r).copyTo(obs);
            eval_and_update_history(obs,est);
        }
        // TODO: remove first samples if vector too big
    }
    else
    {
        eval_and_update_history(estimates, observations);
    }

    return true;
}

int MetaM::getNsamples()
{
    return errorHistory.rows-1;
}

void MetaM::setWindow(int window)
{
    this->window =  window;
}

void MetaM::setSmooth(int smooth)
{
    this->smooth = smooth;
}

float MetaM::getLearningProgress()
{
    if(use_per_frame_eval)
    {
        if(smoothedHistory.rows < 3)
            return 100;
    }
    else
    {
        if(smoothedHistory.rows < 3*INPUT_DATA_SIZE)
            return 100*INPUT_DATA_SIZE;
    }
    int startIdx = smoothedHistory.rows - window;
    int endIdx = smoothedHistory.rows;
    cv::Vec4f line = get_linear_regression(startIdx, endIdx, false);
    if(use_per_frame_eval)
        return -reg_slope(line)*100;
    else
        return -reg_slope(line)*100*INPUT_DATA_SIZE;
}

float MetaM::getLearningError()
{
    return smoothedHistory.at<float>(smoothedHistory.rows-1);
}

float MetaM::getLearningUncertainty()
{
    /* update smoothed history */
    float smoothingfactor = min(smooth, uncertaintyHistory.rows);
    float smoothedUncertainty = sum(uncertaintyHistory(Rect(0,
                                   uncertaintyHistory.rows-smoothingfactor,
                                   1,
                                   smoothingfactor))).val[0];
    smoothedUncertainty /= smoothingfactor;
    return smoothedUncertainty;
}

Plot MetaM::displayClusterError()
{
    /* For each sample of the window, calculate mean error rate */
    vector<Point2f> meanErrors;
    int startIdx = smoothedHistory.rows - window;
    int endIdx = smoothedHistory.rows;
    Vec4f line = get_linear_regression(startIdx, endIdx,true);
    meanErrors =  getMeanErrorValuePoints(0, smoothedHistory.rows);
    if(smoothedHistory.rows < 1)
        return Plot();

    /* Create linear fit curve */
    Mat linfitMat = Mat::ones(1,meanErrors.size(),CV_32F);
    double slope = reg_slope(line);
    Point2f origin = reg_origin(line);
    for(size_t i = 0 ; i < meanErrors.size() ; i++)
    {
        linfitMat.at<float>( i )
                = origin.y+(meanErrors[i].x - origin.x)*slope;
    }

    /* Prepare plot */
    Mat meanErrMat = Mat(meanErrors);
    meanErrMat = meanErrMat.reshape(1,meanErrors.size());
    meanErrMat = meanErrMat.t();
    meanErrMat.push_back(linfitMat);
    meanErrMat = meanErrMat.t();
    Plot errorPlot(meanErrMat);
    return errorPlot;
}


vector<Point2f> MetaM::getMeanErrorValuePoints(int firstIdx, int lastIdx)
{
    if(firstIdx < 0)
        firstIdx = 0;

    if(lastIdx > smoothedHistory.rows)
        lastIdx = smoothedHistory.rows;

    vector<Point2f> meanErrors;
    for(int i = firstIdx ; i < lastIdx ; i++)
    {
        meanErrors.push_back(Point2f((float)i,smoothedHistory.at<float>(i)));
    }

    return meanErrors;

}

Vec4f MetaM::get_linear_regression(int firstIdx, int lastIdx, bool truncate)
{
    vector<Point2f> meanErrors = getMeanErrorValuePoints(firstIdx,lastIdx);

    if(meanErrors.size() > MAX_SAMPLES_FOR_REGRESSION && truncate)
    {
        random_shuffle(meanErrors.begin(),meanErrors.end());
        meanErrors = vector<Point2f>(meanErrors.begin(),
                                     meanErrors.begin()+MAX_SAMPLES_FOR_REGRESSION);
    }

    Vec4f line;
    if(meanErrors.size() > 1)
    {
        fitLine(meanErrors, line, CV_DIST_L2, 0,0.01,0.01);
    }
    return line;
}


double MetaM::reg_slope(Vec4f line)
{
    return line[1]/line[0];
}

Point2f MetaM::reg_origin(Vec4f line)
{
    return Point2f(line[2], line[3]);
}

void MetaM::resize_input_data(Mat &estimates, Mat &observations)
{
    assert(!estimates.empty());
    estimates.convertTo(estimates, CV_32F);
    observations.convertTo(observations, CV_32F);
    Mat resized_est(INPUT_DATA_SIZE, 1, CV_32F);
    Mat resized_obs(INPUT_DATA_SIZE, 1, CV_32F);
    /* Get vector of the same size as estimates */
    vector<float> v_indices;
    for(int i = 0 ; i < estimates.rows ; i++)
    {
        v_indices.push_back(i);
    }
    std::random_shuffle(v_indices.begin(), v_indices.end());
    /* Fill new matrices */
    for(int i = 0 ; i < INPUT_DATA_SIZE ; i++)
    {
        int k = i % v_indices.size();
        resized_est.at<float>(i) = estimates.at<float>(v_indices[k]);
        resized_obs.at<float>(i) = observations.at<float>(v_indices[k]);
    }

    /* Update */
    estimates = resized_est;
    observations = resized_obs;
}

void MetaM::eval_and_update_history(Mat &estimates, Mat &observations)
{
    /* Calculate error for new samples */
    float eval_score = Eval::evaluate(observations,estimates, evaluation_metrics);
    Mat current_error = Mat(1,1,CV_32F);
    current_error.setTo(1-eval_score);

    if(isnan(eval_score))
    {
        if(!errorHistory.empty())
        {
            current_error = errorHistory.row(errorHistory.rows-1).clone();
        }
    }
    errorHistory.push_back(current_error);

    float uncertainty_score = Eval::getUncertainty(estimates);
    Mat uncertainty = uncertainty_score * Mat::ones(1,1,CV_32F);
    if(isnan(uncertainty_score) && !errorHistory.empty())
    {
        uncertainty = uncertaintyHistory.row(uncertaintyHistory.rows-1).clone();
    }
    uncertaintyHistory.push_back(uncertainty);

    /* update smoothed history */
    float smoothingfactor = min(smooth, errorHistory.rows);
    float oldError = sum(errorHistory(Rect(0,
                                           errorHistory.rows-smoothingfactor,
                                           1,
                                           smoothingfactor))).val[0];
    float smoothedError = oldError/smoothingfactor;
    smoothedHistory.push_back(smoothedError);
}
