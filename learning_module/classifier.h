/**
 * \file classifier.h
 * \brief classifier
 * \author Céline Craye
 * \version 0.1
 * \date 2014
 *
 * Abstract class for classifiers and associated implementations
 * Each implementation basically has a load and save method, as
 * well as a training and prediction methods. Most functions of this
 * class are typically wrapped by LearningM class.
 *
 */

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <cv.h>
#include <ml.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "MyRandomForest.h"
#include "Onlinerandomforest.h"

using namespace std;

class Classifier {
public:
    Classifier();
    virtual ~Classifier();
    virtual bool init(string param_file) = 0;
    virtual bool load(const char* filename) = 0;
    virtual bool save(const char* filename) = 0;
    virtual bool train(cv::Mat & samples, cv::Mat & classes) = 0;
    virtual cv::Mat predict(cv::Mat samples,bool useprobs) = 0;
    bool isTrained();
    bool isInit();
    int getSingleValue();
protected:
    bool TRAINED;
    bool INIT;
    int SINGLEVALUE;
};

class RFClassifier: public Classifier {
    bool init(string param_file = "");
    bool load(const char* filename);
    bool save(const char* filename);
    bool train(cv::Mat & samples, cv::Mat & classes);
    cv::Mat  predict(cv::Mat samples,bool useprobs = false);
    bool save_params(const char* filename);
    bool load_params(const char* filename);
private:
    CvRTrees rf;
    CvRTParams rf_params;
};

class MyRFClassifier: public Classifier {
    bool init(string param_file = "");
    bool load(const char* filename);
    bool save(const char* filename);
    bool train(cv::Mat& samples, cv::Mat& classes);
    cv::Mat  predict(cv::Mat samples,bool useprobs = false);
    bool save_params(const char* filename);
    bool load_params(const char* filename);
public:
    cv::Mat nearestNeighbors(cv::Mat samples);
    cv::Mat noveltyScore(cv::Mat samples);
    void initClusters(cv::Mat trainData, int K);
    cv::Mat getClusters(cv::Mat samples);
private:
    MyRandomForest rf;
    RTParams rf_params;
};

class MLPClassifier: public Classifier {
    bool init(string param_file = "");
    bool load(const char* filename);
    bool save(const char* filename);
    bool train(cv::Mat& samples, cv::Mat& classes);
    cv::Mat  predict(cv::Mat samples,bool useprobs = false);
    bool save_params(const char* filename);
    bool load_params(const char* filename);
private:
    CvANN_MLP mlp;
    CvANN_MLP_TrainParams mlp_params;
};

class OnlineRFClassifier: public Classifier {
    bool init(string param_file = "");
    bool load(const char* filename);
    bool save(const char* filename);
    bool train(cv::Mat& samples, cv::Mat& classes);
    cv::Mat  predict(cv::Mat samples,bool useprobs = false);
    bool save_params(const char* filename);
    bool load_params(const char* filename);
private:
    OnlineRandomForest rf;

};
#endif // CLASSIFIER_H
