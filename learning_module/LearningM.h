/**
 * \file LearningM.h
 * \brief LearningM
 * \author Céline Craye
 * \version 0.1
 * \date Oct 13, 2014
 *
 * This class provides ways to incrementally train and evaluate models related to saliency.
 * Several classifiers are available for training, such as random forest (recommended),
 * online random forest or neural networks.
 * The class provides
 *      - a way to load and save models - load() and save()
 *      - a way to incrementally add data to the dataset - addBatchSamples or addSingleSample
 *      - a way to retrain or update the classifier - train()
 *      - a method to generate saliency maps from feature maps - estimateSaliencyFromFeatureMap
 *
 * On top of the type of classifier you want to use, options are available regarding the
 * dataset. You can choose to balance your dataset (50% salient-not salient), or limit the size
 * of the training data. You need to specify these setups during initialization
 *
 */

#ifndef LEARNINGM_H_
#define LEARNINGM_H_

#include <cv.h>
#include <ml.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "classifier.h"

using namespace std;

class LearningM {
public:
	static const int RANDOM_TREE = 0;
	static const int NEURAL_NET = 1;
	static const int SVM = 2;
	static const int My_RANDOM_TREE = 3;
    static const int ONLINE_RANDOM_TREE = 4;
    static const int NONE = -1;

    /* constructor and init function */
    LearningM(int max_samples = 0,
              bool make_balanced_dataset = false,
              bool use_random_replace = false);
    bool init(int classif,
              int max_samples = 0,
              bool make_balanced_dataset = false,
              bool use_random_replace = false,
              string param_file = "");

    /* destructor and reset function */
	virtual ~LearningM();
    bool reset();

    /* Load and save functions */
    bool load(const char* filename);
    bool save(const char* filename);
    bool loadModel(const char* filename);
    bool saveModel(const char* filename);
    bool loadDataset(const char* filename);
    bool saveDataset(const char* filename);
    bool loadParams(const char* filename);
    bool saveParams(const char* filename);

    /* Dataset-related function */
    bool initTrainingSet(cv::Mat trainingData, cv::Mat trainingClasses);
    bool addBatchSamples(cv::Mat samples, cv::Mat labels);
    bool addSingleSample(cv::Mat sample, cv::Mat label);

    /* training and prediction functions */
	bool train();
    cv::Mat estimateSaliencyFromFeatureMap(vector<cv::Mat> & featureMaps,
                                       int subsampling_fact = 1,
                                       int enhance_display_factor = 0);
    cv::Mat estimateSaliencyFromFeatureMap(vector<cv::Mat> & featureMaps,
                                       cv::Mat superpixels,
                                       int enhance_display_factor = 0);
    cv::Mat predict(cv::Mat samples, bool useprobs = true);

    /* Getters */
    bool isInit();
    bool isTrained();
    int getClassifierType();
    Classifier* getClassifier();
    cv::Mat getTrainingData();

private:
    int max_samples; // max number of samples in the dataset
    bool make_balanced_dataset; // make a 50-50 balanced dataset
    bool use_random_replace; // delete exceeding samples randomly
    int classifierType; // the classifier we want to use
    Classifier* classif; // pointer to our classifier
    cv::Mat trainingData; // the samples dataset
    cv::Mat trainingClasses; // the labels dataset

};

#endif /* LEARNINGM_H_ */
