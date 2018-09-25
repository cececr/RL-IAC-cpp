/**
 * \file Onlinerandomforest.h
 * \brief OnlineRandomForest
 * \author Céline Craye
 * \version 0.1
 * \date 7 / 15 / 2015
 *
 * Class for OpenCV-based Online Random Forest.
 * This is a very simple version where all trees are not upgraded at the same time
 * and redundant data is filtered out
 *
 */

#ifndef ONLINERANDOMFOREST_H
#define ONLINERANDOMFOREST_H

#include <opencv2/ml/ml.hpp>

//using namespace cv;
using namespace std;

class OnlineRandomForest
{
public:
    OnlineRandomForest();
    ~OnlineRandomForest();
    bool init(CvRTParams params, float training_ratio = 0.8,
              int nb_tree_per_update = 4, bool online_saving = false);
    bool clear();
    bool reset_learning();
//    bool clear_tree(int treeIdx);
    bool offline_train(cv::Mat &trainingData, cv::Mat &trainingClasses);
    bool update(cv::Mat &trainingData, cv::Mat &trainingClasses);
    cv::Mat predict(cv::Mat& predictData);
    cv::Mat predict_prob(cv::Mat& predictData);
    bool load(string filename);
    bool load_tree(int treeIdx, string filename);
    bool load_last_updated_trees(string filename);
    bool save(string filename);
    bool save_tree(int treeIdx, string filename);
    bool save_last_updated_trees(string filename);
    bool isInit();
    bool isTrained();
    CvRTParams getParams();
    int getNbClasses();
    float getTrainingRatio();
    float getNbTreePerUpdate();
    vector<int> getLastUpdated();
    bool getOnlineSaving();
    void setLastUpdated(cv::Mat last_updated);
    void setNbClasses(int nb_classes);
private:
    bool train_tree(cv::Mat &trainingData, cv::Mat &trainingClasses, int treeIdx);

    vector<int> last_updated_trees;
    float training_ratio;
    int nb_trees_per_update;
    vector<CvDTree*> forest;
    CvRTParams forest_params;
    CvDTreeParams tree_params;
    int nb_trees;
    int nb_classes;
    bool INIT;
    bool TRAINED;
    bool ONLINE_SAVING;
};

#endif // ONLINERANDOMFOREST_H
