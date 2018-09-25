/*
 * MetaM.h
 *
 *  Created on: Nov 10, 2014
 *      Author: celine
 */

#ifndef METAM_H_
#define METAM_H_
#include <cv.h>
//#include "LearningM.h"
#include "../feature_extractor/FeatureExtractor.h"
#include "../learning_module/LearningM.h"
#include "Plot.h"

//using namespace cv;
using namespace std;

class MetaM {
    static const int MAX_SAMPLES_FOR_REGRESSION = 1000;
public:
    static const int INPUT_DATA_SIZE = 50;
    MetaM(int evaluation_metrics = 0,
          bool use_per_frame_eval = true,
          int window = 15 , int smooth = 30);
	virtual ~MetaM();

    bool updateErrors(cv::Mat estimates, cv::Mat observations);
    float getLearningProgress();
    float getLearningError();
    float getLearningUncertainty();
    Plot displayClusterError();

    int getNsamples();
    void setWindow(int window);
    void setSmooth(int smooth);

private:
    cv::Mat errorHistory;
    cv::Mat smoothedHistory;
    cv::Mat trueLabelsHistory;
    cv::Mat estimatesHistory;
    cv::Mat uncertaintyHistory;
    int window;
    int smooth;
    int evaluation_metrics;
    bool use_per_frame_eval;

    vector<cv::Point2f> getMeanErrorValuePoints(int firstIdx, int lastIdx);
    cv::Vec4f get_linear_regression(int firstIdx, int lastIdx, bool truncate = false);
    vector<cv::Point2f> getValuePointList(int firstIdx, int lastIdx);
    double reg_slope(cv::Vec4f line);
    cv::Point2f reg_origin(cv::Vec4f line);
    void resize_input_data(cv::Mat & estimates, cv::Mat & observations);
    void eval_and_update_history(cv::Mat & estimates, cv::Mat & observations);
};

#endif /* METAM_H_ */
