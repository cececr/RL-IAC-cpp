/**
 * \file bottomupsaliencymethod.h
 * \brief BottomUpSaliencyMethod
 * \author Céline Craye
 * \version 0.1
 * \date 11 / 19 / 2015
 *
 * Custom DOxygen-style template. Provide here a description of the file
 *
 */

#ifndef BOTTOMUPSALIENCYMETHOD_H
#define BOTTOMUPSALIENCYMETHOD_H
#include <cv.h>

class BottomUpSaliencyMethod
{
public:
    BottomUpSaliencyMethod();
    virtual ~BottomUpSaliencyMethod();
    virtual bool loadParam(const char* filename) = 0;
    virtual cv::Mat getSalMap(cv::Mat& image) = 0;
};

#endif // BOTTOMUPSALIENCYMETHOD_H
