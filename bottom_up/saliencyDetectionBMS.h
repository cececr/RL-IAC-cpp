/*****************************************************************************
*	Implemetation of the saliency detction method described in paper
*	"Saliency Detection: A Boolean Map Approach", Jianming Zhang,
*	Stan Sclaroff, ICCV, 2013
*
*	Copyright (C) 2013 Jianming Zhang
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*	If you have problems about this software, please contact: jmzhang@bu.edu
*******************************************************************************/

#ifndef SALIENCYDETECTIONBMS_H_
#define SALIENCYDETECTIONBMS_H_

#include <fstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "bottomupsaliencymethod.h"
using namespace cv;
using namespace std;

class saliencyMapBMS :public BottomUpSaliencyMethod
{
public:
	saliencyMapBMS (const Mat& src, const int dw1, const int ow, const bool nm, const bool hb);
	saliencyMapBMS ();
    virtual ~saliencyMapBMS();
    bool loadParam(const char* filename = 0);
    cv::Mat getSalMap(cv::Mat& image);

	Mat getSaliencyMap();
	void computeSaliency(float step);
	void calculateSaliencyMap(const Mat* src, Mat* dst);
private:
	Mat registerPosition(const Mat& bm);
	Mat getAttentionMap(const Mat& bm);
	Mat _sm;
	Mat _src;
	vector<Mat> _feature_maps;
	RNG _rng;
	int _dilation_width_1;
	int _opening_width;
    bool _normalize;
    bool _handle_border;
};


#endif


