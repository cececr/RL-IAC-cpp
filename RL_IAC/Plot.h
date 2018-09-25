/*
 * Plot.h
 *
 *  Created on: Nov 21, 2014
 *      Author: Céline Craye
 *
 * // Plot Sample
 * //	Mat C = (Mat_<float>(9,1) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
 * //	Mat_<float> C = (Mat_<float>(9,4) << 0,0,-1,-2,
 * //								 1, -1,3,-1,
 * //								 2, 0,-12,0,
 * //								 3, -1,2,1,
 * //								 4, 5,0,2,
 * //								 5, -1,1,3,
 * //								 6, 0,3,4,
 * //								 7, -1,4,5,
 * //								 8, 0,2,6);
 * //	Plot pl(C);
 * //	pl.setSize(Size(300,200));
 * //	pl.setLineStyle(Plot::dotline);
 * //	pl.setColor(Scalar(0,0,255));
 * //	pl.setTitle("Ma courbe");
 * //	Mat a = pl.draw();
 * //
 * //	imshow("a",a);
 * //	waitKey();
 * //	return 0;
 */

#ifndef PLOT_H_
#define PLOT_H_
#include <cv.h>
#include <iostream>
#include <fstream>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

//using namespace cv;
using namespace std;

class Plot {

public:
    static const double BORDER_PERCENT = 0.3;
	static const int    DEF_GRAD_ABS = 8;
	static const int    DEF_GRAD_ORD = 4;
	static const int    BG_COLOR = 255;
	static const int    AXIS_COLOR = 0;
    static const int 	FONT = cv::FONT_HERSHEY_SIMPLEX;

	enum LineStyle  {dots, line, dotline, histogram};
	enum RelativeLoc{up, down, left, right};

	/* Constructors and destructors */
	Plot();
    Plot(cv::Mat data);
    Plot(vector<cv::Point2f> data);
    Plot(vector<float> data);
    Plot(cv::Mat data, cv::Size plotSize, cv::Scalar color, string title, LineStyle style);
	virtual ~Plot();

    Plot clone();

	/* Setting functions */
    void setData(cv::Mat data);
    void setSize(cv::Size plotSize);
	void setTitle(string title);
    void setColor(cv::Scalar color);
	void setLineStyle(LineStyle style);
	void setxrange(double xmin, double xmax);
	void setyrange(double ymin, double ymax);
	void resetxrange();
	void resetyrange();
	void axisOff();
	void axisOn();
	void legendOn();
	void legendOff();
	void gridOn();
	void gridOff();
    void frameOn();
    void frameOff();
	void addLegend(string legend);
    cv::Size getDataSize();
    cv::Mat getData();

	/* Plot drawing function */
    cv::Mat draw();

private:
	double xmin,xmax,ymin,ymax;
    cv::Mat data;
    cv::Scalar color;
	string title;
    cv::Size plotSize;
	LineStyle style;
	int NB_GRAD_ABS;
	int NB_GRAD_ORD;
	bool useAxis;
	bool useLegend;
	bool useGrid;
    bool useFrame;
    vector<cv::Scalar> plotColorList;
	vector<string> legendList;

	void defaultVals();
	template<typename IMAGE_TYPE>
    void drawPoints(cv::Mat& data_representation,
					double dataMin, double dataMax, double axisMin, double axisMax);
    void drawAxis(cv::Mat& plot_representaiton,
					double dataMin, double dataMax, double axisMin, double axisMax);
    void drawArrow(cv::Mat& image, cv::Point p, cv::Point q, cv::Scalar color,
			int arrowMagnitude = 9, int thickness=1, int line_type=8, int shift=0);
    void drawText(cv::Mat& image, string text, cv::Point reference, RelativeLoc loc, float fontCoeff = 1000);
    void drawLegend(cv::Mat& plot_representaiton,cv::Point pt,vector<cv::Scalar> plotColorList);
    void drawFrame(cv::Mat& plot_representation);
	template<typename T> string num2str(T val);
};

class Subplot {
public:
    static const int DEF_WIDTH = 1400;
    static const int DEF_HEIGHT = 1000;
    Subplot();
    Subplot(int nrows, int ncols,  cv::Size plotSize = cv::Size(DEF_WIDTH,DEF_HEIGHT));
    Subplot(vector<Plot> plotList);
    virtual ~Subplot();

    void setPlotSize(cv::Size plotSize);
    void setRowsCols(int rows, int cols);
    void addPlot(Plot plot);
    void addPlots(vector<Plot> plots);

    int getNbPlots();

    cv::Mat draw();

private:
    int nrows;
    int ncols;
    cv::Size plotSize;
    vector<Plot> plotList;

    void addToSubplot(cv::Mat& subplot_representation, cv::Mat plot_representation, int row, int col);
};

#endif /* PLOT_H_ */
