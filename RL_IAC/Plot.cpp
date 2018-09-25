/*
 * Plot.cpp
 *
 *  Created on: Nov 21, 2014
 *      Author: Céline Craye
 */

#include "Plot.h"

using namespace cv;

Plot::Plot()
{
	defaultVals();
}

Plot::~Plot() {
}

Plot::Plot(Mat data)
{
	defaultVals();
	setData(data);
}

Plot::Plot(vector<Point2f> data)
{
	Mat matData(data);
	matData = matData.reshape(1,data.size());
	defaultVals();
	setData(matData.clone());
}
Plot::Plot(vector<float> data)
{
	Mat matData(data);
	defaultVals();
	setData(matData.clone());
}

void Plot::defaultVals()
{
	NB_GRAD_ABS = 0;
	NB_GRAD_ORD = 0;
	useAxis = true;
	useLegend = false;
	useGrid = false;
    useFrame = false;
	this->style = dotline;
	this->plotSize = Size(600,400);
	this->title = "";
	this->color = Scalar(255,0,0);
	xmin = xmax = ymin = ymax = 0;
}

Plot::Plot(Mat data, Size plotSize, Scalar color, string title, LineStyle style)
{
	defaultVals();
	setData(data);
	setSize(plotSize);
	setColor(color);
	setTitle(title);
	setLineStyle(style);
}

Plot Plot::clone()
{
	Plot clone(data, plotSize, color, title, style);
	return clone;
}

void Plot::setData(Mat data)
{
	/*TODO: use templates to make generic data collection and avoid conversion*/
	data.convertTo(data,CV_32F);
	this->data = data;
	NB_GRAD_ORD = DEF_GRAD_ORD;
	if(data.cols>1)
	{
		(data.rows > DEF_GRAD_ABS) ?
				NB_GRAD_ABS = DEF_GRAD_ABS : NB_GRAD_ABS = data.rows-1;
	}
}

void Plot::setSize(Size plotSize)
{
	this->plotSize = plotSize;
}

void Plot::setTitle(string title)
{
	this->title = title;
}

void Plot::setColor(Scalar color)
{
	this->color = color;
}

void Plot::setLineStyle(LineStyle style)
{
	this->style = style;
}

void Plot::setxrange(double xmin, double xmax)
{
	if(xmin < xmax)
	{
		this->xmin = xmin;
		this->xmax = xmax;
	}
}
void Plot::setyrange(double ymin, double ymax)
{
	if(ymin < ymax)
	{
		this->ymin = ymin;
		this->ymax = ymax;
	}
}

void Plot::resetxrange()
{
	this->xmin = 0;
	this->xmax = 0;
}
void Plot::resetyrange()
{
	this->ymin = 0;
	this->ymax = 0;
}

void Plot::axisOff()
{
	useAxis = false;
}

void Plot::axisOn()
{
	useAxis = true;
}

void Plot::legendOn()
{
	useLegend = true;
}

void Plot::legendOff()
{
	useLegend = false;
}

void Plot::gridOn()
{
	useGrid = true;
}

void Plot::gridOff()
{
    useGrid = false;
}

void Plot::frameOn()
{
    useFrame = true;
}

void Plot::frameOff()
{
    useFrame = false;
}

void Plot::addLegend(string str)
{
	legendList.push_back(str);
    useLegend = true;
}

Size Plot::getDataSize()
{
    return data.size();
}

Mat Plot::getData()
{
    return data;
}


Mat Plot::draw()
{
	assert(data.channels() == 1);
	assert(data.cols > 0);

	/* Find min and max for optimal representation */
	if(xmin == 0 && xmax == 0 && data.cols > 1)
	{
		minMaxLoc(data.col(0), &xmin, &xmax);
	}

	if(ymin == 0 && ymax == 0)
	{
		if(data.cols == 1)
		{
			minMaxLoc(data.col(0), &ymin, &ymax); // last col is always data
		}
		else
		{
			minMaxLoc(data(Rect(1,0,data.cols-1,data.rows)), &ymin, &ymax);
		}
	}
	if(ymin == ymax)
	{
		ymin = ymin-1;
		ymax = ymax+1;
	}


	/* create output representation */
	const int frameHeight = (1+BORDER_PERCENT)*plotSize.height;
	const int frameWidth = (1+BORDER_PERCENT)*plotSize.width;
	cv::Mat plot_representation(frameHeight, frameWidth,CV_8UC3, cv::Scalar(BG_COLOR,BG_COLOR,BG_COLOR));

	/* Draw and display values on axis */
	if(useAxis)
	{
		drawAxis(plot_representation, ymin, ymax, xmin, xmax);
	}
    if(useFrame)
    {
        drawFrame(plot_representation);
    }

	/* Draw points */
	Rect r(plotSize.width*BORDER_PERCENT/3,plotSize.height*2*BORDER_PERCENT/3,
			 plotSize.width,plotSize.height);
	plotColorList.clear();
	Mat data_representation = plot_representation(r);
	/*TODO: Use templates to call this function in a generic way */
	drawPoints<float>(data_representation, ymin, ymax, xmin, xmax);

	/* Add to frame */
	data_representation.copyTo(plot_representation(r));



	/* If more than one plot, display legend */
	if(useLegend)
	{
		drawLegend(plot_representation, Point(plotSize.width,plotSize.height*BORDER_PERCENT/3),plotColorList);
	}

	/* Display title */
    drawText(plot_representation,title,Point(frameWidth/2,plotSize.height*BORDER_PERCENT/3),down, 700);

	return plot_representation;
}


template<typename IMAGE_TYPE>
void Plot::drawPoints(Mat& data_representation, double dataMin, double dataMax, double axisMin, double axisMax)
{
	if(data_representation.empty())
	{
		data_representation = Mat(plotSize.height, plotSize.width,
				CV_8UC3, cv::Scalar(BG_COLOR,BG_COLOR,BG_COLOR));
	}

	 Point prevdatapoint;
	 int ndatacols = 0;
	 if(data.cols == 1)
	 {
		 ndatacols = 1;
	 }
	 else
	 {
		 ndatacols = data.cols -1;
	 }
	 for(int col = 0 ; col < ndatacols ; col++)
	 {
		 /* Choose color for plot */
		 Scalar plotColor;
		 if(data.cols <= 2)
		 {
			 plotColor = Scalar(color);
		 }
		 else
		 {
			 Mat matColor = (Mat_<uchar>(1,3) << (double)col/(double)(ndatacols)*255,255,255);
			 matColor = matColor.reshape(3,1);
			 cvtColor(matColor,matColor,CV_HSV2BGR_FULL);
			 plotColor = Scalar(matColor.at<Vec3b>(0,0)) ;
		 }
		 plotColorList.push_back(plotColor);

		 for (int row = 0; row < data.rows; row++)
		 {
			 double datavalue = data.at<IMAGE_TYPE>(row, data.cols-1-col);
//			 double datavalue = data.at<IMAGE_TYPE>(row, col);
			 double norm_datavalue = (datavalue-dataMin)/(dataMax-dataMin);
			 int yValue = (double)data_representation.rows * (1.0 - norm_datavalue);
			 int xValue;

			 if(data.cols == 1) // if only data is provided
			 {
				 xValue = (double)row/(data.rows) * data_representation.cols;
			 }
			 else
			 {
				 double axisValue = data.at< IMAGE_TYPE>(row, 0);
				 double norm_axisvalue = (axisValue-axisMin)/(axisMax-axisMin);
				 xValue = norm_axisvalue * data_representation.cols;
			 }
			 Point datapoint(xValue,yValue);

			 /* Draw */
			 if(style == dots ||style == dotline)
			 {
				 circle(data_representation,datapoint,2,plotColor,CV_FILLED);
			 }
			 if(style == histogram)
			 {
				 int rectwidth = 1/(double)(data.rows-1) * data_representation.cols +1;
				 Rect r(datapoint.x-rectwidth/2,datapoint.y-1,
						 rectwidth,data_representation.rows-datapoint.y+2);
				 cv::rectangle(data_representation,r,plotColor,CV_FILLED);
				 cv::rectangle(data_representation,r,Scalar(AXIS_COLOR),1);
			 }
			 if(row > 0)
			 {
				 if(style == line ||style == dotline )
				 {
					 cv::line(data_representation, prevdatapoint, datapoint , plotColor);
				 }
			 }

			 prevdatapoint = datapoint;
		 }
	 }
}

void Plot::drawAxis(Mat& plot_representation, double dataMin, double dataMax, double axisMin, double axisMax)
{
	/* Check image is not empty*/
	assert(!plot_representation.empty());

	/* Get dimension of graphic objects */
	int arrowLength = plotSize.width/50;
	int gradLength = plotSize.width/50;
	Scalar axisColor(AXIS_COLOR,AXIS_COLOR,AXIS_COLOR);

	/* Draw arrows */
	Point origin(plotSize.width*(BORDER_PERCENT/3),plotSize.height*(1.0 + 2*BORDER_PERCENT/3));
	Point max_abscissa( plotSize.width*(1.0 + 2*BORDER_PERCENT/3) ,origin.y);
	Point max_ordinate( origin.x, plotSize.height*(BORDER_PERCENT/3));
	drawArrow(plot_representation,origin,max_abscissa,axisColor,arrowLength);
	drawArrow(plot_representation,origin,max_ordinate,axisColor,arrowLength);

	int ndash = 20; // if useGrid
	/* Draw graduations and values for x */
	if(data.cols > 1)
	{
		for(int i = 1 ; i < NB_GRAD_ABS+1 ; i++)
		{
			int gradAbscissa = (double)(NB_GRAD_ABS*origin.x + i*(plotSize.width))/(double)NB_GRAD_ABS;
			Point pt1(gradAbscissa,origin.y-gradLength/2);
			Point pt2(gradAbscissa,origin.y+gradLength/2);
			cv::line(plot_representation,pt1,pt2,axisColor);
			double textVal = (double)((NB_GRAD_ABS-i)*axisMin + i*axisMax)/(double)(NB_GRAD_ABS);
			drawText(plot_representation,num2str(textVal),pt2,down);
			if(useGrid)
			{

				for(int j = 1 ; j < ndash-1 ; j++ )
				{
					pt1.y = origin.y + j*(max_ordinate.y - origin.y)/(float)ndash;
					pt2.y = origin.y + (j+0.3)*(max_ordinate.y - origin.y)/(float)ndash;
					cv::line(plot_representation,pt1,pt2,axisColor);
				}
			}
		}
	}

	/* Draw graduations and values for y */
	for(int i = 0 ; i < NB_GRAD_ORD+1 ; i++)
	{
		int gradOrdinate = (double)(NB_GRAD_ORD*origin.y - i*(plotSize.height))/(double)NB_GRAD_ORD;
		Point pt1(origin.x-gradLength/2,gradOrdinate);
		Point pt2(origin.x+gradLength/2,gradOrdinate);
		cv::line(plot_representation,pt1,pt2,axisColor);
		double textVal = (double)((NB_GRAD_ORD-i)*dataMin + i*dataMax)/(double)(NB_GRAD_ORD);
		drawText(plot_representation,num2str(textVal),pt1,left);
		if(useGrid)
		{
			float ratio = abs((float)(max_ordinate.y - origin.y)/(float)(max_abscissa.x - origin.x));
			ndash = 20./ratio;
			for(int j = 1 ; j < ndash-1 ; j++ )
			{
				pt1.x = origin.x + j*(max_abscissa.x - origin.x)/(float)ndash;
				pt2.x = origin.x + (j+0.3)*(max_abscissa.x - origin.x)/(float)ndash;
				cv::line(plot_representation,pt1,pt2,axisColor);
			}
		}
	}
}

void Plot::drawArrow(Mat& image, Point p, Point q, Scalar color,
		int arrowMagnitude , int thickness, int line_type, int shift)
{
	assert(!image.empty());
    //Draw the main line
    cv::line(image, p, q, color, thickness, line_type, shift);
    //compute the angle alpha
    double angle = atan2((double)p.y-q.y, (double)p.x-q.x);
    //compute the coordinates of the first segment
    p.x = (int) ( q.x +  arrowMagnitude * cos(angle + CV_PI/6));
    p.y = (int) ( q.y +  arrowMagnitude * sin(angle + CV_PI/6));
    //Draw the first segment
    cv::line(image, p, q, color, thickness, line_type, shift);
    //compute the coordinates of the second segment
    p.x = (int) ( q.x +  arrowMagnitude * cos(angle - CV_PI/6));
    p.y = (int) ( q.y +  arrowMagnitude * sin(angle - CV_PI/6));
    //Draw the second segment
    cv::line(image, p, q, color, thickness, line_type, shift);
}

void Plot::drawText(Mat& image, string text, Point reference, RelativeLoc loc, float fontCoeff)
{
	assert(!image.empty());
    double fontRatio = (double)plotSize.width/fontCoeff;
	int baseline;
	Size textSize = getTextSize(text, FONT, fontRatio, 1, &baseline);
	Point ptTitle;
	switch(loc)
	{
	case up:
		ptTitle.x = reference.x-textSize.width/2;
		ptTitle.y = reference.y;
		break;
	case down:
		ptTitle.x = reference.x-textSize.width/2;
		ptTitle.y = reference.y+textSize.height;
		break;
	case left:
		ptTitle.x = reference.x-textSize.width;
		ptTitle.y = reference.y+textSize.height/2;
		break;
	case right:
		ptTitle.x = reference.x;
		ptTitle.y = reference.y+textSize.height/2;
		break;
	}
	putText(image, text, ptTitle, FONT,fontRatio, Scalar(AXIS_COLOR));
}

void Plot::drawLegend(Mat& plot_representaiton,Point pt, vector<Scalar> plotColorList)
{
	Point ptTitle = pt;
	for(size_t i = 0 ; i < plotColorList.size() ; i++)
	{
        size_t plotidx = plotColorList.size() - 1 - i;
		ptTitle = pt;
		ptTitle.y = pt.y + 10*plotidx;
		string text = "data " + num2str(plotidx);
		if(plotidx < legendList.size())
		{
			text = legendList[plotidx];
		}
		circle(plot_representaiton,ptTitle, 3, plotColorList[i],CV_FILLED);
		ptTitle.x = ptTitle.x + 10;
		drawText(plot_representaiton, text, ptTitle, right);
    }
}

void Plot::drawFrame(Mat &plot_representation)
{
    cv::line(plot_representation,Point(0,0),Point(0,plot_representation.rows),Scalar(255,0,0),3);
    cv::line(plot_representation,Point(0,0),Point(plot_representation.cols,0),Scalar(255,0,0),3);
    cv::line(plot_representation,Point(plot_representation.cols,plot_representation.rows),
                             Point(0,plot_representation.rows),Scalar(255,0,0),3);
    cv::line(plot_representation,Point(plot_representation.cols,plot_representation.rows),
                             Point(plot_representation.cols,0),Scalar(255,0,0),3);
}


template<typename T> string Plot::num2str(T val)
{
	std::ostringstream strs;
	strs.precision(2);
	strs << val;
	std::string str = strs.str();
	return str;
}

Subplot::Subplot()
{
    nrows = 0;
    ncols = 0;
    plotSize = Size(DEF_WIDTH,DEF_HEIGHT);
}

Subplot::Subplot(int nrows, int ncols, Size plotSize)
{
    this->ncols = ncols;
    this->nrows = nrows;
    this->plotSize = plotSize;
}

Subplot::Subplot(vector<Plot> plotList)
{
    nrows = 0;
    ncols = 0;
    plotSize = Size(DEF_WIDTH,DEF_HEIGHT);
    this->plotList = plotList;
}

Subplot::~Subplot()
{
}

int Subplot::getNbPlots()
{
    return plotList.size();
}

void Subplot::addPlot(Plot plot)
{
    this->plotList.push_back(plot);
}

void Subplot::addPlots(vector<Plot> plots)
{
    for(size_t i = 0 ; i < plots.size() ; i++)
        this->plotList.push_back(plots[i]);
}

Mat Subplot::draw()
{
    assert(!plotList.empty());

    if(nrows == 0 || ncols == 0)
    {
        /* Assign optimal values to rows and cols */
        int squareUp = ceil(sqrt(plotList.size()));
        ncols = squareUp;
        nrows = ceil((double)plotList.size()/(double)ncols);
    }

    if(plotSize.width == DEF_WIDTH || plotSize.height == DEF_HEIGHT)
    {
        /* Assign optimal plot size */
        if(nrows < ncols)
            plotSize.height = plotSize.height/ncols*nrows;
        if(nrows > ncols)
            plotSize.width = plotSize.width/nrows*ncols;
    }

    Mat subplot_representation;
    for(size_t i = 0 ; i < plotList.size() ; i++)
    {
        Plot plot;
        plot = plotList[i];
        if(plot.getDataSize().width > 0)
        {
            /*If required, adjust size */
            plot.setSize(Size(floor((double)(plotSize.width*(1-Plot::BORDER_PERCENT))/(double)ncols),
                              floor((double)(plotSize.height*(1-Plot::BORDER_PERCENT))/(double)nrows)));
            Mat plot_representation = plot.draw();
            int row = i % nrows;
            int col = floor((double)i/(double)nrows);
            if(col >= ncols) break;
            addToSubplot(subplot_representation,plot_representation,row,col);
        }

    }
    return subplot_representation;
}

void Subplot::addToSubplot(Mat& subplot_representation, Mat plot_representation, int row, int col)
{
    if(subplot_representation.empty())
    {
        Scalar bg_color(Plot::BG_COLOR,Plot::BG_COLOR,Plot::BG_COLOR);
        subplot_representation = Mat(plotSize,plot_representation.type(),bg_color);
    }
    Rect r(col*plot_representation.cols,row*plot_representation.rows,
         plot_representation.cols,plot_representation.rows);

    assert(r.x >= 0 && r.y >=0);
    assert(r.x+r.width <= subplot_representation.cols
            && r.y+r.height <= subplot_representation.rows);

    plot_representation.copyTo(subplot_representation(r));
    return;
}
