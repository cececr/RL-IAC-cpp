#ifndef PRINTDEBUG_H
#define PRINTDEBUG_H
#include <cv.h>
#include <highgui.h>
#include <time.h>

//using namespace cv;
using namespace std;

namespace PrintDebug
{
    inline void print(string msg = "")
    {
        cout << msg << endl;
    }

    inline void print(int m, string msg = "")
    {
        cout << msg << " : ";
        cout << m;
        cout << endl;
    }

    inline void print(bool m, string msg = "")
    {
        cout << msg << " : ";
        cout << m;
        cout << endl;
    }

    inline string type2str(int type) {
      string r;

      uchar depth = type & CV_MAT_DEPTH_MASK;
      uchar chans = 1 + (type >> CV_CN_SHIFT);

      switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
      }

      r += "C";
      r += (chans+'0');

      return r;
    }
    inline void print(cv::Rect r, string msg = "")
    {
        cout << msg << " : ";
        cout << " x=" << r.x ;
        cout << ", y=" << r.y ;
        cout << ", width=" << r.width ;
        cout << ", height=" << r.height ;
        cout << endl;
    }
    inline void print(cv::Point r, string msg = "")
    {
        cout << msg << " : ";
        cout << " x=" << r.x ;
        cout << ", y=" << r.y ;
        cout << endl;
    }
    inline void print(cv::Point2f r, string msg = "")
    {
        cout << msg << " : ";
        cout << " x=" << r.x ;
        cout << ", y=" << r.y ;
        cout << endl;
    }
    inline void print(cv::Mat m, string msg = "")
    {

        double minV=0, maxV=0;
        if(m.cols > 0)
            minMaxLoc(m,&minV,&maxV);

        cout << msg << " : ";
        cout << " rows=" << m.rows ;
        cout << ", cols=" << m.cols ;
        cout << ", channels=" << m.channels() ;
        cout << ", type=" << type2str(m.type()) ;
        cout << ", min=" << minV ;
        cout << ", max=" << maxV ;
        cout << endl;
    }

    inline void printMat(cv::Mat m, string msg = "")
    {
        cout << msg << " : ";
        cout << " length=" << m.size() ;
        cout << m;
        cout << endl;
    }

    inline void print(vector<int> v, string msg = "")
    {
        cout << msg << " : ";
        cout << " length=" << v.size() ;
        cout << cv::Mat(v).t();
        cout << endl;
    }
    inline void print(vector<double> v, string msg = "")
    {
        cout << msg << " : ";
        cout << " length=" << v.size() ;
        cout << cv::Mat(v).t();
        cout << endl;
    }
    inline void print(vector<float> v, string msg = "")
    {
        cout << msg << " : ";
        cout << " length=" << v.size() ;
        cout << cv::Mat(v).t();
        cout << endl;
    }

    inline void print(vector<cv::Point> v, string msg = "")
    {
        cout << msg << " : ";
        cout << " length=" << v.size() ;
        cout << " type=Point";
        cout << endl;
    }

    inline void print(cv::Vec3f v, string msg = "")
    {
        cout << msg << " : ";
        cout << " [ " << v[0] ;
        cout << " , " << v[1];
        cout << " , " << v[2] << " ]";
        cout << endl;
    }

    inline void print(cv::Vec3b v, string msg = "")
    {
        cout << msg << " : ";
        cout << " [ " << v[0] ;
        cout << " , " << v[1];
        cout << " , " << v[2] << " ]";
        cout << endl;
    }


    inline void show(cv::Mat m_o, string msg = "")
    {
        cv::Mat m = m_o.clone();
        if(m.cols == 0)
        {
            cout << msg << " : Matrix must not be empty to be displayed" << endl;
            return;
        }

        if( !(m.type() == CV_8U || m.type() == CV_8UC1  || m.type() == CV_8UC3) )
        {
            double minV, maxV;
            cv::minMaxLoc(m,&minV,&maxV);
            if(minV == maxV)
            {
                minV = 0;
            }
            if(m.channels() == 1)
            {
                m.convertTo(m,CV_32F);
                m = (m-minV)/(maxV-minV)*255;
                m.convertTo(m,CV_8U);
            }
            else
            {
                cout << msg << " : Matrix must be CV_8U or single channel to be displayed" << endl;
            }

        }
        cv::imshow(msg,m);
        cv::waitKey(5);
    }

    inline void showAndStop(cv::Mat m, string msg = "")
    {
        show(m,msg);
        cv::waitKey();
    }
    inline void getTimeDiff(clock_t t, string msg = "")
    {
        cout << msg << " : ";
        cout << (float)(clock()-t)/CLOCKS_PER_SEC*1000 << " ms" ;
        cout << endl;
    }
}


#endif // PRINTDEBUG_H

