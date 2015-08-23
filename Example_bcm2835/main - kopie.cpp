#include <QApplication>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <raspicam/raspicam_cv.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
	CvCapture *avi = cvCaptureFromCAM(-1);
//	CvCapture *avi = cvCreateCameraCapture( 0 );
//	VideoCapture cap(0); 	
	cvSetCaptureProperty(avi, CV_CAP_PROP_FRAME_WIDTH, 640);
	cvSetCaptureProperty(avi, CV_CAP_PROP_FRAME_HEIGHT, 480);
	cvSetCaptureProperty(avi, CV_CAP_PROP_FPS, 15);
	cvNamedWindow( "main", 0);
	for(;;)
	{
		IplImage* img = cvQueryFrame(avi);
		if (!img){
			break;
		}
	}
    cvReleaseCapture(&avi);
    cvDestroyWindow( "main");
    return (0);
}
