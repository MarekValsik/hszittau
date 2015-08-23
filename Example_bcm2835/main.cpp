#include <QApplication>
#include <cstdio>
#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;
//load driver!!!
// sudo modprobe bcm2835-v4l2
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    VideoCapture cap(0);
    if( !cap.isOpened() )  {
    printf("Camera failed to open!\n");
        return -1;
    }
    namedWindow("main", WINDOW_AUTOSIZE);
    Mat frame;
    for(;;)
    {
        cap >> frame;
        imshow("main", frame);
        if(waitKey(30) >= 0) break;
    }
    cvDestroyWindow("main");
    return (0);
}
