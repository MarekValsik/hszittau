#include <QApplication>
#include <cstdio>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <raspicam/raspicam_cv.h>

using namespace cv;
using namespace std;
raspicam::RaspiCam_Cv Camera;
Mat image;
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    printf("Hello everyone :-)\n");
    Camera.set( CV_CAP_PROP_FORMAT, CV_8UC3 );
    Camera.set( CV_CAP_PROP_FRAME_WIDTH, 640 );
    Camera.set( CV_CAP_PROP_FRAME_HEIGHT, 480 );
    //Camera.set(CV_CAP_PROP_FPS, 15);
if (!Camera.open()) {cerr<<"Error opening the camera"<<endl;return -1;}
    namedWindow("main", WINDOW_AUTOSIZE);
    while(1) {
        Camera.grab();
        Camera.retrieve (image);
        imshow("main", image);
        if (waitKey(10)>=0) break;
    }
    cvDestroyWindow("main");
    return (0);
}
