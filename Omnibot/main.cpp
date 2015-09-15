/*
Main program for Omnibot
Commented lines (commands/functions) can be used for debugging of this program or for checking states of it
This program was written by Marek Valsik - HSZG + TUL
*/
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
#include <unistd.h>			//Used for UART
#include <fcntl.h>			//Used for UART
#include <termios.h>		//Used for UART
#include <time.h>
#include <termios.h>
#include <math.h>
#include <pthread.h>
using namespace cv;
using namespace std;

const int MAX_SPEED=255;

//basic values for manual control
int speed_manual_forward=40;
int speed_manual_turn=80;
int speed_manual_rotate=30;


int speed_face_rot;
int speed_ball_turn;
int speed_ball_forward=40;
int speed_ball_rot=25;



//varaibles for OpenCV
raspicam::RaspiCam_Cv Camera;
Mat image,frame,frame_face,frame_ball;
Mat frame_face_gray,frame_gray,frame_HSV;
IplImage *ball;
Vec3b hsv_val;
vector<Vec3f> circles;

bool face_ready=0;
bool ball_ready=0;
bool ball_track_start;

int h_min=255;
int s_min=255;
int v_min=255;
int h_max=0;
int s_max=0;
int v_max=0;
int t_h;
int t_s;
int t_v;

int par_h1=100;
int par_h2=40;
int min_rad=20;
int max_rad=200;
//necessary variables for advanced parts of program
int control=1; //1-manual,2-face,3-ball
bool settings=0;
int key;


// variables for threads
pthread_t thread_grab,thread_conv,thread_update,thread_face,thread_scale,thread_ball;
pthread_mutex_t mutex_img = PTHREAD_MUTEX_INITIALIZER;
int ret;


//declaration of functios
void detectFaces(Mat frame);
void showResultImage(Mat frame);

//used haarcascade for face detection
String CASCADE_FACE_NAME = "haarcascade_frontalface_default.xml";
CascadeClassifier face_cascade;

//declaration of threads
void *t_conv(void *arg);
void *t_grab(void *arg);
void *t_update(void *arg);
void *t_ball(void *arg);


//uart communication
int motors[6]={0,0,0,0,0,0};
int last_motors[6]={0,0,0,0,0,0};
int digital[8];
int analog[8];
float analog_cm[8];
int uart0_filestream = -1;
unsigned char tx_buffer[256];
unsigned char rx_buffer[256];
unsigned char message[250];
unsigned char message_number=0x31;
unsigned char CrcTable[] = {
0x00, 0x07, 0x0E, 0x09, 0x1C, 0x1B, 0x12, 0x15, 0x38, 0x3F, 0x36, 0x31,
0x24, 0x23, 0x2A, 0x2D, 0x70, 0x77, 0x7E, 0x79, 0x6C, 0x6B, 0x62, 0x65,
0x48, 0x4F, 0x46, 0x41, 0x54, 0x53, 0x5A, 0x5D, 0xE0, 0xE7, 0xEE, 0xE9,
0xFC, 0xFB, 0xF2, 0xF5, 0xD8, 0xDF, 0xD6, 0xD1, 0xC4, 0xC3, 0xCA, 0xCD,
0x90, 0x97, 0x9E, 0x99, 0x8C, 0x8B, 0x82, 0x85, 0xA8, 0xAF, 0xA6, 0xA1,
0xB4, 0xB3, 0xBA, 0xBD, 0xC7, 0xC0, 0xC9, 0xCE, 0xDB, 0xDC, 0xD5, 0xD2,
0xFF, 0xF8, 0xF1, 0xF6, 0xE3, 0xE4, 0xED, 0xEA, 0xB7, 0xB0, 0xB9, 0xBE,
0xAB, 0xAC, 0xA5, 0xA2, 0x8F, 0x88, 0x81, 0x86, 0x93, 0x94, 0x9D, 0x9A,
0x27, 0x20, 0x29, 0x2E, 0x3B, 0x3C, 0x35, 0x32, 0x1F, 0x18, 0x11, 0x16,
0x03, 0x04, 0x0D, 0x0A, 0x57, 0x50, 0x59, 0x5E, 0x4B, 0x4C, 0x45, 0x42,
0x6F, 0x68, 0x61, 0x66, 0x73, 0x74, 0x7D, 0x7A, 0x89, 0x8E, 0x87, 0x80,
0x95, 0x92, 0x9B, 0x9C, 0xB1, 0xB6, 0xBF, 0xB8, 0xAD, 0xAA, 0xA3, 0xA4,
0xF9, 0xFE, 0xF7, 0xF0, 0xE5, 0xE2, 0xEB, 0xEC, 0xC1, 0xC6, 0xCF, 0xC8,
0xDD, 0xDA, 0xD3, 0xD4, 0x69, 0x6E, 0x67, 0x60, 0x75, 0x72, 0x7B, 0x7C,
0x51, 0x56, 0x5F, 0x58, 0x4D, 0x4A, 0x43, 0x44, 0x19, 0x1E, 0x17, 0x10,
0x05, 0x02, 0x0B, 0x0C, 0x21, 0x26, 0x2F, 0x28, 0x3D, 0x3A, 0x33, 0x34,
0x4E, 0x49, 0x40, 0x47, 0x52, 0x55, 0x5C, 0x5B, 0x76, 0x71, 0x78, 0x7F,
0x6A, 0x6D, 0x64, 0x63, 0x3E, 0x39, 0x30, 0x37, 0x22, 0x25, 0x2C, 0x2B,
0x06, 0x01, 0x08, 0x0F, 0x1A, 0x1D, 0x14, 0x13, 0xAE, 0xA9, 0xA0, 0xA7,
0xB2, 0xB5, 0xBC, 0xBB, 0x96, 0x91, 0x98, 0x9F, 0x8A, 0x8D, 0x84, 0x83,
0xDE, 0xD9, 0xD0, 0xD7, 0xC2, 0xC5, 0xCC, 0xCB, 0xE6, 0xE1, 0xE8, 0xEF,
0xFA, 0xFD, 0xF4, 0xF3
};
//8 bit CRC used in the ATM protocol header. It implements the polynomial x8 + x2 + x + 1 with an initial value of zero.
//CRC() - implements CRC-8-ATM used for checksumming *
//used also in Dukane iQLine (ultrasonic welding machine)    :-)


//function for computing CRC
unsigned char CRC(unsigned char * buf, unsigned char len)
    {
    unsigned char CRC_byte = 0;
    for (int i=0; i < len; i++) {
        CRC_byte = CrcTable[CRC_byte ^ *buf++];
    }
    return CRC_byte;
}


//function for transmitting data
void Transmit(unsigned char len){
    if (uart0_filestream != -1){
        int count = write(uart0_filestream, &tx_buffer[0], len);		//Filestream, bytes to write, number of bytes to write
        if (count < 0){
            printf("UART TX error\n");
        }
    }
}

//decomposition of received string
void Decompose(unsigned char len){
    int pSTX;
    for (int i=len-1; i>=0; i--) {
        if (rx_buffer[i]==2) pSTX=i;
    }

    if ((rx_buffer[pSTX]==2)&(rx_buffer[pSTX+3]=='R')&(rx_buffer[pSTX+1]<=(len-pSTX))) {
        if (rx_buffer[pSTX+4]=='M'){
            //if ((rx_buffer[pSTX+5]=='O')&(rx_buffer[pSTX+6]=='K')) printf("Motors OK\n");
        }

        if (rx_buffer[pSTX+4]=='S'){
            for (int i=0; i<len; i++) {
                digital[i]=rx_buffer[pSTX+5+i];
                analog[i]=rx_buffer[pSTX+13+i];

            }
            //printf("digital: %i %i %i %i %i %i %i %i\n", digital[0],digital[1],digital[2],digital[3],digital[4],digital[5],digital[6],digital[7]);
            //printf("analog: %i %i %i %i %i %i %i %i\n", analog[0],analog[1],analog[2],analog[3],analog[4],analog[5],analog[6],analog[7]);
           // printf("analog_cm: %f %f %f\n", analog_cm[0],analog_cm[1],analog_cm[2]);
        }
    }
    else printf("UART RX error - did you push button on SoccerBoard?\n");
}

//receiving of data
void Receive(){
    if (uart0_filestream != -1)
    {
        // Read up to 255 characters from the port if they are there

        int rx_length = read(uart0_filestream, (void*)rx_buffer, 255);		//Filestream, buffer to store in, number of bytes to read (max)
        if (rx_length < 0)
        {

            //An error occured (will occur if there are no bytes)
        }
        else if (rx_length == 0)
        {
            //No data waiting

        }
        else
        {
            //Bytes received
            rx_buffer[rx_length] = '\0';
            //printf("%i bytes read : %s\n", rx_length, rx_buffer);
            Decompose(rx_length);
        }
    }
}


// composition of the message for robot
void Concatenate(unsigned char len){
    tx_buffer[0]=0x02;
    tx_buffer[1]=len+2;
    tx_buffer[2]=message_number;
    for (int i=0; i<len; i++) {
        tx_buffer[3+i]=message[i];
    }
    tx_buffer[len+3]=CRC(&tx_buffer[1],len+1); //change it on receiver side to &tx_buffer[0]
    Transmit(len+4); //send message
}

//thread for communication with robot
void *t_update(void *arg){

    while (1){

    if (motors!=last_motors){
    message[0]='S';//set
    message[1]='M';//motors
    for (int i=0; i < 6; i++) {
        if (motors[i]<-MAX_SPEED) motors[i]=-MAX_SPEED ;
        if (motors[i]>MAX_SPEED) motors[i]=MAX_SPEED ;
    }
    for (int i=0; i < 6; i++) {
        if (motors[i]<0) message[i*2+2]='B'; else message[i*2+2]='F';
        message[i*2+3]=abs(motors[i]);
    }
    last_motors[0]=motors[0];
    last_motors[1]=motors[1];
    last_motors[2]=motors[2];
    last_motors[3]=motors[3];
    last_motors[4]=motors[4];
    last_motors[5]=motors[5];
    Concatenate(14);
    usleep(10000);
    Receive();
    }
    message[0]='G';//get
    message[1]='S';//sensors
    Concatenate(2);
    usleep(10000);
    Receive();
    }
    pthread_exit((void*)0);
}


//function for face detection
void detectFaces(Mat frame)
{
    vector<Rect> faces;
    if (control==2){
    face_cascade.detectMultiScale(frame_face_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    if (faces.size() != 0) {
        for (size_t i = 0; i < 1 /*faces.size()*/; i++)
        {

            Point center;
            center.x = cvRound((faces[i].x + faces[i].width * 0.5));
            center.y = cvRound((faces[i].y + faces[i].height * 0.5));
            int radius = cvRound((faces[i].width + faces[i].height) * 0.25);
            circle(frame_face, center, radius, Scalar(0, 0, 255), 3, 8, 0);

            speed_face_rot=(center.x-(frame.cols / 2))*75/frame.cols;
            printf("Speed: %i \n",speed_face_rot);

        }
    }
    else speed_face_rot=0;

    motors[0]=speed_face_rot;
    motors[1]=speed_face_rot;
    motors[2]=-speed_face_rot;
    }

    }



// function for grabbing image - fastest way - we always see "realtime" image from camera
void *t_grab(void *arg)
{
    Camera.grab();
    Camera.retrieve (image);

    frame=image;
    //pthread_join(thread_conv,(void **)&ret);
    //pthread_create(&thread_conv,NULL,&t_conv,NULL);
    pthread_exit((void*)0);
}

//thread for face detection
void *t_face(void *arg)
{
    while ((control==2)){
        pthread_join(thread_grab,(void **)&ret);
        //pthread_join(thread_conv,(void **)&ret);
        cvtColor(frame, frame_gray, CV_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);
        resize(frame,frame_face,Size(160,120));
        resize(frame_gray,frame_face_gray,Size(160,120));
        detectFaces(frame_face);
        face_ready=1;
    }
    pthread_exit((void*)0);
}

//thread for ball tracking
void *t_ball(void *arg)
{
    CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));


    while ((control==3)){
        pthread_join(thread_grab,(void **)&ret);
        cvtColor(frame, frame_HSV, CV_BGR2HSV);
        inRange(frame_HSV,Scalar(h_min,s_min,v_min),Scalar(h_max,s_max,v_max),frame_ball);
        //GaussianBlur(frame_ball,frame_ball,Size(9,9),2,2);
        /*
        HoughCircles(frame_ball,circles,CV_HOUGH_GRADIENT,2,frame_ball.rows/4,par_h1,par_h2,min_rad,max_rad);
        for (size_t i = 0; i< circles.size();i++){
                printf("X: %i, Y: %i \n",cvRound(circles[i][0]),cvRound(circles[i][1]));
        }
        */
        ball = new IplImage(frame_ball);
        cvMoments(ball,moments,1);
        double moment01 = cvGetSpatialMoment(moments,0,1);
        double moment10 = cvGetSpatialMoment(moments,1,0);
        double moment00 = cvGetCentralMoment(moments,0,0);
        int x=cvRound(moment10/moment00);
        int y=cvRound(moment01/moment00);

        printf("X: %i, Y: %i \n",x,y);
        cvCircle(ball,cvPoint(x,y),3,CV_RGB(0,255,0),3,8,0);
        if ((moment00>1000)&(ball_track_start)){
            speed_ball_turn=(x-320);
            printf("Speed of turning: %i \n",speed_ball_turn);
            motors[0]=-speed_ball_forward;
            motors[1]=speed_ball_forward;
            motors[2]=-speed_ball_turn;
        }

        else if ((moment00<1000)&(ball_track_start)){

            printf("Speed of rotation: %i \n",speed_ball_turn);
            if (motors[2] < 0)
            {
                motors[0]=speed_ball_rot;
                motors[1]=speed_ball_rot;
                motors[2]=-speed_ball_rot;
            }
            if (motors[2] > 0)
            {
                motors[0]=-speed_ball_rot;
                motors[1]=-speed_ball_rot;
                motors[2]=speed_ball_rot;
            }

        }

        else {
            motors[0]=0;
            motors[1]=0;
            motors[2]=0;


        }
        printf("Moment: %f \n",moment00);
        ball_ready=1;
    }
    pthread_exit((void*)0);
}


//interupt on change of trackbar
void on_trackbar_control(int, void*){
    printf("Control: %i \n",control);
    face_ready=0;
    ball_ready=0;
    if (control==2) {
        pthread_join(thread_ball,(void **)&ret);
        //pthread_create(&thread_conv,NULL,&t_conv,NULL);
        pthread_create(&thread_face,NULL,&t_face,NULL);
        namedWindow("face", WINDOW_AUTOSIZE);
        moveWindow("face",650,0);
    }
    if (control!=2) destroyWindow("face");

    if (control==3) {
        pthread_join(thread_face,(void **)&ret);
        pthread_create(&thread_ball,NULL,&t_ball,NULL);
        namedWindow("ball", WINDOW_AUTOSIZE);
        moveWindow("ball",650,0);
    }
    if (control!=3) destroyWindow("ball");

    if (control==1) {motors[0]=0; motors[1]=0; motors[2]=0;}
    if (control==0) {motors[0]=0; motors[1]=0; motors[2]=0;}
}

//open settings
void on_change_checkbox(int, void*){
    settings=~settings;
    if (settings) {
        namedWindow("setting", WINDOW_AUTOSIZE|CV_GUI_NORMAL);
        resizeWindow("setting",600,300);
        moveWindow("setting",0,0);
        createTrackbar("man_forward","setting",&speed_manual_forward,255,NULL);
        createTrackbar("man_turn","setting",&speed_manual_turn,255,NULL);
        createTrackbar("manual_rotate","setting",&speed_manual_rotate,255,NULL);

        createTrackbar("h_min","setting",&h_min,255,NULL);
        createTrackbar("s_min","setting",&s_min,255,NULL);
        createTrackbar("v_min","setting",&v_min,255,NULL);

        createTrackbar("h_max","setting",&h_max,255,NULL);
        createTrackbar("s_max","setting",&s_max,255,NULL);
        createTrackbar("v_max","setting",&v_max,255,NULL);
    }
    else
        destroyWindow( "setting" );
}

//simple function for manual control
void manual_control(int command){
    switch (command){
    case 1:
        motors[0]=speed_manual_forward;
        motors[1]=-speed_manual_forward;
        motors[2]=speed_manual_turn;
        break;
    case 2:
        motors[0]=speed_manual_forward;
        motors[1]=-speed_manual_forward;
        motors[2]=0;
        break;
    case 3:
        motors[0]=speed_manual_forward;
        motors[1]=-speed_manual_forward;
        motors[2]=-speed_manual_turn;
        break;
    case 4:
        motors[0]=-speed_manual_rotate;
        motors[1]=-speed_manual_rotate;
        motors[2]=speed_manual_rotate;
        break;
    case 5:
        motors[0]=0;
        motors[1]=0;
        motors[2]=0;
        break;
    case 6:
        motors[0]=speed_manual_rotate;
        motors[1]=speed_manual_rotate;
        motors[2]=-speed_manual_rotate;
        break;
    case 7:
        motors[0]=-speed_manual_forward;
        motors[1]=speed_manual_forward;
        motors[2]=speed_manual_turn;
        break;
    case 8:
        motors[0]=-speed_manual_forward;
        motors[1]=speed_manual_forward;
        motors[2]=0;
        break;
    case 9:
        motors[0]=-speed_manual_forward;
        motors[1]=speed_manual_forward;
        motors[2]=-speed_manual_turn;
        break;
    case 0:
        motors[0]=0;
        motors[1]=0;
        motors[2]=0;
        break;
}
//printf("command: %i /n", command);

}


void mouseHandler(int event, int x, int y, int flags, void *param)
{

    switch(event) {


        case CV_EVENT_LBUTTONDOWN:
        if (ball_ready){
            hsv_val = frame_HSV.at<Vec3b>(y,x);
            t_h=hsv_val.val[0];
            t_s=hsv_val.val[1];
            t_v=hsv_val.val[2];
            if (h_min>t_h) h_min=t_h;
            if (h_max<t_h) h_max=t_h;
            if (s_min>t_s) s_min=t_s;
            if (s_max<t_s) s_max=t_s;
            if (v_min>t_v) v_min=t_v;
            if (v_max<t_v) v_max=t_v;



            printf("H=%i, S=%i, V=%i\n",t_h,t_s,t_v);
            }
            break;

        case CV_EVENT_RBUTTONDOWN:

            break;


        case CV_EVENT_MBUTTONDOWN:

            break;

    }

}


// main function
int main(int argc, char *argv[])
{
    //QApplication a(argc, argv);

    printf("Hello everyone :-)\n");
    printf("Control 0 = control disabled\n");
    printf("Control 1 = manual control - numerical keyboard\n");
    printf("Control 2 = face detection\n");
    printf("Control 3 = ball tracking  - click on wanted colors, 'r' - reset colors, 's' - start tracking\n");
    printf("CTRL+P advanced settings\n");

    uart0_filestream = open("/dev/ttyUSB0", O_RDWR | O_NOCTTY | O_NDELAY);		//Open in non blocking read/write mode
    if (uart0_filestream == -1)
    {
        printf("Error - Unable to open UART\n");
        return 0;
    }
    //	B9600, B57600, B115200
    struct termios options;
    tcgetattr(uart0_filestream, &options);
    options.c_cflag = B57600 | CS8 | CLOCAL | CREAD;		//Set baud rate
    options.c_iflag = IGNPAR;
    options.c_oflag = 0;
    options.c_lflag = 0;
    tcflush(uart0_filestream, TCIFLUSH);
    tcsetattr(uart0_filestream, TCSANOW, &options);

    //start communication with robot - do not forget push button on SoccerBoard!!!
    pthread_create(&thread_update,NULL,&t_update,NULL);

    //error - you have not haarcascade in proper folder
    if (!face_cascade.load(CASCADE_FACE_NAME)){
        printf("coldnt load face");
        exit(-1);
    }

    //set camera properties
    Camera.set( CV_CAP_PROP_FORMAT, CV_8UC3 );
    Camera.set( CV_CAP_PROP_FRAME_WIDTH, 640 );
    Camera.set( CV_CAP_PROP_FRAME_HEIGHT, 480 );
    //Camera.set(CV_CAP_PROP_FPS, 15);

if (!Camera.open()) {printf("Error opening the camera - not connected or not updated firmware");return -1;}

    //create window
    namedWindow("main", WINDOW_AUTOSIZE);

    createTrackbar("Control","main",&control,3,on_trackbar_control);

    createButton("but",on_change_checkbox,NULL,CV_CHECKBOX,0);// hidden button for properties

    moveWindow("main",0,0); //set main window to the left upper corner
    cvSetMouseCallback( "main", mouseHandler, NULL );
    bool stop = false;

    pthread_create(&thread_grab,NULL,&t_grab,NULL);
    pthread_join(thread_grab,(void **)&ret);


    while(!stop) {
        pthread_join(thread_grab,(void **)&ret);
        pthread_create(&thread_grab,NULL,&t_grab,NULL);

        //printf("analog: %i %i %i %i %i %i %i %i\n", analog[0],analog[1],analog[2],analog[3],analog[4],analog[5],analog[6],analog[7]);

        imshow("main", frame);
        if ((control==2)&(face_ready)) imshow("face", frame_face);
        //if ((control==3)&(ball_ready)) imshow("ball", frame_ball);
        if ((control==3)&(ball_ready)) cvShowImage("ball", ball);


        key = cvWaitKey(5);

        if (key==27) {stop = true;}

        if (control==3){
        if (key=='r') {h_min=255; s_min=255; v_min=255; h_max=0; s_max=0; v_max=0;}
        else if (key=='s') ball_track_start=~ball_track_start;
        }

        if (control==1){
        if (key==-1) continue; //no key pressed
        else if (key=='0') manual_control(0);
        else if (key=='1') manual_control(1);
        else if (key=='2') manual_control(2);
        else if (key=='3') manual_control(3);
        else if (key=='4') manual_control(4);
        else if (key=='5') manual_control(5);
        else if (key=='6') manual_control(6);
        else if (key=='7') manual_control(7);
        else if (key=='8') manual_control(8);
        else if (key=='9') manual_control(9);

        else printf("Key: %i", key); //what did you push???
        }

    }

    return (0);
    //return a.exec();

}
