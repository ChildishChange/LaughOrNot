#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <FaceTracker/Tracker.h>
#include <iostream>
#include <ExpressionClassifier.h>
#include <math.h>
#include <opencv2/opencv.hpp>  
#include <vector>
using namespace cv;
using namespace std;
#define GRAY CV_RGB(128,128,128)
#define WHITE CV_RGB(255,255,255)
#define BLACK CV_RGB(0,0,0)
#define RED CV_RGB(255,0,0)
/**/
//const cv::Mat emoji = imread("smi.jpeg", CV_LOAD_IMAGE_COLOR);
//const cv::Mat Emo_StandAlone = imread("smiling.jpeg",CV_LOAD_IMAGE_COLOR);
const cv::Mat Emo_L_L = imread("1.jpg", CV_LOAD_IMAGE_COLOR);
const cv::Mat Emo_L_LP = imread("1.png", -1);
const cv::Mat Emo_L_H = imread("3.jpg", CV_LOAD_IMAGE_COLOR); 
const cv::Mat Emo_L_HP = imread("3.png", -1); 
const cv::Mat Emo_T_L = imread("5.jpg", CV_LOAD_IMAGE_COLOR);
const cv::Mat Emo_T_LP = imread("5.png", -1);
const cv::Mat Emo_T_H = imread("6.jpg", CV_LOAD_IMAGE_COLOR);
const cv::Mat Emo_T_HP = imread("6.png", -1);
const cv::Mat Emo_N_L = imread("2.jpg", CV_LOAD_IMAGE_COLOR);
const cv::Mat Emo_N_LP = imread("2.png", -1);
const cv::Mat Emo_N_H = imread("4.jpg", CV_LOAD_IMAGE_COLOR);
const cv::Mat Emo_N_HP = imread("4.png", -1);

/**/
void Draw(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi)
{
    int i,n = shape.rows/2; cv::Point p1,p2;
    //draw triangulation
    /*for(i = 0; i < tri.rows; i++)
    {
        if(visi.at<int>(tri.at<int>(i,0),0) == 0 || visi.at<int>(tri.at<int>(i,1),0) == 0 ||visi.at<int>(tri.at<int>(i,2),0) == 0)
            continue;
        p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
            shape.at<double>(tri.at<int>(i,0)+n,0));
        p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
            shape.at<double>(tri.at<int>(i,1)+n,0));
        cv::line(image,p1,p2,BLACK);
        p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
            shape.at<double>(tri.at<int>(i,0)+n,0));
        p2 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
            shape.at<double>(tri.at<int>(i,2)+n,0));
        cv::line(image,p1,p2,BLACK);
        p1 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
            shape.at<double>(tri.at<int>(i,2)+n,0));
        p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
            shape.at<double>(tri.at<int>(i,1)+n,0));
        cv::line(image,p1,p2,BLACK);
    }*/
    //draw connections
   /* for(i = 0; i < con.cols; i++){
        if(visi.at<int>(con.at<int>(0,i),0) == 0 ||
            visi.at<int>(con.at<int>(1,i),0) == 0)continue;
        p1 = cv::Point(shape.at<double>(con.at<int>(0,i),0),
            shape.at<double>(con.at<int>(0,i)+n,0));
        p2 = cv::Point(shape.at<double>(con.at<int>(1,i),0),
            shape.at<double>(con.at<int>(1,i)+n,0));
        cv::line(image,p1,p2,BLACK,1);
    }*/
    //draw points
   for(i = 0; i < n; i++){    
        if(visi.at<int>(i,0) == 0)continue;
        p1 = cv::Point(shape.at<double>(i,0),shape.at<double>(i+n,0));
        cv::circle(image,p1,2,RED);
    }
    return;
}
bool getEmoji(ExpressionClassifier classifier,cv::Mat &EMO,cv::Mat &EMOP)
{
    //when some emo more than 0.5,use picture
    bool useEmo = false;
    int n = classifier.size();
    for(int i = 0;i<n;i++){
        if(classifier.getProbability(i)>0.4)
        {
            useEmo = true;
            break;
        }
    }
    //choose the highest one
    if(useEmo)
    {
        int max_index = 0;
        double max_pro = classifier.getProbability(0);
        int i = 1;
        for(;i<n;i++)
        { 
            if(classifier.getProbability(i)>max_pro)
            {
                max_pro = classifier.getProbability(i);
                max_index = i;
            }
        }
        if(strcmp(classifier.getDescription(max_index).c_str(),"surprise")==0)
        {   
            if(max_pro<=0.95)
            {    EMO = Emo_T_L;
                EMOP = Emo_T_LP;
            }
            else if(max_pro>0.95)
            {               EMO = Emo_T_H;
                    EMOP = Emo_T_HP;
            }
        }
        else if(strcmp(classifier.getDescription(max_index).c_str(),"neutral")==0)
        {   
             if(max_pro<=0.95)
         {         EMO = Emo_N_L;
                EMOP = Emo_N_LP;
         }
            else if(max_pro>0.95)
           {       EMO = Emo_N_H;
                EMOP = Emo_N_HP;
           }
        }
        else if(strcmp(classifier.getDescription(max_index).c_str(),"smile")==0)
        {   
             if(max_pro<=0.95)
            {      EMO = Emo_L_L;
                EMOP = Emo_L_LP;
            }
            else if(max_pro>0.95)
          {        EMO = Emo_L_H;
                EMOP = Emo_L_HP;
          }
        }
    printf("%d:%s:%f,%d,%d,%d,%d\n",max_index, classifier.getDescription(max_index).c_str(),max_pro,EMO.cols,EMO.rows,EMOP.cols,EMOP.rows);

   }
 //namedWindow("12",WINDOW_AUTOSIZE); 
   //imshow("Smile",EMO); 
   return useEmo;
}

void ROI_AddImage(cv::Mat &src,cv::Mat &logo,cv::Mat &shape)
{
    //现在这个函数显示出来的emoji是不能变化角度的。。所以。。如果要变化角度，我们要做一些改变
    cv::Mat imageROI = src(cv::Rect(shape.at<double>(30,0)-logo.cols/2,shape.at<double>(30+66,0)-logo.rows/2,logo.cols,logo.rows));
    cv::Mat mask ;
    cv::cvtColor(logo,mask,CV_BGR2GRAY);
  //  cv::Mat mask1 = 255-mask;
    logo.copyTo(imageROI,mask);

}

cv::Mat RotateImg(cv::Mat img,cv::Mat shape)
{
    double tang_ = (shape.at<double>(27,0)-shape.at<double>(30,0))/(shape.at<double>(30+66,0)-shape.at<double>(27+66,0));
    double angle = atan(tang_);
   // cout<<angle<<endl;
    int degree = (int)(angle*180/M_PI);
    degree = -degree;
    //cout<<degree<<endl;
    double a = sin(angle), b = cos(angle);
    //resize
    double times = (((shape.at<double>(16,0)-shape.at<double>(0,0))/abs(b))/img.cols);
    times*=2;
    cv::resize(img,img,cv::Size(img.cols*times,img.rows*times));
    //rotate
    int width = img.cols;
    int height = img.rows;
    int width_rotate = int(height * fabs(a) + width * fabs(b));
    int height_rotate = int(width * fabs(a) + height * fabs(b));
    
    float map[6];
    Mat map_matrix = Mat(2, 3, CV_32F, map);
    
    CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
    CvMat map_matrix2 = map_matrix;
    cv2DRotationMatrix(center, degree, 1.0, &map_matrix2);//计算二维旋转的仿射变换矩阵
    map[2] += (width_rotate - width) / 2;
    map[5] += (height_rotate - height) / 2;
    Mat img_rotate;
    
    warpAffine(img, img_rotate, map_matrix, Size(width_rotate, height_rotate), 1, 0, 0);
    return img_rotate;
    
}
void mergeImg(cv::Mat & dst,cv::Mat &src1,cv::Mat &src2)  
{  
        int rows = src1.rows;

        double times = (double)src1.rows/src2.rows;
        
        cv::resize(src2,src2,cv::Size(src2.cols*times,src2.rows*times));

        int cols = src1.cols+5+src2.cols;  

        CV_Assert(src1.type () == src2.type ());  
        dst.create (rows,cols,src1.type ());  
        src1.copyTo (dst(Rect(0,0,src1.cols,src1.rows)));  
        src2.copyTo (dst(Rect(src1. cols+5,0,src2.cols,src2.rows)));  
}  

int cvAdd4cMat_q(cv::Mat &d, cv::Mat &scr, double scale,cv::Mat &shape)    
{  
    cv::Mat dst = d(cv::Rect(shape.at<double>(30,0)-scr.cols/2,shape.at<double>(30+66,0)-scr.rows/2,scr.cols,scr.rows));

    if (dst.channels() != 3 || scr.channels() != 4)    
    {    
        return true;    
    }    
    if (scale < 0.01)    
        return false;    
    std::vector<cv::Mat>scr_channels;    
    std::vector<cv::Mat>dstt_channels;    
    split(scr, scr_channels);    
    split(dst, dstt_channels);    
    CV_Assert(scr_channels.size() == 4 && dstt_channels.size() == 3);    
  
    if (scale < 1)    
    {    
        scr_channels[3] *= scale;    
        scale = 1;    
    }    
    for (int i = 0; i < 3; i++)    
    {    
        dstt_channels[i] = dstt_channels[i].mul(255.0 / scale - scr_channels[3], scale / 255.0);    
        dstt_channels[i] += scr_channels[i].mul(scr_channels[3], scale / 255.0);    
    }    
    merge(dstt_channels, dst);    
    return true;    
}    

int main(){
    //settings
    bool fcheck = true; double scale = 1; int fpd = -1; bool show = true;
     //set other tracking parameters
    std::vector<int> wSize1(1); wSize1[0] = 7;
    std::vector<int> wSize2(3); wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;
    int nIter = 5; double clamp=3,fTol=0.01; 

    FACETRACKER::Tracker model("face2.tracker");
    cv::Mat tri=FACETRACKER::IO::LoadTri("face.tri");
    cv::Mat con=FACETRACKER::IO::LoadCon("face.con");

    //initialize camera and display window
    cv::Mat frame,gray,im; double fps=0; char sss[256]; std::string text; 
    VideoCapture camera(0);
    
    camera.set(CV_CAP_PROP_FRAME_WIDTH, 640.0);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT, 480.0);

    int width = camera.get(CV_CAP_PROP_FRAME_WIDTH), height = camera.get(CV_CAP_PROP_FRAME_HEIGHT);
    printf("width: %d, height: %d\n", width, height);

    int64 t1,t0 = cvGetTickCount(); int fnum=0;
    if(show)
        cvNamedWindow("Smile Detect",1);
/*imshow("asd",Emo_L_L);waitKey();
imshow("asd",Emo_L_H);waitKey();
imshow("asd",Emo_T_L);waitKey();
imshow("asd",Emo_T_H);waitKey();
imshow("asd",Emo_N_L);waitKey();
imshow("asd",Emo_N_H);waitKey();*/
    //load expressions
    ExpressionClassifier classifier;
    classifier.load();

    bool failed = true;
    cout<<endl;

    int facenum = 0;

    while(1){
        camera.read(frame);

        //grab image, resize and flip
        if(scale == 1)im = frame;
        else cv::resize(frame,im,cv::Size(scale*frame.cols,scale*frame.rows));
        cv::flip(im,im,1);
        cv::cvtColor(im,gray,CV_BGR2GRAY);

        //track this image
        std::vector<int> wSize; if(failed)wSize = wSize2; else wSize = wSize1; 
        if(model.Track(gray,wSize,fpd,nIter,clamp,fTol,fcheck) == 0){
            int idx = model._clm.GetViewIdx();
            failed = false;
            Draw(im,model._shape,con,tri,model._clm._visi[idx]);
        }else{
            if(show){
                cv::Mat R(im,cvRect(0,0,160,140));
                R = cv::Scalar(0,0,255);
            }
            model.FrameReset();
            failed = true;
        }

        //draw framerate on display image 
        if(fnum >= 9){      
            t1 = cvGetTickCount();
            fps = 10.0/((double(t1-t0)/cvGetTickFrequency())/1e+6); 
            t0 = t1; fnum = 0;
        }else fnum += 1;
        
        if(show){
            sprintf(sss,"%d frames/sec",(int)round(fps)); text = sss;
            cv::putText(im,text,cv::Point(10,20),
            CV_FONT_HERSHEY_SIMPLEX,0.5,RED);
        }
        
        //classify the points
        const Mat& mean = model._clm._pdm._M;
        const Mat& variation = model._clm._pdm._V;
        const Mat& weights = model._clm._plocal;
        Mat objectPoints = mean + variation * weights;

        classifier.classify(objectPoints);

        int n = classifier.size();
        for(int i = 0; i < n; i++){
            sprintf(sss,"%s: %f", classifier.getDescription(i).c_str(), classifier.getProbability(i));
            
            cv::putText(im,sss,cv::Point(10, 40 + i*20),
            CV_FONT_HERSHEY_SIMPLEX,0.5,RED);
        }

        //printf("Detecting you are smiling!\n");


        // printf("smile: %.4f\n", classifier.getProbability(1) );

        if(show){

        cv::Mat emoticon;
        cv::Mat emoticon_p;
 //  getEmoji(classifier,emoticon,emoticon_p);
      //  cv::Mat rot_emoji = RotateImg(emoticon,model._shape);
       // ROI_AddImage(im,emoticon,model._shape);
           
    if(getEmoji(classifier,emoticon,emoticon_p))
        {   

      //      cv::Mat rot_emoji = RotateImg(emoticon_p,model._shape);
       //   cvAdd4cMat_q(im,rot_emoji,1.0,model._shape);
                    cv::Mat outim;
           mergeImg(outim,im,emoticon);
           imshow("Smile Detect",outim); 
         }
           else
            {
        imshow("Smile Detect",im); 
  //           imshow("ss",emoticon);
        }


            //show image and check for user input
           
            int c = waitKey(10)&0xFF;
            if(c == 27){
                cout<<"Esc"<<endl;
                break;
            }
            else if( char(c) == ' '){
                model.FrameReset();
                cout<<"Redetect"<<endl;
            }else if( char(c) == 'c'){
                //save face model
                facenum ++;
                char facefilename[50];
                sprintf(facefilename, "face/face%d.yml", facenum);
                FileStorage fs(facefilename, FileStorage::WRITE);
                fs << "shape" << model._shape;
                fs << "tri" << tri;
                fs.release();

                FileStorage fs2("face/objectPoints.yml", FileStorage::WRITE);
                fs2 <<   "description" << "emotion" <<
                "samples" << "[";
                fs2 << objectPoints;
                fs2 << "]";
                fs2.release();
            }else if( char(c) == 'v'){
                Mat shape = model._shape;
                double scale = 5;
                Mat test(480*scale, 640*scale, CV_8UC3, Scalar(255,255,255));
                int i, n = 66; cv::Point p1,p2;
                //draw triangulation
                for(i = 0; i < tri.rows; i++){
                    p1 = cv::Point(scale*shape.at<double>(tri.at<int>(i,0),0),
                        scale*shape.at<double>(tri.at<int>(i,0)+n,0));
                    p2 = cv::Point(scale*shape.at<double>(tri.at<int>(i,1),0),
                        scale*shape.at<double>(tri.at<int>(i,1)+n,0));
                    cv::line(test,p1,p2,BLACK);
                    p1 = cv::Point(scale*shape.at<double>(tri.at<int>(i,0),0),
                        scale*shape.at<double>(tri.at<int>(i,0)+n,0));
                    p2 = cv::Point(scale*shape.at<double>(tri.at<int>(i,2),0),
                        scale*shape.at<double>(tri.at<int>(i,2)+n,0));
                    cv::line(test,p1,p2,BLACK);
                    p1 = cv::Point(scale*shape.at<double>(tri.at<int>(i,2),0),
                        scale*shape.at<double>(tri.at<int>(i,2)+n,0));
                    p2 = cv::Point(scale*shape.at<double>(tri.at<int>(i,1),0),
                        scale*shape.at<double>(tri.at<int>(i,1)+n,0));
                    cv::line(test,p1,p2,BLACK);
                }
                //draw points
                for(i = 0; i < n; i++){    
                    p1 = cv::Point(scale*shape.at<double>(i,0),scale*shape.at<double>(i+n,0));
                    char buf[50];
                    sprintf(buf, "%d", i);
                    cv::circle(test,p1,scale*2, RED);
                    p1.x += 10;
                    cv::putText(test, buf, p1, CV_FONT_HERSHEY_SIMPLEX, 0.7, RED);
                }
                imwrite("face/test.png", test);
            }
        }

    }
    return 0;
}
