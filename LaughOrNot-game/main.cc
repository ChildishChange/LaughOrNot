/*
说一下流程
我们现在有几张识别效果还可以的图片
首先我们需要获取这些图片的.yml文件
    写个程序挨个生成
然后我们要把那个图片的特征点标记出来形成另一张图
然后我们要用摄像头获取到的帧来用这张图生成的.yml文件与neutral进行比对
直到大于70%就跳出循环并结束。。
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <FaceTracker/Tracker.h>
#include <iostream>
#include <ExpressionClassifier.h>
#include <math.h>
#include <cstring>
using namespace cv;
using namespace std;
#define GRAY CV_RGB(128,128,128)
#define WHITE CV_RGB(255,255,255)
#define BLACK CV_RGB(0,0,0)
#define RED CV_RGB(255,0,0)
/**/
const cv::Mat emoji = imread("smi.jpeg", CV_LOAD_IMAGE_COLOR);
const cv::Mat Emo_StandAlone = imread("smiling.jpeg",CV_LOAD_IMAGE_COLOR);
const cv::Mat Emo_LOL = imread("smi.jpeg",CV_LOAD_IMAGE_COLOR);
const cv::Mat Emo_Ter = imread("terrified.jpeg",CV_LOAD_IMAGE_COLOR);
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
    }
    //draw connections
    for(i = 0; i < con.cols; i++){
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
bool getEmoji(ExpressionClassifier classifier,cv::Mat &EMO)
{
    //when some emo more than 0.5,use picture
    bool useEmo = false;
    int n = classifier.size();
    for(int i = 1;i<n;i++){
        if(classifier.getProbability(i)>0.5)
        {
            useEmo = true;
            break;
        }
    }
    //choose the highest one
    if(useEmo)
    {
        int max_index = 1;
        double max_pro = classifier.getProbability(1);
        int i = 1;
        for(i = 1;i<n;i++)
        { 
            if(classifier.getProbability(i)>max_pro)
            {
                max_pro = classifier.getProbability(i);
                max_index = i;
            }
        }
        if(strcmp(classifier.getDescription(max_index).c_str(),"surprise")==0)
        {   
            EMO = Emo_Ter;
        }
        else
        {   
            EMO = Emo_StandAlone;
        }
   }
   return useEmo;
}
void ROI_AddImage(cv::Mat &src,cv::Mat &logo,cv::Mat &shape)
{
    //现在这个函数显示出来的emoji是不能变化角度的。。所以。。如果要变化角度，我们要做一些改变
    cv::Mat imageROI = src(cv::Rect(shape.at<double>(30,0)-logo.cols/2,shape.at<double>(30+66,0)-logo.rows/2,logo.cols,logo.rows));
    cv::Mat mask ;
    cv::cvtColor(logo,mask,CV_BGR2GRAY);
    logo.copyTo(imageROI,mask);

    // NO.30
}
cv::Mat RotateImg(cv::Mat img,cv::Mat shape)
{
    //before rotating the image , resize it
    //computer the angle
    double tang_ = (shape.at<double>(27,0)-shape.at<double>(30,0))/(shape.at<double>(30+66,0)-shape.at<double>(27+66,0));
    double angle = atan(tang_);
   // cout<<angle<<endl;
    int degree = (int)(angle*180/M_PI);
    degree = -degree;
    //cout<<degree<<endl;
    double a = sin(angle), b = cos(angle);
    //resize
    double times = (((shape.at<double>(16,0)-shape.at<double>(0,0))/abs(b))/img.cols);
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

int main(){
    //settings
    bool failed;
    bool fcheck = true; double scale = 1; int fpd = -1; bool show = true;
     //set other tracking parameters
    std::vector<int> wSize1(1); wSize1[0] = 7;
    std::vector<int> wSize2(3); wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;
    int nIter = 5; double clamp=3,fTol=0.01; 

   
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
/*============================================*/
    ExpressionClassifier classifier;
    classifier.load("neutral.yml");      
    


    int template_index = 1;


while(template_index<=8)
{
    FACETRACKER::Tracker model("face2.tracker");
    cv::Mat tri=FACETRACKER::IO::LoadTri("face.tri");
    cv::Mat con=FACETRACKER::IO::LoadCon("face.con");

    failed = true;
    ExpressionClassifier classifier;
    classifier.load("neutral.yml");      
    string template_name;
    stringstream stream;
    stream<<template_index;
    
    template_name = "picsrc/"+stream.str()+".jpeg" ;
    cv::Mat temPlate ;
    temPlate = imread(template_name);
    cv::Mat template_im;
    //grab image, resize and flip
    if(scale == 1) 
    {
        template_im = temPlate.clone();
    }
    else
    {
     cv::resize(temPlate,template_im,cv::Size(scale*temPlate.cols,scale*temPlate.rows));
    }
    cv::flip(template_im,template_im,1);
    cv::cvtColor(template_im,gray,CV_BGR2GRAY);
    //track this image
    std::vector<int> wSize; if(failed)wSize = wSize2; else wSize = wSize1; 
    if(model.Track(gray,wSize,fpd,nIter,clamp,fTol,fcheck) == 0)
    {
        int idx = model._clm.GetViewIdx();
       // failed = false;
        Draw(template_im,model._shape,con,tri,model._clm._visi[idx]);
          //classify the points
        const Mat& mean = model._clm._pdm._M;
        const Mat& variation = model._clm._pdm._V;
        const Mat& weights = model._clm._plocal;
        Mat objectPoints = mean + variation * weights;

        classifier.classify(objectPoints);

        char template_temp[50];
        sprintf(template_temp, "ymlsrc/%d.yml", template_index);
        FileStorage fs2(template_temp, FileStorage::WRITE);
        fs2 <<   "description" << "emotion" <<
        "samples" << "[";
        fs2 << objectPoints;
        fs2 << "]";
        fs2.release();

    }

    /*=======================================================*/
    classifier.reset();
    classifier.load("ymlsrc/"+stream.str()+".yml");
    /*=======================================================*/
   
    //this is where i need to make some changes...
    //load expressions and this where i need to make some change
    
   
    failed = true;
    cout<<endl;
    int facenum = 0;

    while(1)
    {
        camera.read(frame);

        cv::Mat yourFace = frame.clone();
        //grab image, resize and flip
        if(scale == 1)im = frame;
        else cv::resize(frame,im,cv::Size(scale*frame.cols,scale*frame.rows));
        cv::flip(im,im,1);
        cv::cvtColor(im,gray,CV_BGR2GRAY);

        //track this image
        std::vector<int> wSize; if(failed)wSize = wSize2; else wSize = wSize1; 
        if(model.Track(gray,wSize,fpd,nIter,clamp,fTol,fcheck) == 0)
        {
            int idx = model._clm.GetViewIdx();
            failed = false;
            Draw(im,model._shape,con,tri,model._clm._visi[idx]);

        }
        else
        {
            if(show)
            {
                cv::Mat R(im,cvRect(0,0,160,140));
                R = cv::Scalar(0,0,255);
            }
            model.FrameReset();
            failed = true;
        }

        //draw framerate on display image 
        if(facenum >= 9)
        {      
            t1 = cvGetTickCount();
            fps = 10.0/((double(t1-t0)/cvGetTickFrequency())/1e+6); 
            t0 = t1; fnum = 0;
        }
        else fnum += 1;
        
        if(show)
        {
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
        for(int i = 0; i < n; i++)
        {
            sprintf(sss,"%s: %f", classifier.getDescription(i).c_str(), classifier.getProbability(i));
            cv::putText(im,sss,cv::Point(10, 40 + i*20),
            CV_FONT_HERSHEY_SIMPLEX,0.5,RED);
        }

        //printf("Detecting you are smiling!\n");
        /*  cv::Mat emoticon;
        if(getEmoji(classifier,emoticon))
        {   

            cv::Mat rot_emoji = RotateImg(emoticon,model._shape);
            ROI_AddImage(im,rot_emoji,model._shape);
        }*/
        //imshow("Smile Detect",im); 
         //  int c = waitKey(10)&0xFF;
        // printf("smile: %.4f\n", classifier.getProbability(1) );

        if(show)
        {
            //show image and check for user input
            cv::Mat outim;
            mergeImg(outim,im,template_im);
            imshow("Smile Detect",outim); 
            int c = waitKey(10)&0xFF;
            /*if(c == 27)
            {
                cout<<"Esc"<<endl;
                break;
            }
            else */if( char(c) == ' '){
                model.FrameReset();
                cout<<"Redetect"<<endl;
            }else if( char(c) == 's'){
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

                cv::imwrite( "yourFace/"+stream.str()+".jpeg",yourFace );


            }else if( char(c) == 'v'){
                Mat shape = model._shape;
                double scale = 5;
                Mat test(480*scale, 640*scale, CV_8UC3, Scalar(255,255,255));
                int i, n = 66; cv::Point p1,p2;
                //draw triangulation
                for(i = 0; i < tri.rows; i++)
                {
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
            else if(c==10)
            {
                break;
            }
        }
        // Point p1,p2;
        // Mat shape = model._shape;
        // p1 = cv::Point(shape.at<double>(23,0),
        //     shape.at<double>(23+66,0));
        // p2 = cv::Point(shape.at<double>(43,0),
        //     shape.at<double>(43+66,0));
        // double distance = sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y));
        // cout<<distance-34<<endl;
       
   }
   template_index++;
}
    return 0;
}
