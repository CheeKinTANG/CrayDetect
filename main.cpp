#include "opencv2/opencv.hpp"
#include <iostream>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vitis/ai/classification.hpp>

#include <vitis/ai/demo.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>
#include <vitis/ai/yolov3.hpp>
#include "process_result.hpp"

using namespace cv;

static cv::Scalar getColor(int label) {
  int c[3];
  for (int i = 1, j = 0; i <= 9; i *= 3, j++) {
    c[j] = ((label / i) % 3) * 127;
  }
  return cv::Scalar(c[2], c[1], c[0]);
}


int main(int argc, char** argv)
{
    VideoCapture cap(0);
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    auto network = vitis::ai::YOLOv3::create("yolov4_leaky_spp_m");
    auto network1 = vitis::ai::Classification::create("resnet50");

    while(true)
    {
        Mat frame;
        cap >> frame;
        
        resize(frame, frame, Size(640,480));
        auto results = network->run(frame);
        
          //Drawing boxes around results
        for (const auto bbox : results.bboxes) {
            int label = bbox.label;
            float xmin = bbox.x * frame.cols ;
            float ymin = bbox.y * frame.rows ;
            float xmax = xmin + bbox.width * frame.cols;
            float ymax = ymin + bbox.height * frame.rows;
            float confidence = bbox.score;
            if (xmax > frame.cols) xmax = frame.cols;
            if (ymax > frame.rows) ymax = frame.rows;

            // LOG_IF(INFO, 1) << "RESULT: label:" << label << "\t xmin:" << xmin << "\t ymin" << ymin
            //                     << "\t xmax:" << xmax << "\t ymax:" << ymax << "\t confi:" << confidence
            //                     << "\n";
            cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                        getColor(label), 1, 1, 0);



            Mat cropped_image = frame(cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)));


            //run classifications on each bounding box
            auto results1 = network1->run(cropped_image);

            process_result(cropped_image, results1, 2);
            imshow("Cropped Image", cropped_image);


        }

        
        // resize(frame, frame, Size(640,480));
        // imshow("Crayfish Detection", frame);
        if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC 


	    // for (const auto &r : face_results.rects) {
        //     int x1 = r.x * frame.cols;
        //     int y1 = r.y * frame.rows;
        //     int x2 = x1 + (r.width * frame.cols);
        //     int y2 = y1 + (r.height * frame.rows);
        //   Point pt1(x1, y1);
        //   // and its bottom right corner.
        //   Point pt2(x2, y2);
        //   cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0));
        // }

        //   resize(frame, frame, Size(640,480));
        //   imshow("this is you, smile! :)", frame);
        //   if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC 
         
    // the camera will be closed automatically upon exit
    // cap.close();
}
}