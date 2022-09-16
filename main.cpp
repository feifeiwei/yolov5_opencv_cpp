#include <fstream>
#include <opencv2/opencv.hpp>



const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};


struct Config
{
    const std::string label_path = "../data/classes.txt";
    const std::string onnx_path = "../data/yolov5s.onnx";
    const float Score_threshold = 0.2;
    const float NMS_threshold = 0.4;
    const float Conf_threshold = 0.4;
    const bool use_cuda = false;
};

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};


class YOLOv5{
public:
    
    YOLOv5(Config config):
         label_path(config.label_path), onnx_path(config.onnx_path), Score_threshold(config.Score_threshold), NMS_threshold(config.NMS_threshold), Conf_threshold(config.Conf_threshold) {
             this->load_class_list();
             this->load_net(net, config.use_cuda);
         };

    void detect(cv::Mat &image);
    
    
private:
    const std::string label_path;
    const std::string onnx_path;
    const float Score_threshold;
    const float NMS_threshold;
    const float Conf_threshold;
    const float Input_width=640;
    const float Input_height=640;
    
    cv::dnn::Net net;
    std::vector<std::string> class_list;
    
    void load_net(cv::dnn::Net &net, bool is_cuda);
    void load_class_list();
    void draw_images(cv::Mat &,std::vector<Detection>&);
    
    ;
    cv::Mat get_input_image(const cv::Mat &source);
};

void  YOLOv5::load_class_list()
{
    std::ifstream ifs(label_path);
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    std::cout <<"loading class txt successful,  num classes: "<<class_list.size() << std::endl;
}

void YOLOv5::load_net(cv::dnn::Net &net, bool is_cuda)
{
    net = cv::dnn::readNet(onnx_path);
    if (is_cuda)
    {
        std::cout << "Running on CUDA\n";
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

cv::Mat YOLOv5::get_input_image(const cv::Mat &source){
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}


void YOLOv5::detect(cv::Mat &image)
{
    cv::Mat blob;
    auto input_image = get_input_image(image);
    cv::dnn::blobFromImage(input_image, blob, 1/255., cv::Size(Input_width, Input_height), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    
    float x_factor = input_image.cols / Input_width;
    float y_factor = input_image.rows / Input_height;
    
    float* data = (float*) outputs[0].data;
    
    const int dimensions = 85;
    const int rows = 25200;
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if (confidence >= Conf_threshold)
        {
            float * classes_scores = data + 5;
            cv::Mat scores(1, class_list.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > Score_threshold)
            {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));

            }
        }
        data += dimensions;
    }
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, Score_threshold, NMS_threshold, nms_result);
    
    std::vector<Detection> output;
    for (int i = 0; i < nms_result.size(); i++)
    {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
    
    this->draw_images(image, output);
}

void YOLOv5::draw_images(cv::Mat &image, std::vector<Detection>& output)
{
    unsigned long detections = output.size();

    for (int i = 0; i < detections; ++i)
    {

        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        const auto color = colors[classId % colors.size()];
        cv::rectangle(image, box, color, 3);

        cv::rectangle(image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        cv::putText(image, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}


int main(int argc, char **argv)
{

    
    Config config;
    YOLOv5 yolov5(config);
    
    cv::Mat frame;
    cv::VideoCapture capture("../demo.mp4");
    if (!capture.isOpened())
    {
        std::cerr << "Error opening video file\n";
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;
//
    while (true)
    {
        capture.read(frame);
        if (frame.empty())
        {
            std::cout << "End of stream\n";
            break;
        }

        std::vector<Detection> output;
        yolov5.detect(frame);

        frame_count++;
        total_frames++;

        if (frame_count >= 30)
        {

            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }
//
        if (fps > 0)
        {

            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            std::string fps_label_str = fps_label.str();

            cv::putText(frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("output", frame);

        if (cv::waitKey(1) != -1)
        {
            capture.release();
            std::cout << "finished by user\n";
            break;
        }
    }
    return 0;
}
