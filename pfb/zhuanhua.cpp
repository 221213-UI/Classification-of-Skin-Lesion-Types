#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

std::vector<std::string> class_names = {"AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <图片路径>" << std::endl;
        return -1;
    }
    
    // 1. 读取并预处理图片
    cv::Mat img = cv::imread(argv[1]);
    if (img.empty()) {
        std::cerr << "无法读取图片: " << argv[1] << std::endl;
        return -1;
    }
    cv::resize(img, img, cv::Size(224, 224));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3, 1.0/255.0);
    
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    cv::subtract(img, mean, img);
    cv::divide(img, std, img);
    
    // 2. HWC -> CHW
    std::vector<float> input(1*3*224*224);
    for (int h = 0; h < 224; h++)
        for (int w = 0; w < 224; w++)
            for (int c = 0; c < 3; c++)
                input[c*224*224 + h*224 + w] = img.at<cv::Vec3f>(h,w)[c];
    
    // 3. ONNX Runtime 推理
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions opts;
    Ort::Session session(env, "/home/wangpp/xm/pfb/my_mobilenet.onnx", opts);
    
    std::vector<int64_t> shape = {1, 3, 224, 224};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memInfo, input.data(), input.size(), shape.data(), shape.size());
    
    const char* inputName = "input";
    const char* outputName = "output";
    std::vector<Ort::Value> output = session.Run(Ort::RunOptions{nullptr}, &inputName, &inputTensor, 1, &outputName, 1);
    
    // 4. Softmax 转换为概率
    float* logits = output[0].GetTensorMutableData<float>();
    int num_classes = 8;
    std::vector<float> probabilities(num_classes);
    float max_logit = *std::max_element(logits, logits + num_classes);
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum_exp += probabilities[i];
    }
    for (int i = 0; i < num_classes; i++) {
        probabilities[i] /= sum_exp;
    }
    
    // 5. 获取结果
    int pred = std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
    float confidence = probabilities[pred];
    
    std::cout << "预测类别: " << class_names[pred] << "，置信度: " << confidence << std::endl;
    
    return 0;
}
