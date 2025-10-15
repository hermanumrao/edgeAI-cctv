#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <iostream>
#include <memory>

int main() {
    // Load model
    const char* model_path = "model.tflite";
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        std::cerr << "Failed to load model " << model_path << "\n";
        return -1;
    }

    // Build interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to build interpreter\n";
        return -1;
    }

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors\n";
        return -1;
    }

    // Set input data (example for float input)
    float* input = interpreter->typed_input_tensor<float>(0);
    input[0] = 1.0f; // your data here

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke tflite!\n";
        return -1;
    }

    // Get output
    float* output = interpreter->typed_output_tensor<float>(0);
    std::cout << "Model output: " << output[0] << "\n";

    return 0;
}
