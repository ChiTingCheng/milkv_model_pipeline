#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <stdio.h>
#include <numeric>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <cviruntime.h>
#include <algorithm>

#define PORT 5005
#define HEADER_SIZE 8  // 4 bytes tensor_id + 4 bytes size
#define CHANNEL_SIZE 256

bool recv_all(int sock, uint8_t* buffer, size_t length) {
    size_t total_received = 0;
    while (total_received < length) {
        ssize_t received = recv(sock, buffer + total_received, length - total_received, 0);
        if (received <= 0) return false;
        total_received += received;
    }
    return true;
}

int main(int argc, char **argv) {
    // load model file
    const char *model_file = argv[1];
    CVI_MODEL_HANDLE model = nullptr;
    int ret = CVI_NN_RegisterModel(model_file, &model);
    if (CVI_RC_SUCCESS != ret) {
        printf("CVI_NN_RegisterModel failed, err %d\n", ret);
        exit(1);
    }
    printf("CVI_NN_RegisterModel succeeded\n");

    // get input output tensors
    CVI_TENSOR *input_tensors;
    CVI_TENSOR *output_tensors;
    int32_t input_num;
    int32_t output_num;
    CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num, &output_tensors,&output_num);
    CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);
    assert(input);
    CVI_TENSOR *output = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, output_tensors, output_num);
    assert(output);

    // nchw
    CVI_SHAPE shape = CVI_NN_TensorShape(input);
    int32_t height = shape.dim[2];
    int32_t width = shape.dim[3];

    int server_fd, client_fd;
    struct sockaddr_in address{};
    socklen_t addrlen = sizeof(address);

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Socket creation failed\n";
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);
    float *input_ptr = (float *)CVI_NN_TensorPtr(input);
    int input_size = 4 * CHANNEL_SIZE * height * width;
    
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Bind failed\n";
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 1) < 0) {
        std::cerr << "Listen failed\n";
        close(server_fd);
        return 1;
    }

    std::cout << "Listening on port " << PORT << "...\n";

    client_fd = accept(server_fd, (struct sockaddr*)&address, &addrlen);
    if (client_fd < 0) {
        std::cerr << "Accept failed\n";
        close(server_fd);
        return 1;
    }

    std::cout << "Connected to client\n";

    std::vector<std::string> labels;
    std::ifstream file(argv[2]);
    if (!file) {
        printf("Didn't find synset_words file\n");
        exit(1);
    } else {
        std::string line;
        while (std::getline(file, line)) {
        labels.push_back(std::string(line));
        }
    }

    while (true) {
        uint8_t header[HEADER_SIZE];
        if (!recv_all(client_fd, header, HEADER_SIZE)) {
            std::cout << "Connection closed or failed\n";
            break;
        }

        uint32_t tensor_id, tensor_size;
        std::memcpy(&tensor_id, header, 4);
        std::memcpy(&tensor_size, header + 4, 4);
        tensor_id = ntohl(tensor_id);
        tensor_size = ntohl(tensor_size);

        std::cout << "\nReceiving tensor ID " << tensor_id << ", size: " << tensor_size << " bytes\n";

        std::vector<uint8_t> tensor_buffer(tensor_size);
        if (!recv_all(client_fd, tensor_buffer.data(), tensor_size)) {
            std::cerr << "Failed to receive full tensor\n";
            break;
        }
        if (tensor_size != input_size) {
            std::cerr << "Mismatch between received tensor size and expected input size\n";
            break;
        }
        memcpy(input_ptr, tensor_buffer.data(), input_size);
        std::cout << "Tensor " << tensor_id << " received.\n";

        // run inference
        CVI_NN_Forward(model, input_tensors, input_num, output_tensors, output_num);
        printf("CVI_NN_Forward succeeded\n");

        float *prob = (float *)CVI_NN_TensorPtr(output);
        int32_t count = CVI_NN_TensorCount(output);
        // find top-k prob and cls
        std::vector<size_t> idx(count);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&prob](size_t idx_0, size_t idx_1) {return prob[idx_0] > prob[idx_1];});
        // show results.
        printf("------\n");
        int top_k_idx = idx[0];
        printf("  %f, idx %d", prob[top_k_idx], top_k_idx);
        if (!labels.empty())
            printf(", %s", labels[top_k_idx].c_str());
        printf("\n");
        printf("------\n");

        // Free immediately
        tensor_buffer.clear();
    }
    CVI_NN_CleanupModel(model);
    printf("CVI_NN_CleanupModel succeeded\n");
    close(client_fd);
    close(server_fd);
    std::cout << "Server shutdown\n";
    return 0;
}