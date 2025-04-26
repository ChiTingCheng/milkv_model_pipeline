#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <vector>

#define PORT 5005
#define MAX_CHUNK_SIZE 4096
#define HEADER_SIZE 12
#define BUFFER_SIZE (MAX_CHUNK_SIZE + HEADER_SIZE)

int main() {
    int sockfd;
    uint8_t buffer[BUFFER_SIZE];
    struct sockaddr_in servaddr{}, cliaddr{};

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        std::cerr << "Socket creation failed\n";
        return 1;
    }

    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(PORT);

    if (bind(sockfd, (const struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
        std::cerr << "Bind failed\n";
        close(sockfd);
        return 1;
    }

    std::cout << "Receiver running on port " << PORT << "...\n";

    // Only one tensor in memory
    int current_tensor_id = -1;
    int total_chunks = 0;
    int received_chunks = 0;
    std::vector<std::vector<uint8_t>> chunks;

    while (true) {
        socklen_t len = sizeof(cliaddr);
        ssize_t n = recvfrom(sockfd, buffer, BUFFER_SIZE, 0,
                             (struct sockaddr*)&cliaddr, &len);
        if (n < HEADER_SIZE) continue;

        int tensor_id, chunk_id, total;
        std::memcpy(&tensor_id, buffer, 4);
        std::memcpy(&chunk_id, buffer + 4, 4);
        std::memcpy(&total, buffer + 8, 4);

        tensor_id = ntohl(tensor_id);
        chunk_id = ntohl(chunk_id);
        total = ntohl(total);

        // New tensor begins
        if (tensor_id != current_tensor_id) {
            current_tensor_id = tensor_id;
            total_chunks = total;
            received_chunks = 0;
            chunks.clear();
            chunks.resize(total_chunks);
            std::cout << "\nReceiving tensor ID " << tensor_id
                      << " with " << total_chunks << " chunks...\n";
        }

        // Store chunk
        if (chunk_id < total_chunks && chunks[chunk_id].empty()) {
            chunks[chunk_id].assign(buffer + HEADER_SIZE, buffer + n);
            received_chunks++;
        }

        // All chunks received
        if (received_chunks == total_chunks) {
            std::cout << "Tensor " << current_tensor_id << " complete.\n";

            // Reassemble tensor
            std::vector<uint8_t> full_tensor;
            for (const auto& chunk : chunks)
                full_tensor.insert(full_tensor.end(), chunk.begin(), chunk.end());

            std::cout << "Total size: " << full_tensor.size() << " bytes\n";

            // Do post-processing here (e.g., ONNX Runtime inference)

            // Free memory immediately
            chunks.clear();
            received_chunks = 0;
            current_tensor_id = -1;

            std::cout << "Buffer cleared.\n";
        }
    }

    close(sockfd);
    return 0;
}

