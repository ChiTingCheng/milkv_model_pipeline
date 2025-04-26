#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <vector>

#define PORT 5005
#define HEADER_SIZE 8  // 4 bytes tensor_id + 4 bytes size

bool recv_all(int sock, uint8_t* buffer, size_t length) {
    size_t total_received = 0;
    while (total_received < length) {
        ssize_t received = recv(sock, buffer + total_received, length - total_received, 0);
        if (received <= 0) return false;
        total_received += received;
    }
    return true;
}

int main() {
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

    std::cout << "🚪 Listening on port " << PORT << "...\n";

    client_fd = accept(server_fd, (struct sockaddr*)&address, &addrlen);
    if (client_fd < 0) {
        std::cerr << "Accept failed\n";
        close(server_fd);
        return 1;
    }

    std::cout << "🔗 Connected to client\n";

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

        // 🚧 Optional: Process tensor here
        std::cout << "Tensor " << tensor_id << " received.\n";

        // Free immediately
        tensor_buffer.clear();
    }

    close(client_fd);
    close(server_fd);
    std::cout << "Server shutdown\n";
    return 0;
}

