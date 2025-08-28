#include <WiFi.h>
#include <HTTPClient.h>

// Replace with your network credentials
const char* ssid = "your_SSID";   
const char* password = "your_PASSWORD";

// Backend URL
const char* backend_url = "http://your-backend-url.com/endpoint";

void setup() {
  // Start the serial communication
  Serial.begin(115200);

  // Connecting to Wi-Fi
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  // Wait until the connection is established
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting...");
  }

  // Print the IP address after successful connection
  Serial.println("Connected to WiFi!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Call the backend API after Wi-Fi is connected
  connectToBackend();
}

void loop() {
  // Optional: You can add code here to make periodic requests or handle other tasks
  // You can also handle reconnections to Wi-Fi here if needed
  delay(10000); // Simple delay to avoid flooding the serial monitor
}

void connectToBackend() {
  HTTPClient http;

  // Specify the backend server URL
  http.begin(backend_url);

  // Send an HTTP GET request to the backend
  int httpCode = http.GET();

  // Check the status of the response
  if (httpCode > 0) {
    // If successful, print the response
    String payload = http.getString();
    Serial.println("Backend Response:");
    Serial.println(payload);
  } else {
    // If failed, print error code
    Serial.println("Error on HTTP request");
  }

  // Close the connection
  http.end();
}
