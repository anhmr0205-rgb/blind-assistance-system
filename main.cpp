#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"
#include "camera_pins.h"
#include <ESP32Servo.h>
#include <ArduinoJson.h>

// Hàm trong file app_httpd.cpp của web camera riêng
void startCameraServer();

// ===================== WIFI =====================
const char* WIFI_SSID     = "Fantom";
const char* WIFI_PASSWORD = "00000000";

// ===================== CHÂN LINH KIỆN =====================
#define TRIG_PIN      41
#define ECHO_PIN      42
#define SERVO_PIN     14

#define ENABLE_BUZZER 1
#define BUZZER_PIN    45
// ===================== SERVO =====================
// Cau hinh moi: quet lien tuc qua lai 120 do (30 <-> 150), tam 90 do
#define CENTER_ANGLE   90
#define MIN_ANGLE      30
#define MAX_ANGLE      150
#define SERVO_STEP     1
#define SERVO_DELAY_MS 15
// Toc do goc ~ SERVO_STEP / SERVO_DELAY_MS = 1/15ms = ~66 do/s
// (tuong duong toc do muot cua cau hinh cu 60-120, step=2/40ms = 50 do/s,
//  dieu chinh nhe de bu bien do quet rong hon ma khong gay giat do qua tai servo)

// ===================== KHOẢNG CÁCH =====================
#define SONAR_INTERVAL_MS 100
#define SONAR_TIMEOUT_US  30000
#define SONAR_FILTER_SIZE 5   // Median filter 5 mau

// ===================== YOLO =====================
#define YOLO_TIMEOUT_MS 5000

// ===================== SERVER DATA =====================
WebServer dataServer(8080);

// ===================== BIẾN TOÀN CỤC =====================
Servo scanServo;

float distanceCm = -1;
float sonarBuffer[SONAR_FILTER_SIZE];
int sonarBufferIndex = 0;
bool sonarBufferFilled = false;

int servoAngle = CENTER_ANGLE;
int servoDir = 1;

bool cameraOK = false;
bool yoloOK = false;
bool buzzerState = false;

uint32_t lastSonarMs = 0;
uint32_t lastServoMs = 0;
uint32_t lastBuzzerMs = 0;
uint32_t lastLogMs = 0;
uint32_t lastYoloMs = 0;
uint32_t lastWifiReconnectMs = 0;
uint32_t lastMemCheckMs = 0;
uint32_t wifiReconnectAttempts = 0;

enum WarningLevel {
  SAFE,
  OBSTACLE,
  DANGEROUS,
  EXTREME_DANGER
};

WarningLevel warningLevel = SAFE;

// ===================== IN LINK WEB =====================
void printLinks() {
  IPAddress ip = WiFi.localIP();

  Serial.println();
  Serial.println("====================================");
  Serial.print("[WEB CAMERA] http://");
  Serial.println(ip);

  Serial.print("[STREAM]     http://");
  Serial.print(ip);
  Serial.println(":81/stream");

  Serial.print("[DATA API]   http://");
  Serial.print(ip);
  Serial.println(":8080/data");

  Serial.print("[YOLO API]   http://");
  Serial.print(ip);
  Serial.println(":8080/yolo?ok=1");
  Serial.println("====================================");
  Serial.println();
}

// ===================== CAMERA =====================
bool initCamera() {
  camera_config_t config;

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk  = XCLK_GPIO_NUM;
  config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href  = HREF_GPIO_NUM;

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size   = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count     = 2;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.grab_mode    = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size   = FRAMESIZE_QVGA;
    config.jpeg_quality = 15;
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK) {
    Serial.printf("[CAMERA] LOI khoi tao: 0x%x\n", err);
    return false;
  }

  Serial.println("[CAMERA] Khoi tao OK");

  // ===================== SUA HUONG CAMERA =====================
  sensor_t *s = esp_camera_sensor_get();

  if (s != NULL) {
    s->set_vflip(s, 1);      // sửa ảnh bị lộn ngược trên/dưới
    s->set_hmirror(s, 1);    // sửa ảnh bị ngược trái/phải

    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);

    Serial.println("[CAMERA] Da sua huong anh: vflip=1, hmirror=1");
  } else {
    Serial.println("[CAMERA] Khong lay duoc sensor");
  }

  return true;
}

// ===================== WIFI =====================
void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("[WIFI] Dang ket noi");

  uint32_t startMs = millis();

  while (WiFi.status() != WL_CONNECTED && millis() - startMs < 15000) {
    Serial.print(".");
    delay(300);
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("[WIFI] Ket noi OK");
    printLinks();
  } else {
    Serial.println();
    Serial.println("[WIFI] Ket noi FAIL");
    Serial.println("[WIFI] Kiem tra WiFi 2.4GHz, SSID, PASSWORD");
  }
}

void updateWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    wifiReconnectAttempts = 0;
    return;
  }

  uint32_t now = millis();

  if (now - lastWifiReconnectMs >= 5000) {
    lastWifiReconnectMs = now;
    wifiReconnectAttempts++;
    Serial.printf("[WIFI] Mat ket noi, dang ket noi lai... (lan %u)\n", wifiReconnectAttempts);
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  }
}

// ===================== HC-SR04P =====================
float readSonarRawCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);

  digitalWrite(TRIG_PIN, LOW);

  // LUU Y: pulseIn co the block CPU toi da SONAR_TIMEOUT_US (30ms).
  // Day la han che cua thu vien Arduino pulseIn, khong the lam non-blocking
  // hoan toan ma khong viet lai bang interrupt/RMT. Da giam tac dong bang cach
  // chi goi ham nay toi da 1 lan moi SONAR_INTERVAL_MS (xem updateSonar),
  // va dat timeout vua du (30ms ~ 5m) de khong giu CPU qua lau khi mat tin hieu echo.
  unsigned long duration = pulseIn(ECHO_PIN, HIGH, SONAR_TIMEOUT_US);

  if (duration == 0) {
    return -1;
  }

  return duration * 0.0343f / 2.0f;
}

// Median filter de loc nhieu xung (spike) cua HC-SR04P,
// dac biet huu ich khi servo dang quay gay rung nhe dau cam bien.
float medianFilter(float newSample) {
  static float lastValid = -1;

  if (newSample > 0) {
    sonarBuffer[sonarBufferIndex] = newSample;
  } else {
    // Mau loi (timeout): giu lai gia tri cu gan nhat trong buffer
    sonarBuffer[sonarBufferIndex] = (lastValid > 0) ? lastValid : -1;
  }

  sonarBufferIndex = (sonarBufferIndex + 1) % SONAR_FILTER_SIZE;
  if (sonarBufferIndex == 0) sonarBufferFilled = true;

  int count = sonarBufferFilled ? SONAR_FILTER_SIZE : sonarBufferIndex;
  if (count == 0) return newSample;

  float temp[SONAR_FILTER_SIZE];
  for (int i = 0; i < count; i++) temp[i] = sonarBuffer[i];

  // sort don gian (mang nho, khong can thuat toan phuc tap)
  for (int i = 0; i < count - 1; i++) {
    for (int j = i + 1; j < count; j++) {
      if (temp[j] < temp[i]) {
        float t = temp[i];
        temp[i] = temp[j];
        temp[j] = t;
      }
    }
  }

  float med = temp[count / 2];
  if (med > 0) lastValid = med;
  return med;
}

void updateSonar() {
  uint32_t now = millis();

  if (now - lastSonarMs < SONAR_INTERVAL_MS) return;

  lastSonarMs = now;
  float raw = readSonarRawCm();
  distanceCm = medianFilter(raw);
}

// ===================== WARNING =====================
// Logic nay dua tren khoang cach tuyet doi, khong phu thuoc goc quet servo,
// nen viec doi bien do quet 60-120 -> 30-150 KHONG anh huong nguong canh bao.
void updateWarning() {
  if (distanceCm < 0) {
    warningLevel = SAFE;
    return;
  }

  float m = distanceCm / 100.0f;

  if (m > 4.0f) {
    warningLevel = SAFE;
  } else if (m >= 2.5f) {
    warningLevel = OBSTACLE;
  } else if (m >= 1.5f) {
    warningLevel = DANGEROUS;
  } else {
    warningLevel = EXTREME_DANGER;
  }
}

const char* warningText() {
  switch (warningLevel) {
    case SAFE:
      return "SAFE";
    case OBSTACLE:
      return "OBSTACLE";
    case DANGEROUS:
      return "DANGEROUS";
    case EXTREME_DANGER:
      return "EXTREME_DANGER";
    default:
      return "UNKNOWN";
  }
}

// ===================== SERVO =====================
void initServo() {
  // Dung allocateTimer(1) thay vi (0) de tranh xung dot voi
  // LEDC_TIMER_0 / LEDC_CHANNEL_0 ma camera dang su dung cho XCLK.
  ESP32PWM::allocateTimer(1);
  scanServo.setPeriodHertz(50);
  scanServo.attach(SERVO_PIN, 500, 2400);
  scanServo.write(CENTER_ANGLE);

  Serial.println("[SERVO] Khoi tao OK (timer 1, goc 30-150, tam 90)");
}

void updateServo() {
  uint32_t now = millis();

  if (now - lastServoMs < SERVO_DELAY_MS) return;

  lastServoMs = now;

  servoAngle += servoDir * SERVO_STEP;

  if (servoAngle >= MAX_ANGLE) {
    servoAngle = MAX_ANGLE;
    servoDir = -1;
  }

  if (servoAngle <= MIN_ANGLE) {
    servoAngle = MIN_ANGLE;
    servoDir = 1;
  }

  scanServo.write(servoAngle);
}

// ===================== BUZZER =====================
void initBuzzer() {
#if ENABLE_BUZZER
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.println("[BUZZER] Khoi tao OK");
#endif
}

void updateBuzzer() {
#if ENABLE_BUZZER
  uint32_t now = millis();

  bool beep = false;
  uint32_t interval = 0;

  // 1. Mất WiFi -> cảnh báo hệ thống
  if (WiFi.status() != WL_CONNECTED) {
    beep = true;
    interval = 300;
  }
  else {
    // 2. WiFi OK -> chỉ cảnh báo khi vật cản dưới 1.5m
    switch (warningLevel) {

      case SAFE:
      case OBSTACLE:     // >1.5m
        beep = false;
        break;

      case DANGEROUS:    // 0.5m - 1.5m
        beep = true;
        interval = 300;
        break;

      case EXTREME_DANGER:   // <0.5m
        beep = true;
        interval = 100;
        break;
    }
  }

  if (!beep) {
    digitalWrite(BUZZER_PIN, LOW);
    buzzerState = false;
    return;
  }

  if (now - lastBuzzerMs >= interval) {
    lastBuzzerMs = now;
    buzzerState = !buzzerState;
    digitalWrite(BUZZER_PIN, buzzerState);
  }
#endif
}
// ===================== DATA API =====================
void handleData() {
  JsonDocument doc;

  doc["distance_cm"] = distanceCm;
  doc["distance_m"] = distanceCm > 0 ? distanceCm / 100.0f : -1;
  doc["angle"] = servoAngle;
  doc["warning"] = warningText();

  doc["wifi_ok"] = WiFi.status() == WL_CONNECTED;
  doc["camera_ok"] = cameraOK;
  doc["yolo_ok"] = yoloOK;

  String output;
  serializeJson(doc, output);

  dataServer.sendHeader("Access-Control-Allow-Origin", "*");
  dataServer.send(200, "application/json", output);
}

void handleYolo() {
  if (dataServer.hasArg("ok") && dataServer.arg("ok") == "1") {
    lastYoloMs = millis();
    yoloOK = true;
    dataServer.send(200, "text/plain", "OK");
    Serial.println("[YOLO] Heartbeat OK");
  } else {
    dataServer.send(400, "text/plain", "BAD_REQUEST");
  }
}

void initDataServer() {
  dataServer.on("/data", HTTP_GET, handleData);
  dataServer.on("/yolo", HTTP_GET, handleYolo);
  dataServer.begin();

  Serial.println("[DATA SERVER] OK cong 8080");
}

void updateYolo() {
  if (millis() - lastYoloMs > YOLO_TIMEOUT_MS) {
    yoloOK = false;
  }
}

// ===================== MEMORY MONITOR =====================
// Theo doi heap / PSRAM con trong de phat hien som memory leak
// hoac thieu bo nho khi stream camera chay lau.
void updateMemoryMonitor() {
  uint32_t now = millis();

  if (now - lastMemCheckMs < 10000) return;

  lastMemCheckMs = now;

  uint32_t freeHeap = ESP.getFreeHeap();
  uint32_t freePsram = psramFound() ? ESP.getFreePsram() : 0;

  Serial.printf("[MEM] FreeHeap=%u bytes  FreePSRAM=%u bytes\n", freeHeap, freePsram);

  if (freeHeap < 20000) {
    Serial.println("[MEM] CANH BAO: Heap thap, co the sap het bo nho!");
  }
}

// ===================== LOG =====================
void updateLog() {
  uint32_t now = millis();

  if (now - lastLogMs < 1000) return;

  lastLogMs = now;

  Serial.printf("[LOG] dist=%.1fcm angle=%d warn=%s wifi=%d cam=%d yolo=%d\n",
                distanceCm,
                servoAngle,
                warningText(),
                WiFi.status() == WL_CONNECTED,
                cameraOK,
                yoloOK);
}

// ===================== SETUP =====================
void setup() {
  Serial.begin(115200);
  delay(300);

  Serial.println();
  Serial.println("====================================");
  Serial.println(" ESP32-S3-CAM HO TRO NGUOI KHIEM THI");
  Serial.println(" Camera + HC-SR04P + Servo + Buzzer");
  Serial.println(" Servo quet 30-150 do (bien do 120 do)");
  Serial.println("====================================");

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  for (int i = 0; i < SONAR_FILTER_SIZE; i++) sonarBuffer[i] = -1;

  initBuzzer();
  initServo();

  connectWiFi();

  cameraOK = initCamera();

  if (cameraOK) {
    startCameraServer();
    Serial.println("[WEB CAMERA] startCameraServer OK");
  } else {
    Serial.println("[WEB CAMERA] Khong chay vi camera loi");
  }

  initDataServer();

  lastYoloMs = millis();
  yoloOK = false;

  if (WiFi.status() == WL_CONNECTED) {
    printLinks();
  }
}

// ===================== LOOP =====================
void loop() {
  updateWiFi();

  updateSonar();
  updateWarning();

  updateServo();
  updateYolo();
  updateBuzzer();

  dataServer.handleClient();

  updateLog();
  updateMemoryMonitor();

  // Nhuong CPU cho cac task he thong (WiFi, watchdog) de tranh
  // Task Watchdog Timer reset khi loop chay lien tuc qua nhanh.
  yield();
}