// XIAO ESP32S3 Sense: camera + on-device cat/dog classification over USB serial (see laptop/camlink.py).
// Commands: 'C' JPEG, 'K'+N*N*3 classify, 'I' status, 'R'+q|v|s|x resolution.
// Reply: 4-byte tag (IMG1/RES1/TXT1/ERR1) + uint32 LE length + payload.

#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <tflm_esp32.h>
#include <new>
#include "model_data.h"

const size_t ARENA_BYTES = 1024 * 1024;  // model working memory, in PSRAM

String statusLog;  // collected here and sent on 'I', instead of printing
void note(const String &s) { statusLog += s; statusLog += '\n'; }

void sendPacket(const char *tag, const uint8_t *data, uint32_t len) {
  uint8_t hdr[8] = {(uint8_t)tag[0], (uint8_t)tag[1], (uint8_t)tag[2], (uint8_t)tag[3],
                    (uint8_t)len, (uint8_t)(len >> 8), (uint8_t)(len >> 16), (uint8_t)(len >> 24)};
  Serial.write(hdr, 8);
  if (len) Serial.write(data, len);
  Serial.flush();
}
void sendText(const char *tag, const String &s) { sendPacket(tag, (const uint8_t *)s.c_str(), s.length()); }

bool cameraOK = false;

bool initCamera() {
  camera_config_t c = {};
  c.pin_pwdn = -1;  c.pin_reset = -1;  c.pin_xclk = 10;
  c.pin_sccb_sda = 40;  c.pin_sccb_scl = 39;
  c.pin_d7 = 48; c.pin_d6 = 11; c.pin_d5 = 12; c.pin_d4 = 14;
  c.pin_d3 = 16; c.pin_d2 = 18; c.pin_d1 = 17; c.pin_d0 = 15;
  c.pin_vsync = 38; c.pin_href = 47; c.pin_pclk = 13;
  c.xclk_freq_hz = 20000000;
  c.ledc_timer = LEDC_TIMER_0;
  c.ledc_channel = LEDC_CHANNEL_0;
  c.pixel_format = PIXFORMAT_JPEG;   // the camera compresses itself: sharp image, small transfer
  c.frame_size = FRAMESIZE_UXGA;     // buffers sized for the biggest mode we may switch to...
  c.jpeg_quality = 10;               // 0-63, LOWER = better quality (10 is high quality)
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;  // always the newest frame
  if (esp_camera_init(&c) != ESP_OK) return false;
  sensor_t *s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_VGA);  // ...then start at 640 x 480
  note("Camera sensor PID 0x" + String((unsigned int)s->id.PID, HEX) + ", 640x480 JPEG");
  return true;
}

constexpr int kNumOps = 14;
tflite::MicroMutableOpResolver<kNumOps> resolver;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr, *output = nullptr;
uint8_t *arena = nullptr;
bool modelTried = false, modelOK = false;

bool initModel() {
  modelTried = true;
  const tflite::Model *model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) { note("Model schema version mismatch"); return false; }
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddPad();
  resolver.AddRelu6();
  resolver.AddMean();
  resolver.AddAveragePool2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  arena = (uint8_t *)heap_caps_malloc(ARENA_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!arena) { note("No PSRAM - set Tools > PSRAM > OPI PSRAM"); return false; }

  // prebuilt tflm_esp32 MicroInterpreter is larger than sizeof() reports, so placement-new into a padded buffer
  static const size_t SLACK = 8192;
  alignas(16) static uint8_t interpreterMem[SLACK];
  memset(interpreterMem, 0xA5, SLACK);
  interpreter = new (interpreterMem) tflite::MicroInterpreter(model, resolver, arena, ARENA_BYTES);
  size_t used = SLACK;
  while (used > 0 && interpreterMem[used - 1] == 0xA5) used--;
  note("MicroInterpreter: header says " + String(sizeof(tflite::MicroInterpreter)) + " bytes, library used ~" +
       String(used));
  if (!heap_caps_check_integrity_all(false)) { note("Memory corrupted while creating the model"); return false; }

  if (interpreter->AllocateTensors() != kTfLiteOk) { note("AllocateTensors failed (op missing or arena too small)"); return false; }
  input = interpreter->input(0);
  output = interpreter->output(0);
  note("Model ready: input " + String(input->dims->data[1]) + "x" + String(input->dims->data[2]) + "x" +
       String(input->dims->data[3]) + ", scale " + String(input->params.scale, 6) + ", zero point " +
       String(input->params.zero_point));
  return true;
}

void handleCapture() {
  if (!cameraOK) { sendText("ERR1", "camera not started"); return; }
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { sendText("ERR1", "capture failed"); return; }
  sendPacket("IMG1", fb->buf, fb->len);
  esp_camera_fb_return(fb);  // give the buffer back so the camera can reuse it
}

uint8_t *rgbIn = nullptr;

void handleClassify() {
  const int N = MODEL_INPUT_SIZE;
  const size_t want = (size_t)N * N * 3;
  if (!rgbIn) rgbIn = (uint8_t *)heap_caps_malloc(want, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  size_t got = rgbIn ? Serial.readBytes(rgbIn, want) : 0;  // the laptop sends the pixels right after 'K'
  if (got != want) { sendText("ERR1", "expected " + String(want) + " pixel bytes, got " + String(got)); return; }

  if (!modelTried) modelOK = initModel();
  if (!modelOK) { sendText("ERR1", "model not running:\n" + statusLog); return; }

  // pixels 0..255 -> -1..1 (as in training) -> int8 using the model's own scale and zero point
  const float scale = input->params.scale;
  const int zp = input->params.zero_point;
  for (size_t i = 0; i < want; i++) {
    int q = (int)lroundf((rgbIn[i] / 127.5f - 1.0f) / scale) + zp;
    input->data.int8[i] = (int8_t)constrain(q, -128, 127);
  }
  uint32_t t0 = millis();
  TfLiteStatus st = interpreter->Invoke();
  uint32_t ms = millis() - t0;
  if (st != kTfLiteOk) { sendText("ERR1", "model run failed"); return; }

  String json = "{\"ms\":" + String(ms) + ",\"labels\":[";
  for (int i = 0; i < MODEL_NUM_LABELS; i++) json += String(i ? "," : "") + "\"" + MODEL_LABELS[i] + "\"";
  json += "],\"probs\":[";
  for (int i = 0; i < MODEL_NUM_LABELS; i++) {
    float p = (output->data.int8[i] - output->params.zero_point) * output->params.scale;  // int8 -> probability
    json += String(i ? "," : "") + String(p, 4);
  }
  json += "]}";
  sendText("RES1", json);
}

void handleResolution() {
  int c = -1;
  for (int i = 0; i < 200 && c < 0; i++) { c = Serial.read(); if (c < 0) delay(1); }
  framesize_t fs = c == 'q' ? FRAMESIZE_QVGA : c == 's' ? FRAMESIZE_SVGA : c == 'x' ? FRAMESIZE_XGA : FRAMESIZE_VGA;
  if (!cameraOK) { sendText("ERR1", "camera not started"); return; }
  sensor_t *s = esp_camera_sensor_get();
  s->set_framesize(s, fs);
  for (int i = 0; i < 2; i++) {  // drop frames taken at the old size
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }
  sendText("TXT1", "resolution set");
}

void handleInfo() {
  String s = statusLog;
  s += "input_size=" + String(MODEL_INPUT_SIZE) + "\n";
  s += "free PSRAM " + String(ESP.getFreePsram() / 1024) + " KB, free RAM " + String(ESP.getFreeHeap() / 1024) + " KB\n";
  sendText("TXT1", s);
}

void setup() {
  esp_log_level_set("*", ESP_LOG_NONE);  // stop library log text from mixing into image bytes
  Serial.setRxBufferSize(32768);         // room for the pixels the laptop sends
  Serial.begin(115200);                  // (speed is ignored on native USB; it always runs at USB speed)
  Serial.setTimeout(3000);
  Serial.setTxTimeoutMs(2000);
  note(psramFound() ? "PSRAM OK" : "PSRAM NOT FOUND - set Tools > PSRAM > OPI PSRAM");
  cameraOK = initCamera();
  if (!cameraOK) note("Camera init failed - is the camera board clipped on?");
}

void loop() {
  int c = Serial.read();
  if (c < 0) { delay(1); return; }
  switch (c) {
    case 'C': handleCapture(); break;
    case 'K': handleClassify(); break;
    case 'I': handleInfo(); break;
    case 'R': handleResolution(); break;
    default: break;  // ignore anything else
  }
}
