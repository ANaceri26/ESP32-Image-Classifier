// XIAO ESP32S3 Sense: live MJPEG stream + on-device cat/dog classification over Wi-Fi.
// Board XIAO_ESP32S3, PSRAM OPI. Needs tflm_esp32 and model_data.h from training/train.py.

#include <WiFi.h>
#include <ESPmDNS.h>
#include "esp_http_server.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "esp_heap_caps.h"
#include <tflm_esp32.h>
#include <new>
#include "model_data.h"

void step(const char *what) {
  bool ok = heap_caps_check_integrity_all(true);
  Serial.printf("      %-34s heap %s\n", what, ok ? "OK" : "*** CORRUPTED ***");
  Serial.flush();
}

const char *WIFI_SSID = "YOUR_HOTSPOT_NAME";      // same network as your laptop (2.4 GHz)
const char *WIFI_PASS = "YOUR_HOTSPOT_PASSWORD";
const char *AP_SSID = "CatDogCam";                // fallback network if the above fails
const char *AP_PASS = "catdog123";
const char *HOSTNAME = "catdogcam";               // also reachable as http://catdogcam.local
const float CONFIDENT = 0.70f;                    // below this, say "not sure"
const size_t ARENA_BYTES = 1024 * 1024;           // model working memory, in PSRAM (8 MB available)
#ifdef LED_BUILTIN
const int LED_PIN = LED_BUILTIN;                  // lights when the last photo was a dog (active LOW)
#else
const int LED_PIN = 21;
#endif

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
  c.pixel_format = PIXFORMAT_RGB565;  // raw pixels: the model reads them directly; the stream JPEG-encodes them
  c.frame_size = FRAMESIZE_QVGA;      // 320 x 240
  c.fb_count = 2;                     // one for the live stream, one for "Take image"
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;   // always the newest frame
  return esp_camera_init(&c) == ESP_OK;
}

constexpr int kNumOps = 14;
tflite::MicroMutableOpResolver<kNumOps> resolver;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr, *output = nullptr;
uint8_t *arena = nullptr;

bool initModel() {
  step("before GetModel");
  const tflite::Model *model = tflite::GetModel(g_model);
  step("after GetModel");
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch");
    return false;
  }
  // ops used by MobileNetV2; add here if AllocateTensors reports a missing op
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddPad();
  resolver.AddRelu6();
  resolver.AddMean();            // GlobalAveragePooling2D
  resolver.AddAveragePool2D();
  resolver.AddFullyConnected();  // Dense layer
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  step("after adding ops");

  arena = (uint8_t *)heap_caps_malloc(ARENA_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  step("after arena malloc");
  if (!arena) {
    Serial.println("No PSRAM! Set Tools > PSRAM > OPI PSRAM");
    return false;
  }
  // prebuilt tflm_esp32 MicroInterpreter is larger than sizeof() reports, so placement-new into a padded buffer
  static const size_t SLACK = 8192;
  alignas(16) static uint8_t interpreterMem[SLACK];
  memset(interpreterMem, 0xA5, SLACK);
  interpreter = new (interpreterMem) tflite::MicroInterpreter(model, resolver, arena, ARENA_BYTES);
  size_t used = SLACK;
  while (used > 0 && interpreterMem[used - 1] == 0xA5) used--;
  Serial.printf("      MicroInterpreter: header says %u bytes, library really used ~%u bytes\n",
                (unsigned)sizeof(tflite::MicroInterpreter), (unsigned)used);
  step("after creating MicroInterpreter");
  TfLiteStatus alloc = interpreter->AllocateTensors();
  step("after AllocateTensors");
  if (alloc != kTfLiteOk) {
    Serial.println("AllocateTensors failed (arena too small or op missing)");
    return false;
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.printf("Model ready: input %dx%dx%dx%d int8 (scale %.5f, zero point %d), model %u KB\n",
                input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3],
                input->params.scale, (int)input->params.zero_point, g_model_len / 1024);
  return true;
}

// center crop, area resize, scale to -1..1, quantize (must match training)
void frameToInput(const camera_fb_t *fb) {
  const int W = fb->width, H = fb->height, N = input->dims->data[1];
  const int side = min(W, H), x0 = (W - side) / 2, y0 = (H - side) / 2;
  const float scale = input->params.scale;
  const int zp = input->params.zero_point;
  int8_t *dst = input->data.int8;
  for (int oy = 0; oy < N; oy++) {
    for (int ox = 0; ox < N; ox++) {
      int sx0 = x0 + ox * side / N, sx1 = x0 + (ox + 1) * side / N;
      int sy0 = y0 + oy * side / N, sy1 = y0 + (oy + 1) * side / N;
      uint32_t r = 0, g = 0, b = 0, n = 0;
      for (int y = sy0; y < sy1; y++)
        for (int x = sx0; x < sx1; x++) {
          const uint8_t *p = fb->buf + 2 * (y * W + x);
          uint16_t px = (p[0] << 8) | p[1];  // camera stores RGB565 high byte first
          r += ((px >> 11) & 0x1F) << 3;
          g += ((px >> 5) & 0x3F) << 2;
          b += (px & 0x1F) << 3;
          n++;
        }
      float rgb[3] = {(float)r / n, (float)g / n, (float)b / n};
      for (int c = 0; c < 3; c++) {
        float v = rgb[c] / 127.5f - 1.0f;                // same as MobileNetV2 preprocess_input
        int q = (int)lroundf(v / scale) + zp;            // float -> int8
        *dst++ = (int8_t)constrain(q, -128, 127);
      }
    }
  }
}

httpd_handle_t webServer = nullptr, streamServer = nullptr;
uint8_t *snapJpg = nullptr;   // the last photo that was classified
size_t snapLen = 0;

const char PAGE[] = R"HTML(<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Cat or Dog</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--ink:#e8eaed;--dim:#9aa0a6;--acc:#4f8cff;--line:#2a2e37}
*{box-sizing:border-box}body{margin:0;font:15px/1.4 system-ui,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}
main{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px;max-width:1100px;margin:0 auto;padding:16px}
@media(max-width:800px){main{grid-template-columns:minmax(0,1fr)}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--dim);margin:0 0 10px}
img{width:100%;border-radius:8px;display:block;background:#000;aspect-ratio:4/3;object-fit:contain}
button{width:100%;margin-top:12px;padding:14px;font:600 17px system-ui;border:0;border-radius:8px;background:var(--acc);color:#fff;cursor:pointer}
button:disabled{opacity:.6;cursor:default}
#label{font-size:44px;font-weight:700;margin:12px 0 4px;text-transform:capitalize}
.bar{display:flex;align-items:center;gap:8px;margin-top:6px}.bar span{width:34px;color:var(--dim)}
.bar div{flex:1;height:10px;background:var(--line);border-radius:5px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
.dim{color:var(--dim);font-size:13px}
</style></head><body><main>
<section class="card"><h2>Live view</h2><img id="live" alt="Live camera view">
<button id="snap" onclick="take()">Take image</button><div class="dim" style="margin-top:8px">Tip: press Space to take an image. The square in the middle of the frame is what the model sees.</div></section>
<section class="card"><h2>Result</h2><img id="shot" alt="Last photo taken">
<div id="label">-</div>
<div class="bar"><span>Cat</span><div><i id="bc" style="width:0"></i></div><b id="pc">-</b></div>
<div class="bar"><span>Dog</span><div><i id="bd" style="width:0"></i></div><b id="pd">-</b></div>
<div class="dim" id="info" style="margin-top:10px">Take an image to classify it.</div></section>
</main><script>
live.src = `http://${location.hostname}:81/stream`;
async function take(){
  snap.disabled = true; snap.textContent = "Thinking...";
  try{
    const r = await (await fetch("/capture",{method:"POST"})).json();
    shot.src = "/snapshot.jpg?" + Date.now();
    label.textContent = r.label;
    bc.style.width = r.cat*100 + "%"; pc.textContent = (r.cat*100).toFixed(0) + "%";
    bd.style.width = r.dog*100 + "%"; pd.textContent = (r.dog*100).toFixed(0) + "%";
    info.textContent = `Model ran on the ESP32 in ${r.ms} ms`;
  }catch(e){ info.textContent = "Couldn't reach the ESP - is it still on the network?"; }
  snap.disabled = false; snap.textContent = "Take image";
}
document.addEventListener("keydown", e => { if (e.code === "Space" && !snap.disabled) { e.preventDefault(); take(); } });
</script></body></html>)HTML";

esp_err_t pageHandler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, PAGE, HTTPD_RESP_USE_STRLEN);
}

esp_err_t captureHandler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { httpd_resp_send_500(req); return ESP_FAIL; }

  frameToInput(fb);
  uint32_t t0 = millis();
  TfLiteStatus st = interpreter->Invoke();
  uint32_t ms = millis() - t0;

  uint8_t *jpg = nullptr; size_t len = 0;  // keep this exact frame so the page can show it
  if (frame2jpg(fb, 80, &jpg, &len)) { free(snapJpg); snapJpg = jpg; snapLen = len; }
  esp_camera_fb_return(fb);
  if (st != kTfLiteOk) { httpd_resp_send_500(req); return ESP_FAIL; }

  float probs[8] = {0};
  int best = 0;
  for (int i = 0; i < MODEL_NUM_LABELS; i++) {  // int8 output -> probabilities
    probs[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    if (probs[i] > probs[best]) best = i;
  }
  const char *label = probs[best] >= CONFIDENT ? MODEL_LABELS[best] : "not sure";
  digitalWrite(LED_PIN, strcmp(label, "dog") == 0 ? LOW : HIGH);
  Serial.printf("Photo: %-8s cat %3.0f%%  dog %3.0f%%  (model %lu ms)\n", label, probs[0] * 100, probs[1] * 100,
                (unsigned long)ms);

  char json[120];
  snprintf(json, sizeof(json), "{\"label\":\"%s\",\"cat\":%.3f,\"dog\":%.3f,\"ms\":%lu}", label, probs[0], probs[1],
           (unsigned long)ms);
  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
}

esp_err_t snapshotHandler(httpd_req_t *req) {
  if (!snapJpg) { httpd_resp_send_404(req); return ESP_FAIL; }
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  return httpd_resp_send(req, (const char *)snapJpg, snapLen);
}

#define BOUNDARY "catdogframe"
esp_err_t streamHandler(httpd_req_t *req) {
  httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=" BOUNDARY);
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  char header[80];
  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { delay(10); continue; }
    uint8_t *jpg = nullptr; size_t len = 0;
    bool ok = frame2jpg(fb, 60, &jpg, &len);
    esp_camera_fb_return(fb);
    if (!ok) return ESP_FAIL;
    int hlen = snprintf(header, sizeof(header), "\r\n--" BOUNDARY "\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                        (unsigned)len);
    esp_err_t res = httpd_resp_send_chunk(req, header, hlen);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)jpg, len);
    free(jpg);
    if (res != ESP_OK) break;  // browser tab closed
  }
  return ESP_OK;
}

void startWeb() {
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.stack_size = 16384;  // the model runs inside this server's task
  cfg.server_port = 80;
  if (httpd_start(&webServer, &cfg) == ESP_OK) {
    httpd_uri_t page = {"/", HTTP_GET, pageHandler, nullptr};
    httpd_uri_t cap = {"/capture", HTTP_POST, captureHandler, nullptr};
    httpd_uri_t snap = {"/snapshot.jpg", HTTP_GET, snapshotHandler, nullptr};
    httpd_register_uri_handler(webServer, &page);
    httpd_register_uri_handler(webServer, &cap);
    httpd_register_uri_handler(webServer, &snap);
  }
  httpd_config_t scfg = HTTPD_DEFAULT_CONFIG();
  scfg.server_port = 81;
  scfg.ctrl_port = cfg.ctrl_port + 1;  // each server needs its own control port
  scfg.stack_size = 8192;
  if (httpd_start(&streamServer, &scfg) == ESP_OK) {
    httpd_uri_t stream = {"/stream", HTTP_GET, streamHandler, nullptr};
    httpd_register_uri_handler(streamServer, &stream);
  }
}

void startWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // lower latency for the video stream
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("Joining \"%s\"", WIFI_SSID);
  for (int i = 0; i < 30 && WiFi.status() != WL_CONNECTED; i++) { delay(500); Serial.print("."); }
  IPAddress ip;
  if (WiFi.status() == WL_CONNECTED) {
    ip = WiFi.localIP();
    Serial.println(" connected");
  } else {  // fallback: make our own network
    Serial.printf(" failed. Making my own network \"%s\" (password %s) - join it from the laptop.\n", AP_SSID, AP_PASS);
    WiFi.disconnect(true);
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASS);
    ip = WiFi.softAPIP();
  }
  if (MDNS.begin(HOSTNAME)) MDNS.addService("http", "tcp", 80);
  Serial.printf("\n>>> Open in your laptop's browser:  http://%s   (or http://%s.local)\n\n",
                ip.toString().c_str(), HOSTNAME);
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 5000) delay(10);  // wait (max 5 s) for Serial Monitor to reconnect
  delay(500);
  Serial.println("\n=== Cat/Dog cam starting ===");
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);
  Serial.printf("[1/4] PSRAM: %s (%u KB free)\n", psramFound() ? "found" : "NOT FOUND - set Tools > PSRAM > OPI PSRAM",
                (unsigned)(ESP.getFreePsram() / 1024));
  step("at start");
  Serial.println("[2/4] Starting camera...");
  if (!initCamera()) { Serial.println("Camera init failed - is the camera board clipped on?"); while (true) delay(1000); }
  Serial.println("      camera OK");
  step("right after camera init");
  delay(1000);  // let the camera fill frames for a second
  step("1 s after camera init");
  Serial.println("[3/4] Loading model...");
  if (!initModel()) { while (true) delay(1000); }
  Serial.println("[4/4] Starting Wi-Fi...");
  startWifi();
  startWeb();
}

void loop() {
  delay(1000);  // everything runs in the web server tasks
}
