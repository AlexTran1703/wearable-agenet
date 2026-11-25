// ESP-12F ECG node — microSD data logging and transfer (MDPI BioMedInformatics prototype).
//
// Acquisition is mimicked by replaying a stored 5-min simulator session from the SD card at 200 Hz (the ADC on A0
// will replace SdReplaySource later; logging and transfer do not change). Every acquisition is written to a new
// log file LOG/Rnnnn.BIN (single lead II, 12-bit AD8232 ADC codes as uint16 LE) with a sidecar LOG/Rnnnn.JSN (fs, leads, samples,
// SHA-256 computed while recording).
//
// Serial commands (460800 baud, one per line):
//   PING                       -> PONG
//   INFO                       -> INFO,fw=...,sd=0|1,wifi=0|1,ip=...,source=sd|adc
//   LS [dir]                   -> FILE,<path>,<bytes> ... then OK,LS
//   SHA <path>                 -> SHA,<path>,<hex sha256>
//   RECORD <session> [seconds] -> paced acquisition from ECG/<session> into a new log; OK,RECORD,<log>,<n>,<sha>
//   DUMP <path>                -> BIN,<path>,<bytes>\n then raw bytes, then \nSHA,<hex>  (serial transfer)
//   WIFI                       -> connects (credentials in secrets.h); OK,WIFI,<ip>
//   PUT <path> <bytes>         -> READY,<bytes>; then raw bytes; OK,PUT,<path>,<n>,<sha>  (serial file copy to SD)
//   UPLOAD <path>              -> HTTP POST to GATEWAY_URL with X-Device-ID, X-Timestamp, X-Nonce,
//                                 X-Payload-SHA256, X-Signature = HMAC-SHA256(key, id|ts|nonce|sha)
#include <Arduino.h>
#include <ESP8266HTTPClient.h>
#include <ESP8266WiFi.h>
#include <WiFiClientSecureBearSSL.h>
#include <SD.h>
#include <SPI.h>
#include <bearssl/bearssl_hash.h>
#include <bearssl/bearssl_hmac.h>
#include <time.h>

#include "config.h"
#include "secrets.h"

static const char *FW = "ecgnode-sd-1.5.0";
static bool sdOk = false;

// ---------------------------------------------------------------- helpers
static String hex(const uint8_t *b, size_t n) {
  static const char *H = "0123456789abcdef";
  String s; s.reserve(2 * n);
  for (size_t i = 0; i < n; i++) { s += H[b[i] >> 4]; s += H[b[i] & 15]; }
  return s;
}

static size_t unhex(const char *s, uint8_t *out, size_t max) {
  size_t n = 0;
  auto v = [](char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; };
  while (s[0] && s[1] && n < max) { out[n++] = (v(s[0]) << 4) | v(s[1]); s += 2; }
  return n;
}

static String sha256File(const String &path) {
  File f = SD.open(path, "r");
  if (!f) return "";
  br_sha256_context c; br_sha256_init(&c);
  uint8_t buf[512]; int n;
  while ((n = f.read(buf, sizeof buf)) > 0) { br_sha256_update(&c, buf, n); yield(); }
  f.close();
  uint8_t d[32]; br_sha256_out(&c, d);
  return hex(d, 32);
}

static String nextLogName() {
  if (!SD.exists("/LOG")) SD.mkdir("/LOG");
  for (int i = 1; i < 10000; i++) {
    char p[24]; snprintf(p, sizeof p, "/LOG/R%04d.BIN", i);
    if (!SD.exists(p)) return String(p);
  }
  return "";
}

// ---------------------------------------------------------------- sample sources
struct Source {
  virtual bool begin() = 0;
  virtual bool next(int16_t frame[N_LEADS]) = 0;       // one sample per lead
  virtual void end() {}
  virtual ~Source() {}
};

struct SdReplaySource : Source {                        // mimics the ADC with a stored session
  String path; File f;
  explicit SdReplaySource(const String &p) : path(p) {}
  bool begin() override { f = SD.open(path, "r"); return (bool)f; }
  bool next(int16_t fr[N_LEADS]) override { return f.read((uint8_t *)fr, 2 * N_LEADS) == 2 * N_LEADS; }
  void end() override { if (f) f.close(); }
};

struct AdcSource : Source {                             // placeholder for the A0 front end (single lead)
  bool begin() override { return true; }
  bool next(int16_t fr[N_LEADS]) override { fr[0] = (int16_t)(analogRead(A0) << 2); return true; }  // 10-bit A0 scaled to 12-bit codes
};

// ---------------------------------------------------------------- acquisition / logging
static void record(const String &session, uint32_t seconds) {
  String src = session.startsWith("/") ? session : "/ECG/" + session;
#if ECG_SOURCE_ADC
  AdcSource s;
#else
  SdReplaySource s(src);
#endif
  if (!s.begin()) { Serial.println("ERR,RECORD,source " + src); return; }
  String logPath = nextLogName();
  File out = SD.open(logPath, "w");
  if (!out) { s.end(); Serial.println("ERR,RECORD,cannot create log"); return; }
  br_sha256_context c; br_sha256_init(&c);
  static int16_t chunk[CHUNK_SAMPLES * N_LEADS];
  const uint32_t period = 1000000UL / FS_HZ, total = seconds * FS_HZ;
  uint32_t n = 0, k = 0, late = 0, t0 = micros();
  int16_t fr[N_LEADS];
  while (n < total && s.next(fr)) {
    uint32_t due = t0 + n * period;                     // pace at 200 Hz like a real ADC
    while ((int32_t)(micros() - due) < 0) {}
    if ((int32_t)(micros() - due) > (int32_t)period) late++;
    memcpy(&chunk[k * N_LEADS], fr, sizeof fr);
    n++; k++;
    if (k == CHUNK_SAMPLES) {
      out.write((uint8_t *)chunk, k * 2 * N_LEADS); br_sha256_update(&c, chunk, k * 2 * N_LEADS); k = 0; yield();
    }
  }
  if (k) { out.write((uint8_t *)chunk, k * 2 * N_LEADS); br_sha256_update(&c, chunk, k * 2 * N_LEADS); }
  out.close(); s.end();
  uint8_t d[32]; br_sha256_out(&c, d);
  String sha = hex(d, 32);
  String meta = logPath; meta.replace(".BIN", ".JSN");
  File j = SD.open(meta, "w");
  j.printf("{\"fw\":\"%s\",\"source\":\"%s\",\"fs\":%d,\"leads\":\"II\",\"format\":\"" SAMPLE_FORMAT "\",\"samples\":%lu,"
           "\"duration_ms\":%lu,\"late_samples\":%lu,\"sha256\":\"%s\"}\n",
           FW, src.c_str(), FS_HZ, (unsigned long)n, (unsigned long)((micros() - t0) / 1000), (unsigned long)late, sha.c_str());
  j.close();
  Serial.printf("OK,RECORD,%s,%lu,%s,late=%lu\n", logPath.c_str(), (unsigned long)n, sha.c_str(), (unsigned long)late);
}

// ---------------------------------------------------------------- transfer
static void dump(const String &path) {                  // serial transfer of a file
  File f = SD.open(path, "r");
  if (!f) { Serial.println("ERR,DUMP,open " + path); return; }
  Serial.printf("BIN,%s,%lu\n", path.c_str(), (unsigned long)f.size());
  br_sha256_context c; br_sha256_init(&c);
  uint8_t buf[256]; int n;
  while ((n = f.read(buf, sizeof buf)) > 0) { Serial.write(buf, n); br_sha256_update(&c, buf, n); yield(); }
  f.close();
  uint8_t d[32]; br_sha256_out(&c, d);
  Serial.print("\nSHA,"); Serial.println(hex(d, 32));
}

static bool wifiUp() {
  if (WiFi.status() == WL_CONNECTED) return true;
  WiFi.mode(WIFI_STA); WiFi.setSleepMode(WIFI_NONE_SLEEP);  // modem sleep delays TCP ACKs: throttles uploads
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  for (int i = 0; i < 60 && WiFi.status() != WL_CONNECTED; i++) delay(250);
  if (WiFi.status() != WL_CONNECTED) return false;
  configTime(0, 0, "pool.ntp.org", "time.google.com");
  for (int i = 0; i < 40 && time(nullptr) < 1700000000; i++) delay(250);
  return true;
}

static bool clockOk() {                                 // signed timestamps need NTP time; never send without it
  for (int attempt = 0; attempt < 2 && time(nullptr) < 1700000000; attempt++) {
    configTime(0, 0, "pool.ntp.org", "time.google.com", "time.cloudflare.com");
    for (int i = 0; i < 80 && time(nullptr) < 1700000000; i++) delay(250);
  }
  return time(nullptr) >= 1700000000;
}

static void upload(const String &path) {
  if (!wifiUp()) { Serial.println("ERR,UPLOAD,wifi"); return; }
  if (!clockOk()) { Serial.println("ERR,UPLOAD,clock not synchronised"); return; }
  String sha = sha256File(path);
  if (sha.isEmpty()) { Serial.println("ERR,UPLOAD,open " + path); return; }
  char ts[16]; snprintf(ts, sizeof ts, "%lu", (unsigned long)time(nullptr));
  uint8_t nb[16]; for (int i = 0; i < 16; i += 4) { uint32_t r = RANDOM_REG32; memcpy(nb + i, &r, 4); }
  String nonce = hex(nb, 16);
  String msg = String(DEVICE_ID) + "|" + ts + "|" + nonce + "|" + sha;
  uint8_t key[32]; size_t kl = unhex(DEVICE_KEY_HEX, key, sizeof key);
  br_hmac_key_context kc; br_hmac_key_init(&kc, &br_sha256_vtable, key, kl);
  br_hmac_context hc; br_hmac_init(&hc, &kc, 0); br_hmac_update(&hc, msg.c_str(), msg.length());
  uint8_t mac[32]; br_hmac_out(&hc, mac);
  // GATEWAY_URL = https://host[:port]/path ; HTTPS to EC2 with the server certificate pinned by SHA-1 fingerprint.
  String url = GATEWAY_URL, host, rpath = "/";
  uint16_t port = 443;
  int hs = url.indexOf("://") + 3, ps = url.indexOf('/', hs);
  host = url.substring(hs, ps < 0 ? url.length() : ps);
  if (ps >= 0) rpath = url.substring(ps);
  int colon = host.indexOf(':');
  if (colon >= 0) { port = host.substring(colon + 1).toInt(); host = host.substring(0, colon); }
  File f = SD.open(path, "r");
  uint32_t size = f.size();
  BearSSL::WiFiClientSecure client;
  client.setFingerprint(GATEWAY_FP);
  client.setBufferSizes(4096, 8192);                      // 8-kB TLS records: fewer round trips on the long path
  client.setNoDelay(true);
  client.setTimeout(30000);
  uint32_t t0 = millis();
  if (!client.connect(host.c_str(), port)) {               // TCP + TLS handshake + fingerprint check
    char e[64]; int code = client.getLastSSLError(e, sizeof e); f.close();
    Serial.printf("ERR,UPLOAD,connect,ssl=%d,%s\n", code, e); return;
  }
  uint32_t tConn = millis() - t0;
  client.setSync(false);                                   // do not wait for an ACK after every write
  client.printf("POST %s HTTP/1.1\r\nHost: %s\r\nContent-Type: application/octet-stream\r\nContent-Length: %lu\r\n"
                "X-Device-ID: %s\r\nX-Timestamp: %s\r\nX-Nonce: %s\r\nX-Payload-SHA256: %s\r\nX-Signature: %s\r\n"
                "X-Sample-Format: " SAMPLE_FORMAT "\r\nX-Firmware: %s\r\nConnection: close\r\n\r\n",
                rpath.c_str(), host.c_str(), (unsigned long)size, DEVICE_ID, ts, nonce.c_str(), sha.c_str(),
                hex(mac, 32).c_str(), FW);
  static uint8_t buf[4096];                                // explicit chunked write: every byte confirmed sent
  uint32_t sent = 0, last = millis();
  int n;
  bool ok = true;
  while (ok && (n = f.read(buf, sizeof buf)) > 0) {
    size_t off = 0;
    while (off < (size_t)n) {
      size_t w = client.write(buf + off, n - off);
      if (w > 0) { off += w; last = millis(); }
      else if (!client.connected() || millis() - last > 30000) { ok = false; break; }
      else delay(2);
      yield();
    }
    sent += off;
  }
  f.close();
  uint32_t tSent = millis() - t0;
  if (!ok) { client.stop(); Serial.printf("ERR,UPLOAD,send,%lu/%lu bytes\n", (unsigned long)sent, (unsigned long)size); return; }
  String status = client.readStringUntil('\n');            // "HTTP/1.1 201 Created"
  int code = status.length() > 12 ? status.substring(9, 12).toInt() : -1;
  int clen = -1;
  for (;;) {                                               // headers
    String h = client.readStringUntil('\n'); h.trim();
    if (h.isEmpty()) break;
    if (h.substring(0, 15).equalsIgnoreCase("Content-Length:")) clen = h.substring(15).toInt();
  }
  String body;
  while (client.connected() || client.available()) {
    if (client.available()) { body += (char)client.read(); if (clen >= 0 && (int)body.length() >= clen) break; }
    else delay(1);
  }
  client.stop();
  body.trim();
  Serial.printf("OK,UPLOAD,%s,http=%d,ms=%lu,connect_ms=%lu,sent_ms=%lu,%s\n", path.c_str(), code,
                (unsigned long)(millis() - t0), (unsigned long)tConn, (unsigned long)tSent, body.c_str());
}

static void put(const String &arg) {                    // serial upload of a file onto the SD card
  int s1 = arg.indexOf(' ');
  if (s1 < 0) { Serial.println("ERR,PUT,usage PUT <path> <bytes>"); return; }
  String path = arg.substring(0, s1);
  uint32_t total = arg.substring(s1 + 1).toInt();
  int slash = path.lastIndexOf('/');
  if (slash > 0) SD.mkdir(path.substring(0, slash));
  if (SD.exists(path)) SD.remove(path);
  File f = SD.open(path, "w");
  if (!f) { Serial.println("ERR,PUT,open " + path); return; }
  br_sha256_context c; br_sha256_init(&c);
  static uint8_t buf[512];
  uint32_t got = 0, last = millis();
  Serial.printf("READY,%lu\n", (unsigned long)total);
  while (got < total && millis() - last < 5000) {      // chunked: host sends <= 512 B, waits for '.' after each
    uint32_t want = min((uint32_t)sizeof buf, total - got), have = 0;
    while (have < want && millis() - last < 5000) {
      int n = Serial.readBytes(buf + have, want - have);
      if (n > 0) { have += n; last = millis(); }
      yield();
    }
    f.write(buf, have); br_sha256_update(&c, buf, have); got += have;
    Serial.write('.');
  }
  f.close();
  uint8_t d[32]; br_sha256_out(&c, d);
  if (got != total) { Serial.printf("ERR,PUT,short %lu/%lu\n", (unsigned long)got, (unsigned long)total); return; }
  Serial.printf("OK,PUT,%s,%lu,%s\n", path.c_str(), (unsigned long)got, hex(d, 32).c_str());
}

// ---------------------------------------------------------------- command loop
static void list(const String &dir) {
  File d = SD.open(dir.isEmpty() ? "/ECG" : dir);
  if (!d) { Serial.println("ERR,LS"); return; }
  for (File e = d.openNextFile(); e; e = d.openNextFile()) {
    Serial.printf("FILE,%s,%lu\n", e.fullName(), (unsigned long)e.size()); e.close(); yield();
  }
  d.close(); Serial.println("OK,LS");
}

void setup() {
  Serial.setRxBufferSize(2048);
  Serial.begin(SERIAL_BAUD);
  delay(200);
  WiFi.mode(WIFI_OFF);                                   // radio only for UPLOAD / WIFI
  sdOk = SD.begin(SD_CS_PIN, SD_SPI_HZ);
  Serial.printf("\nBOOT,%s,sd=%d\n", FW, sdOk);
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n'); line.trim();
  int sp = line.indexOf(' ');
  String cmd = sp < 0 ? line : line.substring(0, sp), arg = sp < 0 ? "" : line.substring(sp + 1);
  cmd.toUpperCase();
  if (cmd == "PING") Serial.println("PONG");
  else if (cmd == "INFO")
    Serial.printf("INFO,fw=%s,sd=%d,wifi=%d,ip=%s,source=%s\n", FW, sdOk, WiFi.status() == WL_CONNECTED,
                  WiFi.localIP().toString().c_str(), ECG_SOURCE_ADC ? "adc" : "sd");
  else if (!sdOk) { sdOk = SD.begin(SD_CS_PIN, SD_SPI_HZ); if (!sdOk) Serial.println("ERR,SD,not mounted"); }
  else if (cmd == "LS") list(arg);
  else if (cmd == "SHA") Serial.printf("SHA,%s,%s\n", arg.c_str(), sha256File(arg).c_str());
  else if (cmd == "RECORD") {
    int s2 = arg.indexOf(' ');
    record(s2 < 0 ? arg : arg.substring(0, s2), s2 < 0 ? SESSION_SECONDS : arg.substring(s2 + 1).toInt());
  }
  else if (cmd == "DUMP") dump(arg);
  else if (cmd == "WIFI") { if (wifiUp()) Serial.printf("OK,WIFI,%s\n", WiFi.localIP().toString().c_str()); else Serial.println("ERR,WIFI"); }
  else if (cmd == "UPLOAD") upload(arg);
  else if (cmd == "PUT") put(arg);
  else Serial.println("ERR,unknown," + cmd);
}
