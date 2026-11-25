// Copy to secrets.h (git-ignored) and fill in. Never commit secrets.h.
#pragma once
#define WIFI_SSID      "<wifi-ssid>"
#define WIFI_PASSWORD  "<wifi-password>"
#define DEVICE_ID      "wearable-01"
#define DEVICE_KEY_HEX "<same 64-hex key as server/config/device_key.env>"
#define GATEWAY_URL    "https://<server-host>/v1/records"
// SHA-1 fingerprint of the gateway TLS certificate (pinned), e.g. openssl x509 -in tls.crt -noout -fingerprint -sha1
#define GATEWAY_FP     "<AA:BB:...>"
