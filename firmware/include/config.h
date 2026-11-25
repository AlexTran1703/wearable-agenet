// Hardware and protocol configuration of the ESP-12F ECG node.
#pragma once

// microSD on the hardware SPI bus: D5 = GPIO14 SCLK, D6 = GPIO12 MISO, D7 = GPIO13 MOSI, D8 = GPIO15 CS.
// GPIO15 must be LOW at boot: a CS pull-up on the SD module prevents booting.
#define SD_CS_PIN        15
#define SD_SPI_HZ        20000000

#define SERIAL_BAUD      460800
#define FS_HZ            200          // acquisition rate (samples per second per lead)
#define N_LEADS          1            // single lead II
// AD8232 analogue front end -> 12-bit ADC: codes 0..4095 in uint16 words, mid-scale bias 2048,
// 1 code = 3.3 V / 4096 / 1100 = 0.7324 uV at the electrodes
#define ADC_BITS         12
#define ADC_VREF_MV      3300
#define AFE_GAIN         1100
#define ADC_OFFSET       2048
#define SAMPLE_FORMAT    "adc12le;fs=200;leads=II;vref_mV=3300;gain=1100;offset=2048"
#define SESSION_SECONDS  300          // 5-min acquisition unit
#define CHUNK_SAMPLES    200          // samples buffered before each SD write (1 s)

// ECG source: 0 = microSD replay of a stored session (mimics acquisition), 1 = on-board ADC A0 
#define ECG_SOURCE_ADC   0
