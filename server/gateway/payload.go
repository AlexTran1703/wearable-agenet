package main

// Waveform representations accepted by POST /v1/records (X-Payload-Encoding):
//   bin       raw int16 little-endian, interleaved leads (the device format; default)
//   bin+gzip  gzip-compressed bin
//   csv       one sample per line, leads comma-separated, integer microvolts
//   json      {"samples": [[ii, i], ...]} integer microvolts
// All are decoded to the canonical int16 LE interleaved array that is analysed and stored.

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

type Format struct {
	FS        int
	Leads     []string
	UVPerUnit float64 // microvolts per (offset-corrected) count
	ADCBits   int     // 0 = signed int16 microvolts; 12 = unsigned 12-bit ADC codes
	Offset    int     // ADC mid-scale bias subtracted before scaling
}

var ErrFormat = errors.New("malformed_metadata")

// ParseFormat reads X-Sample-Format. Two forms are accepted:
//
//	int16le;uV;fs=200;leads=II,I                                 signed microvolts (1 count = 1 uV)
//	adc12le;fs=200;leads=II;vref_mV=3300;gain=1100;offset=2048   AD8232 + 12-bit ADC codes (uint16 words)
func ParseFormat(s string) (Format, error) {
	f := Format{UVPerUnit: 1}
	parts := strings.Split(s, ";")
	var kv []string
	switch {
	case len(parts) >= 4 && parts[0] == "int16le" && parts[1] == "uV":
		kv = parts[2:]
	case len(parts) >= 3 && parts[0] == "adc12le":
		f.ADCBits, kv = 12, parts[1:]
	default:
		return f, ErrFormat
	}
	vref, gain := 0.0, 0.0
	f.Offset = -1
	for _, p := range kv {
		kv := strings.SplitN(p, "=", 2)
		if len(kv) != 2 {
			return f, ErrFormat
		}
		switch kv[0] {
		case "fs":
			v, err := strconv.Atoi(kv[1])
			if err != nil || v < 100 || v > 2000 {
				return f, ErrFormat
			}
			f.FS = v
		case "leads":
			f.Leads = strings.Split(kv[1], ",")
		case "vref_mV":
			vref, _ = strconv.ParseFloat(kv[1], 64)
		case "gain":
			gain, _ = strconv.ParseFloat(kv[1], 64)
		case "offset":
			f.Offset, _ = strconv.Atoi(kv[1])
		}
	}
	if f.FS == 0 || len(f.Leads) == 0 || len(f.Leads) > 12 || f.Leads[0] != "II" {
		return f, ErrFormat // the model input is Lead II, expected first
	}
	if f.ADCBits == 12 {
		if vref <= 0 || gain <= 0 || f.Offset < 0 || f.Offset > 4095 {
			return f, ErrFormat
		}
		f.UVPerUnit = vref * 1000 / 4096 / gain
	}
	return f, nil
}

// FromADC converts 12-bit ADC codes (uint16 LE words) to offset-corrected signed int16 counts.
func FromADC(body []byte, offset int) ([]byte, error) {
	out := make([]byte, len(body))
	for i := 0; i+1 < len(body); i += 2 {
		c := int(binary.LittleEndian.Uint16(body[i:]))
		if c > 4095 {
			return nil, fmt.Errorf("%w: ADC code above 12 bits", ErrFormat)
		}
		binary.LittleEndian.PutUint16(out[i:], uint16(int16(c-offset)))
	}
	return out, nil
}

func clamp16(v int64) (int16, error) {
	if v < -32768 || v > 32767 {
		return 0, fmt.Errorf("%w: sample out of int16 range", ErrFormat)
	}
	return int16(v), nil
}

// Decode converts the transmitted payload into canonical int16 LE bytes.
func Decode(enc string, body []byte, nLeads int) ([]byte, error) {
	switch enc {
	case "", "bin":
		if len(body) == 0 || len(body)%(2*nLeads) != 0 {
			return nil, fmt.Errorf("%w: length not a multiple of frame size", ErrFormat)
		}
		return body, nil
	case "bin+gzip":
		r, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("%w: gzip", ErrFormat)
		}
		out, err := io.ReadAll(io.LimitReader(r, 64<<20))
		if err != nil {
			return nil, fmt.Errorf("%w: gzip", ErrFormat)
		}
		return Decode("bin", out, nLeads)
	case "csv":
		out := make([]byte, 0, len(body)/2)
		sc := bufio.NewScanner(bytes.NewReader(body))
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" {
				continue
			}
			cols := strings.Split(line, ",")
			if len(cols) != nLeads {
				return nil, fmt.Errorf("%w: csv columns", ErrFormat)
			}
			for _, c := range cols {
				v, err := strconv.ParseInt(c, 10, 64)
				if err != nil {
					return nil, fmt.Errorf("%w: csv value", ErrFormat)
				}
				s, err := clamp16(v)
				if err != nil {
					return nil, err
				}
				out = binary.LittleEndian.AppendUint16(out, uint16(s))
			}
		}
		return Decode("bin", out, nLeads)
	case "json":
		var j struct {
			Samples [][]int64 `json:"samples"`
		}
		if err := json.Unmarshal(body, &j); err != nil {
			return nil, fmt.Errorf("%w: json", ErrFormat)
		}
		out := make([]byte, 0, 2*nLeads*len(j.Samples))
		for _, fr := range j.Samples {
			if len(fr) != nLeads {
				return nil, fmt.Errorf("%w: json frame", ErrFormat)
			}
			for _, v := range fr {
				s, err := clamp16(v)
				if err != nil {
					return nil, err
				}
				out = binary.LittleEndian.AppendUint16(out, uint16(s))
			}
		}
		return Decode("bin", out, nLeads)
	}
	return nil, fmt.Errorf("%w: unknown encoding %q", ErrFormat, enc)
}
