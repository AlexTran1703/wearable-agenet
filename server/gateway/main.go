// AGENet telemedicine gateway (the "EC2" service; runs on the local machine during the LAN phase).
//
//	POST /v1/records              authenticated 5-min ECG upload -> verify -> decode -> AGENet -> MongoDB
//	GET  /v1/records/{id}         record metadata (no waveform)
//	GET  /v1/records/{id}/result  AGENet analysis
//	GET  /health                  service, MongoDB and inference-worker status
//
// Secrets are read from git-ignored files and never logged: device keys (KEY=VALUE env file with DEVICE_ID and
// DEVICE_KEY_HEX) and the MongoDB connection string (single-line file).
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"golang.org/x/crypto/acme/autocert"
)

const maxBody = 8 << 20

type Server struct {
	ver    *Verifier
	store  *Store // nil -> analysis only (-no-db)
	worker string
	model  string
	http   *http.Client
}

func readEnv(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out := map[string]string{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		l := strings.TrimSpace(sc.Text())
		if l == "" || strings.HasPrefix(l, "#") {
			continue
		}
		if kv := strings.SplitN(l, "=", 2); len(kv) == 2 {
			out[strings.TrimSpace(kv[0])] = strings.Trim(strings.TrimSpace(kv[1]), "\"")
		}
	}
	return out, sc.Err()
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func newID() string {
	b := make([]byte, 6)
	_, _ = rand.Read(b)
	return time.Now().UTC().Format("20060102T150405Z") + "-" + hex.EncodeToString(b)
}

func secStatus(err error) int {
	switch {
	case errors.Is(err, ErrReplay):
		return http.StatusConflict
	case errors.Is(err, ErrHash), errors.Is(err, ErrBadNonce), errors.Is(err, ErrHeaders):
		return http.StatusBadRequest
	default:
		return http.StatusUnauthorized
	}
}

func (s *Server) postRecord(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBody+1))
	if err != nil {
		log.Printf("receive failed from %s after %d bytes in %.1fs (Content-Length %d): %v", r.RemoteAddr, len(body),
			time.Since(t0).Seconds(), r.ContentLength, err)
	}
	if err != nil || len(body) > maxBody {
		writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "payload_too_large"})
		return
	}
	tRecv := time.Now()
	sec, err := s.ver.Verify(r.Header, body, time.Now())
	if err != nil {
		log.Printf("reject device=%q reason=%s", sec.DeviceID, err)
		writeJSON(w, secStatus(err), map[string]string{"error": err.Error()})
		return
	}
	tSec := time.Now()
	fmtHdr := r.Header.Get("X-Sample-Format")
	f, err := ParseFormat(fmtHdr)
	if err != nil {
		log.Printf("reject device=%q reason=%s", sec.DeviceID, err)
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	enc := r.Header.Get("X-Payload-Encoding")
	if enc == "" {
		enc = "bin"
	}
	wave, err := Decode(enc, body, len(f.Leads))
	if err == nil && f.ADCBits == 12 {
		wave, err = FromADC(wave, f.Offset)
	}
	if err != nil {
		log.Printf("reject device=%q reason=%v", sec.DeviceID, err)
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": ErrFormat.Error(), "detail": err.Error()})
		return
	}
	tDec := time.Now()
	url := fmt.Sprintf("%s/infer?fs=%d&leads=%d&lead=0&uv=%g", s.worker, f.FS, len(f.Leads), f.UVPerUnit)
	resp, err := s.http.Post(url, "application/octet-stream", bytes.NewReader(wave))
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "inference_unavailable"})
		return
	}
	var analysis map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&analysis)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		writeJSON(w, http.StatusUnprocessableEntity, map[string]any{"error": "analysis_failed", "detail": analysis["error"]})
		return
	}
	tInf := time.Now()
	id := newID()
	nSamples := len(wave) / (2 * len(f.Leads))
	timing := bson.M{"receive_ms": ms(tRecv.Sub(t0)), "security_ms": ms(tSec.Sub(tRecv)), "decode_ms": ms(tDec.Sub(tSec)),
		"analysis_total_ms": ms(tInf.Sub(tDec)), "worker": analysis["timing"]}
	counts := analysis["counts"]
	if s.store != nil {
		doc := bson.M{
			"_id": id, "device_id": sec.DeviceID, "device_timestamp": sec.DeviceTS, "nonce": sec.Nonce,
			"received_at": time.Now().UTC(), "sha256_payload": sec.SHA256, "encoding": enc, "payload_bytes": len(body),
			"waveform_bytes": len(wave), "fs": f.FS, "leads": f.Leads, "uV_per_count": f.UVPerUnit, "adc_bits": f.ADCBits, "adc_offset": f.Offset, "n_samples": nSamples,
			"duration_s": float64(nSamples) / float64(f.FS), "sample_format": fmtHdr,
			"firmware": r.Header.Get("X-Firmware"), "model_version": s.model,
			"waveform": bson.Binary{Subtype: 0x00, Data: wave}, "analysis": analysis, "timing": timing,
		}
		ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
		err = s.store.Insert(ctx, doc)
		cancel()
		if mongo.IsDuplicateKeyError(err) {
			writeJSON(w, http.StatusConflict, map[string]string{"error": ErrReplay.Error()})
			return
		}
		if err != nil {
			log.Printf("db insert failed: %v", err)
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "storage_failed"})
			return
		}
	}
	tDB := time.Now()
	timing["db_ms"] = ms(tDB.Sub(tInf))
	timing["server_total_ms"] = ms(tDB.Sub(t0))
	if s.store != nil { // complete the stored timing after the response path has been measured
		go func(set bson.M) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			_ = s.store.SetFields(ctx, id, set)
		}(bson.M{"timing.db_ms": timing["db_ms"], "timing.server_total_ms": timing["server_total_ms"]})
	}
	log.Printf("accept id=%s device=%s enc=%s bytes=%d beats=%v counts=%v total=%.0fms", id, sec.DeviceID, enc, len(body),
		analysis["n_classified"], counts, timing["server_total_ms"])
	writeJSON(w, http.StatusCreated, map[string]any{"id": id, "n_classified": analysis["n_classified"], "counts": counts, "timing": timing})
}

func (s *Server) getRecord(result bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.store == nil {
			writeJSON(w, http.StatusNotImplemented, map[string]string{"error": "no_database"})
			return
		}
		proj := bson.M{"waveform": 0, "analysis": 0}
		if result {
			proj = bson.M{"analysis": 1, "model_version": 1, "device_id": 1, "received_at": 1}
		}
		doc, err := s.store.Get(r.Context(), r.PathValue("id"), proj)
		if errors.Is(err, mongo.ErrNoDocuments) {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "not_found"})
			return
		} else if err != nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "storage_failed"})
			return
		}
		if !result {
			if n, err := s.store.StorageBytes(r.Context(), r.PathValue("id")); err == nil {
				doc["stored_bson_bytes"] = n
			}
		}
		writeJSON(w, http.StatusOK, doc)
	}
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	out := map[string]any{"status": "ok", "model": s.model}
	if resp, err := s.http.Get(s.worker + "/health"); err == nil {
		var h map[string]any
		_ = json.NewDecoder(resp.Body).Decode(&h)
		resp.Body.Close()
		out["worker"] = h
	} else {
		out["worker"], out["status"] = "down", "degraded"
	}
	if s.store != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		if err := s.store.Ping(ctx); err != nil {
			out["mongodb"], out["status"] = "down", "degraded"
		} else {
			out["mongodb"] = "ok"
		}
	}
	writeJSON(w, http.StatusOK, out)
}

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	keys := flag.String("devices", "../../secrets/device_key.env", "device key env file (DEVICE_ID, DEVICE_KEY_HEX)")
	mongoFile := flag.String("mongo-uri-file", "../../mongodb", "file holding the MongoDB connection string")
	db := flag.String("db", "agenet_telemed", "MongoDB database")
	noDB := flag.Bool("no-db", false, "run without MongoDB (analysis only)")
	worker := flag.String("worker", "http://127.0.0.1:8090", "inference worker URL (localhost)")
	model := flag.String("model", "AGENet-B3 nv_B3_wce_s1337 (ONNX)", "model version stored with each record")
	certFile := flag.String("tls-cert", "", "TLS certificate file (enables HTTPS)")
	keyFile := flag.String("tls-key", "", "TLS private key file")
	acmeDomain := flag.String("acme-domain", "", "DNS name for an automatic Let's Encrypt certificate")
	acmeCache := flag.String("acme-cache", "/var/lib/agenet/acme", "certificate cache directory")
	flag.Parse()

	env, err := readEnv(*keys)
	if err != nil {
		log.Fatalf("device keys: %v", err)
	}
	k, err := hex.DecodeString(env["DEVICE_KEY_HEX"])
	if err != nil || env["DEVICE_ID"] == "" || len(k) < 16 {
		log.Fatal("device keys: DEVICE_ID / DEVICE_KEY_HEX missing or invalid")
	}
	s := &Server{ver: NewVerifier(map[string][]byte{env["DEVICE_ID"]: k}), worker: *worker, model: *model,
		http: &http.Client{Timeout: 60 * time.Second}}
	if !*noDB {
		raw, err := os.ReadFile(*mongoFile)
		if err != nil {
			log.Fatal("mongodb uri file: cannot read")
		}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		s.store, err = OpenStore(ctx, strings.TrimSpace(string(raw)), *db)
		cancel()
		if err != nil {
			log.Fatal("mongodb: connection failed (check network / IP access list)") // driver errors may echo the URI
		}
		log.Printf("mongodb connected (db %s)", *db)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/records", s.postRecord)
	mux.HandleFunc("GET /v1/records/{id}", s.getRecord(false))
	mux.HandleFunc("GET /v1/records/{id}/result", s.getRecord(true))
	mux.HandleFunc("GET /health", s.health)
	srv := &http.Server{Addr: *addr, Handler: mux, ReadTimeout: 120 * time.Second, WriteTimeout: 120 * time.Second,
		TLSConfig: &tls.Config{MinVersion: tls.VersionTLS12}}
	switch {
	case *acmeDomain != "": // automatic Let's Encrypt certificate (needs a DNS name pointing at this host, port 80 open)
		m := &autocert.Manager{Prompt: autocert.AcceptTOS, HostPolicy: autocert.HostWhitelist(*acmeDomain),
			Cache: autocert.DirCache(*acmeCache)}
		srv.TLSConfig.GetCertificate = m.GetCertificate
		go func() { log.Fatal(http.ListenAndServe(":80", m.HTTPHandler(nil))) }()
		log.Printf("gateway listening on %s (HTTPS, Let's Encrypt for %s)", *addr, *acmeDomain)
		log.Fatal(srv.ListenAndServeTLS("", ""))
	case *certFile != "":
		log.Printf("gateway listening on %s (HTTPS, certificate %s)", *addr, *certFile)
		log.Fatal(srv.ListenAndServeTLS(*certFile, *keyFile))
	default:
		log.Printf("gateway listening on %s (plain HTTP; use only behind TLS)", *addr)
		log.Fatal(srv.ListenAndServe())
	}
}
