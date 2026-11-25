package main

// Request authentication and integrity (functional security layer, not a new protocol):
//   X-Payload-SHA256 = SHA-256(body)                         integrity of the transmitted bytes
//   X-Signature      = HMAC-SHA256(K_device, id|ts|nonce|sha) authenticity, bound to the payload hash
//   X-Timestamp      within +-MaxSkew of server time            freshness
//   X-Nonce          128-bit, accepted once within NonceTTL     replay protection

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"regexp"
	"strconv"
	"sync"
	"time"
)

var (
	ErrUnknownDevice = errors.New("unknown_device")
	ErrStale         = errors.New("stale_timestamp")
	ErrBadNonce      = errors.New("malformed_nonce")
	ErrReplay        = errors.New("replayed_nonce")
	ErrSignature     = errors.New("bad_signature")
	ErrHash          = errors.New("payload_hash_mismatch")
	ErrHeaders       = errors.New("missing_security_headers")
)

var hex32 = regexp.MustCompile(`^[0-9a-f]{32}$`)
var hex64 = regexp.MustCompile(`^[0-9a-f]{64}$`)

type Verifier struct {
	keys     map[string][]byte // device id -> HMAC key (never logged)
	MaxSkew  time.Duration
	NonceTTL time.Duration
	mu       sync.Mutex
	seen     map[string]time.Time
}

func NewVerifier(keys map[string][]byte) *Verifier {
	return &Verifier{keys: keys, MaxSkew: 5 * time.Minute, NonceTTL: 15 * time.Minute, seen: map[string]time.Time{}}
}

type SecInfo struct {
	DeviceID string
	DeviceTS int64
	Nonce    string
	SHA256   string
}

// Verify checks headers and body; the nonce is only consumed when every other check passed.
func (v *Verifier) Verify(h http.Header, body []byte, now time.Time) (SecInfo, error) {
	id, ts, nonce, sha, sig := h.Get("X-Device-ID"), h.Get("X-Timestamp"), h.Get("X-Nonce"), h.Get("X-Payload-SHA256"), h.Get("X-Signature")
	info := SecInfo{DeviceID: id, Nonce: nonce, SHA256: sha}
	if id == "" || ts == "" || nonce == "" || sha == "" || sig == "" {
		return info, ErrHeaders
	}
	key, ok := v.keys[id]
	if !ok {
		return info, ErrUnknownDevice
	}
	t, err := strconv.ParseInt(ts, 10, 64)
	if err != nil {
		return info, ErrStale
	}
	info.DeviceTS = t
	if d := now.Sub(time.Unix(t, 0)); d > v.MaxSkew || d < -v.MaxSkew {
		return info, ErrStale
	}
	if !hex32.MatchString(nonce) || !hex64.MatchString(sha) {
		return info, ErrBadNonce
	}
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(id + "|" + ts + "|" + nonce + "|" + sha))
	want := mac.Sum(nil)
	got, err := hex.DecodeString(sig)
	if err != nil || !hmac.Equal(want, got) {
		return info, ErrSignature
	}
	d := sha256.Sum256(body)
	if hex.EncodeToString(d[:]) != sha {
		return info, ErrHash
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	for n, at := range v.seen { // small map: prune expired nonces
		if now.Sub(at) > v.NonceTTL {
			delete(v.seen, n)
		}
	}
	if _, dup := v.seen[id+"|"+nonce]; dup {
		return info, ErrReplay
	}
	v.seen[id+"|"+nonce] = now
	return info, nil
}
