package threshold

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math/big"
)

// PRF is HMAC-SHA256 keyed by the mask secret.
func PRF(key []byte, id string) []byte {
	h := hmac.New(sha256.New, key)
	h.Write([]byte(id))
	return h.Sum(nil)
}

// DeriveMaskExponent returns r = 2^a + PRF(key,id) mod 2^Br.
func DeriveMaskExponent(key []byte, id string, a uint, Br uint) *big.Int {
	x := new(big.Int).SetBytes(PRF(key, id))
	x.Mod(x, new(big.Int).Lsh(big.NewInt(1), Br))
	return new(big.Int).Add(new(big.Int).Lsh(big.NewInt(1), a), x)
}

// GeneratePRFMask derives one positive mask per SIMD slot.
func GeneratePRFMask(key []byte, idPrefix string, length int, a uint, Br uint) []*big.Int {
	mask := make([]*big.Int, length)
	for i := range mask {
		mask[i] = DeriveMaskExponent(key, fmt.Sprintf("%s:%d", idPrefix, i), a, Br)
	}
	return mask
}

// SelectReconstructor deterministically elects D_k* from public seed and call id.
func SelectReconstructor(publicSeed []byte, callID string, parties int) int {
	if parties <= 0 {
		panic("party count must be positive")
	}
	x := new(big.Int).SetBytes(PRF(publicSeed, callID))
	return int(new(big.Int).Mod(x, big.NewInt(int64(parties))).Int64())
}

// Uint64Prefix is a quick debug helper.
func Uint64Prefix(x *big.Int) uint64 {
	b := x.Bytes()
	if len(b) < 8 {
		return binary.BigEndian.Uint64(append(make([]byte, 8-len(b)), b...))
	}
	return binary.BigEndian.Uint64(b[:8])
}
