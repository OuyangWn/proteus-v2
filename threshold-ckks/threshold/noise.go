package threshold

import (
	crand "crypto/rand"
	"encoding/binary"
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// AddFloodingNoise adds demo slot-domain Gaussian noise to a ciphertext.
func AddFloodingNoise(params ckks.Parameters, ct *rlwe.Ciphertext, length int, sigma float64) *rlwe.Ciphertext {
	if length <= 0 || length > params.MaxSlots() {
		panic("invalid flooding noise length")
	}
	if sigma == 0 {
		return ct.CopyNew()
	}

	noise := make([]float64, length)
	for i := range noise {
		noise[i] = sigma * sampleStandardNormal()
	}

	out, err := ckks.NewEvaluator(params, nil).AddNew(ct, noise)
	if err != nil {
		panic(err)
	}
	return out
}

// SampleGaussian draws demo Gaussian noise for partial decrypt shares.
func SampleGaussian(ringQ *ring.Ring, sigma float64) ring.Poly {
	prng, _ := sampling.NewPRNG()
	sampler, _ := ring.NewSampler(prng, ringQ, ring.DiscreteGaussian{Sigma: sigma, Bound: 6 * sigma}, false)
	e := ringQ.NewPoly()
	sampler.Read(e)
	return e
}

func sampleStandardNormal() float64 {
	u1, u2 := sampleUnitFloat64(), sampleUnitFloat64()
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

func sampleUnitFloat64() float64 {
	var b [8]byte
	if _, err := crand.Read(b[:]); err != nil {
		panic(err)
	}
	return (float64(binary.BigEndian.Uint64(b[:])>>11) + 0.5) / 9007199254740992.0
}
