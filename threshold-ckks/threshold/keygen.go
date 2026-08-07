package threshold

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type ThresholdKeys struct {
	PartySK []*rlwe.SecretKey
	TotalSK *rlwe.SecretKey
	PK      *rlwe.PublicKey
}

// AddSecretKeys sums party shares into the aggregate secret key.
func AddSecretKeys(params ckks.Parameters, shares []*rlwe.SecretKey) *rlwe.SecretKey {
	total, rp := rlwe.NewSecretKey(params), params.GetRLWEParameters()
	for _, sk := range shares {
		rp.RingQ().Add(total.Value.Q, sk.Value.Q, total.Value.Q)
		rp.RingP().Add(total.Value.P, sk.Value.P, total.Value.P)
	}
	return total
}

// GenerateKeys creates secret shares, their aggregate key, and the public key.
func GenerateKeys(params ckks.Parameters, parties int) *ThresholdKeys {
	kgen := ckks.NewKeyGenerator(params)
	keys := &ThresholdKeys{PartySK: make([]*rlwe.SecretKey, parties)}
	for i := 0; i < parties; i++ {
		keys.PartySK[i] = rlwe.NewSecretKey(params)
		kgen.GenSecretKey(keys.PartySK[i])
	}
	keys.TotalSK = AddSecretKeys(params, keys.PartySK)
	keys.PK = rlwe.NewPublicKey(params)
	kgen.GenPublicKey(keys.TotalSK, keys.PK)
	fmt.Println("Threshold key generation finished")
	return keys
}
