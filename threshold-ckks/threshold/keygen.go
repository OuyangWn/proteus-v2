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

func AddSecretKeys(params ckks.Parameters, shares []*rlwe.SecretKey) *rlwe.SecretKey {
	total := rlwe.NewSecretKey(params)
	rp := params.GetRLWEParameters()

	for _, sk := range shares {
		rp.RingQ().Add(total.Value.Q, sk.Value.Q, total.Value.Q)
		rp.RingP().Add(total.Value.P, sk.Value.P, total.Value.P)
	}

	return total
}

func GenerateKeys(params ckks.Parameters, parties int) *ThresholdKeys {
	kgen := ckks.NewKeyGenerator(params)

	keys := &ThresholdKeys{
		PartySK: make([]*rlwe.SecretKey, parties),
	}

	// 每个 DCe_h 生成自己的 secret share
	for i := 0; i < parties; i++ {
		keys.PartySK[i] = rlwe.NewSecretKey(params)
		kgen.GenSecretKey(keys.PartySK[i])
	}

	// sk = Σ sk_h
	keys.TotalSK = AddSecretKeys(params, keys.PartySK)

	// 从聚合 secret key 生成 pk
	keys.PK = rlwe.NewPublicKey(params)
	kgen.GenPublicKey(keys.TotalSK, keys.PK)

	fmt.Println("Threshold key generation finished")

	return keys
}
