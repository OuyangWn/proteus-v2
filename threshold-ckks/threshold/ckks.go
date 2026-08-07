package threshold

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// EncryptVector encrypts a float vector with CKKS public key.
func EncryptVector(params ckks.Parameters, pk *rlwe.PublicKey, values []float64) *rlwe.Ciphertext {
	encoder := ckks.NewEncoder(params)

	pt := ckks.NewPlaintext(params, params.MaxLevel())
	pt.Scale = params.DefaultScale()

	if err := encoder.Encode(values, pt); err != nil {
		panic(err)
	}

	encryptor := ckks.NewEncryptor(params, pk)

	ct, err := encryptor.EncryptNew(pt)
	if err != nil {
		panic(err)
	}

	return ct
}

// SubCiphertexts computes Enc(x)-Enc(y).
func SubCiphertexts(params ckks.Parameters, a, b *rlwe.Ciphertext) *rlwe.Ciphertext {
	evaluator := ckks.NewEvaluator(params, nil)

	ct, err := evaluator.SubNew(a, b)
	if err != nil {
		panic(err)
	}

	return ct
}
