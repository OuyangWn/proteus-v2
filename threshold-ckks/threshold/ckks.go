package threshold

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// EncryptVector encodes a batched vector and encrypts it with the public key.
func EncryptVector(params ckks.Parameters, pk *rlwe.PublicKey, values []float64) *rlwe.Ciphertext {
	return EncryptVectorAtLevelScale(params, pk, values, params.MaxLevel(), params.DefaultScale())
}

// EncryptVectorAtLevelScale encrypts a vector while matching an existing ciphertext level/scale.
func EncryptVectorAtLevelScale(params ckks.Parameters, pk *rlwe.PublicKey, values []float64, level int, scale rlwe.Scale) *rlwe.Ciphertext {
	encoder := ckks.NewEncoder(params)
	pt := ckks.NewPlaintext(params, level)
	pt.Scale = scale
	if err := encoder.Encode(values, pt); err != nil {
		panic(err)
	}
	ct, err := ckks.NewEncryptor(params, pk).EncryptNew(pt)
	if err != nil {
		panic(err)
	}
	return ct
}

// AddCiphertexts computes Enc(x)+Enc(y).
func AddCiphertexts(params ckks.Parameters, a, b *rlwe.Ciphertext) *rlwe.Ciphertext {
	ct, err := ckks.NewEvaluator(params, nil).AddNew(a, b)
	if err != nil {
		panic(err)
	}
	return ct
}

// SubCiphertexts computes Enc(x)-Enc(y).
func SubCiphertexts(params ckks.Parameters, a, b *rlwe.Ciphertext) *rlwe.Ciphertext {
	ct, err := ckks.NewEvaluator(params, nil).SubNew(a, b)
	if err != nil {
		panic(err)
	}
	return ct
}
