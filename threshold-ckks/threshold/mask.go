package threshold

import (
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// MultiplyByMask applies one constant to every SIMD slot and rescales the product.
func MultiplyByMask(params ckks.Parameters, ct *rlwe.Ciphertext, r *big.Int) *rlwe.Ciphertext {
	mask := make([]*big.Int, params.MaxSlots())
	for i := range mask {
		mask[i] = r
	}
	return MultiplyByMaskVector(params, ct, mask)
}

// MultiplyByMaskVector applies a PRF-derived mask vector slotwise and rescales.
func MultiplyByMaskVector(params ckks.Parameters, ct *rlwe.Ciphertext, masks []*big.Int) *rlwe.Ciphertext {
	if len(masks) > params.MaxSlots() {
		panic("mask vector exceeds CKKS slots")
	}

	encoder := ckks.NewEncoder(params)
	pt := ckks.NewPlaintext(params, ct.Level())
	pt.Scale = ct.Scale

	mask := make([]complex128, params.MaxSlots())
	for i, r := range masks {
		r64, _ := new(big.Float).SetInt(r).Float64()
		mask[i] = complex(r64, 0)
	}
	if err := encoder.Encode(mask, pt); err != nil {
		panic(err)
	}

	evaluator := ckks.NewEvaluator(params, nil)
	out, err := evaluator.MulNew(ct, pt)
	if err != nil {
		panic(err)
	}
	if err = evaluator.Rescale(out, out); err != nil {
		panic(err)
	}
	return out
}
