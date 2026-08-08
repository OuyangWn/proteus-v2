package protocols

import (
	"fmt"

	"threshold-ckks/threshold"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// SelectByEncryptedOneHot computes sum_i Enc(bit_i)*Enc(value_i).
func SelectByEncryptedOneHot(ctx HomContext, selector []*rlwe.Ciphertext, values []*rlwe.Ciphertext) *rlwe.Ciphertext {
	validateEncryptedSelector(ctx, selector, values)

	var acc *rlwe.Ciphertext
	for i := range selector {
		term, err := ctx.Evaluator.MulRelinNew(selector[i], values[i])
		if err != nil {
			panic(fmt.Errorf("encrypted selector multiply index %d: %w", i, err))
		}
		if err = ctx.Evaluator.Rescale(term, term); err != nil {
			panic(fmt.Errorf("encrypted selector rescale index %d: %w", i, err))
		}
		if acc == nil {
			acc = term
			continue
		}
		if err = ctx.Evaluator.Add(acc, term, acc); err != nil {
			panic(fmt.Errorf("encrypted selector accumulate index %d: %w", i, err))
		}
	}
	return acc
}

// SelectByEncryptedBit computes Enc(bit)*Enc(left) + (1-Enc(bit))*Enc(right).
func SelectByEncryptedBit(ctx HomContext, pk *rlwe.PublicKey, bit, left, right *rlwe.Ciphertext, length int) *rlwe.Ciphertext {
	validateBinarySelectInputs(ctx, pk, bit, left, right, length)

	notBit, err := ctx.Evaluator.SubNew(encryptRepeated(ctx, pk, 1, length, bit), bit)
	if err != nil {
		panic(fmt.Errorf("encrypted bit complement: %w", err))
	}

	leftTerm, err := ctx.Evaluator.MulRelinNew(bit, left)
	if err != nil {
		panic(fmt.Errorf("encrypted binary select left multiply: %w", err))
	}
	if err = ctx.Evaluator.Rescale(leftTerm, leftTerm); err != nil {
		panic(fmt.Errorf("encrypted binary select left rescale: %w", err))
	}

	rightTerm, err := ctx.Evaluator.MulRelinNew(notBit, right)
	if err != nil {
		panic(fmt.Errorf("encrypted binary select right multiply: %w", err))
	}
	if err = ctx.Evaluator.Rescale(rightTerm, rightTerm); err != nil {
		panic(fmt.Errorf("encrypted binary select right rescale: %w", err))
	}
	if err = ctx.Evaluator.Add(leftTerm, rightTerm, leftTerm); err != nil {
		panic(fmt.Errorf("encrypted binary select add: %w", err))
	}
	return leftTerm
}

func encryptRepeated(ctx HomContext, pk *rlwe.PublicKey, value float64, length int, template *rlwe.Ciphertext) *rlwe.Ciphertext {
	values := make([]float64, length)
	for i := range values {
		values[i] = value
	}
	return threshold.EncryptVectorAtLevelScale(ctx.Params, pk, values, template.Level(), template.Scale)
}

func validateBinarySelectInputs(ctx HomContext, pk *rlwe.PublicKey, bit, left, right *rlwe.Ciphertext, length int) {
	if ctx.Evaluator == nil {
		panic("homomorphic evaluator must not be nil")
	}
	if pk == nil || bit == nil || left == nil || right == nil {
		panic("nil binary select key or ciphertext")
	}
	if length <= 0 || length > ctx.Params.MaxSlots() {
		panic("invalid binary select length")
	}
}

func validateEncryptedSelector(ctx HomContext, selector []*rlwe.Ciphertext, values []*rlwe.Ciphertext) {
	if ctx.Evaluator == nil {
		panic("homomorphic evaluator must not be nil")
	}
	if len(selector) == 0 || len(selector) != len(values) {
		panic("selector and value dimensions mismatch")
	}
	for i := range selector {
		if selector[i] == nil || values[i] == nil {
			panic("nil selector or value ciphertext")
		}
	}
}
