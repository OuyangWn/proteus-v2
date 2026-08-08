package protocols

import (
	"testing"

	"threshold-ckks/threshold"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestSPASAMixesArgmaxAndRandomActionsPerSlot(t *testing.T) {
	params, err := ckks.NewParametersFromLiteral(ckks.ExampleParameters128BitLogN14LogQP438)
	if err != nil {
		t.Fatal(err)
	}

	keys := threshold.GenerateKeys(params, 3)
	rlk := ckks.NewKeyGenerator(params).GenRelinearizationKeyNew(keys.TotalSK)
	ctx := NewHomContext(params, rlwe.NewMemEvaluationKeySet(rlk))

	// slot0 exploits state 1 -> argmax action 1; slot1 explores -> random action 0.
	state := []*rlwe.Ciphertext{
		threshold.EncryptVector(params, keys.PK, []float64{0, 0}),
		threshold.EncryptVector(params, keys.PK, []float64{1, 0}),
		threshold.EncryptVector(params, keys.PK, []float64{0, 1}),
	}
	qTable := [][]*rlwe.Ciphertext{
		{threshold.EncryptVector(params, keys.PK, []float64{5, 9}), threshold.EncryptVector(params, keys.PK, []float64{2, 0})},
		{threshold.EncryptVector(params, keys.PK, []float64{1, 4}), threshold.EncryptVector(params, keys.PK, []float64{8, 7})},
		{threshold.EncryptVector(params, keys.PK, []float64{3, 2}), threshold.EncryptVector(params, keys.PK, []float64{6, 5})},
	}
	exploitBit := threshold.EncryptVector(params, keys.PK, []float64{1, 0})
	randomAction := []*rlwe.Ciphertext{
		threshold.EncryptVector(params, keys.PK, []float64{1, 1}),
		threshold.EncryptVector(params, keys.PK, []float64{0, 0}),
	}
	opts := DefaultSPASAOptions(2, []byte("spasa-mask-key"), []byte("spasa-public-seed"), "spasa-test")

	result := SPASA(ctx, keys, state, qTable, exploitBit, randomAction, opts)

	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Row[0], 2), []float64{1, 2})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Row[1], 2), []float64{8, 5})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.SMAA.Max, 2), []float64{8, 5})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Action[0], 2), []float64{0, 1})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Action[1], 2), []float64{1, 0})
	assertInteractiveStats(t, result.Stats, 1, 0)
}
