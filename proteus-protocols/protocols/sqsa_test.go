package protocols

import (
	"testing"

	"threshold-ckks/threshold"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestSQSASelectsEncryptedStateActionValuePerSlot(t *testing.T) {
	params, err := ckks.NewParametersFromLiteral(ckks.ExampleParameters128BitLogN14LogQP438)
	if err != nil {
		t.Fatal(err)
	}

	keys := threshold.GenerateKeys(params, 3)
	rlk := ckks.NewKeyGenerator(params).GenRelinearizationKeyNew(keys.TotalSK)
	ctx := NewHomContext(params, rlwe.NewMemEvaluationKeySet(rlk))

	// slot0 selects (state 1, action 1), slot1 selects (state 2, action 0).
	state := []*rlwe.Ciphertext{
		threshold.EncryptVector(params, keys.PK, []float64{0, 0}),
		threshold.EncryptVector(params, keys.PK, []float64{1, 0}),
		threshold.EncryptVector(params, keys.PK, []float64{0, 1}),
	}
	action := []*rlwe.Ciphertext{
		threshold.EncryptVector(params, keys.PK, []float64{0, 1}),
		threshold.EncryptVector(params, keys.PK, []float64{1, 0}),
	}
	qTable := [][]*rlwe.Ciphertext{
		{threshold.EncryptVector(params, keys.PK, []float64{5, 9}), threshold.EncryptVector(params, keys.PK, []float64{2, 0})},
		{threshold.EncryptVector(params, keys.PK, []float64{1, 4}), threshold.EncryptVector(params, keys.PK, []float64{8, 7})},
		{threshold.EncryptVector(params, keys.PK, []float64{3, 2}), threshold.EncryptVector(params, keys.PK, []float64{6, 5})},
	}

	result := SQSAWithStats(ctx, state, action, qTable)
	got := result.Value
	assertCloseVector(t, decryptVector(params, keys.TotalSK, got, 2), []float64{8, 2})
	if got.Degree() != 1 {
		t.Fatalf("SQSA output must be relinearized to degree 1, got %d", got.Degree())
	}
	assertLocalStats(t, result.Stats)
}
