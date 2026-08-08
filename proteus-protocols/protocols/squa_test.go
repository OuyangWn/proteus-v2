package protocols

import (
	"testing"

	"threshold-ckks/threshold"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestSQUAUpdatesOnlySelectedStateActionSlots(t *testing.T) {
	params, err := ckks.NewParametersFromLiteral(ckks.ExampleParameters128BitLogN14LogQP438)
	if err != nil {
		t.Fatal(err)
	}

	keys := threshold.GenerateKeys(params, 3)
	rlk := ckks.NewKeyGenerator(params).GenRelinearizationKeyNew(keys.TotalSK)
	ctx := NewHomContext(params, rlwe.NewMemEvaluationKeySet(rlk))

	// slot0 updates (state 1, action 1) by +5; slot1 updates (state 2, action 0) by -2.
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
	alpha := threshold.EncryptVector(params, keys.PK, []float64{0.5, 0.5})
	delta := threshold.EncryptVector(params, keys.PK, []float64{10, -4})

	result := SQUAWithStats(ctx, qTable, delta, state, action, alpha)
	updated := result.QTable

	assertCloseVector(t, decryptVector(params, keys.TotalSK, updated[0][0], 2), []float64{5, 9})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, updated[0][1], 2), []float64{2, 0})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, updated[1][0], 2), []float64{1, 4})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, updated[1][1], 2), []float64{13, 7})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, updated[2][0], 2), []float64{3, 0})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, updated[2][1], 2), []float64{6, 5})
	assertLocalStats(t, result.Stats)
}
