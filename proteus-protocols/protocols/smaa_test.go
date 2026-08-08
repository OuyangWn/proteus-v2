package protocols

import (
	"testing"

	"threshold-ckks/threshold"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestSMAAReturnsEncryptedMaxAndArgmaxOneHot(t *testing.T) {
	params, err := ckks.NewParametersFromLiteral(ckks.ExampleParameters128BitLogN14LogQP438)
	if err != nil {
		t.Fatal(err)
	}

	keys := threshold.GenerateKeys(params, 3)
	rlk := ckks.NewKeyGenerator(params).GenRelinearizationKeyNew(keys.TotalSK)
	ctx := NewHomContext(params, rlwe.NewMemEvaluationKeySet(rlk))

	values := []*rlwe.Ciphertext{
		threshold.EncryptVector(params, keys.PK, []float64{1, 2}),
		threshold.EncryptVector(params, keys.PK, []float64{8, 5}),
		threshold.EncryptVector(params, keys.PK, []float64{3, 9}),
	}
	opts := DefaultSMAAOptions(2, []byte("smaa-mask-key"), []byte("smaa-public-seed"), "smaa-test")

	result := SMAA(ctx, keys, values, opts)

	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Max, 2), []float64{8, 9})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Argmax[0], 2), []float64{0, 0})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Argmax[1], 2), []float64{1, 0})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Argmax[2], 2), []float64{0, 1})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Index, 2), []float64{2, 3})
	if result.Comparisons != 2 || len(result.CompareStats) != 2 {
		t.Fatalf("expected two SPDCmp calls, got comparisons=%d stats=%d", result.Comparisons, len(result.CompareStats))
	}
	if result.Max.Degree() != 1 {
		t.Fatalf("SMAA max output must be degree 1, got %d", result.Max.Degree())
	}
	assertInteractiveStats(t, result.Stats, 2, 0)
}
