package protocols

import (
	"math"
	"testing"

	"threshold-ckks/threshold"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestSOHLSelectsEncryptedRowPerSlot(t *testing.T) {
	params, err := ckks.NewParametersFromLiteral(ckks.ExampleParameters128BitLogN14LogQP438)
	if err != nil {
		t.Fatal(err)
	}

	keys := threshold.GenerateKeys(params, 3)
	rlk := ckks.NewKeyGenerator(params).GenRelinearizationKeyNew(keys.TotalSK)
	ctx := NewHomContext(params, rlwe.NewMemEvaluationKeySet(rlk))

	// Two SIMD tasks: slot0 selects state 1, slot1 selects state 2.
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

	result := SOHLWithStats(ctx, state, qTable)
	row := result.Row
	assertCloseVector(t, decryptVector(params, keys.TotalSK, row[0], 2), []float64{1, 2})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, row[1], 2), []float64{8, 5})
	if row[0].Degree() != 1 || row[1].Degree() != 1 {
		t.Fatalf("SOHL outputs must be relinearized to degree 1, got %d and %d", row[0].Degree(), row[1].Degree())
	}
	assertLocalStats(t, result.Stats)

	saqsa := SAQSAWithStats(ctx, state, qTable)
	assertLocalStats(t, saqsa.Stats)
}

func decryptVector(params ckks.Parameters, sk *rlwe.SecretKey, ct *rlwe.Ciphertext, length int) []float64 {
	values := make([]float64, length)
	if err := ckks.NewEncoder(params).Decode(rlwe.NewDecryptor(params, sk).DecryptNew(ct), values); err != nil {
		panic(err)
	}
	return values
}

func assertCloseVector(t *testing.T, got, want []float64) {
	t.Helper()
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-3 {
			t.Fatalf("slot %d: got %.8f, want %.8f", i, got[i], want[i])
		}
	}
}

func assertLocalStats(t *testing.T, stats ProtocolStats) {
	t.Helper()
	if stats.Duration <= 0 {
		t.Fatal("expected positive protocol duration")
	}
	if stats.Communication.TotalBytes != 0 || stats.SPDCmpCalls != 0 || stats.SEDTPCalls != 0 {
		t.Fatalf("expected local-only stats, got %+v", stats)
	}
}

func assertInteractiveStats(t *testing.T, stats ProtocolStats, minSPDCmp int, minSEDTP int) {
	t.Helper()
	if stats.Duration <= 0 {
		t.Fatal("expected positive protocol duration")
	}
	if stats.SPDCmpCalls < minSPDCmp || stats.SEDTPCalls < minSEDTP {
		t.Fatalf("unexpected call counts: got spdcmp=%d sedtp=%d, want at least %d/%d", stats.SPDCmpCalls, stats.SEDTPCalls, minSPDCmp, minSEDTP)
	}
	if (minSPDCmp+minSEDTP) > 0 && stats.Communication.TotalBytes <= 0 {
		t.Fatalf("expected positive communication bytes, got %+v", stats.Communication)
	}
}
