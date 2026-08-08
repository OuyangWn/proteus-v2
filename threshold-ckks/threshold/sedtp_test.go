package threshold

import (
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestSEDTPTransformsCiphertextToTargetKey(t *testing.T) {
	params, err := ckks.NewParametersFromLiteral(ckks.ExampleParameters128BitLogN14LogQP438)
	if err != nil {
		t.Fatal(err)
	}

	keysA, keysB := GenerateKeys(params, 3), GenerateKeys(params, 3)
	want := []float64{-1.25, 0, 3.5, 8.75}
	ctA := EncryptVectorAtLevelScale(params, keysA.PK, want, params.MaxLevel()-2, params.DefaultScale())
	opts := DefaultSEDTPOptions(len(want), []byte("deterministic-public-seed"), "sedtp-test")

	result := SEDTPWithTranscript(params, keysA, keysB.PK, ctA, opts)
	got := decryptTestVector(params, keysB.TotalSK, result.Ciphertext, len(want))

	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-3 {
			t.Fatalf("slot %d: got %.8f, want %.8f", i, got[i], want[i])
		}
	}
	if result.Stats.Communication.TotalBytes == 0 || result.Stats.Durations.Total == 0 {
		t.Fatal("expected non-zero SEDTP stats")
	}
	if result.Ciphertext.Level() != params.MaxLevel() {
		t.Fatalf("SEDTP should refresh to max level: got %d, want %d", result.Ciphertext.Level(), params.MaxLevel())
	}
}

func decryptTestVector(params ckks.Parameters, sk *rlwe.SecretKey, ct *rlwe.Ciphertext, length int) []float64 {
	values := make([]float64, length)
	if err := ckks.NewEncoder(params).Decode(rlwe.NewDecryptor(params, sk).DecryptNew(ct), values); err != nil {
		panic(err)
	}
	return values
}
