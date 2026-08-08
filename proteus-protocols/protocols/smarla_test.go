package protocols

import (
	"testing"

	"threshold-ckks/threshold"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestSMARLAStepRunsOneEncryptedVDNUpdate(t *testing.T) {
	params, err := ckks.NewParametersFromLiteral(ckks.ExampleParameters128BitLogN14LogQP438)
	if err != nil {
		t.Fatal(err)
	}

	keys := threshold.GenerateKeys(params, 3)
	rlk := ckks.NewKeyGenerator(params).GenRelinearizationKeyNew(keys.TotalSK)
	ctx := NewHomContext(params, rlwe.NewMemEvaluationKeySet(rlk))

	qTable := [][]*rlwe.Ciphertext{
		{
			threshold.EncryptVector(params, keys.PK, []float64{1}),
			threshold.EncryptVector(params, keys.PK, []float64{4}),
		},
		{
			threshold.EncryptVector(params, keys.PK, []float64{3}),
			threshold.EncryptVector(params, keys.PK, []float64{6}),
		},
	}
	agent := SMARLAAgentInput{
		QTable: qTable,
		State: []*rlwe.Ciphertext{
			threshold.EncryptVector(params, keys.PK, []float64{1}),
			threshold.EncryptVector(params, keys.PK, []float64{0}),
		},
		NextState: []*rlwe.Ciphertext{
			threshold.EncryptVector(params, keys.PK, []float64{0}),
			threshold.EncryptVector(params, keys.PK, []float64{1}),
		},
		ExploitBit: threshold.EncryptVector(params, keys.PK, []float64{1}),
		RandomAction: []*rlwe.Ciphertext{
			threshold.EncryptVector(params, keys.PK, []float64{1}),
			threshold.EncryptVector(params, keys.PK, []float64{0}),
		},
	}
	reward := threshold.EncryptVector(params, keys.PK, []float64{2})
	alpha := threshold.EncryptVector(params, keys.PK, []float64{0.1})
	gamma := threshold.EncryptVector(params, keys.PK, []float64{0.5})
	opts := DefaultSMARLAOptions(1, []byte("smarla-mask-key"), []byte("smarla-public-seed"), "smarla-test")

	result := SMARLAStep(ctx, keys, []SMARLAAgentInput{agent}, reward, alpha, gamma, opts)

	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Actions[0][0], 1), []float64{0})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Actions[0][1], 1), []float64{1})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.CurrentQTotal, 1), []float64{4})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.NextMaxQTotal, 1), []float64{6})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Delta, 1), []float64{1})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.UpdatedQTables[0][0][0], 1), []float64{1})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.UpdatedQTables[0][0][1], 1), []float64{4.1})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.UpdatedQTables[0][1][0], 1), []float64{3})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.UpdatedQTables[0][1][1], 1), []float64{6})
	if result.Refresh == nil {
		t.Fatal("expected same-key SEDTP refresh for next max")
	}
	assertAllQCellsAtMaxLevel(t, params, result.UpdatedQTables)
	assertInteractiveStats(t, result.Stats, 2, 7)
}

func TestSMARLAStepSupportsHospitalsAndPatientsParallel(t *testing.T) {
	params, err := ckks.NewParametersFromLiteral(ckks.ExampleParameters128BitLogN14LogQP438)
	if err != nil {
		t.Fatal(err)
	}

	keys := threshold.GenerateKeys(params, 3)
	rlk := ckks.NewKeyGenerator(params).GenRelinearizationKeyNew(keys.TotalSK)
	ctx := NewHomContext(params, rlwe.NewMemEvaluationKeySet(rlk))
	enc := func(values []float64) *rlwe.Ciphertext { return threshold.EncryptVector(params, keys.PK, values) }

	agent0Q := [][]*rlwe.Ciphertext{
		{enc([]float64{1, 10}), enc([]float64{4, 2})},
		{enc([]float64{3, 7}), enc([]float64{6, 1})},
	}
	agent1Q := [][]*rlwe.Ciphertext{
		{enc([]float64{2, 4}), enc([]float64{5, 8})},
		{enc([]float64{9, 1}), enc([]float64{3, 6})},
	}
	agents := []SMARLAAgentInput{
		{
			QTable:       agent0Q,
			State:        []*rlwe.Ciphertext{enc([]float64{1, 0}), enc([]float64{0, 1})},
			NextState:    []*rlwe.Ciphertext{enc([]float64{0, 1}), enc([]float64{1, 0})},
			ExploitBit:   enc([]float64{1, 1}),
			RandomAction: []*rlwe.Ciphertext{enc([]float64{1, 1}), enc([]float64{0, 0})},
		},
		{
			QTable:       agent1Q,
			State:        []*rlwe.Ciphertext{enc([]float64{0, 1}), enc([]float64{1, 0})},
			NextState:    []*rlwe.Ciphertext{enc([]float64{1, 0}), enc([]float64{0, 1})},
			ExploitBit:   enc([]float64{1, 1}),
			RandomAction: []*rlwe.Ciphertext{enc([]float64{1, 1}), enc([]float64{0, 0})},
		},
	}
	reward := enc([]float64{5, 4})
	alpha := enc([]float64{0.2, 0.1})
	gamma := enc([]float64{0.5, 0.25})
	opts := DefaultSMARLAOptions(2, []byte("smarla-multi-mask-key"), []byte("smarla-multi-public-seed"), "smarla-multi")

	result := SMARLAStep(ctx, keys, agents, reward, alpha, gamma, opts)

	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Actions[0][0], 2), []float64{0, 1})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Actions[0][1], 2), []float64{1, 0})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Actions[1][0], 2), []float64{1, 0})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Actions[1][1], 2), []float64{0, 1})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.CurrentQTotal, 2), []float64{13, 15})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.NextMaxQTotal, 2), []float64{11, 16})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Delta, 2), []float64{-2.5, -7})

	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.UpdatedQTables[0][0][0], 2), []float64{1, 10})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.UpdatedQTables[0][0][1], 2), []float64{3.5, 2})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.UpdatedQTables[0][1][0], 2), []float64{3, 6.3})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.UpdatedQTables[0][1][1], 2), []float64{6, 1})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.UpdatedQTables[1][0][0], 2), []float64{2, 4})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.UpdatedQTables[1][0][1], 2), []float64{5, 7.3})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.UpdatedQTables[1][1][0], 2), []float64{8.5, 1})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.UpdatedQTables[1][1][1], 2), []float64{3, 6})
	assertAllQCellsAtMaxLevel(t, params, result.UpdatedQTables)
	assertInteractiveStats(t, result.Stats, 4, 13)
}

func TestSMARLAStepSupportsMoreThanTwoStatesAndActions(t *testing.T) {
	params, err := ckks.NewParametersFromLiteral(ckks.ExampleParameters128BitLogN14LogQP438)
	if err != nil {
		t.Fatal(err)
	}

	keys := threshold.GenerateKeys(params, 3)
	rlk := ckks.NewKeyGenerator(params).GenRelinearizationKeyNew(keys.TotalSK)
	ctx := NewHomContext(params, rlwe.NewMemEvaluationKeySet(rlk))
	enc := func(values []float64) *rlwe.Ciphertext { return threshold.EncryptVector(params, keys.PK, values) }

	agent := SMARLAAgentInput{
		QTable: [][]*rlwe.Ciphertext{
			{enc([]float64{0}), enc([]float64{2}), enc([]float64{1})},
			{enc([]float64{3}), enc([]float64{1}), enc([]float64{2})},
			{enc([]float64{1}), enc([]float64{4}), enc([]float64{0})},
		},
		State:        []*rlwe.Ciphertext{enc([]float64{1}), enc([]float64{0}), enc([]float64{0})},
		NextState:    []*rlwe.Ciphertext{enc([]float64{0}), enc([]float64{1}), enc([]float64{0})},
		ExploitBit:   enc([]float64{1}),
		RandomAction: []*rlwe.Ciphertext{enc([]float64{1}), enc([]float64{0}), enc([]float64{0})},
	}

	result := SMARLAStep(ctx, keys, []SMARLAAgentInput{agent}, enc([]float64{1}), enc([]float64{0.1}), enc([]float64{0.5}), DefaultSMARLAOptions(1, []byte("smarla-3x3-mask-key"), []byte("smarla-3x3-public-seed"), "smarla-3x3"))

	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Actions[0][0], 1), []float64{0})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Actions[0][1], 1), []float64{1})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Actions[0][2], 1), []float64{0})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.CurrentQTotal, 1), []float64{2})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.NextMaxQTotal, 1), []float64{3})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.Delta, 1), []float64{0.5})
	assertCloseVector(t, decryptVector(params, keys.TotalSK, result.UpdatedQTables[0][0][1], 1), []float64{2.05})
	assertAllQCellsAtMaxLevel(t, params, result.UpdatedQTables)
	assertInteractiveStats(t, result.Stats, 4, 13)
}

func assertAllQCellsAtMaxLevel(t *testing.T, params ckks.Parameters, qTables [][][]*rlwe.Ciphertext) {
	t.Helper()
	for i := range qTables {
		for s := range qTables[i] {
			for a := range qTables[i][s] {
				if qTables[i][s][a].Level() != params.MaxLevel() {
					t.Fatalf("agent %d state %d action %d level: got %d, want %d", i, s, a, qTables[i][s][a].Level(), params.MaxLevel())
				}
			}
		}
	}
}
