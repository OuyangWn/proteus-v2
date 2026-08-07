package main

import (
	"crypto/rand"
	"fmt"

	"threshold-ckks/threshold"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func main() {
	params, err := ckks.NewParametersFromLiteral(ckks.ExampleParameters128BitLogN14LogQP438)
	if err != nil {
		panic(err)
	}
	fmt.Println("Max slots:", params.MaxSlots())

	keys := threshold.GenerateKeys(params, 3)
	fmt.Println("Party count:", len(keys.PartySK))
	fmt.Println("PK ready:", keys.PK != nil)

	// Demo only: create two encrypted input vectors.
	x, y := []float64{0.3, 1.2, 5, 2.1}, []float64{2.6, 0.4, 5, 0.2}
	ctX := threshold.EncryptVector(params, keys.PK, x)
	ctY := threshold.EncryptVector(params, keys.PK, y)

	maskKey := make([]byte, 32)
	if _, err = rand.Read(maskKey); err != nil {
		panic(err)
	}
	publicSeed := make([]byte, 32)
	if _, err = rand.Read(publicSeed); err != nil {
		panic(err)
	}

	// Upper-layer API shape: SPDCmp(Enc(x), Enc(y)) -> Enc(f).
	opts := threshold.DefaultSPDCmpOptions(len(x), maskKey, "state0:a1:a2", publicSeed, "call-0001")
	result := threshold.SPDCmpWithTranscript(params, keys, ctX, ctY, opts)
	fmt.Println("Reconstructor:", result.Reconstructor)
	fmt.Println("Partial shares:", result.ShareCount)
	fmt.Println("Transcript call:", result.Transcript.Broadcast.CallID)
	fmt.Println("Transcript result from:", result.Transcript.Result.From)
	fmt.Println("Comparison bits:", result.Bits)
	fmt.Println("Time CSO broadcast:", result.Stats.Durations.CSOBroadcast)
	fmt.Println("Time share generation:", result.Stats.Durations.ShareGeneration)
	fmt.Println("Time reconstruction:", result.Stats.Durations.Reconstruction)
	fmt.Println("Time total:", result.Stats.Durations.Total)
	fmt.Printf("Comm bytes: broadcast=%d, broadcast_delivered=%d, shares=%d, result=%d, total=%d\n",
		result.Stats.Communication.BroadcastBytes,
		result.Stats.Communication.BroadcastDeliveredBytes,
		result.Stats.Communication.ShareBytes,
		result.Stats.Communication.ResultBytes,
		result.Stats.Communication.TotalBytes,
	)

	decryptor := rlwe.NewDecryptor(params, keys.TotalSK)
	ptF := decryptor.DecryptNew(result.Ciphertext)
	decodedF := make([]float64, len(result.Bits))
	if err = ckks.NewEncoder(params).Decode(ptF, decodedF); err != nil {
		panic(err)
	}
	fmt.Println("Enc(f) decrypt check:", decodedF)
}
