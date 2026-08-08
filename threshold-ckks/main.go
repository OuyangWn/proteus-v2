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
	fmt.Println("\nSPDCmp")
	fmt.Println("Reconstructor:", result.Reconstructor)
	fmt.Println("Partial shares:", result.ShareCount)
	fmt.Println("Transcript call:", result.Transcript.Broadcast.CallID)
	fmt.Println("Transcript result from:", result.Transcript.Result.From)
	fmt.Println("Comparison bits:", result.Bits)
	printStats(result.Stats)

	decryptor := rlwe.NewDecryptor(params, keys.TotalSK)
	ptF := decryptor.DecryptNew(result.Ciphertext)
	decodedF := make([]float64, len(result.Bits))
	if err = ckks.NewEncoder(params).Decode(ptF, decodedF); err != nil {
		panic(err)
	}
	fmt.Println("Enc(f) decrypt check:", decodedF)

	// SEDTP key-domain transform: Enc_pkA(x) -> Enc_pkB(x).
	targetKeys := threshold.GenerateKeys(params, 3)
	sedtpSeed := make([]byte, 32)
	if _, err = rand.Read(sedtpSeed); err != nil {
		panic(err)
	}
	sedtpOpts := threshold.DefaultSEDTPOptions(len(x), sedtpSeed, "sedtp-0001")
	sedtp := threshold.SEDTPWithTranscript(params, keys, targetKeys.PK, ctX, sedtpOpts)
	fmt.Println("\nSEDTP")
	fmt.Println("Reconstructor:", sedtp.Reconstructor)
	fmt.Println("Partial shares:", sedtp.ShareCount)
	fmt.Println("Transcript call:", sedtp.Transcript.Broadcast.CallID)
	fmt.Println("Transcript result from:", sedtp.Transcript.Result.From)
	printStats(sedtp.Stats)
	fmt.Println("Enc_pkB(x) decrypt check:", decryptVector(params, targetKeys.TotalSK, sedtp.Ciphertext, len(x)))
}

func printStats(stats threshold.ProtocolStats) {
	fmt.Println("Time CSO broadcast:", stats.Durations.CSOBroadcast)
	fmt.Println("Time share generation:", stats.Durations.ShareGeneration)
	fmt.Println("Time reconstruction:", stats.Durations.Reconstruction)
	if stats.Durations.CSOFinalize > 0 {
		fmt.Println("Time CSO finalize:", stats.Durations.CSOFinalize)
	}
	fmt.Println("Time total:", stats.Durations.Total)
	fmt.Printf("Comm bytes: broadcast=%d, broadcast_delivered=%d, shares=%d, result=%d, total=%d\n",
		stats.Communication.BroadcastBytes,
		stats.Communication.BroadcastDeliveredBytes,
		stats.Communication.ShareBytes,
		stats.Communication.ResultBytes,
		stats.Communication.TotalBytes,
	)
}

func decryptVector(params ckks.Parameters, sk *rlwe.SecretKey, ct *rlwe.Ciphertext, length int) []float64 {
	values := make([]float64, length)
	if err := ckks.NewEncoder(params).Decode(rlwe.NewDecryptor(params, sk).DecryptNew(ct), values); err != nil {
		panic(err)
	}
	return values
}
