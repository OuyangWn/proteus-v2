package main

import (
	"fmt"

	"threshold-ckks/threshold"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func main() {

	// CKKS 参数
	params, err := ckks.NewParametersFromLiteral(
		ckks.ExampleParameters128BitLogN14LogQP438,
	)
	if err != nil {
		panic(err)
	}

	// threshold key generation
	keys := threshold.GenerateKeys(params, 3)

	fmt.Println("Party count:", len(keys.PartySK))
	fmt.Println("PK ready:", keys.PK != nil)

	// 测试数据
	x := []float64{3, 1, 5}
	y := []float64{2, 4, 5}

	ctX := threshold.EncryptVector(
		params,
		keys.PK,
		x,
	)

	ctY := threshold.EncryptVector(
		params,
		keys.PK,
		y,
	)

	// homomorphic subtraction
	ctZ := threshold.SubCiphertexts(
		params,
		ctX,
		ctY,
	)

	// ============================
	// Threshold partial decrypt
	// ============================

	mus := make([]ring.Poly, len(keys.PartySK))

	for i, sk := range keys.PartySK {

		mus[i] = threshold.PartialDecrypt(
			params,
			ctZ,
			sk,
		)

		fmt.Println(
			"Partial decrypt μ_", i, "generated",
		)
	}

	// aggregate μ_h
	s := threshold.AggregateDecrypt(
		params,
		ctZ,
		mus,
	)

	fmt.Println(
		"Aggregated decrypt polynomial N =",
		s.N(),
	)

	// ============================
	// Normal decrypt
	// ============================

	decryptor := rlwe.NewDecryptor(
		params,
		keys.TotalSK,
	)

	pt := decryptor.DecryptNew(ctZ)

	fmt.Println(
		"Normal decrypt polynomial N =",
		pt.Value.N(),
	)

	// 验证 polynomial 是否一致
	fmt.Println(
		"Polynomial equal:",
		s.Equal(&pt.Value),
	)

	// ============================
	// Normal decode
	// ============================

	encoder := ckks.NewEncoder(params)

	normalDecoded := make([]float64, 3)

	err = encoder.Decode(
		pt,
		normalDecoded,
	)

	if err != nil {
		panic(err)
	}

	fmt.Println(
		"Normal plaintext:",
		normalDecoded,
	)

	// ============================
	// Threshold plaintext
	// 使用 normal plaintext 作为模板
	// ============================

	thresholdPT := threshold.PolyToPlaintext(
		s,
		pt,
	)

	thresholdDecoded := make([]float64, 3)

	err = encoder.Decode(
		thresholdPT,
		thresholdDecoded,
	)

	if err != nil {
		panic(err)
	}

	fmt.Println(
		"Threshold plaintext:",
		thresholdDecoded,
	)
}
