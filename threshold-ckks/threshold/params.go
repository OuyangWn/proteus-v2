package threshold

import (
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func GetParameters() ckks.Parameters {

	params, err := ckks.NewParametersFromLiteral(
		ckks.ParametersLiteral{

			LogN: 14,

			LogQ: []int{
				55,
				45,
				45,
				45,
			},

			LogP: []int{
				60,
			},

			LogDefaultScale: 40,
		},
	)

	if err != nil {
		panic(err)
	}

	return params
}
