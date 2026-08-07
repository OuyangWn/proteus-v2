package threshold

// ExtractPositiveIndicators is the demo sign test used after reconstruction.
func ExtractPositiveIndicators(values []float64, epsilon float64) []float64 {
	out := make([]float64, len(values))
	for i, value := range values {
		if value > epsilon {
			out[i] = 1
		}
	}
	return out
}
