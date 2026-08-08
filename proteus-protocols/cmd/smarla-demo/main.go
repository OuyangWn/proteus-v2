package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"proteus-protocols/protocols"
	"threshold-ckks/threshold"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const tolerance = 2e-2

type trainingScenario struct {
	Hospitals  int              `json:"hospitals"`
	Patients   int              `json:"patients"`
	States     int              `json:"states"`
	Actions    int              `json:"actions"`
	Alpha      []float64        `json:"alpha"`
	Gamma      []float64        `json:"gamma"`
	InitialQ   [][][][]float64  `json:"initial_q"`
	Trajectory []trajectoryStep `json:"trajectory"`
}

type trajectoryStep struct {
	State        [][]int   `json:"state"`
	NextState    [][]int   `json:"next_state"`
	Reward       []float64 `json:"reward"`
	Exploit      []float64 `json:"exploit"`
	RandomAction []int     `json:"random_action"`
}

type plainStepResult struct {
	Q             [][][][]float64
	Actions       [][]int
	CurrentQTotal []float64
	NextMaxQTotal []float64
	Delta         []float64
}

func main() {
	steps := flag.Int("steps", 0, "training steps to run; 0 uses the scenario length")
	hospitals := flag.Int("hospitals", 2, "number of hospitals/agents")
	patients := flag.Int("patients", 2, "number of SIMD patient/task slots")
	states := flag.Int("states", 3, "number of states per hospital")
	actions := flag.Int("actions", 3, "number of actions per hospital")
	parties := flag.Int("parties", 3, "number of threshold custodians")
	alpha := flag.Float64("alpha", 0.2, "default learning rate for each patient slot")
	gamma := flag.Float64("gamma", 0.5, "default discount factor for each patient slot")
	dataPath := flag.String("data", "", "optional JSON scenario file")
	refreshNextMax := flag.Bool("refresh-next-max", true, "refresh next-max ciphertext before TD update")
	refreshQ := flag.Bool("refresh-q", true, "refresh updated Q-table ciphertexts for multi-round training")
	flag.Parse()

	scenario := buildScenario(*dataPath, *steps, *hospitals, *patients, *states, *actions, *alpha, *gamma)
	validateScenario(scenario)

	params, err := ckks.NewParametersFromLiteral(ckks.ExampleParameters128BitLogN14LogQP438)
	if err != nil {
		panic(err)
	}
	if scenario.Patients > params.MaxSlots() {
		panic(fmt.Sprintf("patients %d exceed CKKS max slots %d", scenario.Patients, params.MaxSlots()))
	}
	if *parties <= 0 {
		panic("parties must be positive")
	}

	keys := threshold.GenerateKeys(params, *parties)
	rlk := ckks.NewKeyGenerator(params).GenRelinearizationKeyNew(keys.TotalSK)
	ctx := protocols.NewHomContext(params, rlwe.NewMemEvaluationKeySet(rlk))

	plainQ := clonePlainQ(scenario.InitialQ)
	encQ := encryptQTables(params, keys.PK, plainQ)
	totalStats := protocols.ProtocolStats{}
	start := time.Now()

	fmt.Printf("SMARLA demo: n=%d hospitals, m=%d patients, states=%d, actions=%d, steps=%d\n",
		scenario.Hospitals, scenario.Patients, scenario.States, scenario.Actions, len(scenario.Trajectory))
	if *dataPath != "" {
		fmt.Printf("scenario file=%s\n", *dataPath)
	}

	for step, sample := range scenario.Trajectory {
		agents := make([]protocols.SMARLAAgentInput, scenario.Hospitals)
		for h := 0; h < scenario.Hospitals; h++ {
			agents[h] = protocols.SMARLAAgentInput{
				QTable:       encQ[h],
				State:        encryptedOneHot(params, keys.PK, sample.State[h], scenario.States),
				NextState:    encryptedOneHot(params, keys.PK, sample.NextState[h], scenario.States),
				ExploitBit:   threshold.EncryptVector(params, keys.PK, sample.Exploit),
				RandomAction: encryptedOneHot(params, keys.PK, sample.RandomAction, scenario.Actions),
			}
		}

		opts := protocols.DefaultSMARLAOptions(scenario.Patients, []byte("demo-mask-key"), []byte("demo-public-seed"), fmt.Sprintf("demo-step-%d", step))
		opts.RefreshNextMax, opts.RefreshUpdatedQ = *refreshNextMax, *refreshQ
		encResult := protocols.SMARLAStep(ctx, keys, agents,
			threshold.EncryptVector(params, keys.PK, sample.Reward),
			threshold.EncryptVector(params, keys.PK, scenario.Alpha),
			threshold.EncryptVector(params, keys.PK, scenario.Gamma),
			opts,
		)
		plainResult := baselineStep(plainQ, sample.State, sample.NextState, sample.Exploit, sample.RandomAction, sample.Reward, scenario.Alpha, scenario.Gamma)
		plainQ, encQ = plainResult.Q, encResult.UpdatedQTables
		totalStats = mergeStats(totalStats, encResult.Stats)

		current := decryptVector(params, keys.TotalSK, encResult.CurrentQTotal, scenario.Patients)
		nextMax := decryptVector(params, keys.TotalSK, encResult.NextMaxQTotal, scenario.Patients)
		delta := decryptVector(params, keys.TotalSK, encResult.Delta, scenario.Patients)
		maxDiff := maxQDiff(params, keys.TotalSK, encQ, plainQ, scenario.Patients)
		if maxDiff > tolerance {
			panic(fmt.Sprintf("encrypted/plain Q mismatch at step %d: max diff %.6f", step, maxDiff))
		}

		fmt.Printf("\nstep %d\n", step)
		for h, chosen := range plainResult.Actions {
			fmt.Printf("  actions hospital%d=%v\n", h, chosen)
		}
		fmt.Printf("  reward=%v alpha=%v gamma=%v\n", roundSlice(sample.Reward), roundSlice(scenario.Alpha), roundSlice(scenario.Gamma))
		fmt.Printf("  current Q_total enc=%v plain=%v\n", roundSlice(current), roundSlice(plainResult.CurrentQTotal))
		fmt.Printf("  next max Q_total enc=%v plain=%v\n", roundSlice(nextMax), roundSlice(plainResult.NextMaxQTotal))
		fmt.Printf("  delta enc=%v plain=%v\n", roundSlice(delta), roundSlice(plainResult.Delta))
		fmt.Printf("  max Q-table diff=%.6f\n", maxDiff)
		printStats("  step stats", encResult.Stats)
	}

	fmt.Printf("\ncompleted in %s\n", time.Since(start))
	printStats("total stats", totalStats)
}

func buildScenario(path string, steps, hospitals, patients, states, actions int, alpha, gamma float64) trainingScenario {
	if path == "" {
		if steps == 0 {
			steps = 2
		}
		return syntheticScenario(hospitals, patients, states, actions, steps, alpha, gamma)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	var scenario trainingScenario
	if err := json.Unmarshal(raw, &scenario); err != nil {
		panic(err)
	}
	fillScenarioDefaults(&scenario, steps, hospitals, patients, states, actions, alpha, gamma)
	return scenario
}

func fillScenarioDefaults(s *trainingScenario, steps, hospitals, patients, states, actions int, alpha, gamma float64) {
	if s.Hospitals == 0 {
		s.Hospitals = firstPositive(len(s.InitialQ), hospitals)
	}
	if s.States == 0 && len(s.InitialQ) > 0 {
		s.States = len(s.InitialQ[0])
	}
	if s.Actions == 0 && len(s.InitialQ) > 0 && len(s.InitialQ[0]) > 0 {
		s.Actions = len(s.InitialQ[0][0])
	}
	if s.Patients == 0 {
		s.Patients = inferPatients(*s, patients)
	}
	if s.States == 0 {
		s.States = states
	}
	if s.Actions == 0 {
		s.Actions = actions
	}
	if len(s.InitialQ) == 0 {
		s.InitialQ = syntheticInitialQ(s.Hospitals, s.Patients, s.States, s.Actions)
	}
	s.Alpha = normalizeFloatVector("alpha", s.Alpha, s.Patients, alpha)
	s.Gamma = normalizeFloatVector("gamma", s.Gamma, s.Patients, gamma)

	if steps > 0 {
		if len(s.Trajectory) < steps {
			panic(fmt.Sprintf("scenario has %d trajectory steps, cannot run %d", len(s.Trajectory), steps))
		}
		s.Trajectory = s.Trajectory[:steps]
	}
	if len(s.Trajectory) == 0 {
		count := steps
		if count == 0 {
			count = 2
		}
		s.Trajectory = syntheticTrajectory(s.Hospitals, s.Patients, s.States, s.Actions, count)
	}
	for i := range s.Trajectory {
		if len(s.Trajectory[i].Exploit) == 0 {
			s.Trajectory[i].Exploit = filledFloat64(s.Patients, 1)
		}
		if len(s.Trajectory[i].RandomAction) == 0 {
			s.Trajectory[i].RandomAction = make([]int, s.Patients)
		}
	}
}

func syntheticScenario(hospitals, patients, states, actions, steps int, alpha, gamma float64) trainingScenario {
	return trainingScenario{
		Hospitals:  hospitals,
		Patients:   patients,
		States:     states,
		Actions:    actions,
		Alpha:      filledFloat64(patients, alpha),
		Gamma:      filledFloat64(patients, gamma),
		InitialQ:   syntheticInitialQ(hospitals, patients, states, actions),
		Trajectory: syntheticTrajectory(hospitals, patients, states, actions, steps),
	}
}

func syntheticInitialQ(hospitals, patients, states, actions int) [][][][]float64 {
	q := make([][][][]float64, hospitals)
	for h := 0; h < hospitals; h++ {
		q[h] = make([][][]float64, states)
		for s := 0; s < states; s++ {
			q[h][s] = make([][]float64, actions)
			for a := 0; a < actions; a++ {
				q[h][s][a] = make([]float64, patients)
				for j := 0; j < patients; j++ {
					q[h][s][a][j] = float64(1+h*7+s*3+a*2+j) + 0.1*float64((h+s+a+j)%3)
				}
			}
		}
	}
	return q
}

func syntheticTrajectory(hospitals, patients, states, actions, steps int) []trajectoryStep {
	trajectory := make([]trajectoryStep, steps)
	for t := 0; t < steps; t++ {
		step := trajectoryStep{
			State:        makeIntMatrix(hospitals, patients),
			NextState:    makeIntMatrix(hospitals, patients),
			Reward:       make([]float64, patients),
			Exploit:      filledFloat64(patients, 1),
			RandomAction: make([]int, patients),
		}
		for h := 0; h < hospitals; h++ {
			for j := 0; j < patients; j++ {
				step.State[h][j] = (t + h + j) % states
				step.NextState[h][j] = (step.State[h][j] + 1) % states
			}
		}
		for j := 0; j < patients; j++ {
			step.Reward[j] = 5 + 0.5*float64(j) - 0.25*float64(t)
			step.RandomAction[j] = (t + j) % actions
		}
		trajectory[t] = step
	}
	return trajectory
}

func baselineStep(q [][][][]float64, state, nextState [][]int, exploit []float64, randomAction []int, reward, alpha, gamma []float64) plainStepResult {
	out := clonePlainQ(q)
	hospitals, patients := len(q), len(reward)
	actionsChosen := make([][]int, hospitals)
	currentQTotal, nextMaxQTotal, delta := make([]float64, patients), make([]float64, patients), make([]float64, patients)

	for h := 0; h < hospitals; h++ {
		actionsChosen[h] = make([]int, patients)
		for j := 0; j < patients; j++ {
			greedy := rightBiasedArgmax(q[h][state[h][j]], j)
			a := randomAction[j]
			if exploit[j] >= 0.5 {
				a = greedy
			}
			actionsChosen[h][j] = a
			currentQTotal[j] += q[h][state[h][j]][a][j]
			nextMaxQTotal[j] += q[h][nextState[h][j]][rightBiasedArgmax(q[h][nextState[h][j]], j)][j]
		}
	}
	for j := 0; j < patients; j++ {
		delta[j] = reward[j] + gamma[j]*nextMaxQTotal[j] - currentQTotal[j]
	}
	for h := 0; h < hospitals; h++ {
		for j := 0; j < patients; j++ {
			out[h][state[h][j]][actionsChosen[h][j]][j] += alpha[j] * delta[j]
		}
	}
	return plainStepResult{Q: out, Actions: actionsChosen, CurrentQTotal: currentQTotal, NextMaxQTotal: nextMaxQTotal, Delta: delta}
}

func rightBiasedArgmax(row [][]float64, patient int) int {
	best := 0
	for a := 1; a < len(row); a++ {
		if row[a][patient] >= row[best][patient] {
			best = a
		}
	}
	return best
}

func encryptQTables(params ckks.Parameters, pk *rlwe.PublicKey, q [][][][]float64) [][][]*rlwe.Ciphertext {
	out := make([][][]*rlwe.Ciphertext, len(q))
	for h := range q {
		out[h] = make([][]*rlwe.Ciphertext, len(q[h]))
		for s := range q[h] {
			out[h][s] = make([]*rlwe.Ciphertext, len(q[h][s]))
			for a := range q[h][s] {
				out[h][s][a] = threshold.EncryptVector(params, pk, q[h][s][a])
			}
		}
	}
	return out
}

func encryptedOneHot(params ckks.Parameters, pk *rlwe.PublicKey, selected []int, dimension int) []*rlwe.Ciphertext {
	out := make([]*rlwe.Ciphertext, dimension)
	for i := 0; i < dimension; i++ {
		values := make([]float64, len(selected))
		for slot, value := range selected {
			if value == i {
				values[slot] = 1
			}
		}
		out[i] = threshold.EncryptVector(params, pk, values)
	}
	return out
}

func decryptVector(params ckks.Parameters, sk *rlwe.SecretKey, ct *rlwe.Ciphertext, length int) []float64 {
	values := make([]float64, length)
	if err := ckks.NewEncoder(params).Decode(rlwe.NewDecryptor(params, sk).DecryptNew(ct), values); err != nil {
		panic(err)
	}
	return values
}

func maxQDiff(params ckks.Parameters, sk *rlwe.SecretKey, encrypted [][][]*rlwe.Ciphertext, plain [][][][]float64, patients int) float64 {
	maxDiff := 0.0
	for h := range encrypted {
		for s := range encrypted[h] {
			for a := range encrypted[h][s] {
				got := decryptVector(params, sk, encrypted[h][s][a], patients)
				for j, value := range got {
					maxDiff = math.Max(maxDiff, math.Abs(value-plain[h][s][a][j]))
				}
			}
		}
	}
	return maxDiff
}

func clonePlainQ(q [][][][]float64) [][][][]float64 {
	out := make([][][][]float64, len(q))
	for h := range q {
		out[h] = make([][][]float64, len(q[h]))
		for s := range q[h] {
			out[h][s] = make([][]float64, len(q[h][s]))
			for a := range q[h][s] {
				out[h][s][a] = append([]float64(nil), q[h][s][a]...)
			}
		}
	}
	return out
}

func validateScenario(s trainingScenario) {
	if s.Hospitals <= 0 || s.Patients <= 0 || s.States <= 0 || s.Actions <= 0 {
		panic("hospitals, patients, states, and actions must be positive")
	}
	if len(s.Alpha) != s.Patients || len(s.Gamma) != s.Patients {
		panic("alpha/gamma vector length must match patients")
	}
	validateQ(s.InitialQ, s.Hospitals, s.States, s.Actions, s.Patients)
	if len(s.Trajectory) == 0 {
		panic("trajectory must contain at least one step")
	}
	for i, step := range s.Trajectory {
		validateIndexMatrix(fmt.Sprintf("step %d state", i), step.State, s.Hospitals, s.Patients, s.States)
		validateIndexMatrix(fmt.Sprintf("step %d next_state", i), step.NextState, s.Hospitals, s.Patients, s.States)
		if len(step.Reward) != s.Patients || len(step.Exploit) != s.Patients || len(step.RandomAction) != s.Patients {
			panic(fmt.Sprintf("step %d reward/exploit/random_action length must match patients", i))
		}
		for j, action := range step.RandomAction {
			if action < 0 || action >= s.Actions {
				panic(fmt.Sprintf("step %d random action slot %d out of range", i, j))
			}
		}
		for j, value := range step.Exploit {
			if value < 0 || value > 1 {
				panic(fmt.Sprintf("step %d exploit slot %d must be in [0,1]", i, j))
			}
		}
	}
}

func validateQ(q [][][][]float64, hospitals, states, actions, patients int) {
	if len(q) != hospitals {
		panic("initial_q hospital dimension mismatch")
	}
	for h := 0; h < hospitals; h++ {
		if len(q[h]) != states {
			panic(fmt.Sprintf("initial_q hospital %d state dimension mismatch", h))
		}
		for s := 0; s < states; s++ {
			if len(q[h][s]) != actions {
				panic(fmt.Sprintf("initial_q hospital %d state %d action dimension mismatch", h, s))
			}
			for a := 0; a < actions; a++ {
				if len(q[h][s][a]) != patients {
					panic(fmt.Sprintf("initial_q hospital %d state %d action %d patient dimension mismatch", h, s, a))
				}
			}
		}
	}
}

func validateIndexMatrix(name string, matrix [][]int, rows, cols, limit int) {
	if len(matrix) != rows {
		panic(name + " row dimension mismatch")
	}
	for r := 0; r < rows; r++ {
		if len(matrix[r]) != cols {
			panic(fmt.Sprintf("%s row %d length mismatch", name, r))
		}
		for c, value := range matrix[r] {
			if value < 0 || value >= limit {
				panic(fmt.Sprintf("%s[%d][%d]=%d out of range [0,%d)", name, r, c, value, limit))
			}
		}
	}
}

func normalizeFloatVector(name string, values []float64, length int, fill float64) []float64 {
	if len(values) == 0 {
		return filledFloat64(length, fill)
	}
	if len(values) == 1 && length > 1 {
		return filledFloat64(length, values[0])
	}
	if len(values) != length {
		panic(fmt.Sprintf("%s length must be 1 or match patients", name))
	}
	return values
}

func inferPatients(s trainingScenario, fallback int) int {
	if len(s.InitialQ) > 0 && len(s.InitialQ[0]) > 0 && len(s.InitialQ[0][0]) > 0 {
		return len(s.InitialQ[0][0][0])
	}
	if len(s.Trajectory) > 0 && len(s.Trajectory[0].Reward) > 0 {
		return len(s.Trajectory[0].Reward)
	}
	return fallback
}

func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func filledFloat64(length int, value float64) []float64 {
	out := make([]float64, length)
	for i := range out {
		out[i] = value
	}
	return out
}

func makeIntMatrix(rows, cols int) [][]int {
	out := make([][]int, rows)
	for i := range out {
		out[i] = make([]int, cols)
	}
	return out
}

func mergeStats(a, b protocols.ProtocolStats) protocols.ProtocolStats {
	a.Duration += b.Duration
	a.Communication.BroadcastBytes += b.Communication.BroadcastBytes
	a.Communication.BroadcastDeliveredBytes += b.Communication.BroadcastDeliveredBytes
	a.Communication.ShareBytes += b.Communication.ShareBytes
	a.Communication.ResultBytes += b.Communication.ResultBytes
	a.Communication.TotalBytes += b.Communication.TotalBytes
	a.SPDCmpCalls += b.SPDCmpCalls
	a.SEDTPCalls += b.SEDTPCalls
	return a
}

func printStats(label string, stats protocols.ProtocolStats) {
	fmt.Printf("%s: time=%s, comm=%.2f MB, SPDCmp=%d, SEDTP=%d\n",
		label,
		stats.Duration,
		float64(stats.Communication.TotalBytes)/(1024*1024),
		stats.SPDCmpCalls,
		stats.SEDTPCalls,
	)
}

func roundSlice(values []float64) []float64 {
	out := make([]float64, len(values))
	for i, value := range values {
		out[i] = math.Round(value*1000) / 1000
	}
	return out
}
