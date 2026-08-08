package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"proteus-protocols/protocols"
	"threshold-ckks/threshold"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const qTolerance = 5e-2

const (
	frozenLakeWidth   = 3
	frozenLakeStates  = 9
	frozenLakeActions = 4
)

type medicalFrozenLakeEnv struct {
	hospitals, patients int
	starts              [][]int
	current             [][]int
	goals               []int
	hazards             []map[int]bool
}

type transition struct {
	state, nextState [][]int
	exploit          [][]float64
	randomAction     [][]int
	action           [][]int
	agentReward      [][]float64
	teamReward       []float64
}

type plainStep struct {
	q                     [][][][]float64
	currentQ, nextMax, td []float64
}

type rewardSnapshot struct {
	teamAvg    float64
	agentAvg   []float64
	agent0Task []float64
}

type curvePoint struct {
	step            int
	epsilon         float64
	teamAvg         float64
	agentAvg        []float64
	agent0Task      []float64
	greedyTeamAvg   float64
	trainTeamAvg    float64
	trainAgentAvg   []float64
	trainAgent0Task []float64
	duration        time.Duration
	commMB          float64
	spdcmp, sedtp   int
	maxQDiff        float64
}

func main() {
	steps := flag.Int("steps", 6, "training epochs/episodes")
	horizon := flag.Int("horizon", 8, "maximum environment steps per trajectory")
	hospitals := flag.Int("hospitals", 3, "number of hospitals/agents")
	patients := flag.Int("patients", 5, "number of SIMD patient/task slots")
	states := flag.Int("states", frozenLakeStates, "medical FrozenLake states; fixed to 9")
	actions := flag.Int("actions", frozenLakeActions, "medical FrozenLake actions; fixed to 4")
	parties := flag.Int("parties", 3, "number of threshold custodians")
	alpha := flag.Float64("alpha", 0.3, "learning rate")
	gamma := flag.Float64("gamma", 0.4, "discount factor")
	epsStart := flag.Float64("epsilon-start", 0.7, "initial exploration probability")
	epsEnd := flag.Float64("epsilon-end", 0.05, "minimum exploration probability")
	epsDecay := flag.Float64("epsilon-decay", 0.82, "multiplicative epsilon decay")
	window := flag.Int("window", 5, "moving average window for CSV curves")
	seed := flag.Int64("seed", 7, "deterministic simulation seed")
	outDir := flag.String("out", "./results/frozenlake-medical-n3-m5", "output directory for reward curves")
	flag.Parse()

	validateFlags(*steps, *horizon, *hospitals, *patients, *states, *actions, *parties, *alpha, *gamma, *epsStart, *epsEnd, *epsDecay, *window)
	env := newMedicalFrozenLakeEnv(*hospitals, *patients)

	params, err := ckks.NewParametersFromLiteral(ckks.ExampleParameters128BitLogN14LogQP438)
	if err != nil {
		panic(err)
	}
	if env.patients > params.MaxSlots() {
		panic(fmt.Sprintf("patients %d exceed CKKS max slots %d", env.patients, params.MaxSlots()))
	}

	keys := threshold.GenerateKeys(params, *parties)
	rlk := ckks.NewKeyGenerator(params).GenRelinearizationKeyNew(keys.TotalSK)
	ctx := protocols.NewHomContext(params, rlwe.NewMemEvaluationKeySet(rlk))
	rng := rand.New(rand.NewSource(*seed))

	alphaVec, gammaVec := filled(env.patients, *alpha), filled(env.patients, *gamma)
	plainQ := env.initialQ()
	encQ := encryptQ(params, keys.PK, plainQ)
	points := make([]curvePoint, 0, *steps)
	totalStats := protocols.ProtocolStats{}
	start := time.Now()

	fmt.Printf("Medical FrozenLake MARL: n=%d agents, m=%d tasks, states=%d, actions=%d, epochs=%d, horizon=%d\n",
		env.hospitals, env.patients, frozenLakeStates, frozenLakeActions, *steps, *horizon)
	fmt.Printf("patient initial states per agent=%v\n", env.starts)
	for epoch := 0; epoch < *steps; epoch++ {
		epsilon := math.Max(*epsEnd, *epsStart*math.Pow(*epsDecay, float64(epoch)))
		env.resetEpisode()
		active := filledBool(env.patients, true)
		episodeAgentReward := makeFloatMatrix(env.hospitals, env.patients)
		episodeTeamReward := make([]float64, env.patients)
		epochStats := protocols.ProtocolStats{}
		maxDiff := 0.0
		executedSteps := 0
		lastState0, lastAction0 := []int{}, []int{}

		for t := 0; t < *horizon && anyActive(active); t++ {
			activeBefore := append([]bool(nil), active...)
			tr := env.sampleTransition(epsilon, rng, activeBefore)
			selectedActions, selectionStats := selectEncryptedActions(ctx, keys, encQ, tr, epoch, t)
			env.applyActions(&tr, decodeEncryptedActions(params, keys.TotalSK, selectedActions, env.patients), active)

			agents := make([]protocols.SMARLAAgentInput, env.hospitals)
			for h := 0; h < env.hospitals; h++ {
				agents[h] = protocols.SMARLAAgentInput{
					QTable:         encQ[h],
					State:          encryptedOneHot(params, keys.PK, tr.state[h], frozenLakeStates),
					NextState:      encryptedOneHot(params, keys.PK, tr.nextState[h], frozenLakeStates),
					ActionOverride: selectedActions[h],
				}
			}

			alphaStep := activeAlpha(alphaVec, activeBefore)
			opts := protocols.DefaultSMARLAOptions(env.patients, []byte("train-mask-key"), []byte("train-public-seed"), fmt.Sprintf("train-epoch-%d-step-%d", epoch, t))
			encResult := protocols.SMARLAStep(ctx, keys, agents,
				threshold.EncryptVector(params, keys.PK, tr.teamReward),
				threshold.EncryptVector(params, keys.PK, alphaStep),
				threshold.EncryptVector(params, keys.PK, gammaVec),
				opts,
			)

			plain := baselineUpdate(plainQ, tr.state, tr.nextState, tr.action, tr.teamReward, alphaStep, gammaVec)
			plainQ, encQ = plain.q, encResult.UpdatedQTables
			stepStats := mergeStats(selectionStats, encResult.Stats)
			epochStats = mergeStats(epochStats, stepStats)
			totalStats = mergeStats(totalStats, stepStats)
			maxDiff = math.Max(maxDiff, maxQDiff(params, keys.TotalSK, encQ, plainQ, env.patients))
			if maxDiff > qTolerance {
				panic(fmt.Sprintf("encrypted/plain Q mismatch at epoch %d step %d: max diff %.6f", epoch, t, maxDiff))
			}

			for h := 0; h < env.hospitals; h++ {
				for j := 0; j < env.patients; j++ {
					episodeAgentReward[h][j] += tr.agentReward[h][j]
				}
			}
			for j := 0; j < env.patients; j++ {
				episodeTeamReward[j] += tr.teamReward[j]
			}
			lastState0, lastAction0 = append([]int(nil), tr.state[0]...), append([]int(nil), tr.action[0]...)
			executedSteps++
		}

		greedyEval := env.evaluateGreedyEpisode(plainQ, *horizon)
		point := curvePoint{
			step:            epoch,
			epsilon:         epsilon,
			teamAvg:         avg(episodeTeamReward),
			agentAvg:        agentAverages(episodeAgentReward),
			agent0Task:      append([]float64(nil), episodeAgentReward[0]...),
			greedyTeamAvg:   greedyEval.teamAvg,
			trainTeamAvg:    avg(episodeTeamReward),
			trainAgentAvg:   agentAverages(episodeAgentReward),
			trainAgent0Task: append([]float64(nil), episodeAgentReward[0]...),
			duration:        epochStats.Duration,
			commMB:          float64(epochStats.Communication.TotalBytes) / (1024 * 1024),
			spdcmp:          epochStats.SPDCmpCalls,
			sedtp:           epochStats.SEDTPCalls,
			maxQDiff:        maxDiff,
		}
		points = append(points, point)

		fmt.Printf("epoch=%02d eps=%.3f return=%.3f greedy=%.3f traj_steps=%d agents=%v agent0_tasks=%v last_states0=%v last_actions0=%v comm=%.2fMB time=%s\n",
			epoch, epsilon, point.teamAvg, point.greedyTeamAvg, executedSteps, round(point.agentAvg), round(point.agent0Task), lastState0, lastAction0, point.commMB, point.duration)
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		panic(err)
	}
	csvPath := filepath.Join(*outDir, "reward_curves.csv")
	if err := writeRewardCSV(csvPath, points, env.hospitals, env.patients, *window); err != nil {
		panic(err)
	}
	fmt.Printf("\ncompleted in %s\n", time.Since(start))
	printStats("total stats", totalStats)
	fmt.Printf("curves written:\n  %s\n", csvPath)
}

func newMedicalFrozenLakeEnv(hospitals, patients int) *medicalFrozenLakeEnv {
	baseStarts := []int{0, 1, 3, 4, 6}
	env := &medicalFrozenLakeEnv{
		hospitals: hospitals,
		patients:  patients,
		starts:    makeIntMatrix(hospitals, patients),
		current:   makeIntMatrix(hospitals, patients),
		goals:     make([]int, hospitals),
		hazards:   make([]map[int]bool, hospitals),
	}
	for h := 0; h < hospitals; h++ {
		env.goals[h] = 8
		env.hazards[h] = map[int]bool{5: true}
		if h%3 == 1 {
			env.hazards[h] = map[int]bool{3: true}
		} else if h%3 == 2 {
			env.hazards[h] = map[int]bool{7: true}
		}
		for j := 0; j < patients; j++ {
			start := baseStarts[j%len(baseStarts)]
			if env.hazards[h][start] || start == env.goals[h] {
				start = (start + h + 1) % frozenLakeStates
			}
			env.starts[h][j], env.current[h][j] = start, start
		}
	}
	return env
}

func (e *medicalFrozenLakeEnv) initialQ() [][][][]float64 {
	q := make([][][][]float64, e.hospitals)
	for h := 0; h < e.hospitals; h++ {
		q[h] = make([][][]float64, frozenLakeStates)
		for s := 0; s < frozenLakeStates; s++ {
			q[h][s] = make([][]float64, frozenLakeActions)
			for a := 0; a < frozenLakeActions; a++ {
				q[h][s][a] = make([]float64, e.patients)
				for j := 0; j < e.patients; j++ {
					q[h][s][a][j] = e.priorQ(h, s, a)
				}
			}
		}
	}
	return q
}

func (e *medicalFrozenLakeEnv) resetEpisode() {
	for h := 0; h < e.hospitals; h++ {
		copy(e.current[h], e.starts[h])
	}
}

func (e *medicalFrozenLakeEnv) sampleTransition(epsilon float64, rng *rand.Rand, active []bool) transition {
	tr := transition{
		state:        makeIntMatrix(e.hospitals, e.patients),
		nextState:    makeIntMatrix(e.hospitals, e.patients),
		exploit:      makeFloatMatrix(e.hospitals, e.patients),
		randomAction: makeIntMatrix(e.hospitals, e.patients),
		action:       makeIntMatrix(e.hospitals, e.patients),
		agentReward:  makeFloatMatrix(e.hospitals, e.patients),
		teamReward:   make([]float64, e.patients),
	}
	for h := 0; h < e.hospitals; h++ {
		for j := 0; j < e.patients; j++ {
			tr.state[h][j] = e.current[h][j]
			if !active[j] {
				tr.exploit[h][j] = 1
				continue
			}
			tr.randomAction[h][j] = rng.Intn(frozenLakeActions)
			if rng.Float64() >= epsilon {
				tr.exploit[h][j] = 1
			}
		}
	}
	return tr
}

func (e *medicalFrozenLakeEnv) applyActions(tr *transition, actions [][]int, active []bool) {
	tr.action = actions
	tr.nextState = makeIntMatrix(e.hospitals, e.patients)
	tr.agentReward = makeFloatMatrix(e.hospitals, e.patients)
	tr.teamReward = make([]float64, e.patients)
	for j := 0; j < e.patients; j++ {
		if !active[j] {
			for h := 0; h < e.hospitals; h++ {
				tr.nextState[h][j] = tr.state[h][j]
			}
			continue
		}
		hazard := false
		allGoal := true
		for h := 0; h < e.hospitals; h++ {
			if e.isGoal(h, e.current[h][j]) {
				// Already at goal — freeze, no step, no reward.
				tr.nextState[h][j] = e.current[h][j]
				tr.agentReward[h][j] = 0
			} else {
				tr.nextState[h][j] = e.stepState(tr.state[h][j], tr.action[h][j])
				tr.agentReward[h][j] = e.reward(h, tr.state[h][j], tr.nextState[h][j])
				e.current[h][j] = tr.nextState[h][j]
				if e.isHazard(h, tr.nextState[h][j]) {
					hazard = true
				}
			}
			tr.teamReward[j] += tr.agentReward[h][j] / float64(e.hospitals)
			if !e.isGoal(h, e.current[h][j]) {
				allGoal = false
			}
		}
		if hazard || allGoal {
			active[j] = false
		}
	}
}

func (e *medicalFrozenLakeEnv) evaluateGreedyEpisode(
	q [][][][]float64, horizon int,
) rewardSnapshot {
	agentReward := makeFloatMatrix(e.hospitals, e.patients)
	teamReward := make([]float64, e.patients)
	state := makeIntMatrix(e.hospitals, e.patients)
	for h := 0; h < e.hospitals; h++ {
		copy(state[h], e.starts[h])
	}
	active := filledBool(e.patients, true)

	for t := 0; t < horizon && anyActive(active); t++ {
		for j := 0; j < e.patients; j++ {
			if !active[j] {
				continue
			}
			hazard := false
			allGoal := true
			for h := 0; h < e.hospitals; h++ {
				if e.isGoal(h, state[h][j]) {
					// Frozen at goal, no reward.
					continue
				}
				action := rightBiasedArgmax(q[h][state[h][j]], j)
				next := e.stepState(state[h][j], action)
				reward := e.reward(h, state[h][j], next)
				agentReward[h][j] += reward
				teamReward[j] += reward / float64(e.hospitals)
				state[h][j] = next
				if e.isHazard(h, next) {
					hazard = true
				}
			}
			// Check all-goal condition after stepping.
			for h := 0; h < e.hospitals; h++ {
				if !e.isGoal(h, state[h][j]) {
					allGoal = false
					break
				}
			}
			if hazard || allGoal {
				active[j] = false
			}
		}
	}
	return rewardSnapshot{
		teamAvg:    avg(teamReward),
		agentAvg:   agentAverages(agentReward),
		agent0Task: append([]float64(nil), agentReward[0]...),
	}
}

func (e *medicalFrozenLakeEnv) priorQ(hospital, state, action int) float64 {
	if e.isTerminal(hospital, state) || e.isGoal(hospital, state) {
		return 0
	}
	best := e.bestActionTowardGoal(hospital, state)
	if action == best {
		return 15
	}
	next := e.stepState(state, action)
	switch {
	case e.hazards[hospital][next]:
		return -8
	case e.distance(next, e.goals[hospital]) < e.distance(state, e.goals[hospital]):
		return 5
	default:
		return -1
	}
}

func (e *medicalFrozenLakeEnv) reward(hospital, state, next int) float64 {
	switch {
	case e.hazards[hospital][next]:
		return -10 // fell into hazard — strong penalty
	case next == e.goals[hospital]:
		return 20 // reached goal — strong bonus
	case e.distance(next, e.goals[hospital]) < e.distance(state, e.goals[hospital]):
		return -0.1 // moved closer to goal — slight step cost
	case next == state:
		return -0.5 // bumped into wall — moderate step cost
	default:
		return -0.3 // moved but not closer — slight step cost
	}
}

func (e *medicalFrozenLakeEnv) bestActionTowardGoal(hospital, state int) int {
	best, bestScore := 0, math.Inf(-1)
	for action := 0; action < frozenLakeActions; action++ {
		next := e.stepState(state, action)
		score := -float64(e.distance(next, e.goals[hospital]))
		if e.hazards[hospital][next] {
			score -= 10
		}
		if next == e.goals[hospital] {
			score += 10
		}
		if score > bestScore {
			best, bestScore = action, score
		}
	}
	return best
}

func (e *medicalFrozenLakeEnv) stepState(state, action int) int {
	row, col := state/frozenLakeWidth, state%frozenLakeWidth
	switch action {
	case 0:
		row--
	case 1:
		col++
	case 2:
		row++
	case 3:
		col--
	}
	if row < 0 || row >= frozenLakeWidth || col < 0 || col >= frozenLakeWidth {
		return state
	}
	return row*frozenLakeWidth + col
}

func (e *medicalFrozenLakeEnv) distance(a, b int) int {
	ar, ac := a/frozenLakeWidth, a%frozenLakeWidth
	br, bc := b/frozenLakeWidth, b%frozenLakeWidth
	return abs(ar-br) + abs(ac-bc)
}

func (e *medicalFrozenLakeEnv) isTerminal(hospital, state int) bool {
	return e.hazards[hospital][state]
}

func (e *medicalFrozenLakeEnv) isGoal(hospital, state int) bool {
	return state == e.goals[hospital]
}

func (e *medicalFrozenLakeEnv) isHazard(hospital, state int) bool {
	return e.hazards[hospital][state]
}

func baselineUpdate(q [][][][]float64, state, nextState, action [][]int, reward, alpha, gamma []float64) plainStep {
	out := cloneQ(q)
	hospitals, patients := len(q), len(reward)
	currentQ, nextMax, td := make([]float64, patients), make([]float64, patients), make([]float64, patients)
	for h := 0; h < hospitals; h++ {
		for j := 0; j < patients; j++ {
			currentQ[j] += q[h][state[h][j]][action[h][j]][j]
			nextMax[j] += q[h][nextState[h][j]][rightBiasedArgmax(q[h][nextState[h][j]], j)][j]
		}
	}
	for j := 0; j < patients; j++ {
		td[j] = reward[j] + gamma[j]*nextMax[j] - currentQ[j]
	}
	for h := 0; h < hospitals; h++ {
		for j := 0; j < patients; j++ {
			out[h][state[h][j]][action[h][j]][j] += alpha[j] * td[j]
		}
	}
	return plainStep{q: out, currentQ: currentQ, nextMax: nextMax, td: td}
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

func encryptQ(params ckks.Parameters, pk *rlwe.PublicKey, q [][][][]float64) [][][]*rlwe.Ciphertext {
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

func selectEncryptedActions(ctx protocols.HomContext, keys *threshold.ThresholdKeys, qTables [][][]*rlwe.Ciphertext, tr transition, epoch int, trajectoryStep int) ([][]*rlwe.Ciphertext, protocols.ProtocolStats) {
	actions := make([][]*rlwe.Ciphertext, len(qTables))
	stats := protocols.ProtocolStats{}
	for h := range qTables {
		result := protocols.SPASA(ctx, keys,
			encryptedOneHot(ctx.Params, keys.PK, tr.state[h], frozenLakeStates),
			qTables[h],
			threshold.EncryptVector(ctx.Params, keys.PK, tr.exploit[h]),
			encryptedOneHot(ctx.Params, keys.PK, tr.randomAction[h], frozenLakeActions),
			protocols.DefaultSPASAOptions(len(tr.randomAction[h]), []byte("train-mask-key"), []byte("train-public-seed"), fmt.Sprintf("train-epoch-%d-step-%d:agent:%d:select", epoch, trajectoryStep, h)),
		)
		actions[h] = result.Action
		stats = mergeStats(stats, result.Stats)
	}
	return actions, stats
}

func decodeEncryptedActions(params ckks.Parameters, sk *rlwe.SecretKey, actions [][]*rlwe.Ciphertext, patients int) [][]int {
	selected := make([][]int, len(actions))
	for h := range actions {
		selected[h] = make([]int, patients)
		got := make([][]float64, len(actions[h]))
		for a := range actions[h] {
			got[a] = decryptVector(params, sk, actions[h][a], patients)
		}
		for j := 0; j < patients; j++ {
			best := 0
			for a := 1; a < len(got); a++ {
				if got[a][j] > got[best][j] {
					best = a
				}
			}
			selected[h][j] = best
		}
	}
	return selected
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

func writeRewardCSV(path string, points []curvePoint, hospitals, patients, window int) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()
	header := []string{"epoch", "epsilon", "team_return", "team_return_ma", "greedy_team_return", "train_team_return"}
	for h := 0; h < hospitals; h++ {
		header = append(header, fmt.Sprintf("agent%d_return", h), fmt.Sprintf("agent%d_return_ma", h), fmt.Sprintf("agent%d_train_return", h))
	}
	for j := 0; j < patients; j++ {
		header = append(header, fmt.Sprintf("agent0_task%d_return", j), fmt.Sprintf("agent0_task%d_return_ma", j), fmt.Sprintf("agent0_task%d_train_return", j))
	}
	header = append(header, "duration_ms", "comm_mb", "spdcmp_calls", "sedtp_calls", "max_q_diff")
	if err := writer.Write(header); err != nil {
		return err
	}

	team := make([]float64, len(points))
	agents := make([][]float64, hospitals)
	tasks := make([][]float64, patients)
	for i, p := range points {
		team[i] = p.teamAvg
		for h := 0; h < hospitals; h++ {
			if agents[h] == nil {
				agents[h] = make([]float64, len(points))
			}
			agents[h][i] = p.agentAvg[h]
		}
		for j := 0; j < patients; j++ {
			if tasks[j] == nil {
				tasks[j] = make([]float64, len(points))
			}
			tasks[j][i] = p.agent0Task[j]
		}
	}

	for i, p := range points {
		row := []string{itoa(p.step), ftoa(p.epsilon), ftoa(p.teamAvg), ftoa(movingAverage(team, i, window)), ftoa(p.greedyTeamAvg), ftoa(p.trainTeamAvg)}
		for h := 0; h < hospitals; h++ {
			row = append(row, ftoa(p.agentAvg[h]), ftoa(movingAverage(agents[h], i, window)), ftoa(p.trainAgentAvg[h]))
		}
		for j := 0; j < patients; j++ {
			row = append(row, ftoa(p.agent0Task[j]), ftoa(movingAverage(tasks[j], i, window)), ftoa(p.trainAgent0Task[j]))
		}
		row = append(row, ftoa(float64(p.duration.Microseconds())/1000), ftoa(p.commMB), itoa(p.spdcmp), itoa(p.sedtp), ftoa(p.maxQDiff))
		if err := writer.Write(row); err != nil {
			return err
		}
	}
	return writer.Error()
}


func validateFlags(steps, horizon, hospitals, patients, states, actions, parties int, alpha, gamma, epsStart, epsEnd, epsDecay float64, window int) {
	if steps <= 0 || horizon <= 0 || hospitals <= 0 || patients <= 0 || states <= 0 || actions <= 1 || parties <= 0 || window <= 0 {
		panic("steps, horizon, hospitals, patients, states, parties, and window must be positive; actions must be > 1")
	}
	if states != frozenLakeStates || actions != frozenLakeActions {
		panic("medical FrozenLake uses exactly 9 states and 4 actions")
	}
	if alpha <= 0 || alpha > 1 || gamma < 0 || gamma > 1 {
		panic("alpha must be in (0,1], gamma must be in [0,1]")
	}
	if epsStart < 0 || epsStart > 1 || epsEnd < 0 || epsEnd > 1 || epsDecay <= 0 || epsDecay > 1 {
		panic("epsilon values must be probabilities and epsilon-decay must be in (0,1]")
	}
}

func agentAverages(reward [][]float64) []float64 {
	out := make([]float64, len(reward))
	for h := range reward {
		out[h] = avg(reward[h])
	}
	return out
}

func cloneQ(q [][][][]float64) [][][][]float64 {
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

func makeIntMatrix(rows, cols int) [][]int {
	out := make([][]int, rows)
	for i := range out {
		out[i] = make([]int, cols)
	}
	return out
}

func makeFloatMatrix(rows, cols int) [][]float64 {
	out := make([][]float64, rows)
	for i := range out {
		out[i] = make([]float64, cols)
	}
	return out
}

func filled(length int, value float64) []float64 {
	out := make([]float64, length)
	for i := range out {
		out[i] = value
	}
	return out
}

func activeAlpha(alpha []float64, active []bool) []float64 {
	out := make([]float64, len(alpha))
	for i := range alpha {
		if active[i] {
			out[i] = alpha[i]
		}
	}
	return out
}

func filledBool(length int, value bool) []bool {
	out := make([]bool, length)
	for i := range out {
		out[i] = value
	}
	return out
}

func anyActive(active []bool) bool {
	for _, value := range active {
		if value {
			return true
		}
	}
	return false
}

func avg(values []float64) float64 {
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	return sum / float64(len(values))
}

func movingAverage(values []float64, at, window int) float64 {
	start := at - window + 1
	if start < 0 {
		start = 0
	}
	return avg(values[start : at+1])
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
		label, stats.Duration, float64(stats.Communication.TotalBytes)/(1024*1024), stats.SPDCmpCalls, stats.SEDTPCalls)
}

func round(values []float64) []float64 {
	out := make([]float64, len(values))
	for i, value := range values {
		out[i] = math.Round(value*1000) / 1000
	}
	return out
}

func clamp01(value float64) float64 {
	return math.Max(0, math.Min(1, value))
}

func abs(value int) int {
	if value < 0 {
		return -value
	}
	return value
}

func ftoa(value float64) string {
	return strconv.FormatFloat(value, 'f', 6, 64)
}

func itoa(value int) string {
	return strconv.Itoa(value)
}
