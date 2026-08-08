// Plaintext Medical FrozenLake MARL training.
//
// This is the plaintext (non-encrypted) counterpart of cmd/marl-train.
// It uses the same environment, reward structure, hyper-parameters,
// and VDN-style Q-learning, but performs all computations in the clear
// without CKKS homomorphic encryption or threshold cryptography.
//
// Run:
//
//	go run ./cmd/marl-train-plain -steps 6 -hospitals 3 -patients 5
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
)

// ── FrozenLake geometry (3×3 grid) ──────────────────────────────────────────

const (
	frozenLakeWidth   = 3 // 3 columns
	frozenLakeStates  = 9 // 3×3 = 9 discrete states
	frozenLakeActions = 4 // up(0) right(1) down(2) left(3)
)

// ── Environment ─────────────────────────────────────────────────────────────

type medicalFrozenLakeEnv struct {
	hospitals, patients int
	starts              [][]int // per-hospital per-patient start state
	current             [][]int // per-hospital per-patient current state
	goals               []int   // per-hospital goal state
	hazards             []map[int]bool // per-hospital hazard set
}

type transition struct {
	state, nextState   [][]int
	action             [][]int
	agentReward        [][]float64
	teamReward         []float64
}

type rewardSnapshot struct {
	teamAvg    float64
	agentAvg   []float64
	agent0Task []float64
}

type curvePoint struct {
	epoch          int
	epsilon        float64
	teamAvg        float64
	greedyTeamAvg  float64
	agentAvg       []float64
	agent0Task     []float64
	duration       time.Duration
}

func main() {
	steps := flag.Int("steps", 6, "training epochs/episodes")
	horizon := flag.Int("horizon", 8, "maximum environment steps per trajectory")
	hospitals := flag.Int("hospitals", 3, "number of hospitals/agents")
	patients := flag.Int("patients", 5, "number of SIMD patient/task slots")
	alpha := flag.Float64("alpha", 0.3, "learning rate")
	gamma := flag.Float64("gamma", 0.4, "discount factor")
	epsStart := flag.Float64("epsilon-start", 0.7, "initial exploration probability")
	epsEnd := flag.Float64("epsilon-end", 0.05, "minimum exploration probability")
	epsDecay := flag.Float64("epsilon-decay", 0.82, "multiplicative epsilon decay")
	window := flag.Int("window", 5, "moving average window for CSV curves")
	seed := flag.Int64("seed", 7, "deterministic simulation seed")
	outDir := flag.String("out", "./results/frozenlake-medical-plain", "output directory for reward curves")
	flag.Parse()

	validateFlags(*steps, *horizon, *hospitals, *patients, *alpha, *gamma,
		*epsStart, *epsEnd, *epsDecay, *window)

	env := newMedicalFrozenLakeEnv(*hospitals, *patients)
	rng := rand.New(rand.NewSource(*seed))

	alphaVec := filled(env.patients, *alpha)
	gammaVec := filled(env.patients, *gamma)
	plainQ := env.initialQ()

	points := make([]curvePoint, 0, *steps)
	start := time.Now()

	fmt.Printf("Medical FrozenLake MARL (plaintext): n=%d agents, m=%d tasks, states=%d, actions=%d, epochs=%d, horizon=%d\n",
		env.hospitals, env.patients, frozenLakeStates, frozenLakeActions, *steps, *horizon)
	fmt.Printf("patient initial states per agent=%v\n", env.starts)

	for epoch := 0; epoch < *steps; epoch++ {
		epsilon := math.Max(*epsEnd, *epsStart*math.Pow(*epsDecay, float64(epoch)))
		env.resetEpisode()
		active := filledBool(env.patients, true)
		episodeAgentReward := makeFloatMatrix(env.hospitals, env.patients)
		episodeTeamReward := make([]float64, env.patients)

		executedSteps := 0
		for t := 0; t < *horizon && anyActive(active); t++ {
			// 1. Build transition: record current states + select plain ε-greedy actions.
			tr := env.buildTransition(plainQ, epsilon, rng, active)

			// 2. Step environment: next states, rewards, terminal detection.
			env.applyActions(&tr, active)

			// 3. VDN-style Q-update (plaintext).
			plainQ = vdnUpdate(plainQ, tr.state, tr.nextState, tr.action,
				tr.teamReward, alphaVec, gammaVec)

			// Accumulate episode rewards.
			for h := 0; h < env.hospitals; h++ {
				for j := 0; j < env.patients; j++ {
					episodeAgentReward[h][j] += tr.agentReward[h][j]
				}
			}
			for j := 0; j < env.patients; j++ {
				episodeTeamReward[j] += tr.teamReward[j]
			}
			executedSteps++
		}

		// Evaluate greedy policy.
		greedyEval := env.evaluateGreedyEpisode(plainQ, *horizon)

		point := curvePoint{
			epoch:         epoch,
			epsilon:       epsilon,
			teamAvg:       avg(episodeTeamReward),
			greedyTeamAvg: greedyEval.teamAvg,
			agentAvg:      agentAverages(episodeAgentReward),
			agent0Task:    append([]float64(nil), episodeAgentReward[0]...),
			duration:      0, // per-epoch timing omitted for simplicity
		}
		points = append(points, point)

		fmt.Printf("epoch=%02d eps=%.3f train_team_return=%.3f greedy_team_return=%.3f traj_steps=%d agent_returns=%v agent0_task_returns=%v\n",
			epoch, epsilon, point.teamAvg, point.greedyTeamAvg, executedSteps,
			round(point.agentAvg), round(point.agent0Task))
	}

	// Write output.
	if err := os.MkdirAll(*outDir, 0755); err != nil {
		panic(err)
	}
	csvPath := filepath.Join(*outDir, "reward_curves.csv")
	if err := writeRewardCSV(csvPath, points, env.hospitals, env.patients, *window); err != nil {
		panic(err)
	}
	fmt.Printf("\ncompleted in %s\n", time.Since(start))
	fmt.Printf("curves written:\n  %s\n", csvPath)
}

// ── Environment methods ─────────────────────────────────────────────────────

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
		env.goals[h] = 8 // bottom-right corner (state index 8)
		env.hazards[h] = map[int]bool{5: true}
		switch h % 3 {
		case 1:
			env.hazards[h] = map[int]bool{3: true}
		case 2:
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

func (e *medicalFrozenLakeEnv) resetEpisode() {
	for h := 0; h < e.hospitals; h++ {
		copy(e.current[h], e.starts[h])
	}
}

// buildTransition records current states and selects actions via plain ε-greedy.
func (e *medicalFrozenLakeEnv) buildTransition(
	q [][][][]float64, epsilon float64, rng *rand.Rand, active []bool,
) transition {
	tr := transition{
		state:       makeIntMatrix(e.hospitals, e.patients),
		action:      makeIntMatrix(e.hospitals, e.patients),
		nextState:   makeIntMatrix(e.hospitals, e.patients),
		agentReward: makeFloatMatrix(e.hospitals, e.patients),
		teamReward:  make([]float64, e.patients),
	}
	for h := 0; h < e.hospitals; h++ {
		for j := 0; j < e.patients; j++ {
			tr.state[h][j] = e.current[h][j]
			if !active[j] {
				tr.action[h][j] = 0
				continue
			}
			if rng.Float64() < epsilon {
				tr.action[h][j] = rng.Intn(frozenLakeActions)
			} else {
				tr.action[h][j] = argmaxQ(q[h][e.current[h][j]], j)
			}
		}
	}
	return tr
}

// applyActions steps the environment for one transition.
// Mutates env.current and active in place.
// Termination: hazard by ANY agent → all terminate;
// all agents at goal → all terminate (success).
// Agents already at their goal are frozen (stay, earn 0).
func (e *medicalFrozenLakeEnv) applyActions(tr *transition, active []bool) {
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

func (e *medicalFrozenLakeEnv) reward(hospital, state, next int) float64 {
	switch {
	case e.hazards[hospital][next]:
		return -10 // fell into hazard — strong penalty — no penalty (terminal signal comes from isTerminal)
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

func (e *medicalFrozenLakeEnv) stepState(state, action int) int {
	row, col := state/frozenLakeWidth, state%frozenLakeWidth
	switch action {
	case 0: // up
		row--
	case 1: // right
		col++
	case 2: // down
		row++
	case 3: // left
		col--
	}
	if row < 0 || row >= frozenLakeWidth || col < 0 || col >= frozenLakeWidth {
		return state // wall bounce
	}
	return row*frozenLakeWidth + col
}

func (e *medicalFrozenLakeEnv) distance(a, b int) int {
	ar, ac := a/frozenLakeWidth, a%frozenLakeWidth
	br, bc := b/frozenLakeWidth, b%frozenLakeWidth
	return abs(ar-br) + abs(ac-bc)
}

func (e *medicalFrozenLakeEnv) isHazard(hospital, state int) bool {
	return e.hazards[hospital][state]
}

func (e *medicalFrozenLakeEnv) isGoal(hospital, state int) bool {
	return state == e.goals[hospital]
}

// isTerminal: true if hazard (for priorQ zeroing).
func (e *medicalFrozenLakeEnv) isTerminal(hospital, state int) bool {
	return e.hazards[hospital][state]
}

// Grid layout (3×3):
//
//	0  1  2
//	3  4  5
//	6  7  8
//
// State index = row * 3 + col.

// evaluateGreedyEpisode runs a full greedy (ε=0) episode and returns reward stats.
// Termination: hazard→stop, all-goal→stop, frozen-at-goal agents earn 0.
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
				action := argmaxQ(q[h][state[h][j]], j)
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

// ── VDN-style Q-update (plaintext) ──────────────────────────────────────────

// vdnUpdate performs one step of VDN-style Q-learning.
//
//	Q_tot(s, a)    = Σ_h Q_h(s_h, a_h)           [current estimate]
//	max Q_tot(s')  = Σ_h max_{a'} Q_h(s'_h, a')  [next-state max per agent, then sum]
//	δ = r + γ · max Q_tot(s') − Q_tot(s, a)
//	Q_h(s_h, a_h) += α · δ                         [same δ for all agents]
func vdnUpdate(
	q [][][][]float64,
	state, nextState, action [][]int,
	reward, alpha, gamma []float64,
) [][][][]float64 {
	out := cloneQ(q)
	hospitals := len(q)
	patients := len(reward)

	currentQ := make([]float64, patients)
	nextMax := make([]float64, patients)
	td := make([]float64, patients)

	for h := 0; h < hospitals; h++ {
		for j := 0; j < patients; j++ {
			currentQ[j] += q[h][state[h][j]][action[h][j]][j]
			nextMax[j] += q[h][nextState[h][j]][argmaxQ(q[h][nextState[h][j]], j)][j]
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
	return out
}

// ── Action selection helpers ────────────────────────────────────────────────

// argmaxQ returns the action index with the highest Q-value for patient j.
// Ties are broken to the right (higher action index).
func argmaxQ(row [][]float64, patient int) int {
	best := 0
	for a := 1; a < len(row); a++ {
		if row[a][patient] >= row[best][patient] {
			best = a
		}
	}
	return best
}

// ── CSV output ──────────────────────────────────────────────────────────────

func writeRewardCSV(path string, points []curvePoint, hospitals, patients, window int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	header := []string{"epoch", "epsilon", "team_return", "team_return_ma", "greedy_team_return"}
	for h := 0; h < hospitals; h++ {
		header = append(header,
			fmt.Sprintf("agent%d_return", h),
			fmt.Sprintf("agent%d_return_ma", h))
	}
	for j := 0; j < patients; j++ {
		header = append(header,
			fmt.Sprintf("agent0_task%d_return", j),
			fmt.Sprintf("agent0_task%d_return_ma", j))
	}
	header = append(header, "duration_ms")
	if err := w.Write(header); err != nil {
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
		row := []string{
			itoa(p.epoch), ftoa(p.epsilon),
			ftoa(p.teamAvg), ftoa(movingAverage(team, i, window)),
			ftoa(p.greedyTeamAvg),
		}
		for h := 0; h < hospitals; h++ {
			row = append(row, ftoa(p.agentAvg[h]), ftoa(movingAverage(agents[h], i, window)))
		}
		for j := 0; j < patients; j++ {
			row = append(row, ftoa(p.agent0Task[j]), ftoa(movingAverage(tasks[j], i, window)))
		}
		row = append(row, ftoa(float64(p.duration.Microseconds())/1000))
		if err := w.Write(row); err != nil {
			return err
		}
	}
	return w.Error()
}

// ── Validation ──────────────────────────────────────────────────────────────

func validateFlags(steps, horizon, hospitals, patients int,
	alpha, gamma, epsStart, epsEnd, epsDecay float64, window int) {
	if steps <= 0 || horizon <= 0 || hospitals <= 0 || patients <= 0 || window <= 0 {
		panic("steps, horizon, hospitals, patients, and window must be positive")
	}
	if alpha <= 0 || alpha > 1 || gamma < 0 || gamma > 1 {
		panic("alpha must be in (0,1], gamma must be in [0,1]")
	}
	if epsStart < 0 || epsStart > 1 || epsEnd < 0 || epsEnd > 1 || epsDecay <= 0 || epsDecay > 1 {
		panic("epsilon values must be probabilities and epsilon-decay must be in (0,1]")
	}
}

// ── Generic helpers ─────────────────────────────────────────────────────────

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

func filled(length int, value float64) []float64 {
	out := make([]float64, length)
	for i := range out {
		out[i] = value
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
	for _, v := range active {
		if v {
			return true
		}
	}
	return false
}

func agentAverages(reward [][]float64) []float64 {
	out := make([]float64, len(reward))
	for h := range reward {
		out[h] = avg(reward[h])
	}
	return out
}

func avg(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
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

func round(values []float64) []float64 {
	out := make([]float64, len(values))
	for i, v := range values {
		out[i] = math.Round(v*1000) / 1000
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
