// ==========================================================================
// AURA-Cache: Research Suite & Interactive Simulator Engine
// ==========================================================================

// --- Cache & Simulation State ---
const NUM_SETS = 4;
const NUM_WAYS = 16;
const BLOCK_SIZE = 64; // Bytes

let cache = []; // Set array -> Array of 16 Ways
let secondaryLruCache = [];
let secondaryRandomCache = [];

let registers = new Array(16).fill(0);
let instructionQueue = [];
let executionHistory = [];
let currentStepIndex = -1;

// Metrics per policy
let stats = {
    lru: { accesses: 0, hits: 0, misses: 0 },
    phase_aware: { accesses: 0, hits: 0, misses: 0 },
    random: { accesses: 0, hits: 0, misses: 0 }
};

// Rolling telemetry window (last 50 accesses)
let accessWindow = [];
const WINDOW_SIZE = 50;

// Auto-play timer
let playInterval = null;

// Chart references
let comparisonCanvas, comparisonCtx;
let timelineCanvas, timelineCtx;
let timelineHistory = [];

// Guided Tour State
const TOUR_STEPS = [
    {
        section: 'sec-simulator',
        title: 'Hardware Simulation Lab',
        badge: 'Step 1 of 6',
        desc: 'Execute cycle-by-cycle memory accesses. Inspect CPU registers, 16-way cache sets, and live neural Q-inferences.'
    },
    {
        section: 'sec-motivation',
        title: 'The Problem: LRU Thrashing',
        badge: 'Step 2 of 6',
        desc: 'Understand why traditional recency heuristics fail during loops and streaming scans, leading to near 100% miss rates.'
    },
    {
        section: 'sec-phases',
        title: 'Workload Phase Discovery',
        badge: 'Step 3 of 6',
        desc: 'Explore the 4 macroscopic execution phases identified by online K-Means clustering and sliding telemetry windows.'
    },
    {
        section: 'sec-brain',
        title: 'Deep Q-Network Decision Brain',
        badge: 'Step 4 of 6',
        desc: 'Inspect the 28-dimensional state tensor and multi-layer perceptron topology that predicts optimal Belady victim lines.'
    },
    {
        section: 'sec-evaluation',
        title: 'Empirical Review Gallery',
        badge: 'Step 5 of 6',
        desc: 'Browse the 9 publication-grade figures showing +14.2% Belady match accuracy, phase ablation proof, and heatmaps.'
    },
    {
        section: 'sec-deployment',
        title: 'ChampSim C++ & Hardware Overhead',
        badge: 'Step 6 of 6',
        desc: 'Review hardware feasibility: 4.35 KB quantized footprint (<3.4% tag array area) and drop-in C++ ChampSim integration.'
    }
];
let currentTourIndex = 0;

// Code Snippets for Deployment Section
const CODE_SNIPPETS = {
    cpp: `// replacement/phase_aware_rl/phase_aware_rl.cc
long phase_aware_rl::find_victim(uint32_t cpu, uint64_t instr_id, long set, 
                                 const BLOCK* current_set, uint64_t ip, 
                                 uint64_t full_addr, uint32_t type)
{
    // 1. Extract online telemetry and predict phase cluster
    auto [miss_rate, write_ratio, access_rate] = get_phase_features();
    int cluster_id = predict_cluster(miss_rate, write_ratio, access_rate);

    // 2. Formulate 28-dimensional state vector (16 recency + 8 type + 4 cluster)
    std::vector<float> state = build_state_vector(set, type, cluster_id);

    // 3. Fast feedforward neural pass using on-chip weights
    std::vector<float> q_values = evaluate_dqn(state);

    // 4. Return cache way index with maximum Q-value
    return std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end()));
}`,
    json: `{
  "executable_name": "champsim",
  "block_size": 64,
  "page_size": 4096,
  "heartbeat_frequency": 10000000,
  "num_cores": 1,
  "LLC": {
    "frequency": 4000,
    "sets": 2048,
    "ways": 16,
    "rq_size": 32,
    "wq_size": 32,
    "pq_size": 32,
    "mshr_size": 64,
    "latency": 20,
    "max_tag_check": 4,
    "max_fill": 2,
    "prefetcher": "no",
    "replacement": "phase_aware_rl"
  }
}`,
    bash: `# 1. Configure ChampSim with Phase-Aware RL replacement
./config.sh champsim_config.json

# 2. Build the optimized simulator binary
make -j4

# 3. Execute against SPEC CPU benchmark trace
bin/champsim --warmup_instructions 1000000 \\
             --simulation_instructions 10000000 \\
             traces/429.mcf-22B.champsimtrace.xz`
};

// ==========================================================================
// Initialization Functions
// ==========================================================================

function initCache() {
    cache = [];
    for (let s = 0; s < NUM_SETS; s++) {
        let setWays = [];
        for (let w = 0; w < NUM_WAYS; w++) {
            setWays.push({
                wayIndex: w,
                valid: 0,
                tag: '0x000',
                address: 0,
                recency: w, // 0 = MRU, 15 = LRU initially
                accessCount: 0
            });
        }
        cache.push(setWays);
    }
}

function initSecondaryCaches() {
    secondaryLruCache = [];
    secondaryRandomCache = [];
    for (let s = 0; s < NUM_SETS; s++) {
        let setWaysLru = [];
        let setWaysRand = [];
        for (let w = 0; w < NUM_WAYS; w++) {
            setWaysLru.push({ wayIndex: w, valid: 0, tag: '0x000', recency: w });
            setWaysRand.push({ wayIndex: w, valid: 0, tag: '0x000' });
        }
        secondaryLruCache.push(setWaysLru);
        secondaryRandomCache.push(setWaysRand);
    }
}

function initRegisters() {
    const bank = document.getElementById('registers-bank');
    if (!bank) return;
    bank.innerHTML = '';
    for (let i = 0; i < 16; i++) {
        const chip = document.createElement('div');
        chip.className = 'reg-chip';
        chip.id = `reg-chip-${i}`;
        chip.innerHTML = `
            <span class="reg-id">g${i}</span>
            <span class="reg-val" id="reg-val-${i}">0000</span>
        `;
        bank.appendChild(chip);
    }
}

function initRecencyBars() {
    const grid = document.getElementById('nn-recency-bars');
    if (!grid) return;
    grid.innerHTML = '';
    for (let i = 0; i < 16; i++) {
        const bar = document.createElement('div');
        bar.className = 'r-bar-col';
        bar.id = `r-bar-${i}`;
        bar.style.height = `${Math.max(10, (15 - i) * 6)}%`;
        if (i === 0) bar.classList.add('mru');
        if (i === 15) bar.classList.add('lru');
        grid.appendChild(bar);
    }
}

function renderCache() {
    const container = document.getElementById('cache-visualization');
    if (!container) return;
    container.innerHTML = '';

    for (let s = 0; s < NUM_SETS; s++) {
        const setBlock = document.createElement('div');
        setBlock.className = 'cache-set-block';
        setBlock.id = `cache-set-${s}`;

        const setHead = document.createElement('div');
        setHead.className = 'set-header';
        setHead.innerHTML = `
            <span class="set-tag">SET 0x${s.toString(16).toUpperCase()} (Index ${s})</span>
            <span class="set-meta">16-Way Associative · 64B Line</span>
        `;
        setBlock.appendChild(setHead);

        const waysGrid = document.createElement('div');
        waysGrid.className = 'ways-grid';

        for (let w = 0; w < NUM_WAYS; w++) {
            const way = cache[s][w];
            const chip = document.createElement('div');
            chip.className = `way-chip ${way.recency === 0 ? 'mru-line' : ''}`;
            chip.id = `way-chip-${s}-${w}`;

            // Recency percentage (0=MRU full bar, 15=LRU small bar)
            const recencyPct = Math.max(10, ((15 - way.recency) / 15) * 100);

            chip.innerHTML = `
                <span class="w-id">W${w}</span>
                <span class="w-tag">${way.valid ? way.tag : 'EMPTY'}</span>
                <div class="recency-track">
                    <div class="recency-bar" style="width: ${way.valid ? recencyPct : 0}%"></div>
                </div>
            `;
            waysGrid.appendChild(chip);
        }
        setBlock.appendChild(waysGrid);
        container.appendChild(setBlock);
    }
}

// ==========================================================================
// Address Parsing & Workload Presets
// ==========================================================================

function parseAddress(addrHex) {
    const addr = parseInt(addrHex, 16);
    if (isNaN(addr)) return null;

    const blockAddr = Math.floor(addr / BLOCK_SIZE);
    const setIndex = blockAddr % NUM_SETS;
    const tag = '0x' + Math.floor(blockAddr / NUM_SETS).toString(16).toUpperCase();

    return { addr, setIndex, tag, blockAddr };
}

const PRESETS = {
    sequential: Array.from({ length: 32 }, (_, i) => `0x${(i * 64).toString(16)}`),
    stride: Array.from({ length: 32 }, (_, i) => `0x${(i * 256).toString(16)}`),
    loop: (() => {
        // 18 distinct block addresses into 4 sets (triggers heavy thrashing in targeted sets)
        const baseAddresses = Array.from({ length: 18 }, (_, i) => i * 256);
        let trace = [];
        for (let r = 0; r < 4; r++) {
            baseAddresses.forEach(addr => trace.push(`0x${addr.toString(16)}`));
        }
        return trace;
    })(),
    'phase-change': (() => {
        let trace = [];
        // Phase 1: Sequential high-locality streaming
        for (let i = 0; i < 18; i++) trace.push(`0x${(i * 64).toString(16)}`);
        // Phase 2: Pointer chasing / scattered addresses
        for (let i = 0; i < 20; i++) trace.push(`0x${(0x10000 + ((i * 137) % 64) * 256).toString(16)}`);
        // Phase 3: Tight iterative kernel loop
        const kernel = [0x2000, 0x2040, 0x2080, 0x20C0, 0x2100, 0x2140];
        for (let r = 0; r < 4; r++) {
            kernel.forEach(addr => trace.push(`0x${addr.toString(16)}`));
        }
        return trace;
    })()
};

function loadPreset(presetName) {
    const trace = PRESETS[presetName];
    if (trace) {
        document.getElementById('trace-input').value = trace.join('\n');
    }
}

// ==========================================================================
// Neural Decision Engine & K-Means Phase Inference
// ==========================================================================

function getPhaseFeatures() {
    const sz = accessWindow.length;
    if (sz === 0) return { missRate: 0, writeFrac: 0, accessRate: 0 };

    const misses = accessWindow.filter(a => !a.hit).length;
    const writes = accessWindow.filter(a => a.type === 'WRITE').length;

    const missRate = misses / sz;
    const writeFrac = writes / sz;
    const accessRate = sz / (sz * 100);

    return { missRate, writeFrac, accessRate };
}

function predictPhaseCluster(missRate, writeFrac, accessRate) {
    if (!window.KMEANS_MODEL || !KMEANS_MODEL.scaler_mean) return 0;

    const reuseFrac = 0.05;
    const raw = [missRate, writeFrac, reuseFrac, accessRate];
    const scaled = new Array(4);

    for (let i = 0; i < 4; i++) {
        const mean = KMEANS_MODEL.scaler_mean[i];
        const scale = KMEANS_MODEL.scaler_scale[i] > 1e-6 ? KMEANS_MODEL.scaler_scale[i] : 1.0;
        scaled[i] = (raw[i] - mean) / scale;
    }

    let bestCluster = 0;
    let minDist = Infinity;

    for (let c = 0; c < KMEANS_MODEL.cluster_centers.length; c++) {
        let dist = 0;
        for (let i = 0; i < 4; i++) {
            let diff = scaled[i] - KMEANS_MODEL.cluster_centers[c][i];
            dist += diff * diff;
        }
        if (dist < minDist) {
            minDist = dist;
            bestCluster = c;
        }
    }
    return bestCluster;
}

function forwardDQN(inputVector) {
    if (!window.DQN_WEIGHTS || DQN_WEIGHTS.length === 0) {
        return Array.from({ length: 16 }, () => Math.random());
    }

    let activations = [...inputVector];

    function evaluateLayer(input, layer, relu = true) {
        const outDim = layer.bias.length;
        const inDim = input.length;
        const output = new Array(outDim).fill(0);

        for (let i = 0; i < outDim; i++) {
            let val = layer.bias[i];
            for (let j = 0; j < inDim; j++) {
                val += layer.weight[i][j] * input[j];
            }
            if (relu && val < 0) val = 0;
            output[i] = val;
        }
        return output;
    }

    for (let i = 0; i < DQN_WEIGHTS.length; i++) {
        const isLast = (i === DQN_WEIGHTS.length - 1);
        activations = evaluateLayer(activations, DQN_WEIGHTS[i], !isLast);
    }
    return activations;
}

function chooseVictimWay(setIndex, policy, accessTypeVal) {
    const ways = cache[setIndex];

    if (policy === 'random') {
        return Math.floor(Math.random() * NUM_WAYS);
    }

    if (policy === 'lru') {
        let lruWay = 0;
        let maxRecency = -1;
        for (let w = 0; w < NUM_WAYS; w++) {
            if (ways[w].recency > maxRecency) {
                maxRecency = ways[w].recency;
                lruWay = w;
            }
        }
        return lruWay;
    }

    if (policy === 'phase_aware') {
        const { missRate, writeFrac, accessRate } = getPhaseFeatures();
        const clusterId = predictPhaseCluster(missRate, writeFrac, accessRate);

        // Update Phase Pill in Header
        const phasePill = document.getElementById('phase-display');
        phasePill.textContent = `P${clusterId}`;
        const phaseColors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'];
        phasePill.style.color = phaseColors[clusterId] || '#a855f7';

        // 1. Build Recency feature (16 dims)
        const recencyNorm = new Array(NUM_WAYS);
        for (let w = 0; w < NUM_WAYS; w++) {
            recencyNorm[w] = ways[w].recency / (NUM_WAYS - 1);
        }

        // 2. Build Type feature (8 dims)
        const typeOneHot = new Array(8).fill(0);
        const typeIdx = (accessTypeVal === 'WRITE') ? 1 : 0;
        typeOneHot[typeIdx] = 1;

        // 3. Build Cluster One-Hot (4 dims)
        const clusterOneHot = new Array(4).fill(0);
        clusterOneHot[clusterId] = 1;

        // 28-dimensional input vector
        const stateVector = [...recencyNorm, ...typeOneHot, ...clusterOneHot];

        // Update Neural Deck in UI
        updateNeuralUI(recencyNorm, typeIdx, clusterId, clusterOneHot);

        // Run neural forward pass
        const qValues = forwardDQN(stateVector);

        // Victim is way with maximum Q-value
        let victimWay = 0;
        let maxQ = -Infinity;
        for (let w = 0; w < NUM_WAYS; w++) {
            if (qValues[w] > maxQ) {
                maxQ = qValues[w];
                victimWay = w;
            }
        }

        // Update UI decision box
        const targetDisplay = document.getElementById('target-way-display');
        if (targetDisplay) targetDisplay.textContent = `Way ${victimWay} (Q: ${maxQ.toFixed(3)})`;

        return victimWay;
    }

    return 0;
}

function updateNeuralUI(recencyNorm, typeIdx, clusterId, clusterOneHot) {
    // Update recency bars
    for (let w = 0; w < NUM_WAYS; w++) {
        const bar = document.getElementById(`r-bar-${w}`);
        if (bar) {
            bar.style.height = `${Math.max(10, recencyNorm[w] * 100)}%`;
        }
    }

    // Update Type pill
    const typePill = document.getElementById('nn-type-pill');
    if (typePill) typePill.textContent = (typeIdx === 1) ? 'WRITE (0x1)' : 'LOAD (0x0)';

    // Update Cluster pill
    const clusterPill = document.getElementById('nn-phase-pill');
    if (clusterPill) clusterPill.textContent = `[${clusterOneHot.join(', ')}]`;

    const clusterTag = document.getElementById('nn-cluster-tag');
    if (clusterTag) clusterTag.textContent = `Phase P${clusterId} Active`;
}

// ==========================================================================
// Simulation Cycle Execution
// ==========================================================================

function loadTraceAndInitialize() {
    const rawText = document.getElementById('trace-input').value;
    const lines = rawText.split('\n').map(l => l.trim()).filter(l => l.length > 0);

    instructionQueue = lines;
    currentStepIndex = -1;
    executionHistory = [];
    timelineHistory = [];

    initCache();
    initSecondaryCaches();
    initRegisters();
    initRecencyBars();

    stats = {
        lru: { accesses: 0, hits: 0, misses: 0 },
        phase_aware: { accesses: 0, hits: 0, misses: 0 },
        random: { accesses: 0, hits: 0, misses: 0 }
    };
    accessWindow = [];

    renderCache();
    renderStats(getActivePolicy());
    drawCharts();
    updateStepDisplay();

    appendLog('Trace loaded into memory queue. Initialized 4-Set 16-Way Cache.', 'info');
}

function getActivePolicy() {
    const checked = document.querySelector('input[name="repl-policy"]:checked');
    return checked ? checked.value : 'phase_aware';
}

function stepForward() {
    if (currentStepIndex + 1 >= instructionQueue.length) {
        if (playInterval) togglePlay();
        appendLog('Reached end of execution trace queue.', 'info');
        return;
    }

    currentStepIndex++;
    const addrHex = instructionQueue[currentStepIndex];
    const parsed = parseAddress(addrHex);

    if (!parsed) {
        appendLog(`Invalid address format: ${addrHex}`, 'miss');
        return;
    }

    const { addr, setIndex, tag } = parsed;
    const activePolicy = getActivePolicy();
    const accessType = (currentStepIndex % 4 === 0) ? 'WRITE' : 'READ';

    // Update CPU Registers
    const regIdx = currentStepIndex % 16;
    registers[regIdx] = addr & 0xFFFF;
    const regValElem = document.getElementById(`reg-val-${regIdx}`);
    const regChipElem = document.getElementById(`reg-chip-${regIdx}`);
    if (regValElem) regValElem.textContent = registers[regIdx].toString(16).toUpperCase().padStart(4, '0');
    if (regChipElem) {
        regChipElem.classList.add('active-write');
        setTimeout(() => regChipElem.classList.remove('active-write'), 300);
    }

    // Update Program Counter
    const pcElem = document.getElementById('pc-display');
    if (pcElem) pcElem.textContent = `0x${((currentStepIndex + 1) * 4).toString(16).toUpperCase().padStart(2, '0')}`;

    // Execute on Primary Active Policy Cache
    const ways = cache[setIndex];
    let isHit = false;
    let hitWayIdx = -1;
    let emptyWayIdx = -1;

    for (let w = 0; w < NUM_WAYS; w++) {
        if (ways[w].valid && ways[w].tag === tag) {
            isHit = true;
            hitWayIdx = w;
            break;
        }
        if (!ways[w].valid && emptyWayIdx === -1) {
            emptyWayIdx = w;
        }
    }

    let evictedWayIdx = -1;

    if (isHit) {
        stats[activePolicy].hits++;
        updateRecencyOnHit(setIndex, hitWayIdx);
        appendLog(`[${activePolicy.toUpperCase()}] HIT on Set 0x${setIndex.toString(16)} / Way ${hitWayIdx} (Tag: ${tag})`, 'hit');
    } else {
        stats[activePolicy].misses++;
        if (emptyWayIdx !== -1) {
            ways[emptyWayIdx].valid = 1;
            ways[emptyWayIdx].tag = tag;
            ways[emptyWayIdx].address = addr;
            updateRecencyOnInsert(setIndex, emptyWayIdx);
            appendLog(`[${activePolicy.toUpperCase()}] MISS (Compulsory). Filled Set 0x${setIndex.toString(16)} Way ${emptyWayIdx}`, 'miss');
        } else {
            evictedWayIdx = chooseVictimWay(setIndex, activePolicy, accessType);
            const oldTag = ways[evictedWayIdx].tag;
            ways[evictedWayIdx].tag = tag;
            ways[evictedWayIdx].address = addr;
            updateRecencyOnInsert(setIndex, evictedWayIdx);
            appendLog(`[${activePolicy.toUpperCase()}] MISS & EVICT! Replaced Way ${evictedWayIdx} (Evicted: ${oldTag}, New: ${tag})`, 'miss');
        }
    }
    stats[activePolicy].accesses++;

    // Track telemetry window
    accessWindow.push({ hit: isHit, type: accessType });
    if (accessWindow.length > WINDOW_SIZE) accessWindow.shift();

    // Run parallel baseline simulations
    simulateSecondaryLRU(setIndex, tag);
    simulateSecondaryRandom(setIndex, tag);

    // Save Timeline History for charts
    const { missRate } = getPhaseFeatures();
    const clusterId = predictPhaseCluster(missRate, 0.2, 0.05);
    timelineHistory.push({ missRate, cluster: clusterId });
    if (timelineHistory.length > 30) timelineHistory.shift();

    // Update UI elements
    renderCache();
    highlightCacheChip(setIndex, hitWayIdx, evictedWayIdx, isHit);
    renderStats(activePolicy);
    drawCharts();
    updateStepDisplay();
}

function updateRecencyOnHit(setIndex, hitWay) {
    const ways = cache[setIndex];
    const oldRecency = ways[hitWay].recency;
    for (let w = 0; w < NUM_WAYS; w++) {
        if (ways[w].recency < oldRecency) {
            ways[w].recency++;
        }
    }
    ways[hitWay].recency = 0; // MRU
}

function updateRecencyOnInsert(setIndex, insertedWay) {
    const ways = cache[setIndex];
    for (let w = 0; w < NUM_WAYS; w++) {
        if (w !== insertedWay) {
            ways[w].recency++;
        }
    }
    ways[insertedWay].recency = 0; // MRU
}

function highlightCacheChip(setIdx, hitWayIdx, evictedWayIdx, isHit) {
    if (isHit && hitWayIdx !== -1) {
        const chip = document.getElementById(`way-chip-${setIdx}-${hitWayIdx}`);
        if (chip) {
            chip.classList.add('hit');
            setTimeout(() => chip.classList.remove('hit'), 500);
        }
    } else if (evictedWayIdx !== -1) {
        const chip = document.getElementById(`way-chip-${setIdx}-${evictedWayIdx}`);
        if (chip) {
            chip.classList.add('victim');
            setTimeout(() => chip.classList.remove('victim'), 600);
        }
    }
}

// Background parallel simulation for LRU and Random
function simulateSecondaryLRU(setIndex, tag) {
    const ways = secondaryLruCache[setIndex];
    let isHit = false;
    let hitWay = -1;
    let emptyWay = -1;

    for (let w = 0; w < NUM_WAYS; w++) {
        if (ways[w].valid && ways[w].tag === tag) {
            isHit = true;
            hitWay = w;
            break;
        }
        if (!ways[w].valid && emptyWay === -1) emptyWay = w;
    }

    if (isHit) {
        stats.lru.hits++;
        const oldR = ways[hitWay].recency;
        for (let w = 0; w < NUM_WAYS; w++) {
            if (ways[w].recency < oldR) ways[w].recency++;
        }
        ways[hitWay].recency = 0;
    } else {
        stats.lru.misses++;
        let targetWay = emptyWay;
        if (targetWay === -1) {
            let maxR = -1;
            for (let w = 0; w < NUM_WAYS; w++) {
                if (ways[w].recency > maxR) {
                    maxR = ways[w].recency;
                    targetWay = w;
                }
            }
        }
        ways[targetWay].valid = 1;
        ways[targetWay].tag = tag;
        for (let w = 0; w < NUM_WAYS; w++) {
            if (w !== targetWay) ways[w].recency++;
        }
        ways[targetWay].recency = 0;
    }
    stats.lru.accesses++;
}

function simulateSecondaryRandom(setIndex, tag) {
    const ways = secondaryRandomCache[setIndex];
    let isHit = false;
    let emptyWay = -1;

    for (let w = 0; w < NUM_WAYS; w++) {
        if (ways[w].valid && ways[w].tag === tag) {
            isHit = true;
            break;
        }
        if (!ways[w].valid && emptyWay === -1) emptyWay = w;
    }

    if (isHit) {
        stats.random.hits++;
    } else {
        stats.random.misses++;
        const target = (emptyWay !== -1) ? emptyWay : Math.floor(Math.random() * NUM_WAYS);
        ways[target].valid = 1;
        ways[target].tag = tag;
    }
    stats.random.accesses++;
}

function stepBackward() {
    if (currentStepIndex <= 0) return;
    const targetIndex = currentStepIndex - 1;
    loadTraceAndInitialize();
    for (let i = 0; i <= targetIndex; i++) {
        stepForward();
    }
}

function togglePlay() {
    const btn = document.getElementById('btn-play');
    const playText = document.getElementById('play-text');
    const playIcon = document.getElementById('play-icon');

    if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
        if (playText) playText.textContent = 'Auto Play';
        if (playIcon) playIcon.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
        btn.classList.remove('btn-primary');
    } else {
        const speed = parseInt(document.getElementById('speed-slider').value);
        playInterval = setInterval(stepForward, speed);
        if (playText) playText.textContent = 'Pause';
        if (playIcon) playIcon.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
        btn.classList.add('btn-primary');
    }
}

function updateStepDisplay() {
    const display = document.getElementById('step-display');
    if (display) {
        display.textContent = `${currentStepIndex + 1} / ${instructionQueue.length}`;
    }
    const prevBtn = document.getElementById('btn-prev');
    if (prevBtn) prevBtn.disabled = (currentStepIndex <= 0);
}

function renderStats(policy) {
    const polStats = stats[policy] || { hits: 0, misses: 0, accesses: 0 };
    const hit = polStats.hits;
    const miss = polStats.misses;
    const total = polStats.accesses;
    const rate = total > 0 ? (hit / total * 100).toFixed(1) : '0.0';

    const accElem = document.getElementById('stat-accesses');
    const hitsElem = document.getElementById('stat-hits');
    const missElem = document.getElementById('stat-misses');
    const rateElem = document.getElementById('stat-hitrate');

    if (accElem) accElem.textContent = total;
    if (hitsElem) hitsElem.textContent = hit;
    if (missElem) missElem.textContent = miss;
    if (rateElem) rateElem.textContent = `${rate}%`;

    const { missRate, writeFrac, accessRate } = getPhaseFeatures();
    const rMiss = document.getElementById('stat-rolling-miss');
    const wFrac = document.getElementById('stat-write-frac');
    const aRate = document.getElementById('stat-access-rate');

    if (rMiss) rMiss.textContent = missRate.toFixed(2);
    if (wFrac) wFrac.textContent = writeFrac.toFixed(2);
    if (aRate) aRate.textContent = accessRate.toFixed(4);
}

function appendLog(message, type = 'info') {
    const logBox = document.getElementById('execution-log');
    if (!logBox) return;
    const row = document.createElement('div');
    row.className = `log-row ${type}-row`;
    row.textContent = `[Cycle ${currentStepIndex + 1}] ${message}`;
    logBox.appendChild(row);
    logBox.scrollTop = logBox.scrollHeight;
}

// ==========================================================================
// Chart Rendering (Canvas)
// ==========================================================================

function initCharts() {
    comparisonCanvas = document.getElementById('comparison-chart');
    if (comparisonCanvas) comparisonCtx = comparisonCanvas.getContext('2d');

    timelineCanvas = document.getElementById('phase-timeline-chart');
    if (timelineCanvas) timelineCtx = timelineCanvas.getContext('2d');

    drawCharts();
}

function drawCharts() {
    if (!comparisonCtx || !timelineCtx) return;

    // 1. Comparison Bar Chart
    const cW = comparisonCanvas.width;
    const cH = comparisonCanvas.height;
    comparisonCtx.clearRect(0, 0, cW, cH);

    const policies = ['Phase DQN', 'Base LRU', 'Random'];
    const misses = [stats.phase_aware.misses, stats.lru.misses, stats.random.misses];
    const maxM = Math.max(...misses, 4);
    const colors = ['#2563eb', '#64748b', '#ef4444'];

    for (let i = 0; i < 3; i++) {
        const barHeight = (misses[i] / maxM) * (cH - 45);
        const x = 25 + i * 85;
        const y = cH - 20 - barHeight;

        comparisonCtx.fillStyle = colors[i];
        comparisonCtx.fillRect(x, y, 55, barHeight);

        comparisonCtx.fillStyle = '#0f172a';
        comparisonCtx.font = 'bold 10px JetBrains Mono';
        comparisonCtx.textAlign = 'center';
        comparisonCtx.fillText(misses[i], x + 27, y - 4);

        comparisonCtx.fillStyle = '#475569';
        comparisonCtx.font = '9px Space Grotesk';
        comparisonCtx.fillText(policies[i], x + 27, cH - 6);
    }

    // 2. Timeline Chart
    const tW = timelineCanvas.width;
    const tH = timelineCanvas.height;
    timelineCtx.clearRect(0, 0, tW, tH);

    if (timelineHistory.length > 1) {
        const stepX = tW / (timelineHistory.length - 1);
        timelineCtx.beginPath();
        timelineCtx.strokeStyle = '#2563eb';
        timelineCtx.lineWidth = 2;

        for (let i = 0; i < timelineHistory.length; i++) {
            const x = i * stepX;
            const y = tH - 10 - (timelineHistory[i].missRate * (tH - 25));
            if (i === 0) timelineCtx.moveTo(x, y);
            else timelineCtx.lineTo(x, y);
        }
        timelineCtx.stroke();

        // Draw points with phase color
        const phaseColors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'];
        for (let i = 0; i < timelineHistory.length; i++) {
            const x = i * stepX;
            const y = tH - 10 - (timelineHistory[i].missRate * (tH - 25));
            timelineCtx.beginPath();
            timelineCtx.arc(x, y, 3, 0, 2 * Math.PI);
            timelineCtx.fillStyle = phaseColors[timelineHistory[i].cluster] || '#00f2fe';
            timelineCtx.fill();
        }
    } else {
        timelineCtx.fillStyle = '#64748b';
        timelineCtx.font = '10px Space Grotesk';
        timelineCtx.textAlign = 'center';
        timelineCtx.fillText('Execute steps to populate timeline', tW / 2, tH / 2);
    }
}

// ==========================================================================
// Guided Tour Assistant Logic
// ==========================================================================

function startTour() {
    currentTourIndex = 0;
    const dock = document.getElementById('tour-dock');
    if (dock) dock.style.display = 'block';
    renderTourStep();
}

function renderTourStep() {
    const step = TOUR_STEPS[currentTourIndex];
    document.getElementById('tour-step-label').textContent = step.badge;
    document.getElementById('tour-title').textContent = step.title;
    document.getElementById('tour-desc').textContent = step.desc;

    // Switch section tab automatically
    switchJourneySection(step.section);

    document.getElementById('btn-tour-prev').disabled = (currentTourIndex === 0);
    const nextBtn = document.getElementById('btn-tour-next');
    if (currentTourIndex === TOUR_STEPS.length - 1) {
        nextBtn.textContent = 'Finish Tour';
    } else {
        nextBtn.textContent = 'Next Step';
    }
}

function nextTourStep() {
    if (currentTourIndex < TOUR_STEPS.length - 1) {
        currentTourIndex++;
        renderTourStep();
    } else {
        closeTour();
    }
}

function prevTourStep() {
    if (currentTourIndex > 0) {
        currentTourIndex--;
        renderTourStep();
    }
}

function closeTour() {
    const dock = document.getElementById('tour-dock');
    if (dock) dock.style.display = 'none';
}

function switchJourneySection(targetId) {
    // Update tabs
    document.querySelectorAll('.journey-tab').forEach(tab => {
        if (tab.getAttribute('data-target') === targetId) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });

    // Update sections
    document.querySelectorAll('.journey-section').forEach(sec => {
        if (sec.id === targetId) {
            sec.classList.add('active');
        } else {
            sec.classList.remove('active');
        }
    });

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ==========================================================================
// Lightbox Modal for Review Figures
// ==========================================================================

function openLightbox(imgSrc, title, desc) {
    const modal = document.getElementById('lightbox-modal');
    const img = document.getElementById('lightbox-img');
    const titleElem = document.getElementById('lightbox-title');
    const descElem = document.getElementById('lightbox-desc');

    if (modal && img) {
        img.src = imgSrc;
        if (titleElem) titleElem.textContent = title;
        if (descElem) descElem.textContent = desc;
        modal.classList.add('active');
    }
}

function closeLightbox() {
    const modal = document.getElementById('lightbox-modal');
    if (modal) modal.classList.remove('active');
}

// ==========================================================================
// UI Event Bindings
// ==========================================================================

function bindEvents() {
    // Navigation Tabs
    document.querySelectorAll('.journey-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            const target = tab.getAttribute('data-target');
            switchJourneySection(target);
        });
    });

    // Guided Tour Controls
    const startTourBtn = document.getElementById('btn-start-tour');
    if (startTourBtn) startTourBtn.addEventListener('click', startTour);
    document.getElementById('btn-tour-next').addEventListener('click', nextTourStep);
    document.getElementById('btn-tour-prev').addEventListener('click', prevTourStep);
    document.getElementById('btn-tour-close').addEventListener('click', closeTour);

    // Simulator Controls
    document.getElementById('btn-load-run').addEventListener('click', loadTraceAndInitialize);
    document.getElementById('btn-step').addEventListener('click', stepForward);
    document.getElementById('btn-prev').addEventListener('click', stepBackward);
    document.getElementById('btn-play').addEventListener('click', togglePlay);
    document.getElementById('btn-reset').addEventListener('click', loadTraceAndInitialize);

    // Preset buttons
    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadPreset(btn.getAttribute('data-preset'));
            loadTraceAndInitialize();
        });
    });

    // Policy Radio Cards
    document.querySelectorAll('input[name="repl-policy"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            document.querySelectorAll('.policy-card').forEach(c => c.classList.remove('selected'));
            radio.closest('.policy-card').classList.add('selected');
            renderStats(e.target.value);
        });
    });

    // Speed Slider
    const speedSlider = document.getElementById('speed-slider');
    const speedVal = document.getElementById('speed-val');
    speedSlider.addEventListener('input', () => {
        if (speedVal) speedVal.textContent = `${speedSlider.value}ms`;
        if (playInterval) {
            togglePlay();
            togglePlay();
        }
    });

    // Clear Audit Log
    const clearLogBtn = document.getElementById('btn-clear-log');
    if (clearLogBtn) {
        clearLogBtn.addEventListener('click', () => {
            const logBox = document.getElementById('execution-log');
            if (logBox) logBox.innerHTML = '<div class="log-row info-row">Audit log cleared.</div>';
        });
    }

    // Thrashing Demo Trigger from Section 2
    const thrashBtn = document.getElementById('btn-demo-thrash');
    if (thrashBtn) {
        thrashBtn.addEventListener('click', () => {
            switchJourneySection('sec-simulator');
            loadPreset('loop');
            document.querySelectorAll('.btn-preset').forEach(b => {
                b.classList.toggle('active', b.getAttribute('data-preset') === 'loop');
            });
            loadTraceAndInitialize();
            appendLog('Loaded Thrashing Loop pattern into simulator (18 blocks in 16-way set).', 'miss');
        });
    }

    // Lightbox triggers for Figures
    document.querySelectorAll('.figure-card').forEach(card => {
        card.addEventListener('click', () => {
            const imgSrc = card.getAttribute('data-img');
            const title = card.getAttribute('data-title');
            const desc = card.getAttribute('data-desc');
            openLightbox(imgSrc, title, desc);
        });
    });

    const closeLightboxBtn = document.getElementById('btn-lightbox-close');
    const lightboxBackdrop = document.getElementById('lightbox-backdrop');
    if (closeLightboxBtn) closeLightboxBtn.addEventListener('click', closeLightbox);
    if (lightboxBackdrop) lightboxBackdrop.addEventListener('click', closeLightbox);

    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeLightbox();
    });

    // Code Deck Tab Switching in Section 6
    document.querySelectorAll('.c-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.c-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            const fileKey = tab.getAttribute('data-file');
            const codeDisplay = document.getElementById('code-display');
            if (codeDisplay && CODE_SNIPPETS[fileKey]) {
                codeDisplay.textContent = CODE_SNIPPETS[fileKey];
            }
        });
    });
}

// ==========================================================================
// Entrypoint
// ==========================================================================

window.addEventListener('DOMContentLoaded', () => {
    initCache();
    initSecondaryCaches();
    initRegisters();
    initRecencyBars();
    loadPreset('phase-change');
    loadTraceAndInitialize();
    bindEvents();
    initCharts();
});
