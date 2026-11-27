// ============================================================================
// Split or Steal — 3D Apartment Replay Theater
// Main Three.js Scene, Lighting Director, Camera Director & Beat Sequencer
// ============================================================================

import * as THREE from '../vendor/three.module.js';
import { GLTFLoader } from '../vendor/GLTFLoader.js';
import { OrbitControls } from '../vendor/OrbitControls.js';
import { CHARACTER_META, createCharacterRig } from './characters.js';
import { createNegotiationTable } from './table_props.js';
import { TheaterAudio } from './audio.js';

// ----------------------------------------------------------------------------
// Stage Placement Constants (aligned with apartment.glb North Room bounds)
// ----------------------------------------------------------------------------
const APARTMENT_Y_OFFSET = 1.485; // Lifts native floor (-1.485) to y = 0.0; ceiling ≈ +2.34m
const TABLE_POS = new THREE.Vector3(0.25, 0.0, -2.25);
const TABLE_ROT_Y = Math.PI; // Seat A -> world (+1.30, 0.46, -2.25) [screen-left], Seat B -> (-0.80, 0.46, -2.25) [screen-right]

const SEAT_A_WORLD = new THREE.Vector3(1.30, 1.10, -2.25);
const SEAT_B_WORLD = new THREE.Vector3(-0.80, 1.10, -2.25);
const TABLE_CENTER_TOP = new THREE.Vector3(0.25, 0.78, -2.25);

// All cameras kept strictly inside clear North Room air (y <= 2.05m, z >= -3.84m)
const CAM_SHOTS = {
  overview: {
    pos: new THREE.Vector3(0.25, 1.56, -3.80),
    target: new THREE.Vector3(0.25, 0.94, -2.25),
  },
  seatAFocus: {
    pos: new THREE.Vector3(-0.16, 1.37, -3.34),
    target: new THREE.Vector3(1.16, 1.07, -2.25),
  },
  seatAThought: {
    pos: new THREE.Vector3(0.38, 1.32, -3.12),
    target: new THREE.Vector3(1.26, 1.11, -2.25),
  },
  seatBFocus: {
    pos: new THREE.Vector3(0.66, 1.37, -3.34),
    target: new THREE.Vector3(-0.66, 1.07, -2.25),
  },
  seatBThought: {
    pos: new THREE.Vector3(0.12, 1.32, -3.12),
    target: new THREE.Vector3(-0.76, 1.11, -2.25),
  },
  seatAPov: {
    pos: new THREE.Vector3(0.96, 1.29, -2.30),
    target: new THREE.Vector3(-0.72, 1.05, -2.25),
  },
  seatBPov: {
    pos: new THREE.Vector3(-0.46, 1.29, -2.30),
    target: new THREE.Vector3(1.22, 1.05, -2.25),
  },
  reveal: {
    pos: new THREE.Vector3(0.25, 1.88, -3.34),
    target: new THREE.Vector3(0.25, 0.78, -2.25),
  },
};

const LIGHT_MODES = {
  normal: {
    hemi: 1.05,
    ambient: 0.55,
    fillFront: 3.1,
    fillBack: 1.55,
    tableSpot: 5.2,
    noirSpot: 0.0,
    exposure: 1.12,
  },
  thought: {
    hemi: 0.22,
    ambient: 0.09,
    fillFront: 0.32,
    fillBack: 0.16,
    tableSpot: 0.85,
    noirSpot: 10.5,
    exposure: 1.0,
  },
  reveal: {
    hemi: 0.85,
    ambient: 0.42,
    fillFront: 2.8,
    fillBack: 1.4,
    tableSpot: 7.4,
    noirSpot: 0.0,
    exposure: 1.16,
  },
};

// ----------------------------------------------------------------------------
// DOM References
// ----------------------------------------------------------------------------
const dom = {
  canvas: document.getElementById('stage-canvas'),
  loadingOverlay: document.getElementById('loading-overlay'),
  loadingText: document.getElementById('loading-text'),
  gameSelect: document.getElementById('game-select'),
  btnTheme: document.getElementById('btn-theme'),
  camButtons: Array.from(document.querySelectorAll('[data-cam]')),
  roundButtons: Array.from(document.querySelectorAll('[data-round]')),
  viewButtons: Array.from(document.querySelectorAll('[data-view]')),
  orbitHint: document.getElementById('orbit-hint'),

  seatACard: document.getElementById('seat-a-card'),
  seatADot: document.getElementById('seat-a-dot'),
  seatAName: document.getElementById('seat-a-name'),
  seatATitle: document.getElementById('seat-a-title'),
  seatABankroll: document.getElementById('seat-a-bankroll'),
  seatAFinal: document.getElementById('seat-a-final'),
  seatAAction: document.getElementById('seat-a-action'),

  seatBCard: document.getElementById('seat-b-card'),
  seatBDot: document.getElementById('seat-b-dot'),
  seatBName: document.getElementById('seat-b-name'),
  seatBTitle: document.getElementById('seat-b-title'),
  seatBBankroll: document.getElementById('seat-b-bankroll'),
  seatBFinal: document.getElementById('seat-b-final'),
  seatBAction: document.getElementById('seat-b-action'),

  stagePot: document.getElementById('stage-pot'),
  stagePhase: document.getElementById('stage-phase'),

  beatCard: document.getElementById('beat-card'),
  beatProgressFill: document.getElementById('beat-progress-fill'),
  beatBadge: document.getElementById('beat-badge'),
  beatSpeaker: document.getElementById('beat-speaker'),
  beatCounter: document.getElementById('beat-counter'),
  beatTextBody: document.getElementById('beat-text-body'),
  beatText: document.getElementById('beat-text'),

  beatRevealBody: document.getElementById('beat-reveal-body'),
  revealOutcomeTag: document.getElementById('reveal-outcome-tag'),
  revealSummary: document.getElementById('reveal-summary'),
  revealColA: document.getElementById('reveal-col-a'),
  revealNameA: document.getElementById('reveal-name-a'),
  revealActionA: document.getElementById('reveal-action-a'),
  revealPayoutA: document.getElementById('reveal-payout-a'),
  revealThoughtA: document.getElementById('reveal-thought-a'),
  revealColB: document.getElementById('reveal-col-b'),
  revealNameB: document.getElementById('reveal-name-b'),
  revealActionB: document.getElementById('reveal-action-b'),
  revealPayoutB: document.getElementById('reveal-payout-b'),
  revealThoughtB: document.getElementById('reveal-thought-b'),

  btnPrev: document.getElementById('btn-prev'),
  btnPlay: document.getElementById('btn-play'),
  btnNext: document.getElementById('btn-next'),
  btnSpeed: document.getElementById('btn-speed'),
  btnVoice: document.getElementById('btn-voice'),
  btnAmbient: document.getElementById('btn-ambient'),
  btnSpoiler: document.getElementById('btn-spoiler'),
};

// ----------------------------------------------------------------------------
// Application State
// ----------------------------------------------------------------------------
const state = {
  games: [],
  gameIndex: 0,
  beats: [],
  visibleIndices: [],
  currentBeatIndex: 0,
  viewMode: 'both', // 'both' | 'spoken' | 'thoughts'
  camMode: 'auto', // 'auto' | 'overview' | 'seatA' | 'seatB' | 'free'
  isPlaying: false,
  playbackSpeed: 1, // 1 | 1.5
  showSpoilers: false,
  renderedRound: null,
  cardsRevealed: false,
  beatToken: 0,
  autoTimer: null,
  reactionTimer: null,
};

const audio = new TheaterAudio();

// ----------------------------------------------------------------------------
// Three.js Scene, Renderer, Camera & Controls
// ----------------------------------------------------------------------------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x161412);
scene.fog = new THREE.FogExp2(0x161412, 0.055);

const renderer = new THREE.WebGLRenderer({
  canvas: dom.canvas,
  antialias: true,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = LIGHT_MODES.normal.exposure;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;

const camera = new THREE.PerspectiveCamera(
  56,
  window.innerWidth / window.innerHeight,
  0.05,
  45
);
camera.position.copy(CAM_SHOTS.overview.pos);

const controls = new OrbitControls(camera, dom.canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.copy(CAM_SHOTS.overview.target);
controls.minDistance = 0.65;
controls.maxDistance = 3.2;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minPolarAngle = 0.25;
controls.enabled = false;

// Smooth camera director target vectors
const camDesiredPos = CAM_SHOTS.overview.pos.clone();
const camDesiredTarget = CAM_SHOTS.overview.target.clone();
const camCurrentTarget = CAM_SHOTS.overview.target.clone();
camera.lookAt(camCurrentTarget);

// ----------------------------------------------------------------------------
// Atmospheric Lighting Rig
// ----------------------------------------------------------------------------
const hemiLight = new THREE.HemisphereLight(0xfff4e6, 0x3b312a, LIGHT_MODES.normal.hemi);
scene.add(hemiLight);

const ambientLight = new THREE.AmbientLight(0xfff1e0, LIGHT_MODES.normal.ambient);
scene.add(ambientLight);

const roomFillFront = new THREE.PointLight(0xffdfb8, LIGHT_MODES.normal.fillFront, 8.5, 1.35);
roomFillFront.position.set(0.25, 2.08, -3.30);
scene.add(roomFillFront);

const roomFillBack = new THREE.PointLight(0xe4ecff, LIGHT_MODES.normal.fillBack, 7.5, 1.45);
roomFillBack.position.set(0.25, 1.95, -1.30);
scene.add(roomFillBack);

const tableSpot = new THREE.SpotLight(
  0xfff0d8,
  LIGHT_MODES.normal.tableSpot,
  7.5,
  Math.PI * 0.36,
  0.55,
  1.15
);
tableSpot.position.set(0.25, 2.25, -2.38);
tableSpot.target.position.copy(TABLE_CENTER_TOP);
tableSpot.castShadow = true;
tableSpot.shadow.mapSize.set(1024, 1024);
tableSpot.shadow.bias = -0.0005;
tableSpot.shadow.near = 0.4;
tableSpot.shadow.far = 5.5;
scene.add(tableSpot);
scene.add(tableSpot.target);

// Tight noir spotlight focused on the thinking character during Hidden <thought>
const noirSpot = new THREE.SpotLight(
  0xffcb96,
  0.0,
  5.5,
  Math.PI * 0.2,
  0.65,
  1.1
);
noirSpot.position.set(0.95, 2.22, -2.65);
noirSpot.target.position.copy(SEAT_A_WORLD);
noirSpot.castShadow = true;
noirSpot.shadow.mapSize.set(512, 512);
noirSpot.shadow.bias = -0.0006;
scene.add(noirSpot);
scene.add(noirSpot.target);

const lightTargets = {
  ...LIGHT_MODES.normal,
  noirPos: noirSpot.position.clone(),
  noirLook: noirSpot.target.position.clone(),
};

function setLightingMode(mode, focusSeat = null) {
  const preset = LIGHT_MODES[mode] || LIGHT_MODES.normal;
  lightTargets.hemi = preset.hemi;
  lightTargets.ambient = preset.ambient;
  lightTargets.fillFront = preset.fillFront;
  lightTargets.fillBack = preset.fillBack;
  lightTargets.tableSpot = preset.tableSpot;
  lightTargets.noirSpot = preset.noirSpot;
  lightTargets.exposure = preset.exposure;

  if (mode === 'thought' && (focusSeat === 'A' || focusSeat === 'B')) {
    const head = focusSeat === 'A' ? SEAT_A_WORLD : SEAT_B_WORLD;
    lightTargets.noirLook.copy(head);
    lightTargets.noirPos.set(
      THREE.MathUtils.lerp(head.x, TABLE_POS.x, 0.25),
      2.22,
      head.z - 0.42
    );
  }
}

// ----------------------------------------------------------------------------
// Elegant Architectural Fallback Room (shown only if apartment.glb fails)
// ----------------------------------------------------------------------------
function buildFallbackRoom() {
  const group = new THREE.Group();
  group.name = 'fallback_apartment_room';

  const floorMat = new THREE.MeshStandardMaterial({
    color: 0x4b3a2d,
    roughness: 0.78,
    metalness: 0.04,
  });
  const wallMat = new THREE.MeshStandardMaterial({
    color: 0xd9d2c7,
    roughness: 0.9,
    metalness: 0.0,
  });
  const trimMat = new THREE.MeshStandardMaterial({
    color: 0x26221e,
    roughness: 0.65,
    metalness: 0.08,
  });

  const floor = new THREE.Mesh(new THREE.PlaneGeometry(7.2, 7.2), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0.25, 0, -2.25);
  floor.receiveShadow = true;
  group.add(floor);

  const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(7.2, 7.2), wallMat);
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.set(0.25, 2.42, -2.25);
  group.add(ceiling);

  // Back, left, right, front walls enclosing the room around TABLE_POS
  const backWall = new THREE.Mesh(new THREE.PlaneGeometry(7.2, 2.5), wallMat);
  backWall.position.set(0.25, 1.25, -0.55);
  backWall.rotation.y = Math.PI;
  backWall.receiveShadow = true;
  group.add(backWall);

  const frontWall = new THREE.Mesh(new THREE.PlaneGeometry(7.2, 2.5), wallMat);
  frontWall.position.set(0.25, 1.25, -5.0);
  group.add(frontWall);

  const leftWall = new THREE.Mesh(new THREE.PlaneGeometry(5.5, 2.5), wallMat);
  leftWall.position.set(3.2, 1.25, -2.5);
  leftWall.rotation.y = -Math.PI / 2;
  group.add(leftWall);

  const rightWall = new THREE.Mesh(new THREE.PlaneGeometry(5.5, 2.5), wallMat);
  rightWall.position.set(-2.7, 1.25, -2.5);
  rightWall.rotation.y = Math.PI / 2;
  group.add(rightWall);

  // Warm rug beneath the negotiation table
  const rug = new THREE.Mesh(
    new THREE.PlaneGeometry(3.2, 2.1),
    new THREE.MeshStandardMaterial({ color: 0x2a2724, roughness: 0.94, metalness: 0.0 })
  );
  rug.rotation.x = -Math.PI / 2;
  rug.position.set(0.25, 0.004, -2.25);
  rug.receiveShadow = true;
  group.add(rug);

  // Architectural ceiling beam & warm pendant fixture above table
  const beam = new THREE.Mesh(new THREE.BoxGeometry(6.8, 0.12, 0.18), trimMat);
  beam.position.set(0.25, 2.36, -2.25);
  group.add(beam);

  return group;
}

const fallbackRoom = buildFallbackRoom();
scene.add(fallbackRoom);

// ----------------------------------------------------------------------------
// Negotiation Table & Active Character Rigs
// ----------------------------------------------------------------------------
const table = createNegotiationTable();
table.group.position.copy(TABLE_POS);
table.group.rotation.y = TABLE_ROT_Y;
scene.add(table.group);

let rigA = null;
let rigB = null;

function mountCharacters(charIdA, charIdB) {
  if (rigA) {
    table.seatAAnchor.remove(rigA.group);
    rigA.dispose();
    rigA = null;
  }
  if (rigB) {
    table.seatBAnchor.remove(rigB.group);
    rigB.dispose();
    rigB = null;
  }

  rigA = createCharacterRig(charIdA, 'A');
  rigB = createCharacterRig(charIdB, 'B');
  const showPlates = state.camMode === 'free';
  rigA.setNameplateVisible(showPlates);
  rigB.setNameplateVisible(showPlates);
  table.seatAAnchor.add(rigA.group);
  table.seatBAnchor.add(rigB.group);
}

// ----------------------------------------------------------------------------
// Load Photogrammetry Apartment GLB (with critical metalness/roughness fix)
// ----------------------------------------------------------------------------
function loadApartmentGLB() {
  return new Promise((resolve) => {
    const loader = new GLTFLoader();
    loader.load(
      './assets/apartment.glb',
      (gltf) => {
        const apartment = gltf.scene;
        apartment.name = 'apartment_glb';
        apartment.position.set(0, APARTMENT_Y_OFFSET, 0);

        apartment.traverse((child) => {
          if (!child.isMesh) return;
          child.receiveShadow = true;
          const mats = Array.isArray(child.material) ? child.material : [child.material];
          mats.forEach((mat) => {
            if (!mat) return;
            // Critical fix: GLB omits metallicFactor/roughnessFactor -> defaults to metalness=1 (black)
            if ('metalness' in mat) mat.metalness = 0.0;
            if ('roughness' in mat) mat.roughness = 0.88;
            mat.side = THREE.DoubleSide;
            mat.needsUpdate = true;
          });
        });

        scene.add(apartment);
        fallbackRoom.visible = false;
        resolve(true);
      },
      undefined,
      (err) => {
        console.warn('[Stage] apartment.glb unavailable, keeping architectural fallback room.', err);
        fallbackRoom.visible = true;
        resolve(false);
      }
    );
  });
}

// ----------------------------------------------------------------------------
// Beat Sequencer Construction
// ----------------------------------------------------------------------------
function formatMoney(n) {
  return `$${Number(n || 0).toLocaleString('en-US')}`;
}

function classifyOutcome(actionA, actionB) {
  if (actionA === 'SPLIT' && actionB === 'SPLIT') return 'mutual_split';
  if (actionA === 'STEAL' && actionB === 'SPLIT') return 'a_steals';
  if (actionA === 'SPLIT' && actionB === 'STEAL') return 'b_steals';
  return 'mutual_crash';
}

function buildBeatsForGame(game) {
  const beats = [];
  let runningA = 0;
  let runningB = 0;

  for (const roundObj of game.rounds) {
    const r = roundObj.round;
    const stake = roundObj.stake;

    // 1. Round Intro Beat
    const introText =
      r === 1
        ? game.tagline
        : `Stakes escalate to ${formatMoney(stake)}. Current score: ${game.player_a.display_name} ${formatMoney(runningA)} vs ${game.player_b.display_name} ${formatMoney(runningB)}.`;

    beats.push({
      type: 'intro',
      round: r,
      stake,
      title: `Round ${r} of ${game.rounds.length}`,
      text: introText,
      bankrollA: runningA,
      bankrollB: runningB,
    });

    // 2. Negotiation Turns (Private Thought -> Spoken Dialogue)
    for (const turn of roundObj.turns) {
      const thought = (turn.thought || '').trim();
      const speech = (turn.speech || '').trim();

      if (thought) {
        beats.push({
          type: 'thought',
          round: r,
          stake,
          turnNumber: turn.turn,
          seat: turn.seat,
          characterId: turn.character_id,
          speaker: turn.speaker,
          text: thought,
          audioUrl: null,
          silentTurn: speech.length === 0,
          bankrollA: runningA,
          bankrollB: runningB,
        });
      }

      if (speech) {
        beats.push({
          type: 'speech',
          round: r,
          stake,
          turnNumber: turn.turn,
          seat: turn.seat,
          characterId: turn.character_id,
          speaker: turn.speaker,
          text: speech,
          audioUrl: turn.audio_url || null,
          bankrollA: runningA,
          bankrollB: runningB,
        });
      }
    }

    // 3. Simultaneous Decision Reveal Beat
    const decA = roundObj.decision_a;
    const decB = roundObj.decision_b;
    runningA = decA.bankroll;
    runningB = decB.bankroll;

    beats.push({
      type: 'reveal',
      round: r,
      stake,
      decisionA: decA,
      decisionB: decB,
      outcome: classifyOutcome(decA.action, decB.action),
      bankrollA: runningA,
      bankrollB: runningB,
    });
  }

  return beats;
}

function recomputeVisibleIndices() {
  state.visibleIndices = [];
  state.beats.forEach((beat, idx) => {
    if (state.viewMode === 'both') {
      state.visibleIndices.push(idx);
    } else if (state.viewMode === 'spoken') {
      if (beat.type !== 'thought') state.visibleIndices.push(idx);
    } else if (state.viewMode === 'thoughts') {
      if (beat.type !== 'speech') state.visibleIndices.push(idx);
    }
  });
}

function nearestVisibleBeat(targetIdx) {
  if (!state.visibleIndices.length) return 0;
  if (state.visibleIndices.includes(targetIdx)) return targetIdx;
  let best = state.visibleIndices[0];
  let bestDist = Math.abs(best - targetIdx);
  for (const idx of state.visibleIndices) {
    const d = Math.abs(idx - targetIdx);
    if (d < bestDist) {
      best = idx;
      bestDist = d;
    }
  }
  return best;
}

// ----------------------------------------------------------------------------
// Camera Director
// ----------------------------------------------------------------------------
function updateCameraTargetForBeat(beat) {
  if (state.camMode === 'free') return;

  let shot = CAM_SHOTS.overview;
  if (state.camMode === 'overview') {
    shot = CAM_SHOTS.overview;
  } else if (state.camMode === 'seatA') {
    shot = CAM_SHOTS.seatAPov;
  } else if (state.camMode === 'seatB') {
    shot = CAM_SHOTS.seatBPov;
  } else if (state.camMode === 'auto' && beat) {
    if (beat.type === 'intro') {
      shot = CAM_SHOTS.overview;
    } else if (beat.type === 'thought') {
      shot = beat.seat === 'A' ? CAM_SHOTS.seatAThought : CAM_SHOTS.seatBThought;
    } else if (beat.type === 'speech') {
      shot = beat.seat === 'A' ? CAM_SHOTS.seatAFocus : CAM_SHOTS.seatBFocus;
    } else if (beat.type === 'reveal') {
      shot = CAM_SHOTS.reveal;
    }
  }

  camDesiredPos.copy(shot.pos);
  camDesiredTarget.copy(shot.target);
}

function setCameraMode(mode) {
  state.camMode = mode;
  const isFree = mode === 'free';
  controls.enabled = isFree;
  dom.orbitHint.hidden = !isFree;
  if (rigA) rigA.setNameplateVisible(isFree);
  if (rigB) rigB.setNameplateVisible(isFree);

  if (isFree) {
    controls.target.copy(camCurrentTarget);
    controls.update();
  } else {
    updateCameraTargetForBeat(state.beats[state.currentBeatIndex]);
  }

  dom.camButtons.forEach((btn) => {
    const active = btn.dataset.cam === mode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', String(active));
  });
}

// ----------------------------------------------------------------------------
// Beat Presentation & Playback Engine
// ----------------------------------------------------------------------------
function clearTimers() {
  if (state.autoTimer) {
    clearTimeout(state.autoTimer);
    state.autoTimer = null;
  }
  if (state.reactionTimer) {
    clearTimeout(state.reactionTimer);
    state.reactionTimer = null;
  }
}

function flashBankroll(el, action, payout) {
  el.classList.remove('flash-blue', 'flash-orange');
  if (!payout) return;
  el.classList.add(action === 'STEAL' ? 'flash-orange' : 'flash-blue');
  setTimeout(() => {
    el.classList.remove('flash-blue', 'flash-orange');
  }, 1400);
}

function setDecisionBadge(el, action) {
  el.classList.remove('split', 'steal', 'pending');
  if (!action) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  el.textContent = action;
  el.classList.add(action === 'STEAL' ? 'steal' : 'split');
}

function estimateBeatDurationMs(beat) {
  if (!beat) return 4000;
  if (beat.type === 'intro') return Math.round(3600 / state.playbackSpeed);
  if (beat.type === 'reveal') return Math.round(6200 / state.playbackSpeed);
  const words = (beat.text || '').split(/\s+/).filter(Boolean).length;
  const baseMs = Math.min(9200, Math.max(3000, 1650 + words * 190));
  return Math.round(baseMs / state.playbackSpeed);
}

function applyBeat(beatIndex, { triggerAuto = true } = {}) {
  clearTimers();
  audio.stopSpeech();

  const game = state.games[state.gameIndex];
  if (!game || !state.beats.length) return;

  const clampedIdx = nearestVisibleBeat(beatIndex);
  state.currentBeatIndex = clampedIdx;
  const beat = state.beats[clampedIdx];
  state.beatToken += 1;
  const token = state.beatToken;

  // Progress bar & beat counter across visible beats
  const visPos = Math.max(0, state.visibleIndices.indexOf(clampedIdx));
  const visTotal = Math.max(1, state.visibleIndices.length);
  dom.beatCounter.textContent = `${String(visPos + 1).padStart(2, '0')} / ${String(visTotal).padStart(2, '0')}`;
  dom.beatProgressFill.style.width = `${((visPos + 1) / visTotal) * 100}%`;

  // Update prev/next disabled states
  dom.btnPrev.disabled = visPos <= 0;
  dom.btnNext.disabled = visPos >= visTotal - 1;

  // Update Round buttons & Stake indicator
  dom.roundButtons.forEach((btn) => {
    btn.classList.toggle('active', Number(btn.dataset.round) === beat.round);
  });
  dom.stagePot.textContent = `POT ${formatMoney(beat.stake)}`;

  // Sync 3D Table Pot & Card state
  if (state.renderedRound !== beat.round) {
    table.setPot(beat.stake);
    table.resetCards();
    state.renderedRound = beat.round;
    state.cardsRevealed = false;
  } else if (beat.type !== 'reveal' && state.cardsRevealed) {
    table.setPot(beat.stake);
    table.resetCards();
    state.cardsRevealed = false;
  }

  // Update Seat Bankrolls
  dom.seatABankroll.textContent = formatMoney(beat.bankrollA);
  dom.seatBBankroll.textContent = formatMoney(beat.bankrollB);

  // Reset Seat highlight states
  dom.seatACard.classList.remove('speaking', 'thinking');
  dom.seatBCard.classList.remove('speaking', 'thinking');

  // Reset beat card classes
  dom.beatCard.classList.remove(
    'mode-intro',
    'mode-thought',
    'mode-speech',
    'mode-reveal',
    'outcome-split',
    'outcome-steal',
    'outcome-crash'
  );
  document.body.classList.toggle('thought-mode', beat.type === 'thought');

  // Camera & Lighting & Rig states per beat type
  updateCameraTargetForBeat(beat);

  if (beat.type === 'intro') {
    dom.stagePhase.textContent = `ROUND ${beat.round} INTRO`;
    dom.beatCard.classList.add('mode-intro');
    dom.beatBadge.textContent = `ROUND ${beat.round} · ${formatMoney(beat.stake)}`;
    dom.beatSpeaker.textContent = game.title;
    dom.beatTextBody.hidden = false;
    dom.beatRevealBody.hidden = true;
    dom.beatText.textContent = beat.text;

    setDecisionBadge(dom.seatAAction, null);
    setDecisionBadge(dom.seatBAction, null);

    if (rigA) rigA.setState('idle');
    if (rigB) rigB.setState('idle');
    setLightingMode('normal');
    audio.setMood('intro');
  } else if (beat.type === 'thought') {
    const meta = CHARACTER_META[beat.characterId] || {};
    dom.stagePhase.textContent = `HIDDEN THOUGHT`;
    dom.beatCard.classList.add('mode-thought');
    dom.beatBadge.textContent = beat.silentTurn
      ? `PRIVATE THOUGHT · SILENT TURN`
      : `PRIVATE THOUGHT · SEAT ${beat.seat}`;
    dom.beatSpeaker.textContent = beat.speaker;
    dom.beatTextBody.hidden = false;
    dom.beatRevealBody.hidden = true;
    dom.beatText.textContent = beat.text;
    dom.beatCard.style.setProperty('--char-accent', meta.accentColor || '#EF4E14');

    setDecisionBadge(dom.seatAAction, null);
    setDecisionBadge(dom.seatBAction, null);

    if (beat.seat === 'A') {
      dom.seatACard.classList.add('thinking');
      if (rigA) rigA.setState('thinking');
      if (rigB) rigB.setState('idle');
    } else {
      dom.seatBCard.classList.add('thinking');
      if (rigA) rigA.setState('idle');
      if (rigB) rigB.setState('thinking');
    }

    setLightingMode('thought', beat.seat);
    audio.setMood('thought');
  } else if (beat.type === 'speech') {
    const meta = CHARACTER_META[beat.characterId] || {};
    dom.stagePhase.textContent = `NEGOTIATION`;
    dom.beatCard.classList.add('mode-speech');
    dom.beatBadge.textContent = `SPOKEN · SEAT ${beat.seat}`;
    dom.beatSpeaker.textContent = beat.speaker;
    dom.beatTextBody.hidden = false;
    dom.beatRevealBody.hidden = true;
    dom.beatText.textContent = `"${beat.text}"`;
    dom.beatCard.style.setProperty('--char-accent', meta.accentColor || '#141414');

    setDecisionBadge(dom.seatAAction, null);
    setDecisionBadge(dom.seatBAction, null);

    if (beat.seat === 'A') {
      dom.seatACard.classList.add('speaking');
      if (rigA) rigA.setState('speaking');
      if (rigB) rigB.setState('idle');
    } else {
      dom.seatBCard.classList.add('speaking');
      if (rigA) rigA.setState('idle');
      if (rigB) rigB.setState('speaking');
    }

    setLightingMode('normal');
    audio.setMood('speech');
  } else if (beat.type === 'reveal') {
    const { decisionA, decisionB, outcome } = beat;
    dom.stagePhase.textContent = `DECISION REVEAL`;
    dom.beatCard.classList.add('mode-reveal');
    dom.beatTextBody.hidden = true;
    dom.beatRevealBody.hidden = false;

    const outcomeMeta = {
      mutual_split: {
        cls: 'outcome-split',
        badge: `ROUND ${beat.round} REVEAL`,
        headline: 'MUTUAL SPLIT',
        summary: `Both players cooperate and split ${formatMoney(beat.stake)} evenly.`,
      },
      a_steals: {
        cls: 'outcome-steal',
        badge: `ROUND ${beat.round} BETRAYAL`,
        headline: `${decisionA.speaker.toUpperCase()} STEALS THE POT`,
        summary: `${decisionA.speaker} takes all ${formatMoney(beat.stake)}; ${decisionB.speaker} walks away with $0.`,
      },
      b_steals: {
        cls: 'outcome-steal',
        badge: `ROUND ${beat.round} BETRAYAL`,
        headline: `${decisionB.speaker.toUpperCase()} STEALS THE POT`,
        summary: `${decisionB.speaker} takes all ${formatMoney(beat.stake)}; ${decisionA.speaker} walks away with $0.`,
      },
      mutual_crash: {
        cls: 'outcome-crash',
        badge: `ROUND ${beat.round} CRASH`,
        headline: 'MUTUAL STEAL — POT BURNS TO $0',
        summary: `Both chose STEAL. Neither player collects a dollar from the ${formatMoney(beat.stake)} pot.`,
      },
    }[outcome];

    dom.beatCard.classList.add(outcomeMeta.cls);
    dom.beatBadge.textContent = outcomeMeta.badge;
    dom.beatSpeaker.textContent = `${decisionA.action} vs ${decisionB.action}`;
    dom.revealOutcomeTag.textContent = outcomeMeta.headline;
    dom.revealSummary.textContent = outcomeMeta.summary;

    // Populate columns
    dom.revealNameA.textContent = decisionA.speaker;
    setDecisionBadge(dom.revealActionA, decisionA.action);
    dom.revealPayoutA.textContent = decisionA.payout > 0 ? `+${formatMoney(decisionA.payout)}` : '$0';
    dom.revealColA.className = `reveal-col ${decisionA.action === 'STEAL' ? 'steal' : 'split'}`;

    dom.revealNameB.textContent = decisionB.speaker;
    setDecisionBadge(dom.revealActionB, decisionB.action);
    dom.revealPayoutB.textContent = decisionB.payout > 0 ? `+${formatMoney(decisionB.payout)}` : '$0';
    dom.revealColB.className = `reveal-col ${decisionB.action === 'STEAL' ? 'steal' : 'split'}`;

    const showDecisionThoughts = state.viewMode !== 'spoken';
    dom.revealThoughtA.hidden = !showDecisionThoughts || !decisionA.thought;
    dom.revealThoughtB.hidden = !showDecisionThoughts || !decisionB.thought;
    dom.revealThoughtA.textContent = decisionA.thought || '';
    dom.revealThoughtA.title = decisionA.thought || '';
    dom.revealThoughtB.textContent = decisionB.thought || '';
    dom.revealThoughtB.title = decisionB.thought || '';

    setDecisionBadge(dom.seatAAction, decisionA.action);
    setDecisionBadge(dom.seatBAction, decisionB.action);
    flashBankroll(dom.seatABankroll, decisionA.action, decisionA.payout);
    flashBankroll(dom.seatBBankroll, decisionB.action, decisionB.payout);

    // 3D Card Flip & Chip Transfer
    table.setPot(beat.stake);
    table.revealDecisions(decisionA.action, decisionB.action, decisionA.payout, decisionB.payout);
    state.cardsRevealed = true;

    // Initial reveal posture -> reaction after cards slam onto the table
    if (rigA) rigA.setState(decisionA.action === 'STEAL' ? 'reveal_steal' : 'reveal_split');
    if (rigB) rigB.setState(decisionB.action === 'STEAL' ? 'reveal_steal' : 'reveal_split');

    state.reactionTimer = setTimeout(() => {
      if (state.beatToken !== token) return;
      if (outcome === 'mutual_split') {
        if (rigA) rigA.setState('react_win');
        if (rigB) rigB.setState('react_win');
      } else if (outcome === 'a_steals') {
        if (rigA) rigA.setState('react_win');
        if (rigB) rigB.setState('react_betrayed');
      } else if (outcome === 'b_steals') {
        if (rigA) rigA.setState('react_betrayed');
        if (rigB) rigB.setState('react_win');
      } else {
        if (rigA) rigA.setState('react_crash');
        if (rigB) rigB.setState('react_crash');
      }
    }, 920);

    setLightingMode('reveal');
    audio.setMood('reveal');
    audio.playCardFlipCue(outcome);
  }

  // Optional Voice playback + Auto-advance scheduling
  let voiceSpoken = false;
  if (audio.voiceEnabled && (beat.type === 'speech' || beat.type === 'thought')) {
    voiceSpoken = audio.speakBeat({
      text: beat.text,
      characterId: beat.characterId,
      audioUrl: beat.audioUrl,
      isThought: beat.type === 'thought',
      onEnd: () => {
        if (state.isPlaying && state.beatToken === token) {
          clearTimers();
          state.autoTimer = setTimeout(() => {
            if (state.isPlaying && state.beatToken === token) stepBeat(1);
          }, 650);
        }
      },
    });
  }

  if (triggerAuto && state.isPlaying) {
    const fallbackMs = voiceSpoken
      ? Math.max(14000, estimateBeatDurationMs(beat) * 2.2)
      : estimateBeatDurationMs(beat);
    state.autoTimer = setTimeout(() => {
      if (state.isPlaying && state.beatToken === token) {
        stepBeat(1);
      }
    }, fallbackMs);
  }
}

function stepBeat(delta) {
  if (!state.visibleIndices.length) return;
  const visPos = state.visibleIndices.indexOf(state.currentBeatIndex);
  const nextPos = visPos + delta;

  if (nextPos < 0) return;
  if (nextPos >= state.visibleIndices.length) {
    setPlaying(false);
    return;
  }

  applyBeat(state.visibleIndices[nextPos]);
}

function jumpToRound(roundNum) {
  const idx = state.beats.findIndex((b) => b.type === 'intro' && b.round === roundNum);
  if (idx >= 0) {
    applyBeat(idx);
  }
}

function setPlaying(playing) {
  state.isPlaying = playing;
  dom.btnPlay.innerHTML = playing ? '&#10074;&#10074; Pause' : '&#9654; Play';

  if (playing) {
    // If already at the final beat, restart from Beat 1
    const visPos = state.visibleIndices.indexOf(state.currentBeatIndex);
    if (visPos >= state.visibleIndices.length - 1) {
      applyBeat(state.visibleIndices[0]);
      return;
    }
    applyBeat(state.currentBeatIndex, { triggerAuto: true });
  } else {
    clearTimers();
    audio.stopSpeech();
  }
}

// ----------------------------------------------------------------------------
// Game Loading & Switching
// ----------------------------------------------------------------------------
function updateSpoilerDisplay() {
  const game = state.games[state.gameIndex];
  if (!game) return;
  dom.seatAFinal.hidden = !state.showSpoilers;
  dom.seatBFinal.hidden = !state.showSpoilers;
  dom.seatAFinal.textContent = `Final ${formatMoney(game.total_a)}`;
  dom.seatBFinal.textContent = `Final ${formatMoney(game.total_b)}`;
  dom.btnSpoiler.textContent = state.showSpoilers ? 'Final: Shown' : 'Final: Hidden';
  dom.btnSpoiler.setAttribute('aria-pressed', String(state.showSpoilers));
}

function loadGame(index, { autoplay = false } = {}) {
  if (!state.games[index]) return;
  setPlaying(false);
  state.gameIndex = index;
  dom.gameSelect.value = String(index);

  const game = state.games[index];
  const metaA = CHARACTER_META[game.player_a.character_id] || {};
  const metaB = CHARACTER_META[game.player_b.character_id] || {};

  // Populate Seat HUD identities
  dom.seatAName.textContent = game.player_a.display_name;
  dom.seatATitle.textContent = metaA.title || metaA.era || '';
  dom.seatADot.style.background = metaA.accentColor || '#EF4E14';

  dom.seatBName.textContent = game.player_b.display_name;
  dom.seatBTitle.textContent = metaB.title || metaB.era || '';
  dom.seatBDot.style.background = metaB.accentColor || '#2F6FDB';

  updateSpoilerDisplay();

  // Swap 3D character rigs cleanly
  mountCharacters(game.player_a.character_id, game.player_b.character_id);
  table.setPot(game.rounds[0]?.stake || 100);
  table.resetCards();
  state.renderedRound = game.rounds[0]?.round || 1;
  state.cardsRevealed = false;

  // Build beats and enter Beat 1
  state.beats = buildBeatsForGame(game);
  recomputeVisibleIndices();
  applyBeat(state.visibleIndices[0] || 0, { triggerAuto: false });

  if (autoplay) {
    setPlaying(true);
  }
}

// ----------------------------------------------------------------------------
// UI Event Bindings
// ----------------------------------------------------------------------------
function syncThemeButton() {
  const isDark = document.documentElement.classList.contains('dark');
  dom.btnTheme.textContent = isDark ? 'Light HUD' : 'Dark HUD';
}

dom.btnTheme.addEventListener('click', () => {
  const isDark = document.documentElement.classList.toggle('dark');
  try {
    localStorage.setItem('sos-theater-theme', isDark ? 'dark' : 'light');
  } catch (_) {}
  syncThemeButton();
});
syncThemeButton();

dom.gameSelect.addEventListener('change', (e) => {
  const idx = Number(e.target.value);
  if (!Number.isNaN(idx)) loadGame(idx);
});

dom.camButtons.forEach((btn) => {
  btn.addEventListener('click', () => setCameraMode(btn.dataset.cam));
});

dom.roundButtons.forEach((btn) => {
  btn.addEventListener('click', () => jumpToRound(Number(btn.dataset.round)));
});

dom.viewButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    const mode = btn.dataset.view;
    if (mode === state.viewMode) return;
    state.viewMode = mode;
    dom.viewButtons.forEach((b) => {
      const active = b.dataset.view === mode;
      b.classList.toggle('active', active);
      b.setAttribute('aria-pressed', String(active));
    });
    recomputeVisibleIndices();
    applyBeat(state.currentBeatIndex);
  });
});

dom.btnPrev.addEventListener('click', () => stepBeat(-1));
dom.btnNext.addEventListener('click', () => stepBeat(1));
dom.btnPlay.addEventListener('click', () => setPlaying(!state.isPlaying));

dom.btnSpeed.addEventListener('click', () => {
  state.playbackSpeed = state.playbackSpeed === 1 ? 1.5 : 1;
  dom.btnSpeed.textContent = `${state.playbackSpeed}x`;
  dom.btnSpeed.classList.toggle('active', state.playbackSpeed > 1);
  audio.setPlaybackRate(state.playbackSpeed);
});

dom.btnVoice.addEventListener('click', () => {
  const on = audio.toggleVoice();
  dom.btnVoice.textContent = on ? 'Voice: On' : 'Voice: Off';
  dom.btnVoice.setAttribute('aria-pressed', String(on));
  if (on) {
    const beat = state.beats[state.currentBeatIndex];
    if (beat && (beat.type === 'speech' || beat.type === 'thought')) {
      audio.speakBeat({
        text: beat.text,
        characterId: beat.characterId,
        audioUrl: beat.audioUrl,
        isThought: beat.type === 'thought',
      });
    }
  }
});

dom.btnAmbient.addEventListener('click', () => {
  const on = audio.toggleAmbient();
  dom.btnAmbient.textContent = on ? 'Room: On' : 'Room: Off';
  dom.btnAmbient.setAttribute('aria-pressed', String(on));
});

dom.btnSpoiler.addEventListener('click', () => {
  state.showSpoilers = !state.showSpoilers;
  updateSpoilerDisplay();
});

// Keyboard shortcuts: Space = Play/Pause, Left/Right = Prev/Next, 1/2/3 = Round Jump
window.addEventListener('keydown', (e) => {
  const tag = document.activeElement?.tagName;
  if (tag === 'SELECT' || tag === 'INPUT' || tag === 'TEXTAREA') return;

  if (e.code === 'Space') {
    e.preventDefault();
    setPlaying(!state.isPlaying);
  } else if (e.code === 'ArrowRight') {
    e.preventDefault();
    stepBeat(1);
  } else if (e.code === 'ArrowLeft') {
    e.preventDefault();
    stepBeat(-1);
  } else if (e.key === '1' || e.key === '2' || e.key === '3') {
    jumpToRound(Number(e.key));
  }
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ----------------------------------------------------------------------------
// Main Render Loop
// ----------------------------------------------------------------------------
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const elapsed = clock.elapsedTime;

  // 1. Update 3D props & character procedural animations
  table.update(dt, elapsed);
  if (rigA) rigA.update(dt, elapsed);
  if (rigB) rigB.update(dt, elapsed);

  // 2. Smooth lighting director transitions (including Noir Thought spotlight)
  const lightEase = 1 - Math.exp(-5.0 * dt);
  hemiLight.intensity = THREE.MathUtils.lerp(hemiLight.intensity, lightTargets.hemi, lightEase);
  ambientLight.intensity = THREE.MathUtils.lerp(ambientLight.intensity, lightTargets.ambient, lightEase);
  roomFillFront.intensity = THREE.MathUtils.lerp(roomFillFront.intensity, lightTargets.fillFront, lightEase);
  roomFillBack.intensity = THREE.MathUtils.lerp(roomFillBack.intensity, lightTargets.fillBack, lightEase);
  tableSpot.intensity = THREE.MathUtils.lerp(tableSpot.intensity, lightTargets.tableSpot, lightEase);
  noirSpot.intensity = THREE.MathUtils.lerp(noirSpot.intensity, lightTargets.noirSpot, lightEase);
  noirSpot.position.lerp(lightTargets.noirPos, lightEase);
  noirSpot.target.position.lerp(lightTargets.noirLook, lightEase);
  noirSpot.target.updateMatrixWorld();
  renderer.toneMappingExposure = THREE.MathUtils.lerp(
    renderer.toneMappingExposure,
    lightTargets.exposure,
    lightEase
  );

  // 3. Smooth Camera Director (with gentle subtle cinema breathing when locked)
  if (state.camMode === 'free') {
    controls.update();
    camCurrentTarget.copy(controls.target);
  } else {
    const camEase = 1 - Math.exp(-4.2 * dt);
    const swayX = Math.sin(elapsed * 0.42) * 0.018;
    const swayY = Math.cos(elapsed * 0.31) * 0.008;
    camera.position.x = THREE.MathUtils.lerp(camera.position.x, camDesiredPos.x + swayX, camEase);
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, camDesiredPos.y + swayY, camEase);
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, camDesiredPos.z, camEase);
    camCurrentTarget.lerp(camDesiredTarget, camEase);
    camera.lookAt(camCurrentTarget);
  }

  renderer.render(scene, camera);
}

// ----------------------------------------------------------------------------
// Bootstrap
// ----------------------------------------------------------------------------
async function init() {
  try {
    dom.loadingText.textContent = 'Loading apartment stage and 5 highlight matches...';

    const [dataRes] = await Promise.all([
      fetch('./data/highlights.json'),
      loadApartmentGLB(),
    ]);

    if (!dataRes.ok) {
      throw new Error(`Failed to fetch highlights.json (HTTP ${dataRes.status})`);
    }
    const payload = await dataRes.json();
    state.games = payload.games || [];

    // Populate 5-Game Selector
    dom.gameSelect.innerHTML = '';
    state.games.forEach((g, idx) => {
      const opt = document.createElement('option');
      opt.value = String(idx);
      opt.textContent = `${idx + 1}. ${g.title}`;
      dom.gameSelect.appendChild(opt);
    });

    if (state.games.length > 0) {
      loadGame(0, { autoplay: false });
    }
  } catch (err) {
    console.error('[Stage] Initialization error:', err);
    dom.beatText.textContent = `Could not load match archive: ${err.message}`;
  } finally {
    dom.loadingOverlay.classList.add('done');
  }
}

// Expose read-only inspection hook for debugging/testing
window.__THEATER__ = {
  state,
  loadGame,
  applyBeat,
  jumpToRound,
  setCameraMode,
};

animate();
init();
