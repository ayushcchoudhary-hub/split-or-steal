import * as THREE from '../vendor/three.module.js';

/**
 * Exact architectural dimensions and Seat A / Seat B anchor coordinates
 * relative to the negotiation table group origin (floor at y = 0).
 */
export const TABLE_METRICS = {
  length: 1.6, // X axis: -0.80 to +0.80
  width: 0.95, // Z axis: -0.475 to +0.475
  height: 0.74, // Tabletop top surface Y
  seatY: 0.46, // Chair seat cushion top surface Y
  seatA: { x: -1.05, y: 0.46, z: 0 }, // Faces +X
  seatB: { x: 1.05, y: 0.46, z: 0 }, // Faces -X
  cardA: { x: -0.44, y: 0.747, z: 0.0 },
  cardB: { x: 0.44, y: 0.747, z: 0.0 },
  potCenter: { x: 0, y: 0.74, z: 0 },
};

const COLOR_SPLIT = '#2F6FDB';
const COLOR_STEAL = '#EF4E14';
const COLOR_BRASS = '#C9A24B';

function createCanvas(w, h) {
  if (typeof document === 'undefined') return null;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

function toTexture(canvas) {
  if (!canvas) return null;
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Rich procedural walnut wood-grain canvas texture for the tabletop & legs.
 */
function createWalnutTexture() {
  const w = 512;
  const h = 512;
  const canvas = createCanvas(w, h);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');

  const grad = ctx.createLinearGradient(0, 0, w, h);
  grad.addColorStop(0, '#4A2F1F');
  grad.addColorStop(0.5, '#3C2417');
  grad.addColorStop(1, '#4E3221');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // Fine horizontal organic walnut grain waves
  for (let i = 0; i < 160; i++) {
    const y = (i / 160) * h;
    ctx.strokeStyle = i % 3 === 0 ? 'rgba(26, 14, 8, 0.28)' : 'rgba(110, 74, 46, 0.16)';
    ctx.lineWidth = 1 + (i % 4) * 0.6;
    ctx.beginPath();
    ctx.moveTo(0, y);
    for (let x = 0; x <= w; x += 32) {
      const wave = Math.sin(x * 0.015 + i * 0.42) * 3.5 + Math.cos(x * 0.032 - i * 0.2) * 1.8;
      ctx.lineTo(x, y + wave);
    }
    ctx.stroke();
  }

  const tex = toTexture(canvas);
  if (tex) {
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(2, 1.2);
  }
  return tex;
}

/**
 * Subtle matte dark architectural leather/felt mat texture with card zone markers.
 */
function createTableMatTexture(viewFromNegZ = false) {
  const w = 1024;
  const h = 512;
  const canvas = createCanvas(w, h);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');

  // Deep charcoal-slate ground
  const grad = ctx.createRadialGradient(w / 2, h / 2, 40, w / 2, h / 2, w * 0.6);
  grad.addColorStop(0, '#22262E');
  grad.addColorStop(1, '#16191F');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // Subtle architectural hairline border
  ctx.strokeStyle = 'rgba(201, 162, 75, 0.45)';
  ctx.lineWidth = 3;
  ctx.strokeRect(18, 18, w - 36, h - 36);

  // Subtle dashed center negotiation divider
  ctx.strokeStyle = 'rgba(201, 162, 75, 0.28)';
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.moveTo(w / 2, 34);
  ctx.lineTo(w / 2, h - 34);
  ctx.stroke();
  ctx.setLineDash([]);

  // Card placement frames on left (Seat A at -X) & right (Seat B at +X)
  const drawCardSlot = (cx, label) => {
    const sw = 186;
    const sh = 248;
    const sx = cx - sw / 2;
    const sy = h / 2 - sh / 2;
    ctx.strokeStyle = 'rgba(201, 162, 75, 0.34)';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.roundRect(sx, sy, sw, sh, 12);
    ctx.stroke();

    ctx.fillStyle = 'rgba(201, 162, 75, 0.42)';
    ctx.font = '600 18px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(label, cx, sy + sh + 26);
  };

  if (viewFromNegZ) {
    // Rotate canvas 180deg so text reads upright from -Z while keeping SEAT A at -X and SEAT B at +X
    ctx.save();
    ctx.translate(w / 2, h / 2);
    ctx.rotate(Math.PI);
    ctx.translate(-w / 2, -h / 2);
    drawCardSlot(512 + 336, 'SEAT A · DECISION');
    drawCardSlot(512 - 336, 'SEAT B · DECISION');
    ctx.restore();
  } else {
    // Mat width is 1.34m -> x = ±0.44m maps to cx = 512 ± (0.44 / 0.67) * 512 ≈ 512 ± 336
    drawCardSlot(512 - 336, 'SEAT A · DECISION');
    drawCardSlot(512 + 336, 'SEAT B · DECISION');
  }

  return toTexture(canvas);
}

/**
 * Creates a crisp editorial Decision Card face or back texture.
 * @param {'BACK'|'SPLIT'|'STEAL'} variant
 */
function createCardTexture(variant) {
  const w = 384;
  const h = 512;
  const canvas = createCanvas(w, h);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');

  if (variant === 'BACK') {
    const bg = ctx.createLinearGradient(0, 0, w, h);
    bg.addColorStop(0, '#23262E');
    bg.addColorStop(1, '#121419');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    // Ornate double brass frame
    ctx.strokeStyle = COLOR_BRASS;
    ctx.lineWidth = 8;
    ctx.strokeRect(18, 18, w - 36, h - 36);

    ctx.strokeStyle = 'rgba(201, 162, 75, 0.45)';
    ctx.lineWidth = 2;
    ctx.strokeRect(32, 32, w - 64, h - 64);

    // Center circular brass medallion
    ctx.strokeStyle = COLOR_BRASS;
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, 92, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = 'rgba(201, 162, 75, 0.1)';
    ctx.fill();

    // Crisp "?" brass emblem
    ctx.fillStyle = '#E0BE6F';
    ctx.font = '700 132px Georgia, "Times New Roman", serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('?', w / 2, h / 2 + 8);

    ctx.fillStyle = 'rgba(224, 190, 111, 0.72)';
    ctx.font = '600 18px "IBM Plex Mono", ui-monospace, monospace';
    ctx.fillText('SPLIT OR STEAL', w / 2, 74);
    ctx.fillText('SEALED VOTE', w / 2, h - 72);
    return toTexture(canvas);
  }

  const isSplit = variant === 'SPLIT';
  const primary = isSplit ? COLOR_SPLIT : COLOR_STEAL;
  const darkBg = isSplit ? '#183C7D' : '#8A2807';

  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, primary);
  bg.addColorStop(1, darkBg);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  // Crisp ivory editorial border
  ctx.strokeStyle = 'rgba(255, 250, 242, 0.85)';
  ctx.lineWidth = 8;
  ctx.strokeRect(18, 18, w - 36, h - 36);

  ctx.strokeStyle = 'rgba(255, 250, 242, 0.35)';
  ctx.lineWidth = 2;
  ctx.strokeRect(32, 32, w - 64, h - 64);

  // Top & bottom corner markers
  ctx.fillStyle = 'rgba(255, 250, 242, 0.88)';
  ctx.font = '700 22px "IBM Plex Mono", ui-monospace, monospace';
  ctx.textAlign = 'center';
  ctx.fillText(isSplit ? 'COOPERATE' : 'DEFECT', w / 2, 74);
  ctx.fillText(isSplit ? '50 / 50' : 'TAKE ALL', w / 2, h - 64);

  // Bold geometric center icon
  ctx.save();
  ctx.translate(w / 2, h / 2 - 48);
  ctx.strokeStyle = '#FFFDF9';
  ctx.fillStyle = '#FFFDF9';
  ctx.lineWidth = 8;
  if (isSplit) {
    // Interlocking twin balanced circles (mutual split)
    ctx.beginPath();
    ctx.arc(-28, 0, 38, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(28, 0, 38, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    // Sharp bold lightning/dagger diamond (steal)
    ctx.beginPath();
    ctx.moveTo(0, -52);
    ctx.lineTo(44, 0);
    ctx.lineTo(0, 52);
    ctx.lineTo(-44, 0);
    ctx.closePath();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, -26);
    ctx.lineTo(20, 0);
    ctx.lineTo(0, 26);
    ctx.lineTo(-20, 0);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();

  // Giant unmistakable action typography
  ctx.fillStyle = '#FFFFFF';
  ctx.font = '800 84px "Bricolage Grotesque", "Helvetica Neue", Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(variant, w / 2, h / 2 + 86);

  return toTexture(canvas);
}

/**
 * Draws the live pot plaque onto a shared canvas for dynamic stake / outcome updates.
 */
function drawPotPlaque(canvas, mainText, subText, accentColor = COLOR_BRASS) {
  if (!canvas) return;
  const w = canvas.width;
  const h = canvas.height;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#15171D';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 8;
  ctx.strokeRect(8, 8, w - 16, h - 16);

  ctx.fillStyle = accentColor;
  ctx.font = '700 26px "IBM Plex Mono", ui-monospace, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(subText, w / 2, 46);

  ctx.fillStyle = '#F8F4EC';
  ctx.font = '800 66px "Bricolage Grotesque", "Helvetica Neue", Arial, sans-serif';
  ctx.fillText(mainText, w / 2, 114);
}

/**
 * Creates a rounded rectangle THREE.Shape centered at (0, 0).
 */
function roundedRectShape(w, h, r) {
  const s = new THREE.Shape();
  const hw = w / 2;
  const hh = h / 2;
  s.moveTo(-hw + r, -hh);
  s.lineTo(hw - r, -hh);
  s.quadraticCurveTo(hw, -hh, hw, -hh + r);
  s.lineTo(hw, hh - r);
  s.quadraticCurveTo(hw, hh, hw - r, hh);
  s.lineTo(-hw + r, hh);
  s.quadraticCurveTo(-hw, hh, -hw, hh - r);
  s.lineTo(-hw, -hh + r);
  s.quadraticCurveTo(-hw, -hh, -hw + r, -hh);
  return s;
}

/**
 * Builds a sleek mid-century architectural dining/club chair at origin (floor y = 0),
 * seat cushion top at y = 0.46m, facing +X (rotated 180deg for Seat B).
 */
function buildChair(walnutMat, brassMat, leatherMat) {
  const chair = new THREE.Group();

  // Four angled tapered walnut legs + brass foot ferrules resting cleanly at y = 0
  const legPositions = [
    [-0.19, -0.19],
    [-0.19, 0.19],
    [0.19, -0.19],
    [0.19, 0.19],
  ];
  legPositions.forEach(([lx, lz]) => {
    const legGroup = new THREE.Group();
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.017, 0.012, 0.43, 12), walnutMat);
    leg.position.y = 0.215;
    legGroup.add(leg);

    const ferrule = new THREE.Mesh(new THREE.CylinderGeometry(0.013, 0.0115, 0.045, 12), brassMat);
    ferrule.position.y = 0.0225;
    legGroup.add(ferrule);

    legGroup.position.set(lx, 0, lz);
    legGroup.rotation.z = lx > 0 ? -0.07 : 0.07;
    legGroup.rotation.x = lz > 0 ? 0.07 : -0.07;
    chair.add(legGroup);
  });

  // Under-seat walnut frame & cross-rails
  const frame = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.036, 0.44), walnutMat);
  frame.position.y = 0.415;
  chair.add(frame);

  // Upholstered dark espresso leather seat cushion (top at y = 0.46m)
  const seatGeo = new THREE.BoxGeometry(0.47, 0.05, 0.47);
  const seat = new THREE.Mesh(seatGeo, leatherMat);
  seat.position.y = 0.435;
  chair.add(seat);

  // Backrest upright posts (positioned behind character torso at x = -0.23m)
  for (const sz of [-0.19, 0.19]) {
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.017, 0.48, 12), walnutMat);
    post.position.set(-0.235, 0.65, sz);
    post.rotation.z = 0.08;
    chair.add(post);

    // Subtle mid-century side armrest (clears table underside y = 0.685m)
    const armSupport = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.013, 0.20, 10), brassMat);
    armSupport.position.set(0.08, 0.53, sz * 1.15);
    chair.add(armSupport);

    const armRest = new THREE.Mesh(new THREE.CapsuleGeometry(0.018, 0.34, 8, 12), walnutMat);
    armRest.position.set(-0.06, 0.635, sz * 1.15);
    armRest.rotation.z = Math.PI / 2;
    chair.add(armRest);
  }

  // Sculpted curved bentwood walnut backrest wrapping behind seated character (-X = 1.5*PI in CylinderGeometry)
  const backrest = new THREE.Mesh(
    new THREE.CylinderGeometry(0.245, 0.25, 0.21, 24, 1, true, Math.PI * 1.16, Math.PI * 0.68),
    walnutMat
  );
  backrest.name = 'chairBackrest';
  backrest.position.set(-0.01, 0.77, 0);
  backrest.material = walnutMat.clone();
  backrest.material.side = THREE.DoubleSide;
  chair.add(backrest);

  // Brass trim strip across upper backrest spine (centered at -X, matching backrest)
  const brassSpine = new THREE.Mesh(
    new THREE.TorusGeometry(0.246, 0.0055, 8, 24, Math.PI * 0.66),
    brassMat
  );
  brassSpine.name = 'chairBrassSpine';
  brassSpine.position.set(-0.01, 0.872, 0);
  brassSpine.rotation.x = Math.PI / 2;
  brassSpine.rotation.z = Math.PI * 0.67;
  chair.add(brassSpine);

  return chair;
}

/**
 * Formats dollar amounts cleanly (`$100`, `$1,000`, etc.).
 */
function formatMoney(n) {
  const num = Math.max(0, Math.round(Number(n) || 0));
  return `$${num.toLocaleString('en-US')}`;
}

/**
 * Creates the architectural walnut-and-brass negotiation table, seated chairs,
 * dynamic 3D chip pot pedestal, and animated physical SPLIT/STEAL Decision Cards.
 *
 * Returns:
 * {
 *   group,
 *   seatAAnchor,
 *   seatBAnchor,
 *   metrics,
 *   setPot(stake),
 *   resetCards(),
 *   revealDecisions(actionA, actionB, payoutA, payoutB),
 *   update(dt, elapsed),
 *   dispose()
 * }
 */
export function createNegotiationTable(options = {}) {
  const viewFromNegZ = Boolean(options.viewFromNegZ);
  const cardYaw = viewFromNegZ ? Math.PI : 0;
  const group = new THREE.Group();
  group.name = 'negotiationTable';

  const textures = [];
  const walnutTex = createWalnutTexture();
  const matTex = createTableMatTexture(viewFromNegZ);
  const cardBackTex = createCardTexture('BACK');
  const cardSplitTex = createCardTexture('SPLIT');
  const cardStealTex = createCardTexture('STEAL');
  [walnutTex, matTex, cardBackTex, cardSplitTex, cardStealTex].forEach((t) => t && textures.push(t));

  // Core materials
  const walnutMat = new THREE.MeshStandardMaterial({
    color: 0x5d3d29,
    map: walnutTex || null,
    roughness: 0.44,
    metalness: 0.06,
  });
  const darkWalnutMat = new THREE.MeshStandardMaterial({
    color: 0x352217,
    map: walnutTex || null,
    roughness: 0.48,
    metalness: 0.08,
  });
  const brassMat = new THREE.MeshStandardMaterial({
    color: 0xcfa74e,
    roughness: 0.25,
    metalness: 0.88,
  });
  const leatherMat = new THREE.MeshStandardMaterial({
    color: 0x221b18,
    roughness: 0.62,
    metalness: 0.08,
  });
  const feltMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: matTex || null,
    roughness: 0.85,
    metalness: 0.04,
  });

  // --- 1. SLEEK WALNUT & BRASS ARCHITECTURAL TABLETOP (1.60m x 0.95m x 0.74m) ---
  const slabShape = roundedRectShape(1.58, 0.93, 0.075);
  const slabGeo = new THREE.ExtrudeGeometry(slabShape, {
    depth: 0.044,
    bevelEnabled: true,
    bevelThickness: 0.006,
    bevelSize: 0.01,
    bevelSegments: 3,
  });
  // ExtrudeGeometry extrudes along +Z; rotate -PI/2 around X so flat slab lies on XZ with top at y = 0.74
  slabGeo.rotateX(-Math.PI / 2);
  const tabletop = new THREE.Mesh(slabGeo, walnutMat);
  tabletop.position.y = 0.69; // 0.69 + 0.044 + 0.006 bevel = 0.740m top!
  group.add(tabletop);

  // Recessed brushed-brass perimeter inlay frame flush with tabletop surface
  const inlayOuter = new THREE.Mesh(new THREE.BoxGeometry(1.40, 0.003, 0.78), brassMat);
  inlayOuter.position.y = 0.7405;
  group.add(inlayOuter);

  // Dark architectural negotiation mat inset inside the brass inlay
  const tableMat = new THREE.Mesh(new THREE.BoxGeometry(1.36, 0.004, 0.74), feltMat);
  tableMat.position.y = 0.741;
  group.add(tableMat);

  // Under-apron structural beam
  const apron = new THREE.Mesh(new THREE.BoxGeometry(1.32, 0.05, 0.66), darkWalnutMat);
  apron.position.y = 0.66;
  group.add(apron);

  // Architectural flared walnut trestle legs + brass stretcher bar (grounded at y = 0)
  for (const sx of [-0.56, 0.56]) {
    for (const sz of [-0.31, 0.31]) {
      const legPivot = new THREE.Group();
      legPivot.position.set(sx, 0, sz);

      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.021, 0.68, 16), darkWalnutMat);
      leg.position.y = 0.34;
      legPivot.add(leg);

      const footCap = new THREE.Mesh(new THREE.CylinderGeometry(0.023, 0.021, 0.07, 16), brassMat);
      footCap.position.y = 0.035;
      legPivot.add(footCap);

      legPivot.rotation.z = sx > 0 ? -0.09 : 0.09;
      legPivot.rotation.x = sz > 0 ? 0.08 : -0.08;
      group.add(legPivot);
    }

    // Side cross-brace between front & back legs
    const sideBrace = new THREE.Mesh(new THREE.BoxGeometry(0.038, 0.038, 0.58), darkWalnutMat);
    sideBrace.position.set(sx, 0.21, 0);
    group.add(sideBrace);
  }

  // Brushed brass central architectural stretcher connecting the two trestles
  const stretcher = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, 1.12, 16), brassMat);
  stretcher.position.set(0, 0.21, 0);
  stretcher.rotation.z = Math.PI / 2;
  group.add(stretcher);

  // --- 2. TWO SEATED ARCHITECTURAL CHAIRS + CHARACTER MOUNT ANCHORS ---
  const chairA = buildChair(walnutMat, brassMat, leatherMat);
  chairA.name = 'chairA';
  chairA.position.set(TABLE_METRICS.seatA.x, 0, TABLE_METRICS.seatA.z);
  group.add(chairA);

  const chairB = buildChair(walnutMat, brassMat, leatherMat);
  chairB.name = 'chairB';
  chairB.position.set(TABLE_METRICS.seatB.x, 0, TABLE_METRICS.seatB.z);
  chairB.rotation.y = Math.PI;
  group.add(chairB);

  const seatAAnchor = new THREE.Group();
  seatAAnchor.name = 'seatAAnchor';
  seatAAnchor.position.set(TABLE_METRICS.seatA.x, TABLE_METRICS.seatA.y, TABLE_METRICS.seatA.z);
  group.add(seatAAnchor);

  const seatBAnchor = new THREE.Group();
  seatBAnchor.name = 'seatBAnchor';
  seatBAnchor.position.set(TABLE_METRICS.seatB.x, TABLE_METRICS.seatB.y, TABLE_METRICS.seatB.z);
  group.add(seatBAnchor);

  // --- 3. CENTER POT PEDESTAL + DUAL-SIDED POT PLAQUE + DYNAMIC 3D CHIPS ---
  const potGroup = new THREE.Group();
  potGroup.name = 'potGroup';
  potGroup.position.set(0, 0.743, 0);
  group.add(potGroup);

  const pedestalBase = new THREE.Mesh(
    new THREE.CylinderGeometry(0.145, 0.16, 0.018, 36),
    new THREE.MeshStandardMaterial({ color: 0x17191f, roughness: 0.36, metalness: 0.35 })
  );
  pedestalBase.position.y = 0.009;
  potGroup.add(pedestalBase);

  const pedestalTrim = new THREE.Mesh(new THREE.TorusGeometry(0.15, 0.0055, 10, 40), brassMat);
  pedestalTrim.position.y = 0.017;
  pedestalTrim.rotation.x = Math.PI / 2;
  potGroup.add(pedestalTrim);

  // Crisp angled dual-sided canvas pot plaque (readable from +Z and -Z)
  const plaqueCanvas = createCanvas(512, 168);
  const plaqueTex = toTexture(plaqueCanvas);
  if (plaqueTex) textures.push(plaqueTex);
  drawPotPlaque(plaqueCanvas, '$100', 'ROUND POT', COLOR_BRASS);

  const plaqueMat = new THREE.MeshStandardMaterial({
    map: plaqueTex || null,
    roughness: 0.32,
    metalness: 0.12,
  });

  function buildPlaqueStand(zSign) {
    const stand = new THREE.Group();
    stand.position.set(0, 0.036, zSign * 0.175);
    stand.rotation.y = zSign > 0 ? 0 : Math.PI;
    stand.rotation.x = -0.42;

    const backing = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.08, 0.01), brassMat);
    stand.add(backing);

    const face = new THREE.Mesh(new THREE.PlaneGeometry(0.228, 0.072), plaqueMat);
    face.position.z = 0.006;
    stand.add(face);
    return stand;
  }
  // Single clean camera-facing pot plaque (keeps silhouette behind the chip tray completely clean)
  potGroup.add(buildPlaqueStand(viewFromNegZ ? -1 : 1));

  // Container for dynamic 3D chip stacks (rebuilt by setPot)
  const chipsContainer = new THREE.Group();
  chipsContainer.name = 'chipsContainer';
  potGroup.add(chipsContainer);

  const chipGeo = new THREE.CylinderGeometry(0.024, 0.024, 0.0072, 20);
  const goldChipMat = new THREE.MeshStandardMaterial({
    color: 0xe6b94c,
    emissive: 0x5c3e08,
    emissiveIntensity: 0.25,
    roughness: 0.24,
    metalness: 0.86,
  });
  const ivoryChipMat = new THREE.MeshStandardMaterial({
    color: 0xefe7d6,
    emissive: 0x241d12,
    emissiveIntensity: 0.12,
    roughness: 0.38,
    metalness: 0.22,
  });
  const obsidianChipMat = new THREE.MeshStandardMaterial({
    color: 0x262b36,
    emissive: 0x101318,
    emissiveIntensity: 0.15,
    roughness: 0.34,
    metalness: 0.45,
  });

  let currentStake = 100;
  let chipItems = []; // { mesh, homePos, startPos, targetPos, targetRot, delay, isCrashed }

  function clearChips() {
    while (chipsContainer.children.length > 0) {
      chipsContainer.remove(chipsContainer.children[0]);
    }
    chipItems = [];
  }

  function populateChips(stake) {
    clearChips();
    goldChipMat.color.setHex(0xe6b94c);
    goldChipMat.emissiveIntensity = 0.25;
    ivoryChipMat.color.setHex(0xefe7d6);
    obsidianChipMat.color.setHex(0x262b36);

    // Define sculptural stack layouts scaled dynamically to $100 / $500 / $1,000
    let stacks;
    if (stake <= 150) {
      stacks = [
        { x: 0, z: 0, count: 6, tier: 'gold' },
        { x: -0.056, z: 0.016, count: 3, tier: 'ivory' },
        { x: 0.056, z: -0.016, count: 3, tier: 'ivory' },
      ];
    } else if (stake <= 600) {
      stacks = [
        { x: 0, z: 0, count: 10, tier: 'gold' },
        { x: -0.058, z: 0.032, count: 6, tier: 'gold' },
        { x: 0.058, z: 0.032, count: 6, tier: 'gold' },
        { x: -0.058, z: -0.032, count: 5, tier: 'obsidian' },
        { x: 0.058, z: -0.032, count: 5, tier: 'obsidian' },
      ];
    } else {
      // Grand $1,000 centerpiece bullion pyramid
      stacks = [
        { x: 0, z: 0, count: 14, tier: 'gold' },
        { x: -0.06, z: 0, count: 9, tier: 'gold' },
        { x: 0.06, z: 0, count: 9, tier: 'gold' },
        { x: -0.03, z: 0.054, count: 7, tier: 'gold' },
        { x: 0.03, z: 0.054, count: 7, tier: 'gold' },
        { x: -0.03, z: -0.054, count: 6, tier: 'obsidian' },
        { x: 0.03, z: -0.054, count: 6, tier: 'obsidian' },
      ];
    }

    const chipH = 0.0076;
    const baseY = 0.018 + chipH / 2;

    stacks.forEach((st, stackIdx) => {
      for (let i = 0; i < st.count; i++) {
        const mat =
          st.tier === 'gold'
            ? i % 4 === 3
              ? obsidianChipMat
              : goldChipMat
            : st.tier === 'ivory'
              ? i % 2 === 0
                ? ivoryChipMat
                : goldChipMat
              : i % 2 === 0
                ? obsidianChipMat
                : goldChipMat;

        const chip = new THREE.Mesh(chipGeo, mat);
        chip.castShadow = true;
        chip.receiveShadow = true;

        // Tiny organic stack jitter so it looks hand-stacked rather than robotic
        const jx = Math.sin(stackIdx * 7.1 + i * 2.3) * 0.0014;
        const jz = Math.cos(stackIdx * 5.3 + i * 3.1) * 0.0014;
        const homePos = new THREE.Vector3(st.x + jx, baseY + i * chipH, st.z + jz);
        chip.position.copy(homePos);
        chipsContainer.add(chip);

        chipItems.push({
          mesh: chip,
          homePos,
          startPos: homePos.clone(),
          targetPos: homePos.clone(),
          startRot: new THREE.Euler(0, 0, 0),
          targetRot: new THREE.Euler(0, 0, 0),
          delay: 0,
          arcHeight: 0.09,
          stackIdx,
          chipIdx: i,
        });
      }
    });
  }

  // --- 4. TWO PHYSICAL 3D DECISION CARDS + IMPACT GLOW RINGS ---
  const cardWidth = 0.21; // along X
  const cardHeight = 0.011; // thickness along Y
  const cardDepth = 0.275; // along Z
  const cardGeo = new THREE.BoxGeometry(cardWidth, cardHeight, cardDepth);
  const cardEdgeMat = new THREE.MeshStandardMaterial({
    color: 0xd0ac58,
    roughness: 0.3,
    metalness: 0.8,
  });

  function createCardActor(name, restX) {
    const faceMat = new THREE.MeshStandardMaterial({
      map: cardSplitTex || null,
      roughness: 0.35,
      metalness: 0.08,
    });
    const backMat = new THREE.MeshStandardMaterial({
      map: cardBackTex || null,
      roughness: 0.42,
      metalness: 0.16,
    });

    // BoxGeometry face order: +X, -X, +Y (top = face-down back), -Y (bottom = revealed face), +Z, -Z
    const materials = [cardEdgeMat, cardEdgeMat, backMat, faceMat, cardEdgeMat, cardEdgeMat];
    const mesh = new THREE.Mesh(cardGeo, materials);
    mesh.name = name;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    const restPos = new THREE.Vector3(restX, 0.743 + cardHeight / 2, 0);
    mesh.position.copy(restPos);
    mesh.rotation.y = cardYaw;
    group.add(mesh);

    // Expanding impact glow ring on the tabletop beneath each card
    const ringMat = new THREE.MeshBasicMaterial({
      color: COLOR_SPLIT,
      transparent: true,
      opacity: 0,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const shockRing = new THREE.Mesh(new THREE.RingGeometry(0.11, 0.145, 36), ringMat);
    shockRing.rotation.x = -Math.PI / 2;
    shockRing.position.set(restX, 0.7445, 0);
    shockRing.visible = false;
    group.add(shockRing);

    // Soft lingering under-card colored halo once revealed
    const haloMat = new THREE.MeshBasicMaterial({
      color: COLOR_SPLIT,
      transparent: true,
      opacity: 0,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const underHalo = new THREE.Mesh(new THREE.PlaneGeometry(cardWidth + 0.055, cardDepth + 0.055), haloMat);
    underHalo.rotation.x = -Math.PI / 2;
    underHalo.position.set(restX, 0.7438, 0);
    underHalo.visible = false;
    group.add(underHalo);

    return {
      mesh,
      faceMat,
      restPos,
      shockRing,
      ringMat,
      underHalo,
      haloMat,
      action: 'SPLIT',
      animating: false,
      revealed: false,
      elapsed: 0,
      delay: 0,
      impacted: false,
    };
  }

  const cardA = createCardActor('decisionCardA', TABLE_METRICS.cardA.x);
  const cardB = createCardActor('decisionCardB', TABLE_METRICS.cardB.x);

  // Chip transfer animation state
  const chipAnim = {
    active: false,
    elapsed: 0,
    duration: 0.78,
    mode: 'idle', // 'split' | 'winA' | 'winB' | 'crash'
  };

  // Enable clean shadows on table & chairs
  group.traverse((child) => {
    if (child.isMesh && child !== cardA.shockRing && child !== cardB.shockRing && child !== cardA.underHalo && child !== cardB.underHalo) {
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });

  // --- PUBLIC API ---
  function setPot(stake) {
    currentStake = Math.max(0, Number(stake) || 100);
    chipAnim.active = false;
    chipAnim.mode = 'idle';
    populateChips(currentStake);
    drawPotPlaque(plaqueCanvas, formatMoney(currentStake), 'ROUND POT', COLOR_BRASS);
    if (plaqueTex) plaqueTex.needsUpdate = true;
  }

  function resetCards() {
    [cardA, cardB].forEach((c) => {
      c.animating = false;
      c.revealed = false;
      c.elapsed = 0;
      c.impacted = false;
      c.mesh.position.copy(c.restPos);
      c.mesh.rotation.set(0, cardYaw, 0);
      c.shockRing.visible = false;
      c.ringMat.opacity = 0;
      c.underHalo.visible = false;
      c.haloMat.opacity = 0;
    });

    // Restore chips cleanly back to center pedestal for current stake
    setPot(currentStake);
  }

  function revealDecisions(actionA, actionB, payoutA = 0, payoutB = 0) {
    const actA = String(actionA || 'SPLIT').toUpperCase() === 'STEAL' ? 'STEAL' : 'SPLIT';
    const actB = String(actionB || 'SPLIT').toUpperCase() === 'STEAL' ? 'STEAL' : 'SPLIT';

    // Reset chip positions cleanly to center before launching transfer
    populateChips(currentStake);

    const setupCard = (card, action, delay) => {
      card.action = action;
      card.faceMat.map = action === 'STEAL' ? cardStealTex : cardSplitTex;
      card.faceMat.needsUpdate = true;
      const colorHex = action === 'STEAL' ? COLOR_STEAL : COLOR_SPLIT;
      card.ringMat.color.set(colorHex);
      card.haloMat.color.set(colorHex);
      card.mesh.position.copy(card.restPos);
      card.mesh.rotation.set(0, cardYaw, 0);
      card.shockRing.visible = false;
      card.underHalo.visible = false;
      card.animating = true;
      card.revealed = false;
      card.impacted = false;
      card.elapsed = 0;
      card.delay = delay;
    };

    setupCard(cardA, actA, 0.0);
    setupCard(cardB, actB, 0.11); // slight dramatic stagger

    // Plan chip transfer trajectory triggered right after card slam
    const pA = Number(payoutA) || 0;
    const pB = Number(payoutB) || 0;
    chipAnim.active = true;
    chipAnim.elapsed = 0;

    const chipH = 0.0076;
    const tableChipBaseY = chipH / 2 + 0.002;

    if (pA > 0 && pB > 0) {
      chipAnim.mode = 'split';
      let countA = 0;
      let countB = 0;
      chipItems.forEach((item, idx) => {
        item.startPos.copy(item.homePos);
        item.startRot.set(0, 0, 0);
        item.targetRot.set(0, 0, 0);
        const toA = idx % 2 === 0;
        const stackN = toA ? countA++ : countB++;
        const col = stackN % 2;
        const row = Math.floor(stackN / 2);
        const destX = toA ? -0.62 - col * 0.056 : 0.62 + col * 0.056;
        const destZ = (col === 0 ? -0.18 : -0.12) + (toA ? 0 : 0.3);
        item.targetPos.set(destX, tableChipBaseY + row * chipH, destZ);
        item.delay = 0.68 + (idx / Math.max(1, chipItems.length)) * 0.22;
        item.arcHeight = 0.11 + (idx % 3) * 0.025;
      });
      drawPotPlaque(plaqueCanvas, `${formatMoney(pA)} / ${formatMoney(pB)}`, 'MUTUAL SPLIT', COLOR_SPLIT);
    } else if (pA > 0 && pB === 0) {
      chipAnim.mode = 'winA';
      chipItems.forEach((item, idx) => {
        item.startPos.copy(item.homePos);
        item.startRot.set(0, 0, 0);
        item.targetRot.set(0, 0, 0);
        const col = idx % 3;
        const row = Math.floor(idx / 3);
        item.targetPos.set(-0.62 - (col % 2) * 0.054, tableChipBaseY + row * chipH, -0.21 + col * 0.055);
        item.delay = 0.68 + (idx / Math.max(1, chipItems.length)) * 0.25;
        item.arcHeight = 0.12 + (idx % 4) * 0.02;
      });
      drawPotPlaque(plaqueCanvas, `A +${formatMoney(pA)}`, 'SEAT A STEALS', COLOR_STEAL);
    } else if (pB > 0 && pA === 0) {
      chipAnim.mode = 'winB';
      chipItems.forEach((item, idx) => {
        item.startPos.copy(item.homePos);
        item.startRot.set(0, 0, 0);
        item.targetRot.set(0, 0, 0);
        const col = idx % 3;
        const row = Math.floor(idx / 3);
        item.targetPos.set(0.62 + (col % 2) * 0.054, tableChipBaseY + row * chipH, 0.18 - col * 0.055);
        item.delay = 0.68 + (idx / Math.max(1, chipItems.length)) * 0.25;
        item.arcHeight = 0.12 + (idx % 4) * 0.02;
      });
      drawPotPlaque(plaqueCanvas, `B +${formatMoney(pB)}`, 'SEAT B STEALS', COLOR_STEAL);
    } else {
      // Mutual STEAL/STEAL crash: scatter outward, topple, and dim to $0
      chipAnim.mode = 'crash';
      chipItems.forEach((item, idx) => {
        item.startPos.copy(item.homePos);
        item.startRot.set(0, 0, 0);
        const angle = idx * 2.39996; // golden angle scatter
        const dist = 0.14 + ((idx * 37) % 100) * 0.0017;
        item.targetPos.set(Math.cos(angle) * dist, tableChipBaseY + 0.002, Math.sin(angle) * dist * 0.85);
        item.targetRot.set(
          Math.sin(idx * 1.9) * 0.18,
          angle,
          Math.cos(idx * 2.7) * 0.18
        );
        item.delay = 0.64 + (idx % 5) * 0.03;
        item.arcHeight = 0.08 + (idx % 3) * 0.03;
      });
      drawPotPlaque(plaqueCanvas, '$0 · CRASH', 'MUTUAL STEAL', COLOR_STEAL);
    }

    if (plaqueTex) plaqueTex.needsUpdate = true;
  }

  function updateCardActor(card, dt, elapsed) {
    if (card.animating) {
      card.elapsed += dt;
      const localT = card.elapsed - card.delay;
      const flipDuration = 0.56;
      const settleDuration = 0.18;

      if (localT < 0) {
        // Pre-launch slight anticipation crouch
        const prep = Math.max(0, 1 + localT / Math.max(0.05, card.delay));
        card.mesh.position.y = card.restPos.y + Math.sin(prep * Math.PI) * 0.006;
      } else if (localT < flipDuration) {
        const u = localT / flipDuration;
        // Parabolic upward leap peaking at +0.29m, crisp slam onto table
        const lift = Math.sin(u * Math.PI) * 0.29 * (1 - 0.15 * u);
        // Smooth S-curve 180deg flip around X axis so -Y face lands upright on +Y
        const easeFlip = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
        card.mesh.position.set(card.restPos.x, card.restPos.y + lift, card.restPos.z);
        card.mesh.rotation.set(easeFlip * Math.PI, cardYaw, Math.sin(u * Math.PI) * 0.12 * (card === cardA ? 1 : -1));
      } else {
        // Card slam impact + micro-bounce + shockwave ring
        if (!card.impacted) {
          card.impacted = true;
          card.shockRing.visible = true;
          card.underHalo.visible = true;
        }
        const postT = localT - flipDuration;
        const bounce = Math.exp(-postT * 22) * Math.abs(Math.sin(postT * 36)) * 0.016;
        card.mesh.position.set(card.restPos.x, card.restPos.y + bounce, card.restPos.z);
        card.mesh.rotation.set(Math.PI, cardYaw, 0);

        // Expanding glowing impact ring
        const ringU = Math.min(1, postT / 0.55);
        const ringScale = 1.0 + ringU * 1.45;
        card.shockRing.scale.set(ringScale, ringScale, 1);
        card.ringMat.opacity = (1 - ringU) * 0.85;
        if (ringU >= 1) card.shockRing.visible = false;

        // Lingering soft under-card highlight
        card.haloMat.opacity = Math.min(0.45, postT * 2.2);

        if (postT >= settleDuration + 0.45) {
          card.animating = false;
          card.revealed = true;
          card.mesh.position.copy(card.restPos);
          card.mesh.rotation.set(Math.PI, cardYaw, 0);
        }
      }
    } else if (card.revealed) {
      // Gentle breathing glow under revealed face-up card
      card.underHalo.visible = true;
      card.haloMat.opacity = 0.28 + Math.sin(elapsed * 3.0) * 0.08;
    }
  }

  function update(dt = 0.016, elapsed = 0) {
    const clampedDt = Math.min(Math.max(dt, 0.0001), 0.1);

    updateCardActor(cardA, clampedDt, elapsed);
    updateCardActor(cardB, clampedDt, elapsed);

    if (chipAnim.active) {
      chipAnim.elapsed += clampedDt;
      let allDone = true;

      for (let i = 0; i < chipItems.length; i++) {
        const item = chipItems[i];
        const t = (chipAnim.elapsed - item.delay) / chipAnim.duration;
        if (t <= 0) {
          allDone = false;
          continue;
        }
        const u = Math.min(1, t);
        if (u < 1) allDone = false;

        // Smooth cubic ease-out glide with parabolic arc
        const ease = 1 - Math.pow(1 - u, 3);
        item.mesh.position.lerpVectors(item.startPos, item.targetPos, ease);
        item.mesh.position.y += Math.sin(u * Math.PI) * item.arcHeight;

        item.mesh.rotation.set(
          item.startRot.x + (item.targetRot.x - item.startRot.x) * ease,
          item.startRot.y + (item.targetRot.y - item.startRot.y) * ease,
          item.startRot.z + (item.targetRot.z - item.startRot.z) * ease
        );
      }

      // On mutual STEAL crash, smoothly dim & desaturate the scattered chips
      if (chipAnim.mode === 'crash' && chipAnim.elapsed > 0.68) {
        const dimU = Math.min(1, (chipAnim.elapsed - 0.68) / 0.7);
        goldChipMat.color.lerp(new THREE.Color(0x55524d), dimU * 0.12);
        goldChipMat.emissiveIntensity = Math.max(0, 0.25 * (1 - dimU));
        ivoryChipMat.color.lerp(new THREE.Color(0x63605b), dimU * 0.12);
      }

      if (allDone) {
        chipAnim.active = false;
      }
    }
  }

  function dispose() {
    group.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        mats.forEach((m) => m.dispose());
      }
    });
    textures.forEach((tex) => tex && tex.dispose());
    textures.length = 0;
  }

  // Initialize default $100 pot on creation
  setPot(100);

  return {
    group,
    seatAAnchor,
    seatBAnchor,
    metrics: TABLE_METRICS,
    setPot,
    resetCards,
    revealDecisions,
    update,
    dispose,
  };
}
