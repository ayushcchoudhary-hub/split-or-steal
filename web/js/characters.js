import * as THREE from '../vendor/three.module.js';

/**
 * Character cast metadata for Split or Steal 3D Apartment Replay Theater.
 */
export const CHARACTER_META = {
  machiavelli: {
    id: 'machiavelli',
    name: 'Niccolò Machiavelli',
    shortName: 'Machiavelli',
    monogram: 'NM',
    era: '1469–1527',
    title: 'Florentine Diplomat',
    accentColor: '#B52B27',
    secondaryColor: '#D4A553',
    palette: {
      primary: '#8B1D22',
      secondary: '#261B1A',
      skin: '#DEC2A2',
      trim: '#C99B49',
      accent: '#B52B27',
      glow: '#E54B34',
      hair: '#2B201B',
    },
  },
  gandhi: {
    id: 'gandhi',
    name: 'Mahatma Gandhi',
    shortName: 'Gandhi',
    monogram: 'MG',
    era: '1869–1948',
    title: 'Architect of Satyagraha',
    accentColor: '#D99B52',
    secondaryColor: '#F2EAE0',
    palette: {
      primary: '#EFE7DA',
      secondary: '#D9CEBC',
      skin: '#B7815A',
      trim: '#C7964A',
      accent: '#D99B52',
      glow: '#F2BC6E',
      hair: '#3E3630',
    },
  },
  richard_nixon: {
    id: 'richard_nixon',
    name: 'Richard Nixon',
    shortName: 'Nixon',
    monogram: 'RN',
    era: '1913–1994',
    title: '37th U.S. President',
    accentColor: '#C42B2B',
    secondaryColor: '#2B313B',
    palette: {
      primary: '#242933',
      secondary: '#171A21',
      skin: '#E1BF9F',
      trim: '#C42B2B',
      accent: '#C42B2B',
      glow: '#E84A43',
      hair: '#2E3138',
    },
  },
  oscar_wilde: {
    id: 'oscar_wilde',
    name: 'Oscar Wilde',
    shortName: 'Wilde',
    monogram: 'OW',
    era: '1854–1900',
    title: 'Master of Paradox',
    accentColor: '#1E825C',
    secondaryColor: '#4EE08E',
    palette: {
      primary: '#124A36',
      secondary: '#0B2D21',
      skin: '#ECD2BE',
      trim: '#45D482',
      accent: '#1E825C',
      glow: '#3FD989',
      hair: '#3E2B1F',
    },
  },
  warren_buffett: {
    id: 'warren_buffett',
    name: 'Warren Buffett',
    shortName: 'Buffett',
    monogram: 'WB',
    era: '1930–present',
    title: 'Oracle of Omaha',
    accentColor: '#2B5B9E',
    secondaryColor: '#DCE1EA',
    palette: {
      primary: '#1E3354',
      secondary: '#14233B',
      skin: '#E4C2A5',
      trim: '#9C343A',
      accent: '#2B5B9E',
      glow: '#4C8CE6',
      hair: '#D6DBE3',
    },
  },
  van_gogh: {
    id: 'van_gogh',
    name: 'Vincent van Gogh',
    shortName: 'Van Gogh',
    monogram: 'VG',
    era: '1853–1890',
    title: 'Painter of Arles',
    accentColor: '#E0A122',
    secondaryColor: '#28559A',
    palette: {
      primary: '#275194',
      secondary: '#1B3A6B',
      skin: '#E3BF9E',
      trim: '#E0A122',
      accent: '#E0A122',
      glow: '#F2B830',
      hair: '#B94E1E',
    },
  },
};

/**
 * Seat mounting coordinates relative to the negotiation table origin (floor y = 0).
 * Character rig local origin (0, 0, 0) sits flush on the chair seat cushion (y = 0.46m).
 */
export const SEAT_CONFIG = {
  A: { x: -1.05, y: 0.46, z: 0, facingY: Math.PI / 2 },
  B: { x: 1.05, y: 0.46, z: 0, facingY: -Math.PI / 2 },
};

const VALID_STATES = new Set([
  'idle',
  'speaking',
  'thinking',
  'reveal_split',
  'reveal_steal',
  'react_win',
  'react_betrayed',
  'react_crash',
]);

function createCanvas(w, h) {
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  return canvas;
}

function makeCanvasTexture(canvas) {
  if (!canvas) return null;
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Procedural high-DPI portrait medallion / monogram emblem texture for the plinth front.
 */
function createMedallionTexture(meta) {
  const size = 256;
  const canvas = createCanvas(size, size);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  const cx = size / 2;
  const cy = size / 2;

  // Deep radial medallion ground
  const bgGrad = ctx.createRadialGradient(cx, cy - 26, 12, cx, cy, 124);
  bgGrad.addColorStop(0, '#2C2623');
  bgGrad.addColorStop(0.65, '#171413');
  bgGrad.addColorStop(1, '#0E0C0B');
  ctx.fillStyle = bgGrad;
  ctx.beginPath();
  ctx.arc(cx, cy, 124, 0, Math.PI * 2);
  ctx.fill();

  // Outer brass ring & guilloché ticks
  ctx.strokeStyle = '#D4AF5A';
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.arc(cx, cy, 118, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = meta.accentColor;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(cx, cy, 106, 0, Math.PI * 2);
  ctx.stroke();

  // Subtle decorative ticks around inner rim
  ctx.save();
  ctx.translate(cx, cy);
  ctx.strokeStyle = 'rgba(212, 175, 90, 0.45)';
  ctx.lineWidth = 2;
  for (let i = 0; i < 36; i++) {
    ctx.rotate((Math.PI * 2) / 36);
    ctx.beginPath();
    ctx.moveTo(0, -98);
    ctx.lineTo(0, -91);
    ctx.stroke();
  }
  ctx.restore();

  // Character-specific iconic mini-emblem at upper medallion
  ctx.save();
  ctx.translate(cx, cy - 42);
  ctx.strokeStyle = meta.accentColor;
  ctx.fillStyle = meta.accentColor;
  ctx.lineWidth = 3.5;

  if (meta.id === 'machiavelli') {
    // Florentine hood silhouette & diamond crest
    ctx.beginPath();
    ctx.moveTo(-26, 14);
    ctx.quadraticCurveTo(-24, -18, 0, -24);
    ctx.quadraticCurveTo(24, -18, 26, 14);
    ctx.lineTo(14, 18);
    ctx.lineTo(0, 6);
    ctx.lineTo(-14, 18);
    ctx.closePath();
    ctx.fill();
  } else if (meta.id === 'gandhi') {
    // Iconic wire spectacles
    ctx.strokeStyle = '#E5BE72';
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    ctx.arc(-16, 0, 12, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(16, 0, 12, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-4, -1);
    ctx.quadraticCurveTo(0, -6, 4, -1);
    ctx.stroke();
  } else if (meta.id === 'richard_nixon') {
    // Three presidential stars & power tie chevron
    for (const sx of [-22, 0, 22]) {
      ctx.beginPath();
      ctx.arc(sx, -8, 4.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.beginPath();
    ctx.moveTo(0, 2);
    ctx.lineTo(9, 14);
    ctx.lineTo(0, 24);
    ctx.lineTo(-9, 14);
    ctx.closePath();
    ctx.fill();
  } else if (meta.id === 'oscar_wilde') {
    // Green carnation rosette
    ctx.fillStyle = '#45E088';
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      ctx.beginPath();
      ctx.arc(Math.cos(a) * 10, Math.sin(a) * 10, 7, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = '#14523A';
    ctx.beginPath();
    ctx.arc(0, 0, 6, 0, Math.PI * 2);
    ctx.fill();
  } else if (meta.id === 'warren_buffett') {
    // Tortoiseshell glasses + rising compounding curve
    ctx.strokeStyle = '#6FA7F0';
    ctx.lineWidth = 3.5;
    ctx.strokeRect(-26, -10, 21, 16);
    ctx.strokeRect(5, -10, 21, 16);
    ctx.beginPath();
    ctx.moveTo(-5, -4);
    ctx.lineTo(5, -4);
    ctx.stroke();
  } else if (meta.id === 'van_gogh') {
    // Sunlit Provence swirling sun
    ctx.fillStyle = '#F0B632';
    ctx.beginPath();
    ctx.arc(0, 0, 12, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#F0B632';
    ctx.lineWidth = 3;
    for (let i = 0; i < 10; i++) {
      const a = (i / 10) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(Math.cos(a) * 16, Math.sin(a) * 16);
      ctx.lineTo(Math.cos(a) * 24, Math.sin(a) * 24);
      ctx.stroke();
    }
  }
  ctx.restore();

  // Crisp editorial monogram
  ctx.fillStyle = '#F6EFE4';
  ctx.font = '700 66px Georgia, "Times New Roman", serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(meta.monogram, cx, cy + 28);

  // Subtle era ribbon below monogram
  ctx.fillStyle = '#D4AF5A';
  ctx.font = '600 17px "IBM Plex Mono", ui-monospace, monospace';
  ctx.fillText(meta.era, cx, cy + 74);

  return makeCanvasTexture(canvas);
}

/**
 * Crisp floating editorial nameplate sprite texture above each seated figurine.
 */
function createNameplateTexture(meta, seat) {
  const w = 512;
  const h = 136;
  const canvas = createCanvas(w, h);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');

  // Rounded dark editorial plaque with subtle border
  const pad = 10;
  const r = 20;
  const x = pad;
  const y = pad;
  const bw = w - pad * 2;
  const bh = h - pad * 2;

  ctx.clearRect(0, 0, w, h);

  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + bw - r, y);
  ctx.quadraticCurveTo(x + bw, y, x + bw, y + r);
  ctx.lineTo(x + bw, y + bh - r);
  ctx.quadraticCurveTo(x + bw, y + bh, x + bw - r, y + bh);
  ctx.lineTo(x + r, y + bh);
  ctx.quadraticCurveTo(x, y + bh, x, y + bh - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();

  ctx.fillStyle = 'rgba(17, 19, 24, 0.88)';
  ctx.fill();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = 'rgba(218, 188, 128, 0.42)';
  ctx.stroke();

  // Left accent bar
  ctx.save();
  ctx.clip();
  ctx.fillStyle = meta.accentColor;
  ctx.fillRect(x, y, 14, bh);
  ctx.restore();

  // Seat pill top-right
  const pillW = 92;
  const pillH = 28;
  const px = x + bw - pillW - 18;
  const py = y + 18;
  ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.strokeStyle = meta.accentColor;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.roundRect(px, py, pillW, pillH, 8);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#EAE4D9';
  ctx.font = '700 15px "IBM Plex Mono", ui-monospace, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(`SEAT ${seat}`, px + pillW / 2, py + pillH / 2 + 1);

  // Character name
  ctx.fillStyle = '#F7F3EC';
  ctx.font = '700 34px "Bricolage Grotesque", "Helvetica Neue", Arial, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(meta.name, x + 32, y + 58);

  // Title & Era line
  ctx.fillStyle = '#BEB6A8';
  ctx.font = '500 20px "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif';
  ctx.fillText(`${meta.title}  ·  ${meta.era}`, x + 32, y + 92);

  return makeCanvasTexture(canvas);
}

/**
 * Procedural woven straw texture for Van Gogh's iconic Arles hat.
 */
function createStrawTexture() {
  const size = 256;
  const canvas = createCanvas(size, size);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#DFAA38';
  ctx.fillRect(0, 0, size, size);

  for (let y = 0; y < size; y += 6) {
    ctx.fillStyle = y % 12 === 0 ? 'rgba(184, 122, 22, 0.35)' : 'rgba(255, 232, 156, 0.35)';
    ctx.fillRect(0, y, size, 3);
    for (let x = 0; x < size; x += 16) {
      ctx.fillStyle = 'rgba(130, 80, 14, 0.16)';
      ctx.fillRect(x + ((y / 6) % 2) * 8, y, 6, 4);
    }
  }
  const tex = makeCanvasTexture(canvas);
  if (tex) {
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(3, 2);
  }
  return tex;
}

function enableShadows(obj, cast = true, receive = true) {
  obj.traverse((child) => {
    if (child.isMesh) {
      child.castShadow = cast;
      child.receiveShadow = receive;
    }
  });
}

/**
 * Helper to create a clean articulated stylized arm (upper arm + elbow + forearm + hand)
 * pivoted at the shoulder.
 */
function buildArm(side, coatMat, cuffMat, skinMat) {
  const sign = side === 'left' ? 1 : -1;
  const armGroup = new THREE.Group();
  armGroup.name = side === 'left' ? 'leftArm' : 'rightArm';

  // Shoulder cap
  const shoulder = new THREE.Mesh(new THREE.SphereGeometry(0.056, 16, 14), coatMat);
  armGroup.add(shoulder);

  // Upper arm descending slightly outward & forward
  const upperGeo = new THREE.CapsuleGeometry(0.044, 0.14, 8, 14);
  const upper = new THREE.Mesh(upperGeo, coatMat);
  upper.position.set(sign * 0.015, -0.095, 0.025);
  upper.rotation.x = 0.24;
  upper.rotation.z = sign * 0.14;
  armGroup.add(upper);

  // Forearm pivoting from elbow reaching forward toward table
  const forearmGroup = new THREE.Group();
  forearmGroup.position.set(sign * 0.025, -0.175, 0.055);
  armGroup.add(forearmGroup);

  const elbow = new THREE.Mesh(new THREE.SphereGeometry(0.042, 12, 12), coatMat);
  forearmGroup.add(elbow);

  const foreGeo = new THREE.CapsuleGeometry(0.036, 0.135, 8, 14);
  const forearm = new THREE.Mesh(foreGeo, coatMat);
  forearm.position.set(-sign * 0.02, -0.015, 0.095);
  forearm.rotation.x = Math.PI / 2 - 0.12;
  forearm.rotation.z = sign * 0.18;
  forearmGroup.add(forearm);

  // Crisp cuff ring
  const cuff = new THREE.Mesh(new THREE.CylinderGeometry(0.037, 0.039, 0.024, 16), cuffMat);
  cuff.position.set(-sign * 0.036, -0.024, 0.172);
  cuff.rotation.x = Math.PI / 2 - 0.12;
  cuff.rotation.z = sign * 0.18;
  forearmGroup.add(cuff);

  // Stylized sculpted hand
  const hand = new THREE.Group();
  hand.position.set(-sign * 0.044, -0.026, 0.212);
  const palm = new THREE.Mesh(new THREE.SphereGeometry(0.034, 14, 12), skinMat);
  palm.scale.set(0.88, 0.62, 1.25);
  hand.add(palm);

  const thumb = new THREE.Mesh(new THREE.CapsuleGeometry(0.011, 0.026, 6, 8), skinMat);
  thumb.position.set(-sign * 0.026, 0.008, 0.01);
  thumb.rotation.x = Math.PI / 2.2;
  thumb.rotation.z = -sign * 0.5;
  hand.add(thumb);

  forearmGroup.add(hand);

  return { armGroup, forearmGroup, hand };
}

/**
 * Builds the core sculpted head, facial planes, eyes, nose, ears, and articulated jaw/chin.
 */
function buildSculptedHead(skinMat, eyeMat, eyebrowMat, opts = {}) {
  const headGroup = new THREE.Group();
  headGroup.name = 'headGroup';

  const headScaleX = opts.headScaleX ?? 1.0;
  const headScaleY = opts.headScaleY ?? 1.12;
  const headScaleZ = opts.headScaleZ ?? 1.04;

  // Sculpted neck
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.056, 0.068, 0.095, 18), skinMat);
  neck.position.set(0, 0.035, -0.005);
  headGroup.add(neck);

  // Main cranium
  const cranium = new THREE.Mesh(new THREE.SphereGeometry(0.112, 24, 22), skinMat);
  cranium.position.set(0, 0.158, 0.004);
  cranium.scale.set(headScaleX, headScaleY, headScaleZ);
  headGroup.add(cranium);

  // Subtle sculpted cheekbones
  for (const s of [-1, 1]) {
    const cheek = new THREE.Mesh(new THREE.SphereGeometry(0.046, 14, 12), skinMat);
    cheek.position.set(s * 0.056, 0.132, 0.072);
    cheek.scale.set(0.95, 0.78, 0.72);
    headGroup.add(cheek);

    // Sculpted ears
    const ear = new THREE.Mesh(new THREE.SphereGeometry(0.024, 12, 10), skinMat);
    ear.position.set(s * 0.106 * headScaleX, 0.15, -0.004);
    ear.scale.set(0.42, 1.15, 0.75);
    headGroup.add(ear);
  }

  // Brow ridge
  const browHeight = opts.heavyBrow ? 0.026 : 0.017;
  const browDepth = opts.heavyBrow ? 0.042 : 0.028;
  const brow = new THREE.Mesh(
    new THREE.BoxGeometry(0.128, browHeight, browDepth),
    skinMat
  );
  brow.position.set(0, 0.184, 0.094);
  headGroup.add(brow);

  // Expressive eyebrows
  for (const s of [-1, 1]) {
    const eb = new THREE.Mesh(new THREE.CapsuleGeometry(0.0065, 0.032, 6, 8), eyebrowMat);
    eb.position.set(s * 0.036, 0.186, 0.108 + (opts.heavyBrow ? 0.006 : 0));
    eb.rotation.z = Math.PI / 2 + s * (opts.browTilt ?? 0.06);
    headGroup.add(eb);
  }

  // Recessed expressive eyes with subtle warm catchlight
  const glintMat = new THREE.MeshBasicMaterial({ color: 0xfef7ea });
  for (const s of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.0125, 12, 10), eyeMat);
    eye.position.set(s * 0.036, 0.162, 0.104);
    eye.scale.set(1.05, 0.85, 0.65);
    headGroup.add(eye);

    const glint = new THREE.Mesh(new THREE.SphereGeometry(0.0036, 8, 6), glintMat);
    glint.position.set(s * 0.036 + 0.003, 0.165, 0.111);
    headGroup.add(glint);
  }

  // Sculpted nose bridge
  const noseLen = opts.noseLength ?? 0.048;
  const noseProj = opts.noseProjection ?? 0.028;
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.018, noseLen, 6), skinMat);
  nose.position.set(0, 0.144, 0.112 + (noseProj - 0.028) * 0.6);
  nose.rotation.x = Math.PI * 0.44;
  nose.scale.set(0.85, 1.0, noseProj / 0.028);
  headGroup.add(nose);

  // Articulated jaw / chin group (moves subtly while speaking)
  const jawOrChinGroup = new THREE.Group();
  jawOrChinGroup.name = 'jawOrChinGroup';
  jawOrChinGroup.position.set(0, 0.114, 0.035);
  headGroup.add(jawOrChinGroup);

  const jawWidth = opts.angularJaw ? 0.094 : 0.078;
  const jawMesh = new THREE.Mesh(
    opts.angularJaw
      ? new THREE.BoxGeometry(jawWidth, 0.056, 0.09)
      : new THREE.SphereGeometry(0.068, 16, 14),
    skinMat
  );
  if (opts.angularJaw) {
    jawMesh.position.set(0, -0.016, 0.024);
  } else {
    jawMesh.position.set(0, -0.018, 0.022);
    jawMesh.scale.set(0.95, 0.84, 1.02);
  }
  jawOrChinGroup.add(jawMesh);

  // Subtle sculpted mouth line
  const mouthMat = new THREE.MeshStandardMaterial({
    color: 0x6f4338,
    roughness: 0.72,
    metalness: 0.02,
  });
  const mouth = new THREE.Mesh(new THREE.CapsuleGeometry(0.0048, 0.032, 6, 8), mouthMat);
  mouth.position.set(0, -0.006, 0.072);
  mouth.rotation.z = Math.PI / 2;
  jawOrChinGroup.add(mouth);

  return { headGroup, jawOrChinGroup };
}

/**
 * Character-specific costume, silhouette, headgear, spectacles, and side props.
 */
function dressCharacter(id, meta, parts, resources) {
  const { torsoGroup, headGroup, jawOrChinGroup, leftArmObj, rightArmObj, skinMat, brassMat } = parts;
  const p = meta.palette;

  const coatMat = new THREE.MeshStandardMaterial({
    color: p.primary,
    roughness: id === 'oscar_wilde' ? 0.62 : 0.68,
    metalness: id === 'machiavelli' ? 0.14 : 0.06,
  });
  const secondaryMat = new THREE.MeshStandardMaterial({
    color: p.secondary,
    roughness: 0.72,
    metalness: 0.08,
  });
  const trimMat = new THREE.MeshStandardMaterial({
    color: p.trim,
    roughness: 0.36,
    metalness: 0.35,
  });
  const shirtMat = new THREE.MeshStandardMaterial({
    color: 0xf4f0e8,
    roughness: 0.8,
    metalness: 0.02,
  });
  const hairMat = new THREE.MeshStandardMaterial({
    color: p.hair,
    roughness: 0.75,
    metalness: 0.05,
  });

  // Base sculpted torso silhouette (tailored bust/jacket)
  const chestGeo = new THREE.CylinderGeometry(0.156, 0.132, 0.31, 24);
  const chest = new THREE.Mesh(chestGeo, coatMat);
  chest.position.set(0, 0.195, 0);
  chest.scale.set(1.16, 1.0, 0.76);
  torsoGroup.add(chest);

  // Sculpted shoulder mantle
  const shoulders = new THREE.Mesh(new THREE.CapsuleGeometry(0.072, 0.25, 10, 18), coatMat);
  shoulders.position.set(0, 0.32, 0);
  shoulders.rotation.z = Math.PI / 2;
  shoulders.scale.set(1.0, 1.0, 0.96);
  torsoGroup.add(shoulders);

  // Lower flared waist skirt/coat hem merging smoothly onto plinth
  const waist = new THREE.Mesh(new THREE.CylinderGeometry(0.138, 0.165, 0.11, 24), secondaryMat);
  waist.position.set(0, 0.055, 0);
  waist.scale.set(1.12, 1.0, 0.82);
  torsoGroup.add(waist);

  // --- ICONIC SILHOUETTES PER CHARACTER ---
  if (id === 'machiavelli') {
    // High Renaissance standing crimson collar framing the jaw
    const collarGeo = new THREE.CylinderGeometry(0.115, 0.105, 0.115, 22, 1, true, -Math.PI * 0.78, Math.PI * 1.56);
    const collarMat = new THREE.MeshStandardMaterial({
      color: 0x9b1f24,
      roughness: 0.48,
      metalness: 0.14,
      side: THREE.DoubleSide,
    });
    const collar = new THREE.Mesh(collarGeo, collarMat);
    collar.position.set(0, 0.39, -0.01);
    collar.rotation.y = Math.PI;
    torsoGroup.add(collar);

    // Gold brocade doublet front trim + Florentine cape folds
    const trimBar = new THREE.Mesh(new THREE.BoxGeometry(0.036, 0.29, 0.025), brassMat);
    trimBar.position.set(0, 0.195, 0.112);
    torsoGroup.add(trimBar);

    for (const s of [-1, 1]) {
      const drape = new THREE.Mesh(new THREE.CapsuleGeometry(0.044, 0.24, 8, 12), secondaryMat);
      drape.position.set(s * 0.14, 0.21, 0.068);
      drape.rotation.z = -s * 0.14;
      torsoGroup.add(drape);
    }

    // Iconic Florentine crimson cap/hood (cappuccio)
    const hoodMat = new THREE.MeshStandardMaterial({
      color: 0xa62026,
      roughness: 0.5,
      metalness: 0.1,
    });
    const capCrown = new THREE.Mesh(new THREE.SphereGeometry(0.128, 22, 18, 0, Math.PI * 2, 0, Math.PI * 0.58), hoodMat);
    capCrown.position.set(0, 0.175, -0.008);
    capCrown.scale.set(1.06, 1.02, 1.12);
    headGroup.add(capCrown);

    // Sculpted rolled brim of the Florentine cap
    const capRoll = new THREE.Mesh(new THREE.TorusGeometry(0.118, 0.024, 12, 24), hoodMat);
    capRoll.position.set(0, 0.212, -0.004);
    capRoll.rotation.x = Math.PI / 2 - 0.18;
    headGroup.add(capRoll);

    // Sweeping Renaissance side hood lappets framing cheeks
    for (const s of [-1, 1]) {
      const lappet = new THREE.Mesh(new THREE.CapsuleGeometry(0.028, 0.135, 8, 12), hoodMat);
      lappet.position.set(s * 0.108, 0.115, -0.022);
      lappet.rotation.z = s * 0.12;
      lappet.rotation.x = 0.1;
      headGroup.add(lappet);
    }
  } else if (id === 'gandhi') {
    // Serene handspun warm ivory khadi shawl draped diagonally across chest & left shoulder
    const shawlMat = new THREE.MeshStandardMaterial({
      color: 0xf2eadc,
      roughness: 0.9,
      metalness: 0.01,
    });
    const shawlFoldMat = new THREE.MeshStandardMaterial({
      color: 0xe3d7c3,
      roughness: 0.92,
      metalness: 0.01,
    });

    // Expose bare warm sculptural right shoulder accent slightly under draped shawl
    const diagSash = new THREE.Mesh(new THREE.TorusGeometry(0.168, 0.048, 14, 28), shawlMat);
    diagSash.position.set(-0.01, 0.22, 0.015);
    diagSash.rotation.set(0.35, 0.25, 0.72);
    diagSash.scale.set(1.05, 1.18, 0.72);
    torsoGroup.add(diagSash);

    const shawlCollar = new THREE.Mesh(new THREE.TorusGeometry(0.105, 0.032, 12, 24), shawlFoldMat);
    shawlCollar.position.set(0.01, 0.34, 0.01);
    shawlCollar.rotation.x = Math.PI / 2 + 0.16;
    shawlCollar.rotation.y = 0.12;
    torsoGroup.add(shawlCollar);

    // Hanging shawl tail over left chest
    const tail = new THREE.Mesh(new THREE.CapsuleGeometry(0.046, 0.22, 8, 12), shawlMat);
    tail.position.set(0.115, 0.19, 0.088);
    tail.rotation.z = -0.08;
    torsoGroup.add(tail);

    // Iconic round wire-frame spectacles
    const wireMat = new THREE.MeshStandardMaterial({
      color: 0xc9a354,
      roughness: 0.22,
      metalness: 0.92,
    });
    const lensMat = new THREE.MeshPhysicalMaterial({
      color: 0xeef6ff,
      roughness: 0.08,
      metalness: 0.05,
      transparent: true,
      opacity: 0.28,
    });
    for (const s of [-1, 1]) {
      const rim = new THREE.Mesh(new THREE.TorusGeometry(0.031, 0.0036, 10, 26), wireMat);
      rim.position.set(s * 0.039, 0.161, 0.118);
      headGroup.add(rim);

      const lens = new THREE.Mesh(new THREE.CircleGeometry(0.029, 20), lensMat);
      lens.position.set(s * 0.039, 0.161, 0.118);
      headGroup.add(lens);

      // Temple arms back to ears
      const temple = new THREE.Mesh(new THREE.CylinderGeometry(0.0022, 0.0022, 0.122, 8), wireMat);
      temple.position.set(s * 0.084, 0.163, 0.062);
      temple.rotation.x = Math.PI / 2;
      temple.rotation.z = -s * 0.22;
      headGroup.add(temple);
    }
    // Delicate nose bridge arch
    const bridge = new THREE.Mesh(new THREE.TorusGeometry(0.011, 0.0026, 8, 14, Math.PI), wireMat);
    bridge.position.set(0, 0.162, 0.12);
    headGroup.add(bridge);

    // Gentle sculpted gray-white mustache on articulated jaw
    const stacheMat = new THREE.MeshStandardMaterial({ color: 0xc9c2b8, roughness: 0.8 });
    const stache = new THREE.Mesh(new THREE.CapsuleGeometry(0.0085, 0.038, 6, 10), stacheMat);
    stache.position.set(0, 0.011, 0.075);
    stache.rotation.z = Math.PI / 2;
    jawOrChinGroup.add(stache);
  } else if (id === 'richard_nixon') {
    // White dress shirt V-front & crisp shirt collar
    const shirtV = new THREE.Mesh(new THREE.ConeGeometry(0.076, 0.19, 4), shirtMat);
    shirtV.position.set(0, 0.265, 0.095);
    shirtV.rotation.z = Math.PI;
    shirtV.rotation.y = Math.PI / 4;
    shirtV.scale.set(1.0, 1.0, 0.35);
    torsoGroup.add(shirtV);

    // Peaked dark charcoal lapels
    for (const s of [-1, 1]) {
      const lapel = new THREE.Mesh(new THREE.BoxGeometry(0.044, 0.20, 0.022), secondaryMat);
      lapel.position.set(s * 0.064, 0.255, 0.112);
      lapel.rotation.z = s * 0.28;
      torsoGroup.add(lapel);
    }

    // Bold crimson presidential power tie + Windsor knot
    const tieMat = new THREE.MeshStandardMaterial({ color: 0xc92626, roughness: 0.42, metalness: 0.08 });
    const knot = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.014, 0.026, 10), tieMat);
    knot.position.set(0, 0.338, 0.116);
    torsoGroup.add(knot);

    const tieBlade = new THREE.Mesh(new THREE.ConeGeometry(0.026, 0.19, 4), tieMat);
    tieBlade.position.set(0, 0.228, 0.118);
    tieBlade.rotation.z = Math.PI;
    tieBlade.rotation.y = Math.PI / 4;
    tieBlade.scale.set(1.05, 1.0, 0.28);
    torsoGroup.add(tieBlade);

    // Tiny presidential gold lapel pin
    const flagPin = new THREE.Mesh(new THREE.BoxGeometry(0.022, 0.014, 0.008), brassMat);
    flagPin.position.set(0.084, 0.288, 0.125);
    torsoGroup.add(flagPin);

    // Swept-back dark presidential hair with distinguished silver temples
    const hairCrown = new THREE.Mesh(new THREE.SphereGeometry(0.118, 20, 16, 0, Math.PI * 2, 0, Math.PI * 0.52), hairMat);
    hairCrown.position.set(0, 0.175, -0.012);
    hairCrown.scale.set(1.04, 1.05, 1.08);
    headGroup.add(hairCrown);

    // Widow's peak hairline crest
    const peak = new THREE.Mesh(new THREE.ConeGeometry(0.036, 0.044, 4), hairMat);
    peak.position.set(0, 0.232, 0.088);
    peak.rotation.x = 0.65;
    peak.rotation.z = Math.PI;
    headGroup.add(peak);

    const silverMat = new THREE.MeshStandardMaterial({ color: 0x8e95a1, roughness: 0.65 });
    for (const s of [-1, 1]) {
      const sideHair = new THREE.Mesh(new THREE.BoxGeometry(0.026, 0.078, 0.085), silverMat);
      sideHair.position.set(s * 0.104, 0.188, -0.01);
      headGroup.add(sideHair);
    }
  } else if (id === 'oscar_wilde') {
    // Deep emerald velvet shawl lapels
    const velvetLapelMat = new THREE.MeshStandardMaterial({
      color: 0x0a2e21,
      roughness: 0.35,
      metalness: 0.16,
    });
    for (const s of [-1, 1]) {
      const lapel = new THREE.Mesh(new THREE.CapsuleGeometry(0.028, 0.20, 8, 12), velvetLapelMat);
      lapel.position.set(s * 0.072, 0.25, 0.112);
      lapel.rotation.z = s * 0.24;
      torsoGroup.add(lapel);
    }

    // Puffed ivory silk Victorian cravat + pearl stickpin
    const cravatMat = new THREE.MeshStandardMaterial({ color: 0xf7f2e6, roughness: 0.42, metalness: 0.08 });
    const cravatTop = new THREE.Mesh(new THREE.SphereGeometry(0.044, 16, 14), cravatMat);
    cravatTop.position.set(0, 0.332, 0.114);
    cravatTop.scale.set(1.15, 0.9, 0.68);
    torsoGroup.add(cravatTop);

    const cravatFold = new THREE.Mesh(new THREE.ConeGeometry(0.056, 0.14, 12), cravatMat);
    cravatFold.position.set(0, 0.258, 0.112);
    cravatFold.rotation.z = Math.PI;
    cravatFold.scale.set(1.0, 1.0, 0.45);
    torsoGroup.add(cravatFold);

    const pearl = new THREE.Mesh(
      new THREE.SphereGeometry(0.011, 12, 10),
      new THREE.MeshStandardMaterial({ color: 0xfffdf8, roughness: 0.15, metalness: 0.3 })
    );
    pearl.position.set(0, 0.32, 0.145);
    torsoGroup.add(pearl);

    // Unmistakable bright green carnation lapel pin (multi-petal rosette!)
    const carnationGroup = new THREE.Group();
    carnationGroup.position.set(0.082, 0.292, 0.134);
    const carnMat = new THREE.MeshStandardMaterial({
      color: 0x3be883,
      emissive: 0x165e33,
      emissiveIntensity: 0.45,
      roughness: 0.45,
    });
    for (let i = 0; i < 6; i++) {
      const a = (i / 6) * Math.PI * 2;
      const petal = new THREE.Mesh(new THREE.SphereGeometry(0.012, 10, 8), carnMat);
      petal.position.set(Math.cos(a) * 0.012, Math.sin(a) * 0.012, 0.004);
      carnationGroup.add(petal);
    }
    const centerBud = new THREE.Mesh(new THREE.SphereGeometry(0.010, 10, 8), carnMat);
    centerBud.position.set(0, 0, 0.01);
    carnationGroup.add(centerBud);
    torsoGroup.add(carnationGroup);

    // Sweeping Victorian parted wavy hair crest
    const hairMain = new THREE.Mesh(new THREE.SphereGeometry(0.125, 22, 18), hairMat);
    hairMain.position.set(0, 0.185, -0.022);
    hairMain.scale.set(1.12, 1.06, 1.06);
    headGroup.add(hairMain);

    for (const s of [-1, 1]) {
      const wave = new THREE.Mesh(new THREE.CapsuleGeometry(0.036, 0.135, 8, 12), hairMat);
      wave.position.set(s * 0.114, 0.165, -0.002);
      wave.rotation.z = s * 0.22;
      wave.rotation.x = 0.12;
      headGroup.add(wave);

      // Sweeping bangs framing center part
      const crest = new THREE.Mesh(new THREE.SphereGeometry(0.054, 14, 12), hairMat);
      crest.position.set(s * 0.052, 0.258, 0.064);
      crest.scale.set(1.2, 0.65, 0.9);
      headGroup.add(crest);
    }
  } else if (id === 'warren_buffett') {
    // Crisp light-blue Oxford shirt V-neck & neat burgundy tie
    const oxfordMat = new THREE.MeshStandardMaterial({ color: 0xe3ecf7, roughness: 0.78 });
    const shirtV = new THREE.Mesh(new THREE.ConeGeometry(0.072, 0.18, 4), oxfordMat);
    shirtV.position.set(0, 0.265, 0.095);
    shirtV.rotation.z = Math.PI;
    shirtV.rotation.y = Math.PI / 4;
    shirtV.scale.set(1.0, 1.0, 0.35);
    torsoGroup.add(shirtV);

    for (const s of [-1, 1]) {
      const lapel = new THREE.Mesh(new THREE.BoxGeometry(0.042, 0.195, 0.022), secondaryMat);
      lapel.position.set(s * 0.062, 0.252, 0.112);
      lapel.rotation.z = s * 0.26;
      torsoGroup.add(lapel);
    }

    const tieMat = new THREE.MeshStandardMaterial({ color: 0x8f2d34, roughness: 0.45, metalness: 0.1 });
    const knot = new THREE.Mesh(new THREE.CylinderGeometry(0.017, 0.013, 0.024, 10), tieMat);
    knot.position.set(0, 0.336, 0.116);
    torsoGroup.add(knot);

    const tieBlade = new THREE.Mesh(new THREE.ConeGeometry(0.024, 0.18, 4), tieMat);
    tieBlade.position.set(0, 0.23, 0.118);
    tieBlade.rotation.z = Math.PI;
    tieBlade.rotation.y = Math.PI / 4;
    tieBlade.scale.set(1.0, 1.0, 0.28);
    torsoGroup.add(tieBlade);

    // Classic tortoiseshell glasses (warm amber-brown rounded-rect frames)
    const tortoiseMat = new THREE.MeshStandardMaterial({
      color: 0x7a421d,
      roughness: 0.28,
      metalness: 0.18,
    });
    const lensMat = new THREE.MeshPhysicalMaterial({
      color: 0xf0f7ff,
      roughness: 0.06,
      transparent: true,
      opacity: 0.26,
    });
    for (const s of [-1, 1]) {
      const frame = new THREE.Group();
      frame.position.set(s * 0.042, 0.162, 0.118);
      // Rounded-square frame built from 4 sleek bars
      const topBar = new THREE.Mesh(new THREE.BoxGeometry(0.062, 0.008, 0.008), tortoiseMat);
      topBar.position.y = 0.024;
      const botBar = new THREE.Mesh(new THREE.BoxGeometry(0.058, 0.007, 0.008), tortoiseMat);
      botBar.position.y = -0.022;
      const outerBar = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.048, 0.008), tortoiseMat);
      outerBar.position.x = s * 0.028;
      const innerBar = new THREE.Mesh(new THREE.BoxGeometry(0.007, 0.046, 0.008), tortoiseMat);
      innerBar.position.x = -s * 0.027;
      const glass = new THREE.Mesh(new THREE.PlaneGeometry(0.052, 0.042), lensMat);
      frame.add(topBar, botBar, outerBar, innerBar, glass);
      headGroup.add(frame);

      const temple = new THREE.Mesh(new THREE.BoxGeometry(0.007, 0.007, 0.118), tortoiseMat);
      temple.position.set(s * 0.088, 0.172, 0.06);
      temple.rotation.y = -s * 0.2;
      headGroup.add(temple);
    }
    const gBridge = new THREE.Mesh(new THREE.BoxGeometry(0.024, 0.007, 0.008), tortoiseMat);
    gBridge.position.set(0, 0.172, 0.119);
    headGroup.add(gBridge);

    // Warm silver swept hair over temples & side-parted crown
    const silverHairMat = new THREE.MeshStandardMaterial({ color: 0xdce0e6, roughness: 0.62 });
    const hairTop = new THREE.Mesh(new THREE.SphereGeometry(0.118, 20, 16, 0, Math.PI * 2, 0, Math.PI * 0.46), silverHairMat);
    hairTop.position.set(-0.006, 0.184, -0.008);
    hairTop.scale.set(1.06, 0.96, 1.06);
    headGroup.add(hairTop);

    const sideSweep = new THREE.Mesh(new THREE.CapsuleGeometry(0.028, 0.115, 8, 12), silverHairMat);
    sideSweep.position.set(0.015, 0.258, 0.048);
    sideSweep.rotation.z = Math.PI / 2 - 0.14;
    headGroup.add(sideSweep);

    for (const s of [-1, 1]) {
      const puff = new THREE.Mesh(new THREE.SphereGeometry(0.042, 12, 10), silverHairMat);
      puff.position.set(s * 0.104, 0.182, -0.012);
      puff.scale.set(0.55, 1.15, 1.15);
      headGroup.add(puff);
    }
  } else if (id === 'van_gogh') {
    // Open-collar cobalt painter's smock + warm cream undershirt + ochre neckerchief
    const underShirt = new THREE.Mesh(new THREE.CylinderGeometry(0.068, 0.062, 0.09, 16), shirtMat);
    underShirt.position.set(0, 0.34, 0.02);
    torsoGroup.add(underShirt);

    const scarfMat = new THREE.MeshStandardMaterial({ color: 0xdca02a, roughness: 0.78 });
    const scarf = new THREE.Mesh(new THREE.TorusGeometry(0.088, 0.022, 10, 22), scarfMat);
    scarf.position.set(0, 0.345, 0.015);
    scarf.rotation.x = Math.PI / 2 + 0.18;
    torsoGroup.add(scarf);

    // Subtle impasto paint dabs on cobalt smock
    const dabColors = [0xf2c029, 0xe85a2a, 0x54a8e8];
    dabColors.forEach((hex, idx) => {
      const dab = new THREE.Mesh(
        new THREE.SphereGeometry(0.013, 8, 6),
        new THREE.MeshStandardMaterial({ color: hex, roughness: 0.45 })
      );
      dab.position.set(-0.07 + idx * 0.025, 0.21 - idx * 0.024, 0.114);
      dab.scale.set(1.4, 0.7, 0.25);
      torsoGroup.add(dab);
    });

    // Sunlit ochre straw hat (wide brim + domed crown + cobalt ribbon!)
    const strawTex = createStrawTexture();
    if (strawTex) resources.textures.push(strawTex);
    const strawMat = new THREE.MeshStandardMaterial({
      color: 0xebbb44,
      map: strawTex || null,
      roughness: 0.82,
      metalness: 0.02,
    });
    const hatGroup = new THREE.Group();
    hatGroup.position.set(0, 0.242, -0.012);
    hatGroup.rotation.set(-0.14, 0, -0.08); // slightly rakish Provençal tilt

    const brim = new THREE.Mesh(new THREE.CylinderGeometry(0.245, 0.255, 0.014, 32), strawMat);
    hatGroup.add(brim);

    const crown = new THREE.Mesh(new THREE.CylinderGeometry(0.118, 0.132, 0.095, 24), strawMat);
    crown.position.y = 0.048;
    hatGroup.add(crown);

    const crownTop = new THREE.Mesh(new THREE.SphereGeometry(0.118, 20, 12, 0, Math.PI * 2, 0, Math.PI * 0.4), strawMat);
    crownTop.position.y = 0.072;
    crownTop.scale.set(1, 0.35, 1);
    hatGroup.add(crownTop);

    const hatBand = new THREE.Mesh(
      new THREE.CylinderGeometry(0.133, 0.135, 0.024, 24),
      new THREE.MeshStandardMaterial({ color: 0x1d407c, roughness: 0.65 })
    );
    hatBand.position.y = 0.018;
    hatGroup.add(hatBand);
    headGroup.add(hatGroup);

    // Tousled copper-ginger hair peeking under hat
    for (const s of [-1, 1]) {
      const tuft = new THREE.Mesh(new THREE.SphereGeometry(0.044, 12, 10), hairMat);
      tuft.position.set(s * 0.104, 0.182, -0.01);
      tuft.scale.set(0.62, 1.1, 1.1);
      headGroup.add(tuft);
    }

    // Unmistakable sculpted ginger beard & mustache mounted on jawOrChinGroup
    const beard = new THREE.Mesh(new THREE.ConeGeometry(0.068, 0.105, 14), hairMat);
    beard.position.set(0, -0.048, 0.042);
    beard.rotation.x = Math.PI + 0.14;
    beard.scale.set(1.06, 1.0, 0.78);
    jawOrChinGroup.add(beard);

    const beardCheeks = new THREE.Mesh(new THREE.SphereGeometry(0.062, 14, 12), hairMat);
    beardCheeks.position.set(0, -0.016, 0.048);
    beardCheeks.scale.set(1.18, 0.72, 0.88);
    jawOrChinGroup.add(beardCheeks);

    // Tiny wooden painter's palette prop held neatly at his left side/hand
    const paletteGroup = new THREE.Group();
    paletteGroup.position.set(0.045, 0.012, 0.032);
    paletteGroup.rotation.set(0.28, -0.45, 0.15);

    const woodMat = new THREE.MeshStandardMaterial({ color: 0x9b683b, roughness: 0.65 });
    const paletteBoard = new THREE.Mesh(new THREE.CylinderGeometry(0.088, 0.088, 0.009, 20), woodMat);
    paletteBoard.scale.set(1.22, 1.0, 0.92);
    paletteGroup.add(paletteBoard);

    // Paint dabs around palette rim
    const palettePigments = [0xf5c828, 0x265ab8, 0xd93c23, 0x2ca35b, 0xf7f5ee];
    palettePigments.forEach((hex, idx) => {
      const angle = -0.9 + idx * 0.56;
      const blob = new THREE.Mesh(
        new THREE.SphereGeometry(0.014, 10, 8),
        new THREE.MeshStandardMaterial({ color: hex, roughness: 0.35 })
      );
      blob.position.set(Math.cos(angle) * 0.068, 0.008, Math.sin(angle) * 0.052);
      blob.scale.set(1.1, 0.45, 1.1);
      paletteGroup.add(blob);
    });

    // Fine wooden paintbrush resting across right hand
    const brushGroup = new THREE.Group();
    const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.0045, 0.003, 0.17, 8), woodMat);
    handle.rotation.x = Math.PI / 2 - 0.35;
    handle.position.set(0, 0.015, 0.02);
    const tip = new THREE.Mesh(
      new THREE.ConeGeometry(0.006, 0.024, 8),
      new THREE.MeshStandardMaterial({ color: 0xf4c42a, roughness: 0.45 })
    );
    tip.position.set(0, 0.046, 0.102);
    tip.rotation.x = Math.PI / 2 - 0.35;
    brushGroup.add(handle, tip);

    leftArmObj.hand.add(paletteGroup);
    rightArmObj.hand.add(brushGroup);
  }
}

/**
 * Creates a stylized sculptural 3D figurine/bust seated at the negotiation table.
 *
 * Local coordinate space:
 * - Origin (0, 0, 0) rests cleanly on the chair seat cushion surface (table seat Y = 0.46m).
 * - Total figurine height is ~0.76m (up to ~0.84m with hat/hair crest).
 * - Internal `facingGroup` rotates +PI/2 for Seat A (facing +X) and -PI/2 for Seat B (facing -X).
 *
 * @param {string} characterId One of the 6 cast IDs in CHARACTER_META
 * @param {'A'|'B'} seat Table seat identifier
 */
export function createCharacterRig(characterId, seat = 'A') {
  const meta = CHARACTER_META[characterId] || CHARACTER_META.machiavelli;
  const resolvedSeat = seat === 'B' ? 'B' : 'A';
  const resources = { textures: [] };

  const group = new THREE.Group();
  group.name = `characterRig_${meta.id}_${resolvedSeat}`;
  group.userData = { characterId: meta.id, seat: resolvedSeat };

  // Internal facing group so all animations work in intuitive +Z-forward local space
  const facingGroup = new THREE.Group();
  facingGroup.name = 'facingGroup';
  facingGroup.rotation.y = resolvedSeat === 'A' ? Math.PI / 2 : -Math.PI / 2;
  group.add(facingGroup);

  // Shared premium materials
  const plinthMat = new THREE.MeshStandardMaterial({
    color: 0x1d1a19,
    roughness: 0.42,
    metalness: 0.22,
  });
  const brassMat = new THREE.MeshStandardMaterial({
    color: 0xc9a24b,
    roughness: 0.26,
    metalness: 0.86,
  });
  const accentRingMat = new THREE.MeshStandardMaterial({
    color: meta.accentColor,
    emissive: meta.accentColor,
    emissiveIntensity: 0.35,
    roughness: 0.35,
    metalness: 0.3,
  });
  const skinMat = new THREE.MeshStandardMaterial({
    color: meta.palette.skin,
    roughness: meta.id === 'machiavelli' ? 0.42 : 0.56,
    metalness: meta.id === 'machiavelli' ? 0.14 : 0.04,
  });
  const eyeMat = new THREE.MeshStandardMaterial({
    color: 0x1a1716,
    roughness: 0.18,
    metalness: 0.1,
  });
  const eyebrowMat = new THREE.MeshStandardMaterial({
    color: meta.palette.hair,
    roughness: 0.72,
  });

  // --- 1. ARCHITECTURAL SEAT PLINTH + PORTRAIT MEDALLION ---
  const plinthGroup = new THREE.Group();
  plinthGroup.name = 'plinthGroup';
  facingGroup.add(plinthGroup);

  const plinthBase = new THREE.Mesh(new THREE.CylinderGeometry(0.185, 0.205, 0.052, 32), plinthMat);
  plinthBase.position.y = 0.026;
  plinthGroup.add(plinthBase);

  const plinthStep = new THREE.Mesh(new THREE.CylinderGeometry(0.165, 0.175, 0.028, 32), plinthMat);
  plinthStep.position.y = 0.066;
  plinthGroup.add(plinthStep);

  const plinthBrassTrim = new THREE.Mesh(new THREE.TorusGeometry(0.188, 0.006, 10, 36), brassMat);
  plinthBrassTrim.position.y = 0.052;
  plinthBrassTrim.rotation.x = Math.PI / 2;
  plinthGroup.add(plinthBrassTrim);

  const plinthGlowRing = new THREE.Mesh(new THREE.TorusGeometry(0.17, 0.0055, 10, 36), accentRingMat);
  plinthGlowRing.position.y = 0.079;
  plinthGlowRing.rotation.x = Math.PI / 2;
  plinthGroup.add(plinthGlowRing);

  // Procedural high-DPI portrait medallions mounted symmetrically on both table-view sides of the plinth
  const medTex = createMedallionTexture(meta);
  if (medTex) resources.textures.push(medTex);

  const medFaceMat = new THREE.MeshStandardMaterial({
    map: medTex || null,
    color: 0xffffff,
    roughness: 0.34,
    metalness: 0.12,
  });

  for (const sideSign of [-1, 1]) {
    const medallionGroup = new THREE.Group();
    medallionGroup.position.set(sideSign * 0.122, 0.112, 0.136);
    medallionGroup.rotation.y = sideSign * 0.72;
    medallionGroup.rotation.x = -0.2;

    const medBezel = new THREE.Mesh(new THREE.CylinderGeometry(0.054, 0.057, 0.014, 28), brassMat);
    medBezel.rotation.x = Math.PI / 2;
    medallionGroup.add(medBezel);

    const medFace = new THREE.Mesh(new THREE.CircleGeometry(0.049, 28), medFaceMat);
    medFace.position.z = 0.008;
    medallionGroup.add(medFace);
    plinthGroup.add(medallionGroup);
  }

  // --- 2. EXPRESSIVE TORSO, HEAD, JAW & ARMS HIERARCHY ---
  const torsoGroup = new THREE.Group();
  torsoGroup.name = 'torsoGroup';
  torsoGroup.position.set(0, 0.08, 0);
  facingGroup.add(torsoGroup);

  const headOpts =
    meta.id === 'richard_nixon'
      ? { heavyBrow: true, angularJaw: true, noseLength: 0.054, noseProjection: 0.036, browTilt: -0.1 }
      : meta.id === 'machiavelli'
        ? { noseLength: 0.052, noseProjection: 0.034, browTilt: -0.06, headScaleX: 0.94 }
        : meta.id === 'gandhi'
          ? { headScaleX: 0.96, headScaleY: 1.14, browTilt: 0.08 }
          : meta.id === 'oscar_wilde'
            ? { headScaleX: 1.04, headScaleY: 1.12, browTilt: 0.1 }
            : {};

  const { headGroup, jawOrChinGroup } = buildSculptedHead(skinMat, eyeMat, eyebrowMat, headOpts);
  headGroup.position.set(0, 0.39, 0.005);
  torsoGroup.add(headGroup);

  const armCoatMat = new THREE.MeshStandardMaterial({
    color: meta.palette.primary,
    roughness: 0.68,
    metalness: 0.06,
  });
  const cuffMat = new THREE.MeshStandardMaterial({
    color: meta.id === 'gandhi' ? 0xf2eadc : 0xf4f0e8,
    roughness: 0.78,
  });

  const leftArmObj = buildArm('left', armCoatMat, cuffMat, skinMat);
  const leftArm = leftArmObj.armGroup;
  leftArm.position.set(0.185, 0.31, 0.01);
  torsoGroup.add(leftArm);

  const rightArmObj = buildArm('right', armCoatMat, cuffMat, skinMat);
  const rightArm = rightArmObj.armGroup;
  rightArm.position.set(-0.185, 0.31, 0.01);
  torsoGroup.add(rightArm);

  dressCharacter(
    meta.id,
    meta,
    { torsoGroup, headGroup, jawOrChinGroup, leftArmObj, rightArmObj, skinMat, brassMat },
    resources
  );

  // Enable clean shadows on sculpted bust & plinth
  enableShadows(plinthGroup, true, true);
  enableShadows(torsoGroup, true, true);

  // --- 3. THOUGHT-AURA HALO RING (hidden unless thinking) ---
  const thoughtHalo = new THREE.Group();
  thoughtHalo.name = 'thoughtHalo';
  thoughtHalo.position.set(0, 0.86, 0);
  thoughtHalo.visible = false;
  facingGroup.add(thoughtHalo);

  const haloOuterMat = new THREE.MeshBasicMaterial({
    color: 0xf4c05b,
    transparent: true,
    opacity: 0.0,
    depthWrite: false,
  });
  const haloInnerMat = new THREE.MeshBasicMaterial({
    color: 0xffe4a8,
    transparent: true,
    opacity: 0.0,
    depthWrite: false,
  });
  const outerRing = new THREE.Mesh(new THREE.TorusGeometry(0.19, 0.0065, 10, 48), haloOuterMat);
  outerRing.rotation.x = Math.PI / 2;
  thoughtHalo.add(outerRing);

  const innerRing = new THREE.Mesh(new THREE.TorusGeometry(0.138, 0.004, 8, 40), haloInnerMat);
  innerRing.rotation.x = Math.PI / 2;
  thoughtHalo.add(innerRing);

  // Subtle orbiting contemplation motes
  const motes = [];
  for (let i = 0; i < 6; i++) {
    const mote = new THREE.Mesh(new THREE.SphereGeometry(0.0085, 8, 8), haloInnerMat);
    thoughtHalo.add(mote);
    motes.push(mote);
  }

  // --- 4. EXPRESSIVE RIM LIGHT ---
  const rimLight = new THREE.PointLight(new THREE.Color(meta.palette.glow), 0.35, 2.2, 2.0);
  rimLight.name = 'rimLight';
  rimLight.position.set(0, 0.78, -0.26);
  facingGroup.add(rimLight);

  // --- 5. FLOATING EDITORIAL CANVAS NAMEPLATE SPRITE ---
  const plateTex = createNameplateTexture(meta, resolvedSeat);
  if (plateTex) resources.textures.push(plateTex);
  const plateMat = new THREE.SpriteMaterial({
    map: plateTex || null,
    transparent: true,
    depthTest: true,
    depthWrite: false,
  });
  const nameplate = new THREE.Sprite(plateMat);
  nameplate.name = 'nameplate';
  nameplate.scale.set(0.64, 0.17, 1);
  nameplate.position.set(0, 1.04, 0);
  facingGroup.add(nameplate);

  // --- 6. PROCEDURAL ANIMATION STATE MACHINE ---
  let currentState = 'idle';
  let stateTime = 0;
  let stateOpts = {};
  let haloAlpha = 0;

  const baseRimColor = new THREE.Color(meta.palette.glow);
  const thinkRimColor = new THREE.Color('#F2B857');
  const splitRimColor = new THREE.Color('#2F6FDB');
  const stealRimColor = new THREE.Color('#EF4E14');
  const winRimColor = new THREE.Color('#E8B949');
  const betrayedRimColor = new THREE.Color('#5C73A8');
  const crashRimColor = new THREE.Color('#D94E28');
  const targetRimColor = baseRimColor.clone();

  // Smoothed pose state (for zero-snap transitions)
  const pose = {
    torsoX: 0,
    torsoY: 0,
    torsoZ: 0,
    torsoPosY: 0.08,
    torsoPosZ: 0,
    headX: 0,
    headY: 0,
    headZ: 0,
    jawX: 0,
    lArmX: 0,
    lArmY: 0,
    lArmZ: 0,
    rArmX: 0,
    rArmY: 0,
    rArmZ: 0,
    rimIntensity: 0.35,
    plinthGlow: 0.35,
  };

  function setState(state, opts = {}) {
    const next = VALID_STATES.has(state) ? state : 'idle';
    if (next !== currentState || opts.force) {
      currentState = next;
      stateTime = 0;
    }
    stateOpts = opts || {};
    if (opts.instant) {
      update(0.5, 0);
    }
  }

  function damp(cur, target, lambda, dt) {
    return cur + (target - cur) * (1 - Math.exp(-lambda * dt));
  }

  const phaseOffset = resolvedSeat === 'A' ? 0.0 : 1.73;

  function update(dt = 0.016, elapsed = 0) {
    const clampedDt = Math.min(Math.max(dt, 0.0001), 0.1);
    stateTime += clampedDt;
    const t = elapsed + phaseOffset;
    const intensity = stateOpts.intensity ?? 1.0;

    // Default neutral resting targets
    let tgtTorsoX = 0.02 + Math.sin(t * 1.6) * 0.014;
    let tgtTorsoY = Math.sin(t * 0.65) * 0.025;
    let tgtTorsoZ = 0;
    let tgtTorsoPosY = 0.08 + Math.sin(t * 1.6) * 0.005;
    let tgtTorsoPosZ = 0;

    let tgtHeadX = Math.sin(t * 1.15) * 0.028;
    let tgtHeadY = Math.sin(t * 0.72) * 0.055;
    let tgtHeadZ = Math.cos(t * 0.85) * 0.024;
    let tgtJawX = 0;

    let tgtLArmX = -0.05 + Math.sin(t * 1.5) * 0.02;
    let tgtLArmY = 0.04;
    let tgtLArmZ = 0.06;
    let tgtRArmX = -0.05 + Math.cos(t * 1.4) * 0.02;
    let tgtRArmY = -0.04;
    let tgtRArmZ = -0.06;

    let tgtRimIntensity = 0.35;
    let tgtPlinthGlow = 0.32;
    let tgtHaloAlpha = 0.0;
    targetRimColor.copy(baseRimColor);

    if (currentState === 'speaking') {
      // Attentive forward lean toward opponent + expressive rhythmic nods & hand gestures
      const cadence = Math.sin(t * 5.8) * 0.5 + Math.sin(t * 9.3) * 0.32 + Math.cos(t * 3.1) * 0.18;
      tgtTorsoX = 0.135 + Math.sin(t * 2.4) * 0.022;
      tgtTorsoY = Math.sin(t * 1.35) * 0.05;
      tgtTorsoPosZ = 0.036;
      tgtTorsoPosY = 0.083 + Math.sin(t * 3.2) * 0.006;

      tgtHeadX = -0.04 + cadence * 0.075 * intensity;
      tgtHeadY = Math.sin(t * 1.85) * 0.095;
      tgtHeadZ = Math.cos(t * 2.1) * 0.055;
      tgtJawX = Math.max(0, Math.sin(t * 12.5) * 0.55 + Math.cos(t * 7.8) * 0.45) * 0.14 * intensity;

      // Emphatic open gesturing right hand & steady supporting left hand
      tgtRArmX = -0.34 - Math.sin(t * 4.2) * 0.14 * intensity;
      tgtRArmY = 0.14 + Math.cos(t * 2.7) * 0.1;
      tgtRArmZ = -0.12 - Math.sin(t * 3.4) * 0.08;

      tgtLArmX = -0.18 - Math.cos(t * 3.1) * 0.08 * intensity;
      tgtLArmY = -0.08;
      tgtLArmZ = 0.12;

      tgtRimIntensity = 1.65 + Math.sin(t * 4.5) * 0.2;
      tgtPlinthGlow = 0.95;
      targetRimColor.copy(baseRimColor);
    } else if (currentState === 'thinking') {
      // Lean back slightly, contemplative head tilt, pulsing golden thought halo overhead
      tgtTorsoX = -0.095 + Math.sin(t * 1.1) * 0.012;
      tgtTorsoY = -0.04 + Math.sin(t * 0.5) * 0.03;
      tgtTorsoPosZ = -0.026;
      tgtTorsoPosY = 0.078 + Math.sin(t * 1.1) * 0.004;

      tgtHeadX = -0.085 + Math.sin(t * 0.9) * 0.025;
      tgtHeadY = 0.09 + Math.sin(t * 0.55) * 0.05;
      tgtHeadZ = 0.145 + Math.cos(t * 0.8) * 0.025;

      // Right hand raised thoughtfully toward chin
      tgtRArmX = -0.78;
      tgtRArmY = 0.44;
      tgtRArmZ = 0.16;

      tgtLArmX = -0.04;
      tgtLArmY = 0.08;
      tgtLArmZ = 0.08;

      tgtRimIntensity = 1.45 + Math.sin(t * 2.6) * 0.35;
      tgtPlinthGlow = 0.8;
      tgtHaloAlpha = 0.88;
      targetRimColor.copy(thinkRimColor);
    } else if (currentState === 'reveal_split') {
      // Open upright cooperative posture, open palms gesture, affirming nod
      const nodPulse = Math.exp(-stateTime * 1.8) * Math.sin(stateTime * 7.0) * 0.14;
      tgtTorsoX = 0.035;
      tgtTorsoPosZ = 0.015;
      tgtTorsoPosY = 0.088;

      tgtHeadX = 0.04 + nodPulse + Math.sin(t * 2.2) * 0.035;
      tgtHeadY = 0;
      tgtHeadZ = 0.03;

      // Wide welcoming open arms
      tgtLArmX = -0.28;
      tgtLArmY = 0.32;
      tgtLArmZ = 0.32;
      tgtRArmX = -0.28;
      tgtRArmY = -0.32;
      tgtRArmZ = -0.32;

      tgtRimIntensity = 1.85;
      tgtPlinthGlow = 1.15;
      targetRimColor.copy(splitRimColor);
    } else if (currentState === 'reveal_steal') {
      // Sharp forward predator lean & cocked head angle
      tgtTorsoX = 0.21;
      tgtTorsoY = -0.06;
      tgtTorsoPosZ = 0.056;
      tgtTorsoPosY = 0.075;

      tgtHeadX = -0.12;
      tgtHeadY = 0.11;
      tgtHeadZ = -0.16;

      // Right arm drives decisively forward onto table
      tgtRArmX = -0.44;
      tgtRArmY = 0.18;
      tgtRArmZ = -0.04;
      tgtLArmX = 0.08;
      tgtLArmY = -0.08;
      tgtLArmZ = 0.14;

      tgtRimIntensity = 2.05;
      tgtPlinthGlow = 1.3;
      targetRimColor.copy(stealRimColor);
    } else if (currentState === 'react_win') {
      // Satisfied slow nod & upright proud posture
      tgtTorsoX = -0.055;
      tgtTorsoPosZ = -0.01;
      tgtTorsoPosY = 0.092 + Math.sin(t * 2.0) * 0.005;

      tgtHeadX = 0.03 + Math.sin(stateTime * 2.6) * 0.085;
      tgtHeadY = Math.sin(t * 0.9) * 0.04;
      tgtHeadZ = 0.04;

      tgtLArmX = -0.22;
      tgtLArmY = 0.14;
      tgtLArmZ = 0.18;
      tgtRArmX = -0.22;
      tgtRArmY = -0.14;
      tgtRArmZ = -0.18;

      tgtRimIntensity = 1.9;
      tgtPlinthGlow = 1.1;
      targetRimColor.copy(winRimColor);
    } else if (currentState === 'react_betrayed') {
      // Stunned recoil back + slow disbelieving head shake
      const recoil = Math.min(1, stateTime * 4.0);
      tgtTorsoX = -0.185 * recoil;
      tgtTorsoPosZ = -0.052 * recoil;
      tgtTorsoPosY = 0.072;

      tgtHeadX = 0.07;
      tgtHeadY = Math.sin(stateTime * 3.8) * 0.24 * Math.exp(-stateTime * 0.25);
      tgtHeadZ = -0.09;

      // Defensive pulled-back arms
      tgtLArmX = -0.35;
      tgtLArmY = 0.26;
      tgtLArmZ = 0.22;
      tgtRArmX = -0.35;
      tgtRArmY = -0.26;
      tgtRArmZ = -0.22;

      tgtRimIntensity = 0.85;
      tgtPlinthGlow = 0.25;
      targetRimColor.copy(betrayedRimColor);
    } else if (currentState === 'react_crash') {
      // Tense mutual standoff freeze & slight jitter shake
      const tremor = Math.exp(-stateTime * 1.6) * Math.sin(stateTime * 34.0) * 0.025;
      tgtTorsoX = 0.075;
      tgtTorsoZ = tremor;
      tgtTorsoPosZ = 0.022;
      tgtTorsoPosY = 0.078;

      tgtHeadX = -0.04;
      tgtHeadY = tremor * 1.8;
      tgtHeadZ = -0.05 + tremor;

      tgtLArmX = -0.14;
      tgtLArmY = 0.04;
      tgtLArmZ = 0.05;
      tgtRArmX = -0.14;
      tgtRArmY = -0.04;
      tgtRArmZ = -0.05;

      tgtRimIntensity = 1.45;
      tgtPlinthGlow = 0.75;
      targetRimColor.copy(crashRimColor);
    }

    const speed = currentState === 'react_crash' || currentState === 'reveal_steal' ? 11.0 : 6.8;
    pose.torsoX = damp(pose.torsoX, tgtTorsoX, speed, clampedDt);
    pose.torsoY = damp(pose.torsoY, tgtTorsoY, speed, clampedDt);
    pose.torsoZ = damp(pose.torsoZ, tgtTorsoZ, speed, clampedDt);
    pose.torsoPosY = damp(pose.torsoPosY, tgtTorsoPosY, speed, clampedDt);
    pose.torsoPosZ = damp(pose.torsoPosZ, tgtTorsoPosZ, speed, clampedDt);

    pose.headX = damp(pose.headX, tgtHeadX, speed + 1.5, clampedDt);
    pose.headY = damp(pose.headY, tgtHeadY, speed + 1.5, clampedDt);
    pose.headZ = damp(pose.headZ, tgtHeadZ, speed + 1.5, clampedDt);
    pose.jawX = damp(pose.jawX, tgtJawX, 18.0, clampedDt);

    pose.lArmX = damp(pose.lArmX, tgtLArmX, speed, clampedDt);
    pose.lArmY = damp(pose.lArmY, tgtLArmY, speed, clampedDt);
    pose.lArmZ = damp(pose.lArmZ, tgtLArmZ, speed, clampedDt);
    pose.rArmX = damp(pose.rArmX, tgtRArmX, speed, clampedDt);
    pose.rArmY = damp(pose.rArmY, tgtRArmY, speed, clampedDt);
    pose.rArmZ = damp(pose.rArmZ, tgtRArmZ, speed, clampedDt);

    pose.rimIntensity = damp(pose.rimIntensity, tgtRimIntensity, 5.5, clampedDt);
    pose.plinthGlow = damp(pose.plinthGlow, tgtPlinthGlow, 5.5, clampedDt);

    // Apply transforms
    torsoGroup.rotation.set(pose.torsoX, pose.torsoY, pose.torsoZ);
    torsoGroup.position.set(0, pose.torsoPosY, pose.torsoPosZ);

    headGroup.rotation.set(pose.headX, pose.headY, pose.headZ);
    jawOrChinGroup.rotation.x = pose.jawX;

    leftArm.rotation.set(pose.lArmX, pose.lArmY, pose.lArmZ);
    rightArm.rotation.set(pose.rArmX, pose.rArmY, pose.rArmZ);

    rimLight.intensity = pose.rimIntensity;
    rimLight.color.lerp(targetRimColor, 1 - Math.exp(-5.0 * clampedDt));
    accentRingMat.emissiveIntensity = pose.plinthGlow;

    // Thought halo smooth fade, pulse, and orbiting motes
    haloAlpha = damp(haloAlpha, tgtHaloAlpha, 6.0, clampedDt);
    if (haloAlpha > 0.01) {
      thoughtHalo.visible = true;
      const pulse = 1.0 + Math.sin(t * 3.0) * 0.045;
      thoughtHalo.scale.set(pulse, 1.0, pulse);
      thoughtHalo.position.y = 0.87 + Math.sin(t * 2.1) * 0.015;
      outerRing.rotation.z = t * 0.45;
      innerRing.rotation.z = -t * 0.7;
      haloOuterMat.opacity = haloAlpha * 0.82;
      haloInnerMat.opacity = haloAlpha * 0.65;
      motes.forEach((m, idx) => {
        const a = t * 1.15 + (idx / motes.length) * Math.PI * 2;
        m.position.set(Math.cos(a) * 0.165, Math.sin(t * 2.4 + idx) * 0.018, Math.sin(a) * 0.165);
      });
    } else {
      thoughtHalo.visible = false;
    }

    // Gentle hover on nameplate
    nameplate.position.y = 1.04 + Math.sin(t * 1.4) * 0.008;
  }

  function getWorldHeadPosition(target = new THREE.Vector3()) {
    group.updateWorldMatrix(true, false);
    headGroup.getWorldPosition(target);
    target.y += 0.15;
    return target;
  }

  function setNameplateVisible(visible) {
    nameplate.visible = Boolean(visible);
  }

  function dispose() {
    group.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        mats.forEach((m) => {
          if (m.map) m.map.dispose();
          m.dispose();
        });
      }
    });
    resources.textures.forEach((tex) => tex && tex.dispose());
    resources.textures.length = 0;
  }

  return {
    group,
    facingGroup,
    torsoGroup,
    headGroup,
    jawOrChinGroup,
    leftArm,
    rightArm,
    rimLight,
    thoughtHalo,
    nameplate,
    meta,
    seat: resolvedSeat,
    characterId: meta.id,
    setState,
    getState: () => currentState,
    update,
    getWorldHeadPosition,
    setNameplateVisible,
    dispose,
  };
}
