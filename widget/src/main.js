import * as THREE from 'three';

// ── Element colors (Jmol palette) ──────────────────────────────────────────
const ELEM_COLORS = {
  H:0xFFFFFF, He:0xD9FFFF, Li:0xCC80FF, Be:0xC2FF00, B:0xFFB5B5,
  C:0x909090, N:0x3050F8, O:0xFF0D0D, F:0x90E050, Ne:0xB3E3F5,
  Na:0xAB5CF2, Mg:0x8AFF00, Al:0xBFA6A6, Si:0xF0C8A0, P:0xFF8000,
  S:0xFFFF30, Cl:0x1FF01F, Ar:0x80D1E3, K:0x8F40D4, Ca:0x3DFF00,
  Sc:0xE6E6E6, Ti:0xBFC2C7, V:0xA6A6AB, Cr:0x8A99C7, Mn:0x9C7AC7,
  Fe:0xE06633, Co:0xF090A0, Ni:0x50D050, Cu:0xC88033, Zn:0x7D80B0,
  Ga:0xC28F8F, Ge:0x668F8F, As:0xBD80E3, Se:0xFFA100, Br:0xA62929,
  Kr:0x5CB8D1, Rb:0x702EB0, Sr:0x00FF00, Y:0x94FFFF, Zr:0x94E0E0,
  Nb:0x73C2C9, Mo:0x54B5B5, Tc:0x3B9E9E, Ru:0x248F8F, Rh:0x0A7D8C,
  Pd:0x006985, Ag:0xC0C0C0, Cd:0xFFD98F, In:0xA67573, Sn:0x668080,
  Sb:0x9E63B5, Te:0xD47A00, I:0x940094, Xe:0x429EB0, Cs:0x57178F,
  Ba:0x00C900, La:0x70D4FF, Ce:0xFFFFC7, Pr:0xD9FFC7, Nd:0xC7FFC7,
  Pm:0xA3FFC7, Sm:0x8FFFC7, Eu:0x61FFC7, Gd:0x45FFC7, Tb:0x30FFC7,
  Dy:0x1FFFC7, Ho:0x00FF9C, Er:0x00E675, Tm:0x00D452, Yb:0x00BF38,
  Lu:0x00AB24, Hf:0x4DC2FF, Ta:0x4DA6FF, W:0x2194D6, Re:0x267DAB,
  Os:0x266696, Ir:0x175487, Pt:0xD0D0E0, Au:0xFFD123, Hg:0xB8B8D0,
  Tl:0xA6544D, Pb:0x575961, Bi:0x9E4FB5, Po:0xAB5C00, At:0x754F45,
  Rn:0x428296, Fr:0x420066, Ra:0x007D00, Ac:0x70ABFA, Th:0x00BAFF,
  Pa:0x00A1FF, U:0x008FFF, Np:0x0080FF, Pu:0x006BFF,
};

const ELEM_RADII = {
  H:0.31, He:0.28, Li:1.28, Be:0.96, B:0.84, C:0.76, N:0.71, O:0.66,
  F:0.57, Ne:0.58, Na:1.66, Mg:1.41, Al:1.21, Si:1.11, P:1.07, S:1.05,
  Cl:1.02, Ar:1.06, K:2.03, Ca:1.76, Sc:1.70, Ti:1.60, V:1.53, Cr:1.39,
  Mn:1.61, Fe:1.52, Co:1.50, Ni:1.24, Cu:1.32, Zn:1.22, Ga:1.22, Ge:1.20,
  As:1.19, Se:1.20, Br:1.20, Kr:1.16, Rb:2.20, Sr:1.95, Y:1.90, Zr:1.75,
  Nb:1.64, Mo:1.54, Tc:1.47, Ru:1.46, Rh:1.42, Pd:1.39, Ag:1.45, Cd:1.44,
  In:1.42, Sn:1.39, Sb:1.39, Te:1.38, I:1.39, Xe:1.40, Cs:2.44, Ba:2.15,
  La:2.07, Ce:2.04, Pr:2.03, Nd:2.01, Pm:1.99, Sm:1.98, Eu:1.98, Gd:1.96,
  Tb:1.94, Dy:1.92, Ho:1.92, Er:1.89, Tm:1.90, Yb:1.87, Lu:1.87, Hf:1.75,
  Ta:1.70, W:1.62, Re:1.51, Os:1.44, Ir:1.41, Pt:1.36, Au:1.36, Hg:1.32,
  Tl:1.45, Pb:1.46, Bi:1.48, Po:1.40, At:1.50, Rn:1.50,
  Th:2.06, U:1.96,
};

function elemColor(s) { return new THREE.Color(ELEM_COLORS[s] ?? 0x999999); }
function elemRadius(s) { return (ELEM_RADII[s] ?? 1.2) * 0.4; }

// ── PMG JSON → {sites, lattice} ────────────────────────────────────────────
// Origin convention: unit cell centre (frac 0.5,0.5,0.5) maps to Cartesian (0,0,0).
// Each fractional coord is first wrapped into [0,1) so all atoms stay inside the cell.
function parsePmg(pmg) {
  const mat = pmg.lattice.matrix;

  // Cartesian position of the cell centre (frac 0.5,0.5,0.5)
  const cx = 0.5*mat[0][0] + 0.5*mat[1][0] + 0.5*mat[2][0];
  const cy = 0.5*mat[0][1] + 0.5*mat[1][1] + 0.5*mat[2][1];
  const cz = 0.5*mat[0][2] + 0.5*mat[1][2] + 0.5*mat[2][2];

  const sites = pmg.sites.map(site => {
    const sp = site.species[0]?.element ?? site.label ?? 'X';
    // wrap fractional coords into [0, 1)
    const fa = ((site.abc[0] % 1) + 1) % 1;
    const fb = ((site.abc[1] % 1) + 1) % 1;
    const fc = ((site.abc[2] % 1) + 1) % 1;
    const x = fa*mat[0][0] + fb*mat[1][0] + fc*mat[2][0] - cx;
    const y = fa*mat[0][1] + fb*mat[1][1] + fc*mat[2][1] - cy;
    const z = fa*mat[0][2] + fb*mat[1][2] + fc*mat[2][2] - cz;
    return { elem: sp, x, y, z };
  });
  return { sites, lattice: mat, cx, cy, cz };
}

// ── Three.js scene ─────────────────────────────────────────────────────────
const wrap = document.getElementById('canvas-wrap');
const W = () => wrap.clientWidth;
const H = () => wrap.clientHeight;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0xf0f4f8);  // light blue-grey background
renderer.setSize(W(), H());
wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, W() / H(), 0.1, 500);

const ambLight = new THREE.AmbientLight(0xffffff, 0.7);
scene.add(ambLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(2, 3, 4);
scene.add(dirLight);
const dirLight2 = new THREE.DirectionalLight(0xc8d8ff, 0.35);
dirLight2.position.set(-3, -1, -2);
scene.add(dirLight2);

// ── Orbit controls ─────────────────────────────────────────────────────────
let isDragging = false, isRightDrag = false;
let lastX = 0, lastY = 0;
let rotX = 0.4, rotY = 0.5;
let panX = 0, panY = 0;
let zoom = 1.0;
let radius = 10;

const canvas = renderer.domElement;

canvas.addEventListener('mousedown', e => {
  isDragging = true;
  isRightDrag = e.button === 2;
  lastX = e.clientX; lastY = e.clientY;
  e.preventDefault();
});
canvas.addEventListener('contextmenu', e => e.preventDefault());
window.addEventListener('mouseup', () => { isDragging = false; });
window.addEventListener('mousemove', e => {
  if (!isDragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  if (isRightDrag) {
    panX -= dx * 0.02 * zoom;
    panY += dy * 0.02 * zoom;
  } else {
    rotY += dx * 0.008;
    rotX += dy * 0.008;
    rotX = Math.max(-Math.PI/2, Math.min(Math.PI/2, rotX));
  }
  updateCamera();
});
canvas.addEventListener('wheel', e => {
  zoom *= e.deltaY > 0 ? 1.1 : 0.9;
  zoom = Math.max(0.1, Math.min(10, zoom));
  updateCamera();
  e.preventDefault();
}, { passive: false });

let lastTouchDist = null;
canvas.addEventListener('touchstart', e => {
  if (e.touches.length === 1) {
    isDragging = true; isRightDrag = false;
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
  } else if (e.touches.length === 2) {
    isDragging = false;
    lastTouchDist = Math.hypot(
      e.touches[0].clientX - e.touches[1].clientX,
      e.touches[0].clientY - e.touches[1].clientY,
    );
  }
  e.preventDefault();
}, { passive: false });
canvas.addEventListener('touchmove', e => {
  if (e.touches.length === 1 && isDragging) {
    const dx = e.touches[0].clientX - lastX, dy = e.touches[0].clientY - lastY;
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
    rotY += dx * 0.008; rotX += dy * 0.008;
    rotX = Math.max(-Math.PI/2, Math.min(Math.PI/2, rotX));
    updateCamera();
  } else if (e.touches.length === 2) {
    const dist = Math.hypot(
      e.touches[0].clientX - e.touches[1].clientX,
      e.touches[0].clientY - e.touches[1].clientY,
    );
    if (lastTouchDist) { zoom *= lastTouchDist / dist; zoom = Math.max(0.1, Math.min(10, zoom)); updateCamera(); }
    lastTouchDist = dist;
  }
  e.preventDefault();
}, { passive: false });
canvas.addEventListener('touchend', () => { isDragging = false; lastTouchDist = null; });

function updateCamera() {
  const x = radius * zoom * Math.sin(rotY) * Math.cos(rotX);
  const y = radius * zoom * Math.sin(rotX);
  const z = radius * zoom * Math.cos(rotY) * Math.cos(rotX);
  camera.position.set(x + panX, y + panY, z);
  camera.lookAt(panX, panY, 0);
}

// ── Structure rendering ─────────────────────────────────────────────────────
let structGroup = new THREE.Group();
scene.add(structGroup);

function buildStructure(pmg) {
  structGroup.clear();
  scene.remove(structGroup);
  structGroup = new THREE.Group();
  scene.add(structGroup);

  // sites already centred at cell centre (Cartesian origin = cell centre)
  const { sites, lattice, cx, cy, cz } = parsePmg(pmg);

  const geoCache = {};
  const matCache = {};
  const legendMap = {};

  sites.forEach(site => {
    const el = site.elem;
    if (!geoCache[el]) {
      geoCache[el] = new THREE.SphereGeometry(elemRadius(el), 22, 18);
      matCache[el] = new THREE.MeshPhongMaterial({
        color: elemColor(el),
        shininess: 80,
        specular: new THREE.Color(0x666666),
      });
    }
    const mesh = new THREE.Mesh(geoCache[el], matCache[el]);
    mesh.position.set(site.x, site.y, site.z);
    structGroup.add(mesh);
    legendMap[el] = ELEM_COLORS[el] ?? 0x999999;
  });

  // Unit cell wireframe — corners at frac 0/1, shifted so cell centre = origin
  const [a, b, c] = lattice;
  const verts = [
    [0,0,0],[1,0,0],[0,1,0],[1,1,0],
    [0,0,1],[1,0,1],[0,1,1],[1,1,1],
  ].map(([fa,fb,fc]) => new THREE.Vector3(
    fa*a[0] + fb*b[0] + fc*c[0] - cx,
    fa*a[1] + fb*b[1] + fc*c[1] - cy,
    fa*a[2] + fb*b[2] + fc*c[2] - cz,
  ));
  const edges = [
    [0,1],[2,3],[4,5],[6,7],
    [0,2],[1,3],[4,6],[5,7],
    [0,4],[1,5],[2,6],[3,7],
  ];
  const pts = [];
  edges.forEach(([i,j]) => { pts.push(verts[i].clone(), verts[j].clone()); });
  const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
  const lineMat = new THREE.LineBasicMaterial({ color: 0x6b7280, opacity: 0.7, transparent: true });
  structGroup.add(new THREE.LineSegments(lineGeo, lineMat));

  // Legend
  const legendEl = document.getElementById('legend');
  legendEl.innerHTML = Object.entries(legendMap)
    .map(([el, hex]) => {
      const col = hex.toString(16).padStart(6, '0');
      return `<div class="legend-item">
        <div class="legend-dot" style="background:#${col}"></div>
        <span>${el}</span>
      </div>`;
    }).join('');

  // Camera distance (sites already centred at origin)
  let maxDist = 0;
  sites.forEach(s => {
    const d = Math.sqrt(s.x**2 + s.y**2 + s.z**2);
    if (d > maxDist) maxDist = d;
  });
  radius = Math.max(maxDist * 1.8, 8);
  zoom = 1; panX = 0; panY = 0; rotX = 0.4; rotY = 0.5;
  updateCamera();
}

// ── Resize ─────────────────────────────────────────────────────────────────
function onResize() {
  renderer.setSize(W(), H());
  camera.aspect = W() / H();
  camera.updateProjectionMatrix();
}
new ResizeObserver(onResize).observe(wrap);

// ── Animate ────────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
updateCamera();
animate();

// ── Data & UI ──────────────────────────────────────────────────────────────
const BASE = import.meta.env.BASE_URL;

let allData = null;
let currentCatId = null;
let currentFormula = null;
let polymorphIdx = 0;

const catSelect = document.getElementById('cat-select');
const formulaList = document.getElementById('formula-list');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');

async function init() {
  const res = await fetch(`${BASE}data/structures.json`);
  allData = await res.json();

  allData.categories.forEach(cat => {
    const opt = document.createElement('option');
    opt.value = cat.id;
    opt.textContent = `${cat.label} (${cat.entries.length})`;
    catSelect.appendChild(opt);
  });

  catSelect.addEventListener('change', () => selectCategory(catSelect.value));
  prevBtn.addEventListener('click', () => { polymorphIdx--; showCurrent(); });
  nextBtn.addEventListener('click', () => { polymorphIdx++; showCurrent(); });

  selectCategory(allData.categories[0].id);
}

function selectCategory(catId) {
  currentCatId = catId;
  catSelect.value = catId;

  const cat = allData.categories.find(c => c.id === catId);
  const groups = groupByFormula(cat.entries);
  const formulas = Array.from(groups.keys());

  formulaList.innerHTML = '';
  formulas.forEach(formula => {
    const btn = document.createElement('button');
    btn.className = 'formula-item';
    const group = groups.get(formula);
    btn.innerHTML = `<span>${formula}</span>${group.length > 1 ? `<span class="badge">${group.length}</span>` : ''}`;
    btn.dataset.formula = formula;
    btn.addEventListener('click', () => selectFormula(formula));
    formulaList.appendChild(btn);
  });

  selectFormula(formulas[0]);
}

function groupByFormula(entries) {
  const map = new Map();
  entries.forEach(e => {
    const arr = map.get(e.formula) ?? [];
    arr.push(e);
    map.set(e.formula, arr);
  });
  return map;
}

function selectFormula(formula) {
  currentFormula = formula;
  polymorphIdx = 0;

  formulaList.querySelectorAll('.formula-item').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.formula === formula);
  });
  const active = formulaList.querySelector('.formula-item.active');
  if (active) active.scrollIntoView({ block: 'nearest' });

  showCurrent();
}

function showCurrent() {
  const cat = allData.categories.find(c => c.id === currentCatId);
  const groups = groupByFormula(cat.entries);
  const entries = groups.get(currentFormula) ?? [];

  polymorphIdx = Math.max(0, Math.min(polymorphIdx, entries.length - 1));
  const entry = entries[polymorphIdx];
  if (!entry) return;

  prevBtn.disabled = polymorphIdx === 0;
  nextBtn.disabled = polymorphIdx === entries.length - 1;

  buildStructure(entry.pmg);
}

init().catch(console.error);
