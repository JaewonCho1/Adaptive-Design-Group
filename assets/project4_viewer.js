// Project 4 -- live 3D comparison of quadcopter airframe designs.
// Reads assets/project4_results.json (written by project4_finalize.py).
// Members are drawn as tubes along cubic Hermite curves that use both end
// rotations of every beam element (the element's own shape functions).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const NSEG = 8;          // segments along each element
const NRING = 8;         // vertices around each tube
const SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];   // single-hue blue
const DIV_NEG = ['#7f1d1d', '#b83a3a', '#e07b7b', '#f0efec'];    // compression (red) -> neutral
const DIV_POS = ['#f0efec', '#86b6ef', '#2a78d6', '#104281'];    // neutral -> tension (blue)

function hexToRgb(h) {
  const n = parseInt(h.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
function rampColor(stops, t) {
  t = Math.min(1, Math.max(0, t));
  const x = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = hexToRgb(stops[i]), b = hexToRgb(stops[i + 1]);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
function divColor(v, vmax) {
  const t = v / vmax;
  return t < 0 ? rampColor(DIV_NEG, 1 + t) : rampColor(DIV_POS, t);
}

const FIELDS = {
  util: {
    label: 'Fatigue utilization  σ_vm / σ_allow  (R2)', kind: 'seq',
    get: (d, e) => d.util[e], range: () => [0, 1], limit: 1, fmt: v => (100 * v).toFixed(0) + '%'
  },
  axial: {
    label: 'Axial force N (N): compression ← → tension', kind: 'div',
    get: (d, e) => d.N[e], range: (ds) => { const m = Math.max(...ds.map(d => Math.max(...d.N.map(Math.abs)))); return [-m, m]; },
    fmt: v => v.toFixed(0)
  },
  radius: {
    label: 'Tube outer radius r (mm)', kind: 'seq',
    get: (d, e) => d.r[e] * 1000, range: (ds, spec) => [spec.r_min * 1000, spec.r_max * 1000],
    fmt: v => v.toFixed(1)
  },
  slender: {
    label: 'Member slenderness L / r_gyr  (R6)', kind: 'seq',
    get: (d, e) => d.slender[e], range: (ds, spec) => [0, spec.slender_max], limit: 'spec', fmt: v => v.toFixed(0)
  },
};

function el(tag, attrs = {}, html = '') {
  const e = document.createElement(tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (html) e.innerHTML = html;
  return e;
}

export async function mountViewer(root) {
  const data = await (await fetch(root.dataset.src || 'assets/project4_results.json')).json();
  const spec = data.spec;
  const ne = data.elements.length;
  const designs = data.designs;
  const byName = Object.fromEntries(designs.map(d => [d.name, d]));

  // ---------- DOM ----------
  root.classList.add('p4v');
  const bar = el('div', { class: 'p4v-bar' });
  const selL = el('select', { 'aria-label': 'left design' });
  const selR = el('select', { 'aria-label': 'right design' });
  for (const d of designs) { selL.append(new Option(d.label, d.name)); selR.append(new Option(d.label, d.name)); }
  selL.value = 'initial'; selR.value = 'optimized';
  const selField = el('select', { 'aria-label': 'color by' });
  for (const k in FIELDS) selField.append(new Option(FIELDS[k].label.split('  ')[0].split(' (')[0], k));
  const selAnim = el('select', { 'aria-label': 'shape shown' });
  [['static', 'Deformed shape under max thrust'], ['mode', 'Animate: 1st vibration mode'],
   ['buck', 'Animate: 1st buckling mode'], ['ramp', 'Animate: thrust 0 → max']].forEach(([v, t]) => selAnim.append(new Option(t, v)));
  const slider = el('input', { type: 'range', min: '0', max: '100', value: '40', 'aria-label': 'deformation scale' });
  const scaleTxt = el('span', { class: 'p4v-small' });
  const selRad = el('select', { 'aria-label': 'tube radius display' });
  selRad.append(new Option('tubes: true radius', '1'));
  selRad.append(new Option('tubes: radius × 2', '2'));
  const chkProp = el('input', { type: 'checkbox', id: 'p4v-prop' });
  chkProp.checked = true;
  const lab = (t, c) => { const l = el('label', {}, `<span>${t}</span>`); l.append(c); return l; };
  bar.append(lab('Left', selL), lab('Right', selR), lab('Color', selField), lab('Show', selAnim));
  const sl = el('label', {}, '<span>Deformation</span>'); sl.append(slider, scaleTxt);
  const pl = el('label', { for: 'p4v-prop' }, ''); pl.append(chkProp, document.createTextNode(' propeller disks'));
  bar.append(sl, lab('', selRad), pl);
  root.append(bar);

  const legend = el('div', { class: 'p4v-legend' });
  root.append(legend);
  const panesDiv = el('div', { class: 'p4v-panes' });
  root.append(panesDiv);
  const tip = el('div', { class: 'p4v-tip' });
  tip.style.display = 'none';
  document.body.append(tip);
  root.append(el('p', { class: 'p4v-small' },
    'Drag to orbit (the two views are linked), scroll to zoom, hover a member for its values. ' +
    'Dark pyramids: payload mounts (clamped). Orange arrows: rotor thrust. ' +
    '◆ black diamond: member over its limit for the colored quantity. ' +
    'Geometry and tube radii are true scale unless “radius × 2” is chosen; displacements are exaggerated by the factor shown.'));

  // ---------- shared scales ----------
  const maxU = Math.max(...designs.map(d => {
    let m = 0; for (let i = 0; i < d.u.length; i += 6) m = Math.max(m, Math.hypot(d.u[i], d.u[i + 1], d.u[i + 2])); return m;
  }));
  const VIS = 0.06; // metres of visual displacement at 100 %

  // ---------- panes ----------
  function makePane(side) {
    const wrap = el('div', { class: 'p4v-pane' });
    const title = el('div', { class: 'p4v-title' });
    const holder = el('div', { class: 'p4v-canvas' });
    const metrics = el('div', { class: 'p4v-metrics' });
    wrap.append(title, holder, metrics);
    panesDiv.append(wrap);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setClearColor(0xfcfcfb, 1);
    holder.append(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 20);
    camera.up.set(0, 0, 1);
    camera.position.set(0.95, -1.12, 0.78);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.05);
    controls.update();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a80, 1.6));
    const dl = new THREE.DirectionalLight(0xffffff, 1.6);
    dl.position.set(1, -2, 3);
    scene.add(dl);
    const grid = new THREE.GridHelper(2, 20, 0xd8d7d2, 0xe9e8e4);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -0.002;
    scene.add(grid);
    // tube mesh
    const nv = ne * (NSEG + 1) * NRING;
    const pos = new Float32Array(nv * 3);
    const col = new Float32Array(nv * 3);
    const idx = [];
    for (let e = 0; e < ne; e++) {
      const b = e * (NSEG + 1) * NRING;
      for (let s = 0; s < NSEG; s++) for (let k = 0; k < NRING; k++) {
        const a0 = b + s * NRING + k, a1 = b + s * NRING + (k + 1) % NRING;
        const c0 = a0 + NRING, c1 = a1 + NRING;
        idx.push(a0, c0, a1, a1, c0, c1);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(idx);
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.1 }));
    scene.add(mesh);
    const deco = new THREE.Group();
    scene.add(deco);
    const marks = new THREE.Group();
    scene.add(marks);
    return { side, wrap, title, holder, metrics, renderer, scene, camera, controls, mesh, geo, pos, col, deco, marks, d: null };
  }
  const panes = [makePane('L'), makePane('R')];

  // linked cameras
  let syncing = false;
  for (const p of panes) {
    p.controls.addEventListener('change', () => {
      if (syncing) return;
      syncing = true;
      for (const q of panes) if (q !== p) {
        q.camera.position.copy(p.camera.position);
        q.camera.quaternion.copy(p.camera.quaternion);
        q.controls.target.copy(p.controls.target);
        q.controls.update();
      }
      syncing = false;
    });
  }

  function setDesign(p, d) {
    p.d = d;
    p.title.textContent = d.label;
    // decorations: supports, motors, thrust arrows, propeller disks
    p.deco.clear();
    const X = d.nodes;
    const node = i => new THREE.Vector3(X[3 * i], X[3 * i + 1], X[3 * i + 2]);
    const pyr = new THREE.ConeGeometry(0.016, 0.03, 4);
    pyr.rotateX(Math.PI / 2);
    for (const s of data.supports) {
      const m = new THREE.Mesh(pyr, new THREE.MeshStandardMaterial({ color: 0x2b2b2b }));
      const v = node(s);
      m.position.set(v.x, v.y, v.z - 0.017);
      p.deco.add(m);
    }
    const propGroup = new THREE.Group();
    propGroup.name = 'props';
    data.motors.forEach((mi, k) => {
      const v = node(mi);
      const body = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 0.03, 20), new THREE.MeshStandardMaterial({ color: 0x55544f }));
      body.rotation.x = Math.PI / 2;
      body.position.set(v.x, v.y, v.z + 0.015);
      p.deco.add(body);
      const arr = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(v.x, v.y, v.z + 0.035), 0.14, 0xeb6834, 0.035, 0.022);
      p.deco.add(arr);
      const disk = new THREE.Mesh(new THREE.CircleGeometry(spec.prop_diameter / 2, 48),
        new THREE.MeshBasicMaterial({ color: 0x52514e, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false }));
      disk.position.set(v.x, v.y, v.z + 0.04);
      propGroup.add(disk);
    });
    p.deco.add(propGroup);
    propGroup.visible = chkProp.checked;
    // metrics panel
    let h = '<table><tr><th></th><th>requirement</th><th>value</th><th>limit</th><th></th></tr>';
    for (const m of d.metrics) {
      const v = m.unit === 'kg' ? m.value.toFixed(3) : (Math.abs(m.value) >= 100 ? m.value.toFixed(0) : m.value.toFixed(2));
      const st = m.ok === null ? '' : (m.ok ? '<span class="ok">✓ pass</span>' : '<span class="bad">✗ FAIL</span>');
      h += `<tr><td>${m.id}</td><td>${m.name}</td><td>${v} ${m.unit}</td><td>${m.limit ?? 'minimize'}</td><td>${st}</td></tr>`;
    }
    p.metrics.innerHTML = h + '</table>';
  }

  // ---------- geometry update ----------
  const R = new Float32Array(9);
  function updateGeometry(p, field, s) {
    const d = p.d, X = d.nodes, E = data.elements, pos = p.pos;
    const rs = parseFloat(selRad.value);
    for (let e = 0; e < ne; e++) {
      const i = E[e][0], j = E[e][1];
      for (let k = 0; k < 9; k++) R[k] = d.R[9 * e + k];
      const xi = [X[3 * i], X[3 * i + 1], X[3 * i + 2]], xj = [X[3 * j], X[3 * j + 1], X[3 * j + 2]];
      const L = Math.hypot(xj[0] - xi[0], xj[1] - xi[1], xj[2] - xi[2]);
      // local DOFs
      const loc = (n, off) => [0, 1, 2].map(a => R[3 * a] * field[6 * n + off] + R[3 * a + 1] * field[6 * n + off + 1] + R[3 * a + 2] * field[6 * n + off + 2]);
      const di = loc(i, 0), ti = loc(i, 3), dj = loc(j, 0), tj = loc(j, 3);
      const rad = d.r[e] * rs;
      const b = e * (NSEG + 1) * NRING;
      for (let q = 0; q <= NSEG; q++) {
        const x = q / NSEG;
        const N1 = 1 - 3 * x * x + 2 * x * x * x, N2 = x - 2 * x * x + x * x * x, N3 = 3 * x * x - 2 * x * x * x, N4 = -x * x + x * x * x;
        const ua = (1 - x) * di[0] + x * dj[0];
        const v = N1 * di[1] + N2 * L * ti[2] + N3 * dj[1] + N4 * L * tj[2];
        const w = N1 * di[2] - N2 * L * ti[1] + N3 * dj[2] - N4 * L * tj[1];
        const lx = x * L + s * ua, ly = s * v, lz = s * w;
        const cx = xi[0] + lx * R[0] + ly * R[3] + lz * R[6];
        const cy = xi[1] + lx * R[1] + ly * R[4] + lz * R[7];
        const cz = xi[2] + lx * R[2] + ly * R[5] + lz * R[8];
        for (let k = 0; k < NRING; k++) {
          const a = 2 * Math.PI * k / NRING, c = Math.cos(a) * rad, sn = Math.sin(a) * rad;
          const o = 3 * (b + q * NRING + k);
          pos[o] = cx + c * R[3] + sn * R[6];
          pos[o + 1] = cy + c * R[4] + sn * R[7];
          pos[o + 2] = cz + c * R[5] + sn * R[8];
        }
      }
    }
    p.geo.attributes.position.needsUpdate = true;
    p.geo.computeVertexNormals();
    p.geo.computeBoundingSphere();
  }

  function colorRange() {
    const F = FIELDS[selField.value];
    return F.range(panes.map(p => p.d), spec);
  }

  function updateColors(p) {
    const F = FIELDS[selField.value];
    const [lo, hi] = colorRange();
    const lim = F.limit === 'spec' ? spec.slender_max : F.limit;
    p.marks.clear();
    const markGeo = new THREE.OctahedronGeometry(0.011);
    const markMat = new THREE.MeshBasicMaterial({ color: 0x0b0b0b });
    for (let e = 0; e < ne; e++) {
      const v = F.get(p.d, e);
      const c = F.kind === 'div' ? divColor(v, hi) : rampColor(SEQ, (v - lo) / (hi - lo));
      const b = e * (NSEG + 1) * NRING;
      for (let k = 0; k < (NSEG + 1) * NRING; k++) { p.col[3 * (b + k)] = c[0]; p.col[3 * (b + k) + 1] = c[1]; p.col[3 * (b + k) + 2] = c[2]; }
      if (lim !== undefined && v > lim) {
        const m = new THREE.Mesh(markGeo, markMat);
        m.userData.e = e;
        p.marks.add(m);
      }
    }
    p.geo.attributes.color.needsUpdate = true;
  }

  function placeMarks(p) {
    for (const m of p.marks.children) {
      const e = m.userData.e;
      const b = e * (NSEG + 1) * NRING + (NSEG / 2) * NRING;
      let x = 0, y = 0, z = 0;
      for (let k = 0; k < NRING; k++) { x += p.pos[3 * (b + k)]; y += p.pos[3 * (b + k) + 1]; z += p.pos[3 * (b + k) + 2]; }
      m.position.set(x / NRING, y / NRING, z / NRING + 0.02);
    }
  }

  function updateLegend() {
    const F = FIELDS[selField.value];
    const [lo, hi] = colorRange();
    let grad;
    if (F.kind === 'div') grad = `linear-gradient(90deg, ${DIV_NEG.join(',')}, ${DIV_POS.slice(1).join(',')})`;
    else grad = `linear-gradient(90deg, ${SEQ.join(',')})`;
    const mid = F.kind === 'div' ? 0 : (lo + hi) / 2;
    const lim = F.limit === 'spec' ? spec.slender_max : F.limit;
    legend.innerHTML = `<div class="p4v-small"><b>${F.label}</b>${lim !== undefined ? ' — ◆ marks members above the limit (' + F.fmt(lim) + ')' : ''}</div>
      <div class="p4v-cbar" style="background:${grad}"></div>
      <div class="p4v-ticks"><span>${F.fmt(lo)}</span><span>${F.fmt(mid)}</span><span>${F.fmt(hi)}${F.kind === 'seq' && lim !== undefined ? ' and above' : ''}</span></div>`;
  }

  function currentField(p, t) {
    const mode = selAnim.value;
    const sv = parseFloat(slider.value) / 100;
    if (mode === 'static') return [p.d.u, sv * VIS / maxU];
    if (mode === 'ramp') return [p.d.u, sv * VIS / maxU * (0.5 - 0.5 * Math.cos(2 * Math.PI * t / 2.4))];
    const f = mode === 'mode' ? p.d.mode : p.d.buck;
    return [f, sv * VIS * Math.sin(2 * Math.PI * t / 1.6)];
  }

  function updateScaleText() {
    const mode = selAnim.value;
    const sv = parseFloat(slider.value) / 100;
    if (mode === 'static' || mode === 'ramp') scaleTxt.textContent = ` displacements × ${(sv * VIS / maxU).toFixed(0)}`;
    else {
      const t = panes.map(p => mode === 'mode' ? `${p.d.mode_freq.toFixed(1)} Hz` : `λ = ${p.d.buck_lam.toFixed(1)}`);
      scaleTxt.textContent = ` mode shape, arbitrary amplitude (${t[0]} | ${t[1]})`;
    }
  }

  function resize() {
    for (const p of panes) {
      const w = p.holder.clientWidth, h = p.holder.clientHeight;
      p.renderer.setSize(w, h, false);
      p.camera.aspect = w / h;
      p.camera.updateProjectionMatrix();
    }
  }

  function refresh() {
    setDesign(panes[0], byName[selL.value]);
    setDesign(panes[1], byName[selR.value]);
    for (const p of panes) updateColors(p);
    updateLegend();
    updateScaleText();
    needsGeo = true;
  }
  let needsGeo = true;
  selL.onchange = selR.onchange = selField.onchange = refresh;
  selAnim.onchange = () => { updateScaleText(); needsGeo = true; };
  slider.oninput = () => { updateScaleText(); needsGeo = true; };
  selRad.onchange = () => { needsGeo = true; };
  chkProp.onchange = () => { for (const p of panes) { const g = p.deco.getObjectByName('props'); if (g) g.visible = chkProp.checked; } };
  new ResizeObserver(resize).observe(panesDiv);
  refresh();
  resize();

  // ---------- hover ----------
  const ray = new THREE.Raycaster();
  const ptr = new THREE.Vector2();
  for (const p of panes) {
    const cv = p.renderer.domElement;
    cv.addEventListener('pointermove', ev => {
      const r = cv.getBoundingClientRect();
      ptr.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
      ptr.y = -((ev.clientY - r.top) / r.height) * 2 + 1;
      ray.setFromCamera(ptr, p.camera);
      const hit = ray.intersectObject(p.mesh)[0];
      if (!hit) { tip.style.display = 'none'; return; }
      const e = Math.floor(hit.faceIndex / (NSEG * NRING * 2));
      const d = p.d, m = data.elem_member[e], g = data.groups[data.member_group[m]];
      const util = d.util[e], vm = Math.max(d.vm[2 * e], d.vm[2 * e + 1]);
      const over = util > 1 || d.slender[e] > spec.slender_max;
      tip.innerHTML = `<b>member ${m}</b> (${g})<br>element ${e} (two elements per member)<br>` +
        `radius ${(d.r[e] * 1000).toFixed(2)} mm, wall ${(d.r[e] * 100).toFixed(2)} mm<br>` +
        `length ${(d.L_member[m] * 1000).toFixed(0)} mm, L/r = ${d.slender[e].toFixed(0)}<br>` +
        `axial force ${d.N[e].toFixed(1)} N (${d.N[e] < 0 ? 'compression' : 'tension'})<br>` +
        `peak von Mises ${vm.toFixed(1)} MPa<br>fatigue utilization ${(100 * util).toFixed(0)}%` +
        (over ? '<br><b>◆ OVER LIMIT</b>' : '');
      tip.style.display = 'block';
      tip.style.left = (ev.clientX + 14) + 'px';
      tip.style.top = (ev.clientY + 14) + 'px';
    });
    cv.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
  }

  // ---------- render loop ----------
  const t0 = performance.now();
  function frame() {
    const t = (performance.now() - t0) / 1000;
    const animated = selAnim.value !== 'static';
    for (const p of panes) {
      if (animated || needsGeo) {
        const [f, s] = currentField(p, t);
        updateGeometry(p, f, s);
        placeMarks(p);
      }
      p.renderer.render(p.scene, p.camera);
    }
    needsGeo = false;
    requestAnimationFrame(frame);
  }
  frame();
}

const rootEl = document.getElementById('p4-viewer');
if (rootEl) mountViewer(rootEl).catch(err => { rootEl.textContent = 'Viewer failed to load: ' + err; });
