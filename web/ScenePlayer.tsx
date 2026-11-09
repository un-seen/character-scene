import { useEffect, useRef, useState } from 'react';
import { Pause, Play } from 'lucide-react';
import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/examples/jsm/utils/SkeletonUtils.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/**
 * THE SCENE, PLAYED. Each step of a live scene (the `scene` program) brings
 * one clip per character (.glb, keyed on the rig's bones), which plays
 * on that character's own model by bone name. A step starts each body where
 * the last one left it and blends its pose in over half a second; the
 * speaker's line shows in a bubble over its head while the step plays. When
 * the newest step has played, the bodies hold their last pose until the next
 * arrives. Ported from the simulator's page (server/static/index.html:
 * applyStep, showNow, faceClip, hipsTurned, blendPose).
 */

export type SceneMember = {
  /** the cast member, the same through renames and new looks (older scenes: absent, the look is the key) */
  id?: string;
  character: string;
  name: string;
  kind: 'speak' | 'listen';
  says: string;
  act: string;
  clip: string | null;
  frames: number | null;
  fps: number | null;
  end_yaw: number | null;
};
export type SceneStep = { n: number; members: SceneMember[] };

const SPACING = 1.4;
const ROOT_BONES = ['ROOT-MAIN', 'ROOT'];
const FADE = 0.5;
const CUTOUT = /hair|afro|ponytail|bob|braid|short0|long0|brow|lash/i;
const WPS = 2.5;

type Rest = { p: THREE.Vector3; q: THREE.Quaternion; s: THREE.Vector3 };
type PoseKey = { b: THREE.Bone; p: THREE.Vector3; q: THREE.Quaternion; s: THREE.Vector3; top: boolean; w: THREE.Matrix4 };
type Actor = {
  id: string;
  character: string;
  root: THREE.Group;
  mixer: THREE.AnimationMixer;
  hips: THREE.Object3D | null;
  head: THREE.Object3D | null;
  bones: THREE.Bone[];
  rest: Map<THREE.Bone, Rest>;
  faces: THREE.Mesh[];
  end: THREE.Vector3;
  yaw: number;
  blend: { pose: PoseKey[]; t: number } | null;
};
type Part = { action: THREE.AnimationAction; root: THREE.Vector3; yaw: number; member: SceneMember };
type Played = { n: number; parts: Map<Actor, Part>; duration: number; ids: Set<string>; members: SceneMember[] };
/** What the timeline overlay draws: every step made so far, the one playing and how far into it. */
type Timeline = { steps: { n: number; duration: number; members: SceneMember[] }[]; at: number; frac: number; paused: boolean };

function dress(o: THREE.Mesh) {
  for (const m of ([] as THREE.Material[]).concat(o.material)) {
    const cut = CUTOUT.test(m.name || o.name);
    m.transparent = false;
    m.depthWrite = true;
    m.alphaTest = cut ? 0.5 : 0;
    m.side = cut ? THREE.DoubleSide : THREE.FrontSide;
    m.needsUpdate = true;
  }
}

/** A clip's face keys, one track in the file, split into one track per key for each of the person's meshes that has it. */
function faceClip(a: Actor, gltf: GLTF) {
  const clip = gltf.animations[0];
  const json = gltf.parser.json as { meshes?: { extras?: { targetNames?: string[] } }[] };
  const mesh = (json.meshes || []).find((m) => m.extras && m.extras.targetNames);
  const names = mesh?.extras?.targetNames ?? [];
  const n = names.length;
  const tracks: THREE.KeyframeTrack[] = [];
  for (const t of clip.tracks) {
    if (!t.name.endsWith('.morphTargetInfluences')) {
      tracks.push(t);
      continue;
    }
    if (!n) continue;
    const k = t.times.length;
    for (const m of a.faces)
      names.forEach((name, j) => {
        const idx = m.morphTargetDictionary?.[name];
        if (idx === undefined) return;
        const v = new Float32Array(k);
        for (let f = 0; f < k; f++) v[f] = t.values[f * n + j];
        tracks.push(new THREE.NumberKeyframeTrack(`${m.uuid}.morphTargetInfluences[${idx}]`, t.times as unknown as number[], v as unknown as number[]));
      });
  }
  return new THREE.AnimationClip(clip.name, clip.duration, tracks);
}

function hipsAt(a: Actor, action: THREE.AnimationAction, t: number) {
  const all = (a.mixer as unknown as { _actions: THREE.AnimationAction[] })._actions;
  const saved = all.map((x) => [x, x.enabled, x.time, x.getEffectiveWeight()] as const);
  for (const x of all) x.enabled = false;
  action.enabled = true;
  action.play();
  action.time = t;
  action.setEffectiveWeight(1);
  a.mixer.update(0);
  const p = new THREE.Vector3();
  a.hips?.getWorldPosition(p);
  p.sub(a.root.position);
  for (const [x, e, tm, w] of saved) {
    x.enabled = e;
    x.time = tm;
    x.setEffectiveWeight(w);
  }
  if (!saved.some(([x]) => x === action)) action.enabled = false;
  a.mixer.update(0);
  return p;
}

function hipsTurned(a: Actor, action: THREE.AnimationAction, yaw: number) {
  const was = a.root.rotation.y;
  a.root.rotation.y = yaw;
  a.root.updateMatrixWorld(true);
  const out = [hipsAt(a, action, 0), hipsAt(a, action, action.getClip().duration)];
  a.root.rotation.y = was;
  a.root.updateMatrixWorld(true);
  return out;
}

function capturePose(a: Actor): PoseKey[] {
  a.root.updateMatrixWorld(true);
  return a.bones.map((b) => ({ b, p: b.position.clone(), q: b.quaternion.clone(), s: b.scale.clone(), top: !(b.parent as THREE.Bone | null)?.isBone, w: b.matrixWorld.clone() }));
}

function rebase(a: Actor, pose: PoseKey[]) {
  a.root.updateMatrixWorld(true);
  const inv = new THREE.Matrix4();
  for (const x of pose)
    if (x.top && x.b.parent) {
      inv.copy(x.b.parent.matrixWorld).invert();
      new THREE.Matrix4().multiplyMatrices(inv, x.w).decompose(x.p, x.q, x.s);
    }
  return pose;
}

const _bv = new THREE.Vector3();
const _bq = new THREE.Quaternion();
function blendPose(a: Actor, dt: number) {
  if (!a.blend) return;
  a.blend.t += dt;
  const u0 = Math.min(1, a.blend.t / FADE);
  const u = u0 * u0 * (3 - 2 * u0);
  for (const x of a.blend.pose) {
    _bv.copy(x.b.position);
    x.b.position.lerpVectors(x.p, _bv, u);
    _bq.copy(x.b.quaternion);
    x.b.quaternion.slerpQuaternions(x.q, _bq, u);
    _bv.copy(x.b.scale);
    x.b.scale.lerpVectors(x.s, _bv, u);
  }
  if (u0 >= 1) a.blend = null;
}

function floorTexture() {
  const cv = document.createElement('canvas');
  cv.width = cv.height = 64;
  const g = cv.getContext('2d')!;
  g.fillStyle = '#a8a39a';
  g.fillRect(0, 0, 64, 64);
  g.fillStyle = '#9b968d';
  g.fillRect(0, 0, 32, 32);
  g.fillRect(32, 32, 32, 32);
  const t = new THREE.CanvasTexture(cv);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(75, 75);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export default function ScenePlayer({
  steps,
  models,
  className = '',
}: {
  steps: SceneStep[];
  models: Record<string, string>;
  className?: string;
}) {
  const box = useRef<HTMLDivElement>(null);
  const bubbleRef = useRef<HTMLDivElement>(null);
  const engine = useRef<{
    add: (s: SceneStep) => void;
    seen: Set<number>;
    seek: (i: number) => void;
    toggle: () => void;
  } | null>(null);
  const [tl, setTl] = useState<Timeline>({ steps: [], at: -1, frac: 0, paused: false });
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState('');

  // The world, built once.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    el.prepend(renderer.domElement);
    renderer.domElement.className = 'block h-full w-full rounded-xl';
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#d9d6cf');
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 200);
    camera.position.set(0, 1.7, 6.5);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.95, 0);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI * 0.49;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8478, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(4, 9, 3);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    scene.add(sun, sun.target);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(300, 300), new THREE.MeshStandardMaterial({ map: floorTexture(), roughness: 0.9 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    const loader = new GLTFLoader();
    const load = (url: string) => new Promise<GLTF | null>((res) => loader.load(url, res, undefined, () => res(null)));
    const modelCache: Record<string, Promise<GLTF | null>> = {};
    const actors = new Map<string, Actor>();
    // every step made, kept so the timeline can go back to one; `at` is the one playing
    const history: Played[] = [];
    let at = -1;
    let paused = false;
    let cur: Played | null = null;
    let stepT = 0;
    let shownAt = 0;
    const show = () =>
      setTl({
        steps: history.map((h) => ({ n: h.n, duration: h.duration, members: h.members })),
        at,
        frac: cur ? Math.min(1, stepT / cur.duration) : 0,
        paused,
      });
    let chain = Promise.resolve();
    const follow = new THREE.Vector3(0, 0.95, 0);
    const clock = new THREE.Clock();
    let raf = 0;

    /** A member's body: made the first time it is seen; made again in its new look where it stands when its look changes. */
    async function actorFor(m: SceneMember, slot: number, count: number) {
      const id = m.id || m.character;
      const was = actors.get(id);
      if (was && was.character === m.character) return was;
      const url = models[m.character];
      if (!url) return was || null;
      modelCache[url] = modelCache[url] || load(url);
      const model = await modelCache[url];
      if (!model) return was || null;
      const body = SkeletonUtils.clone(model.scene);
      const root = new THREE.Group();
      root.add(body);
      root.position.x = (slot - (count - 1) / 2) * SPACING;
      let hips: THREE.Object3D | null = null;
      let head: THREE.Object3D | null = null;
      const bones: THREE.Bone[] = [];
      const faces: THREE.Mesh[] = [];
      body.traverse((o) => {
        const b = o as THREE.Bone;
        const mesh = o as THREE.Mesh;
        if (b.isBone) bones.push(b);
        if (mesh.isMesh && mesh.morphTargetDictionary) faces.push(mesh);
        if (mesh.isMesh) {
          dress(mesh);
          mesh.castShadow = true;
          mesh.frustumCulled = false;
        }
        if (b.isBone && o.name === 'DEF-HIPS') hips = o;
        if (b.isBone && o.name === 'DEF-HEAD') head = o;
      });
      const rest = new Map(bones.map((b) => [b, { p: b.position.clone(), q: b.quaternion.clone(), s: b.scale.clone() }]));
      const a: Actor = { id, character: m.character, root, mixer: new THREE.AnimationMixer(body), hips, head, bones, rest, faces, end: root.position.clone(), yaw: 0, blend: null };
      if (was) {
        // the new look takes the old one's place and heading, then the old body goes
        a.end = was.end.clone();
        a.yaw = was.yaw;
        root.position.copy(was.root.position);
        root.rotation.y = was.root.rotation.y;
        was.mixer.stopAllAction();
        scene.remove(was.root);
      }
      scene.add(root);
      actors.set(id, a);
      return a;
    }

    async function prepare(s: SceneStep) {
      const parts = new Map<Actor, Part>();
      const live = s.members.filter((m) => m.clip);
      await Promise.all(
        live.map(async (m) => {
          const a = await actorFor(m, s.members.indexOf(m), s.members.length);
          const gltf = m.clip ? await load(m.clip) : null;
          if (!a || !gltf || !gltf.animations.length) return;
          const action = a.mixer.clipAction(faceClip(a, gltf));
          action.setLoop(THREE.LoopOnce, 1);
          action.clampWhenFinished = true;
          const yaw = a.yaw;
          const [h0, h1] = hipsTurned(a, action, yaw);
          const root = new THREE.Vector3(a.end.x - h0.x, 0, a.end.z - h0.z);
          a.end = new THREE.Vector3(root.x + h1.x, 0, root.z + h1.z);
          a.yaw = yaw + THREE.MathUtils.degToRad(m.end_yaw || 0);
          parts.set(a, { action, root, yaw, member: m });
        }),
      );
      const words = s.members.reduce((k, m) => k + (m.says ? m.says.split(/\s+/).length : 0), 0);
      const clipS = Math.max(0, ...live.map((m) => (m.frames || 0) / (m.fps || 24)));
      history.push({ n: s.n, parts, duration: Math.max(clipS, words / WPS + 1.2), ids: new Set(s.members.map((m) => m.id || m.character)), members: s.members });
      if (!cur) play(history.length - 1);
      show();
    }

    /** Play step i of the history from its start; the next ones follow it. */
    function play(i: number) {
      const s = history[i];
      if (!s) return;
      const prev = cur;
      for (const [a, part] of Array.from(s.parts.entries())) {
        const from = prev && prev.parts.has(a) ? rebase(a, capturePose(a)) : null;
        a.root.position.copy(part.root);
        a.root.rotation.y = part.yaw;
        part.action.reset();
        part.action.enabled = true;
        part.action.setEffectiveWeight(1);
        part.action.play();
        for (const x of (a.mixer as unknown as { _actions: THREE.AnimationAction[] })._actions) if (x !== part.action) x.stop();
        for (const n of ROOT_BONES) {
          const b = a.bones.find((x) => x.name === n);
          const r = b && a.rest.get(b);
          if (b && r) {
            b.position.copy(r.p);
            b.quaternion.copy(r.q);
            b.scale.copy(r.s);
          }
        }
        a.blend = from ? { pose: from.filter((x: PoseKey) => !ROOT_BONES.includes(x.b.name)), t: 0 } : null;
      }
      for (const a of Array.from(actors.values())) a.root.visible = s.ids.has(a.id);   // someone who left the cast is not shown
      cur = s;
      at = i;
      stepT = 0;
      show();
    }

    function bubble() {
      const el2 = bubbleRef.current;
      if (!el2) return;
      const speaker = cur && Array.from(cur.parts.entries()).find(([, p]) => p.member.says);
      if (!speaker) {
        el2.style.display = 'none';
        return;
      }
      const [a, p] = speaker;
      const words = p.member.says.split(/\s+/);
      const shown = words.slice(0, Math.max(1, Math.ceil((stepT - 0.3) * WPS))).join(' ');
      const at = new THREE.Vector3();
      (a.head || a.root).getWorldPosition(at);
      at.y += 0.35;
      at.project(camera);
      const r = renderer.domElement.getBoundingClientRect();
      el2.style.display = 'block';
      el2.style.left = `${((at.x + 1) / 2) * r.width}px`;
      el2.style.top = `${((1 - at.y) / 2) * r.height}px`;
      el2.textContent = `${p.member.name}: ${shown}`;
    }

    function resize() {
      const r = el!.getBoundingClientRect();
      renderer.setSize(r.width, r.height, false);
      camera.aspect = r.width / Math.max(1, r.height);
      camera.updateProjectionMatrix();
    }
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    function loop() {
      raf = requestAnimationFrame(loop);
      const dt = paused ? 0 : Math.min(clock.getDelta(), 0.1);
      if (paused) clock.getDelta();
      for (const a of Array.from(actors.values())) {
        a.mixer.update(dt);
        blendPose(a, dt);
      }
      if (cur) {
        stepT += dt;
        if (stepT >= cur.duration && at + 1 < history.length) play(at + 1);
      }
      const now = performance.now();
      if (now - shownAt > 150) {
        shownAt = now;
        show();
      }
      const c = new THREE.Vector3();
      let k = 0;
      for (const a of Array.from(actors.values()))
        if (a.hips) {
          const p = new THREE.Vector3();
          a.hips.getWorldPosition(p);
          c.add(p);
          k++;
        }
      if (k) {
        c.multiplyScalar(1 / k);
        c.y = 0.95;
        const d = c.clone().sub(follow).multiplyScalar(Math.min(1, dt * 3));
        follow.add(d);
        controls.target.add(d);
        camera.position.add(d);
        sun.position.set(follow.x + 4, 9, follow.z + 3);
        sun.target.position.copy(follow);
      }
      controls.update();
      renderer.render(scene, camera);
      bubble();
    }
    loop();

    engine.current = {
      seek: (i) => {
        // going back: the bodies stand where that step starts, and later steps follow again
        if (history[i]) play(i);
      },
      toggle: () => {
        paused = !paused;
        show();
      },
      seen: new Set(),
      add: (s) => {
        setLoading(true);
        chain = chain
          .then(() => prepare(s))
          .catch(() => setFailed('A clip could not be loaded.'))
          .finally(() => setLoading(false));
      },
    };

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      engine.current = null;
    };
    // the models are fixed for the page's life
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // New steps, in order, each once.
  useEffect(() => {
    const e = engine.current;
    if (!e) return;
    for (const s of [...steps].sort((x, y) => x.n - y.n)) {
      if (e.seen.has(s.n)) continue;
      e.seen.add(s.n);
      e.add(s);
    }
  }, [steps]);

  return (
    <div ref={box} className={`relative overflow-hidden rounded-xl border border-border bg-card/60 ${className}`}>
      <div
        ref={bubbleRef}
        className="pointer-events-none absolute z-10 hidden max-w-[16rem] -translate-x-1/2 -translate-y-full rounded-xl border border-border bg-background/95 px-3 py-1.5 text-sm shadow-sm"
        aria-live="polite"
      />
      {(loading || failed) && (
        <p className={`absolute left-3 text-xs text-muted-foreground ${tl.steps.length ? 'bottom-14' : 'bottom-2'}`}>{failed || 'Loading the clips…'}</p>
      )}
      {tl.steps.length > 0 && (
        <div className="absolute inset-x-2 bottom-2 z-10 flex items-center gap-2 rounded-xl border border-border bg-background/85 px-2 py-1.5 shadow-sm backdrop-blur">
          <button
            type="button"
            onClick={() => engine.current?.toggle()}
            aria-label={tl.paused ? 'Play' : 'Pause'}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-foreground text-background hover:bg-foreground/90"
          >
            {tl.paused ? <Play className="h-4 w-4" aria-hidden /> : <Pause className="h-4 w-4" aria-hidden />}
          </button>
          <div className="flex h-8 min-w-0 flex-1 items-stretch gap-1" role="group" aria-label="Steps">
            {tl.steps.map((st, i) => {
              const said = st.members.map((m) => `${m.name}: ${m.says ? `"${m.says}"` : 'listens'}${m.act ? ` (${m.act})` : ''}`).join('\n');
              const on = i === tl.at;
              return (
                <button
                  key={st.n}
                  type="button"
                  onClick={() => engine.current?.seek(i)}
                  title={`Step ${st.n}\n${said}`}
                  aria-label={`Step ${st.n}: ${said}`}
                  aria-current={on ? 'step' : undefined}
                  style={{ flexGrow: st.duration, flexBasis: 0 }}
                  className={`relative min-w-[1.25rem] overflow-hidden rounded-md text-left transition-colors ${on ? 'bg-foreground/15' : i < tl.at ? 'bg-foreground/[0.07] hover:bg-foreground/10' : 'bg-foreground/[0.04] hover:bg-foreground/10'}`}
                >
                  {on && <span className="absolute inset-y-0 left-0 bg-foreground/20" style={{ width: `${tl.frac * 100}%` }} aria-hidden />}
                  <span className="relative block truncate px-1.5 text-[11px] leading-8 text-muted-foreground">{st.n}</span>
                </button>
              );
            })}
          </div>
          <span className="shrink-0 font-mono text-[11px] text-muted-foreground">
            {tl.at + 1}/{tl.steps.length}
          </span>
        </div>
      )}
    </div>
  );
}
