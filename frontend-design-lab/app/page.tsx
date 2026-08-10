"use client";

import { useState } from "react";

type Page = "live" | "teach" | "library" | "motion" | "settings";

const navigation: { id: Page; label: string; hint: string }[] = [
  { id: "live", label: "Live", hint: "01" },
  { id: "teach", label: "Teach", hint: "02" },
  { id: "library", label: "Library", hint: "03" },
  { id: "motion", label: "Motion", hint: "04" },
  { id: "settings", label: "Settings", hint: "05" },
];

function CameraStage({ live }: { live: boolean }) {
  return (
    <section className={`camera-stage ${live ? "is-live" : ""}`} aria-label="Camera preview">
      <div className="camera-grid" aria-hidden="true" />
      <div className="camera-corners" aria-hidden="true" />
      <div className="camera-meta"><span>{live ? "TRACKING / RIGHT HAND" : "HANDSPACE / READY"}</span><span>CAMERA 01</span></div>
      {live ? <div className="landmark-hand" aria-hidden="true"><i /><i /><i /><i /><i /><i /><i /><i /><i /></div> : <div className="camera-prompt"><strong>Place your hand inside the frame</strong><span>Hold a gesture steady to begin recognition</span></div>}
      <div className="camera-footer"><span>LANDMARKS {live ? "ON" : "OFF"}</span><span>960 × 540&nbsp;&nbsp;·&nbsp;&nbsp;30 FPS</span></div>
    </section>
  );
}

export default function Home() {
  const [page, setPage] = useState<Page>("live");
  const [tracking, setTracking] = useState(false);
  const [result, setResult] = useState("UNKNOWN");
  const [samples, setSamples] = useState(0);
  const [motionRecording, setMotionRecording] = useState(false);
  const [notice, setNotice] = useState("Session is local. Camera frames are never stored.");

  const simulateRecognition = () => {
    const next = tracking ? "VICTORY" : "UNKNOWN";
    setResult(next);
    setNotice(next === "UNKNOWN" ? "Show a learned gesture inside the guide." : "Stable output confirmed across three frames.");
  };

  return (
    <main className="app-shell">
      <a className="skip-link" href="#workspace">Skip to workspace</a>
      <header className="app-header">
        <button className="brand" onClick={() => setPage("live")} aria-label="Open live recognition"><span className="brand-mark" aria-hidden="true" /><span>ADAPTIVE <em>/</em> GESTURE</span></button>
        <div className="header-status"><span className="signal-dot" aria-hidden="true" /> LOCAL SESSION <span className="header-separator">/</span> CAMERA 01</div>
      </header>

      <div className="app-frame">
        <nav className="side-nav" aria-label="Primary navigation">
          {navigation.map((item) => <button key={item.id} className={page === item.id ? "nav-item active" : "nav-item"} onClick={() => setPage(item.id)}><span>{item.hint}</span>{item.label}</button>)}
        </nav>

        <section className="workspace" id="workspace">
          {page === "live" && <div className="live-view">
            <div className="stage-column"><div className="page-intro"><p>LIVE RECOGNITION</p><h1>Read the gesture.<br />Keep the signal clear.</h1></div><CameraStage live={tracking} /><div className="stage-controls"><button className="primary-action" onClick={() => { setTracking(!tracking); setResult("UNKNOWN"); setNotice(!tracking ? "Camera tracking enabled. Hold a learned gesture in frame." : "Camera tracking paused."); }}>{tracking ? "Pause tracking" : "Start tracking"}</button><button className="text-action" onClick={simulateRecognition}>Check signal</button><span>{tracking ? "Tracking right hand" : "Camera ready"}</span></div></div>
            <aside className="signal-panel" aria-live="polite"><div><p className="eyebrow">STABLE OUTPUT</p><div className={result === "UNKNOWN" ? "output-word muted" : "output-word"}>{result}</div><p className="output-context">{result === "UNKNOWN" ? "Waiting for a known gesture" : "Static gesture · 0.87 confidence"}</p><div className="confidence-track"><span style={{ width: result === "UNKNOWN" ? "12%" : "87%" }} /></div><div className="signal-number">CONFIDENCE&nbsp;&nbsp;{result === "UNKNOWN" ? "—" : "0.87"}</div></div><div className="feedback"><p className="eyebrow">FEEDBACK LOOP</p><p>Help the local model refine its memory.</p><div className="feedback-buttons"><button onClick={() => setNotice("Positive feedback stored for this session.")} disabled={result === "UNKNOWN"}>Correct</button><button onClick={() => setPage("teach")} disabled={result === "UNKNOWN"}>Correct it</button></div></div><div className="session-line"><span>03</span><p>learned gestures<br /><small>{notice}</small></p></div></aside>
          </div>}

          {page === "teach" && <div className="teach-view"><div className="page-intro"><p>PERSONALIZED LEARNING</p><h1>Teach one gesture<br />at a time.</h1></div><div className="teach-layout"><section className="teach-guide"><div className="teach-index">0{Math.min(samples + 1, 3)}</div><div><p className="eyebrow">THREE CLEAN EXAMPLES</p><h2>Victory</h2><p>Keep your hand still. The system selects the clearest landmark sample from each capture.</p></div><div className="capture-row">{[1, 2, 3].map((item) => <span key={item} className={samples >= item ? "capture done" : "capture"}>{samples >= item ? "Captured" : `Sample ${item}`}</span>)}</div><button className="primary-action full" onClick={() => { const next = Math.min(samples + 1, 3); setSamples(next); setNotice(next === 3 ? "Gesture saved to the local library." : `Sample ${next} accepted. Capture the next example.`); }}> {samples >= 3 ? "Gesture saved" : "Capture sample"}</button></section><section className="teach-preview"><CameraStage live={samples > 0} /></section></div></div>}

          {page === "library" && <div className="library-view"><div className="library-head"><div className="page-intro"><p>LOCAL LIBRARY</p><h1>Known by you.</h1></div><button className="primary-action" onClick={() => setPage("teach")}>Teach a gesture</button></div><div className="gesture-list">{[{ name: "Victory", type: "STATIC", detail: "3 prototypes" }, { name: "Wave", type: "MOTION", detail: "3 trajectories" }, { name: "Chill", type: "STATIC", detail: "2 prototypes" }].map((gesture, index) => <article className="gesture-row" key={gesture.name}><span className="gesture-count">0{index + 1}</span><h2>{gesture.name}</h2><p>{gesture.type}</p><span>{gesture.detail}</span><button onClick={() => setNotice(`${gesture.name} is selected for refinement.`)}>Refine</button></article>)}</div></div>}

          {page === "motion" && <div className="motion-view"><div className="motion-copy"><p className="eyebrow">DYNAMIC RECOGNITION</p><h1>Follow the movement,<br />not the frame.</h1><p>Record a complete gesture. Matching begins only after the movement resolves.</p><button className="primary-action" onClick={() => { setMotionRecording(!motionRecording); setNotice(!motionRecording ? "Recording movement. Complete the gesture, then finish." : "Motion example stored locally for review."); }}>{motionRecording ? "Finish demonstration" : "Start demonstration"}</button><div className="motion-readout"><span className={motionRecording ? "signal-dot" : "signal-dot idle"} /> {motionRecording ? "RECORDING / 01.8S" : "IDLE / READY"}</div></div><div className="trajectory" aria-label="Motion trajectory visualization"><span className="axis vertical" /><span className="axis horizontal" /><span className="ring ring-a" /><span className="ring ring-b" /><span className="ring ring-c" /><span className={motionRecording ? "path active" : "path"} /><span className="path-start" /><span className="path-end" /><p>LIVE TRAJECTORY / RIGHT HAND</p><div className="trajectory-footer"><span>START</span><b className={motionRecording ? "recording" : ""} /><span>RELEASE TO MATCH</span></div></div></div>}

          {page === "settings" && <div className="settings-view"><div className="page-intro"><p>SYSTEM</p><h1>Private by default.</h1></div><div className="setting-list"><div><p>CAMERA STORAGE</p><strong>Frames remain in memory only</strong></div><div><p>GESTURE MEMORY</p><strong>Local landmark data</strong></div><div><p>STABILIZATION</p><strong>3-frame confirmation</strong></div></div></div>}
        </section>
      </div>
      <footer>ADAPTIVE GESTURE LAB <span>FRONTEND PROTOTYPE / NO MODEL CONNECTED</span></footer>
    </main>
  );
}
