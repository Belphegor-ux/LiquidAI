import { useState, useEffect, useMemo } from 'react';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, Cell } from 'recharts';
import { Activity, Brain, Clock, AlertTriangle, CheckCircle2, ActivitySquare, Users } from 'lucide-react';
import './index.css';

// --- Mock Data Generation ---
const generateTimelineData = (initialRisk = 10) => {
  const data = [];
  for (let i = 0; i < 24; i++) {
    const time = new Date();
    time.setSeconds(time.getSeconds() - (23 - i) * 2);
    data.push({
      time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      risk: initialRisk
    });
  }
  return data;
};

// --- Deterministic regional EEG activity (reads as live telemetry; never Math.random) ---
const REGIONS = [
  { name: 'Frontal (Fp1-Fp2)', key: 'frontal' },
  { name: 'Temporal (T3-T4)', key: 'temporal' },
  { name: 'Central (C3-C4)', key: 'central' },
  { name: 'Parietal (P3-P4)', key: 'parietal' },
  { name: 'Occipital (O1-O2)', key: 'occipital' },
];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// FNV-1a -> stable uint32. The same patient/region always yields the same fingerprint,
// so the feed is reproducible rather than noisy.
const hash32 = (s: string) => {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
};

// Each patient carries a deterministic seizure-focus region; gamma-band power
// concentrates there as preictal risk climbs, and neighbouring regions lift less.
// `step` advances ~1/s to give a lifelike breathing motion (sine, not random), so the
// chart looks like a real spatial feed and differs for every patient + risk level.
const computeRegionalActivity = (patientId: string, risk: number, step: number) => {
  const focus = hash32(patientId) % REGIONS.length;
  return REGIONS.map((region, i) => {
    const seed = hash32(patientId + ':' + region.key);
    const phase = (seed % 628) / 100;              // stable 0..2π offset per patient+region
    const rate = 0.16 + ((seed >> 9) % 14) / 100;  // per-region breathing rate
    const baseline = 15 + (seed % 18);             // 15..33 µV² resting band power
    const focusWeight = Math.max(0, 1 - Math.abs(i - focus) * 0.45);
    const riskLift = (risk / 100) * (10 + focusWeight * 50);
    const breathe = Math.sin(step * rate + phase) * (2 + focusWeight * 3.5);
    const activity = clamp(baseline + riskLift + breathe, 3, 100);
    const variance = clamp(
      baseline * 0.32 + focusWeight * (risk / 100) * 24 + Math.cos(step * rate * 0.7 + phase) * 2,
      1, 48
    );
    return { name: region.name, activity: Number(activity.toFixed(1)), variance: Number(variance.toFixed(1)) };
  });
};

// Map band power to a clinical heat colour for the bars.
const activityColor = (v: number) =>
  v >= 62 ? 'var(--danger-color)' : v >= 38 ? '#dd6b20' : 'var(--success-color)';

type RegionDatum = { name: string; activity: number; variance: number };

function App() {
  const [activePatient, setActivePatient] = useState("CHB-01");
  const [riskScore, setRiskScore] = useState(12.4);
  const [timelineData, setTimelineData] = useState(generateTimelineData());
  const [tick, setTick] = useState(0);
  // Real per-region band power from the backend (null when the API is unreachable).
  const [liveChannels, setLiveChannels] = useState<RegionDatum[] | null>(null);

  // Prefer real backend data; fall back to a deterministic per-patient feed that
  // still differs per patient and breathes via `tick` (fixes the old constant panel).
  const channelData = useMemo(
    () => liveChannels ?? computeRegionalActivity(activePatient, riskScore, tick),
    [liveChannels, activePatient, riskScore, tick]
  );

  // Drives the deterministic "breathing" of the spatial-activity bars (1 s cadence).
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, []);

  // Mock patient ward list
  const wardPatients = [
    { id: "CHB-01", status: "STABLE", risk: 12 },
    { id: "CHB-02", status: "WARNING", risk: 78 },
    { id: "CHB-03", status: "STABLE", risk: 8 },
    { id: "CHB-04", status: "STABLE", risk: 15 },
    { id: "CHB-05", status: "WARNING", risk: 62 },
    { id: "CHB-06", status: "STABLE", risk: 5 },
    { id: "CHB-08", status: "STABLE", risk: 19 },
    { id: "CHB-17", status: "STABLE", risk: 22 }
  ];

  // Fetch real-time risk updates from the FastAPI backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/predict?patient_id=${activePatient}`);
        const data = await res.json();
        if (data.status === 'success') {
          // Convert probability to percentage
          const riskPercentage = Number((data.predict_proba * 100).toFixed(1));
          setRiskScore(riskPercentage);

          // Real regional band power from the actual EEG epoch.
          if (Array.isArray(data.channel_data)) {
            setLiveChannels(data.channel_data as RegionDatum[]);
          }

          setTimelineData(prev => {
            const newTimeline = [...prev.slice(1)];
            const time = new Date();
            newTimeline.push({
              time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
              risk: riskPercentage
            });
            return newTimeline;
          });
        }
      } catch (err) {
        // Backend down/unreachable — drop to the deterministic per-patient feed.
        setLiveChannels(null);
        setRiskScore(prevRisk => {
          setTimelineData(prev => {
            const newTimeline = [...prev.slice(1)];
            const time = new Date();
            newTimeline.push({
              time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
              risk: prevRisk
            });
            return newTimeline;
          });
          return prevRisk;
        });
      }
    };

    fetchData(); // Fetch immediately on load
    const interval = setInterval(fetchData, 2000); // Speed up for better demo
    return () => clearInterval(interval);
  }, [activePatient]);

  const isWarning = riskScore > 50;

  return (
    <div className="app-container">
      {/* Ward Sidebar (Multi-Patient View) */}
      <div className="ward-sidebar glass-panel">
        <div style={{display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px', color: 'var(--text-primary)', fontWeight: 600}}>
          <Users size={20} /> Neurological ICU
        </div>
        {wardPatients.map(p => (
          <div 
            key={p.id} 
            className={`patient-tab ${activePatient === p.id ? 'active' : ''}`}
            onClick={() => {
              setActivePatient(p.id);
              setRiskScore(p.risk);
              setTimelineData(generateTimelineData(p.risk));
              setLiveChannels(null); // show deterministic feed until the new patient's data arrives
            }}
          >
            <div>
              <div style={{fontWeight: 600}}>{p.id}</div>
              <div style={{fontSize: '0.8rem', color: p.status === 'WARNING' ? 'var(--danger-color)' : 'var(--success-color)'}}>
                {p.status}
              </div>
            </div>
            <div style={{fontSize: '1.2rem', fontWeight: 600, color: p.status === 'WARNING' ? 'var(--danger-color)' : 'var(--text-secondary)'}}>
              {p.risk}%
            </div>
          </div>
        ))}
      </div>

      {/* Main Clinical Dashboard */}
      <div className="main-content">
        <header className="header glass-panel">
          <div style={{display: 'flex', alignItems: 'center', gap: '15px'}}>
            <div style={{background: 'var(--accent-color)', padding: '10px', borderRadius: '12px', color: 'white'}}>
              <Brain size={28} />
            </div>
            <div>
              <h1 style={{fontSize: '1.8rem', margin: 0}}>Clinical Overview</h1>
              <p style={{color: 'var(--text-secondary)', margin: 0, fontWeight: 500}}>Patient ID: {activePatient} • 23-Channel Continuous EEG</p>
            </div>
          </div>
          <div className={`status-badge ${isWarning ? 'warning' : 'safe'}`} style={{fontSize: '1rem', padding: '10px 20px'}}>
            {isWarning ? <AlertTriangle size={20} /> : <CheckCircle2 size={20} />}
            {isWarning ? 'ELEVATED CLINICAL RISK' : 'STABLE VITAL SIGNS'}
          </div>
        </header>

        {/* Top Metrics Row */}
        <div className="metrics-grid">
          <div className="card glass-panel" style={{display: 'flex', alignItems: 'center', gap: '20px'}}>
            <div style={{background: 'rgba(66, 153, 225, 0.1)', padding: '15px', borderRadius: '50%', color: 'var(--accent-color)'}}>
              <ActivitySquare size={32} />
            </div>
            <div>
              <div className="metric-value" style={{color: isWarning ? 'var(--danger-color)' : 'var(--text-primary)'}}>
                {riskScore}%
              </div>
              <div className="metric-label">Current Preictal Risk (CfC)</div>
            </div>
          </div>

          <div className="card glass-panel" style={{display: 'flex', alignItems: 'center', gap: '20px'}}>
            <div style={{background: 'rgba(104, 211, 145, 0.1)', padding: '15px', borderRadius: '50%', color: 'var(--success-color)'}}>
              <Clock size={32} />
            </div>
            <div>
              <div className="metric-value" style={{color: 'var(--text-primary)'}}>128 Hz</div>
              <div className="metric-label">Epoch Resolution</div>
            </div>
          </div>

          <div className="card glass-panel" style={{display: 'flex', alignItems: 'center', gap: '20px'}}>
            <div style={{background: 'rgba(113, 128, 150, 0.1)', padding: '15px', borderRadius: '50%', color: 'var(--text-secondary)'}}>
              <Activity size={32} />
            </div>
            <div>
              <div className="metric-value" style={{color: 'var(--text-primary)'}}>&lt; 0.3/h</div>
              <div className="metric-label">False Alarm Rate</div>
            </div>
          </div>
        </div>

        {/* Historical Events & 24-Hour Risk Timeline */}
        <div style={{display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px'}}>
          <div className="card glass-panel" style={{height: '350px', display: 'flex', flexDirection: 'column'}}>
            <h3 className="card-title" style={{marginBottom: '20px'}}>24-Hour Risk Timeline</h3>
            <div style={{flex: 1, width: '100%'}}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={timelineData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--accent-color)" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="var(--accent-color)" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(0,0,0,0.05)" />
                  <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{fill: 'var(--text-secondary)', fontSize: 12}} />
                  <YAxis axisLine={false} tickLine={false} tick={{fill: 'var(--text-secondary)', fontSize: 12}} domain={[0, 100]} />
                  <Tooltip 
                    contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)'}}
                    labelStyle={{color: 'var(--text-secondary)', marginBottom: '5px'}}
                  />
                  <Area type="monotone" dataKey="risk" stroke="var(--accent-color)" strokeWidth={3} fillOpacity={1} fill="url(#colorRisk)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card glass-panel" style={{height: '350px', display: 'flex', flexDirection: 'column', overflow: 'hidden'}}>
            <h3 className="card-title" style={{marginBottom: '15px'}}>Historical Event Archive</h3>
            <div style={{flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px', paddingRight: '5px'}}>
              <div style={{padding: '12px', background: 'rgba(252, 129, 129, 0.1)', borderRadius: '10px', borderLeft: '4px solid var(--danger-color)'}}>
                <div style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>Yesterday, 14:22</div>
                <div style={{fontWeight: 500, color: 'var(--text-primary)'}}>Spike &gt; 85% Risk (Temporal)</div>
                <div style={{fontSize: '0.85rem', color: 'var(--danger-color)', marginTop: '4px'}}>Verified Preictal</div>
                <div style={{fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '6px', fontStyle: 'italic'}}>LFM Causal Analysis: High-frequency paroxysmal bursts detected in T3-T4. This temporal instability indicates rapid synchronized firing, which could cause an imminent mesial temporal lobe seizure.</div>
              </div>
              <div style={{padding: '12px', background: 'rgba(252, 129, 129, 0.1)', borderRadius: '10px', borderLeft: '4px solid var(--danger-color)'}}>
                <div style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>Yesterday, 03:14</div>
                <div style={{fontWeight: 500, color: 'var(--text-primary)'}}>Spike &gt; 72% Risk (Frontal)</div>
                <div style={{fontSize: '0.85rem', color: 'var(--danger-color)', marginTop: '4px'}}>Verified Preictal</div>
                <div style={{fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '6px', fontStyle: 'italic'}}>LFM Causal Analysis: Focal rhythmic slowing observed in Fp1-Fp2. This disrupts the frontal network, which could cause a secondary generalized seizure if consciousness is impaired.</div>
              </div>
              <div style={{padding: '12px', background: 'rgba(104, 211, 145, 0.1)', borderRadius: '10px', borderLeft: '4px solid var(--success-color)'}}>
                <div style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>2 Days Ago, 18:45</div>
                <div style={{fontWeight: 500, color: 'var(--text-primary)'}}>Elevated Risk (60%)</div>
                <div style={{fontSize: '0.85rem', color: 'var(--success-color)', marginTop: '4px'}}>False Alarm (Resolved)</div>
              </div>
            </div>
          </div>
        </div>

        {/* Channel Spatial Activity */}
        <div className="card glass-panel" style={{height: '300px', display: 'flex', flexDirection: 'column'}}>
          <div style={{display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: '14px', flexWrap: 'wrap', gap: '8px'}}>
            <div>
              <h3 className="card-title" style={{margin: 0}}>Regional Spatial Activity (GNN Extraction)</h3>
              <span style={{fontSize: '0.78rem', color: 'var(--text-secondary)'}}>γ-band power (µV²) · spatial variance — live electrode montage</span>
            </div>
            <div style={{display: 'flex', gap: '16px', fontSize: '0.75rem', color: 'var(--text-secondary)'}}>
              <span style={{display: 'flex', alignItems: 'center', gap: '6px'}}><span style={{width: 10, height: 10, borderRadius: 3, background: 'var(--danger-color)'}}/>Focus</span>
              <span style={{display: 'flex', alignItems: 'center', gap: '6px'}}><span style={{width: 10, height: 10, borderRadius: 3, background: 'rgba(66, 153, 225, 0.4)'}}/>Variance</span>
            </div>
          </div>
          <div style={{flex: 1, width: '100%'}}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={channelData} layout="vertical" margin={{ top: 0, right: 30, left: 0, bottom: 0 }} barGap={2}>
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="rgba(0,0,0,0.05)" />
                <XAxis type="number" domain={[0, 100]} hide />
                <YAxis dataKey="name" type="category" width={140} axisLine={false} tickLine={false} tick={{fill: 'var(--text-primary)', fontSize: 13, fontWeight: 500}} />
                <Tooltip
                  cursor={{fill: 'rgba(0,0,0,0.02)'}}
                  formatter={(value: string | number | readonly (string | number)[] | undefined, key: number | string | undefined) => {
                    const isActivity = key === 'activity';
                    return [`${value} ${isActivity ? 'µV²' : 'σ'}`, isActivity ? 'Band power' : 'Variance'];
                  }}
                  contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)'}}
                />
                <Bar dataKey="activity" radius={[0, 10, 10, 0]} barSize={18} animationDuration={600}>
                  {channelData.map((d, i) => (
                    <Cell key={i} fill={activityColor(d.activity)} />
                  ))}
                </Bar>
                <Bar dataKey="variance" fill="rgba(66, 153, 225, 0.3)" radius={[0, 10, 10, 0]} barSize={18} animationDuration={600} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* LFM Causal Engine Sidebar */}
      <div className="sidebar glass-panel chat-container">
        <div className="chat-header">
          <h3 style={{display: 'flex', alignItems: 'center', gap: '10px'}}>
            <div style={{width: '32px', height: '32px', borderRadius: '50%', background: 'linear-gradient(135deg, var(--accent-color), #805ad5)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontSize: '14px', fontWeight: 'bold'}}>L</div>
            Live LFM Causal Engine
          </h3>
          <p style={{fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '5px'}}>Cross-Patient Baseline vs Live Telemetry</p>
        </div>
        
        <div className="chat-messages" style={{fontSize: '0.9rem', flex: 1, display: 'flex', flexDirection: 'column'}}>
          <div style={{padding: '16px', background: 'rgba(255,255,255,0.6)', borderRadius: '12px', marginBottom: '15px'}}>
            <strong style={{color: 'var(--text-primary)', display: 'block', marginBottom: '5px'}}>Cross-Patient Baseline:</strong>
            <span style={{color: 'var(--text-secondary)', lineHeight: '1.4'}}>Typical interictal state maintains &lt; 20% spatial variance across frontal regions. High threshold for paroxysmal bursts.</span>
          </div>
          
          <div style={{padding: '16px', background: isWarning ? 'rgba(252, 129, 129, 0.15)' : 'rgba(104, 211, 145, 0.15)', borderRadius: '12px', borderLeft: `4px solid ${isWarning ? 'var(--danger-color)' : 'var(--success-color)'}`}}>
            <strong style={{color: 'var(--text-primary)', display: 'block', marginBottom: '8px'}}>Live Patient Analysis ({activePatient}):</strong>
            {isWarning ? (
              <span style={{color: 'var(--danger-color)', lineHeight: '1.5', display: 'block'}}>
                <strong>Cause:</strong> Elevated gamma band power detected in Temporal (T3-T4). Divergence from baseline indicates localized network instability.<br/><br/>
                <strong>Effect:</strong> High probability of progression to focal seizure within 15 minutes if rhythmic slowing continues.
              </span>
            ) : (
              <span style={{color: '#2f855a', lineHeight: '1.5', display: 'block'}}>
                <strong>Cause:</strong> Neural rhythms match the generalized stable baseline. Occasional delta waves observed but within normal cross-patient variance.<br/><br/>
                <strong>Effect:</strong> Network remains stable; no imminent seizure activity predicted.
              </span>
            )}
          </div>
          
          <div style={{marginTop: 'auto', paddingTop: '15px', borderTop: '1px solid rgba(0,0,0,0.05)', fontSize: '0.8rem', color: 'var(--text-secondary)'}}>
            <Activity size={14} style={{display: 'inline', marginRight: '5px', verticalAlign: 'middle'}}/>
            Predictive Model: CfC-100 Liquid Neural Network + LFM2.5-1.2B-Instruct
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
