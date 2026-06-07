import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import { Activity, Brain, Clock, AlertTriangle, CheckCircle2, Send, ActivitySquare, Users } from 'lucide-react';
import './index.css';

// --- Mock Data Generation ---
const generateTimelineData = () => {
  const data = [];
  let baseRisk = 10;
  for (let i = 0; i < 24; i++) {
    const time = new Date();
    time.setHours(time.getHours() - (23 - i));
    
    // Create a spike around 4 hours ago
    if (i === 20) baseRisk = 65;
    else if (i === 21) baseRisk = 85;
    else if (i === 22) baseRisk = 40;
    else baseRisk = Math.max(5, baseRisk * 0.8 + Math.random() * 10);
    
    data.push({
      time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      risk: Math.round(baseRisk)
    });
  }
  return data;
};

const generateChannelData = () => {
  const regions = ['Frontal (Fp1-Fp2)', 'Temporal (T3-T4)', 'Central (C3-C4)', 'Parietal (P3-P4)', 'Occipital (O1-O2)'];
  return regions.map(name => ({
    name,
    activity: Math.floor(Math.random() * 60) + 10,
    variance: Math.floor(Math.random() * 30)
  }));
};

function App() {
  const [activePatient, setActivePatient] = useState("CHB-01");
  const [messages, setMessages] = useState([
    { id: 1, text: `System active. Analyzing ${activePatient} 24-hour telemetry.`, sender: "bot" }
  ]);
  const [inputText, setInputText] = useState("");
  const [riskScore, setRiskScore] = useState(12.4);
  const [timelineData, setTimelineData] = useState(generateTimelineData());
  const [channelData, setChannelData] = useState(generateChannelData());

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
        const res = await fetch('http://localhost:8000/api/predict');
        const data = await res.json();
        if (data.status === 'success') {
          // Convert probability to percentage
          const riskPercentage = Number((data.predict_proba * 100).toFixed(1));
          setRiskScore(riskPercentage);
          setChannelData(generateChannelData()); // Future: connect to real channel data
        }
      } catch (err) {
        console.error("Failed to connect to backend", err);
      }
    };

    fetchData(); // Fetch immediately on load
    const interval = setInterval(fetchData, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    
    const newMsg = { id: Date.now(), text: inputText, sender: "user" };
    setMessages(prev => [...prev, newMsg]);
    setInputText("");
    
    setTimeout(() => {
      setMessages(prev => [...prev, {
        id: Date.now(),
        text: "Analyzing... The spike at 14:00 was characterized by high-frequency gamma bursts in the T3-T4 channels. This local instability disrupts the neural rhythm, which could rapidly spread and cause a focal impaired awareness seizure.",
        sender: "bot"
      }]);
    }, 1200);
  };

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
              setTimelineData(generateTimelineData());
              setChannelData(generateChannelData());
              setMessages([{ 
                id: Date.now(), 
                text: `Switched to monitoring ${p.id}. Live telemetry connected. LFM Initializing clinical context...`, 
                sender: "bot" 
              }]);
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
          <h3 className="card-title" style={{marginBottom: '20px'}}>Regional Spatial Activity (GNN Extraction)</h3>
          <div style={{flex: 1, width: '100%'}}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={channelData} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="rgba(0,0,0,0.05)" />
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{fill: 'var(--text-primary)', fontSize: 13, fontWeight: 500}} />
                <Tooltip cursor={{fill: 'rgba(0,0,0,0.02)'}} contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)'}} />
                <Bar dataKey="activity" fill="var(--accent-color)" radius={[0, 10, 10, 0]} barSize={20} />
                <Bar dataKey="variance" fill="rgba(66, 153, 225, 0.3)" radius={[0, 10, 10, 0]} barSize={20} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Chatbot Sidebar */}
      <div className="sidebar glass-panel chat-container">
        <div className="chat-header">
          <h3 style={{display: 'flex', alignItems: 'center', gap: '10px'}}>
            <div style={{width: '32px', height: '32px', borderRadius: '50%', background: 'linear-gradient(135deg, var(--accent-color), #805ad5)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontSize: '14px', fontWeight: 'bold'}}>L</div>
            Liquid LFM Assistant
          </h3>
          <p style={{fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '5px'}}>Clinical AI Explainer • Connected</p>
        </div>
        
        <div className="chat-messages">
          {messages.map(msg => (
            <div key={msg.id} className={`message ${msg.sender}`}>
              {msg.text}
            </div>
          ))}
        </div>

        <form className="chat-input" onSubmit={handleSend}>
          <input 
            type="text" 
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Ask about patient anomalies..." 
          />
          <button type="submit" style={{padding: '0 15px', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
            <Send size={18} />
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
