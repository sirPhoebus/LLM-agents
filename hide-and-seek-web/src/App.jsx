import React from 'react';
import { Scoreboard } from './components/Scoreboard';
import { Maze } from './components/Maze';
import { useGameState, PHASES } from './hooks/useGameState';
import { Play, RotateCcw, Info } from 'lucide-react';

function App() {
  const { phase, timeLeft, maze, agents, score, clues, toggleWall, startGame, resetGame } = useGameState();

  return (
    <div className="App">
      <header style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <h1 style={{
          fontSize: '3.5rem',
          fontWeight: '800',
          background: 'linear-gradient(to right, #38bdf8, #818cf8)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: '0.5rem'
        }}>
          Hide & Seek
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '1.2rem' }}>
          Differentiable Multi-Agent Symbolic Simulation
        </p>
      </header>

      <Scoreboard phase={phase} timeLeft={timeLeft} score={score} />

      <main style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2rem' }}>
        <Maze maze={maze} agents={agents} clues={clues} toggleWall={toggleWall} phase={phase} />

        <div style={{ display: 'flex', gap: '1rem' }}>
          {phase === PHASES.SETUP || phase === PHASES.RESULTS ? (
            <button className="button-primary" onClick={startGame} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Play size={20} />
              {phase === PHASES.RESULTS ? 'Play Again' : 'Start Simulation'}
            </button>
          ) : (
            <button className="button-secondary" onClick={resetGame} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <RotateCcw size={20} />
              Reset
            </button>
          )}
        </div>

        {phase === PHASES.SETUP && (
          <div className="glass-card" style={{ padding: '1rem 2rem', display: 'flex', alignItems: 'center', gap: '1rem', color: 'var(--text-muted)' }}>
            <Info size={20} />
            <p>Click on the grid cells to place or remove walls before starting.</p>
          </div>
        )}
      </main>

      <footer style={{ marginTop: '4rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
        <p>Built with React & Framer Motion for Advanced Agentic Coding Showcase</p>
      </footer>
    </div>
  );
}

export default App;
