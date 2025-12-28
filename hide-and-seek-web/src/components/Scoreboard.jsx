import React from 'react';
import { Timer, Trophy, Eye, EyeOff } from 'lucide-react';
import { PHASES } from '../hooks/useGameState';

export const Scoreboard = ({ phase, timeLeft, score }) => {
    const getPhaseText = () => {
        switch (phase) {
            case PHASES.SETUP: return 'Design the Maze';
            case PHASES.HIDING: return 'Hiders are Hiding...';
            case PHASES.SEEKING: return 'Seekers are Hunting!';
            case PHASES.RESULTS: return 'Match Over';
            default: return '';
        }
    };

    return (
        <div className="glass-card" style={{
            width: '100%',
            padding: '1.5rem',
            marginBottom: '2rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{
                    background: 'rgba(255,255,255,0.1)',
                    padding: '0.5rem 1rem',
                    borderRadius: '0.5rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem'
                }}>
                    <Timer size={20} color="#38bdf8" />
                    <span style={{ fontSize: '1.25rem', fontWeight: 'bold', width: '3rem' }}>
                        {timeLeft}s
                    </span>
                </div>
                <h2 style={{ fontSize: '1.5rem', margin: 0, color: 'var(--accent-primary)' }}>
                    {getPhaseText()}
                </h2>
            </div>

            <div style={{ display: 'flex', gap: '2rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <EyeOff size={20} color="#4ade80" />
                    <span style={{ fontSize: '1.2rem' }}>Hidden: {4 - score.seekers}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Eye size={20} color="#f87171" />
                    <span style={{ fontSize: '1.2rem' }}>Found: {score.seekers}</span>
                </div>
            </div>
        </div>
    );
};
