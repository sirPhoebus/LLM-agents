import React from 'react';
import { motion } from 'framer-motion';
import { GAME_CONFIG } from '../config/gameConfig';

export const Agent = ({ agent }) => {
    const isHider = agent.type === 'HIDER';

    if (agent.found && isHider && agent.reward === 0) return null;

    return (
        <motion.div
            initial={false}
            animate={{
                x: agent.x * GAME_CONFIG.CELL_SIZE,
                y: agent.y * GAME_CONFIG.CELL_SIZE,
                scale: agent.found ? 0.8 : 1,
                opacity: agent.found && isHider ? 0.4 : 1
            }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            style={{
                position: 'absolute',
                width: GAME_CONFIG.CELL_SIZE,
                height: GAME_CONFIG.CELL_SIZE,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.5rem',
                zIndex: 10,
                pointerEvents: 'none'
            }}
        >
            <div style={{
                width: '80%',
                height: '80%',
                borderRadius: '50%',
                background: isHider ? agent.found ? '#94a3b8' : GAME_CONFIG.COLORS.HIDER : GAME_CONFIG.COLORS.SEEKER,
                boxShadow: agent.found ? 'none' : `0 0 15px ${isHider ? GAME_CONFIG.COLORS.HIDER : GAME_CONFIG.COLORS.SEEKER}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: agent.found ? '2px solid rgba(255,255,255,0.2)' : '2px solid white',
                transition: 'all 0.3s'
            }}>
                {isHider ? agent.found ? '😱' : '👻' : '🔍'}
            </div>
            {agent.reward !== 0 && (
                <motion.div
                    initial={{ y: 0, opacity: 0 }}
                    animate={{ y: -20, opacity: 1 }}
                    style={{
                        position: 'absolute',
                        top: -10,
                        fontWeight: 'bold',
                        color: agent.reward > 0 ? '#4ade80' : '#f87171',
                        textShadow: '0 0 5px black'
                    }}
                >
                    {agent.reward > 0 ? `+${agent.reward}` : agent.reward}
                </motion.div>
            )}
        </motion.div>
    );
};
