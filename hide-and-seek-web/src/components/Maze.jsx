import React from 'react';
import { GAME_CONFIG } from '../config/gameConfig';
import { Agent } from './Agent';

export const Maze = ({ maze, agents, clues, toggleWall, phase }) => {
    const size = GAME_CONFIG.GRID_SIZE * GAME_CONFIG.CELL_SIZE;

    return (
        <div className="glass-card" style={{
            padding: '20px',
            position: 'relative',
            width: 'fit-content',
            margin: '0 auto',
            background: 'rgba(15, 23, 42, 0.8)',
        }}>
            <div style={{
                position: 'relative',
                width: size,
                height: size,
                display: 'grid',
                gridTemplateColumns: `repeat(${GAME_CONFIG.GRID_SIZE}, ${GAME_CONFIG.CELL_SIZE}px)`,
                gridTemplateRows: `repeat(${GAME_CONFIG.GRID_SIZE}, ${GAME_CONFIG.CELL_SIZE}px)`,
                border: '2px solid var(--glass-border)',
                borderRadius: '4px',
                overflow: 'hidden'
            }}>
                {maze.map((row, y) =>
                    row.map((cell, x) => (
                        <div
                            key={`${x}-${y}`}
                            onClick={() => toggleWall(x, y)}
                            style={{
                                width: GAME_CONFIG.CELL_SIZE,
                                height: GAME_CONFIG.CELL_SIZE,
                                backgroundColor: cell === 1 ? GAME_CONFIG.COLORS.WALL : GAME_CONFIG.COLORS.FLOOR,
                                border: '0.5px solid rgba(255,255,255,0.05)',
                                cursor: phase === 'SETUP' ? 'pointer' : 'default',
                                transition: 'background-color 0.2s',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            }}
                        >
                            {x === Math.floor(GAME_CONFIG.GRID_SIZE / 2) && y === Math.floor(GAME_CONFIG.GRID_SIZE / 2) && (
                                <div style={{
                                    width: '30%',
                                    height: '30%',
                                    borderRadius: '50%',
                                    background: 'rgba(255,255,255,0.1)',
                                    border: '1px dashed rgba(255,255,255,0.3)'
                                }} />
                            )}
                        </div>
                    ))
                )}

                {clues && clues.map((clue, i) => (
                    <div
                        key={`clue-${i}`}
                        style={{
                            position: 'absolute',
                            left: clue.x * GAME_CONFIG.CELL_SIZE + GAME_CONFIG.CELL_SIZE / 2 - 2,
                            top: clue.y * GAME_CONFIG.CELL_SIZE + GAME_CONFIG.CELL_SIZE / 2 - 2,
                            width: 4,
                            height: 4,
                            borderRadius: '50%',
                            background: 'rgba(74, 222, 128, 0.4)',
                            boxShadow: '0 0 4px rgba(74, 222, 128, 0.8)',
                            pointerEvents: 'none'
                        }}
                    />
                ))}

                {agents.map(agent => (
                    <Agent key={agent.id} agent={agent} />
                ))}
            </div>
        </div>
    );
};
