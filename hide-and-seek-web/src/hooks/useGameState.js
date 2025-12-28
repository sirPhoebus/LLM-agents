import { useState, useEffect, useCallback, useRef } from 'react';
import { GAME_CONFIG } from '../config/gameConfig';

export const PHASES = {
    SETUP: 'SETUP',
    HIDING: 'HIDING',
    SEEKING: 'SEEKING',
    RESULTS: 'RESULTS'
};

export const useGameState = () => {
    const [phase, setPhase] = useState(PHASES.SETUP);
    const [timeLeft, setTimeLeft] = useState(0);
    const [maze, setMaze] = useState(() => {
        const grid = Array(GAME_CONFIG.GRID_SIZE).fill().map(() => Array(GAME_CONFIG.GRID_SIZE).fill(0));
        return grid;
    });
    const [gameData, setGameData] = useState({
        agents: [],
        clues: [],
        score: { hiders: 0, seekers: 0 }
    });

    const timerRef = useRef(null);
    const loopRef = useRef(null);

    const center = Math.floor(GAME_CONFIG.GRID_SIZE / 2);

    const initAgents = useCallback(() => {
        const hiders = Array(GAME_CONFIG.HIDER_COUNT).fill().map((_, i) => ({
            id: `hider-${i}`,
            type: 'HIDER',
            x: center,
            y: center,
            found: false,
            reward: 0
        }));
        const seekers = Array(GAME_CONFIG.SEEKER_COUNT).fill().map((_, i) => ({
            id: `seeker-${i}`,
            type: 'SEEKER',
            x: center,
            y: center
        }));
        setGameData({
            agents: [...hiders, ...seekers],
            clues: [],
            score: { hiders: 0, seekers: 0 }
        });
    }, [center]);

    const toggleWall = (x, y) => {
        if (phase !== PHASES.SETUP) return;
        if (x === center && y === center) return; // Can't block start
        const newMaze = [...maze];
        newMaze[y][x] = newMaze[y][x] === 1 ? 0 : 1;
        setMaze(newMaze);
    };

    const startGame = () => {
        initAgents();
        setPhase(PHASES.HIDING);
        setTimeLeft(GAME_CONFIG.HIDING_DURATION);
    };

    const calculateRewards = useCallback(() => {
        setGameData(prev => ({
            ...prev,
            agents: prev.agents.map(a => {
                if (a.type === 'HIDER') {
                    const reward = a.found ? GAME_CONFIG.REWARD_FOUND : GAME_CONFIG.REWARD_WELL_HIDDEN;
                    return { ...a, reward };
                }
                return a;
            })
        }));
    }, []);

    const nextPhase = useCallback(() => {
        if (phase === PHASES.HIDING) {
            setPhase(PHASES.SEEKING);
            setTimeLeft(GAME_CONFIG.SEEKING_DURATION);
        } else if (phase === PHASES.SEEKING) {
            setPhase(PHASES.RESULTS);
            calculateRewards();
        }
    }, [phase, calculateRewards]);

    // Timer effect
    useEffect(() => {
        if (timeLeft > 0 && (phase === PHASES.HIDING || phase === PHASES.SEEKING)) {
            timerRef.current = setTimeout(() => setTimeLeft(prev => prev - 1), 1000);
        } else if (timeLeft === 0 && (phase === PHASES.HIDING || phase === PHASES.SEEKING)) {
            nextPhase();
        }
        return () => clearTimeout(timerRef.current);
    }, [timeLeft, phase, nextPhase]);

    // Movement Logic
    const moveAgents = useCallback(() => {
        setGameData(prev => {
            // Create a deep copy of agents to avoid mutating previous state
            const newAgents = prev.agents.map(a => ({ ...a }));
            const currentClues = [...prev.clues];
            const newClues = [];
            let newSeekerScore = prev.score.seekers;

            newAgents.forEach(agent => {
                if (agent.type === 'HIDER' && agent.found) return;
                if (agent.type === 'SEEKER' && phase === PHASES.HIDING) return;
                if (agent.type === 'HIDER' && phase !== PHASES.HIDING) return; // Hiders stop after hiding phase

                const moves = [
                    { x: 0, y: 1 }, { x: 0, y: -1 }, { x: 1, y: 0 }, { x: -1, y: 0 }
                ];

                // Shuffle moves to add non-deterministic behavior
                moves.sort(() => Math.random() - 0.5);

                let bestMove = { x: 0, y: 0 };

                if (agent.type === 'HIDER') {
                    // Hiders move away from center and seekers
                    let maxScore = -1000;
                    moves.forEach(m => {
                        const nx = agent.x + m.x;
                        const ny = agent.y + m.y;
                        if (nx >= 0 && nx < GAME_CONFIG.GRID_SIZE && ny >= 0 && ny < GAME_CONFIG.GRID_SIZE && maze[ny][nx] === 0) {
                            const distToCenter = Math.abs(nx - center) + Math.abs(ny - center);

                            // Penalty for being near corners (prevents getting stuck)
                            const cornerPenalty = (nx === 0 || nx === GAME_CONFIG.GRID_SIZE - 1 || ny === 0 || ny === GAME_CONFIG.GRID_SIZE - 1) ? -2 : 0;

                            // Score based on distance and variety
                            const score = distToCenter + cornerPenalty + (Math.random() * 2);

                            if (score > maxScore) {
                                maxScore = score;
                                bestMove = m;
                            }
                        }
                    });

                    // Leave a symbolic clue (breadcrumb)
                    if (bestMove.x !== 0 || bestMove.y !== 0) {
                        newClues.push({ x: agent.x, y: agent.y });
                    }
                } else {
                    // Seekers follow hiders if very close, otherwise follow clues (symbolic)
                    let minDist = 1000;
                    const target = newAgents.find(a => a.type === 'HIDER' && !a.found);
                    const smellRange = 4;

                    // Initialize seeker memory if not exists
                    if (!agent.memory) agent.memory = [];

                    if (target && (Math.abs(agent.x - target.x) + Math.abs(agent.y - target.y)) <= smellRange) {
                        // Direct tracking if very close
                        moves.forEach(m => {
                            const nx = agent.x + m.x;
                            const ny = agent.y + m.y;
                            if (nx >= 0 && nx < GAME_CONFIG.GRID_SIZE && ny >= 0 && ny < GAME_CONFIG.GRID_SIZE && maze[ny][nx] === 0) {
                                const dist = Math.abs(nx - target.x) + Math.abs(ny - target.y);
                                if (dist < minDist) {
                                    minDist = dist;
                                    bestMove = m;
                                }
                            }
                        });
                    } else {
                        // Follow the LATEST symbolic clue
                        const relevantClue = [...currentClues].reverse().find(c => {
                            const d = Math.abs(c.x - agent.x) + Math.abs(c.y - agent.y);
                            return d > 0 && d < 10; // Ignore clues they are already on
                        });

                        if (relevantClue) {
                            moves.forEach(m => {
                                const nx = agent.x + m.x;
                                const ny = agent.y + m.y;
                                if (nx >= 0 && nx < GAME_CONFIG.GRID_SIZE && ny >= 0 && ny < GAME_CONFIG.GRID_SIZE && maze[ny][nx] === 0) {
                                    // Add small random noise to break ties and prevent static behaviors
                                    const dist = Math.abs(nx - relevantClue.x) + Math.abs(ny - relevantClue.y) + (Math.random() * 0.1);
                                    if (dist < minDist) {
                                        minDist = dist;
                                        bestMove = m;
                                    }
                                }
                            });
                        }
                        else {
                            // Random exploration with avoidance of recent tiles (memory)
                            const validMoves = moves.filter(m => {
                                const nx = agent.x + m.x;
                                const ny = agent.y + m.y;
                                return nx >= 0 && nx < GAME_CONFIG.GRID_SIZE && ny >= 0 && ny < GAME_CONFIG.GRID_SIZE && maze[ny][nx] === 0;
                            });

                            if (validMoves.length > 0) {
                                // Score moves based on memory (penalize recently visited)
                                let bestExploreMove = validMoves[0];
                                let minPenalty = 1000;

                                validMoves.forEach(m => {
                                    const nx = agent.x + m.x;
                                    const ny = agent.y + m.y;
                                    const memoryIndex = agent.memory.findIndex(pos => pos.x === nx && pos.y === ny);
                                    const penalty = memoryIndex === -1 ? 0 : (agent.memory.length - memoryIndex);

                                    if (penalty < minPenalty) {
                                        minPenalty = penalty;
                                        bestExploreMove = m;
                                    }
                                });
                                bestMove = bestExploreMove;
                            }
                        }
                    }

                    // Update seeker memory
                    agent.memory.push({ x: agent.x, y: agent.y });
                    if (agent.memory.length > 15) agent.memory.shift();
                }

                agent.x += bestMove.x;
                agent.y += bestMove.y;

                // Check for "found"
                if (agent.type === 'SEEKER') {
                    newAgents.forEach(hider => {
                        if (hider.type === 'HIDER' && !hider.found && hider.x === agent.x && hider.y === agent.y) {
                            hider.found = true;
                            newSeekerScore += 1;
                        }
                    });
                }
            });

            // Update clues (keep only recent clues)
            return {
                agents: newAgents,
                clues: [...currentClues, ...newClues].slice(-40),
                score: { ...prev.score, seekers: newSeekerScore }
            };
        });
    }, [maze, center, phase]);

    useEffect(() => {
        if (phase === PHASES.HIDING || phase === PHASES.SEEKING) {
            loopRef.current = setInterval(moveAgents, GAME_CONFIG.MOVE_INTERVAL);
        } else {
            clearInterval(loopRef.current);
        }
        return () => clearInterval(loopRef.current);
    }, [phase, moveAgents]);

    return {
        phase,
        timeLeft,
        maze,
        agents: gameData.agents,
        score: gameData.score,
        clues: gameData.clues,
        toggleWall,
        startGame,
        resetGame: () => setPhase(PHASES.SETUP)
    };
};
