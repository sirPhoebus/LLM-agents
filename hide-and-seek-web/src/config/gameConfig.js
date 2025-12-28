export const GAME_CONFIG = {
    GRID_SIZE: 15,
    CELL_SIZE: 40,
    HIDER_COUNT: 4,
    SEEKER_COUNT: 2,
    HIDING_DURATION: 10, // seconds
    SEEKING_DURATION: 60, // seconds
    MOVE_INTERVAL: 500, // ms
    REWARD_WELL_HIDDEN: 100,
    REWARD_FOUND: -50,
    SYMBOLIC_CLUES: true,
    COLORS: {
        WALL: '#334155',
        FLOOR: '#1e293b',
        HIDER: '#4ade80',
        SEEKER: '#f87171',
        CENTER: '#e2e8f0'
    }
};
