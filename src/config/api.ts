/**
 * API Configuration
 * 
 * Update API_BASE_URL with your deployed backend URL
 */

// Set this to your deployed Python backend URL
// Examples:
// - Local: 'http://localhost:8000'
// - Heroku: 'https://your-app.herokuapp.com'
// - Railway: 'https://your-app.railway.app'
// - Custom domain: 'https://api.yourdomain.com'
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  health: `${API_BASE_URL}/health`,
  transcribe: `${API_BASE_URL}/transcribe`,
  summarize: `${API_BASE_URL}/summarize`,
  voiceToNotes: `${API_BASE_URL}/voice-to-notes`,
} as const;