"""
Real-Time MindState WebSocket Server
Provides real-time cognitive state detection and recommendations via WebSocket
"""

import asyncio
import websockets
import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Set
import logging

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from behavioral_analyzer.behavior_simulator import BehaviorDataGenerator
from behavioral_analyzer.cognitive_state_detector import CognitiveStateClassifier
from recommendation_engine.mindstate_recommender import MindStateRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeMindStateServer:
    """Real-time WebSocket server for MindState system"""
    
    def __init__(self):
        self.app = FastAPI(title="MindState Real-Time API")
        self.connected_clients: Set[WebSocket] = set()
        
        # Initialize MindState components
        self.behavior_generator = BehaviorDataGenerator()
        self.cognitive_classifier = CognitiveStateClassifier()
        self.recommender = MindStateRecommender()
        
        # Session tracking
        self.active_sessions = {}
        self.is_system_trained = False
        
        # Setup routes
        self.setup_routes()
        self.setup_content_database()
        
        logger.info("MindState Real-Time Server initialized")
    
    def setup_content_database(self):
        """Load content database for recommendations"""
        
        content_db = [
            {
                'title': 'Friends - The One Where...',
                'genre': 'Comedy/Sitcom',
                'duration': 22,
                'type': 'tv_series',
                'description': 'Classic sitcom about six friends in New York.',
                'plot': 'Light-hearted situations with familiar, beloved characters.'
            },
            {
                'title': 'The Office (US)',
                'genre': 'Comedy/Mockumentary',
                'duration': 22,
                'type': 'tv_series',
                'description': 'Workplace comedy set in a paper company.',
                'plot': 'Character-driven humor with minimal cognitive load.'
            },
            {
                'title': 'Inception',
                'genre': 'Sci-Fi/Thriller',
                'duration': 148,
                'type': 'movie',
                'description': 'A thief who enters dreams to steal secrets.',
                'plot': 'Complex multi-layered narrative requiring high attention and analysis.'
            },
            {
                'title': 'Planet Earth',
                'genre': 'Documentary/Nature',
                'duration': 50,
                'type': 'documentary',
                'description': 'Stunning wildlife documentary series.',
                'plot': 'Educational content with beautiful visuals and moderate complexity.'
            },
            {
                'title': 'The Great British Baking Show',
                'genre': 'Reality/Competition',
                'duration': 60,
                'type': 'tv_series',
                'description': 'Gentle baking competition with positive atmosphere.',
                'plot': 'Calming, predictable format perfect for relaxation.'
            },
            {
                'title': 'Cosmos: A Space-Time Odyssey',
                'genre': 'Documentary/Science',
                'duration': 43,
                'type': 'documentary',
                'description': 'Exploration of space and scientific discovery.',
                'plot': 'Complex scientific concepts requiring focused attention.'
            },
            {
                'title': 'Avatar: The Last Airbender',
                'genre': 'Animation/Adventure',
                'duration': 23,
                'type': 'tv_series',
                'description': 'Young Aang masters airbending to save the world.',
                'plot': 'Engaging story with moderate complexity and emotional depth.'
            },
            {
                'title': 'Mindfulness Guide',
                'genre': 'Educational/Wellness',
                'duration': 15,
                'type': 'educational',
                'description': 'Guided meditation and mindfulness exercises.',
                'plot': 'Calming instructional content for stress relief.'
            }
        ]
        
        self.recommender.load_content_database(content_db)
        logger.info(f"Loaded {len(content_db)} content items")
    
    def train_cognitive_system(self):
        """Train the cognitive detection system with synthetic data"""
        
        if self.is_system_trained:
            return
        
        logger.info("Training cognitive state detection system...")
        
        try:
            # Generate training data
            training_data = self.behavior_generator.generate_dataset(sessions_per_state=25)
            
            # Train classifier
            results = self.cognitive_classifier.train(training_data)
            best_accuracy = max(results.values(), key=lambda x: x['test_accuracy'])['test_accuracy']
            
            self.is_system_trained = True
            logger.info(f"System trained successfully - Accuracy: {best_accuracy:.1%}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.is_system_trained = False
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket_connection(websocket)
        
        @self.app.get("/")
        async def get_dashboard():
            return HTMLResponse(self.get_dashboard_html())
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "system_trained": self.is_system_trained,
                "active_connections": len(self.connected_clients),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/train")
        async def train_system():
            self.train_cognitive_system()
            return {"message": "System training completed", "trained": self.is_system_trained}
    
    async def handle_websocket_connection(self, websocket: WebSocket):
        """Handle WebSocket connections for real-time updates"""
        
        await websocket.accept()
        self.connected_clients.add(websocket)
        session_id = f"session_{len(self.active_sessions) + 1}"
        
        # Initialize session
        self.active_sessions[session_id] = {
            'start_time': datetime.now(),
            'behavioral_buffer': [],
            'state_history': [],
            'current_state': 'discovery_mode',
            'recommendations': []
        }
        
        logger.info(f"New WebSocket connection: {session_id}")
        
        try:
            # Train system if not already trained
            if not self.is_system_trained:
                await websocket.send_text(json.dumps({
                    'type': 'system_status',
                    'message': 'Training cognitive detection system...',
                    'training': True
                }))
                self.train_cognitive_system()
                await websocket.send_text(json.dumps({
                    'type': 'system_status',
                    'message': 'System ready for real-time detection!',
                    'training': False,
                    'trained': True
                }))
            
            await self.stream_realtime_data(websocket, session_id)
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected normally: {session_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {session_id}: {e}")
        finally:
            self.connected_clients.discard(websocket)
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            logger.info(f"Cleaned up session: {session_id}")
    
    async def stream_realtime_data(self, websocket: WebSocket, session_id: str):
        """Stream real-time behavioral data and cognitive predictions"""
        
        session = self.active_sessions[session_id]
        error_count = 0
        max_errors = 5
        
        while True:
            try:
                # Calculate session duration
                session_duration = (datetime.now() - session['start_time']).total_seconds()
                
                # Generate realistic behavioral event
                behavioral_event = self.behavior_generator.generate_realtime_event(session_duration)
                
                # Add to session buffer (keep last 20 events for analysis)
                session['behavioral_buffer'].append(behavioral_event)
                if len(session['behavioral_buffer']) > 20:
                    session['behavioral_buffer'].pop(0)
                
                # Predict cognitive state if we have enough data
                current_state = session['current_state']
                confidence = 0.8
                recommendations = session.get('recommendations', {})
                
                if len(session['behavioral_buffer']) >= 5 and self.is_system_trained:
                    try:
                        buffer_df = pd.DataFrame(session['behavioral_buffer'])
                        
                        prediction = self.cognitive_classifier.predict(buffer_df)
                        current_state = prediction['state']
                        confidence = prediction['confidence']
                        
                        # Update session state
                        session['current_state'] = current_state
                        session['state_history'].append({
                            'timestamp': datetime.now(),
                            'state': current_state,
                            'confidence': confidence
                        })
                        
                        # Get fresh recommendations periodically
                        if (not session['recommendations'] or 
                            len(session['state_history']) % 8 == 0):  # Update every 8 predictions
                            
                            recommendations = self.recommender.get_recommendations(
                                behavioral_data=buffer_df,
                                num_recommendations=3
                            )
                            session['recommendations'] = recommendations
                        
                    except Exception as pred_error:
                        logger.warning(f"Prediction error for {session_id}: {pred_error}")
                        # Continue with previous state
                        pass
                
                # Prepare real-time update
                update = {
                    'type': 'realtime_update',
                    'timestamp': datetime.now().isoformat(),
                    'session_duration': round(session_duration, 1),
                    'behavioral_metrics': {
                        'typing_speed': round(behavioral_event.get('typing_speed', 0), 1),
                        'typing_errors': round(behavioral_event.get('typing_errors', 0), 3),
                        'mouse_precision': round(behavioral_event.get('mouse_precision', 0), 3),
                        'session_focus': round(behavioral_event.get('session_focus', 0), 3),
                        'app_switches': round(behavioral_event.get('app_switches', 0), 2)
                    },
                    'cognitive_state': {
                        'current_state': current_state,
                        'confidence': round(confidence, 3),
                        'state_description': self.get_state_description(current_state)
                    },
                    'recommendations': recommendations.get('recommendations', [])[:3] if recommendations else []
                }
                
                # Send update to client
                await websocket.send_text(json.dumps(update, default=str))
                
                # Reset error count on successful send
                error_count = 0
                
                # Wait before next update (1 Hz - every 1 second for stability)
                await asyncio.sleep(1.0)
                
            except WebSocketDisconnect:
                logger.info(f"Client disconnected: {session_id}")
                break
            except Exception as e:
                error_count += 1
                logger.error(f"Error in real-time stream for {session_id}: {e}")
                
                if error_count >= max_errors:
                    logger.error(f"Too many errors for {session_id}, closing connection")
                    break
                
                # Wait longer on error
                await asyncio.sleep(2.0)
    
    def get_state_description(self, state: str) -> str:
        """Get human-readable state description"""
        descriptions = {
            'stress_state': 'Feeling stressed - need calming content',
            'recovery_mode': 'In recovery mode - prefer familiar, easy content',
            'discovery_mode': 'Discovery mode - ready to explore',
            'learning_mode': 'Learning focused - want educational content',
            'flow_state': 'Deep focus - can handle complex content'
        }
        return descriptions.get(state, 'Current cognitive state')
    
    def get_dashboard_html(self) -> str:
        """Generate the real-time dashboard HTML with improved connection handling"""
        
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MindState - Real-Time Cognitive Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
            margin: 10px 0;
        }
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .card h3 {
            margin: 0 0 15px 0;
            font-size: 1.3em;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            font-size: 1.1em;
        }
        .metric-value {
            font-weight: bold;
            color: #ffd700;
        }
        .state-indicator {
            text-align: center;
            padding: 20px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            margin: 15px 0;
        }
        .state-name {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .confidence-bar {
            width: 100%;
            height: 20px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcf7f);
            transition: width 0.5s ease;
            border-radius: 10px;
        }
        .recommendation {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #ffd700;
        }
        .recommendation-title {
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 5px;
        }
        .recommendation-details {
            font-size: 0.9em;
            opacity: 0.8;
            margin: 5px 0;
        }
        .status {
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .status.connecting {
            background: rgba(255, 193, 7, 0.3);
        }
        .status.connected {
            background: rgba(40, 167, 69, 0.3);
        }
        .status.error {
            background: rgba(220, 53, 69, 0.3);
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        .controls button {
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            margin: 0 10px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .controls button:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        @media (max-width: 1200px) {
            .dashboard {
                grid-template-columns: 1fr 1fr;
            }
        }
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 MindState</h1>
        <p>Real-Time Cognitive State Detection & Recommendations</p>
        <div id="connection-status" class="status connecting">Connecting to real-time stream...</div>
    </div>

    <div class="controls">
        <button onclick="connect()">🔌 Reconnect</button>
        <button onclick="toggleUpdates()">⏸️ Pause Updates</button>
        <button onclick="clearData()">🗑️ Clear Data</button>
    </div>

    <div class="dashboard">
        <!-- Behavioral Metrics -->
        <div class="card">
            <h3>📊 Behavioral Metrics</h3>
            <div class="metric">
                <span>Typing Speed:</span>
                <span class="metric-value" id="typing-speed">-- WPM</span>
            </div>
            <div class="metric">
                <span>Error Rate:</span>
                <span class="metric-value" id="error-rate">--%</span>
            </div>
            <div class="metric">
                <span>Mouse Precision:</span>
                <span class="metric-value" id="mouse-precision">--</span>
            </div>
            <div class="metric">
                <span>Focus Score:</span>
                <span class="metric-value" id="focus-score">--</span>
            </div>
            <div class="metric">
                <span>App Switches:</span>
                <span class="metric-value" id="app-switches">--/min</span>
            </div>
            <div class="metric">
                <span>Session Duration:</span>
                <span class="metric-value" id="session-duration">--s</span>
            </div>
        </div>

        <!-- Cognitive State -->
        <div class="card">
            <h3>🧠 Cognitive State</h3>
            <div class="state-indicator">
                <div class="state-name" id="current-state">Initializing...</div>
                <div id="state-description">Connecting to cognitive detection system</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidence-fill" style="width: 0%"></div>
                </div>
                <div id="confidence-text">Confidence: --%</div>
            </div>
        </div>

        <!-- Recommendations -->
        <div class="card">
            <h3>🎬 Smart Recommendations</h3>
            <div id="recommendations-container">
                <div class="recommendation">
                    <div class="recommendation-title">Loading recommendations...</div>
                    <div class="recommendation-details">Analyzing your cognitive state for personalized suggestions</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let reconnectAttempts = 0;
        let maxReconnectAttempts = 10;
        let updatesPaused = false;
        let reconnectInterval;

        function connect() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
            
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);

            ws.onopen = function(event) {
                console.log('Connected to MindState real-time stream');
                document.getElementById('connection-status').className = 'status connected';
                document.getElementById('connection-status').textContent = '✅ Connected to real-time stream';
                reconnectAttempts = 0;
                clearInterval(reconnectInterval);
            };

            ws.onmessage = function(event) {
                if (!updatesPaused) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                }
            };

            ws.onclose = function(event) {
                console.log('WebSocket connection closed:', event.code, event.reason);
                document.getElementById('connection-status').className = 'status error';
                
                if (event.code === 1001) {
                    document.getElementById('connection-status').textContent = '🔄 Connection closed - attempting to reconnect...';
                } else {
                    document.getElementById('connection-status').textContent = '❌ Connection lost - attempting to reconnect...';
                }
                
                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000); // Exponential backoff
                    console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
                    
                    reconnectInterval = setTimeout(() => {
                        connect();
                    }, delay);
                } else {
                    document.getElementById('connection-status').textContent = '❌ Connection failed - click Reconnect to try again';
                }
            };

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }

        function toggleUpdates() {
            updatesPaused = !updatesPaused;
            const button = event.target;
            button.textContent = updatesPaused ? '▶️ Resume Updates' : '⏸️ Pause Updates';
        }

        function clearData() {
            // Reset all displays
            document.getElementById('typing-speed').textContent = '-- WPM';
            document.getElementById('error-rate').textContent = '--%';
            document.getElementById('mouse-precision').textContent = '--';
            document.getElementById('focus-score').textContent = '--';
            document.getElementById('app-switches').textContent = '--/min';
            document.getElementById('session-duration').textContent = '--s';
            document.getElementById('current-state').textContent = 'Initializing...';
            document.getElementById('state-description').textContent = 'Waiting for data...';
            document.getElementById('confidence-fill').style.width = '0%';
            document.getElementById('confidence-text').textContent = 'Confidence: --%';
        }

        function updateDashboard(data) {
            if (data.type === 'realtime_update') {
                // Update behavioral metrics
                const metrics = data.behavioral_metrics;
                document.getElementById('typing-speed').textContent = `${metrics.typing_speed} WPM`;
                document.getElementById('error-rate').textContent = `${(metrics.typing_errors * 100).toFixed(1)}%`;
                document.getElementById('mouse-precision').textContent = metrics.mouse_precision.toFixed(2);
                document.getElementById('focus-score').textContent = metrics.session_focus.toFixed(2);
                document.getElementById('app-switches').textContent = `${metrics.app_switches}/min`;
                document.getElementById('session-duration').textContent = `${data.session_duration}s`;

                // Update cognitive state
                const state = data.cognitive_state;
                document.getElementById('current-state').textContent = state.current_state.replace('_', ' ').toUpperCase();
                document.getElementById('state-description').textContent = state.state_description;
                
                const confidence = state.confidence * 100;
                document.getElementById('confidence-fill').style.width = `${confidence}%`;
                document.getElementById('confidence-text').textContent = `Confidence: ${confidence.toFixed(0)}%`;

                // Update recommendations
                updateRecommendations(data.recommendations);
            } else if (data.type === 'system_status') {
                document.getElementById('connection-status').textContent = data.message;
                if (data.training) {
                    document.getElementById('connection-status').className = 'status connecting';
                } else if (data.trained) {
                    document.getElementById('connection-status').className = 'status connected';
                }
            }
        }

        function updateRecommendations(recommendations) {
            const container = document.getElementById('recommendations-container');
            
            if (!recommendations || recommendations.length === 0) {
                container.innerHTML = `
                    <div class="recommendation">
                        <div class="recommendation-title">Analyzing your state...</div>
                        <div class="recommendation-details">Building personalized recommendations</div>
                    </div>
                `;
                return;
            }

            container.innerHTML = '';
            recommendations.forEach((rec, index) => {
                const recElement = document.createElement('div');
                recElement.className = 'recommendation';
                recElement.innerHTML = `
                    <div class="recommendation-title">${rec.title}</div>
                    <div class="recommendation-details">
                        ${rec.duration} min • ${rec.complexity_category} complexity (${rec.complexity_score.toFixed(2)})
                    </div>
                    <div class="recommendation-details">
                        💡 ${rec.why_recommended}
                    </div>
                `;
                container.appendChild(recElement);
            });
        }

        // Start connection when page loads
        connect();
    </script>
</body>
</html>
        """

def start_server():
    """Start the MindState real-time server"""
    
    server = RealTimeMindStateServer()
    
    print("🚀 Starting MindState Real-Time Server...")
    print("📊 Dashboard will be available at: http://localhost:8000")
    print("🔌 WebSocket endpoint: ws://localhost:8000/ws")
    print("💻 Press Ctrl+C to stop the server")
    print("\nServer Features:")
    print("  - Auto-reconnection on connection loss")
    print("  - Error recovery and stability")
    print("  - Real-time cognitive state detection")
    print("  - Adaptive recommendations")
    
    uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()
