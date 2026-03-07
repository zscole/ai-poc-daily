#!/usr/bin/env python3
"""
AgentOps Simple Demo - Simplified version without external dependencies
Demonstrates the multi-agent coordination concept using only built-in Python libraries.
"""

import asyncio
import sqlite3
import json
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime

@dataclass
class AgentMessage:
    sender_id: str
    recipient_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class SimpleAgentRuntime:
    """Simplified agent runtime using only built-in libraries"""
    
    def __init__(self, db_path: str = "simple_agentops.db"):
        self.db_path = db_path
        self.agents = {}
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_states (
                agent_id TEXT,
                timestamp REAL,
                state_data TEXT,
                PRIMARY KEY (agent_id, timestamp)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_id TEXT,
                recipient_id TEXT,
                message_type TEXT,
                payload TEXT,
                timestamp REAL,
                delivered BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def record_agent_state(self, agent_id: str, state_data: Dict[str, Any]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO agent_states (agent_id, timestamp, state_data)
            VALUES (?, ?, ?)
        ''', (agent_id, time.time(), json.dumps(state_data)))
        
        conn.commit()
        conn.close()
    
    async def send_message(self, message: AgentMessage):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO agent_messages (sender_id, recipient_id, message_type, payload, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (message.sender_id, message.recipient_id, message.message_type, 
              json.dumps(message.payload), message.timestamp))
        
        conn.commit()
        conn.close()
        
        print(f"[MESSAGE] {message.sender_id} -> {message.recipient_id}: {message.message_type}")

class SimpleAgent:
    """Base agent class for demo"""
    
    def __init__(self, agent_id: str, runtime: SimpleAgentRuntime):
        self.agent_id = agent_id
        self.runtime = runtime
        self.running = False
    
    async def send_message(self, recipient_id: str, message_type: str, payload: Dict[str, Any]):
        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload
        )
        await self.runtime.send_message(message)
    
    async def execute_cycle(self) -> Dict[str, Any]:
        """Override in subclasses"""
        return {}

class SimpleScoutAgent(SimpleAgent):
    """Simulated scout agent for service discovery"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.service_count = 5
    
    async def execute_cycle(self) -> Dict[str, Any]:
        # Simulate discovering services
        services = []
        for i in range(self.service_count):
            services.append({
                'id': f'service-{i+1}',
                'name': f'microservice-{i+1}',
                'status': 'running',
                'cpu_usage': random.uniform(10, 30),
                'memory_usage': random.uniform(20, 40)
            })
        
        # Occasionally discover new services
        if random.random() < 0.3:
            self.service_count += 1
            print(f"[SCOUT] Discovered new service: service-{self.service_count}")
        
        results = {
            'discovered_services': services,
            'total_services': len(services),
            'discovery_time': time.time()
        }
        
        await self.runtime.record_agent_state(self.agent_id, results)
        return results

class SimpleHealthAgent(SimpleAgent):
    """Simulated health monitoring agent"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_cpu = 25.0
        self.baseline_memory = 35.0
    
    async def execute_cycle(self) -> Dict[str, Any]:
        # Simulate system metrics
        current_cpu = self.baseline_cpu + random.uniform(-10, 20)
        current_memory = self.baseline_memory + random.uniform(-10, 25)
        
        anomalies = []
        health_score = 100
        
        # Detect anomalies
        if current_cpu > 80:
            anomalies.append({
                'metric': 'cpu_usage',
                'value': current_cpu,
                'severity': 'high',
                'description': f'CPU usage at {current_cpu:.1f}%'
            })
            health_score -= 20
        
        if current_memory > 85:
            anomalies.append({
                'metric': 'memory_usage',
                'value': current_memory,
                'severity': 'high', 
                'description': f'Memory usage at {current_memory:.1f}%'
            })
            health_score -= 15
        
        # Alert on anomalies
        if anomalies:
            await self.send_message('coordinator', 'health_alert', {
                'anomalies': anomalies,
                'severity': 'high' if len(anomalies) > 1 else 'medium'
            })
            print(f"[HEALTH] ALERT: {len(anomalies)} anomalies detected")
        
        results = {
            'cpu_usage': current_cpu,
            'memory_usage': current_memory,
            'anomalies': anomalies,
            'health_score': max(0, health_score),
            'check_time': time.time()
        }
        
        await self.runtime.record_agent_state(self.agent_id, results)
        return results

class SimpleSecurityAgent(SimpleAgent):
    """Simulated security monitoring agent"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scan_counter = 0
    
    async def execute_cycle(self) -> Dict[str, Any]:
        self.scan_counter += 1
        vulnerabilities = []
        security_score = 100
        
        # Simulate vulnerability discovery
        if random.random() < 0.4:  # 40% chance of finding issues
            vuln_types = [
                ('open_port', 'medium', 'Unnecessary port 21 open'),
                ('outdated_package', 'high', 'Outdated SSL library detected'),
                ('weak_config', 'medium', 'Weak encryption configuration'),
                ('default_creds', 'critical', 'Default credentials found')
            ]
            
            selected_vuln = random.choice(vuln_types)
            vulnerabilities.append({
                'type': selected_vuln[0],
                'severity': selected_vuln[1],
                'description': selected_vuln[2],
                'found_at': time.time()
            })
            
            if selected_vuln[1] == 'critical':
                security_score -= 30
            elif selected_vuln[1] == 'high':
                security_score -= 20
            else:
                security_score -= 10
        
        # Alert on critical issues
        critical_vulns = [v for v in vulnerabilities if v['severity'] == 'critical']
        if critical_vulns:
            await self.send_message('coordinator', 'security_alert', {
                'critical_vulnerabilities': critical_vulns,
                'total_vulnerabilities': len(vulnerabilities),
                'severity': 'critical'
            })
            print(f"[SECURITY] CRITICAL ALERT: {len(critical_vulns)} critical vulnerabilities")
        
        results = {
            'vulnerabilities': vulnerabilities,
            'security_score': security_score,
            'scan_number': self.scan_counter,
            'recommendations': self._generate_recommendations(vulnerabilities),
            'scan_time': time.time()
        }
        
        await self.runtime.record_agent_state(self.agent_id, results)
        return results
    
    def _generate_recommendations(self, vulns):
        recommendations = []
        for vuln in vulns:
            if vuln['type'] == 'open_port':
                recommendations.append("Close unnecessary ports")
            elif vuln['type'] == 'outdated_package':
                recommendations.append("Update packages to latest versions")
            elif vuln['type'] == 'weak_config':
                recommendations.append("Strengthen encryption configuration")
            elif vuln['type'] == 'default_creds':
                recommendations.append("Change default passwords immediately")
        return recommendations

class SimpleCoordinatorAgent(SimpleAgent):
    """Coordinator agent that makes strategic decisions"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_counter = 0
    
    async def execute_cycle(self) -> Dict[str, Any]:
        # Get pending messages
        alerts = await self._get_pending_alerts()
        
        # Get latest agent states
        agent_states = await self._get_agent_states()
        
        # Generate system overview
        overview = self._generate_system_overview(agent_states)
        
        # Make decisions
        decisions = self._make_decisions(alerts, agent_states)
        
        results = {
            'system_overview': overview,
            'new_decisions': len(decisions),
            'total_alerts': len(alerts),
            'decision_count': self.decision_counter,
            'recent_decisions': decisions,
            'coordination_time': time.time()
        }
        
        # Display decisions
        for decision in decisions:
            print(f"[COORDINATOR] DECISION: {decision['action']} - {decision['rationale']}")
        
        await self.runtime.record_agent_state(self.agent_id, results)
        return results
    
    async def _get_pending_alerts(self) -> List[Dict]:
        conn = sqlite3.connect(self.runtime.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT sender_id, message_type, payload, timestamp
            FROM agent_messages
            WHERE recipient_id = 'coordinator' AND delivered = FALSE
            ORDER BY timestamp DESC
        ''')
        
        alerts = []
        for row in cursor.fetchall():
            sender_id, message_type, payload, timestamp = row
            alerts.append({
                'sender': sender_id,
                'type': message_type,
                'payload': json.loads(payload),
                'timestamp': timestamp
            })
        
        # Mark as delivered
        cursor.execute('''
            UPDATE agent_messages 
            SET delivered = TRUE 
            WHERE recipient_id = 'coordinator' AND delivered = FALSE
        ''')
        
        conn.commit()
        conn.close()
        return alerts
    
    async def _get_agent_states(self) -> Dict[str, Dict]:
        conn = sqlite3.connect(self.runtime.db_path)
        cursor = conn.cursor()
        
        states = {}
        for agent_id in ['scout', 'health', 'security']:
            cursor.execute('''
                SELECT state_data, timestamp
                FROM agent_states
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (agent_id,))
            
            row = cursor.fetchone()
            if row:
                states[agent_id] = {
                    'data': json.loads(row[0]),
                    'timestamp': row[1]
                }
        
        conn.close()
        return states
    
    def _generate_system_overview(self, agent_states: Dict) -> Dict:
        overview = {
            'infrastructure_health': 'unknown',
            'security_posture': 'unknown',
            'total_services': 0,
            'active_threats': 0,
            'overall_status': 'monitoring'
        }
        
        if 'health' in agent_states:
            health_score = agent_states['health']['data'].get('health_score', 0)
            if health_score >= 80:
                overview['infrastructure_health'] = 'healthy'
            elif health_score >= 60:
                overview['infrastructure_health'] = 'degraded'
            else:
                overview['infrastructure_health'] = 'critical'
        
        if 'security' in agent_states:
            security_score = agent_states['security']['data'].get('security_score', 0)
            vulns = len(agent_states['security']['data'].get('vulnerabilities', []))
            overview['active_threats'] = vulns
            
            if security_score >= 80:
                overview['security_posture'] = 'secure'
            elif security_score >= 60:
                overview['security_posture'] = 'vulnerable'
            else:
                overview['security_posture'] = 'critical'
        
        if 'scout' in agent_states:
            overview['total_services'] = agent_states['scout']['data'].get('total_services', 0)
        
        return overview
    
    def _make_decisions(self, alerts: List[Dict], agent_states: Dict) -> List[Dict]:
        decisions = []
        
        # Security decisions
        security_alerts = [a for a in alerts if a['type'] == 'security_alert']
        for alert in security_alerts:
            self.decision_counter += 1
            if alert['payload'].get('severity') == 'critical':
                decisions.append({
                    'id': self.decision_counter,
                    'action': 'immediate_security_response',
                    'rationale': 'Critical vulnerabilities require immediate action',
                    'priority': 'high',
                    'estimated_impact': 'high risk mitigation'
                })
        
        # Health decisions
        health_alerts = [a for a in alerts if a['type'] == 'health_alert']
        for alert in health_alerts:
            self.decision_counter += 1
            anomalies = alert['payload'].get('anomalies', [])
            if len(anomalies) > 1:
                decisions.append({
                    'id': self.decision_counter,
                    'action': 'scale_resources',
                    'rationale': f'Multiple performance anomalies detected: {len(anomalies)} issues',
                    'priority': 'medium',
                    'estimated_impact': 'improved system performance'
                })
        
        # Proactive decisions based on trends
        if 'health' in agent_states and 'scout' in agent_states:
            health_data = agent_states['health']['data']
            scout_data = agent_states['scout']['data']
            
            avg_service_cpu = sum(s.get('cpu_usage', 0) for s in scout_data.get('discovered_services', [])) / max(1, len(scout_data.get('discovered_services', [])))
            
            if avg_service_cpu > 70 and health_data.get('health_score', 100) < 80:
                self.decision_counter += 1
                decisions.append({
                    'id': self.decision_counter,
                    'action': 'optimize_resource_allocation',
                    'rationale': f'High average service CPU usage: {avg_service_cpu:.1f}%',
                    'priority': 'medium',
                    'estimated_impact': 'better resource distribution'
                })
        
        return decisions

class SimpleDemo:
    """Main demo orchestrator"""
    
    def __init__(self):
        self.runtime = SimpleAgentRuntime()
        self.agents = {}
    
    def setup_agents(self):
        self.agents['scout'] = SimpleScoutAgent('scout', self.runtime)
        self.agents['health'] = SimpleHealthAgent('health', self.runtime)
        self.agents['security'] = SimpleSecurityAgent('security', self.runtime)
        self.agents['coordinator'] = SimpleCoordinatorAgent('coordinator', self.runtime)
    
    async def run_demo_cycle(self):
        """Run one complete cycle of all agents"""
        print(f"\n{'='*50}")
        print(f"DEMO CYCLE - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        
        # Run each agent's cycle
        for agent_name, agent in self.agents.items():
            print(f"\n[{agent_name.upper()}] Executing cycle...")
            await agent.execute_cycle()
            await asyncio.sleep(1)  # Brief pause for readability
        
        print(f"\nCycle complete. Agents coordination in progress...")
        print(f"{'='*50}")
    
    async def run_interactive_demo(self):
        print("AgentOps - Multi-Agent Infrastructure Intelligence Demo")
        print("=" * 60)
        print()
        print("This simplified demo shows how AI agents coordinate to manage infrastructure:")
        print("- Scout Agent: Discovers and monitors services")
        print("- Health Agent: Monitors system performance and detects anomalies") 
        print("- Security Agent: Scans for vulnerabilities and security issues")
        print("- Coordinator Agent: Synthesizes insights and makes strategic decisions")
        print()
        
        self.setup_agents()
        
        print("Starting demo... Press Ctrl+C to stop")
        print()
        
        try:
            cycle_count = 0
            while True:
                cycle_count += 1
                await self.run_demo_cycle()
                
                if cycle_count % 3 == 0:  # Every 3 cycles, show summary
                    await self.show_system_summary()
                
                await asyncio.sleep(5)  # Wait 5 seconds between cycles
                
        except KeyboardInterrupt:
            print("\n\nDemo stopped by user.")
            await self.show_final_summary()
    
    async def show_system_summary(self):
        """Show current system status"""
        print(f"\n{'*'*30} SYSTEM SUMMARY {'*'*30}")
        
        conn = sqlite3.connect(self.runtime.db_path)
        cursor = conn.cursor()
        
        for agent_id in ['scout', 'health', 'security', 'coordinator']:
            cursor.execute('''
                SELECT state_data, timestamp
                FROM agent_states
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (agent_id,))
            
            row = cursor.fetchone()
            if row:
                state = json.loads(row[0])
                age = time.time() - row[1]
                
                print(f"\n{agent_id.upper()} Agent (data age: {age:.1f}s):")
                
                if agent_id == 'scout':
                    print(f"  Services: {state.get('total_services', 0)}")
                elif agent_id == 'health':
                    print(f"  Health Score: {state.get('health_score', 0)}/100")
                    print(f"  CPU: {state.get('cpu_usage', 0):.1f}%")
                    print(f"  Memory: {state.get('memory_usage', 0):.1f}%")
                elif agent_id == 'security':
                    print(f"  Security Score: {state.get('security_score', 0)}/100")
                    print(f"  Vulnerabilities: {len(state.get('vulnerabilities', []))}")
                elif agent_id == 'coordinator':
                    overview = state.get('system_overview', {})
                    print(f"  Infrastructure: {overview.get('infrastructure_health', 'unknown')}")
                    print(f"  Security: {overview.get('security_posture', 'unknown')}")
                    print(f"  Decisions Made: {state.get('decision_count', 0)}")
        
        conn.close()
        print(f"{'*'*75}")
    
    async def show_final_summary(self):
        """Show final summary statistics"""
        print(f"\n{'='*60}")
        print("FINAL DEMO SUMMARY")
        print(f"{'='*60}")
        
        conn = sqlite3.connect(self.runtime.db_path)
        cursor = conn.cursor()
        
        # Count total states recorded
        cursor.execute('SELECT COUNT(*) FROM agent_states')
        total_states = cursor.fetchone()[0]
        
        # Count total messages
        cursor.execute('SELECT COUNT(*) FROM agent_messages')
        total_messages = cursor.fetchone()[0]
        
        # Get coordinator decisions
        cursor.execute('''
            SELECT state_data FROM agent_states 
            WHERE agent_id = 'coordinator'
            ORDER BY timestamp DESC LIMIT 1
        ''')
        row = cursor.fetchone()
        total_decisions = 0
        if row:
            state = json.loads(row[0])
            total_decisions = state.get('decision_count', 0)
        
        print(f"Agent State Updates: {total_states}")
        print(f"Inter-Agent Messages: {total_messages}")
        print(f"Coordinator Decisions: {total_decisions}")
        print("\nThis demonstrates how multiple AI agents can work together")
        print("to autonomously monitor and manage complex infrastructure.")
        print(f"{'='*60}")
        
        conn.close()

async def main():
    demo = SimpleDemo()
    await demo.run_interactive_demo()

if __name__ == "__main__":
    asyncio.run(main())