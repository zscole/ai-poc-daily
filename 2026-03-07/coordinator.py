#!/usr/bin/env python3
"""
AgentOps Coordinator - Central intelligence for agent orchestration
Synthesizes insights from all agents and makes coordinated decisions.
"""

import asyncio
import sqlite3
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from agent_runtime import BaseAgent, AgentRuntime
import structlog

logger = structlog.get_logger()

@dataclass
class Decision:
    """Represents a coordinated decision from multiple agent inputs"""
    decision_id: str
    timestamp: float
    agents_consulted: List[str]
    decision_type: str
    recommendation: Dict[str, Any]
    confidence_score: float
    risk_assessment: str
    implementation_plan: List[str]

class CoordinatorAgent(BaseAgent):
    """Central coordinator that synthesizes insights from all agents"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_history: List[Decision] = []
        self.agent_states: Dict[str, Dict] = {}
        self.active_alerts: List[Dict] = []
        
    async def execute_cycle(self) -> Dict[str, Any]:
        """Coordinate all agents and make strategic decisions"""
        
        # Collect latest states from all agents
        await self._collect_agent_states()
        
        # Process any pending alerts
        await self._process_pending_alerts()
        
        # Generate system overview
        system_overview = self._generate_system_overview()
        
        # Make strategic decisions
        decisions = await self._make_strategic_decisions()
        
        # Update decision history
        self.decision_history.extend(decisions)
        
        results = {
            'system_overview': system_overview,
            'new_decisions': len(decisions),
            'active_alerts': len(self.active_alerts),
            'agent_health': self._assess_agent_health(),
            'recommendations': [d.recommendation for d in decisions[-5:]]  # Last 5 decisions
        }
        
        return results
    
    async def _collect_agent_states(self):
        """Collect latest state from each agent"""
        conn = sqlite3.connect(self.runtime.db_path)
        cursor = conn.cursor()
        
        # Get latest state for each agent
        cursor.execute('''
            SELECT agent_id, state_data, timestamp
            FROM agent_states a1
            WHERE timestamp = (
                SELECT MAX(timestamp)
                FROM agent_states a2
                WHERE a2.agent_id = a1.agent_id
            )
        ''')
        
        for row in cursor.fetchall():
            agent_id, state_data, timestamp = row
            self.agent_states[agent_id] = {
                'data': json.loads(state_data),
                'timestamp': timestamp,
                'age_seconds': time.time() - timestamp
            }
        
        conn.close()
    
    async def _process_pending_alerts(self):
        """Process any pending alerts from agents"""
        conn = sqlite3.connect(self.runtime.db_path)
        cursor = conn.cursor()
        
        # Get undelivered messages for coordinator
        cursor.execute('''
            SELECT id, sender_id, message_type, payload, timestamp
            FROM agent_messages
            WHERE recipient_id = ? AND delivered = FALSE
            ORDER BY timestamp ASC
        ''', ('coordinator',))
        
        for row in cursor.fetchall():
            msg_id, sender_id, msg_type, payload, timestamp = row
            alert_data = {
                'id': msg_id,
                'sender': sender_id,
                'type': msg_type,
                'payload': json.loads(payload),
                'timestamp': timestamp
            }
            
            self.active_alerts.append(alert_data)
            
            # Mark as delivered
            cursor.execute('UPDATE agent_messages SET delivered = TRUE WHERE id = ?', (msg_id,))
        
        conn.commit()
        conn.close()
    
    def _generate_system_overview(self) -> Dict[str, Any]:
        """Generate comprehensive system overview"""
        overview = {
            'infrastructure_health': 'unknown',
            'security_posture': 'unknown',
            'performance_status': 'healthy',
            'discovered_services': 0,
            'active_threats': 0,
            'resource_utilization': {}
        }
        
        # Health metrics
        if 'health' in self.agent_states:
            health_data = self.agent_states['health']['data']
            health_score = health_data.get('health_score', 0)
            
            if health_score >= 80:
                overview['infrastructure_health'] = 'healthy'
            elif health_score >= 60:
                overview['infrastructure_health'] = 'degraded'
            else:
                overview['infrastructure_health'] = 'critical'
            
            overview['performance_status'] = 'healthy' if len(health_data.get('anomalies', [])) == 0 else 'anomalies_detected'
            
            current_metrics = health_data.get('current_metrics', {})
            overview['resource_utilization'] = {
                'cpu_percent': current_metrics.get('cpu_percent', 0),
                'memory_percent': current_metrics.get('memory_percent', 0),
                'process_count': current_metrics.get('process_count', 0)
            }
        
        # Security status
        if 'security' in self.agent_states:
            security_data = self.agent_states['security']['data']
            security_score = security_data.get('security_score', 0)
            
            if security_score >= 80:
                overview['security_posture'] = 'secure'
            elif security_score >= 60:
                overview['security_posture'] = 'vulnerable'
            else:
                overview['security_posture'] = 'critical'
            
            overview['active_threats'] = len(security_data.get('vulnerabilities', []))
        
        # Discovery data
        if 'scout' in self.agent_states:
            scout_data = self.agent_states['scout']['data']
            overview['discovered_services'] = len(scout_data.get('discovered_services', []))
        
        return overview
    
    async def _make_strategic_decisions(self) -> List[Decision]:
        """Make strategic decisions based on all available data"""
        decisions = []
        
        # Decision 1: Resource Optimization
        if 'health' in self.agent_states:
            resource_decision = self._decide_resource_optimization()
            if resource_decision:
                decisions.append(resource_decision)
        
        # Decision 2: Security Response
        security_alerts = [a for a in self.active_alerts if a['type'] == 'security_alert']
        if security_alerts:
            security_decision = self._decide_security_response(security_alerts)
            if security_decision:
                decisions.append(security_decision)
        
        # Decision 3: Scaling Recommendations
        anomaly_alerts = [a for a in self.active_alerts if a['type'] == 'anomaly_alert']
        if anomaly_alerts:
            scaling_decision = self._decide_scaling_action(anomaly_alerts)
            if scaling_decision:
                decisions.append(scaling_decision)
        
        return decisions
    
    def _decide_resource_optimization(self) -> Optional[Decision]:
        """Decide on resource optimization actions"""
        health_data = self.agent_states['health']['data']
        current_metrics = health_data.get('current_metrics', {})
        
        cpu_percent = current_metrics.get('cpu_percent', 0)
        memory_percent = current_metrics.get('memory_percent', 0)
        
        if cpu_percent > 80 or memory_percent > 85:
            return Decision(
                decision_id=f"resource_opt_{int(time.time())}",
                timestamp=time.time(),
                agents_consulted=['health'],
                decision_type='resource_optimization',
                recommendation={
                    'action': 'optimize_resources',
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory_percent,
                    'suggested_actions': [
                        'Identify high-resource processes',
                        'Consider horizontal scaling',
                        'Optimize application configuration'
                    ]
                },
                confidence_score=0.8,
                risk_assessment='low',
                implementation_plan=[
                    '1. Analyze process resource consumption',
                    '2. Implement resource limits',
                    '3. Monitor impact on performance'
                ]
            )
        return None
    
    def _decide_security_response(self, security_alerts: List[Dict]) -> Optional[Decision]:
        """Decide on security response actions"""
        critical_issues = []
        high_issues = []
        
        for alert in security_alerts:
            payload = alert['payload']
            critical_issues.extend(payload.get('critical_issues', []))
            if payload.get('total_issues', 0) > 0:
                high_issues.append(alert)
        
        if critical_issues or len(high_issues) > 2:
            severity = 'critical' if critical_issues else 'high'
            
            return Decision(
                decision_id=f"security_response_{int(time.time())}",
                timestamp=time.time(),
                agents_consulted=['security'],
                decision_type='security_response',
                recommendation={
                    'action': 'immediate_security_remediation',
                    'severity': severity,
                    'critical_issues': len(critical_issues),
                    'total_issues': sum(a['payload'].get('total_issues', 0) for a in high_issues),
                    'immediate_actions': [
                        'Isolate affected services',
                        'Apply security patches',
                        'Review access controls'
                    ]
                },
                confidence_score=0.9,
                risk_assessment='high',
                implementation_plan=[
                    '1. Implement immediate containment',
                    '2. Apply critical security patches',
                    '3. Conduct security audit',
                    '4. Update security policies'
                ]
            )
        return None
    
    def _decide_scaling_action(self, anomaly_alerts: List[Dict]) -> Optional[Decision]:
        """Decide on scaling actions based on anomalies"""
        high_severity_anomalies = []
        
        for alert in anomaly_alerts:
            if alert['payload'].get('severity') == 'high':
                high_severity_anomalies.extend(alert['payload'].get('anomalies', []))
        
        if len(high_severity_anomalies) > 0:
            cpu_anomalies = [a for a in high_severity_anomalies if 'cpu' in a.get('metric', '')]
            memory_anomalies = [a for a in high_severity_anomalies if 'memory' in a.get('metric', '')]
            
            recommended_actions = []
            if cpu_anomalies:
                recommended_actions.append('Consider CPU scaling')
            if memory_anomalies:
                recommended_actions.append('Consider memory optimization')
            
            return Decision(
                decision_id=f"scaling_action_{int(time.time())}",
                timestamp=time.time(),
                agents_consulted=['health'],
                decision_type='scaling_recommendation',
                recommendation={
                    'action': 'evaluate_scaling',
                    'anomaly_count': len(high_severity_anomalies),
                    'cpu_issues': len(cpu_anomalies),
                    'memory_issues': len(memory_anomalies),
                    'recommended_actions': recommended_actions
                },
                confidence_score=0.7,
                risk_assessment='medium',
                implementation_plan=[
                    '1. Analyze resource usage patterns',
                    '2. Test scaling scenarios',
                    '3. Implement gradual scaling',
                    '4. Monitor performance impact'
                ]
            )
        return None
    
    def _assess_agent_health(self) -> Dict[str, str]:
        """Assess health status of all agents"""
        agent_health = {}
        current_time = time.time()
        
        for agent_id, state in self.agent_states.items():
            age = state['age_seconds']
            
            if age < 120:  # Fresh data within 2 minutes
                agent_health[agent_id] = 'healthy'
            elif age < 300:  # Somewhat stale (5 minutes)
                agent_health[agent_id] = 'degraded'
            else:  # Very stale data
                agent_health[agent_id] = 'unhealthy'
        
        return agent_health

async def main():
    """Main coordinator entry point"""
    
    # Initialize runtime with coordinator
    runtime = AgentRuntime()
    
    # Create coordinator with extended cycle interval
    coordinator_config = {'cycle_interval': 45}  # Run every 45 seconds
    coordinator = CoordinatorAgent('coordinator', coordinator_config, runtime)
    
    runtime.register_agent(coordinator)
    
    print("AgentOps Coordinator Starting...")
    print("Monitoring agents: scout, health, security")
    print("Making strategic decisions based on agent insights")
    print("Press Ctrl+C to stop")
    
    await runtime.start_all_agents()

if __name__ == "__main__":
    asyncio.run(main())