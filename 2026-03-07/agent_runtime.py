#!/usr/bin/env python3
"""
AgentOps Runtime - Multi-Agent Infrastructure Intelligence System
Core runtime for spawning, managing, and coordinating infrastructure agents.
"""

import asyncio
import sqlite3
import json
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import structlog

logger = structlog.get_logger()

@dataclass
class AgentMessage:
    """Inter-agent communication message"""
    sender_id: str
    recipient_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class BaseAgent(ABC):
    """Base class for all infrastructure agents"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any], runtime: 'AgentRuntime'):
        self.agent_id = agent_id
        self.config = config
        self.runtime = runtime
        self.running = False
        self.logger = structlog.get_logger().bind(agent_id=agent_id)
        
    @abstractmethod
    async def execute_cycle(self) -> Dict[str, Any]:
        """Execute one complete agent cycle"""
        pass
    
    async def send_message(self, recipient_id: str, message_type: str, payload: Dict[str, Any]):
        """Send message to another agent"""
        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload
        )
        await self.runtime.deliver_message(message)
    
    async def start(self):
        """Start the agent's main loop"""
        self.running = True
        self.logger.info("Agent started")
        
        while self.running:
            try:
                results = await self.execute_cycle()
                await self.runtime.record_agent_state(self.agent_id, results)
                await asyncio.sleep(self.config.get('cycle_interval', 30))
            except Exception as e:
                self.logger.error("Agent cycle failed", error=str(e))
                await asyncio.sleep(5)
    
    def stop(self):
        """Stop the agent"""
        self.running = False
        self.logger.info("Agent stopped")

class ScoutAgent(BaseAgent):
    """Discovers services, ports, and infrastructure dependencies"""
    
    async def execute_cycle(self) -> Dict[str, Any]:
        import docker
        import psutil
        import nmap
        
        results = {
            'discovered_services': [],
            'network_topology': {},
            'resource_inventory': {}
        }
        
        # Discover Docker containers
        try:
            client = docker.from_env()
            containers = client.containers.list()
            
            for container in containers:
                service_info = {
                    'id': container.id[:12],
                    'name': container.name,
                    'image': container.image.tags[0] if container.image.tags else 'unknown',
                    'status': container.status,
                    'ports': container.attrs.get('NetworkSettings', {}).get('Ports', {}),
                    'labels': container.labels
                }
                results['discovered_services'].append(service_info)
                
        except Exception as e:
            self.logger.warning("Docker discovery failed", error=str(e))
        
        # System resource inventory
        results['resource_inventory'] = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_usage': {p.mountpoint: psutil.disk_usage(p.mountpoint)._asdict() 
                          for p in psutil.disk_partitions()},
            'network_interfaces': list(psutil.net_if_addrs().keys())
        }
        
        # Network topology scan (localhost only for demo)
        try:
            nm = nmap.PortScanner()
            nm.scan('127.0.0.1', '22-443')
            
            for host in nm.all_hosts():
                results['network_topology'][host] = {
                    'state': nm[host].state(),
                    'protocols': list(nm[host].all_protocols()),
                    'ports': {port: nm[host]['tcp'][port] 
                             for port in nm[host]['tcp'].keys()} if 'tcp' in nm[host] else {}
                }
        except Exception as e:
            self.logger.warning("Network scan failed", error=str(e))
        
        return results

class HealthAgent(BaseAgent):
    """Monitors performance metrics and detects anomalies"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_metrics = {}
        
    async def execute_cycle(self) -> Dict[str, Any]:
        import psutil
        
        current_metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
            'load_average': psutil.getloadavg(),
            'process_count': len(psutil.pids())
        }
        
        anomalies = []
        
        # Detect anomalies by comparing to baseline
        for metric, value in current_metrics.items():
            if isinstance(value, (int, float)) and metric in self.baseline_metrics:
                baseline = self.baseline_metrics[metric]
                deviation = abs(value - baseline) / baseline if baseline != 0 else 0
                
                if deviation > 0.5:  # 50% deviation threshold
                    anomalies.append({
                        'metric': metric,
                        'current_value': value,
                        'baseline_value': baseline,
                        'deviation_percent': deviation * 100
                    })
        
        # Update baseline (simple moving average)
        for metric, value in current_metrics.items():
            if isinstance(value, (int, float)):
                if metric in self.baseline_metrics:
                    self.baseline_metrics[metric] = 0.9 * self.baseline_metrics[metric] + 0.1 * value
                else:
                    self.baseline_metrics[metric] = value
        
        results = {
            'current_metrics': current_metrics,
            'anomalies': anomalies,
            'health_score': self._calculate_health_score(current_metrics, anomalies)
        }
        
        # Alert on critical anomalies
        if len(anomalies) > 0:
            await self.send_message('coordinator', 'anomaly_alert', {
                'severity': 'high' if len(anomalies) > 2 else 'medium',
                'anomalies': anomalies
            })
        
        return results
    
    def _calculate_health_score(self, metrics: Dict, anomalies: List) -> float:
        """Calculate overall system health score (0-100)"""
        base_score = 100
        
        # Deduct for high resource usage
        if metrics.get('cpu_percent', 0) > 80:
            base_score -= 20
        if metrics.get('memory_percent', 0) > 85:
            base_score -= 15
        
        # Deduct for anomalies
        base_score -= len(anomalies) * 10
        
        return max(0, base_score)

class SecurityAgent(BaseAgent):
    """Scans for vulnerabilities and security misconfigurations"""
    
    async def execute_cycle(self) -> Dict[str, Any]:
        vulnerabilities = []
        security_score = 100
        
        # Check for common security issues
        security_checks = [
            self._check_open_ports(),
            self._check_default_passwords(),
            self._check_unencrypted_connections(),
            self._check_outdated_packages()
        ]
        
        for check_result in security_checks:
            if check_result['issues']:
                vulnerabilities.extend(check_result['issues'])
                security_score -= check_result['severity_weight']
        
        results = {
            'vulnerabilities': vulnerabilities,
            'security_score': max(0, security_score),
            'recommendations': self._generate_recommendations(vulnerabilities)
        }
        
        if len(vulnerabilities) > 0:
            await self.send_message('coordinator', 'security_alert', {
                'critical_issues': [v for v in vulnerabilities if v['severity'] == 'critical'],
                'total_issues': len(vulnerabilities)
            })
        
        return results
    
    def _check_open_ports(self) -> Dict[str, Any]:
        """Check for unnecessarily open ports"""
        import psutil
        
        issues = []
        connections = psutil.net_connections()
        
        # Common risky ports
        risky_ports = [23, 21, 135, 139, 445, 1433, 3389]
        
        for conn in connections:
            if conn.status == 'LISTEN' and conn.laddr.port in risky_ports:
                issues.append({
                    'type': 'open_risky_port',
                    'port': conn.laddr.port,
                    'severity': 'high',
                    'description': f'Risky port {conn.laddr.port} is open and listening'
                })
        
        return {'issues': issues, 'severity_weight': 15}
    
    def _check_default_passwords(self) -> Dict[str, Any]:
        """Check for default/weak passwords (simulated)"""
        # In real implementation, would check databases, web apps, etc.
        issues = []
        
        # Simulate finding default credentials
        if self.config.get('simulate_default_creds', True):
            issues.append({
                'type': 'default_credentials',
                'service': 'example_database',
                'severity': 'critical',
                'description': 'Default credentials detected on database service'
            })
        
        return {'issues': issues, 'severity_weight': 25}
    
    def _check_unencrypted_connections(self) -> Dict[str, Any]:
        """Check for unencrypted network connections"""
        issues = []
        
        # Simulate HTTP vs HTTPS check
        if self.config.get('simulate_http_check', True):
            issues.append({
                'type': 'unencrypted_connection',
                'service': 'web_server',
                'severity': 'medium',
                'description': 'HTTP connections detected, should use HTTPS'
            })
        
        return {'issues': issues, 'severity_weight': 10}
    
    def _check_outdated_packages(self) -> Dict[str, Any]:
        """Check for outdated packages with known vulnerabilities"""
        issues = []
        
        # Simulate package vulnerability check
        if self.config.get('simulate_cve_check', True):
            issues.append({
                'type': 'outdated_package',
                'package': 'example-lib',
                'version': '1.2.3',
                'cve': 'CVE-2023-12345',
                'severity': 'high',
                'description': 'Outdated package with known vulnerability'
            })
        
        return {'issues': issues, 'severity_weight': 20}
    
    def _generate_recommendations(self, vulnerabilities: List[Dict]) -> List[str]:
        """Generate actionable security recommendations"""
        recommendations = []
        
        for vuln in vulnerabilities:
            if vuln['type'] == 'open_risky_port':
                recommendations.append(f"Close or restrict access to port {vuln['port']}")
            elif vuln['type'] == 'default_credentials':
                recommendations.append(f"Change default credentials on {vuln['service']}")
            elif vuln['type'] == 'unencrypted_connection':
                recommendations.append(f"Enable TLS/SSL encryption on {vuln['service']}")
            elif vuln['type'] == 'outdated_package':
                recommendations.append(f"Update {vuln['package']} to patch {vuln['cve']}")
        
        return recommendations

class AgentRuntime:
    """Central runtime for managing and coordinating all agents"""
    
    def __init__(self, db_path: str = "agentops.db"):
        self.db_path = db_path
        self.agents: Dict[str, BaseAgent] = {}
        self.message_queue: List[AgentMessage] = []
        self.logger = structlog.get_logger()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for agent communication and state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Agent states table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_states (
                agent_id TEXT,
                timestamp REAL,
                state_data TEXT,
                PRIMARY KEY (agent_id, timestamp)
            )
        ''')
        
        # Messages table
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
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the runtime"""
        self.agents[agent.agent_id] = agent
        self.logger.info("Agent registered", agent_id=agent.agent_id)
    
    async def record_agent_state(self, agent_id: str, state_data: Dict[str, Any]):
        """Record agent state to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO agent_states (agent_id, timestamp, state_data)
            VALUES (?, ?, ?)
        ''', (agent_id, time.time(), json.dumps(state_data)))
        
        conn.commit()
        conn.close()
    
    async def deliver_message(self, message: AgentMessage):
        """Deliver message between agents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO agent_messages (sender_id, recipient_id, message_type, payload, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (message.sender_id, message.recipient_id, message.message_type, 
              json.dumps(message.payload), message.timestamp))
        
        conn.commit()
        conn.close()
        
        self.logger.info("Message delivered", 
                        sender=message.sender_id,
                        recipient=message.recipient_id,
                        type=message.message_type)
    
    async def start_all_agents(self):
        """Start all registered agents"""
        tasks = []
        for agent in self.agents.values():
            task = asyncio.create_task(agent.start())
            tasks.append(task)
        
        self.logger.info("All agents started", count=len(tasks))
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("Shutting down agents")
            for agent in self.agents.values():
                agent.stop()

async def main():
    """Main entry point for AgentOps runtime"""
    
    # Initialize runtime
    runtime = AgentRuntime()
    
    # Create and register agents
    scout_config = {'cycle_interval': 60, 'discovery_depth': 'basic'}
    health_config = {'cycle_interval': 30, 'anomaly_threshold': 0.5}
    security_config = {'cycle_interval': 300, 'simulate_default_creds': True, 
                      'simulate_http_check': True, 'simulate_cve_check': True}
    
    scout = ScoutAgent('scout', scout_config, runtime)
    health = HealthAgent('health', health_config, runtime)
    security = SecurityAgent('security', security_config, runtime)
    
    runtime.register_agent(scout)
    runtime.register_agent(health)
    runtime.register_agent(security)
    
    print("AgentOps Runtime Starting...")
    print("Registered agents: scout, health, security")
    print("Press Ctrl+C to stop")
    
    # Start all agents
    await runtime.start_all_agents()

if __name__ == "__main__":
    asyncio.run(main())