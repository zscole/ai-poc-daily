#!/usr/bin/env python3
"""
AgentOps Demo - Demonstrates multi-agent coordination scenarios
Shows how agents work together to handle infrastructure challenges.
"""

import asyncio
import sqlite3
import json
import time
import random
from typing import Dict, List, Any
from agent_runtime import AgentRuntime, ScoutAgent, HealthAgent, SecurityAgent
from coordinator import CoordinatorAgent
import structlog

logger = structlog.get_logger()

class DemoScenarioRunner:
    """Runs demonstration scenarios for AgentOps"""
    
    def __init__(self):
        self.runtime = AgentRuntime("demo_agentops.db")
        self.agents = {}
        self.scenarios = {
            '1': self.scenario_service_discovery,
            '2': self.scenario_performance_degradation,
            '3': self.scenario_security_incident,
            '4': self.scenario_resource_optimization
        }
    
    async def setup_agents(self):
        """Setup demo agents with simulated data"""
        
        # Enhanced configs for demo scenarios
        scout_config = {
            'cycle_interval': 20,
            'demo_mode': True,
            'simulate_services': True
        }
        
        health_config = {
            'cycle_interval': 15,
            'demo_mode': True,
            'simulate_load': True
        }
        
        security_config = {
            'cycle_interval': 30,
            'demo_mode': True,
            'simulate_default_creds': True,
            'simulate_http_check': True,
            'simulate_cve_check': True
        }
        
        coordinator_config = {
            'cycle_interval': 25,
            'demo_mode': True
        }
        
        # Create demo agents
        self.agents['scout'] = DemoScoutAgent('scout', scout_config, self.runtime)
        self.agents['health'] = DemoHealthAgent('health', health_config, self.runtime)
        self.agents['security'] = DemoSecurityAgent('security', security_config, self.runtime)
        self.agents['coordinator'] = CoordinatorAgent('coordinator', coordinator_config, self.runtime)
        
        # Register all agents
        for agent in self.agents.values():
            self.runtime.register_agent(agent)
    
    async def run_interactive_demo(self):
        """Run interactive demo with user choices"""
        print("=== AgentOps Multi-Agent Infrastructure Intelligence Demo ===")
        print()
        print("This demo shows how AI agents coordinate to manage infrastructure.")
        print("Each agent has specialized expertise and they work together to solve complex problems.")
        print()
        
        await self.setup_agents()
        
        # Start agents in background
        agent_tasks = []
        for agent in self.agents.values():
            task = asyncio.create_task(agent.start())
            agent_tasks.append(task)
        
        # Let agents run for a moment to establish baseline
        print("Starting agents and establishing baseline...")
        await asyncio.sleep(10)
        
        while True:
            print("\n" + "="*60)
            print("AVAILABLE DEMO SCENARIOS:")
            print("1. Service Discovery - Scout finds new services, others analyze")
            print("2. Performance Degradation - Health detects issues, team responds")
            print("3. Security Incident - Security finds threats, coordinated response")
            print("4. Resource Optimization - System under load, agents optimize")
            print("5. View Current System Status")
            print("0. Exit Demo")
            print()
            
            choice = input("Select scenario (0-5): ").strip()
            
            if choice == '0':
                print("Stopping demo...")
                for agent in self.agents.values():
                    agent.stop()
                break
            
            elif choice == '5':
                await self.show_system_status()
            
            elif choice in self.scenarios:
                print(f"\n{'='*60}")
                await self.scenarios[choice]()
                print(f"{'='*60}")
                input("\nPress Enter to continue...")
            
            else:
                print("Invalid choice. Please select 0-5.")
    
    async def scenario_service_discovery(self):
        """Demo: New service discovery and analysis"""
        print("SCENARIO 1: Service Discovery and Multi-Agent Analysis")
        print("-" * 50)
        print("Simulating: New Docker container deployed")
        print()
        
        # Trigger scout agent to discover new service
        print("1. Scout Agent: Discovering new service...")
        self.agents['scout'].simulate_new_service = True
        await asyncio.sleep(5)
        
        print("2. Health Agent: Establishing baseline metrics...")
        await asyncio.sleep(3)
        
        print("3. Security Agent: Scanning for vulnerabilities...")
        self.agents['security'].simulate_new_scan = True
        await asyncio.sleep(3)
        
        print("4. Coordinator: Synthesizing findings...")
        await asyncio.sleep(3)
        
        # Show results
        print("\nRESULTS:")
        await self._show_latest_coordinator_decision()
    
    async def scenario_performance_degradation(self):
        """Demo: Performance issue detection and response"""
        print("SCENARIO 2: Performance Degradation Response")
        print("-" * 50)
        print("Simulating: High CPU usage and memory pressure")
        print()
        
        print("1. Health Agent: Detecting performance anomalies...")
        self.agents['health'].simulate_high_load = True
        await asyncio.sleep(5)
        
        print("2. Scout Agent: Analyzing affected services...")
        await asyncio.sleep(3)
        
        print("3. Security Agent: Ensuring changes don't create vulnerabilities...")
        await asyncio.sleep(3)
        
        print("4. Coordinator: Recommending optimization strategy...")
        await asyncio.sleep(3)
        
        print("\nRESULTS:")
        await self._show_latest_coordinator_decision()
        
        # Reset simulation
        self.agents['health'].simulate_high_load = False
    
    async def scenario_security_incident(self):
        """Demo: Security incident detection and response"""
        print("SCENARIO 3: Security Incident Response")
        print("-" * 50)
        print("Simulating: Critical vulnerabilities discovered")
        print()
        
        print("1. Security Agent: Detecting critical vulnerabilities...")
        self.agents['security'].simulate_critical_vuln = True
        await asyncio.sleep(5)
        
        print("2. Scout Agent: Mapping affected infrastructure...")
        await asyncio.sleep(3)
        
        print("3. Health Agent: Assessing performance impact of mitigation...")
        await asyncio.sleep(3)
        
        print("4. Coordinator: Orchestrating incident response...")
        await asyncio.sleep(3)
        
        print("\nRESULTS:")
        await self._show_latest_coordinator_decision()
        
        # Reset simulation
        self.agents['security'].simulate_critical_vuln = False
    
    async def scenario_resource_optimization(self):
        """Demo: Proactive resource optimization"""
        print("SCENARIO 4: Proactive Resource Optimization")
        print("-" * 50)
        print("Simulating: System approaching resource limits")
        print()
        
        print("1. Health Agent: Monitoring resource trends...")
        self.agents['health'].simulate_resource_pressure = True
        await asyncio.sleep(5)
        
        print("2. Scout Agent: Analyzing service resource usage...")
        await asyncio.sleep(3)
        
        print("3. Security Agent: Validating optimization proposals...")
        await asyncio.sleep(3)
        
        print("4. Coordinator: Recommending scaling and optimization...")
        await asyncio.sleep(3)
        
        print("\nRESULTS:")
        await self._show_latest_coordinator_decision()
        
        # Reset simulation
        self.agents['health'].simulate_resource_pressure = False
    
    async def show_system_status(self):
        """Show current system status from all agents"""
        print("\nCURRENT SYSTEM STATUS")
        print("-" * 30)
        
        for agent_id in ['scout', 'health', 'security', 'coordinator']:
            state = await self._get_latest_agent_state(agent_id)
            if state:
                print(f"\n{agent_id.upper()} Agent:")
                
                if agent_id == 'scout':
                    services = len(state.get('discovered_services', []))
                    print(f"  Services Discovered: {services}")
                    
                elif agent_id == 'health':
                    health_score = state.get('health_score', 0)
                    anomalies = len(state.get('anomalies', []))
                    print(f"  Health Score: {health_score}/100")
                    print(f"  Active Anomalies: {anomalies}")
                    
                elif agent_id == 'security':
                    security_score = state.get('security_score', 0)
                    vulns = len(state.get('vulnerabilities', []))
                    print(f"  Security Score: {security_score}/100")
                    print(f"  Vulnerabilities: {vulns}")
                    
                elif agent_id == 'coordinator':
                    overview = state.get('system_overview', {})
                    decisions = state.get('new_decisions', 0)
                    print(f"  Infrastructure Health: {overview.get('infrastructure_health', 'unknown')}")
                    print(f"  Security Posture: {overview.get('security_posture', 'unknown')}")
                    print(f"  Recent Decisions: {decisions}")
            else:
                print(f"\n{agent_id.upper()} Agent: No data available")
    
    async def _show_latest_coordinator_decision(self):
        """Show the latest decision from coordinator"""
        state = await self._get_latest_agent_state('coordinator')
        if state and 'recommendations' in state:
            recommendations = state['recommendations']
            if recommendations:
                latest = recommendations[-1]
                print(f"Action: {latest.get('action', 'Unknown')}")
                
                for key, value in latest.items():
                    if key != 'action' and not key.startswith('suggested_'):
                        print(f"  {key}: {value}")
                
                if 'suggested_actions' in latest:
                    print("  Suggested Actions:")
                    for action in latest['suggested_actions']:
                        print(f"    - {action}")
        else:
            print("No recent decisions available")
    
    async def _get_latest_agent_state(self, agent_id: str) -> Dict:
        """Get latest state for specific agent"""
        try:
            conn = sqlite3.connect(self.runtime.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT state_data, timestamp
                FROM agent_states
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (agent_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                state_data, timestamp = row
                return json.loads(state_data)
            return {}
            
        except Exception:
            return {}

class DemoScoutAgent(ScoutAgent):
    """Scout agent with demo simulation capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulate_new_service = False
        self.demo_service_count = 3
    
    async def execute_cycle(self) -> Dict[str, Any]:
        results = await super().execute_cycle()
        
        # Add demo services
        if self.simulate_new_service:
            results['discovered_services'].append({
                'id': f'demo-service-{int(time.time())}',
                'name': 'new-microservice',
                'image': 'nodejs:16-alpine',
                'status': 'running',
                'ports': {'3000/tcp': [{'HostPort': '3000'}]},
                'labels': {'environment': 'production', 'version': '1.0.0'}
            })
            self.simulate_new_service = False
            self.demo_service_count += 1
        
        return results

class DemoHealthAgent(HealthAgent):
    """Health agent with demo simulation capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulate_high_load = False
        self.simulate_resource_pressure = False
    
    async def execute_cycle(self) -> Dict[str, Any]:
        results = await super().execute_cycle()
        
        # Simulate high load scenario
        if self.simulate_high_load:
            results['current_metrics']['cpu_percent'] = 85 + random.uniform(-5, 10)
            results['current_metrics']['memory_percent'] = 88 + random.uniform(-3, 7)
            
            results['anomalies'].append({
                'metric': 'cpu_percent',
                'current_value': results['current_metrics']['cpu_percent'],
                'baseline_value': 45,
                'deviation_percent': 89
            })
        
        # Simulate resource pressure
        elif self.simulate_resource_pressure:
            results['current_metrics']['cpu_percent'] = 75 + random.uniform(-5, 10)
            results['current_metrics']['memory_percent'] = 78 + random.uniform(-3, 7)
            
            results['anomalies'].append({
                'metric': 'memory_percent',
                'current_value': results['current_metrics']['memory_percent'],
                'baseline_value': 50,
                'deviation_percent': 56
            })
        
        # Recalculate health score based on simulated conditions
        results['health_score'] = self._calculate_health_score(
            results['current_metrics'], 
            results['anomalies']
        )
        
        return results

class DemoSecurityAgent(SecurityAgent):
    """Security agent with demo simulation capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulate_new_scan = False
        self.simulate_critical_vuln = False
    
    async def execute_cycle(self) -> Dict[str, Any]:
        results = await super().execute_cycle()
        
        # Simulate new service scan
        if self.simulate_new_scan:
            results['vulnerabilities'].append({
                'type': 'unpatched_service',
                'service': 'new-microservice',
                'severity': 'medium',
                'description': 'Service running outdated Node.js version'
            })
            self.simulate_new_scan = False
        
        # Simulate critical vulnerability discovery
        if self.simulate_critical_vuln:
            results['vulnerabilities'].extend([
                {
                    'type': 'remote_code_execution',
                    'service': 'web-server',
                    'cve': 'CVE-2026-1234',
                    'severity': 'critical',
                    'description': 'Remote code execution vulnerability in web server'
                },
                {
                    'type': 'privilege_escalation',
                    'service': 'database',
                    'cve': 'CVE-2026-5678',
                    'severity': 'high',
                    'description': 'Local privilege escalation in database service'
                }
            ])
        
        # Recalculate security score
        critical_count = len([v for v in results['vulnerabilities'] if v.get('severity') == 'critical'])
        high_count = len([v for v in results['vulnerabilities'] if v.get('severity') == 'high'])
        
        results['security_score'] = max(0, 100 - (critical_count * 30) - (high_count * 15))
        results['recommendations'] = self._generate_recommendations(results['vulnerabilities'])
        
        return results

async def main():
    """Run the AgentOps demo"""
    demo = DemoScenarioRunner()
    
    try:
        await demo.run_interactive_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(main())