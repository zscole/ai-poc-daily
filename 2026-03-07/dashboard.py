#!/usr/bin/env python3
"""
AgentOps Dashboard - Real-time visualization of agent activity and system status
Provides a live terminal interface for monitoring agent orchestration.
"""

import asyncio
import sqlite3
import json
import time
from datetime import datetime
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich import box
import structlog

logger = structlog.get_logger()

class AgentOpsDashboard:
    """Real-time dashboard for AgentOps system monitoring"""
    
    def __init__(self, db_path: str = "agentops.db"):
        self.db_path = db_path
        self.console = Console()
        self.last_update = time.time()
    
    def create_layout(self) -> Layout:
        """Create the dashboard layout"""
        layout = Layout()
        
        # Split screen into main areas
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Split main area
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Split left side
        layout["left"].split_column(
            Layout(name="system_overview", ratio=1),
            Layout(name="agent_status", ratio=2)
        )
        
        # Split right side
        layout["right"].split_column(
            Layout(name="alerts"),
            Layout(name="recent_decisions")
        )
        
        return layout
    
    def get_system_overview(self) -> Table:
        """Generate system overview table"""
        table = Table(title="System Overview", box=box.ROUNDED, title_style="bold blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Value", justify="right")
        
        # Get latest coordinator state
        coordinator_state = self._get_latest_agent_state('coordinator')
        if coordinator_state:
            overview = coordinator_state.get('system_overview', {})
            
            # Infrastructure health
            health = overview.get('infrastructure_health', 'unknown')
            health_color = self._get_status_color(health)
            table.add_row("Infrastructure", f"[{health_color}]{health.upper()}[/{health_color}]", "")
            
            # Security posture
            security = overview.get('security_posture', 'unknown')
            security_color = self._get_status_color(security)
            table.add_row("Security", f"[{security_color}]{security.upper()}[/{security_color}]", "")
            
            # Resource utilization
            resource_util = overview.get('resource_utilization', {})
            cpu_percent = resource_util.get('cpu_percent', 0)
            memory_percent = resource_util.get('memory_percent', 0)
            
            cpu_color = "red" if cpu_percent > 80 else "yellow" if cpu_percent > 60 else "green"
            memory_color = "red" if memory_percent > 85 else "yellow" if memory_percent > 70 else "green"
            
            table.add_row("CPU Usage", f"[{cpu_color}]{cpu_percent:.1f}%[/{cpu_color}]", "")
            table.add_row("Memory Usage", f"[{memory_color}]{memory_percent:.1f}%[/{memory_color}]", "")
            
            # Services and threats
            services = overview.get('discovered_services', 0)
            threats = overview.get('active_threats', 0)
            
            threat_color = "red" if threats > 0 else "green"
            table.add_row("Services", "[blue]ACTIVE[/blue]", str(services))
            table.add_row("Threats", f"[{threat_color}]{threats} FOUND[/{threat_color}]", "")
        
        else:
            table.add_row("Status", "[red]NO DATA[/red]", "Coordinator offline")
        
        return table
    
    def get_agent_status(self) -> Table:
        """Generate agent status table"""
        table = Table(title="Agent Status", box=box.ROUNDED, title_style="bold green")
        table.add_column("Agent", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Last Update", justify="center")
        table.add_column("Key Metrics", justify="left")
        
        agents = ['scout', 'health', 'security', 'coordinator']
        current_time = time.time()
        
        for agent_id in agents:
            state = self._get_latest_agent_state(agent_id)
            
            if state:
                timestamp = state.get('timestamp', 0)
                age_seconds = current_time - timestamp
                
                # Determine status based on data age
                if age_seconds < 120:
                    status = "[green]HEALTHY[/green]"
                elif age_seconds < 300:
                    status = "[yellow]DEGRADED[/yellow]"
                else:
                    status = "[red]STALE[/red]"
                
                # Format last update time
                last_update = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
                
                # Get key metrics for each agent type
                key_metrics = self._get_agent_key_metrics(agent_id, state)
                
                table.add_row(agent_id.upper(), status, last_update, key_metrics)
            else:
                table.add_row(agent_id.upper(), "[red]OFFLINE[/red]", "Never", "No data")
        
        return table
    
    def get_alerts_panel(self) -> Panel:
        """Generate alerts panel"""
        alert_text = Text()
        
        # Get recent alerts
        alerts = self._get_recent_alerts(limit=5)
        
        if alerts:
            alert_text.append("RECENT ALERTS\n", style="bold red")
            for alert in alerts:
                timestamp = datetime.fromtimestamp(alert['timestamp']).strftime("%H:%M:%S")
                sender = alert['sender'].upper()
                msg_type = alert['message_type']
                
                alert_text.append(f"{timestamp} ", style="dim")
                alert_text.append(f"[{sender}] ", style="cyan")
                alert_text.append(f"{msg_type}\n", style="yellow")
        else:
            alert_text.append("No recent alerts", style="dim")
        
        return Panel(alert_text, title="Alerts", border_style="red", title_align="left")
    
    def get_decisions_panel(self) -> Panel:
        """Generate recent decisions panel"""
        decisions_text = Text()
        
        # Get coordinator state with recent decisions
        coordinator_state = self._get_latest_agent_state('coordinator')
        
        if coordinator_state and 'recommendations' in coordinator_state:
            decisions_text.append("RECENT DECISIONS\n", style="bold blue")
            recommendations = coordinator_state['recommendations']
            
            for i, decision in enumerate(recommendations[-3:], 1):  # Show last 3
                action = decision.get('action', 'Unknown')
                decisions_text.append(f"{i}. {action}\n", style="blue")
                
                # Show key details
                if 'cpu_usage' in decision:
                    decisions_text.append(f"   CPU: {decision['cpu_usage']:.1f}%\n", style="dim")
                if 'severity' in decision:
                    decisions_text.append(f"   Severity: {decision['severity']}\n", style="dim")
                if 'anomaly_count' in decision:
                    decisions_text.append(f"   Anomalies: {decision['anomaly_count']}\n", style="dim")
        else:
            decisions_text.append("No recent decisions", style="dim")
        
        return Panel(decisions_text, title="Decisions", border_style="blue", title_align="left")
    
    def _get_latest_agent_state(self, agent_id: str) -> Dict:
        """Get latest state for specific agent"""
        try:
            conn = sqlite3.connect(self.db_path)
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
                result = json.loads(state_data)
                result['timestamp'] = timestamp
                return result
            return {}
            
        except Exception as e:
            logger.error("Failed to get agent state", agent_id=agent_id, error=str(e))
            return {}
    
    def _get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alert messages"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT sender_id, message_type, payload, timestamp
                FROM agent_messages
                WHERE recipient_id = 'coordinator'
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            alerts = []
            for row in cursor.fetchall():
                sender_id, message_type, payload, timestamp = row
                alerts.append({
                    'sender': sender_id,
                    'message_type': message_type,
                    'payload': json.loads(payload),
                    'timestamp': timestamp
                })
            
            conn.close()
            return alerts
            
        except Exception as e:
            logger.error("Failed to get alerts", error=str(e))
            return []
    
    def _get_status_color(self, status: str) -> str:
        """Get color for status display"""
        status_colors = {
            'healthy': 'green',
            'secure': 'green',
            'degraded': 'yellow',
            'vulnerable': 'yellow',
            'critical': 'red',
            'unknown': 'dim'
        }
        return status_colors.get(status.lower(), 'white')
    
    def _get_agent_key_metrics(self, agent_id: str, state: Dict) -> str:
        """Get key metrics for agent display"""
        if agent_id == 'scout':
            services = len(state.get('discovered_services', []))
            return f"{services} services discovered"
        
        elif agent_id == 'health':
            health_score = state.get('health_score', 0)
            anomalies = len(state.get('anomalies', []))
            return f"Health: {health_score:.0f}, Anomalies: {anomalies}"
        
        elif agent_id == 'security':
            security_score = state.get('security_score', 0)
            vulns = len(state.get('vulnerabilities', []))
            return f"Security: {security_score:.0f}, Vulns: {vulns}"
        
        elif agent_id == 'coordinator':
            decisions = state.get('new_decisions', 0)
            alerts = state.get('active_alerts', 0)
            return f"Decisions: {decisions}, Alerts: {alerts}"
        
        return "Active"
    
    def generate_dashboard(self) -> Layout:
        """Generate complete dashboard"""
        layout = self.create_layout()
        
        # Header
        header_text = Text("AgentOps Multi-Agent Infrastructure Intelligence", justify="center")
        header_text.stylize("bold white on blue")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_text.append(f" | {current_time}", style="dim white on blue")
        layout["header"].update(Panel(header_text, box=box.HEAVY))
        
        # Main panels
        layout["system_overview"].update(Panel(self.get_system_overview(), box=box.ROUNDED))
        layout["agent_status"].update(Panel(self.get_agent_status(), box=box.ROUNDED))
        layout["alerts"].update(self.get_alerts_panel())
        layout["recent_decisions"].update(self.get_decisions_panel())
        
        # Footer
        footer_text = Text("Press Ctrl+C to exit | Data refreshes every 5 seconds", justify="center")
        layout["footer"].update(Panel(footer_text, style="dim"))
        
        return layout

async def main():
    """Main dashboard entry point"""
    dashboard = AgentOpsDashboard()
    
    print("Starting AgentOps Dashboard...")
    print("Make sure AgentOps agents are running for data to appear.")
    
    try:
        with Live(dashboard.generate_dashboard(), refresh_per_second=0.2, screen=True):
            while True:
                await asyncio.sleep(5)  # Refresh every 5 seconds
                
    except KeyboardInterrupt:
        print("\nDashboard stopped.")

if __name__ == "__main__":
    asyncio.run(main())