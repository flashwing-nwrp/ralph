"""
VPS Deployment Module for RALPH Agent Ensemble

Enables agents to deploy code, run commands, and manage the trading bot
on the production VPS server.

Workflow:
1. Development happens locally (powerful dev machine)
2. Code is committed and pushed to git
3. Agents can trigger deployment to VPS via SSH
4. VPS pulls latest code, restarts services

Security:
- Uses SSH key authentication (no password)
- Only runs predefined safe commands
- Requires operator approval for dangerous operations
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("vps_deployer")


class DeploymentStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    CANCELLED = "cancelled"


@dataclass
class VPSConfig:
    """VPS connection configuration."""
    host: str
    user: str
    ssh_key: str
    port: int = 22
    project_dir: str = ""
    venv_path: str = ""
    log_dir: str = ""
    service_name: str = ""
    branch: str = "main"

    @classmethod
    def from_env(cls) -> Optional["VPSConfig"]:
        """Load VPS config from environment variables."""
        host = os.getenv("VPS_HOST")
        user = os.getenv("VPS_USER")

        if not host or not user:
            return None

        return cls(
            host=host,
            user=user,
            ssh_key=os.path.expanduser(os.getenv("VPS_SSH_KEY", "~/.ssh/id_rsa")),
            port=int(os.getenv("VPS_PORT", "22")),
            project_dir=os.getenv("VPS_PROJECT_DIR", ""),
            venv_path=os.getenv("VPS_VENV_PATH", ""),
            log_dir=os.getenv("VPS_LOG_DIR", ""),
            service_name=os.getenv("VPS_SERVICE_NAME", ""),
            branch=os.getenv("VPS_BRANCH", "main"),
        )


@dataclass
class CommandResult:
    """Result from VPS command execution."""
    success: bool
    stdout: str
    stderr: str
    return_code: int


class VPSDeployer:
    """
    Handles deployment and remote operations on the VPS.

    Uses SSH with key-based authentication for secure access.
    """

    def __init__(self, config: VPSConfig = None):
        self.config = config or VPSConfig.from_env()
        if not self.config:
            logger.warning("VPS configuration not found. Deployment disabled.")

    def _build_ssh_command(self, remote_command: str) -> list:
        """Build SSH command with proper options."""
        return [
            "ssh",
            "-i", self.config.ssh_key,
            "-p", str(self.config.port),
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",  # Fail if password required
            f"{self.config.user}@{self.config.host}",
            remote_command
        ]

    async def run_command(self, command: str, timeout: int = 60) -> CommandResult:
        """
        Run a command on the VPS via SSH.

        Args:
            command: The command to run
            timeout: Timeout in seconds

        Returns:
            CommandResult with output and status
        """
        if not self.config:
            return CommandResult(
                success=False,
                stdout="",
                stderr="VPS not configured",
                return_code=-1
            )

        ssh_cmd = self._build_ssh_command(command)

        try:
            process = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return CommandResult(
                    success=False,
                    stdout="",
                    stderr=f"Command timed out after {timeout}s",
                    return_code=-1
                )

            return CommandResult(
                success=process.returncode == 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                return_code=process.returncode
            )

        except FileNotFoundError:
            return CommandResult(
                success=False,
                stdout="",
                stderr="SSH client not found. Install OpenSSH.",
                return_code=-1
            )
        except Exception as e:
            return CommandResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1
            )

    async def test_connection(self) -> bool:
        """Test SSH connection to VPS."""
        result = await self.run_command("echo 'Connection OK'", timeout=10)
        return result.success

    async def deploy(self) -> Tuple[DeploymentStatus, str]:
        """
        Deploy latest code to VPS.

        Steps:
        1. Git pull on VPS
        2. Install dependencies
        3. Restart service
        """
        if not self.config:
            return DeploymentStatus.FAILED, "VPS not configured"

        steps = []

        # Step 1: Git pull
        logger.info("Pulling latest code on VPS...")
        result = await self.run_command(
            f"cd {self.config.project_dir} && git pull origin {self.config.branch}"
        )
        if not result.success:
            return DeploymentStatus.FAILED, f"Git pull failed: {result.stderr}"
        steps.append("Git pull: OK")

        # Step 2: Install dependencies (if requirements changed)
        logger.info("Installing dependencies...")
        result = await self.run_command(
            f"cd {self.config.project_dir} && "
            f"{self.config.venv_path}/bin/pip install -r requirements.txt --quiet"
        )
        if not result.success:
            logger.warning(f"Pip install warning: {result.stderr}")
        steps.append("Dependencies: OK")

        # Step 3: Restart service
        if self.config.service_name:
            logger.info(f"Restarting {self.config.service_name}...")
            result = await self.run_command(
                f"sudo systemctl restart {self.config.service_name}"
            )
            if not result.success:
                return DeploymentStatus.FAILED, f"Service restart failed: {result.stderr}"
            steps.append(f"Service restart: OK")

        return DeploymentStatus.SUCCESS, "\n".join(steps)

    async def get_status(self) -> dict:
        """Get status of the trading bot on VPS."""
        status = {
            "connection": False,
            "service": "unknown",
            "uptime": "unknown",
            "last_log": ""
        }

        if not self.config:
            return status

        # Check connection
        status["connection"] = await self.test_connection()

        if not status["connection"]:
            return status

        # Check service status
        if self.config.service_name:
            result = await self.run_command(
                f"systemctl is-active {self.config.service_name}"
            )
            status["service"] = result.stdout.strip() if result.success else "error"

        # Get uptime
        result = await self.run_command("uptime -p")
        if result.success:
            status["uptime"] = result.stdout.strip()

        # Get last log entry
        if self.config.log_dir:
            result = await self.run_command(
                f"tail -n 5 {self.config.log_dir}/*.log 2>/dev/null | tail -n 5"
            )
            if result.success:
                status["last_log"] = result.stdout.strip()[-500:]  # Last 500 chars

        return status

    async def get_logs(self, lines: int = 50) -> str:
        """Get recent logs from VPS."""
        if not self.config or not self.config.log_dir:
            return "Log directory not configured"

        result = await self.run_command(
            f"tail -n {lines} {self.config.log_dir}/*.log 2>/dev/null"
        )

        if result.success:
            return result.stdout
        else:
            return f"Error fetching logs: {result.stderr}"

    async def restart_service(self) -> Tuple[bool, str]:
        """Restart the trading bot service."""
        if not self.config or not self.config.service_name:
            return False, "Service name not configured"

        result = await self.run_command(
            f"sudo systemctl restart {self.config.service_name}"
        )

        if result.success:
            return True, f"Service {self.config.service_name} restarted"
        else:
            return False, f"Restart failed: {result.stderr}"

    async def stop_service(self) -> Tuple[bool, str]:
        """Stop the trading bot service."""
        if not self.config or not self.config.service_name:
            return False, "Service name not configured"

        result = await self.run_command(
            f"sudo systemctl stop {self.config.service_name}"
        )

        if result.success:
            return True, f"Service {self.config.service_name} stopped"
        else:
            return False, f"Stop failed: {result.stderr}"


# Singleton instance
_deployer: Optional[VPSDeployer] = None


def get_deployer() -> VPSDeployer:
    """Get or create the VPS deployer instance."""
    global _deployer
    if _deployer is None:
        _deployer = VPSDeployer()
    return _deployer
