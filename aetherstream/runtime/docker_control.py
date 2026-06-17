"""Optional Docker integration for local companion-service restarts."""

from typing import Callable

try:
    import docker
except ImportError:  # pragma: no cover - optional deployment dependency
    docker = None


class DockerContainerRestarter:
    """Lazily connect to Docker and restart a configured container when enabled."""

    def __init__(self, *, container_name: str, log: Callable[[str], None]):
        self.container_name = container_name
        self._log = log
        self._client = None

    def get_client(self):
        if docker is None:
            self._log('Docker SDK unavailable on this runtime; skip docker client init')
            return None
        if self._client is None:
            try:
                self._client = docker.from_env()
            except Exception as e:
                self._log(f'Failed to connect to Docker: {e}')
        return self._client

    def restart(self, *, timeout: int = 5) -> None:
        try:
            client = self.get_client()
            if client:
                container = client.containers.get(self.container_name)
                container.restart(timeout=timeout)
                self._log(f'{self.container_name} restarted')
        except Exception as e:
            self._log(f'Failed to restart {self.container_name}: {e}')
