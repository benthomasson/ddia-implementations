"""Fencing tokens for distributed locking (DDIA Ch. 8/9)."""


class FencingToken:
    """A fencing token issued with a lock."""

    def __init__(self, token: int, lock_name: str, client_id: str,
                 issued_at: int, ttl: int):
        self.token = token
        self.lock_name = lock_name
        self.client_id = client_id
        self.issued_at = issued_at
        self.ttl = ttl

    def is_expired(self, current_time: int) -> bool:
        """Check if this token's lock has expired."""
        return current_time >= self.issued_at + self.ttl


class LockService:
    """Lock service that issues monotonically increasing fencing tokens."""

    def __init__(self):
        self._counter = 1
        self._locks: dict[str, FencingToken] = {}

    def acquire(self, lock_name: str, client_id: str,
                current_time: int, ttl: int = 10) -> FencingToken | None:
        """Acquire a named lock. Returns a FencingToken or None if held by another client."""
        existing = self._locks.get(lock_name)
        if existing and not existing.is_expired(current_time) and existing.client_id != client_id:
            return None
        token = FencingToken(self._counter, lock_name, client_id, current_time, ttl)
        self._counter += 1
        self._locks[lock_name] = token
        return token

    def release(self, lock_name: str, client_id: str) -> bool:
        """Release a lock. Only the holder can release."""
        existing = self._locks.get(lock_name)
        if existing and existing.client_id == client_id:
            del self._locks[lock_name]
            return True
        return False

    def renew(self, lock_name: str, client_id: str,
              current_time: int, ttl: int = 10) -> bool:
        """Renew a lock's TTL without issuing a new token."""
        existing = self._locks.get(lock_name)
        if not existing or existing.client_id != client_id or existing.is_expired(current_time):
            return False
        existing.issued_at = current_time
        existing.ttl = ttl
        return True

    def is_held(self, lock_name: str, current_time: int) -> dict | None:
        """Check if a lock is currently held."""
        existing = self._locks.get(lock_name)
        if existing and not existing.is_expired(current_time):
            return {
                'client_id': existing.client_id,
                'token': existing.token,
                'expires_at': existing.issued_at + existing.ttl,
            }
        return None

    def get_token_counter(self) -> int:
        """Return the current global token counter."""
        return self._counter


class FencedResourceServer:
    """Key-value store protected by fencing token validation."""

    def __init__(self):
        self._data: dict[str, dict[str, any]] = {}
        self._highest_token: dict[str, int] = {}

    def write(self, resource: str, key: str, value: any,
              fencing_token: int) -> dict:
        """Write to a resource, rejecting stale tokens."""
        highest = self._highest_token.get(resource, 0)
        if fencing_token < highest:
            return {'success': False, 'error': f'Token {fencing_token} is stale (highest seen: {highest})'}
        self._highest_token[resource] = fencing_token
        if resource not in self._data:
            self._data[resource] = {}
        self._data[resource][key] = value
        return {'success': True, 'error': None}

    def read(self, resource: str, key: str) -> any | None:
        """Read a value."""
        return self._data.get(resource, {}).get(key)

    def get_highest_token(self, resource: str) -> int:
        """Return the highest fencing token seen for a resource."""
        return self._highest_token.get(resource, 0)


class UnfencedResourceServer:
    """Key-value store WITHOUT fencing token validation."""

    def __init__(self):
        self._data: dict[str, dict[str, any]] = {}

    def write(self, resource: str, key: str, value: any) -> dict:
        """Write without any token validation."""
        if resource not in self._data:
            self._data[resource] = {}
        self._data[resource][key] = value
        return {'success': True, 'error': None}

    def read(self, resource: str, key: str) -> any | None:
        """Read a value."""
        return self._data.get(resource, {}).get(key)


class Client:
    """Client that uses the lock service and writes to resource servers."""

    def __init__(self, client_id: str, lock_service: LockService):
        self.client_id = client_id
        self._lock_service = lock_service
        self._held_tokens: dict[str, FencingToken] = {}

    def acquire_lock(self, lock_name: str, current_time: int,
                     ttl: int = 10) -> FencingToken | None:
        """Acquire a lock and store the token."""
        token = self._lock_service.acquire(lock_name, self.client_id, current_time, ttl)
        if token:
            self._held_tokens[lock_name] = token
        return token

    def release_lock(self, lock_name: str) -> bool:
        """Release a held lock."""
        result = self._lock_service.release(lock_name, self.client_id)
        if result:
            del self._held_tokens[lock_name]
        return result

    def get_token(self, lock_name: str, current_time: int | None = None) -> FencingToken | None:
        """Get the fencing token for a lock this client holds."""
        token = self._held_tokens.get(lock_name)
        if token and current_time is not None and token.is_expired(current_time):
            del self._held_tokens[lock_name]
            return None
        return token

    def write_to_resource(self, server: FencedResourceServer,
                          resource: str, key: str, value: any,
                          lock_name: str, current_time: int | None = None) -> dict:
        """Write to a fenced resource using the held lock's token."""
        token = self._held_tokens.get(lock_name)
        if not token:
            return {'success': False, 'error': f'Client does not hold lock {lock_name}'}
        if current_time is not None and token.is_expired(current_time):
            del self._held_tokens[lock_name]
            return {'success': False, 'error': f'Lock {lock_name} has expired'}
        return server.write(resource, key, value, token.token)
