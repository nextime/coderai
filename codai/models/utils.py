class FuzzyToolBreaker:
    """
    Circuit breaker to detect when a model is stuck in a loop,
    repeatedly trying the same failing tool call.
    """
    
    def __init__(self, threshold=2):
        self.threshold = threshold
        self.history = {}  # Key: (tool_name, normalized_args_hash) -> Count

    def _normalize(self, data):
        """Recursively sort and clean dictionaries/lists for a stable hash."""
        import re
        import json
        if isinstance(data, dict):
            return {k: self._normalize(v) for k, v in sorted(data.items())}
        if isinstance(data, list):
            # For lists, normalize each element and then sort by string representation
            # to make it order-independent
            normalized_list = [self._normalize(i) for i in data]
            try:
                # Try to sort by string representation to make order-independent
                return sorted(normalized_list, key=lambda x: json.dumps(x, sort_keys=True))
            except Exception:
                # If sorting fails, return as-is
                return normalized_list
        if isinstance(data, str):
            # Strip whitespace and common LLM junk like leading/trailing dots
            return re.sub(r'\s+', '', data).strip(" ._-")
        return data

    def check(self, tool_name: str, arguments: dict):
        import hashlib
        import json
        # 1. Deep sort and clean the arguments
        normalized = self._normalize(arguments)
        
        # 2. Create a stable hash of the structure
        stable_json = json.dumps(normalized, sort_keys=True)
        call_hash = hashlib.md5(stable_json.encode()).hexdigest()
        
        key = (tool_name, call_hash)
        self.history[key] = self.history.get(key, 0) + 1
        count = self.history[key]

        if count > self.threshold:
            return True, (
                f"STOP: You've attempted {tool_name} with these parameters {count} times. "
                "It is failing or looping. DO NOT repeat this call. "
                "Try a different tool, search a different path, or ask the user for help."
            )
        return False, None
    
    def reset(self):
        """Clear the history (useful when starting a new conversation)."""
        self.history = {}
